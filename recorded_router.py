"""
recorded_router.py — POST /reels/upload_recorded (multipart) and
                     POST /reels/upload_recorded_b64 (JSON, base64-encoded)

Both endpoints route through the same trim → score pipeline. The b64
endpoint exists as a workaround for iOS Safari, which hangs indefinitely
on multipart/form-data uploads.

Frontend hits one endpoint, gets one analyze_v2-shape response.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from clip_trimmer import trim_clip
from pose_similarity import extract_keypoints, extract_keypoints_phased
from rubric_scorer import score_with_rubric, result_to_dict
from sport_scorer import compute_sport_similarity, get_profile

# Share state with the existing routers — same dicts, no duplication
try:
    from rubric_router import (  # type: ignore
        rubric_store,
        creator_kps_store,
        _probe_video_info,
    )
    from reels_router import reference_cache  # type: ignore
except ImportError:
    rubric_store = {}
    creator_kps_store = {}
    reference_cache = {}
    def _probe_video_info(path: str):
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        duration_ms = int((frames / fps) * 1000) if fps > 0 else 0
        return duration_ms, float(fps), frames

log = logging.getLogger(__name__)

router = APIRouter(prefix="/reels", tags=["reels-recorded"])

# All temp files live here. Created on demand.
TMP_DIR = Path(tempfile.gettempdir()) / "caydence"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Defensive upload size limit (bytes). 25 MB.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous scoring callable
# ─────────────────────────────────────────────────────────────────────────────

def _score_attempt_sync(
    attempt_path: str,
    reel_id: str,
    sport: Optional[str],
) -> dict:
    """Run the full scoring pipeline on a clip file. Returns the result dict."""
    if reel_id in rubric_store:
        att_duration_ms, _, _ = _probe_video_info(attempt_path)
        sample_target = max(60, min(240, int(att_duration_ms / 100)))
        attempt_kps = extract_keypoints(attempt_path, sample_target)
        attempt_fps = (
            len(attempt_kps) / (att_duration_ms / 1000.0)
            if attempt_kps is not None and att_duration_ms > 0
            else 30.0
        )
    else:
        action_phase = get_profile(sport).get("action_phase_start", 0.0)
        attempt_kps = extract_keypoints_phased(attempt_path, 60, action_phase)
        attempt_fps = 30.0

    if attempt_kps is None or len(attempt_kps) < 5:
        raise HTTPException(
            422,
            "Could not extract pose. Ensure full body is visible and well-lit.",
        )

    if reel_id in rubric_store:
        creator_kps, duration_ms = creator_kps_store[reel_id]
        rubric = rubric_store[reel_id]
        result = score_with_rubric(
            creator_kps, attempt_kps, duration_ms, rubric, None, attempt_fps,
        )
        return {**result_to_dict(result), "sport": sport or "generic"}

    if reel_id not in reference_cache:
        raise HTTPException(
            404,
            f"No rubric or reference for reel '{reel_id}'.",
        )
    ref_kps = reference_cache[reel_id]
    legacy = compute_sport_similarity(ref_kps, attempt_kps, sport)
    return {
        "score": legacy.score,
        "arm_alignment": legacy.arm_alignment,
        "hip_position": legacy.hip_position,
        "timing_sync": legacy.timing_sync,
        "frame_count": legacy.frame_count,
        "confidence": legacy.confidence,
        "used_rubric": False,
        "checks": [],
        "key_frame_mapping": {},
        "sport": sport or "generic",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Transcode helper: webm → mp4 (iPhone Safari produces vp9 webm; OpenCV on
# Render's Linux build cannot decode vp9 reliably)
# ─────────────────────────────────────────────────────────────────────────────

# ffmpeg binary resolution. imageio_ffmpeg ships a static binary so we don't
# depend on Render's base image including ffmpeg. Lazy-import so a missing
# install fails loudly only when the helper is actually called.
def _ffmpeg_exe() -> str:
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


# Sanity ceiling for total_frames. 100k frames at 30fps = ~55 minutes, far
# beyond any 15s recording. Anything above this is a decode garbage value.
_MAX_REASONABLE_FRAMES = 100_000


def _ensure_mp4(input_path: str) -> str:
    """
    Ensure the input video is mp4. If already mp4, return path unchanged.
    If webm, transcode to a sibling _transcoded.mp4 file using ffmpeg
    (imageio-ffmpeg's bundled binary, no host dependency).

    Raises HTTPException(500, "transcode failed") on any ffmpeg failure.
    """
    p = Path(input_path)
    if p.suffix.lower() == ".mp4":
        return input_path

    if p.suffix.lower() != ".webm":
        # Defensive: only mp4 and webm reach here per _MIME_TO_EXT
        raise HTTPException(500, "transcode failed")

    output_path = p.parent / f"{p.stem}_transcoded.mp4"
    cmd = [
        _ffmpeg_exe(),
        "-i", str(p),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-an",       # drop audio — pose pipeline doesn't need it
        "-y",        # overwrite output if exists
        str(output_path),
    ]

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timeout input=%s", input_path)
        raise HTTPException(500, "transcode failed")
    except Exception as e:
        log.exception("ffmpeg subprocess failed input=%s err=%s", input_path, e)
        raise HTTPException(500, "transcode failed")

    elapsed = time.monotonic() - start

    if result.returncode != 0:
        # stderr is bytes; decode best-effort for log only, don't surface to client
        stderr_snippet = (result.stderr or b"").decode("utf-8", errors="replace")[:500]
        log.error(
            "ffmpeg non-zero exit input=%s rc=%d stderr=%s",
            input_path, result.returncode, stderr_snippet,
        )
        raise HTTPException(500, "transcode failed")

    log.info(
        "transcode webm→mp4 input=%s output=%s duration=%.2fs",
        input_path, str(output_path), elapsed,
    )
    return str(output_path)


def _check_decodable(mp4_path: str) -> None:
    """
    Verify the mp4 has a sane frame count before handing it to the pose
    pipeline. cv2.VideoCapture returns garbage values on undecodable inputs
    (e.g. -433498485732174464 on certain webm/vp9 files), which propagate
    into np.linspace and crash the worker. We catch that here.

    Raises HTTPException(500, "video decode failed") on any unreasonable
    frame count.
    """
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if total_frames <= 0 or total_frames > _MAX_REASONABLE_FRAMES:
        log.error(
            "decode check failed path=%s total_frames=%d",
            mp4_path, total_frames,
        )
        raise HTTPException(500, "video decode failed")


# ─────────────────────────────────────────────────────────────────────────────
# Shared post-write pipeline: trim → score → cleanup
# ─────────────────────────────────────────────────────────────────────────────

async def _process_recorded_file(
    raw_path: Path,
    reel_id: str,
    sport: Optional[str],
) -> JSONResponse:
    """
    Given a video file already written to disk, run the trim → score pipeline
    and return the analyze_v2-shape JSON response. Cleans up tmp files in a
    finally block. Used by both upload_recorded (multipart) and
    upload_recorded_b64 (JSON) endpoints.
    """
    transcoded_path: Optional[Path] = None

    try:
        loop = asyncio.get_event_loop()

        # 1. Ensure mp4. Transcodes webm → mp4 via ffmpeg if needed.
        mp4_path_str = await loop.run_in_executor(
            None, _ensure_mp4, str(raw_path)
        )
        if mp4_path_str != str(raw_path):
            transcoded_path = Path(mp4_path_str)

        # 2. Defensive decode check. Catches degenerate mp4s
        # (transcode succeeded but produced 0-frame or corrupt output).
        await loop.run_in_executor(None, _check_decodable, mp4_path_str)

        # 3. Trim. Operates on mp4.
        mp4_path = Path(mp4_path_str)
        trimmed_path = mp4_path.parent / f"{mp4_path.stem}_trimmed.mp4"

        trim_result = await loop.run_in_executor(
            None, trim_clip, mp4_path_str, str(trimmed_path)
        )

        if trim_result.get("trimmed"):
            score_input_path = str(trimmed_path)
        else:
            log.info(
                "trim skipped clip_id=%s original_duration=%.2f window=%s reason=%s",
                raw_path.stem,
                trim_result.get("original_duration_s", 0.0),
                trim_result.get("window"),
                trim_result.get("skip_reason"),
            )
            score_input_path = mp4_path_str

        # 4. Score.
        result = await loop.run_in_executor(
            None, _score_attempt_sync, score_input_path, reel_id, sport
        )

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception:
        log.exception("processing failed clip_id=%s", raw_path.stem)
        raise HTTPException(500, "scoring failed")
    finally:
        # Clean up: original raw, transcoded mp4 (if any), trimmed file.
        cleanup_targets = [raw_path]
        if transcoded_path is not None:
            cleanup_targets.append(transcoded_path)
            cleanup_targets.append(
                transcoded_path.parent / f"{transcoded_path.stem}_trimmed.mp4"
            )
        else:
            cleanup_targets.append(
                raw_path.parent / f"{raw_path.stem}_trimmed{raw_path.suffix}"
            )

        for p in cleanup_targets:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                log.warning("cleanup failed for %s", p)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1 — POST /reels/upload_recorded (multipart, original)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload_recorded")
async def upload_recorded(
    request: Request,
    reel_id: str = Form(...),
    sport: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Multipart form-data upload. Original endpoint, kept working for non-Safari clients."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "invalid file")

    raw_path = TMP_DIR / f"{uuid.uuid4()}.mp4"
    bytes_written = 0
    try:
        with open(raw_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    out.close()
                    raise HTTPException(413, "file too large")
                out.write(chunk)

        if bytes_written == 0:
            raise HTTPException(400, "invalid file")

    except HTTPException:
        try:
            if raw_path.exists():
                raw_path.unlink()
        except Exception:
            pass
        raise

    return await _process_recorded_file(raw_path, reel_id, sport)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2 — POST /reels/upload_recorded_b64 (JSON, base64-encoded)
# ─────────────────────────────────────────────────────────────────────────────

class B64UploadBody(BaseModel):
    reel_id: str = Field(..., min_length=1)
    video_b64: str = Field(..., min_length=1)
    mime_type: str = Field(..., min_length=1)
    sport: Optional[str] = None


_MIME_TO_EXT = {
    "video/webm": ".webm",
    "video/mp4": ".mp4",
}


@router.post("/upload_recorded_b64")
async def upload_recorded_b64(body: B64UploadBody):
    """
    Base64 JSON upload. Workaround for iOS Safari, which hangs on
    multipart/form-data fetch. Same response shape as /upload_recorded.
    """
    # Strip optional data-URL prefix (e.g. "data:video/webm;base64,...")
    b64 = body.video_b64
    if "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    # mime_type can include codec params (e.g. "video/webm;codecs=vp9")
    mime_base = body.mime_type.split(";")[0].strip().lower()
    ext = _MIME_TO_EXT.get(mime_base)
    if ext is None:
        raise HTTPException(400, f"unsupported mime_type: {body.mime_type}")

    try:
        raw_bytes = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "invalid base64")

    if len(raw_bytes) == 0:
        raise HTTPException(400, "invalid file")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")

    raw_path = TMP_DIR / f"{uuid.uuid4()}{ext}"
    try:
        with open(raw_path, "wb") as out:
            out.write(raw_bytes)
    except Exception:
        try:
            if raw_path.exists():
                raw_path.unlink()
        except Exception:
            pass
        log.exception("b64 write failed")
        raise HTTPException(500, "scoring failed")

    return await _process_recorded_file(raw_path, body.reel_id, body.sport)