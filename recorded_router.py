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
import tempfile
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
    trimmed_path = raw_path.parent / f"{raw_path.stem}_trimmed{raw_path.suffix}"

    try:
        loop = asyncio.get_event_loop()
        trim_result = await loop.run_in_executor(
            None, trim_clip, str(raw_path), str(trimmed_path)
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
            score_input_path = str(raw_path)

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
        for p in (raw_path, trimmed_path):
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