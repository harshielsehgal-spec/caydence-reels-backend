"""
recorded_router.py — POST /reels/upload_recorded

Single-call endpoint for the Loveable recording flow:
  1. Receives a recorded clip (≤15s, ≤25MB)
  2. Auto-trims setup/cleanup junk (opportunistic, never user-facing)
  3. Routes the trimmed clip through the existing /reels/analyze_v2 scoring
     path — synchronous response, same shape as analyze_v2's eventual result

The user never sees "trimming." Frontend hits one URL, gets one score.

The pivot context: live MediaPipe framing guidance on Loveable prevents bad
uploads from ever reaching here. So this endpoint trusts the input is at
least recordable and well-framed. Trim is just cleanup, not a quality gate.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse

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
#
# Mirrors the logic in rubric_router._process_attempt_v2 but synchronous and
# pure — takes a video path + reel_id + sport, returns a result dict in the
# same shape /reels/analyze_v2's eventual result has.
#
# This is additive: the existing async _process_attempt_v2 stays unchanged
# (per the constraints). We rebuild the same flow here so the new endpoint
# can return synchronously without going through the jobs queue.

def _score_attempt_sync(
    attempt_path: str,
    reel_id: str,
    sport: Optional[str],
) -> dict:
    """Run the full scoring pipeline on a clip file. Returns the result dict."""
    if reel_id in rubric_store:
        # Adaptive path — dense uniform sampling for rep detection
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

    # Legacy fallback — no rubric defined for this reel
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
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload_recorded")
async def upload_recorded(
    request: Request,
    reel_id: str = Form(...),
    sport: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    Accept a Loveable-recorded clip, trim it internally, score against
    the creator's reel, return result synchronously.

    Frontend hits this endpoint once. Gets the same response shape as
    /reels/analyze_v2's eventual result. Trim is invisible to the user.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "invalid file")

    # Defensive size check — read body in chunks, abort if over limit
    raw_path = TMP_DIR / f"{uuid.uuid4()}.mp4"
    trimmed_path = TMP_DIR / f"{raw_path.stem}_trimmed.mp4"

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

        # Auto-trim — opportunistic, never user-facing
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

        # Score synchronously
        result = await loop.run_in_executor(
            None, _score_attempt_sync, score_input_path, reel_id, sport
        )

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("upload_recorded failed clip_id=%s", raw_path.stem)
        raise HTTPException(500, "scoring failed")
    finally:
        # Always clean up — both files
        for p in (raw_path, trimmed_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                log.warning("cleanup failed for %s", p)