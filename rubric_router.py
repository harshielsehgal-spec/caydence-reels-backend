"""
rubric_router.py — New endpoints for rubric-based scoring.

Adds:
  POST /reels/rubric            — creator builds a rubric for their reel
  POST /reels/analyze_v2        — viewer attempt, scored against rubric if one exists,
                                  falls back to sport_scorer otherwise
  GET  /reels/rubric/{reel_id}  — fetch stored rubric
  DELETE /reels/rubric/{reel_id}— remove stored rubric

Uses in-memory dicts matching reels_router.py's pattern. Migrate to Supabase
later by swapping the three dicts for table queries in one PR.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from pose_similarity import extract_keypoints, extract_keypoints_phased
from rubric_builder import RubricSpec, build_rubric
from rubric_schema import Rubric
from rubric_scorer import score_with_rubric, result_to_dict
from sport_scorer import compute_sport_similarity, get_profile
from adaptive_rubric import build_adaptive_rubric
from feedback_translator import translate_all, summary_line
from first_frame_extractor import populate_cache_safe

# Import existing stores so both routers share state. If reels_router
# is loaded first, its jobs/reference_cache dicts are used here too.
try:
    from reels_router import jobs, reference_cache  # type: ignore
except ImportError:
    jobs: dict = {}
    reference_cache: dict = {}


router = APIRouter(prefix="/reels", tags=["reels-rubric"])


# ─────────────────────────────────────────────────────────────────────────────
# In-memory stores for rubrics + creator keypoints
# ─────────────────────────────────────────────────────────────────────────────
# rubric_store:    reel_id -> Rubric
# creator_kps_store: reel_id -> (keypoints_uniform_array, duration_ms)
#
# We store UNIFORMLY sampled creator keypoints (not phased) because target
# timestamps must map linearly to frame indices.

rubric_store: dict[str, Rubric] = {}
creator_kps_store: dict[str, tuple[np.ndarray, int]] = {}

UPLOAD_DIR = Path(tempfile.gettempdir()) / "caydence_rubric"
UPLOAD_DIR.mkdir(exist_ok=True)


def _probe_duration_ms(video_path: str) -> int:
    """Read video duration in milliseconds using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return 0
    return int((frames / fps) * 1000)


def _probe_video_info(video_path: str) -> tuple[int, float, int]:
    """Return (duration_ms, real_fps, total_frames) from OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 30.0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration_ms = int((frames / fps) * 1000) if fps > 0 else 0
    return duration_ms, float(fps), frames


# ─────────────────────────────────────────────────────────────────────────────
# POST /reels/rubric — build + store rubric from creator video
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/rubric")
async def create_rubric(
    spec_json: str = Form(..., description="JSON string of RubricSpec"),
    reference_video: UploadFile = File(...),
):
    """
    Build a rubric for a reel from the creator's reference video.

    The `spec_json` form field is a JSON-encoded RubricSpec containing:
      - reel_id
      - sport (optional)
      - key_frames: [{id, timestamp_ms}, ...]
      - checks:     [{id, key_frame_id, type, landmarks, weight, ...}, ...]

    Targets are auto-extracted from the creator's pose at each key frame.
    Tolerances default by check type unless explicitly set per check.
    """
    try:
        spec_dict = json.loads(spec_json)
        spec = RubricSpec(**spec_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(400, f"Invalid spec_json: {e}")

    if not reference_video.content_type or not reference_video.content_type.startswith("video/"):
        raise HTTPException(400, "reference_video must be a video file")

    ref_path = str(UPLOAD_DIR / f"{uuid.uuid4()}_ref.mp4")
    with open(ref_path, "wb") as f:
        f.write(await reference_video.read())

    try:
        loop = asyncio.get_event_loop()

        # UNIFORM sampling — rubric target extraction requires linear
        # timestamp->frame mapping. Do NOT use extract_keypoints_phased here.
        creator_kps = await loop.run_in_executor(
            None, extract_keypoints, ref_path, 60
        )
        if creator_kps is None or len(creator_kps) < 5:
            raise HTTPException(
                400,
                "Could not extract pose from reference video. "
                "Ensure full body visible and well-lit."
            )

        duration_ms = _probe_duration_ms(ref_path)
        if duration_ms <= 0:
            raise HTTPException(400, "Could not read reference video duration.")

        try:
            rubric = build_rubric(spec, creator_kps, duration_ms)
        except (ValidationError, ValueError) as e:
            raise HTTPException(400, f"Rubric build failed: {e}")

        rubric_store[spec.reel_id] = rubric
        creator_kps_store[spec.reel_id] = (creator_kps, duration_ms)

        # Populate skeleton cache for ghost-overlay endpoint (additive,
        # safe — never raises). Must happen before the finally cleanup
        # deletes the tmp video file.
        await loop.run_in_executor(None, populate_cache_safe, spec.reel_id, ref_path)

        # Also populate reference_cache so /reels/analyze can use it
        # (stores the phased version for best legacy behavior)
        phased_kps = await loop.run_in_executor(
            None,
            extract_keypoints_phased,
            ref_path,
            60,
            get_profile(spec.sport).get("action_phase_start", 0.0),
        )
        if phased_kps is not None and len(phased_kps) >= 5:
            reference_cache[spec.reel_id] = phased_kps

        return JSONResponse({
            "reel_id": spec.reel_id,
            "rubric": rubric.model_dump(),
            "duration_ms": duration_ms,
            "creator_frame_count": len(creator_kps),
        })
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)


# ─────────────────────────────────────────────────────────────────────────────
# GET / DELETE rubric
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/rubric/{reel_id}")
async def get_rubric(reel_id: str):
    if reel_id not in rubric_store:
        raise HTTPException(404, f"No rubric for reel '{reel_id}'")
    return rubric_store[reel_id].model_dump()


@router.delete("/rubric/{reel_id}")
async def delete_rubric(reel_id: str):
    rubric_store.pop(reel_id, None)
    creator_kps_store.pop(reel_id, None)
    return {"deleted": reel_id}


# ─────────────────────────────────────────────────────────────────────────────
# POST /reels/analyze_v2 — rubric-aware scoring with legacy fallback
# ─────────────────────────────────────────────────────────────────────────────

async def _process_attempt_v2(
    job_id: str,
    attempt_path: str,
    reel_id: str,
    sport: Optional[str],
):
    try:
        jobs[job_id]["status"] = "processing"
        loop = asyncio.get_event_loop()

        # If rubric exists → use DENSE UNIFORM sampling so rep detection works.
        # If legacy path → use phased sampling as before.
        if reel_id in rubric_store:
            att_duration_ms, _, _ = _probe_video_info(attempt_path)
            sample_target = max(60, min(240, int(att_duration_ms / 100)))
            attempt_kps = await loop.run_in_executor(
                None, extract_keypoints, attempt_path, sample_target
            )
            attempt_fps = len(attempt_kps) / (att_duration_ms / 1000.0) \
                if attempt_kps is not None and att_duration_ms > 0 else 30.0
        else:
            action_phase = get_profile(sport).get("action_phase_start", 0.0)
            attempt_kps = await loop.run_in_executor(
                None, extract_keypoints_phased, attempt_path, 60, action_phase
            )
            attempt_fps = 30.0

        if attempt_kps is None or len(attempt_kps) < 5:
            jobs[job_id] = {
                "status": "failed",
                "error": "Could not extract pose. Ensure full body visible in good lighting.",
            }
            return

        if reel_id in rubric_store:
            creator_kps, duration_ms = creator_kps_store[reel_id]
            rubric = rubric_store[reel_id]
            # Use attempt's sampled FPS for rep detection inside scoring
            result = await loop.run_in_executor(
                None,
                score_with_rubric,
                creator_kps, attempt_kps, duration_ms, rubric, None, attempt_fps,
            )
            jobs[job_id] = {
                "status": "complete",
                "result": {**result_to_dict(result), "sport": sport or "generic"},
            }
        else:
            # Legacy path — no rubric defined for this reel
            if reel_id not in reference_cache:
                jobs[job_id] = {
                    "status": "failed",
                    "error": f"No reference or rubric for reel '{reel_id}'. "
                             f"POST /reels/rubric or /reels/analyze with reference_video first.",
                }
                return
            ref_kps = reference_cache[reel_id]
            legacy = await loop.run_in_executor(
                None, compute_sport_similarity, ref_kps, attempt_kps, sport
            )
            jobs[job_id] = {
                "status": "complete",
                "result": {
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
                },
            }
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(attempt_path):
            os.remove(attempt_path)


@router.post("/analyze_v2")
async def analyze_v2(
    background_tasks: BackgroundTasks,
    reel_id: str,
    sport: Optional[str] = None,
    attempt_video: UploadFile = File(...),
):
    """
    Score a viewer attempt. Uses rubric if one was built for `reel_id`,
    otherwise falls back to the legacy sport_scorer path.

    Response shape is a superset of the legacy /reels/analyze response —
    `used_rubric`, `checks`, and `key_frame_mapping` are the new fields.
    """
    if not attempt_video.content_type or not attempt_video.content_type.startswith("video/"):
        raise HTTPException(400, "attempt_video must be a video file")

    if reel_id not in rubric_store and reel_id not in reference_cache:
        raise HTTPException(
            400,
            f"No rubric or cached reference for reel '{reel_id}'. "
            f"Call POST /reels/rubric (preferred) or /reels/analyze with "
            f"reference_video first.",
        )

    job_id = str(uuid.uuid4())
    attempt_path = str(UPLOAD_DIR / f"{job_id}_attempt.mp4")
    with open(attempt_path, "wb") as f:
        f.write(await attempt_video.read())

    jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(
        _process_attempt_v2, job_id, attempt_path, reel_id, sport
    )
    return JSONResponse({
        "job_id": job_id,
        "status": "pending",
        "used_rubric": reel_id in rubric_store,
        "sport": sport or "generic",
    })

# ─────────────────────────────────────────────────────────────────────────────
# POST /reels/rubric_adaptive — build rubric from creator video alone
# ─────────────────────────────────────────────────────────────────────────────
#
# Zero creator config. Upload video → engine figures out which joints matter,
# when the key moments are, what targets to hit, and how strict to be.

@router.post("/rubric_adaptive")
async def create_adaptive_rubric(
    reel_id: str = Form(...),
    reference_video: UploadFile = File(...),
):
    if not reference_video.content_type or not reference_video.content_type.startswith("video/"):
        raise HTTPException(400, "reference_video must be a video file")

    ref_path = str(UPLOAD_DIR / f"{uuid.uuid4()}_adaptive.mp4")
    with open(ref_path, "wb") as f:
        f.write(await reference_video.read())

    try:
        loop = asyncio.get_event_loop()

        # Probe real video info first — we need real FPS for correct rep
        # detection gap calculation.
        duration_ms, real_fps, total_frames = _probe_video_info(ref_path)
        if duration_ms <= 0:
            raise HTTPException(400, "Could not read reference video duration.")

        # Dense sampling for rubric building — need enough frames per rep to
        # detect extrema. Aim for effective 10fps sampled (sufficient for
        # pushups/squats/most sports), capped at 240 frames to keep memory
        # bounded. Minimum 60 frames for short clips.
        sample_target = max(60, min(240, int(duration_ms / 100)))  # 10fps == 1 frame per 100ms
        creator_kps = await loop.run_in_executor(
            None, extract_keypoints, ref_path, sample_target
        )
        if creator_kps is None or len(creator_kps) < 5:
            raise HTTPException(400, "Could not extract pose from reference video.")

        # Effective sampled FPS = how many keypoint frames per real second.
        # This is what rep detection needs for its "min gap between reps"
        # calculation, NOT the raw camera FPS.
        sampled_fps = len(creator_kps) / (duration_ms / 1000.0)

        try:
            result = await loop.run_in_executor(
                None, build_adaptive_rubric, creator_kps, reel_id, sampled_fps
            )
        except ValueError as e:
            raise HTTPException(400, f"Could not build adaptive rubric: {e}")

        rubric_store[reel_id] = result.rubric
        creator_kps_store[reel_id] = (creator_kps, duration_ms)

        # Populate skeleton cache for ghost-overlay endpoint
        await loop.run_in_executor(None, populate_cache_safe, reel_id, ref_path)

        # Populate legacy reference_cache for /reels/analyze fallback
        phased_kps = await loop.run_in_executor(
            None, extract_keypoints_phased, ref_path, 60, 0.0,
        )
        if phased_kps is not None and len(phased_kps) >= 5:
            reference_cache[reel_id] = phased_kps

        return JSONResponse({
            "reel_id": reel_id,
            "rubric": result.rubric.model_dump(),
            "importance_weights": result.importance_weights,
            "rep_count_bottoms": len(result.rep_bottoms),
            "rep_count_tops": len(result.rep_tops),
            "primary_joint": result.primary_joint,
            "static_hold": result.static_hold,
            "duration_ms": duration_ms,
            "real_fps": round(real_fps, 2),
            "sampled_fps": round(sampled_fps, 2),
            "creator_frame_count": len(creator_kps),
        })
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)


# ─────────────────────────────────────────────────────────────────────────────
# POST /reels/feedback_v2 — human-readable feedback from a completed job
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackV2Request(BaseModel):
    job_id: str


@router.post("/feedback_v2")
async def feedback_v2(req: FeedbackV2Request):
    """Convert a completed rubric scoring job into human-readable feedback."""
    if req.job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    job = jobs[req.job_id]
    if job.get("status") != "complete":
        raise HTTPException(400, f"Job not complete (status: {job.get('status')})")
    result = job.get("result") or {}
    checks = result.get("checks") or []
    feedback = translate_all(checks)
    return {
        "summary": summary_line(feedback),
        "items": feedback,
        "score": result.get("score"),
        "rep_count_creator": result.get("rep_count_creator", 0),
        "rep_count_viewer": result.get("rep_count_viewer", 0),
        "per_rep_scores": result.get("per_rep_scores", []),
    }