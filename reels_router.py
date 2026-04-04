from ai_coach import generate_reel_coaching
import os
import uuid
import asyncio
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np

from pose_similarity import extract_keypoints, extract_keypoints_phased
from sport_scorer import compute_sport_similarity, get_profile

router = APIRouter(prefix="/reels", tags=["reels"])

# In-memory stores — replace with Supabase in Phase 2
jobs: dict = {}
reference_cache: dict = {}

UPLOAD_DIR = Path(tempfile.gettempdir()) / "caydence_attempts"
UPLOAD_DIR.mkdir(exist_ok=True)


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | processing | complete | failed
    result: Optional[dict] = None
    error: Optional[str] = None


async def process_attempt(
    job_id: str,
    attempt_path: str,
    ref_keypoints: np.ndarray,
    sport: Optional[str],
):
    """Background task — pose extraction + sport-aware scoring."""
    try:
        jobs[job_id]["status"] = "processing"

        loop = asyncio.get_event_loop()

        # Get action phase start for this sport
        action_phase_start = get_profile(sport).get("action_phase_start", 0.0)

        # Extract attempt keypoints using phase-aware sampling
        attempt_kps = await loop.run_in_executor(
            None, extract_keypoints_phased, attempt_path, 60, action_phase_start
        )

        if attempt_kps is None or len(attempt_kps) < 5:
            jobs[job_id] = {
                "status": "failed",
                "error": (
                    "Could not extract pose from video. "
                    "Ensure your full body is visible in good lighting."
                )
            }
            return

        # Sport-aware scoring
        result = await loop.run_in_executor(
            None,
            compute_sport_similarity,
            ref_keypoints,
            attempt_kps,
            sport,
        )

        jobs[job_id] = {
            "status": "complete",
            "result": {
                "score": result.score,
                "arm_alignment": result.arm_alignment,
                "hip_position": result.hip_position,
                "timing_sync": result.timing_sync,
                "frame_count": result.frame_count,
                "confidence": result.confidence,
                "sport": sport or "generic",
            }
        }

    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(attempt_path):
            os.remove(attempt_path)


@router.post("/analyze")
async def analyze_attempt(
    background_tasks: BackgroundTasks,
    reel_id: str,
    sport: Optional[str] = None,
    attempt_video: UploadFile = File(...),
    reference_video: Optional[UploadFile] = File(None),
):
    """
    Submit attempt video for sport-aware pose similarity scoring.
    Returns job_id immediately — poll /reels/result/{job_id}.

    Parameters:
        reel_id: unique reel identifier
        sport: sport profile key (e.g. 'cricket_bowling')
               defaults to generic scoring if omitted
        attempt_video: user's attempt video
        reference_video: creator's reference video
                        (required on first attempt per reel)
    """
    if not attempt_video.content_type.startswith("video/"):
        raise HTTPException(400, "attempt_video must be a video file.")

    job_id = str(uuid.uuid4())
    attempt_path = str(UPLOAD_DIR / f"{job_id}_attempt.mp4")

    contents = await attempt_video.read()
    with open(attempt_path, "wb") as f:
        f.write(contents)

    # Handle reference keypoints
    if reel_id in reference_cache:
        ref_kps = reference_cache[reel_id]

    elif reference_video is not None:
        if not reference_video.content_type.startswith("video/"):
            raise HTTPException(400, "reference_video must be a video file.")

        ref_path = str(UPLOAD_DIR / f"{job_id}_reference.mp4")
        ref_contents = await reference_video.read()
        with open(ref_path, "wb") as f:
            f.write(ref_contents)

        # Get action phase start for reference extraction too
        action_phase_start = get_profile(sport).get("action_phase_start", 0.0)

        loop = asyncio.get_event_loop()
        ref_kps = await loop.run_in_executor(
            None, extract_keypoints_phased, ref_path, 60, action_phase_start
        )
        os.remove(ref_path)

        if ref_kps is None or len(ref_kps) < 5:
            raise HTTPException(
                400,
                "Could not extract pose from reference video. "
                "Ensure full body is visible."
            )

        reference_cache[reel_id] = ref_kps

    else:
        raise HTTPException(
            400,
            f"No reference found for reel '{reel_id}'. "
            "Pass reference_video on first attempt for this reel."
        )

    # Queue background job
    jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(
        process_attempt, job_id, attempt_path, ref_kps, sport
    )

    return JSONResponse({
        "job_id": job_id,
        "status": "pending",
        "sport": sport or "generic"
    })


@router.get("/result/{job_id}", response_model=JobStatus)
async def get_result(job_id: str):
    """Poll until status is 'complete' or 'failed'."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get("/profiles")
async def list_sport_profiles():
    """Returns all available sport profiles and their weights."""
    from sport_scorer import SPORT_PROFILES
    return {
        k: {
            "display_name": v["display_name"],
            "weights": v["weights"],
            "timing_critical": v["timing_critical"],
        }
        for k, v in SPORT_PROFILES.items()
        if not k.startswith("_")
    }


@router.delete("/cache/{reel_id}")
async def clear_reference_cache(reel_id: str):
    """Clear cached reference keypoints for a reel."""
    reference_cache.pop(reel_id, None)
    return {"cleared": reel_id}

class FeedbackRequest(BaseModel):
    sport: str
    creator_name: str
    arm_score: int
    hip_score: int
    timing_score: int
    overall_score: int


@router.post("/feedback")
async def get_coaching_feedback(req: FeedbackRequest):
    """
    Call Claude API to generate sport-specific coaching feedback.
    Called async after score is returned — never blocks score reveal.
    """
    from ai_coach import generate_reel_coaching
    feedback = await generate_reel_coaching(
        sport=req.sport,
        creator_name=req.creator_name,
        arm_score=req.arm_score,
        hip_score=req.hip_score,
        timing_score=req.timing_score,
        overall_score=req.overall_score,
    )
    return {"feedback": feedback}