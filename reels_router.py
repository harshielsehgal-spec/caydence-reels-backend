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
import httpx

from pose_similarity import extract_keypoints, extract_keypoints_phased
from sport_scorer import compute_sport_similarity, get_profile

router = APIRouter(prefix="/reels", tags=["reels"])

# In-memory stores — replace with Supabase in Phase 2
jobs: dict = {}
reference_cache: dict = {}

UPLOAD_DIR = Path(tempfile.gettempdir()) / "caydence_attempts"
UPLOAD_DIR.mkdir(exist_ok=True)

# Max video size to download (100MB matches frontend limit)
MAX_VIDEO_BYTES = 100 * 1024 * 1024
# Per-download timeout (Render free tier can be slow on cold fetches)
DOWNLOAD_TIMEOUT_SECONDS = 60.0


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | processing | complete | failed
    result: Optional[dict] = None
    error: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """JSON payload from the frontend (iOS-friendly — no multipart)."""
    reel_id: str
    attempt_video_url: str
    athlete_id: Optional[str] = None
    sport: Optional[str] = None
    # Optional override — if frontend already knows the reference URL,
    # pass it to skip the DB lookup. Otherwise backend looks it up.
    reference_video_url: Optional[str] = None


async def _download_video(url: str, dest_path: str) -> None:
    """
    Stream a video from a public URL to local disk.
    Raises HTTPException on failure with a useful error message.
    """
    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        400,
                        f"Failed to fetch video (HTTP {response.status_code}) from {url}"
                    )

                total_bytes = 0
                with open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        total_bytes += len(chunk)
                        if total_bytes > MAX_VIDEO_BYTES:
                            raise HTTPException(413, "Video exceeds 100MB limit.")
                        f.write(chunk)

                if total_bytes == 0:
                    raise HTTPException(400, "Downloaded video is empty (0 bytes).")
    except httpx.TimeoutException:
        raise HTTPException(504, f"Timeout downloading video from {url}")
    except httpx.RequestError as e:
        raise HTTPException(502, f"Network error downloading video: {e}")


async def _lookup_reference_url(reel_id: str) -> tuple[str, Optional[str]]:
    """
    Look up the reference video URL and sport for a reel from Supabase.
    Returns (video_url, sport). Raises HTTPException if not found.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise HTTPException(
            500,
            "Supabase credentials not configured on backend. "
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars."
        )

    # Direct REST query — avoids adding supabase-py as a dependency
    url = f"{supabase_url}/rest/v1/reels?id=eq.{reel_id}&select=video_url,sport"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(
                    500,
                    f"Supabase lookup failed (HTTP {response.status_code}): {response.text}"
                )
            rows = response.json()
            if not rows:
                raise HTTPException(404, f"No reel found with id '{reel_id}'")
            row = rows[0]
            return row["video_url"], row.get("sport")
    except httpx.RequestError as e:
        raise HTTPException(502, f"Supabase network error: {e}")


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


# ─────────────────────────────────────────────────────────────────────────────
# NEW: JSON-based /analyze endpoint (iOS Safari friendly)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_attempt(
    background_tasks: BackgroundTasks,
    req: AnalyzeRequest,
):
    """
    Submit attempt for sport-aware pose similarity scoring.

    Frontend uploads the attempt video to Supabase storage first, then
    POSTs the URL here as JSON. Backend downloads both attempt and
    reference videos itself — avoids iOS Safari multipart upload bugs.

    Returns job_id immediately — poll /reels/result/{job_id}.
    """
    job_id = str(uuid.uuid4())
    attempt_path = str(UPLOAD_DIR / f"{job_id}_attempt.mp4")

    # 1. Download the attempt video from Supabase storage
    try:
        await _download_video(req.attempt_video_url, attempt_path)
    except HTTPException:
        if os.path.exists(attempt_path):
            os.remove(attempt_path)
        raise

    # 2. Get reference keypoints (cached, or download + extract)
    if req.reel_id in reference_cache:
        ref_kps = reference_cache[req.reel_id]
        sport = req.sport
    else:
        # Need to fetch reference video and extract keypoints.
        # Use provided URL or look it up from Supabase.
        if req.reference_video_url:
            reference_url = req.reference_video_url
            sport = req.sport
        else:
            try:
                reference_url, db_sport = await _lookup_reference_url(req.reel_id)
                sport = req.sport or db_sport
            except HTTPException:
                if os.path.exists(attempt_path):
                    os.remove(attempt_path)
                raise

        ref_path = str(UPLOAD_DIR / f"{job_id}_reference.mp4")
        try:
            await _download_video(reference_url, ref_path)
        except HTTPException:
            if os.path.exists(attempt_path):
                os.remove(attempt_path)
            if os.path.exists(ref_path):
                os.remove(ref_path)
            raise

        action_phase_start = get_profile(sport).get("action_phase_start", 0.0)
        loop = asyncio.get_event_loop()
        ref_kps = await loop.run_in_executor(
            None, extract_keypoints_phased, ref_path, 60, action_phase_start
        )

        # Populate skeleton cache for ghost-overlay endpoint
        try:
            from first_frame_extractor import populate_cache_safe
            await loop.run_in_executor(None, populate_cache_safe, req.reel_id, ref_path)
        except Exception:
            pass

        if os.path.exists(ref_path):
            os.remove(ref_path)

        if ref_kps is None or len(ref_kps) < 5:
            if os.path.exists(attempt_path):
                os.remove(attempt_path)
            raise HTTPException(
                400,
                "Could not extract pose from reference video. "
                "Ensure full body is visible."
            )

        reference_cache[req.reel_id] = ref_kps

    # 3. Queue background job
    jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(
        process_attempt, job_id, attempt_path, ref_kps, sport
    )

    return JSONResponse({
        "job_id": job_id,
        "status": "pending",
        "sport": sport or "generic"
    })


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY: Original multipart /analyze endpoint, kept as /analyze_legacy
# Use this only if some other client still uploads files directly.
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze_legacy")
async def analyze_attempt_legacy(
    background_tasks: BackgroundTasks,
    reel_id: str,
    sport: Optional[str] = None,
    attempt_video: UploadFile = File(...),
    reference_video: Optional[UploadFile] = File(None),
):
    """
    LEGACY multipart endpoint. Kept for backward compatibility.
    Prefer POST /reels/analyze (JSON) for new clients, especially iOS Safari.
    """
    if not attempt_video.content_type or not attempt_video.content_type.startswith("video/"):
        raise HTTPException(400, "attempt_video must be a video file.")

    job_id = str(uuid.uuid4())
    attempt_path = str(UPLOAD_DIR / f"{job_id}_attempt.mp4")

    contents = await attempt_video.read()
    with open(attempt_path, "wb") as f:
        f.write(contents)

    if reel_id in reference_cache:
        ref_kps = reference_cache[reel_id]
    elif reference_video is not None:
        if not reference_video.content_type or not reference_video.content_type.startswith("video/"):
            raise HTTPException(400, "reference_video must be a video file.")

        ref_path = str(UPLOAD_DIR / f"{job_id}_reference.mp4")
        ref_contents = await reference_video.read()
        with open(ref_path, "wb") as f:
            f.write(ref_contents)

        action_phase_start = get_profile(sport).get("action_phase_start", 0.0)

        loop = asyncio.get_event_loop()
        ref_kps = await loop.run_in_executor(
            None, extract_keypoints_phased, ref_path, 60, action_phase_start
        )

        try:
            from first_frame_extractor import populate_cache_safe
            await loop.run_in_executor(None, populate_cache_safe, reel_id, ref_path)
        except Exception:
            pass

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