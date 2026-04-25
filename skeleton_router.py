"""
skeleton_router.py — Serves the creator's first-frame skeleton for the
Loveable recording component's ghost overlay.

Reads from raw_first_frame_cache populated at creator-ingestion time.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from first_frame_extractor import raw_first_frame_cache

router = APIRouter(prefix="/reels", tags=["reels-skeleton"])


@router.get("/{reel_id}/skeleton")
async def get_skeleton(reel_id: str):
    """
    Returns the creator's first qualifying frame pose landmarks in raw
    MediaPipe normalized coordinate space (0-1), suitable for overlay on
    a camera viewport. 404 if the reel has not been ingested yet (or
    pose extraction failed during ingestion).
    """
    payload = raw_first_frame_cache.get(reel_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="skeleton not available")
    return payload