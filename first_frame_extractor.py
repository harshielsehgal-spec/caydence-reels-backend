"""
first_frame_extractor.py — Extract creator's first usable frame in RAW MediaPipe
normalized coordinate space (0-1), NOT the hip-anchored / torso-scaled coords
that creator_kps_store and reference_cache use.

Used by the /reels/{reel_id}/skeleton endpoint to serve a ghost-overlay
reference for the Loveable recording component.

The cache is populated at creator-ingestion time only (three call sites in
rubric_router.py and reels_router.py). Attempt-ingestion endpoints do NOT
write to this cache.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cache — single source of truth, co-located with the extractor that writes it
# ─────────────────────────────────────────────────────────────────────────────
#
# Shape: { reel_id: { "frame_index": int,
#                     "landmarks": [{"index", "x", "y", "visibility"}, ...],
#                     "width": int, "height": int } }
#
# Matches existing in-memory dict pattern. No TTL, no eviction.

raw_first_frame_cache: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

# Primary landmarks for "qualifying frame" — same set the frontend will use
# for framing checks (shoulders, hips, knees).
_PRIMARY_LANDMARKS = (11, 12, 23, 24, 25, 26)
_QUALIFY_VISIBILITY = 0.7

# Full-body extras — head + ankles must also be visible for the picked frame
# to represent a fully-in-frame subject (not mid-action with head out of frame).
# Looser threshold because nose and ankles often have lower visibility than
# torso landmarks even when clearly in frame.
_FULL_BODY_LANDMARKS = (0, 27, 28)  # nose, L_ankle, R_ankle
_FULL_BODY_VISIBILITY = 0.5

_MAX_SEARCH_FRAMES = 30

_mp_pose = mp.solutions.pose


def _landmarks_to_payload(lms) -> list[dict]:
    """Convert MediaPipe landmark proto list to the API JSON shape."""
    return [
        {
            "index": idx,
            "x": float(lm.x),
            "y": float(lm.y),
            "visibility": float(lm.visibility),
        }
        for idx, lm in enumerate(lms)
    ]


def _qualifies(lms) -> bool:
    """
    Frame qualifies when:
      - all primary landmarks (shoulders, hips, knees) above 0.7 visibility, AND
      - nose and both ankles above 0.5 visibility (full body in frame)

    The looser threshold for head/ankles reflects that they're often partially
    occluded or near edges even when clearly visible to a human viewer.
    """
    primary_ok = all(lms[i].visibility >= _QUALIFY_VISIBILITY for i in _PRIMARY_LANDMARKS)
    full_body_ok = all(lms[i].visibility >= _FULL_BODY_VISIBILITY for i in _FULL_BODY_LANDMARKS)
    return primary_ok and full_body_ok


def extract_raw_first_frame(video_path: str) -> Optional[dict]:
    """
    Find the first qualifying frame in the first 30 frames of `video_path`
    and return its pose landmarks in raw MediaPipe normalized space (0-1).

    Qualifying = all primary landmarks (11, 12, 23, 24, 25, 26) above 0.7
    visibility. If none of the first 30 frames qualify, falls back to the
    first frame with any pose detected at all.

    Returns the dict shape used by the cache, or None if no pose is detected
    at all in the first 30 frames.

    Never raises. On any unexpected exception, logs and returns None.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning("first_frame: could not open %s", video_path)
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        fallback_payload: Optional[dict] = None

        with _mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        ) as pose:
            for frame_idx in range(_MAX_SEARCH_FRAMES):
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                r = pose.process(rgb)
                if not r.pose_landmarks:
                    continue
                lms = r.pose_landmarks.landmark
                payload = {
                    "frame_index": frame_idx,
                    "landmarks": _landmarks_to_payload(lms),
                    "width": width,
                    "height": height,
                }
                # Keep the first detected pose as fallback in case nothing qualifies
                if fallback_payload is None:
                    fallback_payload = payload
                if _qualifies(lms):
                    cap.release()
                    return payload

        cap.release()
        return fallback_payload  # may be None if no pose detected at all

    except Exception:
        log.exception("first_frame: extraction error for %s", video_path)
        return None


def populate_cache_safe(reel_id: str, video_path: str) -> None:
    """
    Wrapper for use at creator-ingestion call sites. Never raises.
    Logs and continues on any failure so the parent endpoint isn't affected.
    """
    try:
        result = extract_raw_first_frame(video_path)
        if result is not None:
            raw_first_frame_cache[reel_id] = result
            log.info(
                "first_frame: cached reel_id=%s frame_index=%d landmarks=%d",
                reel_id, result["frame_index"], len(result["landmarks"]),
            )
        else:
            log.warning("first_frame: no qualifying pose for reel_id=%s", reel_id)
    except Exception:
        log.exception("first_frame: populate_cache_safe failed for %s", reel_id)