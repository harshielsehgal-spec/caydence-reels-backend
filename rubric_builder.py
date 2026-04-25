"""
rubric_builder.py — Build a complete Rubric from a creator's video.

Input: creator keypoints + a "rubric spec" (key frame timestamps + which
landmarks to measure at each). Output: a Rubric with target_value filled
in for every check, auto-extracted from the creator's pose at the labeled
timestamps.

This is called once per reel, at creation time. After this runs, the rubric
is stored and used to score every viewer attempt for that reel.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from pose_similarity import normalize_keypoints
from rubric_schema import (
    Check,
    CheckType,
    KeyFrame,
    Rubric,
    DEFAULT_TOLERANCE,
    REQUIRED_LANDMARKS,
)
from geometric_scoring import (
    _angle_deg,
    _distance,
    _perp_deviation,
    _signed_axis_delta,
)


# ─────────────────────────────────────────────────────────────────────────────
# Spec models — what the creator sends to build a rubric
# ─────────────────────────────────────────────────────────────────────────────
#
# A rubric spec is the *intent* — key frames and what to measure at each.
# It does NOT contain target_value. Targets are extracted from the creator's
# video by build_rubric().

class CheckSpec(BaseModel):
    id: str
    key_frame_id: str
    type: CheckType
    landmarks: list[int]
    weight: float = Field(..., gt=0, le=1)
    tolerance: Optional[float] = None
    min_visibility: float = 0.5
    axis: Optional[Literal["x", "y"]] = None


class RubricSpec(BaseModel):
    reel_id: str
    sport: Optional[str] = None
    key_frames: list[KeyFrame]
    checks: list[CheckSpec]


# ─────────────────────────────────────────────────────────────────────────────
# Frame timestamp resolution
# ─────────────────────────────────────────────────────────────────────────────

def timestamp_to_frame_index(
    timestamp_ms: int,
    total_frames: int,
    video_duration_ms: int,
) -> int:
    """
    Map a wall-clock timestamp (from the creator's video) to a frame index
    within the *sampled* keypoint array.

    Why this is non-trivial: extract_keypoints_phased does non-uniform sampling
    (dense around action, sparse elsewhere), so the returned keypoints array
    is NOT a uniformly-sampled view of the video. For MVP we assume uniform
    sampling for target extraction — we call extract_keypoints (uniform),
    not extract_keypoints_phased, when building rubrics. The caller must
    pass total_frames = len(keypoints_uniform).
    """
    if video_duration_ms <= 0:
        return 0
    fraction = max(0.0, min(1.0, timestamp_ms / video_duration_ms))
    return int(round(fraction * (total_frames - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Target extraction — the core of auto-filling
# ─────────────────────────────────────────────────────────────────────────────

def extract_target_value(
    frame: np.ndarray,
    check_type: CheckType,
    landmarks: list[int],
    axis: Optional[str] = None,
) -> float:
    """
    Measure the geometric quantity for a check against a single frame.
    Used to populate target_value from the creator's own pose.

    `frame` must be a normalized pose frame of shape (33, 3) or (33, 2).
    Only xy are used; visibility channel is ignored here.
    """
    xy = frame[:, :2] if frame.shape[1] >= 3 else frame
    pts = [xy[i] for i in landmarks]

    if check_type == "angle":
        return _angle_deg(pts[0], pts[1], pts[2])
    if check_type == "distance":
        return _distance(pts[0], pts[1])
    if check_type == "alignment":
        # Target for alignment is always 0 (perfect collinearity).
        # We still return the observed deviation so the creator can see it
        # and decide whether their own pose was actually aligned.
        return _perp_deviation(pts[0], pts[1], pts[2])
    if check_type == "relative_position":
        return _signed_axis_delta(pts[0], pts[1], axis or "x")
    raise ValueError(f"Unknown check type: {check_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Rubric builder
# ─────────────────────────────────────────────────────────────────────────────

def build_rubric(
    spec: RubricSpec,
    creator_keypoints: np.ndarray,
    creator_video_duration_ms: int,
) -> Rubric:
    """
    Build a validated Rubric with auto-extracted target values.

    Args:
        spec: creator's rubric intent (key frames + check definitions, no targets)
        creator_keypoints: UNIFORMLY sampled keypoint array from creator video,
                          shape (N, 33, 3). Call pose_similarity.extract_keypoints
                          (NOT extract_keypoints_phased) so timestamp→frame mapping
                          stays linear.
        creator_video_duration_ms: total video duration for timestamp resolution

    Returns:
        Fully populated Rubric. target_value is filled for every check.
        For alignment checks, target is forced to 0.0 regardless of the
        creator's actual deviation (rubrics measure deviation from ideal).
    """
    normalized = normalize_keypoints(creator_keypoints)

    # Resolve each key frame timestamp to a frame index
    kf_to_frame: dict[str, int] = {}
    for kf in spec.key_frames:
        kf_to_frame[kf.id] = timestamp_to_frame_index(
            kf.timestamp_ms, len(normalized), creator_video_duration_ms
        )

    # Build each check with its extracted target
    built_checks: list[Check] = []
    for cs in spec.checks:
        if len(cs.landmarks) != REQUIRED_LANDMARKS[cs.type]:
            raise ValueError(
                f"Check '{cs.id}' ({cs.type}) needs "
                f"{REQUIRED_LANDMARKS[cs.type]} landmarks, got {len(cs.landmarks)}"
            )

        if cs.key_frame_id not in kf_to_frame:
            raise ValueError(
                f"Check '{cs.id}' references unknown key frame '{cs.key_frame_id}'"
            )

        frame_idx = kf_to_frame[cs.key_frame_id]
        frame = normalized[frame_idx]

        observed = extract_target_value(frame, cs.type, cs.landmarks, cs.axis)

        # Alignment checks: target is ideal (0), not observed.
        # Everything else: target is what the creator themselves did.
        target = 0.0 if cs.type == "alignment" else observed

        built_checks.append(Check(
            id=cs.id,
            key_frame_id=cs.key_frame_id,
            type=cs.type,
            landmarks=cs.landmarks,
            target_value=float(target),
            tolerance=cs.tolerance,  # None -> defaults applied at scoring time
            weight=cs.weight,
            min_visibility=cs.min_visibility,
            axis=cs.axis,
        ))

    return Rubric(
        reel_id=spec.reel_id,
        sport=spec.sport,
        key_frames=spec.key_frames,
        checks=built_checks,
    )


if __name__ == "__main__":
    # Smoke test
    import sys
    class Stub:
        def __getattr__(self, k): return Stub()
    sys.modules.setdefault('mediapipe', Stub())
    sys.modules.setdefault('mediapipe.solutions', Stub())

    np.random.seed(0)
    creator_kps = np.random.rand(60, 33, 3).astype(np.float32)
    creator_kps[:, :, 2] = 0.9

    spec = RubricSpec(
        reel_id="demo-001",
        sport="cricket_bowling",
        key_frames=[
            KeyFrame(id="release", timestamp_ms=1200),
        ],
        checks=[
            CheckSpec(
                id="right_elbow_at_release",
                key_frame_id="release",
                type="angle",
                landmarks=[12, 14, 16],
                weight=0.6,
            ),
            CheckSpec(
                id="hip_rotation_at_release",
                key_frame_id="release",
                type="relative_position",
                landmarks=[24, 23],
                axis="x",
                weight=0.4,
            ),
        ],
    )

    rubric = build_rubric(spec, creator_kps, creator_video_duration_ms=2000)
    print("Built rubric:")
    print(rubric.model_dump_json(indent=2))
