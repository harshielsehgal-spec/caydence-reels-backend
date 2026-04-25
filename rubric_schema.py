"""
rubric_schema.py — Rubric data model for Caydence Reels scoring engine.

A rubric defines how a creator's reel is scored. It consists of:
  - key_frames: labeled moments in the creator's video (e.g. "contact", "release")
  - checks:     geometric measurements taken at those key frames, each with a
                target value, tolerance, and weight

Targets are auto-extracted from the creator's reference video at rubric-creation
time; the creator only picks (a) the key frame timestamp and (b) which landmarks
to measure. Tolerance defaults are set per check type.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Check types
# ─────────────────────────────────────────────────────────────────────────────

CheckType = Literal["angle", "distance", "alignment", "relative_position"]

# Default tolerance per check type. Tuned loosely; creator can override.
#   angle:             ± degrees
#   distance:          ± fraction of torso-height (normalized units)
#   alignment:         max perpendicular deviation in torso-heights
#   relative_position: not used (binary-ish check; tolerance repurposed as margin)
DEFAULT_TOLERANCE: dict[CheckType, float] = {
    "angle": 15.0,
    "distance": 0.15,
    "alignment": 0.10,
    "relative_position": 0.05,
}

# Landmark count required per check type (before target auto-extraction).
REQUIRED_LANDMARKS: dict[CheckType, int] = {
    "angle": 3,              # p1 - p2 (vertex) - p3
    "distance": 2,           # p1 to p2
    "alignment": 3,          # p1, p2, p3 should be collinear
    "relative_position": 2,  # p1 relative to p2 on specified axis
}

# ─────────────────────────────────────────────────────────────────────────────
# Joint-specific default tolerances — FALLBACK for 1-rep videos
# ─────────────────────────────────────────────────────────────────────────────
#
# When the creator's video has only one rep, we can't auto-calibrate tolerance
# from variance (no variance exists). These defaults kick in.
#
# Values tuned from typical human movement ranges — elbows flex sharply,
# shoulders swing widely, knees deep. Looked up, not guessed.
#
# Keys are (vertex_landmark_index,) for angle checks, matching the middle
# landmark of the (p1, vertex, p3) triplet. Unknown vertex -> 15° generic.

DEFAULT_ANGLE_TOLERANCE_BY_VERTEX: dict[int, float] = {
    # Elbows — precise in fitness, more lenient in sports
    13: 12.0,  # left elbow
    14: 12.0,  # right elbow
    # Shoulders — wide natural range
    11: 20.0,  # left shoulder
    12: 20.0,  # right shoulder
    # Hips — wide natural range
    23: 18.0,  # left hip
    24: 18.0,  # right hip
    # Knees — deep flexion possible
    25: 15.0,  # left knee
    26: 15.0,  # right knee
    # Ankles — limited range
    27: 10.0,  # left ankle
    28: 10.0,  # right ankle
}

# Generic fallbacks if vertex isn't in the dict
GENERIC_ANGLE_TOLERANCE = 15.0
GENERIC_DISTANCE_TOLERANCE = 0.15
GENERIC_ALIGNMENT_TOLERANCE = 0.10
GENERIC_RELATIVE_POSITION_TOLERANCE = 0.08

# Minimum variance floor — if creator's own consistency is below this,
# use the defaults above instead. Prevents tolerance collapse on shaky videos.
MIN_VARIANCE_FLOOR: dict[CheckType, float] = {
    "angle": 3.0,            # degrees
    "distance": 0.02,        # torso-normalized units
    "alignment": 0.02,
    "relative_position": 0.015,
}


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class KeyFrame(BaseModel):
    """A labeled moment in the creator's reel."""
    id: str = Field(..., description="Creator-defined label, e.g. 'contact'")
    timestamp_ms: int = Field(..., ge=0, description="Time in ms from video start")


class Check(BaseModel):
    """A single geometric measurement in the rubric."""
    id: str = Field(..., description="Unique check ID, e.g. 'elbow_at_contact'")
    key_frame_id: str = Field(..., description="Which key_frame this check runs at")
    type: CheckType
    landmarks: list[int] = Field(..., description="MediaPipe landmark indices (0-32)")

    # Auto-filled from creator video at rubric-creation time.
    target_value: Optional[float] = Field(
        None,
        description="Auto-extracted from creator's pose at key_frame. "
                    "None until build_rubric() runs.",
    )

    # Defaults per check type; creator can override.
    tolerance: Optional[float] = Field(
        None,
        description="± acceptable deviation. None = use DEFAULT_TOLERANCE[type].",
    )

    weight: float = Field(..., gt=0, le=1, description="Weight in final score")

    min_visibility: float = Field(
        0.5,
        ge=0, le=1,
        description="Skip check if any required landmark in viewer frame is less visible",
    )

    # Axis for relative_position checks: "x" (horizontal) or "y" (vertical).
    # None for other check types.
    axis: Optional[Literal["x", "y"]] = None

    @field_validator("landmarks")
    @classmethod
    def _validate_landmark_indices(cls, v: list[int]) -> list[int]:
        if any(i < 0 or i > 32 for i in v):
            raise ValueError("Landmark indices must be in [0, 32]")
        return v

    @model_validator(mode="after")
    def _validate_landmark_count(self) -> "Check":
        required = REQUIRED_LANDMARKS[self.type]
        if len(self.landmarks) != required:
            raise ValueError(
                f"Check type '{self.type}' requires {required} landmarks, "
                f"got {len(self.landmarks)}"
            )
        if self.type == "relative_position" and self.axis is None:
            raise ValueError("relative_position checks must specify axis ('x' or 'y')")
        return self

    def effective_tolerance(self) -> float:
        return self.tolerance if self.tolerance is not None else DEFAULT_TOLERANCE[self.type]


class Rubric(BaseModel):
    """Complete rubric for a reel."""
    reel_id: str
    sport: Optional[str] = Field(None, description="Optional sport tag for analytics")
    key_frames: list[KeyFrame]
    checks: list[Check]

    @model_validator(mode="after")
    def _validate_references_and_weights(self) -> "Rubric":
        # Every check must reference a valid key_frame.
        kf_ids = {kf.id for kf in self.key_frames}
        for c in self.checks:
            if c.key_frame_id not in kf_ids:
                raise ValueError(
                    f"Check '{c.id}' references unknown key_frame '{c.key_frame_id}'"
                )

        # Weights must sum to ~1.0 (tolerate float drift).
        total_weight = sum(c.weight for c in self.checks)
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Check weights must sum to 1.0, got {total_weight:.4f}"
            )

        # Check IDs must be unique.
        ids = [c.id for c in self.checks]
        if len(ids) != len(set(ids)):
            raise ValueError("Check IDs must be unique within a rubric")

        return self


# ─────────────────────────────────────────────────────────────────────────────
# Supabase DDL
# ─────────────────────────────────────────────────────────────────────────────

SUPABASE_DDL = """
-- Run this in Supabase SQL editor.
--
-- One row per reel. Rubric JSON is stored as jsonb for flexibility;
-- validation happens in Python via Rubric.model_validate().

create table if not exists reels (
    id               uuid primary key default gen_random_uuid(),
    creator_user_id  uuid not null,
    video_url        text not null,
    sport            text,
    created_at       timestamptz not null default now()
);

create table if not exists rubrics (
    reel_id    uuid primary key references reels(id) on delete cascade,
    rubric     jsonb not null,
    version    int not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists attempts (
    id              uuid primary key default gen_random_uuid(),
    reel_id         uuid not null references reels(id) on delete cascade,
    viewer_user_id  uuid not null,
    video_url       text not null,
    score           int,
    breakdown       jsonb,
    confidence      real,
    created_at      timestamptz not null default now()
);

create index if not exists idx_attempts_reel_id on attempts(reel_id);
create index if not exists idx_attempts_viewer  on attempts(viewer_user_id);
"""


if __name__ == "__main__":
    # Sanity check: build an example rubric and validate it.
    example = Rubric(
        reel_id="test-reel-001",
        sport="cricket",
        key_frames=[
            KeyFrame(id="backlift", timestamp_ms=400),
            KeyFrame(id="contact", timestamp_ms=1420),
        ],
        checks=[
            Check(
                id="left_elbow_at_backlift",
                key_frame_id="backlift",
                type="angle",
                landmarks=[11, 13, 15],
                target_value=110.0,
                weight=0.3,
            ),
            Check(
                id="left_elbow_at_contact",
                key_frame_id="contact",
                type="angle",
                landmarks=[11, 13, 15],
                target_value=95.0,
                weight=0.4,
            ),
            Check(
                id="hip_rotation_at_contact",
                key_frame_id="contact",
                type="relative_position",
                landmarks=[23, 24],
                axis="x",
                target_value=0.12,
                weight=0.3,
            ),
        ],
    )
    print("Rubric validated:")
    print(example.model_dump_json(indent=2))