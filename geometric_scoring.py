"""
geometric_scoring.py — Pure geometric check functions for Caydence rubric scoring.

Each check runs on a single normalized pose frame (output of normalize_keypoints
from pose_similarity.py, but with the visibility channel preserved).

Input frame shape: (33, 3) — x, y, visibility. x, y are in torso-normalized
coordinates (hip-centered, torso-scaled).

Each function returns CheckResult with:
  - score:      0-100, where 100 = exact match, 0 = outside tolerance
  - observed:   the measured value (for debugging / breakdown UI)
  - confidence: based on visibility of the landmarks this check depends on
  - passed:     True if confidence >= min_visibility
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rubric_schema import Check


@dataclass
class CheckResult:
    check_id: str
    score: int                 # 0-100
    observed: Optional[float]  # None if skipped due to visibility
    target: float
    tolerance: float
    confidence: float          # min visibility across required landmarks
    passed: bool               # confidence >= min_visibility
    weight: float              # pass-through from check definition


# ─────────────────────────────────────────────────────────────────────────────
# Geometric primitives
# ─────────────────────────────────────────────────────────────────────────────

def _angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at vertex p2, formed by rays p2->p1 and p2->p3, in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points (already torso-normalized)."""
    return float(np.linalg.norm(p1 - p2))


def _perp_deviation(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Perpendicular distance from p2 to the line through p1 and p3.
    Used to measure collinearity — 0 means p1, p2, p3 are perfectly aligned.
    """
    line_vec = p3 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return 0.0
    point_vec = p2 - p1
    # |cross| / |line| gives the perpendicular distance in 2D
    cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    return float(abs(cross) / line_len)


def _signed_axis_delta(p1: np.ndarray, p2: np.ndarray, axis: str) -> float:
    """Signed delta on a single axis (p1 - p2). Positive = p1 is right/below p2."""
    idx = 0 if axis == "x" else 1
    return float(p1[idx] - p2[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _linear_score(observed: float, target: float, tolerance: float) -> int:
    """
    Linear falloff scoring:
      |observed - target| == 0          -> 100
      |observed - target| >= tolerance  -> 0
    """
    if tolerance <= 0:
        return 100 if observed == target else 0
    delta = abs(observed - target)
    score = 100 * max(0.0, 1.0 - delta / tolerance)
    return int(round(score))


# ─────────────────────────────────────────────────────────────────────────────
# Check runners
# ─────────────────────────────────────────────────────────────────────────────

def _extract_landmarks(frame: np.ndarray, indices: list[int]) -> tuple[list[np.ndarray], float]:
    """Return (xy_points, min_visibility) for the requested landmark indices."""
    xy = [frame[i, :2] for i in indices]
    vis = float(min(frame[i, 2] for i in indices))
    return xy, vis


def run_check(check: Check, frame: np.ndarray) -> CheckResult:
    """Dispatch a single check against a normalized pose frame."""
    pts, visibility = _extract_landmarks(frame, check.landmarks)
    tolerance = check.effective_tolerance()
    target = check.target_value
    if target is None:
        raise ValueError(
            f"Check '{check.id}' has no target_value — rubric not built yet"
        )

    passed = visibility >= check.min_visibility

    if not passed:
        return CheckResult(
            check_id=check.id, score=0, observed=None,
            target=target, tolerance=tolerance,
            confidence=visibility, passed=False, weight=check.weight,
        )

    # Dispatch by check type
    if check.type == "angle":
        observed = _angle_deg(pts[0], pts[1], pts[2])
    elif check.type == "distance":
        observed = _distance(pts[0], pts[1])
    elif check.type == "alignment":
        observed = _perp_deviation(pts[0], pts[1], pts[2])
        # For alignment, target is always 0 (perfectly collinear). We still
        # score linearly against tolerance.
        target = 0.0
    elif check.type == "relative_position":
        observed = _signed_axis_delta(pts[0], pts[1], check.axis or "x")
    else:
        raise ValueError(f"Unknown check type: {check.type}")

    score = _linear_score(observed, target, tolerance)

    return CheckResult(
        check_id=check.id,
        score=score,
        observed=observed,
        target=target,
        tolerance=tolerance,
        confidence=visibility,
        passed=True,
        weight=check.weight,
    )


def aggregate_scores(results: list[CheckResult]) -> tuple[int, float]:
    """
    Weighted average of check scores, ignoring failed (low-visibility) checks.
    Returns (overall_score, overall_confidence).

    If all checks failed, returns (0, 0.0).
    If some passed, weights are re-normalized across the passing checks so
    one occluded joint doesn't cap the max score artificially.
    """
    passing = [r for r in results if r.passed]
    if not passing:
        return 0, 0.0

    total_weight = sum(r.weight for r in passing)
    if total_weight <= 0:
        return 0, 0.0

    weighted = sum(r.score * r.weight for r in passing) / total_weight
    avg_conf = sum(r.confidence for r in passing) / len(passing)
    return int(round(weighted)), round(avg_conf, 3)


if __name__ == "__main__":
    # Smoke test: build a fake normalized frame and run each check type.
    np.random.seed(42)
    frame = np.zeros((33, 3))
    # Put some plausible torso-normalized coords
    frame[11] = [-0.3, -1.0, 0.9]   # L shoulder
    frame[13] = [-0.5, -0.3, 0.9]   # L elbow
    frame[15] = [-0.2, 0.2, 0.9]    # L wrist
    frame[23] = [-0.2, 0.0, 0.95]   # L hip
    frame[24] = [0.2, 0.0, 0.95]    # R hip
    frame[12] = [0.3, -1.0, 0.9]    # R shoulder

    from rubric_schema import Check

    angle_check = Check(
        id="l_elbow", key_frame_id="kf1", type="angle",
        landmarks=[11, 13, 15], target_value=90.0, weight=0.5,
    )
    dist_check = Check(
        id="hip_width", key_frame_id="kf1", type="distance",
        landmarks=[23, 24], target_value=0.4, weight=0.25,
    )
    relpos_check = Check(
        id="hips_level", key_frame_id="kf1", type="relative_position",
        landmarks=[24, 23], axis="x", target_value=0.4, weight=0.25,
    )

    results = [
        run_check(angle_check, frame),
        run_check(dist_check, frame),
        run_check(relpos_check, frame),
    ]
    for r in results:
        print(r)

    overall, conf = aggregate_scores(results)
    print(f"\nOverall: {overall}  Confidence: {conf}")
