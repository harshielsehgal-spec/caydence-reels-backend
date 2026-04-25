"""
adaptive_rubric.py — Auto-build a rubric from the creator's video alone.

FINAL version with all review fixes + bug fixes from iteration:
  - Relative variance (kills ambient sway)
  - Bilateral pair grouping
  - Static hold fallback
  - 1-rep tolerance fallback
  - Variance floor
  - Action window cropping (kills setup/cleanup junk)
  - Plausibility filter on rep candidates
  - Tolerance hard cap at 40°
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from pose_similarity import (
    normalize_keypoints,
    _compute_frame_velocities,
    _detect_action_window,
)
from rubric_schema import (
    Check, KeyFrame, Rubric,
    DEFAULT_ANGLE_TOLERANCE_BY_VERTEX, GENERIC_ANGLE_TOLERANCE, MIN_VARIANCE_FLOOR,
)

JOINT_TRIPLETS: list[tuple[str, int, int, int]] = [
    ("l_elbow",    11, 13, 15),
    ("r_elbow",    12, 14, 16),
    ("l_shoulder", 13, 11, 23),
    ("r_shoulder", 14, 12, 24),
    ("l_hip",      11, 23, 25),
    ("r_hip",      12, 24, 26),
    ("l_knee",     23, 25, 27),
    ("r_knee",     24, 26, 28),
]
BILATERAL_PAIRS: list[tuple[str, str]] = [
    ("l_elbow", "r_elbow"), ("l_shoulder", "r_shoulder"),
    ("l_hip", "r_hip"),     ("l_knee", "r_knee"),
]
STATIC_HOLD_VARIANCE_THRESHOLD = 4.0
MAX_ANGLE_TOLERANCE = 40.0  # HARD CAP — applies to every angle check


def _angle_deg(p1, p2, p3):
    v1 = p1 - p2; v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))


def compute_joint_angles_per_frame(normalized_keypoints: np.ndarray) -> dict[str, np.ndarray]:
    xy = normalized_keypoints[:, :, :2]
    return {name: np.array([_angle_deg(f[a], f[b], f[c]) for f in xy])
            for name, a, b, c in JOINT_TRIPLETS}


def _compute_torso_angle(xy: np.ndarray) -> np.ndarray:
    sm = (xy[:, 11] + xy[:, 12]) / 2
    hm = (xy[:, 23] + xy[:, 24]) / 2
    return np.degrees(np.arctan2(sm[:, 0] - hm[:, 0], -(sm[:, 1] - hm[:, 1])))


def compute_relative_variance(joint_angles, normalized_keypoints) -> dict[str, float]:
    torso = _compute_torso_angle(normalized_keypoints[:, :, :2])
    return {name: float(np.var(a - torso, ddof=0)) for name, a in joint_angles.items()}


def compute_importance_weights(rel_variance: dict[str, float]) -> dict[str, float]:
    if not rel_variance:
        return {}
    n = len(rel_variance)
    max_var = max(rel_variance.values())
    if max_var < STATIC_HOLD_VARIANCE_THRESHOLD:
        return {name: 1.0 / n for name in rel_variance}
    total = sum(rel_variance.values())
    if total <= 0:
        return {name: 1.0 / n for name in rel_variance}
    paired: dict[str, float] = {}
    used: set[str] = set()
    for l, r in BILATERAL_PAIRS:
        if l in rel_variance and r in rel_variance:
            paired[f"{l}+{r}"] = rel_variance[l] + rel_variance[r]
            used.update({l, r})
    for name, v in rel_variance.items():
        if name not in used:
            paired[name] = v
    grp_total = sum(paired.values())
    if grp_total <= 0:
        return {name: 1.0 / n for name in rel_variance}
    out: dict[str, float] = {}
    for g, v in paired.items():
        gw = v / grp_total
        if "+" in g:
            a, b = g.split("+")
            out[a] = gw / 2; out[b] = gw / 2
        else:
            out[g] = gw
    return out


def _smooth(s: np.ndarray, window: int = 5) -> np.ndarray:
    if len(s) < window:
        return s.copy()
    return np.convolve(s, np.ones(window) / window, mode="same")


def find_peaks_with_prominence(series, min_prominence, min_distance):
    n = len(series)
    if n < 3:
        return []
    peaks: list[int] = []
    i = 1
    while i < n - 1:
        if series[i] > series[i - 1] and series[i] >= series[i + 1]:
            ls = max(0, i - min_distance); re = min(n, i + min_distance + 1)
            lmin = float(np.min(series[ls:i])) if i > ls else series[i]
            rmin = float(np.min(series[i + 1:re])) if i + 1 < re else series[i]
            if (series[i] - max(lmin, rmin)) >= min_prominence:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
                    i += min_distance
                    continue
        i += 1
    return peaks


def detect_rep_extrema(joint_angles, weights, fps):
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    if not ranked:
        return [], [], ""
    primary = ranked[0][0]
    if primary not in joint_angles:
        return [], [], primary
    series = _smooth(joint_angles[primary], 5)
    std = float(np.std(series))
    prom = max(std * 0.35, 5.0)
    gap = max(6, int(fps * 0.3))
    minima = find_peaks_with_prominence(-series, prom, gap)
    maxima = find_peaks_with_prominence(series, prom, gap)
    return minima, maxima, primary


def _default_angle_tol(vertex_idx: int) -> float:
    return DEFAULT_ANGLE_TOLERANCE_BY_VERTEX.get(vertex_idx, GENERIC_ANGLE_TOLERANCE)


def calibrate_tolerance(values_across_reps, vertex_idx):
    if len(values_across_reps) < 2:
        return _default_angle_tol(vertex_idx)
    std = float(np.std(values_across_reps, ddof=0))
    if std < MIN_VARIANCE_FLOOR["angle"]:
        return _default_angle_tol(vertex_idx)
    return float(max(std * 2.0, MIN_VARIANCE_FLOOR["angle"]))


def crop_to_action_window(keypoints: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Find the longest contiguous window where the body is in a stable position
    (e.g. plank for pushups, standing for squats). Junk frames at start/end
    typically have high torso angle variance because the person is walking
    or transitioning into position.

    Returns (cropped, start_idx, end_idx). Refuses to crop if it would remove
    more than 70% of the video.
    """
    if len(keypoints) < 20:
        return keypoints, 0, len(keypoints)

    xy = keypoints[:, :, :2]
    # Torso angle: orientation of shoulder-midpoint to hip-midpoint vector
    sm = (xy[:, 11] + xy[:, 12]) / 2
    hm = (xy[:, 23] + xy[:, 24]) / 2
    torso = np.degrees(np.arctan2(sm[:, 0] - hm[:, 0], -(sm[:, 1] - hm[:, 1])))

    # Rolling std of torso angle. Stable region (in plank/standing) = low std.
    # Walking / setup = high std.
    window = max(10, len(torso) // 20)
    rolling_std = np.array([
        float(np.std(torso[max(0, i - window // 2):min(len(torso), i + window // 2 + 1)]))
        for i in range(len(torso))
    ])

    threshold = float(np.median(rolling_std))
    stable = rolling_std < threshold

    # Find longest contiguous "stable" run
    best_start, best_end, best_len = 0, len(keypoints), 0
    cur_start: Optional[int] = None
    for i, s in enumerate(stable):
        if s and cur_start is None:
            cur_start = i
        elif not s and cur_start is not None:
            run_len = i - cur_start
            if run_len > best_len:
                best_start, best_end, best_len = cur_start, i, run_len
            cur_start = None
    if cur_start is not None:
        run_len = len(stable) - cur_start
        if run_len > best_len:
            best_start, best_end, best_len = cur_start, len(stable), run_len

    # Refuse pathological crops
    if best_end - best_start < max(20, int(len(keypoints) * 0.3)):
        return keypoints, 0, len(keypoints)

    return keypoints[best_start:best_end], best_start, best_end


def _filter_plausible_rep_frames(rep_frames, joint_angles, primary_joint) -> list[int]:
    """Drop rep candidates >30° from median primary joint angle."""
    if len(rep_frames) < 2 or primary_joint not in joint_angles:
        return rep_frames
    vals = [float(joint_angles[primary_joint][f]) for f in rep_frames
            if 0 <= f < len(joint_angles[primary_joint])]
    if not vals:
        return rep_frames
    median_val = float(np.median(vals))
    keep = [f for f in rep_frames
            if 0 <= f < len(joint_angles[primary_joint])
            and abs(joint_angles[primary_joint][f] - median_val) <= 30.0]
    return keep if keep else rep_frames


@dataclass
class AdaptiveRubricResult:
    rubric: Rubric
    importance_weights: dict[str, float]
    rep_bottoms: list[int]
    rep_tops: list[int]
    primary_joint: str
    fps: float
    static_hold: bool
    debug: dict = field(default_factory=dict)


def _f2ms(f: int, fps: float) -> int:
    return int(round((f / max(fps, 1e-6)) * 1000))


def build_adaptive_rubric(creator_keypoints, reel_id, fps=30.0):
    if len(creator_keypoints) < 5:
        raise ValueError("Need at least 5 frames")

    # STEP 0 — crop to action window, kill setup/cleanup junk
    cropped, crop_start, crop_end = crop_to_action_window(creator_keypoints)
    norm = normalize_keypoints(cropped)
    joint_angles = compute_joint_angles_per_frame(norm)
    rel_var = compute_relative_variance(joint_angles, norm)
    weights = compute_importance_weights(rel_var)
    static_hold = max(rel_var.values()) < STATIC_HOLD_VARIANCE_THRESHOLD
    bottoms, tops, primary = detect_rep_extrema(joint_angles, weights, fps)
    bottoms = _filter_plausible_rep_frames(bottoms, joint_angles, primary)
    tops = _filter_plausible_rep_frames(tops, joint_angles, primary)

    if static_hold:
        rep_frames = [len(norm) // 2]; kf_id = "hold"
    elif bottoms:
        rep_frames = bottoms; kf_id = "bottom"
    elif tops:
        rep_frames = tops; kf_id = "top"
    else:
        rep_frames = [len(norm) // 2]; kf_id = "midpoint"

    rep_frame = int(np.median(rep_frames))
    original_rep_frame = rep_frame + crop_start
    rubric_kf = [KeyFrame(id=kf_id, timestamp_ms=_f2ms(original_rep_frame, fps))]

    checks: list[Check] = []
    for name, a, b, c in JOINT_TRIPLETS:
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        vals = [float(joint_angles[name][rf]) for rf in rep_frames
                if 0 <= rf < len(joint_angles[name])]
        if not vals:
            continue
        target = float(np.median(vals))
        tol = calibrate_tolerance(vals, vertex_idx=b)
        tol = min(tol, MAX_ANGLE_TOLERANCE)  # HARD CAP
        checks.append(Check(
            id=f"{name}_at_{kf_id}",
            key_frame_id=kf_id,
            type="angle",
            landmarks=[a, b, c],
            target_value=target,
            tolerance=tol,
            weight=w,
            min_visibility=0.4,
        ))
    if not checks:
        raise ValueError("No meaningful motion — cannot build rubric")
    total = sum(c.weight for c in checks)
    for c in checks:
        c.weight = c.weight / total

    bottoms_orig = [b + crop_start for b in bottoms]
    tops_orig = [t + crop_start for t in tops]

    rubric = Rubric(reel_id=reel_id, sport=None, key_frames=rubric_kf, checks=checks)
    return AdaptiveRubricResult(
        rubric=rubric, importance_weights=weights,
        rep_bottoms=bottoms_orig, rep_tops=tops_orig,
        primary_joint=primary, fps=fps, static_hold=static_hold,
        debug={
            "relative_variance": rel_var,
            "rep_count_bottoms": len(bottoms),
            "rep_count_tops": len(tops),
            "crop_start": crop_start,
            "crop_end": crop_end,
            "cropped_frames": len(norm),
            "original_frames": len(creator_keypoints),
        },
    )