"""
rep_segmentation.py — Detect reps, pair creator↔viewer reps, slice for per-rep scoring.

Crops to action window first (same as adaptive rubric builder), then detects
reps in the cropped sequence, then translates frame indices back to original
coordinates so downstream scoring uses the same timeline as the input.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from adaptive_rubric import (
    compute_joint_angles_per_frame,
    compute_relative_variance,
    compute_importance_weights,
    detect_rep_extrema,
    crop_to_action_window,
    _filter_plausible_rep_frames,
    _smooth,
)
from pose_similarity import normalize_keypoints


@dataclass
class RepSegment:
    start_frame: int
    peak_frame: int
    end_frame: int
    peak_value: float


def segment_reps(
    keypoints: np.ndarray,
    fps: float = 30.0,
    prefer: str = "bottoms",
) -> tuple[list[RepSegment], str]:
    if len(keypoints) < 5:
        return [], ""
    cropped, crop_start, _ = crop_to_action_window(keypoints)
    norm = normalize_keypoints(cropped)
    angles = compute_joint_angles_per_frame(norm)
    rel_var = compute_relative_variance(angles, norm)
    weights = compute_importance_weights(rel_var)
    bottoms, tops, primary = detect_rep_extrema(angles, weights, fps)
    bottoms = _filter_plausible_rep_frames(bottoms, angles, primary)
    tops = _filter_plausible_rep_frames(tops, angles, primary)

    if prefer == "bottoms" and bottoms:
        peaks = bottoms
    elif tops:
        peaks = tops
    elif bottoms:
        peaks = bottoms
    else:
        return [], primary

    if not primary or primary not in angles:
        return [], primary

    primary_series = _smooth(angles[primary], 5)
    segments: list[RepSegment] = []
    n = len(cropped)
    for i, peak in enumerate(peaks):
        prev_peak = peaks[i - 1] if i > 0 else 0
        next_peak = peaks[i + 1] if i < len(peaks) - 1 else n - 1
        start = (prev_peak + peak) // 2 if i > 0 else 0
        end = (peak + next_peak) // 2 if i < len(peaks) - 1 else n - 1
        segments.append(RepSegment(
            start_frame=start + crop_start,
            peak_frame=peak + crop_start,
            end_frame=end + crop_start,
            peak_value=float(primary_series[peak]),
        ))
    return segments, primary


def pair_reps(
    creator_reps: list[RepSegment],
    viewer_reps: list[RepSegment],
    creator_total_frames: int,
    viewer_total_frames: int,
) -> list[tuple[RepSegment, RepSegment]]:
    if not creator_reps or not viewer_reps:
        return []
    c_norms = [r.peak_frame / max(creator_total_frames, 1) for r in creator_reps]
    v_norms = [r.peak_frame / max(viewer_total_frames, 1) for r in viewer_reps]
    pairs: list[tuple[RepSegment, RepSegment]] = []
    used: set[int] = set()
    for vi, vn in enumerate(v_norms):
        best_ci, best_dist = -1, float("inf")
        for ci, cn in enumerate(c_norms):
            if ci in used:
                continue
            d = abs(cn - vn)
            if d < best_dist:
                best_dist = d
                best_ci = ci
        if best_ci >= 0:
            pairs.append((creator_reps[best_ci], viewer_reps[vi]))
            used.add(best_ci)
    pairs.sort(key=lambda p: p[0].peak_frame)
    return pairs


@dataclass
class RepScoreInput:
    creator_slice: np.ndarray
    viewer_slice: np.ndarray
    creator_peak_local: int
    viewer_peak_local: int


def extract_rep_slices(
    creator_kps: np.ndarray,
    viewer_kps: np.ndarray,
    pair: tuple[RepSegment, RepSegment],
) -> RepScoreInput:
    c_rep, v_rep = pair
    c_slice = creator_kps[c_rep.start_frame:c_rep.end_frame + 1]
    v_slice = viewer_kps[v_rep.start_frame:v_rep.end_frame + 1]
    return RepScoreInput(
        creator_slice=c_slice,
        viewer_slice=v_slice,
        creator_peak_local=c_rep.peak_frame - c_rep.start_frame,
        viewer_peak_local=v_rep.peak_frame - v_rep.start_frame,
    )