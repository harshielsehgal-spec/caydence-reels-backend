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
    diagnose: bool = False,
) -> tuple[list[RepSegment], str] | tuple[list[RepSegment], str, dict]:
    """
    Segment a keypoint sequence into reps.

    When diagnose=True, returns a 3-tuple (segments, primary_joint, diag_dict)
    where diag_dict captures rep-detection internals for debugging:
      - trim_stage: crop_to_action_window decisions
      - primary_signal: which joint was selected, signal stats, raw series
      - peak_detection: pre-filter candidate peaks/bottoms + prominences
      - filters_applied: ordered filter chain showing what was rejected
      - final_reps: surviving rep ranges

    Default (diagnose=False) preserves the original 2-tuple return contract.
    """
    if not diagnose:
        return _segment_reps_impl(keypoints, fps, prefer, capture=None)

    capture: dict = {
        "trim_stage": {},
        "primary_signal": {},
        "peak_detection": {},
        "filters_applied": [],
        "final_reps": [],
    }
    segments, primary = _segment_reps_impl(keypoints, fps, prefer, capture=capture)
    capture["final_reps"] = [
        {"start": int(s.start_frame), "peak": int(s.peak_frame), "end": int(s.end_frame)}
        for s in segments
    ]
    return segments, primary, capture


def _segment_reps_impl(
    keypoints: np.ndarray,
    fps: float,
    prefer: str,
    capture: Optional[dict],
) -> tuple[list[RepSegment], str]:
    if len(keypoints) < 5:
        if capture is not None:
            capture["trim_stage"] = {
                "ran": False, "skipped": True,
                "skip_reason": "input_too_short",
                "input_frame_count": int(len(keypoints)),
                "trimmed_frame_range": None,
            }
        return [], ""

    cropped, crop_start, crop_end = crop_to_action_window(keypoints)
    if capture is not None:
        cropped_full_match = (
            crop_start == 0 and crop_end == len(keypoints)
        )
        capture["trim_stage"] = {
            "ran": not cropped_full_match,
            "input_frame_count": int(len(keypoints)),
            "trimmed_frame_range": [int(crop_start), int(crop_end)],
            "skipped": cropped_full_match,
            "skip_reason": (
                "would_crop_over_70_percent_or_input_too_short"
                if cropped_full_match else None
            ),
        }

    norm = normalize_keypoints(cropped)
    angles = compute_joint_angles_per_frame(norm)
    rel_var = compute_relative_variance(angles, norm)
    weights = compute_importance_weights(rel_var)

    bottoms, tops, primary = detect_rep_extrema(angles, weights, fps)

    # Capture primary signal + pre-filter peak detection state
    if capture is not None:
        if primary and primary in angles:
            series = _smooth(angles[primary], 5)
            std = float(np.std(series))
            prom = max(std * 0.35, 5.0)
            # Downsample series if > 500 samples for payload size
            raw_series_len = len(series)
            if raw_series_len > 500:
                step = raw_series_len // 500
                downsampled = series[::step][:500]
                downsampled_note = (
                    f"downsampled from {raw_series_len} to {len(downsampled)} samples"
                )
            else:
                downsampled = series
                downsampled_note = None

            ranked = sorted(weights.items(), key=lambda kv: -kv[1])
            top3 = ranked[:3]
            rationale = (
                f"highest importance weight ({top3[0][1]:.4f}); "
                f"top 3: " + ", ".join(f"{n}={w:.3f}" for n, w in top3)
            )

            capture["primary_signal"] = {
                "joint_name": primary,
                "selection_rationale": rationale,
                "signal_length": int(raw_series_len),
                "signal_min": round(float(np.min(series)), 4),
                "signal_max": round(float(np.max(series)), 4),
                "signal_mean": round(float(np.mean(series)), 4),
                "signal_std": round(std, 4),
                "signal_amplitude": round(
                    float(np.max(series) - np.min(series)), 4
                ),
                "raw_series": [round(float(v), 4) for v in downsampled],
                "downsample_note": downsampled_note,
            }

            # Re-run peak detection in capture mode to get prominences
            bottom_proms = _peak_prominences(-series, prom, max(6, int(fps * 0.3)))
            top_proms = _peak_prominences(series, prom, max(6, int(fps * 0.3)))
            capture["peak_detection"] = {
                "all_peaks_found": [int(p) for p in tops],
                "all_bottoms_found": [int(b) for b in bottoms],
                "prominence_threshold_used": round(prom, 4),
                "min_distance_used": int(max(6, int(fps * 0.3))),
                "peak_prominences": [round(float(p), 4) for p in top_proms],
                "bottom_prominences": [round(float(p), 4) for p in bottom_proms],
                "note": (
                    "Bottoms = local minima of primary joint angle series. "
                    "All listed entries already passed the prominence threshold; "
                    "if list is short, candidates below threshold were rejected at "
                    "detection time and not surfaced separately by find_peaks_with_prominence."
                ),
            }
        else:
            capture["primary_signal"] = {
                "joint_name": primary or None,
                "selection_rationale": "no joint selected — empty weights or angles",
                "signal_length": 0,
            }
            capture["peak_detection"] = {
                "all_peaks_found": [],
                "all_bottoms_found": [],
                "prominence_threshold_used": None,
            }

    # Filter 1: plausibility filter on bottoms
    bottoms_pre = list(bottoms)
    bottoms = _filter_plausible_rep_frames(bottoms, angles, primary)
    if capture is not None:
        capture["filters_applied"].append(_make_filter_record(
            "plausibility_filter_bottoms",
            bottoms_pre, list(bottoms), angles, primary,
        ))

    # Filter 2: plausibility filter on tops
    tops_pre = list(tops)
    tops = _filter_plausible_rep_frames(tops, angles, primary)
    if capture is not None:
        capture["filters_applied"].append(_make_filter_record(
            "plausibility_filter_tops",
            tops_pre, list(tops), angles, primary,
        ))

    if prefer == "bottoms" and bottoms:
        peaks = bottoms
    elif tops:
        peaks = tops
    elif bottoms:
        peaks = bottoms
    else:
        if capture is not None:
            capture["filters_applied"].append({
                "filter_name": "extrema_selection",
                "indices_before": [],
                "indices_after": [],
                "indices_rejected": [],
                "rejection_reasons": ["no_peaks_or_bottoms_after_filtering"],
            })
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


def _peak_prominences(series, min_prominence, min_distance):
    """
    Returns the actual prominence value for each peak that passed the
    threshold in find_peaks_with_prominence. Mirrors that function's
    logic but captures prominence values instead of just indices.
    """
    n = len(series)
    if n < 3:
        return []
    proms: list[float] = []
    last_peak = -1
    i = 1
    while i < n - 1:
        if series[i] > series[i - 1] and series[i] >= series[i + 1]:
            ls = max(0, i - min_distance); re = min(n, i + min_distance + 1)
            lmin = float(np.min(series[ls:i])) if i > ls else float(series[i])
            rmin = float(np.min(series[i + 1:re])) if i + 1 < re else float(series[i])
            prom = float(series[i]) - max(lmin, rmin)
            if prom >= min_prominence:
                if last_peak < 0 or (i - last_peak) >= min_distance:
                    proms.append(prom)
                    last_peak = i
                    i += min_distance
                    continue
        i += 1
    return proms


def _make_filter_record(
    name: str,
    before: list[int],
    after: list[int],
    angles: dict,
    primary: str,
) -> dict:
    rejected = [int(f) for f in before if f not in after]
    reasons: list[str] = []
    if rejected and primary and primary in angles:
        vals = [float(angles[primary][f]) for f in before
                if 0 <= f < len(angles[primary])]
        if vals:
            median_val = float(np.median(vals))
            for f in rejected:
                if 0 <= f < len(angles[primary]):
                    delta = abs(float(angles[primary][f]) - median_val)
                    reasons.append(
                        f"frame_{f}: |{float(angles[primary][f]):.2f} - "
                        f"median({median_val:.2f})| = {delta:.2f} > 30.0"
                    )
                else:
                    reasons.append(f"frame_{f}: out_of_bounds")
    elif rejected:
        reasons = [f"frame_{f}: filter_applied" for f in rejected]
    return {
        "filter_name": name,
        "indices_before": [int(f) for f in before],
        "indices_after": [int(f) for f in after],
        "indices_rejected": rejected,
        "rejection_reasons": reasons,
        "filter_threshold": "abs(angle - median(angles)) <= 30.0 degrees",
    }


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