"""
pose_alignment.py — DTW-based sequence alignment for Caydence rubric scoring.

Reuses the DTW config from pose_similarity.py but exposes the *warping path*
instead of converting it to a similarity score. The path lets us map
creator-frame indices to viewer-frame indices, which is what rubric scoring
needs.

This module does NOT replace compute_timing_sync — that stays as-is for the
legacy scoring path. It adds a parallel "alignment" capability.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from dtaidistance import dtw_ndim

from pose_similarity import (
    normalize_keypoints,
    extract_angle_sequence,
    _DTW_WINDOW_FRACTION,
    _DTW_DEFAULT_WEIGHTS,
)


def _concat_angle_features(angles: dict, weights: dict) -> np.ndarray:
    """
    Concatenate weighted angle groups into a single (N, D) feature sequence.
    Angles are normalized to [0,1] by dividing by 180.
    Weights are applied before concat so DTW treats higher-weight groups
    as more distance-relevant.
    """
    features = []
    for group, w in weights.items():
        if group not in angles:
            continue
        seq = (angles[group] / 180.0).astype(np.float64) * float(w)
        features.append(seq)
    if not features:
        raise ValueError("No matching angle groups found for given weights")
    return np.concatenate(features, axis=1)


def align_sequences(
    ref_kps: np.ndarray,
    att_kps: np.ndarray,
    sport_weights: Optional[dict] = None,
) -> list[tuple[int, int]]:
    """
    Compute DTW warping path between reference and attempt keypoint sequences.

    Returns a list of (ref_idx, att_idx) pairs. The path is monotonically
    non-decreasing in both indices. Use path_to_mapping() to convert into a
    per-ref-frame lookup.

    Alignment is done on normalized joint angles (camera-angle invariant),
    matching the strategy that gave 7.5/10 scoring accuracy. The difference:
    we extract the path, not the distance.
    """
    if len(ref_kps) < 2 or len(att_kps) < 2:
        # Degenerate case — return identity-ish mapping
        return [(i, i) for i in range(min(len(ref_kps), len(att_kps)))]

    weights = sport_weights or _DTW_DEFAULT_WEIGHTS

    ref_norm = normalize_keypoints(ref_kps)
    att_norm = normalize_keypoints(att_kps)
    ref_angles = extract_angle_sequence(ref_norm)
    att_angles = extract_angle_sequence(att_norm)

    ref_feat = _concat_angle_features(ref_angles, weights)
    att_feat = _concat_angle_features(att_angles, weights)

    n = max(len(ref_feat), len(att_feat))
    window = max(1, int(n * _DTW_WINDOW_FRACTION))

    # dtw_ndim.warping_path returns list of (ref_idx, att_idx) tuples
    path = dtw_ndim.warping_path(ref_feat, att_feat, window=window)
    return [(int(i), int(j)) for i, j in path]


def path_to_mapping(path: list[tuple[int, int]], ref_len: int) -> list[int]:
    """
    Convert a DTW warping path into a per-ref-frame lookup.

    Returns an array of length ref_len where mapping[i] = best attempt-frame
    index for ref frame i. When multiple attempt frames align to one ref frame,
    we pick the median (most stable choice).
    """
    buckets: list[list[int]] = [[] for _ in range(ref_len)]
    for ref_idx, att_idx in path:
        if 0 <= ref_idx < ref_len:
            buckets[ref_idx].append(att_idx)

    mapping: list[int] = []
    last = 0
    for i, bucket in enumerate(buckets):
        if bucket:
            # median is robust to DTW plateau artifacts
            last = int(np.median(bucket))
        # If a ref frame has no alignment, carry forward the previous mapping.
        mapping.append(last)
    return mapping


def map_creator_frame_to_attempt(
    creator_frame_idx: int,
    ref_len: int,
    path: list[tuple[int, int]],
    att_len: int,
) -> int:
    """
    Given a specific creator-frame index, find the best-matching attempt frame.
    Handles out-of-range inputs by clamping to valid ranges.
    """
    mapping = path_to_mapping(path, ref_len)
    clamped = max(0, min(creator_frame_idx, ref_len - 1))
    att_idx = mapping[clamped]
    return max(0, min(att_idx, att_len - 1))


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(7)
    ref = rng.random((50, 33, 3)).astype(np.float32)
    att = rng.random((60, 33, 3)).astype(np.float32)
    ref[:, :, 2] = 0.9
    att[:, :, 2] = 0.85

    path = align_sequences(ref, att)
    print(f"Path length: {len(path)}")
    print(f"First 5 pairs:  {path[:5]}")
    print(f"Last 5 pairs:   {path[-5:]}")
    mapping = path_to_mapping(path, len(ref))
    print(f"Ref frame 25 -> att frame {mapping[25]}")
    print(f"Ref frame 0  -> att frame {mapping[0]}")
    print(f"Ref frame 49 -> att frame {mapping[49]}")
