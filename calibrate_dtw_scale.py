"""
calibrate_dtw_scale.py  —  Caydence DTW scale calibration tool
Angle-based version — matches pose_similarity.py compute_timing_sync exactly.

USAGE:
    python3 calibrate_dtw_scale.py --ref ref_video.mp4 --good good.mp4
    python3 calibrate_dtw_scale.py --ref ref_video.mp4 --good good.mp4 --bad bad.mp4
"""

import argparse
import sys
import math
import numpy as np

try:
    from dtaidistance import dtw_ndim
except ImportError:
    print("ERROR: dtaidistance not installed. Run: pip install \"dtaidistance>=2.3.10\"")
    sys.exit(1)

try:
    from pose_similarity import (
        extract_keypoints,
        normalize_keypoints,
        extract_angle_sequence,
        _DTW_WINDOW_FRACTION,
        _DTW_DEFAULT_WEIGHTS,
        _DTW_SCALE as CURRENT_SCALE,
    )
    print("✓ Loaded from pose_similarity.py")
except ImportError as e:
    print(f"ERROR: Could not import from pose_similarity.py — {e}")
    sys.exit(1)

SCALE_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]


def raw_angle_dtw(ref_kps, att_kps):
    """
    Mirrors compute_timing_sync exactly:
      1. normalize_keypoints
      2. extract_angle_sequence
      3. divide by 180
      4. dtw_ndim.distance per group
    Returns dict of {group: dist}
    """
    ref_norm   = normalize_keypoints(ref_kps)
    att_norm   = normalize_keypoints(att_kps)
    ref_angles = extract_angle_sequence(ref_norm)
    att_angles = extract_angle_sequence(att_norm)

    group_dists = {}
    print(f"\n  {'GROUP':8s}  {'DIST':>8s}  {'NOTE'}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*30}")

    for group in _DTW_DEFAULT_WEIGHTS:
        ref_seq = (ref_angles[group] / 180.0).astype(np.float64)
        att_seq = (att_angles[group] / 180.0).astype(np.float64)
        n       = max(len(ref_seq), len(att_seq))
        window  = max(1, int(n * _DTW_WINDOW_FRACTION))
        dist    = float(dtw_ndim.distance(ref_seq, att_seq, window=window))
        group_dists[group] = dist
        note = "✓ good range" if dist < 1.0 else ("high — diff camera angle?" if dist > 5.0 else "ok")
        print(f"  {group:8s}  {dist:8.4f}  {note}")

    return group_dists


def weighted_score(group_dists, scale):
    total = w_total = 0.0
    for g, d in group_dists.items():
        w = _DTW_DEFAULT_WEIGHTS.get(g, 1.0)
        total   += math.exp(-d / scale) * w
        w_total += w
    return total / w_total if w_total else 0.0


def scale_sweep(good_dists, bad_dists=None):
    print(f"\n  {'SCALE':>6s}  {'GOOD':>6s}  {'BAD':>6s}  {'GAP':>6s}  NOTE")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}")

    best_scale = None
    best_gap   = -1.0

    for scale in SCALE_SWEEP:
        good_score = weighted_score(good_dists, scale)
        bad_score  = weighted_score(bad_dists,  scale) if bad_dists else None

        good_ok  = good_score >= 0.75
        bad_ok   = (bad_score <= 0.45) if bad_score is not None else True
        is_valid = good_ok and bad_ok

        gap     = (good_score - bad_score) if bad_score is not None else good_score
        bad_str = f"{bad_score:.3f}" if bad_score is not None else "  n/a"
        gap_str = f"{gap:.3f}"       if bad_score is not None else "  n/a"
        note    = "← ✓ VALID" if is_valid else ("good too low" if not good_ok else "bad too high")

        if is_valid and gap > best_gap:
            best_gap   = gap
            best_scale = scale

        print(f"  {scale:>6.2f}  {good_score:.3f}  {bad_str}  {gap_str}  {note}")

    return best_scale


def load_video(path):
    print(f"  extracting keypoints from {path} …")
    arr = extract_keypoints(path, max_frames=60)
    if arr is None or len(arr) == 0:
        print(f"  ERROR: could not extract keypoints from {path}")
        sys.exit(1)
    print(f"  extracted {len(arr)} frames  shape={arr.shape}")
    return arr


def main():
    p = argparse.ArgumentParser(description="Calibrate _DTW_SCALE for pose_similarity.py")
    p.add_argument("--ref",  required=True)
    p.add_argument("--good", required=True)
    p.add_argument("--bad",  default=None)
    args = p.parse_args()

    print("=" * 60)
    print("  Caydence DTW scale calibration  (angle-based)")
    print("=" * 60)

    print("\n[1/3] Loading inputs")
    ref_kps  = load_video(args.ref)
    good_kps = load_video(args.good)
    bad_kps  = load_video(args.bad) if args.bad else None

    print(f"\n[2/3] Raw angle DTW distances")
    print(f"\n  — GOOD attempt ({len(good_kps)} frames vs {len(ref_kps)} ref frames)")
    good_dists = raw_angle_dtw(ref_kps, good_kps)

    bad_dists = None
    if bad_kps is not None:
        print(f"\n  — BAD attempt ({len(bad_kps)} frames vs {len(ref_kps)} ref frames)")
        bad_dists = raw_angle_dtw(ref_kps, bad_kps)

    print(f"\n[3/3] Scale sweep")
    print(f"\n  Target: good ≥ 0.75 (75/100),  bad ≤ 0.45 (45/100)")
    best = scale_sweep(good_dists, bad_dists)

    print("\n" + "=" * 60)
    if best:
        print(f"  RECOMMENDED: set _DTW_SCALE = {best} in pose_similarity.py")
    else:
        best_fallback = max(SCALE_SWEEP, key=lambda s: weighted_score(good_dists, s))
        print(f"  No scale satisfies both constraints.")
        print(f"  Best for good-form alone: _DTW_SCALE = {best_fallback}")
        if bad_dists is None:
            print(f"  TIP: run again with --bad bad_video.mp4 for tighter calibration.")

    cur = weighted_score(good_dists, CURRENT_SCALE)
    print(f"\n  Current (_DTW_SCALE={CURRENT_SCALE}):")
    print(f"    good={cur:.3f} ({int(cur*100)}/100)")
    print("=" * 60)


if __name__ == "__main__":
    main()