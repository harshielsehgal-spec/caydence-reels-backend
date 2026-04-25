"""
diagnose_adaptive.py — Read-only diagnostic for the adaptive rubric pipeline.

Runs the full pipeline on ref_video.mp4 (and optionally attempt_video.mp4)
and prints intermediate state at every decision point. Does not modify
any engine code.

Usage:
    python3 diagnose_adaptive.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

# Ensure local imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from pose_similarity import (
    extract_keypoints,
    normalize_keypoints,
    _compute_frame_velocities,
    _detect_action_window,
)
from adaptive_rubric import (
    JOINT_TRIPLETS,
    BILATERAL_PAIRS,
    STATIC_HOLD_VARIANCE_THRESHOLD,
    compute_joint_angles_per_frame,
    compute_relative_variance,
    compute_importance_weights,
    detect_rep_extrema,
    crop_to_action_window,
    _filter_plausible_rep_frames,
    build_adaptive_rubric,
)


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def probe_video(path: str) -> tuple[int, float, int]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration_ms = int((frames / fps) * 1000) if fps > 0 else 0
    return duration_ms, fps, frames


def diagnose(video_path: str, label: str):
    banner(f"DIAGNOSTIC: {label}  ({video_path})")

    # ── 1. Video probe ────────────────────────────────────────────────────
    duration_ms, real_fps, total_frames = probe_video(video_path)
    print(f"\n[1] Video info")
    print(f"    Duration:     {duration_ms} ms  ({duration_ms/1000:.2f} s)")
    print(f"    Real FPS:     {real_fps:.2f}")
    print(f"    Total frames: {total_frames}")

    # ── 2. Extract keypoints (same parameters as rubric_router) ───────────
    sample_target = max(60, min(240, int(duration_ms / 100)))
    print(f"\n[2] Extracting {sample_target} uniform samples...")
    kps = extract_keypoints(video_path, sample_target)
    if kps is None or len(kps) < 5:
        print(f"    EXTRACTION FAILED — got {0 if kps is None else len(kps)} frames")
        return
    sampled_fps = len(kps) / (duration_ms / 1000.0)
    print(f"    Got: {len(kps)} keypoint frames")
    print(f"    Sampled FPS: {sampled_fps:.2f}")

    # ── 3. Action window crop (current logic in adaptive_rubric.py) ──────
    print(f"\n[3] crop_to_action_window() result")
    cropped, crop_start, crop_end = crop_to_action_window(kps)
    kept_pct = 100 * len(cropped) / len(kps) if len(kps) else 0
    print(f"    crop_start: {crop_start}")
    print(f"    crop_end:   {crop_end}")
    print(f"    kept:       {len(cropped)} / {len(kps)} frames  ({kept_pct:.1f}%)")
    if crop_start == 0 and crop_end == len(kps):
        print(f"    >>> NO CROP APPLIED (returned full sequence)")

    # ── 3b. What the LEGACY velocity-based detector says ──────────────────
    print(f"\n[3b] Legacy velocity-based action window (for comparison)")
    vels = _compute_frame_velocities(kps)
    legacy_start, legacy_end = _detect_action_window(vels, phase_start=0.0)
    print(f"    legacy start: {legacy_start}, end: {legacy_end}")
    print(f"    legacy would keep: {legacy_end-legacy_start} frames "
          f"({100*(legacy_end-legacy_start)/len(kps):.1f}%)")
    print(f"    velocity stats: max={vels.max():.4f}  median={np.median(vels):.4f}  "
          f"mean={vels.mean():.4f}")

    # ── 3c. Manually compute torso-stability crop signal ──────────────────
    print(f"\n[3c] Torso-stability signal (what current crop function uses)")
    norm_full = normalize_keypoints(kps)
    xy = norm_full[:, :, :2]
    sm = (xy[:, 11] + xy[:, 12]) / 2
    hm = (xy[:, 23] + xy[:, 24]) / 2
    torso_angle = np.degrees(np.arctan2(sm[:, 0] - hm[:, 0], -(sm[:, 1] - hm[:, 1])))
    window = max(10, len(torso_angle) // 20)
    rolling_std = np.array([
        float(np.std(torso_angle[max(0, i-window//2):min(len(torso_angle), i+window//2+1)]))
        for i in range(len(torso_angle))
    ])
    threshold = float(np.median(rolling_std))
    stable_mask = rolling_std < threshold
    print(f"    torso_angle range: {torso_angle.min():.1f}° to {torso_angle.max():.1f}°  "
          f"std={torso_angle.std():.1f}°")
    print(f"    rolling_std range: {rolling_std.min():.2f} to {rolling_std.max():.2f}")
    print(f"    threshold (median): {threshold:.2f}")
    print(f"    stable frames: {int(stable_mask.sum())} / {len(stable_mask)}")
    # Find the longest stable run for sanity
    longest = 0; cur = 0
    for s in stable_mask:
        if s:
            cur += 1; longest = max(longest, cur)
        else:
            cur = 0
    print(f"    longest contiguous stable run: {longest} frames")

    # ── 4. Per-joint variance ─────────────────────────────────────────────
    norm = normalize_keypoints(cropped)
    joint_angles = compute_joint_angles_per_frame(norm)
    rel_var = compute_relative_variance(joint_angles, norm)

    print(f"\n[4] Per-joint relative variance (on cropped data)")
    print(f"    {'joint':<14} {'rel_var':>12}  raw_angle_range")
    for name, v in sorted(rel_var.items(), key=lambda kv: -kv[1]):
        a = joint_angles[name]
        print(f"    {name:<14} {v:>12.2f}  {a.min():.1f}° to {a.max():.1f}°  "
              f"(std={a.std():.1f}°)")

    # ── 5. Bilateral pairing ──────────────────────────────────────────────
    print(f"\n[5] Bilateral pair sums (this is what determines primary)")
    for left, right in BILATERAL_PAIRS:
        if left in rel_var and right in rel_var:
            total = rel_var[left] + rel_var[right]
            print(f"    {left:<10} + {right:<10} = {total:>10.2f}  "
                  f"(L={rel_var[left]:.2f}, R={rel_var[right]:.2f})")

    # ── 6. Weights and primary joint ──────────────────────────────────────
    weights = compute_importance_weights(rel_var)
    static_hold = max(rel_var.values()) < STATIC_HOLD_VARIANCE_THRESHOLD
    print(f"\n[6] Importance weights (final)")
    print(f"    static_hold mode: {static_hold}")
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<14} weight={w:.4f}")

    # ── 7. Rep detection ──────────────────────────────────────────────────
    bottoms, tops, primary = detect_rep_extrema(joint_angles, weights, sampled_fps)
    print(f"\n[7] Rep detection")
    print(f"    primary joint: {primary}")
    if primary in joint_angles:
        a = joint_angles[primary]
        print(f"    primary signal stats: min={a.min():.1f}°  max={a.max():.1f}°  "
              f"mean={a.mean():.1f}°  std={a.std():.1f}°")
    print(f"    bottoms detected: {len(bottoms)}  -> frames {bottoms}")
    print(f"    tops    detected: {len(tops)}  -> frames {tops}")
    if primary in joint_angles and bottoms:
        print(f"    primary angle at each bottom: "
              f"{[round(float(joint_angles[primary][b]),1) for b in bottoms]}")
    if primary in joint_angles and tops:
        print(f"    primary angle at each top:    "
              f"{[round(float(joint_angles[primary][t]),1) for t in tops]}")

    bottoms_filt = _filter_plausible_rep_frames(bottoms, joint_angles, primary)
    print(f"    after plausibility filter: {len(bottoms_filt)} bottoms -> {bottoms_filt}")

    # ── 8. Final rubric ───────────────────────────────────────────────────
    print(f"\n[8] Final rubric (via build_adaptive_rubric)")
    try:
        result = build_adaptive_rubric(kps, "diagnostic", fps=sampled_fps)
        print(f"    primary_joint: {result.primary_joint}")
        print(f"    rep_count_bottoms: {len(result.rep_bottoms)}")
        print(f"    rep_count_tops:    {len(result.rep_tops)}")
        print(f"    static_hold:       {result.static_hold}")
        print(f"    crop_start/end:    {result.debug.get('crop_start')} / "
              f"{result.debug.get('crop_end')}")
        print(f"    Checks:")
        for c in result.rubric.checks:
            print(f"      {c.id:30s} target={c.target_value:6.1f}°  "
                  f"tol=±{c.tolerance:.1f}°  weight={c.weight:.3f}")
    except Exception as e:
        print(f"    EXCEPTION: {e}")

    return locals()


def main():
    here = Path(__file__).parent
    ref = here / "ref_video.mp4"
    att = here / "attempt_video.mp4"

    if not ref.exists():
        print(f"ERROR: {ref} not found")
        return

    diagnose(str(ref), "REFERENCE")

    if att.exists():
        diagnose(str(att), "ATTEMPT")

    # ── Verdict ──────────────────────────────────────────────────────────
    banner("VERDICT")
    print("(See detailed analysis above. Verdict logic depends on numbers.)")


if __name__ == "__main__":
    main()