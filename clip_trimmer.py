"""
clip_trimmer.py — Auto-trim setup/cleanup junk from a recorded clip.

Uses _compute_frame_velocities + _detect_action_window from pose_similarity.py
(read-only — does not modify them). Writes a trimmed clip to disk via
cv2.VideoWriter with mp4v codec. Quality loss is acceptable because downstream
scoring reads pose landmarks, not pixels.

Trim is opportunistic, not a quality gate. If detection signal is weak, returns
trim_skipped=True and the caller passes through the original clip.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from pose_similarity import (
    extract_keypoints,
    _compute_frame_velocities,
    _detect_action_window,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Trim policy thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Minimum trimmed window duration. Below this we consider the trim suspicious
# and skip it. Pushup-class movements take ≥0.5s/rep; 2s captures at least
# one rep with margin.
MIN_TRIM_WINDOW_SECONDS = 2.0

# Maximum fraction of the original clip we'll keep when "trimming." If the
# detected window is >90% of the clip, the trim isn't doing useful work — skip.
MAX_TRIM_KEEP_FRACTION = 0.90

# Minimum velocity signal strength. If max velocity is below this, the clip
# has no detectable action — skip trim, let the scorer reject it downstream.
MIN_VELOCITY_PEAK = 0.005


def _probe(path: str) -> tuple[float, int, int, int]:
    """Return (fps, total_frames, width, height) for a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, total, w, h


def trim_clip(input_path: str, output_path: str) -> dict:
    """
    Try to trim setup/cleanup junk from input_path. Writes trimmed clip to
    output_path on success.

    Returns:
        {
          "trimmed": bool,
          "skip_reason": Optional[str],
          "window": Optional[(start_frame, end_frame)],
          "original_duration_s": float,
          "trimmed_duration_s": Optional[float],
        }

    Never raises. On any exception, returns trimmed=False with skip_reason set.
    The caller should fall back to the original input_path when trimmed=False.
    """
    result = {
        "trimmed": False,
        "skip_reason": None,
        "window": None,
        "original_duration_s": 0.0,
        "trimmed_duration_s": None,
    }

    try:
        fps, total_frames, width, height = _probe(input_path)
        original_duration = total_frames / fps if fps > 0 else 0.0
        result["original_duration_s"] = round(original_duration, 3)

        if total_frames < 10 or fps <= 0:
            result["skip_reason"] = "clip_too_short_or_invalid_fps"
            return result

        # Step 1: extract keypoints uniformly to compute velocities
        # Use ~10fps sampling so the velocity signal has enough resolution
        # without burning extraction time. Capped at 240 frames for memory.
        sample_target = max(60, min(240, int(total_frames / max(fps / 10.0, 1))))
        kps = extract_keypoints(input_path, sample_target)
        if kps is None or len(kps) < 10:
            result["skip_reason"] = "pose_extraction_too_sparse"
            return result

        # Step 2: velocities → action window (in sampled-frame indices)
        velocities = _compute_frame_velocities(kps)
        if velocities.max() < MIN_VELOCITY_PEAK:
            result["skip_reason"] = "signal_too_weak"
            return result

        sampled_start, sampled_end = _detect_action_window(velocities, phase_start=0.0)
        if sampled_end <= sampled_start:
            result["skip_reason"] = "action_window_invalid"
            return result

        # Step 3: map sampled-frame indices back to ORIGINAL video frame indices
        # The sampling was uniform across [0, total_frames-1] with sample_target points
        sample_indices = np.linspace(0, total_frames - 1, len(kps), dtype=int)
        # Clamp window indices into the kps array
        sampled_start = max(0, min(sampled_start, len(sample_indices) - 1))
        sampled_end = max(0, min(sampled_end, len(sample_indices) - 1))
        orig_start = int(sample_indices[sampled_start])
        orig_end = int(sample_indices[sampled_end])
        if orig_end <= orig_start:
            result["skip_reason"] = "trimmed_window_zero_length"
            return result

        trimmed_duration = (orig_end - orig_start) / fps
        keep_fraction = (orig_end - orig_start) / total_frames

        # Step 4: trim sanity gates
        if trimmed_duration < MIN_TRIM_WINDOW_SECONDS:
            result["skip_reason"] = "window_too_short"
            return result
        if keep_fraction > MAX_TRIM_KEEP_FRACTION:
            result["skip_reason"] = "window_too_long"
            return result

        # Step 5: write the trimmed clip
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            result["skip_reason"] = "writer_init_failed"
            return result

        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_start)
        frames_written = 0
        for _ in range(orig_end - orig_start + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames_written += 1
        cap.release()
        writer.release()

        if frames_written < 5:
            result["skip_reason"] = "frames_written_too_few"
            return result

        result["trimmed"] = True
        result["window"] = (orig_start, orig_end)
        result["trimmed_duration_s"] = round(trimmed_duration, 3)
        return result

    except Exception as e:
        log.exception("trim_clip failed for %s", input_path)
        result["skip_reason"] = f"exception:{type(e).__name__}"
        return result