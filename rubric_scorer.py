"""
rubric_scorer.py — Orchestrates rubric-based scoring.

NEW FLOW (REVIEW FIX #5: segment-then-DTW, not DTW-then-segment):
  1. Segment creator reps and viewer reps independently
  2. Pair reps by normalized timeline position
  3. For each pair, DTW within just that rep (finds the peak-to-peak alignment)
  4. Score geometric checks at the peak frame of each viewer rep
  5. Average per-rep scores — NO rep-count penalty

Backward-compatible entry point score_with_rubric() kept for the existing
/reels/analyze_v2 endpoint. Falls back gracefully if rep detection fails
(single-rep or static-hold rubrics).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from pose_similarity import normalize_keypoints
from pose_alignment import align_sequences, path_to_mapping
from geometric_scoring import run_check, aggregate_scores, CheckResult
from rubric_schema import Rubric
from rubric_builder import timestamp_to_frame_index
from rep_segmentation import segment_reps, pair_reps, extract_rep_slices


@dataclass
class RubricScoreResult:
    score: int
    arm_alignment: int
    hip_position: int
    timing_sync: int
    frame_count: int
    confidence: float
    used_rubric: bool
    checks: list[dict]
    key_frame_mapping: dict
    rep_count_creator: int = 0
    rep_count_viewer: int = 0
    rep_count_paired: int = 0
    per_rep_scores: list[int] = None  # type: ignore


_UPPER = set(range(11, 23))
_LOWER = set(range(23, 33))


def _derive_legacy_scores(results: list[CheckResult], rubric: Rubric) -> tuple[int, int, int]:
    """Back-compat fields for ai_coach.py and existing frontend."""
    check_by_id = {c.id: c for c in rubric.checks}
    up_s, up_w = 0.0, 0.0
    lo_s, lo_w = 0.0, 0.0
    for r in results:
        if not r.passed:
            continue
        spec = check_by_id.get(r.check_id)
        if not spec:
            continue
        lms = set(spec.landmarks)
        if lms & _UPPER:
            up_s += r.score * r.weight; up_w += r.weight
        if lms & _LOWER:
            lo_s += r.score * r.weight; lo_w += r.weight
    arm = int(round(up_s / up_w)) if up_w > 0 else 50
    hip = int(round(lo_s / lo_w)) if lo_w > 0 else 50
    passed = sum(1 for r in results if r.passed)
    total = len(results) or 1
    timing = int(round(100 * passed / total))
    return arm, hip, timing


def _score_single_frame(
    attempt_frame: np.ndarray,
    rubric: Rubric,
) -> tuple[int, list[CheckResult], float]:
    """Run all rubric checks against one normalized-with-visibility frame."""
    results = [run_check(c, attempt_frame) for c in rubric.checks]
    overall, conf = aggregate_scores(results)
    return overall, results, conf


def _attach_visibility(normalized_xy: np.ndarray, raw_kps: np.ndarray) -> np.ndarray:
    """normalize_keypoints returns (N,33,2); reattach visibility for scoring."""
    vis = raw_kps[:, :, 2:3]
    return np.concatenate([normalized_xy, vis], axis=2)


def score_with_rubric(
    creator_keypoints: np.ndarray,
    attempt_keypoints: np.ndarray,
    creator_video_duration_ms: int,
    rubric: Rubric,
    sport_weights: Optional[dict] = None,
    fps: float = 30.0,
) -> RubricScoreResult:
    """
    Rep-aware scoring entry point.

    Strategy:
      - Try to segment reps in both videos.
      - If both have ≥1 rep detected → score each paired rep, average.
      - Else fall back to single-frame scoring at the rubric's key frame
        (works for static holds and single-rep rubrics).
    """
    # Try rep-based scoring
    creator_reps, _ = segment_reps(creator_keypoints, fps=fps)
    viewer_reps, _ = segment_reps(attempt_keypoints, fps=fps)

    att_norm = normalize_keypoints(attempt_keypoints)
    att_full = _attach_visibility(att_norm, attempt_keypoints)

    if creator_reps and viewer_reps:
        pairs = pair_reps(
            creator_reps, viewer_reps,
            len(creator_keypoints), len(attempt_keypoints),
        )
        if pairs:
            return _score_rep_by_rep(
                creator_keypoints, attempt_keypoints,
                att_full, rubric, pairs,
                creator_reps, viewer_reps,
            )

    # Fallback: single-frame scoring at the rubric's designated key frame
    return _score_single_keyframe(
        creator_keypoints, attempt_keypoints,
        att_full, rubric, creator_video_duration_ms,
        creator_reps, viewer_reps,
    )


def _score_rep_by_rep(
    creator_kps: np.ndarray,
    attempt_kps: np.ndarray,
    att_full: np.ndarray,
    rubric: Rubric,
    pairs,
    creator_reps,
    viewer_reps,
) -> RubricScoreResult:
    """Score each paired rep, average. No rep-count penalty."""
    per_rep_scores: list[int] = []
    all_results: list[CheckResult] = []
    kf_mapping: dict = {}

    for idx, pair in enumerate(pairs):
        slice_info = extract_rep_slices(creator_kps, attempt_kps, pair)

        # DTW just this rep
        try:
            path = align_sequences(slice_info.creator_slice, slice_info.viewer_slice)
            mapping = path_to_mapping(path, len(slice_info.creator_slice))
            # Creator's peak (local) → viewer's frame index (local)
            viewer_peak_local = mapping[slice_info.creator_peak_local] \
                if 0 <= slice_info.creator_peak_local < len(mapping) else slice_info.viewer_peak_local
        except Exception:
            viewer_peak_local = slice_info.viewer_peak_local

        # Convert local viewer frame back to global index
        viewer_peak_global = pair[1].start_frame + viewer_peak_local
        viewer_peak_global = max(0, min(viewer_peak_global, len(att_full) - 1))

        frame = att_full[viewer_peak_global]
        overall, results, _ = _score_single_frame(frame, rubric)
        per_rep_scores.append(overall)
        all_results.extend(results)
        kf_mapping[f"rep_{idx + 1}"] = viewer_peak_global

    final_score = int(round(np.mean(per_rep_scores))) if per_rep_scores else 0
    arm, hip, timing = _derive_legacy_scores(all_results, rubric)
    avg_conf = float(np.mean([r.confidence for r in all_results if r.passed])) if all_results else 0.0

    # Check breakdown: use the best-scoring rep's checks so the UI shows
    # achievable-looking numbers, not the worst rep
    best_rep_idx = int(np.argmax(per_rep_scores)) if per_rep_scores else 0
    checks_per_rep = len(rubric.checks)
    best_start = best_rep_idx * checks_per_rep
    best_results = all_results[best_start:best_start + checks_per_rep]

    return RubricScoreResult(
        score=final_score,
        arm_alignment=arm,
        hip_position=hip,
        timing_sync=timing,
        frame_count=len(attempt_kps),
        confidence=round(avg_conf, 2),
        used_rubric=True,
        checks=[{
            "id": r.check_id, "score": r.score, "observed": r.observed,
            "target": r.target, "tolerance": r.tolerance,
            "confidence": r.confidence, "passed": r.passed, "weight": r.weight,
        } for r in best_results],
        key_frame_mapping=kf_mapping,
        rep_count_creator=len(creator_reps),
        rep_count_viewer=len(viewer_reps),
        rep_count_paired=len(pairs),
        per_rep_scores=per_rep_scores,
    )


def _score_single_keyframe(
    creator_kps: np.ndarray,
    attempt_kps: np.ndarray,
    att_full: np.ndarray,
    rubric: Rubric,
    creator_duration_ms: int,
    creator_reps,
    viewer_reps,
) -> RubricScoreResult:
    """Fallback path: DTW whole sequence, score at mapped key frame."""
    try:
        path = align_sequences(creator_kps, attempt_kps)
        mapping = path_to_mapping(path, len(creator_kps))
    except Exception:
        mapping = list(range(min(len(creator_kps), len(attempt_kps))))

    kf_mapping: dict = {}
    all_results: list[CheckResult] = []
    for kf in rubric.key_frames:
        c_idx = timestamp_to_frame_index(kf.timestamp_ms, len(creator_kps), creator_duration_ms)
        a_idx = mapping[c_idx] if 0 <= c_idx < len(mapping) else 0
        a_idx = max(0, min(a_idx, len(att_full) - 1))
        kf_mapping[kf.id] = a_idx
        frame = att_full[a_idx]
        for check in rubric.checks:
            if check.key_frame_id == kf.id:
                all_results.append(run_check(check, frame))

    overall, avg_conf = aggregate_scores(all_results)
    arm, hip, timing = _derive_legacy_scores(all_results, rubric)
    return RubricScoreResult(
        score=overall,
        arm_alignment=arm,
        hip_position=hip,
        timing_sync=timing,
        frame_count=len(attempt_kps),
        confidence=round(avg_conf, 2),
        used_rubric=True,
        checks=[{
            "id": r.check_id, "score": r.score, "observed": r.observed,
            "target": r.target, "tolerance": r.tolerance,
            "confidence": r.confidence, "passed": r.passed, "weight": r.weight,
        } for r in all_results],
        key_frame_mapping=kf_mapping,
        rep_count_creator=len(creator_reps),
        rep_count_viewer=len(viewer_reps),
        rep_count_paired=0,
        per_rep_scores=[],
    )


def result_to_dict(r: RubricScoreResult) -> dict:
    return asdict(r)