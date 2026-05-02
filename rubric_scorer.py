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
    diagnose: bool = False,
) -> RubricScoreResult | tuple[RubricScoreResult, dict]:
    """
    Rep-aware scoring entry point.

    Strategy:
      - Try to segment reps in both videos.
      - If both have ≥1 rep detected → score each paired rep, average.
      - Else fall back to single-frame scoring at the rubric's key frame
        (works for static holds and single-rep rubrics).

    When diagnose=True, returns a tuple (RubricScoreResult, diagnostic_dict)
    instead of just RubricScoreResult. The diagnostic dict has shape
    {"reps": [...]} for the rep-by-rep path, or {"reps": [], "fallback": True}
    for the single-keyframe fallback path. Default is False — return contract
    is preserved unless diagnose is explicitly enabled.
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
                diagnose=diagnose,
            )

    # Fallback: single-frame scoring at the rubric's designated key frame
    result = _score_single_keyframe(
        creator_keypoints, attempt_keypoints,
        att_full, rubric, creator_video_duration_ms,
        creator_reps, viewer_reps,
    )
    if diagnose:
        return result, {"reps": [], "fallback": True}
    return result


def _score_rep_by_rep(
    creator_kps: np.ndarray,
    attempt_kps: np.ndarray,
    att_full: np.ndarray,
    rubric: Rubric,
    pairs,
    creator_reps,
    viewer_reps,
    diagnose: bool = False,
) -> RubricScoreResult | tuple[RubricScoreResult, dict]:
    """Score each paired rep, average. No rep-count penalty.

    When diagnose=True, returns (RubricScoreResult, {"reps": [...]}) where each
    rep entry exposes the DTW path neighborhood, observed/target/tolerance per
    check, and per-landmark visibility. Read-only; does not affect scoring.
    """
    per_rep_scores: list[int] = []
    all_results: list[CheckResult] = []
    kf_mapping: dict = {}
    diag_reps: list[dict] = [] if diagnose else []

    # When diagnose=True, we need normalized creator keypoints with visibility
    # to capture creator_observed values at trajectory anchor frames. Don't
    # waste the work when diagnose=False — scoring math doesn't need this.
    creator_full = None
    if diagnose:
        creator_norm = normalize_keypoints(creator_kps)
        creator_full = _attach_visibility(creator_norm, creator_kps)

    for idx, pair in enumerate(pairs):
        creator_rep, viewer_rep = pair
        slice_info = extract_rep_slices(creator_kps, attempt_kps, pair)

        # DTW just this rep
        path: list[tuple[int, int]] = []
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

        # ── Diagnostic build (read-only, runs alongside scoring) ─────────────
        if diagnose:
            # Map the viewer rep's [start, peak, end] back to the creator
            # frames they DTW-aligned to, in global creator-frame indices.
            v_start_local = 0
            v_end_local = max(0, len(slice_info.viewer_slice) - 1)
            # Reverse mapping (viewer_local → creator_local) by walking the path
            viewer_to_creator: dict[int, int] = {}
            for c_local, v_local in path:
                # Keep the median-equivalent: last write wins is fine for
                # monotonic paths, but prefer first stable mapping
                if v_local not in viewer_to_creator:
                    viewer_to_creator[v_local] = c_local

            def _v_to_c_global(v_local: int) -> Optional[int]:
                if not viewer_to_creator:
                    return None
                # Snap to nearest mapped viewer-local frame
                if v_local in viewer_to_creator:
                    c_local = viewer_to_creator[v_local]
                else:
                    keys = sorted(viewer_to_creator.keys())
                    nearest = min(keys, key=lambda k: abs(k - v_local))
                    c_local = viewer_to_creator[nearest]
                return creator_rep.start_frame + int(c_local)

            creator_match_start = _v_to_c_global(v_start_local)
            creator_match_peak = _v_to_c_global(viewer_peak_local)
            creator_match_end = _v_to_c_global(v_end_local)

            # 5 path entries centered on viewer peak local — what creator
            # frames did DTW match to viewer_peak−2..viewer_peak+2?
            dtw_neighbors: dict[str, Optional[int]] = {}
            for offset in (-2, -1, 0, 1, 2):
                v_local = viewer_peak_local + offset
                if v_local < 0 or v_local >= len(slice_info.viewer_slice):
                    dtw_neighbors[str(offset)] = None
                else:
                    dtw_neighbors[str(offset)] = _v_to_c_global(v_local)

            # Per-check diagnostics for this rep's results.
            check_specs_by_id = {c.id: c for c in rubric.checks}
            checks_diag: list[dict] = []
            # Visibility lookup uses the raw (un-normalized) viewer keypoints
            # at the matched frame, since att_full has normalized xy + raw vis.
            viewer_frame_for_vis = att_full[viewer_peak_global]  # (33, 3)

            for r in results:
                spec = check_specs_by_id.get(r.check_id)
                landmarks = spec.landmarks if spec else []
                vis_dict: dict[str, float] = {}
                for lm_idx in landmarks:
                    if 0 <= lm_idx < viewer_frame_for_vis.shape[0]:
                        vis_dict[str(int(lm_idx))] = round(
                            float(viewer_frame_for_vis[lm_idx, 2]), 4
                        )
                checks_diag.append({
                    "name": r.check_id,
                    "target": round(float(r.target), 4) if r.target is not None else None,
                    "tolerance": round(float(r.tolerance), 4) if r.tolerance is not None else None,
                    "observed": round(float(r.observed), 4) if r.observed is not None else None,
                    "visibility": vis_dict,
                    "score": int(r.score),
                })

            # ── Trajectory build (5 anchors across creator rep) ──────────
            trajectory: list[dict] = []
            dtw_rep_cost: Optional[float] = None

            if path:
                # Compute total DTW alignment cost: sum of feature distances
                # along the warping path, in normalized angle space.
                # Reuse pose_alignment's feature builder + per-pair distance.
                try:
                    from pose_alignment import _concat_angle_features
                    from pose_similarity import (
                        normalize_keypoints as _normk,
                        extract_angle_sequence as _eas,
                        _DTW_DEFAULT_WEIGHTS as _W,
                    )
                    c_feat = _concat_angle_features(
                        _eas(_normk(slice_info.creator_slice)), _W,
                    )
                    v_feat = _concat_angle_features(
                        _eas(_normk(slice_info.viewer_slice)), _W,
                    )
                    cost_sum = 0.0
                    for c_local, v_local in path:
                        if 0 <= c_local < len(c_feat) and 0 <= v_local < len(v_feat):
                            cost_sum += float(np.linalg.norm(
                                c_feat[c_local] - v_feat[v_local]
                            ))
                    dtw_rep_cost = round(cost_sum, 4)
                except Exception:
                    dtw_rep_cost = None

                # Forward mapping (creator_local → viewer_local) by walking path
                creator_to_viewer: dict[int, int] = {}
                for c_local, v_local in path:
                    if c_local not in creator_to_viewer:
                        creator_to_viewer[c_local] = v_local

                def _c_to_v_local(c_local: int) -> Optional[int]:
                    if not creator_to_viewer:
                        return None
                    if c_local in creator_to_viewer:
                        return creator_to_viewer[c_local]
                    keys = sorted(creator_to_viewer.keys())
                    nearest = min(keys, key=lambda k: abs(k - c_local))
                    return creator_to_viewer[nearest]

                # Anchor at 0%, 25%, 50%, 75%, 100% of creator rep range.
                # Ranges are inclusive (extract_rep_slices uses end_frame + 1).
                c_local_max = len(slice_info.creator_slice) - 1
                v_local_max = len(slice_info.viewer_slice) - 1

                for pct in (0.0, 0.25, 0.5, 0.75, 1.0):
                    c_local = int(round(c_local_max * pct))
                    c_local = max(0, min(c_local, c_local_max))
                    v_local = _c_to_v_local(c_local)

                    if v_local is None:
                        trajectory.append({
                            "anchor_pct": pct,
                            "creator_frame": None,
                            "viewer_frame": None,
                            "checks": [],
                        })
                        continue

                    v_local = max(0, min(int(v_local), v_local_max))
                    creator_frame_global = creator_rep.start_frame + c_local
                    viewer_frame_global = viewer_rep.start_frame + v_local
                    creator_frame_global = max(
                        0, min(creator_frame_global, creator_full.shape[0] - 1)
                    )
                    viewer_frame_global = max(
                        0, min(viewer_frame_global, att_full.shape[0] - 1)
                    )

                    c_frame_data = creator_full[creator_frame_global]
                    v_frame_data = att_full[viewer_frame_global]

                    anchor_checks: list[dict] = []
                    for check in rubric.checks:
                        c_result = run_check(check, c_frame_data)
                        v_result = run_check(check, v_frame_data)
                        c_vis: dict[str, float] = {}
                        v_vis: dict[str, float] = {}
                        for lm_idx in check.landmarks:
                            if 0 <= lm_idx < c_frame_data.shape[0]:
                                c_vis[str(int(lm_idx))] = round(
                                    float(c_frame_data[lm_idx, 2]), 4
                                )
                            if 0 <= lm_idx < v_frame_data.shape[0]:
                                v_vis[str(int(lm_idx))] = round(
                                    float(v_frame_data[lm_idx, 2]), 4
                                )
                        anchor_checks.append({
                            "name": check.id,
                            "creator_observed": (
                                round(float(c_result.observed), 4)
                                if c_result.observed is not None else None
                            ),
                            "viewer_observed": (
                                round(float(v_result.observed), 4)
                                if v_result.observed is not None else None
                            ),
                            "creator_visibility": c_vis,
                            "viewer_visibility": v_vis,
                        })

                    trajectory.append({
                        "anchor_pct": pct,
                        "creator_frame": int(creator_frame_global),
                        "viewer_frame": int(viewer_frame_global),
                        "checks": anchor_checks,
                    })

            diag_reps.append({
                "rep_index": idx,
                "viewer_range": {
                    "start": int(viewer_rep.start_frame),
                    "peak": int(viewer_peak_global),
                    "end": int(viewer_rep.end_frame),
                },
                "creator_match": {
                    "start": creator_match_start,
                    "peak": creator_match_peak,
                    "end": creator_match_end,
                },
                "dtw_peak_neighbors": dtw_neighbors,
                "checks": checks_diag,
                "total": int(overall),
                "trajectory": trajectory,
                "dtw_rep_cost": dtw_rep_cost,
            })

    final_score = int(round(np.mean(per_rep_scores))) if per_rep_scores else 0
    arm, hip, timing = _derive_legacy_scores(all_results, rubric)
    avg_conf = float(np.mean([r.confidence for r in all_results if r.passed])) if all_results else 0.0

    # Check breakdown: use the best-scoring rep's checks so the UI shows
    # achievable-looking numbers, not the worst rep
    best_rep_idx = int(np.argmax(per_rep_scores)) if per_rep_scores else 0
    checks_per_rep = len(rubric.checks)
    best_start = best_rep_idx * checks_per_rep
    best_results = all_results[best_start:best_start + checks_per_rep]

    result = RubricScoreResult(
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

    if diagnose:
        return result, {"reps": diag_reps}
    return result


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