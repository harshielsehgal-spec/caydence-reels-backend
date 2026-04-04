import json
import numpy as np
from pathlib import Path
from typing import Optional
from pose_similarity import (
    normalize_keypoints,
    compute_timing_sync,
    SimilarityResult,
)

_PROFILES_PATH = Path(__file__).parent / "sport_profiles.json"
with open(_PROFILES_PATH) as f:
    SPORT_PROFILES: dict = json.load(f)


def get_profile(sport: Optional[str]) -> dict:
    if sport and sport in SPORT_PROFILES:
        return SPORT_PROFILES[sport]
    return SPORT_PROFILES["_generic"]


def score_joint_group(
    ref_kps: np.ndarray,
    att_kps: np.ndarray,
    joint_indices: list,
) -> int:
    """
    Score positional similarity for specific joint group.
    [UNCERTAIN] Assumes MediaPipe landmark order is stable
    across mediapipe 0.10.x patch versions.
    """
    valid = [i for i in joint_indices if i < ref_kps.shape[1]]
    if not valid:
        return 50

    ref_sub = ref_kps[:, valid, :]
    att_sub = att_kps[:, valid, :]

    target_len = len(ref_sub)
    if len(att_sub) != target_len:
        resampled = np.zeros((target_len, len(valid), ref_sub.shape[2]))
        for j in range(len(valid)):
            for d in range(ref_sub.shape[2]):
                resampled[:, j, d] = np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(att_sub)),
                    att_sub[:, j, d]
                )
        att_sub = resampled

    diff = np.abs(ref_sub - att_sub)
    similarity = 1.0 - np.mean(np.clip(diff / 0.3, 0, 1))
    return int(np.clip(similarity * 100, 0, 100))


def compute_sport_similarity(
    ref_keypoints: np.ndarray,
    attempt_keypoints: np.ndarray,
    sport: Optional[str] = None,
) -> SimilarityResult:
    """
    Sport-aware similarity scoring using profile weight vectors.
    Falls back to _generic profile if sport is None or unknown.
    """
    profile = get_profile(sport)
    weights = profile["weights"]
    joint_groups = profile["key_joint_groups"]
    timing_override = profile.get("timing_weight_override")

    ref_norm = normalize_keypoints(ref_keypoints)[:, :, :2]
    att_norm = normalize_keypoints(attempt_keypoints)[:, :, :2]

    # Core scores
    arm_joints = joint_groups.get("arm_alignment", [11, 12, 13, 14, 15, 16])
    arm_score = score_joint_group(ref_norm, att_norm, arm_joints)

    hip_joints = joint_groups.get("hip_rotation", [23, 24, 25, 26])
    hip_score = score_joint_group(ref_norm, att_norm, hip_joints)

    timing_score = compute_timing_sync(ref_keypoints, attempt_keypoints)

    # Optional sport-specific scores
    wrist_score = None
    if "wrist_position" in joint_groups and "wrist_position" in weights:
        wrist_score = score_joint_group(
            ref_norm, att_norm, joint_groups["wrist_position"]
        )

    leg_score = None
    if "leg_extension" in joint_groups and "leg_extension" in weights:
        leg_score = score_joint_group(
            ref_norm, att_norm, joint_groups["leg_extension"]
        )

    full_body_score = None
    if "full_body_alignment" in joint_groups and "full_body_alignment" in weights:
        full_body_score = score_joint_group(
            ref_norm, att_norm, joint_groups["full_body_alignment"]
        )

    back_score = None
    if "back_straightness" in joint_groups and "back_straightness" in weights:
        back_score = score_joint_group(
            ref_norm, att_norm, joint_groups["back_straightness"]
        )

    # Weighted sum
    overall = 0.0
    total_weight = 0.0

    def add(score, key):
        nonlocal overall, total_weight
        w = weights.get(key, 0)
        if w > 0:
            overall += score * w
            total_weight += w

    add(arm_score, "arm_alignment")
    add(hip_score, "hip_rotation")

    t_weight = timing_override if timing_override is not None \
        else weights.get("timing_sync", 0)
    overall += timing_score * t_weight
    total_weight += t_weight

    if wrist_score is not None:
        add(wrist_score, "wrist_position")
    if leg_score is not None:
        add(leg_score, "leg_extension")
    if full_body_score is not None:
        add(full_body_score, "full_body_alignment")
    if back_score is not None:
        add(back_score, "back_straightness")

    final = int(np.clip(overall / total_weight if total_weight > 0 else 0, 0, 100))
    confidence = min(len(attempt_keypoints) / 30.0, 1.0)

    return SimilarityResult(
        score=final,
        arm_alignment=arm_score,
        hip_position=hip_score,
        timing_sync=timing_score,
        frame_count=len(attempt_keypoints),
        confidence=round(confidence, 2),
    )