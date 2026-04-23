import cv2
import gc
import numpy as np
from dataclasses import dataclass
from typing import Optional
import math
import mediapipe as mp
from dtaidistance import dtw_ndim   # pip install "dtaidistance>=2.3.10"

# Initialize at module level — fixes thread executor issues
mp_pose = mp.solutions.pose

# Joint indices
ARM_LANDMARKS  = [11, 12, 13, 14, 15, 16]
HIP_LANDMARKS  = [23, 24, 25, 26]
CORE_LANDMARKS = [11, 12, 23, 24]

# ─────────────────────────────────────────────────────────────────────────────
# DTW config
# ─────────────────────────────────────────────────────────────────────────────

# Sakoe-Chiba band: ±10% of the longer sequence.
_DTW_WINDOW_FRACTION = 0.10

# Score = exp(-dist / scale). Calibrate with calibrate_dtw_scale.py.
# Angles normalised to [0,1] so good-attempt distances are 0.1–0.5.
_DTW_SCALE = 8.0

# Per angle-group weights (Phase 3: override per sport)
_DTW_DEFAULT_WEIGHTS = {"arm": 1.0, "hip": 1.0}


@dataclass
class SimilarityResult:
    score: int
    arm_alignment: int
    hip_position: int
    timing_sync: int
    frame_count: int
    confidence: float


def extract_keypoints(video_path: str, max_frames: int = 60) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    sample_indices = set(
        np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    )
    keypoints = []
    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in sample_indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    kps = np.array([[lm.x, lm.y, lm.visibility]
                                    for lm in result.pose_landmarks.landmark])
                    keypoints.append(kps)
            frame_idx += 1
    cap.release()
    gc.collect()
    return np.array(keypoints) if keypoints else None


def _compute_frame_velocities(keypoints: np.ndarray) -> np.ndarray:
    core = keypoints[:, CORE_LANDMARKS, :2].mean(axis=1)
    velocities = np.linalg.norm(np.diff(core, axis=0), axis=1)
    return np.concatenate([[0], velocities])


def _detect_action_window(velocities: np.ndarray, phase_start: float) -> tuple:
    n = len(velocities)
    hard_start = int(n * phase_start)
    if hard_start >= n - 2:
        return (0, n)
    window = velocities[hard_start:]
    if window.max() < 1e-6:
        return (hard_start, n)
    peak_offset = int(np.argmax(window))
    peak_idx = hard_start + peak_offset
    threshold = window.max() * 0.20
    onset = hard_start
    for i in range(peak_idx, hard_start - 1, -1):
        if velocities[i] < threshold:
            onset = i
            break
    end = n
    for i in range(peak_idx, n):
        if velocities[i] < threshold:
            end = min(i + 2, n)
            break
    if end - onset < int(n * 0.20):
        onset = hard_start
        end = n
    return (onset, end)


def extract_keypoints_phased(
    video_path: str,
    max_frames: int = 60,
    action_phase_start: float = 0.0,
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    probe_count = min(30, total_frames)
    probe_indices = set(np.linspace(0, total_frames - 1, probe_count, dtype=int))
    probe_kps = []
    with mp_pose.Pose(
        static_image_mode=True, model_complexity=0,
        min_detection_confidence=0.4, min_tracking_confidence=0.4,
    ) as pose:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in probe_indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    kps = np.array([[lm.x, lm.y, lm.visibility]
                                    for lm in result.pose_landmarks.landmark])
                    probe_kps.append(kps)
                else:
                    probe_kps.append(np.zeros((33, 3)))
            frame_idx += 1
    if not probe_kps:
        cap.release()
        return None
    probe_arr = np.array(probe_kps)
    velocities = _compute_frame_velocities(probe_arr)
    action_start, action_end = _detect_action_window(velocities, action_phase_start)
    probe_list = sorted(probe_indices)
    fs_action_start = probe_list[min(action_start, len(probe_list) - 1)]
    fs_action_end   = probe_list[min(action_end,   len(probe_list) - 1)]
    action_ratio  = (fs_action_end - fs_action_start) / max(total_frames, 1)
    action_frames = int(max_frames * action_ratio * 3)
    action_frames = min(action_frames, int(max_frames * 0.75))
    other_frames  = max_frames - action_frames
    action_indices = set(
        np.linspace(fs_action_start, fs_action_end - 1,
                    max(action_frames, 1), dtype=int).tolist()
    )
    other_range = list(range(0, fs_action_start)) + list(range(fs_action_end, total_frames))
    if other_range and other_frames > 0:
        other_indices = set(
            np.array(other_range)[
                np.linspace(0, len(other_range) - 1,
                            min(other_frames, len(other_range)), dtype=int)
            ].tolist()
        )
    else:
        other_indices = set()
    sample_indices = action_indices | other_indices
    keypoints = []
    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in sample_indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    kps = np.array([[lm.x, lm.y, lm.visibility]
                                    for lm in result.pose_landmarks.landmark])
                    keypoints.append(kps)
            frame_idx += 1
    cap.release()
    gc.collect()
    return np.array(keypoints) if keypoints else None


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    normalized = []
    for frame in keypoints:
        xy = frame[:, :2].copy()
        hip_mid = (xy[23] + xy[24]) / 2
        xy -= hip_mid
        shoulder_mid = (xy[11] + xy[12]) / 2
        torso_height = np.linalg.norm(shoulder_mid)
        if torso_height > 0.01:
            xy /= torso_height
        normalized.append(xy)
    return np.array(normalized)


def compute_joint_angle(p1, p2, p3) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


def extract_angle_sequence(keypoints: np.ndarray) -> dict:
    arm_angles, hip_angles = [], []
    for frame in keypoints:
        r_elbow = compute_joint_angle(frame[12], frame[14], frame[16])
        l_elbow = compute_joint_angle(frame[11], frame[13], frame[15])
        r_knee  = compute_joint_angle(frame[24], frame[26], frame[28])
        l_knee  = compute_joint_angle(frame[23], frame[25], frame[27])
        r_hip   = compute_joint_angle(frame[12], frame[24], frame[26])
        arm_angles.append([r_elbow, l_elbow])
        hip_angles.append([r_knee, l_knee, r_hip])
    return {"arm": np.array(arm_angles), "hip": np.array(hip_angles)}


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == target_len:
        return seq
    orig = np.linspace(0, 1, len(seq))
    targ = np.linspace(0, 1, target_len)
    return np.array([np.interp(targ, orig, seq[:, i])
                     for i in range(seq.shape[1])]).T


def score_sequence_similarity(ref_seq: np.ndarray, attempt_seq: np.ndarray) -> int:
    attempt_r  = resample_sequence(attempt_seq, len(ref_seq))
    diff       = np.abs(ref_seq - attempt_r)
    similarity = 1.0 - np.mean(np.clip(diff / 90.0, 0, 1))
    return int(np.clip(similarity * 100, 0, 100))


def compute_timing_sync(
    ref_kps:       np.ndarray,
    attempt_kps:   np.ndarray,
    sport_weights: Optional[dict] = None,
) -> int:
    """
    Timing sync using DTW on joint angle sequences — camera-angle invariant.
    Angles (degrees) divided by 180 → [0,1] before DTW.
    Returns int [0-100], identical interface to original.
    """
    if len(ref_kps) < 2 or len(attempt_kps) < 2:
        return 50

    ref_norm   = normalize_keypoints(ref_kps)
    att_norm   = normalize_keypoints(attempt_kps)
    ref_angles = extract_angle_sequence(ref_norm)
    att_angles = extract_angle_sequence(att_norm)

    weights      = sport_weights or _DTW_DEFAULT_WEIGHTS
    score_total  = 0.0
    weight_total = 0.0

    for group, w in weights.items():
        ref_seq = (ref_angles[group] / 180.0).astype(np.float64)
        att_seq = (att_angles[group] / 180.0).astype(np.float64)
        n       = max(len(ref_seq), len(att_seq))
        window  = max(1, int(n * _DTW_WINDOW_FRACTION))
        dist    = float(dtw_ndim.distance(ref_seq, att_seq, window=window))
        score   = float(np.exp(-dist / _DTW_SCALE))
        score_total  += score * w
        weight_total += w

    final = score_total / weight_total if weight_total else 0.0
    return int(np.clip(final * 100, 0, 100))


def compute_similarity(ref_keypoints: np.ndarray,
                        attempt_keypoints: np.ndarray) -> SimilarityResult:
    ref_norm   = normalize_keypoints(ref_keypoints)
    att_norm   = normalize_keypoints(attempt_keypoints)
    ref_angles = extract_angle_sequence(ref_norm)
    att_angles = extract_angle_sequence(att_norm)
    arm_score    = score_sequence_similarity(ref_angles["arm"], att_angles["arm"])
    hip_score    = score_sequence_similarity(ref_angles["hip"], att_angles["hip"])
    timing_score = compute_timing_sync(ref_keypoints, attempt_keypoints)
    overall      = int(arm_score * 0.35 + hip_score * 0.35 + timing_score * 0.30)
    confidence   = min(len(attempt_keypoints) / 30.0, 1.0)
    return SimilarityResult(
        score=overall,
        arm_alignment=arm_score,
        hip_position=hip_score,
        timing_sync=timing_score,
        frame_count=len(attempt_keypoints),
        confidence=round(confidence, 2),
    )


if __name__ == "__main__":
    rng  = np.random.default_rng(42)
    ref  = rng.random((80, 33, 3)).astype(np.float32)
    user = rng.random((90, 33, 3)).astype(np.float32)
    ref[:,  :, 2] = np.clip(rng.normal(0.82, 0.15, (80, 33)), 0, 1)
    user[:, :, 2] = np.clip(rng.normal(0.78, 0.18, (90, 33)), 0, 1)
    timing = compute_timing_sync(ref, user)
    print(f"compute_timing_sync  -> {timing}  (int, drop-in)")
    result = compute_similarity(ref, user)
    print(f"compute_similarity   -> {result}")