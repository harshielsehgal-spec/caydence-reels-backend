"""
verify_skeleton.py — Visual verification that /reels/{reel_id}/skeleton
returns landmarks that actually land on the creator's body.

Workflow:
  1. GET the skeleton from the running server
  2. Read the same frame_index from the source video
  3. Plot landmarks scaled to the frame dimensions
  4. Save annotated image to skeleton_overlay.png
  5. Print a pass/fail verdict based on key joint placement

Usage (on Mac, with server running on port 8002):
    python3 verify_skeleton.py <reel_id> <path_to_creator_video>

Example:
    python3 verify_skeleton.py skel_test ref_video.mp4
"""
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error

import cv2
import numpy as np

HOST = "http://localhost:8002"

# MediaPipe Pose connections (subset — the ones that matter for visual sanity)
POSE_CONNECTIONS = [
    # Face/torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Landmark labels for major joints
LABELS = {
    0: "nose", 11: "L_shoulder", 12: "R_shoulder",
    13: "L_elbow", 14: "R_elbow", 15: "L_wrist", 16: "R_wrist",
    23: "L_hip", 24: "R_hip", 25: "L_knee", 26: "R_knee",
    27: "L_ankle", 28: "R_ankle",
}

PRIMARY_LANDMARKS = (11, 12, 23, 24, 25, 26)  # checks the engine cares about


def fetch_skeleton(reel_id: str) -> dict:
    url = f"{HOST}/reels/{reel_id}/skeleton"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} on {url}: {e.read().decode()}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach server at {HOST}: {e}")


def read_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {frame_index} from {video_path}")
    return frame


def overlay_landmarks(frame: np.ndarray, payload: dict) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    lms = {lm["index"]: lm for lm in payload["landmarks"]}

    # Draw bones
    for a, b in POSE_CONNECTIONS:
        if a not in lms or b not in lms:
            continue
        if lms[a]["visibility"] < 0.3 or lms[b]["visibility"] < 0.3:
            continue
        x1, y1 = int(lms[a]["x"] * w), int(lms[a]["y"] * h)
        x2, y2 = int(lms[b]["x"] * w), int(lms[b]["y"] * h)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw joints + labels
    for idx, lm in lms.items():
        if lm["visibility"] < 0.3:
            continue
        x, y = int(lm["x"] * w), int(lm["y"] * h)
        color = (0, 0, 255) if idx in PRIMARY_LANDMARKS else (255, 255, 0)
        cv2.circle(img, (x, y), 5, color, -1)
        if idx in LABELS:
            cv2.putText(
                img, LABELS[idx], (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
    return img


def sanity_check(payload: dict) -> tuple[bool, list[str]]:
    """Heuristic checks on landmark placement."""
    msgs = []
    lms = {lm["index"]: lm for lm in payload["landmarks"]}

    # All x,y in 0-1
    out_of_range = [
        lm["index"] for lm in payload["landmarks"]
        if not (0.0 <= lm["x"] <= 1.0 and 0.0 <= lm["y"] <= 1.0)
    ]
    if out_of_range:
        msgs.append(f"FAIL: landmarks out of [0,1]: {out_of_range}")

    # Both shoulders should exist with reasonable visibility
    primary_visible = [i for i in PRIMARY_LANDMARKS
                       if i in lms and lms[i]["visibility"] >= 0.5]
    msgs.append(
        f"Primary landmarks (11,12,23,24,25,26) above visibility 0.5: "
        f"{len(primary_visible)}/6"
    )

    # Shoulders should be above hips (smaller y in image coords)
    if 11 in lms and 23 in lms:
        if lms[11]["y"] >= lms[23]["y"]:
            msgs.append("WARN: L_shoulder y >= L_hip y (subject upside down or very rotated?)")

    return not bool(out_of_range), msgs


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 verify_skeleton.py <reel_id> <path_to_creator_video>")
        sys.exit(1)
    reel_id, video_path = sys.argv[1], sys.argv[2]

    print(f"[1] GET {HOST}/reels/{reel_id}/skeleton")
    payload = fetch_skeleton(reel_id)
    print(f"    frame_index: {payload['frame_index']}")
    print(f"    landmarks:   {len(payload['landmarks'])}")
    print(f"    dimensions:  {payload['width']} x {payload['height']}")

    print(f"\n[2] Reading frame {payload['frame_index']} from {video_path}")
    frame = read_frame(video_path, payload["frame_index"])
    print(f"    frame shape: {frame.shape}")

    if (frame.shape[1], frame.shape[0]) != (payload["width"], payload["height"]):
        print(
            f"    WARN: video dim {frame.shape[1]}x{frame.shape[0]} "
            f"≠ payload dim {payload['width']}x{payload['height']}"
        )

    print(f"\n[3] Overlaying landmarks")
    annotated = overlay_landmarks(frame, payload)
    out_path = "skeleton_overlay.png"
    cv2.imwrite(out_path, annotated)
    print(f"    saved -> {out_path}")

    print(f"\n[4] Sanity checks")
    ok, msgs = sanity_check(payload)
    for m in msgs:
        print(f"    {m}")

    print(f"\n[5] VERDICT")
    if ok:
        print(f"    Coords in valid range. Open {out_path} and visually confirm")
        print(f"    joints land on the creator's body.")
    else:
        print(f"    FAIL — see messages above.")


if __name__ == "__main__":
    main()