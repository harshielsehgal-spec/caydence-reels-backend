#!/usr/bin/env bash
# test_rubric_flow.sh — end-to-end smoke test for rubric scoring.
#
# Usage:
#   1. Start server:  python3 main.py
#   2. In another terminal: bash test_rubric_flow.sh
#
# Requires:  ref_video.mp4 and attempt_video.mp4 in the current dir
#            (you already have these)

set -euo pipefail

HOST="${HOST:-http://localhost:8002}"
REEL_ID="test-reel-$(date +%s)"

echo "=== 1. Build rubric for reel: $REEL_ID ==="

# Example spec: two angle checks at one key frame.
# Adjust timestamp_ms to match your ref_video.mp4's actual key moment.
SPEC=$(cat <<EOF
{
  "reel_id": "$REEL_ID",
  "sport": "cricket_bowling",
  "key_frames": [
    {"id": "release", "timestamp_ms": 1200}
  ],
  "checks": [
    {
      "id": "right_elbow_at_release",
      "key_frame_id": "release",
      "type": "angle",
      "landmarks": [12, 14, 16],
      "weight": 0.5
    },
    {
      "id": "left_elbow_at_release",
      "key_frame_id": "release",
      "type": "angle",
      "landmarks": [11, 13, 15],
      "weight": 0.3
    },
    {
      "id": "hip_rotation_at_release",
      "key_frame_id": "release",
      "type": "relative_position",
      "landmarks": [24, 23],
      "axis": "x",
      "weight": 0.2
    }
  ]
}
EOF
)

curl -sS -X POST "$HOST/reels/rubric" \
  -F "spec_json=$SPEC" \
  -F "reference_video=@ref_video.mp4;type=video/mp4" \
  | python3 -m json.tool

echo
echo "=== 2. Fetch stored rubric ==="
curl -sS "$HOST/reels/rubric/$REEL_ID" | python3 -m json.tool

echo
echo "=== 3. Submit attempt video ==="
JOB_RESP=$(curl -sS -X POST "$HOST/reels/analyze_v2?reel_id=$REEL_ID&sport=cricket_bowling" \
  -F "attempt_video=@attempt_video.mp4;type=video/mp4")
echo "$JOB_RESP" | python3 -m json.tool
JOB_ID=$(echo "$JOB_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

echo
echo "=== 4. Poll for result (job: $JOB_ID) ==="
for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 2
    RESULT=$(curl -sS "$HOST/reels/result/$JOB_ID")
    STATUS=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
    echo "[$i] status=$STATUS"
    if [ "$STATUS" = "complete" ] || [ "$STATUS" = "failed" ]; then
        echo "$RESULT" | python3 -m json.tool
        break
    fi
done
