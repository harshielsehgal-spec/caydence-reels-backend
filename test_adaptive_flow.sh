#!/usr/bin/env bash
# test_adaptive_flow.sh — end-to-end test for the adaptive (no-spec) engine.
#
# Usage:
#   1. Start server: python3 main.py
#   2. In another terminal: bash test_adaptive_flow.sh
#
# Requires ref_video.mp4 and attempt_video.mp4 in current dir.

set -euo pipefail

HOST="${HOST:-http://localhost:8002}"
REEL_ID="adaptive-$(date +%s)"

echo "=== 1. Build ADAPTIVE rubric for $REEL_ID (zero config) ==="
curl -sS -X POST "$HOST/reels/rubric_adaptive" \
  -F "reel_id=$REEL_ID" \
  -F "reference_video=@ref_video.mp4;type=video/mp4" \
  | python3 -m json.tool

echo
echo "=== 2. Submit attempt video ==="
JOB_RESP=$(curl -sS -X POST "$HOST/reels/analyze_v2?reel_id=$REEL_ID" \
  -F "attempt_video=@attempt_video.mp4;type=video/mp4")
echo "$JOB_RESP" | python3 -m json.tool
JOB_ID=$(echo "$JOB_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

echo
echo "=== 3. Poll for result ==="
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    sleep 2
    RESULT=$(curl -sS "$HOST/reels/result/$JOB_ID")
    STATUS=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
    echo "[$i] status=$STATUS"
    if [ "$STATUS" = "complete" ] || [ "$STATUS" = "failed" ]; then
        echo "$RESULT" | python3 -m json.tool
        break
    fi
done

echo
echo "=== 4. Get human-readable feedback ==="
curl -sS -X POST "$HOST/reels/feedback_v2" \
  -H "Content-Type: application/json" \
  -d "{\"job_id\": \"$JOB_ID\"}" \
  | python3 -m json.tool
