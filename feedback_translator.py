"""
feedback_translator.py — Turn per-check numeric results into human feedback.

REVIEW FIX #6: "left_elbow: 43/100" is useless UX. This layer produces
"Bend your left elbow more at the bottom" style strings. Structured output
so ai_coach.py / Claude API can polish further before UI display.

Not ML. Just a lookup table keyed by (joint, direction, severity).
"""
from __future__ import annotations
from typing import Literal
from dataclasses import dataclass

Severity = Literal["ok", "minor", "major"]

# Joint → human-readable name + default action verb
_JOINT_LABELS: dict[str, tuple[str, str, str]] = {
    # (human_label, "bend_more" verb, "bend_less" verb)
    "l_elbow":    ("left elbow",    "bend more",  "extend more"),
    "r_elbow":    ("right elbow",   "bend more",  "extend more"),
    "l_shoulder": ("left shoulder", "raise more", "lower"),
    "r_shoulder": ("right shoulder","raise more", "lower"),
    "l_hip":      ("left hip",      "flex more",  "extend more"),
    "r_hip":      ("right hip",     "flex more",  "extend more"),
    "l_knee":     ("left knee",     "bend more",  "straighten more"),
    "r_knee":     ("right knee",    "bend more",  "straighten more"),
}


def _severity(score: int) -> Severity:
    if score >= 85:
        return "ok"
    if score >= 60:
        return "minor"
    return "major"


def _joint_from_check_id(check_id: str) -> str:
    """Check IDs are like 'l_elbow_at_bottom' → 'l_elbow'."""
    return check_id.split("_at_")[0] if "_at_" in check_id else check_id


@dataclass
class FeedbackItem:
    check_id: str
    severity: Severity
    message: str
    score: int
    observed: float
    target: float
    direction: Literal["bend_more", "bend_less", "ok"]


def translate_check(
    check_id: str,
    score: int,
    observed: float,
    target: float,
    check_type: str = "angle",
) -> FeedbackItem:
    """Convert one check result into a human-readable feedback item."""
    sev = _severity(score)
    joint_key = _joint_from_check_id(check_id)
    label, verb_more, verb_less = _JOINT_LABELS.get(
        joint_key, (joint_key.replace("_", " "), "adjust", "adjust")
    )

    if sev == "ok":
        return FeedbackItem(
            check_id=check_id, severity="ok",
            message=f"{label.capitalize()} looks good.",
            score=score, observed=observed, target=target, direction="ok",
        )

    # For angle checks: observed < target means joint is MORE bent than creator
    # (smaller angle = more flexion), so viewer needs to extend.
    # Observed > target means viewer is too extended — needs to bend more.
    if check_type == "angle":
        if observed > target:
            direction = "bend_more"
            action = verb_more
        else:
            direction = "bend_less"
            action = verb_less
    else:
        direction = "bend_more"
        action = "adjust"

    delta = abs(observed - target)
    # Avoid "bend more more" — if action already ends in "more", drop the adverb
    action_ends_in_more = action.strip().endswith("more")
    if sev == "minor":
        adverb = "" if action_ends_in_more else " slightly"
        msg = f"{label.capitalize()}: {action}{adverb} (off by {delta:.0f}°)."
    else:
        adverb = "" if action_ends_in_more else " more"
        msg = f"{label.capitalize()}: {action}{adverb} (off by {delta:.0f}°)."

    return FeedbackItem(
        check_id=check_id, severity=sev, message=msg,
        score=score, observed=observed, target=target, direction=direction,
    )


def translate_all(checks: list[dict]) -> list[dict]:
    """Convert a list of check dicts (as returned by rubric_scorer) into feedback dicts."""
    out = []
    for c in checks:
        if not c.get("passed", True) or c.get("observed") is None:
            continue
        fb = translate_check(
            check_id=c["id"], score=c["score"],
            observed=c["observed"], target=c["target"],
        )
        out.append({
            "check_id": fb.check_id,
            "severity": fb.severity,
            "message": fb.message,
            "direction": fb.direction,
        })
    return out


def summary_line(feedback_items: list[dict]) -> str:
    """One-line summary of the top 1-2 most impactful issues."""
    majors = [f for f in feedback_items if f["severity"] == "major"]
    minors = [f for f in feedback_items if f["severity"] == "minor"]
    if majors:
        return majors[0]["message"]
    if minors:
        return minors[0]["message"]
    return "Great form — keep it up!"