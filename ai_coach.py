import os
import anthropic
from typing import Optional

# Anthropic client — reads ANTHROPIC_API_KEY from environment
_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SPORT_DISPLAY = {
    "cricket_bowling": "cricket bowling",
    "football_freekick": "football free kick",
    "gym_pushup": "gym push-up",
    "badminton_smash": "badminton smash",
    "basketball_layup": "basketball layup",
    "yoga_pose": "yoga",
    "generic": "athletic movement",
}

METRIC_LABELS = {
    "arm": "arm alignment",
    "hip": "hip position",
    "timing": "timing sync",
}


def _weakest_and_strongest(arm: int, hip: int, timing: int) -> tuple:
    scores = {"arm": arm, "hip": hip, "timing": timing}
    weakest = min(scores, key=scores.get)
    strongest = max(scores, key=scores.get)
    return weakest, strongest, scores


async def generate_reel_coaching(
    sport: str,
    creator_name: str,
    arm_score: int,
    hip_score: int,
    timing_score: int,
    overall_score: int,
) -> str:
    """
    Call Claude API and return a 3-sentence sport-specific coaching cue.
    Returns a fallback string if the API call fails.
    """
    sport_label = SPORT_DISPLAY.get(sport, sport.replace("_", " "))
    weakest_key, strongest_key, scores = _weakest_and_strongest(
        arm_score, hip_score, timing_score
    )
    weakest_label = METRIC_LABELS[weakest_key]
    strongest_label = METRIC_LABELS[strongest_key]
    weakest_score = scores[weakest_key]
    strongest_score = scores[strongest_key]
    target_score = min(weakest_score + 7, 100)

    system_prompt = (
        "You are a professional sports performance analyst reviewing pose-matching footage. "
        "You give direct, data-driven coaching — like a coach reviewing match footage, not a motivational chatbot. "
        "Rules: no filler words, no praise like 'great effort' or 'keep it up', no generic tips. "
        "Always reference the specific metric names and scores given. "
        "Max 3 sentences total."
    )

    user_prompt = (
        f"Athlete attempted to match {creator_name}'s {sport_label} technique.\n"
        f"Scores: arm alignment={arm_score}/100, hip position={hip_score}/100, "
        f"timing sync={timing_score}/100, overall={overall_score}/100.\n\n"
        f"Their weakest metric is {weakest_label} at {weakest_score}. "
        f"Their strongest metric is {strongest_label} at {strongest_score}.\n\n"
        f"Give coaching in exactly 3 sentences:\n"
        f"1. Acknowledge what they did well using their {strongest_label} score ({strongest_score}).\n"
        f"2. Identify the specific physical correction needed for their {weakest_label} ({weakest_score}) "
        f"— be concrete and sport-specific to {sport_label}.\n"
        f"3. Set a single performance target for their next attempt "
        f"(aim to get {weakest_label} above {target_score}).\n"
        f"No filler. No encouragement. Just the analysis."
    )

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=120,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    except Exception as e:
        print(f"[ai_coach] Claude API error: {e}")
        # Fallback — deterministic, sport-aware, not generic
        return (
            f"Your {strongest_label} is your strongest asset at {strongest_score}. "
            f"Focus on correcting your {weakest_label} ({weakest_score}) — "
            f"this is the primary gap against {creator_name}'s technique. "
            f"Next attempt: target {weakest_label} above {target_score}."
        )