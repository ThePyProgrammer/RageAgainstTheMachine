"""Prompt construction utilities for opponent personality generation."""

from __future__ import annotations

from opponent.models import OpponentGameEvent
from opponent.services.difficulty_service import MetricsSnapshot


def build_system_prompt(max_taunt_chars: int) -> str:
    return (
        "You are a competitive but friendly game opponent persona.\n"
        "Generate short taunts that feel playful and energetic, never abusive.\n"
        "Hard constraints:\n"
        f"- taunt_text must be <= {max_taunt_chars} characters.\n"
        "- No harassment, hate, slurs, or profanity.\n"
        "- Prioritize stress and frustration over score when suggesting difficulty_target.\n"
        "- Return JSON only with fields: taunt_text (string), difficulty_target (0 to 1).\n"
        "- difficulty_target should rise as stress/frustration rise.\n"
        "- Score should influence difficulty less than stress.\n"
    )


def build_user_prompt(
    event: OpponentGameEvent,
    metrics: MetricsSnapshot,
    prior_difficulty: float,
) -> str:
    near_side = event.event_context.near_side.value if event.event_context and event.event_context.near_side else "none"
    proximity = event.event_context.proximity if event.event_context else None
    proximity_text = f"{proximity:.3f}" if isinstance(proximity, float) else "n/a"

    return (
        "Game event context:\n"
        f"- game_mode: {event.game_mode.value}\n"
        f"- event: {event.event.value}\n"
        f"- score: player={event.score.player}, ai={event.score.ai}\n"
        f"- current_difficulty: {event.current_difficulty:.3f}\n"
        f"- near_side: {near_side}\n"
        f"- proximity: {proximity_text}\n"
        "Command-centre metrics (0 to 1):\n"
        f"- stress: {metrics.stress:.3f}\n"
        f"- frustration: {metrics.frustration:.3f}\n"
        f"- focus: {metrics.focus:.3f}\n"
        f"- alertness: {metrics.alertness:.3f}\n"
        f"- suggested_prior_difficulty: {prior_difficulty:.3f}\n"
        "Return JSON only."
    )

