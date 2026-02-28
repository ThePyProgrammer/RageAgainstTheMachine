"""Rule-based fallback taunt generation."""

from __future__ import annotations

from opponent.models import GameEventType, OpponentGameEvent
from opponent.services.difficulty_service import MetricsSnapshot


class RuleBasedFallbackService:
    """Deterministic fallback taunts when LLM or TTS is unavailable."""

    def generate_taunt(
        self,
        event: OpponentGameEvent,
        metrics: MetricsSnapshot,
    ) -> str:
        stress_pct = int(round(metrics.stress * 100))
        frustration_pct = int(round(metrics.frustration * 100))

        if event.event == GameEventType.AI_SCORE:
            return f"Nice try. You are at {stress_pct}% stress now."
        if event.event == GameEventType.PLAYER_SCORE:
            return f"Lucky hit. Keep that focus up at {int(round(metrics.focus * 100))}%."
        if metrics.frustration > 0.65:
            return f"I see {frustration_pct}% frustration. Stay steady."
        if metrics.stress > 0.7:
            return f"You are sweating at {stress_pct}% stress."
        return "Won't be so lucky next time."

