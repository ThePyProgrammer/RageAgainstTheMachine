"""Rule-based fallback taunt generation."""

from __future__ import annotations

import hashlib

from opponent.models import GameEventType, OpponentGameEvent
from opponent.services.difficulty_service import MetricsSnapshot


class RuleBasedFallbackService:
    """Deterministic fallback taunts when LLM or TTS is unavailable."""

    def generate_taunt(
        self,
        event: OpponentGameEvent,
        metrics: MetricsSnapshot,
    ) -> str:
        dominant_metric = self._dominant_metric(metrics)
        event_bucket = event.event.value
        lines = self._line_pool(event_bucket=event_bucket, dominant_metric=dominant_metric)
        if not lines:
            lines = self._line_pool(event_bucket="default", dominant_metric=dominant_metric)

        idx = self._stable_index(
            f"{event.event_id}|{event.event.value}|{dominant_metric}|{event.score.player}|{event.score.ai}",
            len(lines),
        )
        return lines[idx]

    @staticmethod
    def _dominant_metric(metrics: MetricsSnapshot) -> str:
        values = {
            "stress": metrics.stress,
            "frustration": metrics.frustration,
            "focus": metrics.focus,
            "alertness": metrics.alertness,
        }
        return max(values, key=values.get)

    @staticmethod
    def _stable_index(seed: str, size: int) -> int:
        if size <= 1:
            return 0
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % size

    @staticmethod
    def _line_pool(event_bucket: str, dominant_metric: str) -> list[str]:
        pools: dict[tuple[str, str], list[str]] = {
            ("ai_score", "stress"): [
                "You flinched at the key moment; stress stole that point for me.",
                "That was a pressure point, and stress cashed it in.",
                "You felt the heat and your timing folded right on cue.",
            ],
            ("ai_score", "frustration"): [
                "You swung angry instead of smart; I gladly took the point.",
                "Frustration rushed your read and handed me the lane.",
                "Tilt made that decision for you, not skill.",
            ],
            ("player_score", "focus"): [
                "Clean hit, but your focus flickers when the rally gets ugly.",
                "Nice conversion. Let's see that focus survive real pressure.",
                "You took one, but your focus is already leaking at the edges.",
            ],
            ("player_score", "alertness"): [
                "You were awake for that one. Stay that sharp if you can.",
                "Good reaction there. Keep that alertness from dipping.",
                "Fast read. Let's test how long that alertness lasts.",
            ],
            ("near_score", "stress"): [
                "You were one heartbeat from disaster and stress knows it.",
                "That was a near miss dressed as confidence.",
                "You survived that scramble, but your nerves did not.",
            ],
            ("near_score", "frustration"): [
                "You almost cracked on that scramble; frustration is creeping in.",
                "That rally had panic in it, and I heard every beat.",
                "You escaped the point, not the pressure.",
            ],
            ("default", "stress"): [
                "Pressure is climbing and your rhythm is starting to wobble.",
                "You are playing fast, but stress is steering.",
                "Your hands look calm; your decisions do not.",
            ],
            ("default", "frustration"): [
                "Frustration is writing your playbook one rushed move at a time.",
                "You are arguing with the rally instead of reading it.",
                "Tilt is whispering and you keep listening.",
            ],
            ("default", "focus"): [
                "Your focus flashes bright, then vanishes at the worst time.",
                "You see the first move. I win on the second.",
                "Focus is there, discipline is not.",
            ],
            ("default", "alertness"): [
                "Your alertness is decent, but your anticipation is late.",
                "You react quickly, just not early enough.",
                "You are awake, but still one beat behind.",
            ],
        }
        return pools.get((event_bucket, dominant_metric), [])
