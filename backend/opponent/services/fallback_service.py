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
                "Stress tugged your timing at the worst bounce; I cashed the point.",
                "That point had panic fingerprints all over it.",
                "You held the rally until pressure touched it, then it cracked.",
            ],
            ("ai_score", "frustration"): [
                "You swung for revenge instead of placement, so I took the lane.",
                "Frustration rushed the read and left the corner wide open.",
                "Tilt called the shot; I just collected.",
            ],
            ("player_score", "focus"): [
                "Clean angle. Focus looked sharp for once, I will test it again.",
                "You found the seam. Let's see if your focus survives a longer rally.",
                "Good point. Keep that concentration from slipping on the next exchange.",
            ],
            ("player_score", "alertness"): [
                "Quick read there. Stay that awake when the pace climbs.",
                "Nice reaction. One sleepy bounce and that lead vanishes.",
                "Good pickup. Alertness looked sharp for exactly one rally.",
            ],
            ("near_score", "stress"): [
                "That save was all panic and no plan.",
                "Near miss. Stress is now driving and your paddle is just riding shotgun.",
                "You escaped the point, not the pressure.",
            ],
            ("near_score", "frustration"): [
                "That scramble sounded like frustration in surround sound.",
                "Close call. Frustration is creeping into every recovery step.",
                "You got away with that rally, but your composure did not.",
            ],
            ("default", "stress"): [
                "Tempo looks quick, but stress keeps nudging your decisions off-line.",
                "Your rhythm has a panic hitch and I can hear it every rally.",
                "Hands look steady. Choices look rushed.",
            ],
            ("default", "frustration"): [
                "Frustration is writing your game plan one rushed swing at a time.",
                "You keep arguing with the rally instead of reading the angle.",
                "Tilt keeps whispering and you keep taking notes.",
            ],
            ("default", "focus"): [
                "Focus flashes bright, then vanishes right before the finish.",
                "You read move one. I win on move two.",
                "Concentration is present; discipline is still buffering.",
            ],
            ("default", "alertness"): [
                "Alertness is decent, anticipation still shows up late.",
                "Quick reactions, late reads. That's the gap I farm.",
                "Awake enough to chase, not early enough to control.",
            ],
        }
        return pools.get((event_bucket, dominant_metric), [])
