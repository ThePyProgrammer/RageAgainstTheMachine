"""Session-scoped synthetic metric generation for non-EEG Pong input modes."""

from __future__ import annotations

import random
import time

from opponent.models import EventContextPayload, GameEventType, NearSide, ScorePayload
from opponent.services.difficulty_service import MetricsSnapshot, clamp, clamp01


class SyntheticMetricsService:
    """Generate smooth synthetic command-centre metrics when EEG is unavailable."""

    def __init__(self, seed: int | None = None) -> None:
        if seed is None:
            seed = int(time.time_ns() % (2**32))
        self._rng = random.Random(seed)
        self._state = MetricsSnapshot(
            stress=self._rng.uniform(0.40, 0.55),
            frustration=self._rng.uniform(0.35, 0.50),
            focus=self._rng.uniform(0.50, 0.65),
            alertness=self._rng.uniform(0.45, 0.65),
        )

    def next_metrics(
        self,
        *,
        event: GameEventType,
        score: ScorePayload,
        event_context: EventContextPayload | None,
    ) -> MetricsSnapshot:
        ai_lead = clamp((score.ai - score.player) / 8.0, -1.0, 1.0)
        near_side = event_context.near_side if event_context else None

        player_scored = event == GameEventType.PLAYER_SCORE
        ai_scored = event == GameEventType.AI_SCORE
        near_any = event == GameEventType.NEAR_SCORE
        near_player_goal = near_any and near_side == NearSide.PLAYER_GOAL
        near_ai_goal = near_any and near_side == NearSide.AI_GOAL

        stress_target = clamp01(
            0.46
            + (0.24 * max(ai_lead, 0.0))
            + (0.16 if ai_scored else 0.0)
            + (0.12 if near_player_goal else 0.0)
            - (0.14 if player_scored else 0.0)
        )
        frustration_target = clamp01(
            0.42
            + (0.30 * max(ai_lead, 0.0))
            + (0.20 if ai_scored else 0.0)
            + (0.10 if near_player_goal else 0.0)
            - (0.10 if player_scored else 0.0)
        )
        focus_target = clamp01(
            0.58
            - (0.25 * stress_target)
            - (0.18 * frustration_target)
            + (0.12 if player_scored else 0.0)
            - (0.08 if ai_scored else 0.0)
            + (0.06 if near_ai_goal else 0.0)
        )
        alertness_target = clamp01(
            0.56
            + (0.08 if near_any else 0.0)
            + (0.05 if ai_scored else 0.0)
            - (0.05 if player_scored else 0.0)
            - (0.10 * max(ai_lead, 0.0))
        )

        self._state = MetricsSnapshot(
            stress=self._drift(self._state.stress, stress_target, jitter=0.030),
            frustration=self._drift(self._state.frustration, frustration_target, jitter=0.030),
            focus=self._drift(self._state.focus, focus_target, jitter=0.025),
            alertness=self._drift(self._state.alertness, alertness_target, jitter=0.025),
        )
        return self._state

    def _drift(self, current: float, target: float, *, jitter: float) -> float:
        noise = self._rng.uniform(-jitter, jitter)
        return clamp01(current + ((target - current) * 0.28) + noise)

