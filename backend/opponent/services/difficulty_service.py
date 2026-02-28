"""Difficulty computation utilities for AI opponent adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from opponent.models import EventContextPayload, GameEventType, ScorePayload


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


@dataclass
class MetricsSnapshot:
    stress: float
    frustration: float
    focus: float
    alertness: float


def compute_prior_difficulty(
    event: GameEventType,
    score: ScorePayload,
    metrics: MetricsSnapshot,
    event_context: EventContextPayload | None,
) -> float:
    lead = clamp((score.player - score.ai) / 10.0, -1.0, 1.0)
    event_boost = 0.0

    if event == GameEventType.NEAR_SCORE:
        near_side = event_context.near_side if event_context else None
        if near_side and near_side.value == "player_goal":
            event_boost = 0.03
    elif event == GameEventType.AI_SCORE:
        event_boost = 0.01
    elif event == GameEventType.PLAYER_SCORE:
        event_boost = -0.01

    prior = (
        0.55
        + (0.35 * metrics.stress)
        + (0.10 * metrics.frustration)
        + (0.08 * max(lead, 0.0))
        - (0.04 * max(-lead, 0.0))
        + event_boost
    )

    return clamp01(prior)


def apply_difficulty_control(
    previous: float,
    prior: float,
    model_target: float,
) -> float:
    model_clamped = clamp(model_target, prior - 0.20, prior + 0.20)
    target = (0.70 * model_clamped) + (0.30 * prior)
    delta = clamp(target - previous, -0.08, 0.08)
    return clamp01(previous + delta)

