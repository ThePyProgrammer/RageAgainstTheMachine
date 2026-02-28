"""Difficulty computation utilities for AI opponent adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from opponent.models import EventContextPayload, GameEventType, NearSide, ScorePayload


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
    ai_lead_ratio = clamp((score.ai - score.player) / 8.0, 0.0, 1.0)
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
        + (0.06 * max(lead, 0.0))
        - (0.16 * max(-lead, 0.0))
        - (0.22 * ai_lead_ratio)
        + event_boost
    )

    return clamp01(prior)


def apply_difficulty_control(
    previous: float,
    prior: float,
    model_target: float,
    *,
    event: GameEventType,
    score: ScorePayload,
    event_context: EventContextPayload | None,
) -> float:
    model_clamped = clamp(model_target, prior - 0.20, prior + 0.20)
    target = (0.70 * model_clamped) + (0.30 * prior)

    ai_lead = max(score.ai - score.player, 0)
    max_down = -0.10

    # If the AI is clearly ahead, force the system to ease off quickly.
    if ai_lead >= 3:
        comeback_scale = clamp((ai_lead - 2) / 5.0, 0.0, 1.0)
        relief_cap = 0.62 - (0.28 * comeback_scale)
        target = min(target, relief_cap) - (0.06 * comeback_scale)
        max_down = -0.14

    max_up = 0.08 if _has_upward_evidence(event=event, score=score, event_context=event_context) else 0.05
    if ai_lead >= 3:
        max_up = min(max_up, 0.02)

    delta = clamp(target - previous, max_down, max_up)
    return clamp01(previous + delta)


def _has_upward_evidence(
    *,
    event: GameEventType,
    score: ScorePayload,
    event_context: EventContextPayload | None,
) -> bool:
    if event == GameEventType.PLAYER_SCORE:
        return True

    if score.player >= score.ai:
        return True

    if event != GameEventType.NEAR_SCORE or not event_context:
        return False

    return event_context.near_side == NearSide.AI_GOAL
