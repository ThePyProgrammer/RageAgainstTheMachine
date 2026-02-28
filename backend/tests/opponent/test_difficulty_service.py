from opponent.models import EventContextPayload, GameEventType, NearSide, ScorePayload
from opponent.services.difficulty_service import (
    MetricsSnapshot,
    apply_difficulty_control,
    compute_prior_difficulty,
)


def test_prior_increases_with_stress() -> None:
    score = ScorePayload(player=5, ai=5)
    low_stress = MetricsSnapshot(stress=0.1, frustration=0.2, focus=0.6, alertness=0.7)
    high_stress = MetricsSnapshot(stress=0.9, frustration=0.2, focus=0.6, alertness=0.7)

    prior_low = compute_prior_difficulty(
        event=GameEventType.NEAR_SCORE,
        score=score,
        metrics=low_stress,
        event_context=EventContextPayload(near_side=NearSide.AI_GOAL, proximity=0.4),
    )
    prior_high = compute_prior_difficulty(
        event=GameEventType.NEAR_SCORE,
        score=score,
        metrics=high_stress,
        event_context=EventContextPayload(near_side=NearSide.AI_GOAL, proximity=0.4),
    )

    assert prior_high > prior_low


def test_near_player_goal_event_boost() -> None:
    score = ScorePayload(player=4, ai=4)
    metrics = MetricsSnapshot(stress=0.5, frustration=0.5, focus=0.5, alertness=0.5)

    prior_near_player = compute_prior_difficulty(
        event=GameEventType.NEAR_SCORE,
        score=score,
        metrics=metrics,
        event_context=EventContextPayload(near_side=NearSide.PLAYER_GOAL, proximity=0.2),
    )
    prior_near_ai = compute_prior_difficulty(
        event=GameEventType.NEAR_SCORE,
        score=score,
        metrics=metrics,
        event_context=EventContextPayload(near_side=NearSide.AI_GOAL, proximity=0.2),
    )

    assert prior_near_player > prior_near_ai


def test_smoothing_limits_per_event_jump() -> None:
    final_value = apply_difficulty_control(
        previous=0.40,
        prior=0.90,
        model_target=1.0,
        event=GameEventType.AI_SCORE,
        score=ScorePayload(player=1, ai=4),
        event_context=None,
    )
    assert abs(final_value - 0.40) <= 0.050001


def test_upward_cap_expands_with_evidence() -> None:
    final_value = apply_difficulty_control(
        previous=0.40,
        prior=0.90,
        model_target=1.0,
        event=GameEventType.PLAYER_SCORE,
        score=ScorePayload(player=4, ai=2),
        event_context=None,
    )
    assert abs(final_value - 0.40) <= 0.080001
    assert final_value >= 0.48


def test_downward_cap_allows_faster_relief() -> None:
    final_value = apply_difficulty_control(
        previous=0.70,
        prior=0.20,
        model_target=0.0,
        event=GameEventType.PLAYER_SCORE,
        score=ScorePayload(player=5, ai=4),
        event_context=EventContextPayload(near_side=NearSide.AI_GOAL, proximity=0.3),
    )
    assert abs(final_value - 0.70) <= 0.100001


def test_big_ai_lead_forces_difficulty_down() -> None:
    final_value = apply_difficulty_control(
        previous=0.72,
        prior=0.86,
        model_target=0.92,
        event=GameEventType.AI_SCORE,
        score=ScorePayload(player=1, ai=6),
        event_context=None,
    )
    assert final_value < 0.72
