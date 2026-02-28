from opponent.models import (
    EventContextPayload,
    GameEventType,
    GameMode,
    NearSide,
    OpponentGameEvent,
    ScorePayload,
)
from opponent.services.difficulty_service import MetricsSnapshot
from opponent.services.prompt_builder import build_system_prompt, build_user_prompt


def test_system_prompt_contains_constraints() -> None:
    prompt = build_system_prompt(max_taunt_chars=80)
    assert "playful" in prompt.lower()
    assert "no harassment" in prompt.lower()
    assert "under 80 characters" in prompt.lower()
    assert "default to gradual pressure increases" in prompt.lower()


def test_user_prompt_contains_event_and_metrics() -> None:
    event = OpponentGameEvent(
        type="game_event",
        event_id="evt-1",
        game_mode=GameMode.PONG,
        event=GameEventType.NEAR_SCORE,
        score=ScorePayload(player=10, ai=2),
        current_difficulty=0.64,
        event_context=EventContextPayload(near_side=NearSide.PLAYER_GOAL, proximity=0.2),
        timestamp_ms=1730000000000,
    )
    metrics = MetricsSnapshot(stress=0.64, frustration=0.58, focus=0.42, alertness=0.73)

    prompt = build_user_prompt(event=event, metrics=metrics, prior_difficulty=0.82)
    assert "event type: near_score" in prompt.lower()
    assert "standout signal" in prompt.lower()
    assert "prior suggested difficulty: 0.820" in prompt.lower()
