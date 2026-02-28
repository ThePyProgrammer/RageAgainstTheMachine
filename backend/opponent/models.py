"""Pydantic models for AI opponent websocket contracts."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GameMode(str, Enum):
    PONG = "pong"
    COMBAT = "combat"


class GameEventType(str, Enum):
    PLAYER_SCORE = "player_score"
    AI_SCORE = "ai_score"
    NEAR_SCORE = "near_score"


class NearSide(str, Enum):
    PLAYER_GOAL = "player_goal"
    AI_GOAL = "ai_goal"


class ScorePayload(BaseModel):
    player: int = Field(ge=0)
    ai: int = Field(ge=0)


class EventContextPayload(BaseModel):
    near_side: NearSide | None = None
    proximity: float | None = Field(default=None, ge=0.0, le=1.0)


class OpponentGameEvent(BaseModel):
    type: Literal["game_event"]
    event_id: str = Field(min_length=1)
    game_mode: GameMode
    event: GameEventType
    score: ScorePayload
    current_difficulty: float = Field(ge=0.0, le=1.0)
    event_context: EventContextPayload | None = None
    timestamp_ms: int = Field(ge=0)


class OpponentLLMOutput(BaseModel):
    taunt_text: str = Field(min_length=1)
    difficulty_target: float = Field(ge=0.0, le=1.0)

    @field_validator("taunt_text")
    @classmethod
    def normalize_taunt(cls, value: str) -> str:
        text = " ".join(value.split())
        if not text:
            raise ValueError("taunt_text cannot be empty")
        return text


class DifficultyPayload(BaseModel):
    previous: float = Field(ge=0.0, le=1.0)
    model_target: float = Field(ge=0.0, le=1.0)
    final: float = Field(ge=0.0, le=1.0)


class SpeechPayload(BaseModel):
    mime_type: Literal["audio/mpeg"] = "audio/mpeg"
    audio_base64: str = ""


class MetricsPayload(BaseModel):
    stress: float = Field(ge=0.0, le=1.0)
    frustration: float = Field(ge=0.0, le=1.0)
    focus: float = Field(ge=0.0, le=1.0)
    alertness: float = Field(ge=0.0, le=1.0)


class MetaPayload(BaseModel):
    provider: Literal["responses_speech", "rule_based"] = "rule_based"
    latency_ms: int = Field(ge=0)
    metrics_age_ms: int = Field(ge=-1)


class OpponentUpdate(BaseModel):
    type: Literal["opponent_update"] = "opponent_update"
    event_id: str
    taunt_text: str
    difficulty: DifficultyPayload
    speech: SpeechPayload
    metrics: MetricsPayload
    meta: MetaPayload
    timestamp_ms: int = Field(ge=0)


class OpponentError(BaseModel):
    type: Literal["error"] = "error"
    code: Literal[
        "INVALID_EVENT",
        "OPENAI_ERROR",
        "METRICS_UNAVAILABLE",
        "RATE_LIMIT",
    ]
    message: str
    recoverable: bool

