"""Single-session state for opponent websocket interactions."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class OpponentSessionState:
    current_difficulty: float = 0.5
    last_taunt_monotonic: float = 0.0

    def can_emit_taunt(self, min_interval_ms: int) -> bool:
        now = time.monotonic()
        if self.last_taunt_monotonic == 0.0:
            return True
        elapsed_ms = (now - self.last_taunt_monotonic) * 1000.0
        return elapsed_ms >= min_interval_ms

    def mark_taunt_emitted(self) -> None:
        self.last_taunt_monotonic = time.monotonic()


class OpponentSessionService:
    """Factory for per-connection opponent session state."""

    @staticmethod
    def create_session(initial_difficulty: float = 0.5) -> OpponentSessionState:
        return OpponentSessionState(current_difficulty=initial_difficulty)

