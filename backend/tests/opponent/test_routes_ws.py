import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

import opponent.routes as opponent_routes
from opponent.models import OpponentLLMOutput


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(opponent_routes.router)
    return app


def _valid_event_payload(event_id: str = "evt-1") -> dict:
    return {
        "type": "game_event",
        "event_id": event_id,
        "game_mode": "pong",
        "event": "player_score",
        "score": {"player": 3, "ai": 2},
        "current_difficulty": 0.62,
        "event_context": {"near_side": "ai_goal", "proximity": 0.2},
        "timestamp_ms": 1730000000000,
    }


def test_opponent_websocket_happy_path(monkeypatch) -> None:
    class FakeStreamer:
        @staticmethod
        def get_latest_cc_signal():
            return {
                "payload": {
                    "signals": {
                        "stress": 0.64,
                        "frustration": 0.58,
                        "focus": 0.42,
                        "alertness": 0.73,
                    }
                },
                "monotonic": time.monotonic(),
            }

    monkeypatch.setattr(opponent_routes, "get_active_streamer", lambda: FakeStreamer())
    monkeypatch.setattr(opponent_routes.openai_service, "is_available", lambda: True)
    monkeypatch.setattr(
        opponent_routes.openai_service,
        "generate_structured_output",
        lambda **_: OpponentLLMOutput(
            taunt_text="You're sweating already.",
            difficulty_target=0.82,
        ),
    )
    monkeypatch.setattr(
        opponent_routes.openai_service,
        "generate_speech_base64",
        lambda _: "ZmFrZS1tcDM=",
    )

    app = _build_app()
    with TestClient(app).websocket_connect("/opponent/ws") as ws:
        ws.send_json(_valid_event_payload())
        response = ws.receive_json()
        assert response["type"] == "opponent_update"
        assert response["difficulty"]["previous"] == 0.62
        assert 0 <= response["difficulty"]["final"] <= 1
        assert response["speech"]["audio_base64"] == "ZmFrZS1tcDM="
        assert response["meta"]["provider"] == "responses_speech"


def test_opponent_websocket_invalid_payload() -> None:
    app = _build_app()
    with TestClient(app).websocket_connect("/opponent/ws") as ws:
        ws.send_json({"type": "game_event", "event_id": "missing-fields"})
        response = ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "INVALID_EVENT"


def test_rate_limit_still_emits_difficulty_update(monkeypatch) -> None:
    class FakeStreamer:
        @staticmethod
        def get_latest_cc_signal():
            return {
                "payload": {
                    "signals": {
                        "stress": 0.50,
                        "frustration": 0.50,
                        "focus": 0.50,
                        "alertness": 0.50,
                    }
                },
                "monotonic": time.monotonic(),
            }

    monkeypatch.setattr(opponent_routes, "get_active_streamer", lambda: FakeStreamer())
    monkeypatch.setattr(opponent_routes.openai_service, "is_available", lambda: False)
    monkeypatch.setattr(opponent_routes, "OPPONENT_MIN_TAUNT_INTERVAL_MS", 999999)

    app = _build_app()
    with TestClient(app).websocket_connect("/opponent/ws") as ws:
        ws.send_json(_valid_event_payload(event_id="evt-a"))
        first_update = ws.receive_json()
        assert first_update["type"] == "opponent_update"
        assert first_update["taunt_text"] != ""

        ws.send_json(_valid_event_payload(event_id="evt-b"))
        rate_limit_error = ws.receive_json()
        assert rate_limit_error["type"] == "error"
        assert rate_limit_error["code"] == "RATE_LIMIT"
        second_update = ws.receive_json()
        assert second_update["type"] == "opponent_update"
        assert second_update["difficulty"]["final"] >= 0


def test_openai_failure_falls_back(monkeypatch) -> None:
    class FakeStreamer:
        @staticmethod
        def get_latest_cc_signal():
            return {
                "payload": {
                    "signals": {
                        "stress": 0.70,
                        "frustration": 0.62,
                        "focus": 0.45,
                        "alertness": 0.68,
                    }
                },
                "monotonic": time.monotonic(),
            }

    monkeypatch.setattr(opponent_routes, "get_active_streamer", lambda: FakeStreamer())
    monkeypatch.setattr(opponent_routes.openai_service, "is_available", lambda: True)

    def _raise_error(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        opponent_routes.openai_service,
        "generate_structured_output",
        _raise_error,
    )
    monkeypatch.setattr(
        opponent_routes.openai_service,
        "generate_speech_base64",
        lambda _: "",
    )

    app = _build_app()
    with TestClient(app).websocket_connect("/opponent/ws") as ws:
        ws.send_json(_valid_event_payload(event_id="evt-openai-fail"))
        error_msg = ws.receive_json()
        assert error_msg["type"] == "error"
        assert error_msg["code"] == "OPENAI_ERROR"
        update_msg = ws.receive_json()
        assert update_msg["type"] == "opponent_update"
        assert update_msg["meta"]["provider"] == "rule_based"
