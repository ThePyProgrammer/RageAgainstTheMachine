"""WebSocket routes for AI opponent personality updates."""

from __future__ import annotations

import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from eeg.services.stream_service import get_active_streamer
from opponent.models import (
    MetaPayload,
    MetricsPayload,
    OpponentError,
    OpponentGameEvent,
    OpponentUpdate,
    SpeechPayload,
    DifficultyPayload,
)
from opponent.services.difficulty_service import (
    MetricsSnapshot,
    apply_difficulty_control,
    clamp01,
    compute_prior_difficulty,
)
from opponent.services.fallback_service import RuleBasedFallbackService
from opponent.services.openai_service import OpenAIOpponentService
from opponent.services.prompt_builder import build_system_prompt, build_user_prompt
from opponent.services.session_service import OpponentSessionService
from shared.config.app_config import (
    OPENAI_API_KEY,
    OPPONENT_MAX_TAUNT_CHARS,
    OPPONENT_MIN_TAUNT_INTERVAL_MS,
    OPPONENT_TEXT_MODEL,
    OPPONENT_TIMEOUT_MS,
    OPPONENT_TTS_MODEL,
    OPPONENT_TTS_VOICE,
)
from shared.config.logging import get_logger

router = APIRouter(
    prefix="/opponent",
    tags=["opponent"],
    responses={404: {"description": "Not found"}},
)

logger = get_logger("opponent.routes")

DEFAULT_METRICS = MetricsSnapshot(
    stress=0.5,
    frustration=0.5,
    focus=0.5,
    alertness=0.5,
)

openai_service = OpenAIOpponentService(
    api_key=OPENAI_API_KEY,
    text_model=OPPONENT_TEXT_MODEL,
    tts_model=OPPONENT_TTS_MODEL,
    tts_voice=OPPONENT_TTS_VOICE,
    timeout_ms=OPPONENT_TIMEOUT_MS,
)
fallback_service = RuleBasedFallbackService()
session_service = OpponentSessionService()


@router.websocket("/ws")
async def opponent_websocket(websocket: WebSocket):
    """Bidirectional websocket for opponent event processing."""
    await websocket.accept()
    session = session_service.create_session()

    try:
        while True:
            start_mono = time.monotonic()

            try:
                incoming = await websocket.receive_json()
                event = OpponentGameEvent.model_validate(incoming)
            except ValidationError as exc:
                await websocket.send_json(
                    OpponentError(
                        code="INVALID_EVENT",
                        message=f"Invalid event payload: {exc.errors()}",
                        recoverable=True,
                    ).model_dump()
                )
                continue
            except Exception as exc:
                await websocket.send_json(
                    OpponentError(
                        code="INVALID_EVENT",
                        message=f"Unable to parse websocket payload: {exc}",
                        recoverable=True,
                    ).model_dump()
                )
                continue

            metrics, metrics_age_ms = _get_metrics_snapshot()
            if metrics is None:
                metrics = DEFAULT_METRICS
                metrics_age_ms = -1
                await websocket.send_json(
                    OpponentError(
                        code="METRICS_UNAVAILABLE",
                        message="Using fallback metrics because live command-centre data is unavailable.",
                        recoverable=True,
                    ).model_dump()
                )

            previous = clamp01(event.current_difficulty)
            session.current_difficulty = previous
            prior = compute_prior_difficulty(
                event=event.event,
                score=event.score,
                metrics=metrics,
                event_context=event.event_context,
            )
            model_target = prior
            taunt_text = ""
            audio_base64 = ""
            provider = "rule_based"

            if session.can_emit_taunt(OPPONENT_MIN_TAUNT_INTERVAL_MS):
                provider, taunt_text, model_target, audio_base64 = await _generate_response_bundle(
                    websocket=websocket,
                    event=event,
                    metrics=metrics,
                    prior=prior,
                )
                if taunt_text:
                    session.mark_taunt_emitted()
            else:
                model_target = prior
                await websocket.send_json(
                    OpponentError(
                        code="RATE_LIMIT",
                        message="Taunt generation rate-limited; difficulty update still applied.",
                        recoverable=True,
                    ).model_dump()
                )

            final_difficulty = apply_difficulty_control(
                previous=previous,
                prior=prior,
                model_target=clamp01(model_target),
            )
            session.current_difficulty = final_difficulty

            update = OpponentUpdate(
                event_id=event.event_id,
                taunt_text=taunt_text,
                difficulty=DifficultyPayload(
                    previous=previous,
                    model_target=clamp01(model_target),
                    final=final_difficulty,
                ),
                speech=SpeechPayload(audio_base64=audio_base64),
                metrics=MetricsPayload(
                    stress=metrics.stress,
                    frustration=metrics.frustration,
                    focus=metrics.focus,
                    alertness=metrics.alertness,
                ),
                meta=MetaPayload(
                    provider=provider,
                    latency_ms=max(0, int((time.monotonic() - start_mono) * 1000.0)),
                    metrics_age_ms=metrics_age_ms,
                ),
                timestamp_ms=int(time.time() * 1000),
            )
            await websocket.send_json(update.model_dump())

    except WebSocketDisconnect as exc:
        logger.info("Opponent websocket disconnected: code=%s", exc.code)
    except Exception:
        logger.exception("Unhandled opponent websocket error")
        try:
            await websocket.send_json(
                OpponentError(
                    code="OPENAI_ERROR",
                    message="Unexpected server error in opponent service.",
                    recoverable=True,
                ).model_dump()
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _generate_response_bundle(
    websocket: WebSocket,
    event: OpponentGameEvent,
    metrics: MetricsSnapshot,
    prior: float,
) -> tuple[str, str, float, str]:
    """Generate taunt/difficulty/audio bundle, falling back when needed."""
    provider = "rule_based"
    taunt_text = fallback_service.generate_taunt(event=event, metrics=metrics)
    model_target = prior
    audio_base64 = ""

    if not openai_service.is_available():
        return provider, taunt_text, model_target, audio_base64

    system_prompt = build_system_prompt(max_taunt_chars=OPPONENT_MAX_TAUNT_CHARS)
    user_prompt = build_user_prompt(event=event, metrics=metrics, prior_difficulty=prior)

    try:
        llm_output = openai_service.generate_structured_output(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_taunt_chars=OPPONENT_MAX_TAUNT_CHARS,
        )
        taunt_text = llm_output.taunt_text[:OPPONENT_MAX_TAUNT_CHARS]
        model_target = llm_output.difficulty_target
        provider = "responses_speech"
    except Exception as exc:
        logger.error("OpenAI structured generation failed: %s", exc, exc_info=True)
        await websocket.send_json(
            OpponentError(
                code="OPENAI_ERROR",
                message="Text generation failed; fallback mode active.",
                recoverable=True,
            ).model_dump()
        )
        return provider, taunt_text, model_target, audio_base64

    try:
        audio_base64 = openai_service.generate_speech_base64(taunt_text)
    except Exception as exc:
        logger.error("OpenAI speech generation failed: %s", exc, exc_info=True)
        await websocket.send_json(
            OpponentError(
                code="OPENAI_ERROR",
                message="Speech generation failed; returning text without audio.",
                recoverable=True,
            ).model_dump()
        )

    return provider, taunt_text, model_target, audio_base64


def _get_metrics_snapshot() -> tuple[MetricsSnapshot | None, int]:
    """Read latest backend-owned command-centre metrics from active streamer."""
    try:
        streamer = get_active_streamer()
        getter = getattr(streamer, "get_latest_cc_signal", None)
        if not callable(getter):
            return None, -1

        latest = getter()
        if not latest:
            return None, -1

        payload = latest.get("payload", {})
        signals = payload.get("signals", {})
        metrics = MetricsSnapshot(
            stress=_safe_signal_value(signals.get("stress"), DEFAULT_METRICS.stress),
            frustration=_safe_signal_value(signals.get("frustration"), DEFAULT_METRICS.frustration),
            focus=_safe_signal_value(signals.get("focus"), DEFAULT_METRICS.focus),
            alertness=_safe_signal_value(signals.get("alertness"), DEFAULT_METRICS.alertness),
        )

        emitted_monotonic = latest.get("monotonic")
        if isinstance(emitted_monotonic, (float, int)):
            age_ms = max(0, int((time.monotonic() - float(emitted_monotonic)) * 1000.0))
        else:
            age_ms = -1

        return metrics, age_ms
    except Exception:
        logger.exception("Failed reading command-centre metrics snapshot")
        return None, -1


def _safe_signal_value(raw_value: object, default: float) -> float:
    try:
        return clamp01(float(raw_value))
    except Exception:
        return default
