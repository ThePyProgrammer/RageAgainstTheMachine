"""WebSocket routes for AI opponent personality updates."""

from __future__ import annotations

import re
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from opponent.models import (
    MetaPayload,
    MetricsPayload,
    OpponentError,
    OpponentGameEvent,
    OpponentUpdate,
    SpeechPayload,
    DifficultyPayload,
    GameEventType,
    InputMode,
)
from opponent.services.difficulty_service import (
    MetricsSnapshot,
    apply_difficulty_control,
    clamp01,
    compute_prior_difficulty,
)
from opponent.services.fallback_service import RuleBasedFallbackService
from opponent.services.openai_service import OpenAIOpponentService
from opponent.services.prompt_builder import (
    STYLE_REFERENCE_EXAMPLES,
    build_system_prompt,
    build_user_prompt,
)
from opponent.services.synthetic_metrics_service import SyntheticMetricsService
from opponent.services.session_service import OpponentSessionService
from shared.config.app_config import (
    OPENAI_API_KEY,
    OPPONENT_MAX_TAUNT_CHARS,
    OPPONENT_MIN_TAUNT_INTERVAL_MS,
    OPPONENT_TEXT_MODEL,
    OPPONENT_TEXT_TEMPERATURE,
    OPPONENT_TIMEOUT_MS,
    OPPONENT_TTS_INSTRUCTIONS,
    OPPONENT_TTS_MODEL,
    OPPONENT_TTS_SPEED,
    OPPONENT_TTS_VOICE,
)
from shared.config.logging import get_logger

try:
    from eeg.services.stream_service import get_active_streamer
except Exception:  # pragma: no cover - allows standalone testing without EEG deps
    def get_active_streamer():
        raise RuntimeError("EEG stream service unavailable in current environment")


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
    text_temperature=OPPONENT_TEXT_TEMPERATURE,
    tts_model=OPPONENT_TTS_MODEL,
    tts_voice=OPPONENT_TTS_VOICE,
    tts_instructions=OPPONENT_TTS_INSTRUCTIONS,
    tts_speed=OPPONENT_TTS_SPEED,
    timeout_ms=OPPONENT_TIMEOUT_MS,
)
fallback_service = RuleBasedFallbackService()
session_service = OpponentSessionService()


@router.websocket("/ws")
async def opponent_websocket(websocket: WebSocket):
    """Bidirectional websocket for opponent event processing."""
    await websocket.accept()
    session = session_service.create_session()
    synthetic_metrics_service = SyntheticMetricsService()

    try:
        while True:
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

            # Measure server processing latency from parsed event to emitted update.
            start_mono = time.monotonic()

            if _should_use_synthetic_metrics(event):
                metrics = synthetic_metrics_service.next_metrics(
                    event=event.event,
                    score=event.score,
                    event_context=event.event_context,
                )
                metrics_age_ms = 0
            else:
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
            generation_debug = {
                "path": "skipped_non_score_event",
                "text_ms": 0,
                "tts_ms": 0,
                "reason": "",
            }
            should_generate_taunt = event.event in (GameEventType.PLAYER_SCORE, GameEventType.AI_SCORE)

            if should_generate_taunt:
                if session.can_emit_taunt(OPPONENT_MIN_TAUNT_INTERVAL_MS):
                    (
                        provider,
                        taunt_text,
                        model_target,
                        audio_base64,
                        generation_debug,
                    ) = await _generate_response_bundle(
                        websocket=websocket,
                        event=event,
                        metrics=metrics,
                        prior=prior,
                    )
                    if taunt_text:
                        session.mark_taunt_emitted()
                else:
                    model_target = prior
                    generation_debug = {
                        "path": "rate_limited",
                        "text_ms": 0,
                        "tts_ms": 0,
                        "reason": "min_taunt_interval",
                    }
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
                event=event.event,
                score=event.score,
                event_context=event.event_context,
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

            processing_latency_ms = max(0, int((time.monotonic() - start_mono) * 1000.0))
            logger.info(
                (
                    "opponent_event event_id=%s event=%s input_mode=%s score=%d-%d provider=%s "
                    "latency_ms=%d metrics_age_ms=%d text_ms=%s tts_ms=%s taunt_chars=%d audio_bytes=%d path=%s reason=%s"
                ),
                event.event_id,
                event.event.value,
                event.input_mode.value if event.input_mode else "unspecified",
                event.score.player,
                event.score.ai,
                provider,
                processing_latency_ms,
                metrics_age_ms,
                generation_debug.get("text_ms", 0),
                generation_debug.get("tts_ms", 0),
                len(taunt_text),
                _audio_bytes_estimate(audio_base64),
                generation_debug.get("path", ""),
                generation_debug.get("reason", ""),
            )

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
) -> tuple[str, str, float, str, dict[str, int | str]]:
    """Generate taunt/difficulty/audio bundle, falling back when needed."""
    provider = "rule_based"
    taunt_text = fallback_service.generate_taunt(event=event, metrics=metrics)
    model_target = prior
    audio_base64 = ""
    debug = {
        "path": "rule_based",
        "text_ms": 0,
        "tts_ms": 0,
        "reason": "",
    }

    if not openai_service.is_available():
        debug["reason"] = "openai_unavailable"
        return provider, taunt_text, model_target, audio_base64, debug

    system_prompt = build_system_prompt(max_taunt_chars=OPPONENT_MAX_TAUNT_CHARS)
    user_prompt = build_user_prompt(event=event, metrics=metrics, prior_difficulty=prior)

    try:
        text_start = time.monotonic()
        llm_output = openai_service.generate_structured_output(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_taunt_chars=OPPONENT_MAX_TAUNT_CHARS,
        )
        debug["text_ms"] = max(0, int((time.monotonic() - text_start) * 1000.0))
        taunt_text = _clean_taunt_text(
            llm_output.taunt_text,
            max_chars=OPPONENT_MAX_TAUNT_CHARS,
        )
        if _looks_canned(taunt_text):
            taunt_text = _clean_taunt_text(
                fallback_service.generate_taunt(event=event, metrics=metrics),
                max_chars=OPPONENT_MAX_TAUNT_CHARS,
            )
        model_target = llm_output.difficulty_target
        provider = "responses_speech"
        debug["path"] = "responses_speech"
    except Exception as exc:
        logger.error("OpenAI structured generation failed: %s", exc, exc_info=True)
        debug["reason"] = "text_generation_error"
        await websocket.send_json(
            OpponentError(
                code="OPENAI_ERROR",
                message="Text generation failed; fallback mode active.",
                recoverable=True,
            ).model_dump()
        )
        return provider, taunt_text, model_target, audio_base64, debug

    try:
        tts_start = time.monotonic()
        audio_base64 = openai_service.generate_speech_base64(
            taunt_text,
            instructions=_build_dynamic_tts_instructions(metrics=metrics),
        )
        debug["tts_ms"] = max(0, int((time.monotonic() - tts_start) * 1000.0))
    except Exception as exc:
        logger.error("OpenAI speech generation failed: %s", exc, exc_info=True)
        debug["reason"] = "speech_generation_error"
        await websocket.send_json(
            OpponentError(
                code="OPENAI_ERROR",
                message="Speech generation failed; returning text without audio.",
                recoverable=True,
            ).model_dump()
        )

    return provider, taunt_text, model_target, audio_base64, debug


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


def _should_use_synthetic_metrics(event: OpponentGameEvent) -> bool:
    return event.input_mode in (
        InputMode.KEYBOARD_PADDLE,
        InputMode.KEYBOARD_BALL,
    )


def _audio_bytes_estimate(audio_base64: str) -> int:
    if not audio_base64:
        return 0
    return int((len(audio_base64) * 3) / 4)


def _build_dynamic_tts_instructions(metrics: MetricsSnapshot) -> str:
    dominant, value = _dominant_metric(metrics)
    if dominant == "stress":
        return "Lean into sharp, punchy delivery with a knowing grin in the voice."
    if dominant == "frustration":
        return "Add playful provocation — animated, almost gleeful needling."
    if dominant == "focus":
        return "Crisp and rapid, like a commentator hyping a clutch play."
    return "Bright and bouncy with teasing confidence."


def _dominant_metric(metrics: MetricsSnapshot) -> tuple[str, float]:
    values = {
        "stress": metrics.stress,
        "frustration": metrics.frustration,
        "focus": metrics.focus,
        "alertness": metrics.alertness,
    }
    key = max(values, key=values.get)
    return key, values[key]


def _clean_taunt_text(text: str, max_chars: int) -> str:
    """Normalize whitespace, strip leaked numbers/terms, trim to length."""
    normalized = " ".join(text.split())
    if not normalized:
        return "I'll let you figure that one out."

    # Strip leaked numeric values and percentages.
    normalized = re.sub(r"\b\d+(?:\.\d+)?%?\b", "", normalized)
    normalized = normalized.replace("%", "")
    # Replace leaked internal terminology.
    normalized = re.sub(
        r"\b(difficulty|difficulty_target|target|adjustment)\b",
        "pressure",
        normalized,
        flags=re.IGNORECASE,
    )
    # Collapse repeated punctuation and whitespace.
    normalized = re.sub(r"([.!?]){2,}", r"\1", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()

    if not normalized:
        return "I'll let you figure that one out."
    return _trim_taunt(normalized, max_chars=max_chars)


def _looks_canned(text: str) -> bool:
    lowered = text.lower()
    banned_fragments = (
        "keep up",
        "try harder",
        "ready for",
        "nice shot",
        "can you handle",
        "keep watching",
        "let's see if you can",
        "curveball",
        "powerup",
    )
    if any(fragment in lowered for fragment in banned_fragments):
        return True
    # Reject taunts that copy a style example too closely (5-word overlap).
    words = lowered.split()
    if len(words) >= 5:
        for _, example in STYLE_REFERENCE_EXAMPLES:
            example_lower = example.lower()
            for i in range(len(words) - 4):
                if " ".join(words[i : i + 5]) in example_lower:
                    return True
    return False


def _trim_taunt(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned

    head = cleaned[: max_chars + 1]
    # Prefer ending on punctuation near the limit.
    for punct in (".", "!", "?"):
        idx = head.rfind(punct)
        if idx >= max_chars - 24:
            return head[: idx + 1].strip()

    # Otherwise, cut on the last full word.
    cut = head.rfind(" ")
    if cut > 0:
        return head[:cut].rstrip(" ,.;:!?")

    return head[:max_chars].rstrip(" ,.;:!?")
