"""OpenAI-backed taunt and speech generation."""

from __future__ import annotations

import base64
import json
from typing import Any

from opponent.models import OpponentLLMOutput

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


class OpenAIOpponentService:
    """Calls OpenAI Responses for text and Audio Speech for MP3 synthesis."""

    def __init__(
        self,
        api_key: str | None,
        text_model: str,
        tts_model: str,
        tts_voice: str,
        timeout_ms: int,
    ) -> None:
        self._text_model = text_model
        self._tts_model = tts_model
        self._tts_voice = tts_voice
        self._timeout_seconds = max(timeout_ms, 1000) / 1000.0
        self._client = None

        if api_key and OpenAI is not None:
            self._client = OpenAI(api_key=api_key)

    def is_available(self) -> bool:
        return self._client is not None

    def generate_structured_output(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_taunt_chars: int,
    ) -> OpponentLLMOutput:
        if self._client is None:
            raise RuntimeError("OpenAI client is unavailable")

        schema = {
            "type": "object",
            "properties": {
                "taunt_text": {
                    "type": "string",
                    "maxLength": max_taunt_chars,
                },
                "difficulty_target": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["taunt_text", "difficulty_target"],
            "additionalProperties": False,
        }

        client = self._client
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            client = with_options(timeout=self._timeout_seconds)
        response = client.responses.create(
            model=self._text_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "opponent_output",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        output_text = self._extract_output_text(response)
        if not output_text:
            raise RuntimeError("Responses API returned no output text")

        parsed = json.loads(output_text)
        return OpponentLLMOutput.model_validate(parsed)

    def generate_speech_base64(self, text: str) -> str:
        if self._client is None:
            raise RuntimeError("OpenAI client is unavailable")
        if not text.strip():
            return ""

        client = self._client
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            client = with_options(timeout=self._timeout_seconds)
        speech_response = client.audio.speech.create(
            model=self._tts_model,
            voice=self._tts_voice,
            input=text,
            response_format="mp3",
        )
        audio_bytes = self._extract_audio_bytes(speech_response)
        if not audio_bytes:
            raise RuntimeError("Audio speech API returned empty audio")

        return base64.b64encode(audio_bytes).decode("ascii")

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        direct_text = getattr(response, "output_text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()

        payload = OpenAIOpponentService._to_dict(response)
        for output_item in payload.get("output", []):
            for content_item in output_item.get("content", []):
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        return ""

    @staticmethod
    def _extract_audio_bytes(response: Any) -> bytes:
        if isinstance(response, (bytes, bytearray)):
            return bytes(response)

        content = getattr(response, "content", None)
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)

        read_fn = getattr(response, "read", None)
        if callable(read_fn):
            data = read_fn()
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)

        iter_bytes_fn = getattr(response, "iter_bytes", None)
        if callable(iter_bytes_fn):
            return b"".join(iter_bytes_fn())

        return b""

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        return {}
