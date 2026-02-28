# GPT_TAUNTS.md

Generate contextual, friendly taunts and speech for game events.

## Implementation Location

- Route: `/backend/opponent/routes.py`
- Prompt builder: `/backend/opponent/services/prompt_builder.py`
- OpenAI integration: `/backend/opponent/services/openai_service.py`
- Rule fallback: `/backend/opponent/services/fallback_service.py`

## Trigger Events

Supported v1 events:
- `player_score`
- `ai_score`
- `near_score`

Each event is sent by frontend over `/opponent/ws` with score + current difficulty.

## Prompt Contract

System prompt constraints:
- Keep tone playful and competitive.
- No harassment, hate speech, slurs, or profanity.
- Keep taunt length within `OPPONENT_MAX_TAUNT_CHARS` (default 80).
- Weight stress/frustration more than score for difficulty suggestions.
- Output strict JSON only.

Expected model JSON:

```json
{
  "taunt_text": "string",
  "difficulty_target": 0.0
}
```

## OpenAI APIs Used

1. Responses API
- Purpose: generate structured taunt + `difficulty_target`
- Guard: strict JSON schema

2. Audio Speech API
- Purpose: synthesize taunt text into MP3
- Output encoded into base64 and returned inline to frontend

## Rate Limiting

- Controlled by `OPPONENT_MIN_TAUNT_INTERVAL_MS` (default 1800ms).
- If rate-limited, server still sends difficulty update but skips taunt/speech generation.

## Error Handling

Recoverable error messages may be emitted with codes:
- `INVALID_EVENT`
- `OPENAI_ERROR`
- `METRICS_UNAVAILABLE`
- `RATE_LIMIT`

When OpenAI fails, the backend falls back to rule-based taunts and deterministic difficulty.

## Frontend Consumption

Frontend hook: `/frontend/src/hooks/useAIOpponent.ts`

- Sends `game_event` packets.
- Receives `opponent_update` payloads.
- Decodes `speech.audio_base64` into playable MP3 Blob URL.
- Applies `difficulty.final` to active game logic.
