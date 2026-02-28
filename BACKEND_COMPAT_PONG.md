# Backend Compatibility Guide — Pong

Last updated: 2026-02-28

This guide captures the current Pong backend compatibility contract and the minimum
interface required so teammates can safely add AI opponent and TTS taunt assets
without breaking existing behavior.

## 1) Current status (ground truth)

- Pong rendering and simulation in frontend currently run in a local, client-side
  loop (`frontend/src/features/pong/game/gameLoop.ts` + canvas renderer).
- Server-authoritative Pong simulation is **not** present in this repo.
- Python backend already provides AI opponent event processing + difficulty control at
  `backend/opponent/*`.
- Python backend already provides `speech.audio_base64` in the `opponent_update`
  payload (mp3 bytes encoded as base64).
- Frontend has a reusable opponent websocket hook (`frontend/src/hooks/useAIOpponent.ts`)
  and taunt bubble component (`frontend/src/features/pong/components/TauntBubble.tsx`),
  but Pong page is not yet wired to them.

## 2) Backend compatibility contract

### 2.1 Endpoint map (already implemented)

Run the Python app on port 8000 (`backend/app.py`).

| Feature | Transport | Path | Purpose |
|---|---|---|---|
| BCI status control | HTTP | `POST /bci/start` | Start backend stream |
| BCI stop | HTTP | `POST /bci/stop` | Stop backend stream |
| Stress/cognitive stream | WS | `/bci/ccsignals/ws` | Feed stress/focus/frustration/alertness |
| MI control/prediction | WS | `/mi/ws` | MI prediction for input/intent |
| AI opponent events | WS | `/opponent/ws` | Deterministic taunt/difficulty updates |

### 2.2 AI opponent websocket payload (authoritative)

All objects are modeled in `backend/opponent/models.py` and validated server-side.

#### Client -> Server (`game_event`)

```ts
{
  "type": "game_event",
  "event_id": "evt-1",
  "game_mode": "pong",
  "event": "player_score" | "ai_score" | "near_score",
  "score": { "player": 3, "ai": 2 },
  "current_difficulty": 0.62,
  "event_context": { "near_side": "ai_goal" | "player_goal", "proximity": 0.2 },
  "timestamp_ms": 1730000000000
}
```

#### Server -> Client (`opponent_update`)

```ts
{
  "type": "opponent_update",
  "event_id": "evt-1",
  "taunt_text": "You're sweating already.",
  "difficulty": {
    "previous": 0.62,
    "model_target": 0.82,
    "final": 0.70
  },
  "speech": {
    "mime_type": "audio/mpeg",
    "audio_base64": "base64-encoded-mp3"
  },
  "metrics": {
    "stress": 0.64,
    "frustration": 0.58,
    "focus": 0.42,
    "alertness": 0.73
  },
  "meta": {
    "provider": "responses_speech" | "rule_based",
    "latency_ms": 420,
    "metrics_age_ms": 105
  },
  "timestamp_ms": 1730000000500
}
```

#### Recoverable error message

```ts
{
  "type": "error",
  "code": "INVALID_EVENT" | "OPENAI_ERROR" | "METRICS_UNAVAILABLE" | "RATE_LIMIT",
  "message": "human-readable message",
  "recoverable": true
}
```

#### Config/env contract

Important env vars in `backend/shared/config/app_config.py`:

- `OPENAI_API_KEY`
- `OPPONENT_TEXT_MODEL` (default `gpt-4.1-mini`)
- `OPPONENT_TTS_MODEL` (default `gpt-4o-mini-tts`)
- `OPPONENT_TTS_VOICE` (default `alloy`)
- `OPPONENT_MAX_TAUNT_CHARS` (default `80`)
- `OPPONENT_MIN_TAUNT_INTERVAL_MS` (default `1800`)
- `OPPONENT_TIMEOUT_MS` (default `5000`)

#### Backend tests that must continue to pass

- `backend/tests/opponent/test_routes_ws.py`  
  Valid event flow, invalid payload handling, rate limit behavior, fallback behavior.
- `backend/tests/opponent/test_difficulty_service.py`  
  Stress-first prior, near-goal boost, control smoothing bounds.
- `backend/tests/opponent/test_prompt_builder.py`
  Prompt includes event/metric context and guardrails.

## 3) Frontend compatibility status

Existing files:

- `frontend/src/hooks/useAIOpponent.ts`  
  Opens `API_ENDPOINTS.OPPONENT_WS` (`/opponent/ws`), decodes base64 audio payload to
  a blob URL (`audio/mpeg`), exposes `sendGameEvent` and `playLatestAudio`.
- `frontend/src/features/pong/components/TauntBubble.tsx`  
  Displays text for a fixed duration and supports fade at 80%.
- `frontend/src/config/api.ts`  
  Exposes `API_ENDPOINTS.OPPONENT_WS`.

Missing (Pong-specific):

- `frontend/src/pages/PongPage.tsx` currently does not use `useAIOpponent`
  and does not emit/consume opponent updates.
- No game-state websocket for Pong itself is implemented yet.

## 4) Required Pong taunt transport + UI sync contract

When teammates commit taunt TTS support, preserve this rule:

- **`speech.audio_base64` and `taunt_text` belong to the same `opponent_update`
  message and must trigger at the same presentation time.**

Suggested integration pattern (single source of truth):

1. In Pong page/controller, consume `latestUpdate` from `useAIOpponent`.
2. On each `opponent_update`, create a local `AiTauntEvent`:
   - `text = update.taunt_text`
   - `startedAt = performance.now()` (or adjusted using clock skew estimate if available)
   - `durationMs = 3000` (or contract-configured value)
   - `speechUrl = URL.createObjectURL(mp3Blob)` from `speech.audio_base64`
3. Render `TauntBubble` from that local event and call `Audio.play()` with the same
   `startedAt` seed.
4. If `speech.audio_base64` is empty, still display text bubble and log fallback.
5. If `audio_base64` is non-empty, decode once per update and revoke blob URL on
   unmount/update.
6. Keep the event stream independent from render loop; only update React state for taunt
   visibility.

Minimal timing rule:

- Bubble active window: `Date.now() - startedAt < durationMs`
- Audio: call play at the exact render decision frame immediately after state update.
- This produces synchronous start on the same frame.

## 5) Current alignment with documented systems

- `EVENT_CONTRACTS.md` already documents `/opponent/ws` and `opponent_update`.
- `AI_OPPONENT_DIFFICULTY.md` already documents formula + smoothing + fallback semantics.
- `GPT_TAUNTS.md` already documents OpenAI flow, rate limit, and text+audio outputs.
- No additional compatibility path is needed from `server/` for Pong at this stage
  because the implemented opponent stack already lives in Python.

## 6) Integration checklist for teammates

- Do not rename or flatten opponent payload fields.
- Keep `opponent_update` schema backward compatible:
  - `speech` object must always exist.
  - keep `mime_type` as `"audio/mpeg"` for mp3.
  - keep `event`, `score`, `current_difficulty`, `metrics`, and `meta` semantics.
- Add/adjust taunt/audio features by extending only the frontend Pong wiring and optionally
  backend prompt/svc internals.
- Add regression checks for:
  - `opponent_update` -> bubble/audio visible together in same tick.
  - empty/invalid audio payload -> no crash, text-only fallback.
  - rate-limit branch sends updated difficulty even when taunt is skipped.
- If a future dedicated server-authored Pong socket layer is added, expose a separate
  Pong namespace and do not alter `/opponent/ws`.
