# AI_OPPONENT_DIFFICULTY.md

Adaptive AI opponent difficulty derived from game state + backend command-centre metrics.

## Implementation Location

- Backend route: `/backend/opponent/routes.py`
- Difficulty logic: `/backend/opponent/services/difficulty_service.py`
- Websocket endpoint: `/opponent/ws`

## Inputs

- Frontend event payload (`game_event`):
  - `event`: `player_score | ai_score | near_score`
  - `score.player`, `score.ai`
  - `current_difficulty` in `[0, 1]`
  - optional `input_mode`: `eeg | keyboard_paddle | keyboard_ball`
  - optional near-score context
- Backend metrics snapshot:
  - `stress`, `frustration`, `focus`, `alertness` in `[0, 1]`

## Metric Source Selection

- `input_mode = eeg` (or omitted): use live command-centre metrics from backend EEG streamer.
- `input_mode = keyboard_paddle | keyboard_ball`: use session-scoped synthetic metrics.
- Synthetic metrics are generated as a bounded random walk with event/score bias and always clamped to `[0, 1]`.

## Stress-Dominant Prior

The deterministic prior is computed as:

```text
lead = clamp(-1, 1, (player_score - ai_score) / 10)
prior = clamp01(
  0.55
  + 0.35 * stress
  + 0.10 * frustration
  + 0.08 * max(lead, 0)
  - 0.04 * max(-lead, 0)
  + event_boost
)
```

Event boost:
- `near_score` near player goal: `+0.03`
- `ai_score`: `+0.01`
- `player_score`: `-0.01`
- otherwise: `0`

## Model Target and Control Layer

The model proposes `difficulty_target` in `[0, 1]`.
Backend applies bounded control:

```text
model_clamped = clamp(prior - 0.20, prior + 0.20, model_target)
target = 0.70 * model_clamped + 0.30 * prior
max_up = +0.05 by default
max_up = +0.08 if gameplay evidence supports it
  evidence := player scored OR player is tied/ahead OR near-score at AI goal
final = previous + clamp(target - previous, -0.10, max_up)
final = clamp01(final)
```

This prevents sharp jumps and keeps difficulty smooth.

## Fallback Behavior

- If model output fails: use rule-based taunt and set `model_target = prior`.
- If speech synthesis fails: return text + difficulty with empty audio payload.
- If command-centre metrics are unavailable: use neutral fallback metrics and emit recoverable error.

## Output Contract

Server sends `opponent_update`:

```json
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
    "audio_base64": "..."
  },
  "metrics": {
    "stress": 0.64,
    "frustration": 0.58,
    "focus": 0.42,
    "alertness": 0.73
  },
  "meta": {
    "provider": "responses_speech",
    "latency_ms": 420,
    "metrics_age_ms": 105
  },
  "timestamp_ms": 1730000000500
}
```

## Validation and Tests

- Unit tests: prior formula, stress dominance, smoothing bounds.
- Websocket tests: valid event flow, invalid payload handling, fallback on model failure, rate limiting behavior.
