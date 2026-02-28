# EVENT_CONTRACTS.md

All WebSocket events with TypeScript types and Zod schemas.

## AI Opponent WebSocket (Implemented in FastAPI)

Endpoint: `/opponent/ws`

### Client -> Server (`game_event`)

```json
{
  "type": "game_event",
  "event_id": "evt-1",
  "game_mode": "pong",
  "event": "player_score",
  "score": { "player": 3, "ai": 2 },
  "current_difficulty": 0.62,
  "event_context": { "near_side": "ai_goal", "proximity": 0.2 },
  "timestamp_ms": 1730000000000
}
```

### Server -> Client (`opponent_update`)

```json
{
  "type": "opponent_update",
  "event_id": "evt-1",
  "taunt_text": "You're sweating already.",
  "difficulty": { "previous": 0.62, "model_target": 0.82, "final": 0.70 },
  "speech": { "mime_type": "audio/mpeg", "audio_base64": "..." },
  "metrics": { "stress": 0.64, "frustration": 0.58, "focus": 0.42, "alertness": 0.73 },
  "meta": { "provider": "responses_speech", "latency_ms": 420, "metrics_age_ms": 105 },
  "timestamp_ms": 1730000000500
}
```

### Recoverable Error Message

```json
{
  "type": "error",
  "code": "INVALID_EVENT|OPENAI_ERROR|METRICS_UNAVAILABLE|RATE_LIMIT",
  "message": "human-readable error",
  "recoverable": true
}
```

### Notes

- Opponent events are validated server-side with Pydantic models in `backend/opponent/models.py`.
- Command-centre metrics are backend-authoritative and sourced from live EEG streamers.
- Difficulty updates still flow even when taunt/speech is rate-limited or OpenAI calls fail.

## Type Definitions

Location: `/packages/shared/src/events.ts`

```typescript
import { z } from 'zod';

// ============================================================================
// Client → Server Events
// ============================================================================

export const ConnectEventSchema = z.object({
  userId: z.string().uuid().optional(), // Resume session if provided
});
export type ConnectEvent = z.infer<typeof ConnectEventSchema>;

export const StartCalibrationEventSchema = z.object({
  // No payload
});
export type StartCalibrationEvent = z.infer<typeof StartCalibrationEventSchema>;

export const CalibrationSampleEventSchema = z.object({
  signals: z.object({
    up: z.number(),
    down: z.number(),
    left: z.number(),
    right: z.number(),
  }),
  timestamp: z.number(), // Client timestamp (ms since epoch)
});
export type CalibrationSampleEvent = z.infer<typeof CalibrationSampleEventSchema>;

export const InputCommandEventSchema = z.object({
  direction: z.enum(['LEFT', 'RIGHT', 'NEUTRAL']),
  timestamp: z.number(),
  rawActivation: z.number().optional(), // For debugging
});
export type InputCommandEvent = z.infer<typeof InputCommandEventSchema>;

export const StartGameEventSchema = z.object({
  // No payload (calibration assumed complete)
});
export type StartGameEvent = z.infer<typeof StartGameEventSchema>;

export const PauseGameEventSchema = z.object({
  // No payload
});
export type PauseGameEvent = z.infer<typeof PauseGameEventSchema>;

// ============================================================================
// Server → Client Events
// ============================================================================

export const SessionCreatedEventSchema = z.object({
  sessionId: z.string().uuid(),
  calibrationRequired: z.boolean(),
  calibrationParams: z.object({
    baselineMean: z.number(),
    baselineStd: z.number(),
    thresholdLeft: z.number(),
    thresholdRight: z.number(),
    thresholdEnter: z.number(),
    thresholdExit: z.number(),
    quality: z.number(),
  }).optional(), // Present if resuming session
});
export type SessionCreatedEvent = z.infer<typeof SessionCreatedEventSchema>;

export const CalibrationStepEventSchema = z.object({
  step: z.enum(['baseline', 'left', 'right']),
  trial: z.number(), // 1-indexed (1, 2, 3)
  instruction: z.string(), // Human-readable instruction
  durationMs: z.number(), // How long to hold this step
});
export type CalibrationStepEvent = z.infer<typeof CalibrationStepEventSchema>;

export const CalibrationCompleteEventSchema = z.object({
  params: z.object({
    baselineMean: z.number(),
    baselineStd: z.number(),
    thresholdLeft: z.number(),
    thresholdRight: z.number(),
    thresholdEnter: z.number(),
    thresholdExit: z.number(),
    quality: z.number(), // Separation score (μ_left - μ_right) / pooledStd
  }),
  quality: z.number(),
  status: z.enum(['success', 'retry_suggested', 'failed']),
  message: z.string(),
});
export type CalibrationCompleteEvent = z.infer<typeof CalibrationCompleteEventSchema>;

export const GameStateEventSchema = z.object({
  timestamp: z.number(),
  ball: z.object({
    x: z.number(),
    y: z.number(),
    vx: z.number(),
    vy: z.number(),
  }),
  playerPaddle: z.object({
    y: z.number(), // x is fixed at left side
    vy: z.number(),
  }),
  aiPaddle: z.object({
    y: z.number(), // x is fixed at right side
    vy: z.number(),
  }),
  score: z.object({
    player: z.number(),
    ai: z.number(),
  }),
  stressLevel: z.number(), // 0.0 - 1.0
});
export type GameStateEvent = z.infer<typeof GameStateEventSchema>;

export const TauntMessageEventSchema = z.object({
  text: z.string(),
  durationMs: z.number(), // How long to display
  trigger: z.enum([
    'player_miss',
    'player_score',
    'ai_score',
    'long_rally',
    'stress_spike',
    'calm_streak',
  ]),
});
export type TauntMessageEvent = z.infer<typeof TauntMessageEventSchema>;

export const GameOverEventSchema = z.object({
  winner: z.enum(['player', 'ai']),
  finalScore: z.object({
    player: z.number(),
    ai: z.number(),
  }),
});
export type GameOverEvent = z.infer<typeof GameOverEventSchema>;

export const ErrorEventSchema = z.object({
  code: z.string(), // e.g., "CALIBRATION_FAILED", "THONK_TIMEOUT"
  message: z.string(),
  recoverable: z.boolean(),
});
export type ErrorEvent = z.infer<typeof ErrorEventSchema>;

// ============================================================================
// Event Names (string constants)
// ============================================================================

export const ClientEvents = {
  CONNECT: 'connect',
  START_CALIBRATION: 'start_calibration',
  CALIBRATION_SAMPLE: 'calibration_sample',
  INPUT_COMMAND: 'input_command',
  START_GAME: 'start_game',
  PAUSE_GAME: 'pause_game',
  DISCONNECT: 'disconnect',
} as const;

export const ServerEvents = {
  SESSION_CREATED: 'session_created',
  CALIBRATION_STEP: 'calibration_step',
  CALIBRATION_COMPLETE: 'calibration_complete',
  GAME_STATE: 'game_state',
  TAUNT_MESSAGE: 'taunt_message',
  GAME_OVER: 'game_over',
  ERROR: 'error',
} as const;

// ============================================================================
// Helper Types for Socket.IO
// ============================================================================

export interface ServerToClientEvents {
  [ServerEvents.SESSION_CREATED]: (data: SessionCreatedEvent) => void;
  [ServerEvents.CALIBRATION_STEP]: (data: CalibrationStepEvent) => void;
  [ServerEvents.CALIBRATION_COMPLETE]: (data: CalibrationCompleteEvent) => void;
  [ServerEvents.GAME_STATE]: (data: GameStateEvent) => void;
  [ServerEvents.TAUNT_MESSAGE]: (data: TauntMessageEvent) => void;
  [ServerEvents.GAME_OVER]: (data: GameOverEvent) => void;
  [ServerEvents.ERROR]: (data: ErrorEvent) => void;
}

export interface ClientToServerEvents {
  [ClientEvents.START_CALIBRATION]: (data: StartCalibrationEvent) => void;
  [ClientEvents.CALIBRATION_SAMPLE]: (data: CalibrationSampleEvent) => void;
  [ClientEvents.INPUT_COMMAND]: (data: InputCommandEvent) => void;
  [ClientEvents.START_GAME]: (data: StartGameEvent) => void;
  [ClientEvents.PAUSE_GAME]: (data: PauseGameEvent) => void;
}
```

## Event Flow Diagram

```
CLIENT                                SERVER
  │                                      │
  ├─── connect ─────────────────────────>│
  │<─────────────────── session_created ─┤ {sessionId, calibrationRequired: true}
  │                                      │
  ├─── start_calibration ───────────────>│
  │<────────────────── calibration_step ─┤ {step: "baseline", trial: 1, ...}
  ├─── calibration_sample ──────────────>│ (repeat 2-4s)
  │<────────────────── calibration_step ─┤ {step: "left", trial: 1, ...}
  ├─── calibration_sample ──────────────>│ (repeat 1.5-2s)
  │<────────────────── calibration_step ─┤ {step: "left", trial: 2, ...}
  ├─── calibration_sample ──────────────>│ (repeat 1.5-2s)
  │<────────────────── calibration_step ─┤ {step: "right", trial: 1, ...}
  ├─── calibration_sample ──────────────>│ (repeat 1.5-2s)
  │<────────────────── calibration_step ─┤ {step: "right", trial: 2, ...}
  ├─── calibration_sample ──────────────>│ (repeat 1.5-2s)
  │<─────────────── calibration_complete ┤ {params, quality, status: "success"}
  │                                      │
  ├─── start_game ──────────────────────>│
  │                                      │
  │ GAME LOOP (60 Hz both directions)   │
  ├─── input_command ───────────────────>│ {direction: "LEFT", timestamp}
  │<──────────────────────── game_state ─┤ {ball, paddles, score, stress}
  ├─── input_command ───────────────────>│ {direction: "NEUTRAL", timestamp}
  │<──────────────────────── game_state ─┤
  │<─────────────────── taunt_message ───┤ {text: "Is that all you got?!", ...}
  │ ...                                  │
  │<──────────────────────── game_over ──┤ {winner: "ai", finalScore}
  │                                      │
  ├─── disconnect ──────────────────────>│
```

## Rate Limits

| Event | Client → Server | Server → Client |
|-------|----------------|-----------------|
| `calibration_sample` | Unlimited during calibration | N/A |
| `input_command` | 60 Hz (every ~16ms) | N/A |
| `game_state` | N/A | 60 Hz (every ~16ms) |
| `taunt_message` | N/A | Max 1 per 3 seconds |
| `start_calibration` | Max 1 per 10 seconds | N/A |
| `start_game` | Max 1 per 2 seconds | N/A |

## Error Codes

| Code | Meaning | Recoverable |
|------|---------|-------------|
| `CALIBRATION_FAILED` | Quality score < 1.0 after 3 attempts | Yes (retry) |
| `THONK_TIMEOUT` | No Thonk signals for >2 seconds | Yes (reconnect) |
| `INVALID_SESSION` | Session ID not found | No (restart) |
| `GAME_NOT_STARTED` | Received input before game started | Yes (ignore) |
| `GPT_API_ERROR` | OpenAI API failure | Yes (skip taunt) |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Yes (throttle) |

## Validation

All events must be validated with Zod schemas before processing:

```typescript
// Server-side example
socket.on(ClientEvents.INPUT_COMMAND, (data: unknown) => {
  const parsed = InputCommandEventSchema.safeParse(data);
  if (!parsed.success) {
    socket.emit(ServerEvents.ERROR, {
      code: 'VALIDATION_ERROR',
      message: parsed.error.message,
      recoverable: true,
    });
    return;
  }
  // Process parsed.data
});
```

## Testing Checklist

- [ ] All schemas compile with TypeScript strict mode
- [ ] All events round-trip through Zod parse/serialize
- [ ] Rate limits enforced on server
- [ ] Invalid payloads trigger ERROR event
- [ ] Missing required fields rejected
- [ ] Extra fields ignored (forward compatibility)
- [ ] Timestamps within ±5s of server time (clock drift tolerance)
