# Backend Compatibility Guide — Pong

> How the backend must be structured so the **Pong** game (`frontend/src/features/pong/`) works end-to-end, and what needs to be built to fulfill the AGENTS.md vision.

---

## 1. Current State

**The Pong game is currently a fully client-side demo.** It has:

- **Zero WebSocket connections** — no socket code in the pong feature
- **Zero REST API calls** — no HTTP requests
- **Hardcoded stub values** — stress fixed at `0.24`, taunt text is `"RAGE! Make me hit it."`
- **Stub calibration wizard** — no real EEG connection
- **`DebugStats.thonkConnected`** — always `false`
- **`GameMode: "eeg"`** — defined but unused

All game logic runs locally in `requestAnimationFrame` with client-side physics.

---

## 2. Frontend File Structure

```
frontend/src/features/pong/
├── components/
│   ├── AvatarCustomizer.tsx          # Paddle/ball shape & pattern customization
│   ├── CalibrationWizard.tsx         # EEG calibration wizard (stub)
│   ├── DebugOverlay.tsx              # FPS, latency, Thonk status overlay
│   ├── GameCanvas.tsx                # Canvas + input binding + game loop
│   ├── KeyboardHints.tsx             # Key shortcut hints
│   ├── MenuScreen.tsx                # Play mode selection menu
│   ├── PatternPreview.tsx            # Canvas pattern preview
│   ├── ScoreBoard.tsx                # Player vs AI score display
│   ├── SettingsOverlay.tsx           # Theme + Avatar settings modal
│   ├── StressMeter.tsx               # Stress bar (hardcoded 0.24)
│   ├── TauntBubble.tsx               # Taunt text with fade-out animation
│   └── ThemeCustomizer.tsx           # Color, line, glow settings
├── game/
│   ├── gameLoop.ts                   # Core physics, collision, scoring
│   ├── patterns.ts                   # Procedural pattern tile generation
│   ├── renderer.ts                   # Canvas draw calls
│   └── shapes.ts                     # Shape path drawing utilities
├── state/
│   ├── settingsManager.ts            # localStorage persistence
│   └── usePongSettings.ts            # React settings hook
└── types/
    ├── pongRuntime.ts                # Runtime game state types
    └── pongSettings.ts               # UI/theme/avatar settings types

frontend/src/pages/PongPage.tsx       # Page component, route: /pong
```

---

## 3. Pong Runtime Types (what the backend must understand)

From `frontend/src/features/pong/types/pongRuntime.ts`:

```typescript
type GameMode = "keyboard" | "eeg";
type GameScreen = "menu" | "calibration" | "game" | "paused";

interface DebugStats {
  fps: number;
  latencyMs: number;
  thonkConnected: boolean;
  calibrationQuality?: number;
}

interface RuntimePaddle { x: number; y: number; width: number; height: number; }
interface RuntimeBall   { x: number; y: number; radius: number; }

interface RuntimeState {
  width: number;
  height: number;
  ball: RuntimeBall;
  leftPaddle: RuntimePaddle;
  rightPaddle: RuntimePaddle;
  playerScore: number;
  aiScore: number;
}

type GameInputState = {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  pointerX?: number;
  pointerY?: number;
};
```

---

## 4. What the Backend Needs to Provide (per AGENTS.md)

The AGENTS.md spec calls for:
1. **Server-authoritative game state** — server maintains the truth
2. **Thonk EEG bridge** — real-time EEG signal → paddle movement
3. **GPT taunts** — contextual trash-talk via OpenAI API
4. **Adaptive AI difficulty** — based on inferred stress

### 4.1 Required: Pong Socket.IO Server

Following the combat3d pattern, a Pong-specific Socket.IO server should be created.

#### Expected directory structure:

```
server/pong/
├── package.json              # @rage/pong-server
├── tsconfig.json
└── src/
    ├── index.ts              # Socket.IO server entry point
    ├── contracts.ts          # Zod schemas + event constants
    ├── engine.ts             # Server-side pong physics (or import from shared)
    └── taunts.ts             # GPT taunt integration (rate-limited)
```

#### Expected Socket.IO events:

| Event String | Direction | Payload | Description |
|-------------|-----------|---------|-------------|
| `pong:join` | Client → Server | `{ sessionId: string, mode: "keyboard" \| "eeg" }` | Join/create a game session |
| `pong:session-config` | Server → Client | `{ sessionId: string, acceptedAt: number, mode: string }` | Session accepted |
| `pong:paddle-input` | Client → Server | `{ sessionId: string, y: number, timestamp: number }` | Player paddle position update |
| `pong:state` | Server → Client | `RuntimeState` (see §3) | Authoritative game state broadcast |
| `pong:score` | Server → Client | `{ playerScore: number, aiScore: number }` | Score update |
| `pong:taunt` | Server → Client | `{ text: string, stress: number, timestamp: number }` | AI trash-talk |
| `pong:stress` | Server → Client | `{ stress: number, timestamp: number }` | Stress level update |
| `pong:difficulty` | Server → Client | `{ level: number, aggression: number }` | Difficulty adjustment notification |
| `pong:error` | Server → Client | `{ reason: string }` | Validation/join errors |

#### Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PONG_PORT` | `4002` | Port for pong Socket.IO server |
| `OPENAI_API_KEY` | — | Server-side only, for GPT taunts |

#### Corresponding frontend env var:

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_PONG_WS_URL` | `ws://localhost:4002` | Pong server WebSocket URL |

---

### 4.2 Required: BCI Integration via Python Backend

The Pong frontend's EEG mode should use the **existing Python backend** endpoints to receive EEG data and translate it to paddle movement.

#### Endpoints the Pong game should consume:

| Endpoint | Transport | Purpose |
|----------|-----------|---------|
| `POST /bci/start` | HTTP | Start EEG streaming |
| `POST /bci/stop` | HTTP | Stop EEG streaming |
| `GET /bci/status` | HTTP | Check if EEG is streaming |
| `GET /bci/devices` | HTTP | List available BCI devices |
| `ws://localhost:8000/bci/ws` | Native WS | Receive real-time EEG samples |
| `ws://localhost:8000/bci/ccsignals/ws` | Native WS | Receive stress/focus signals for difficulty adaptation |

#### EEG Stream Payload (`/bci/ws`):
```json
{ "samples": [[sample_index, ts_unix_ms, ts_formatted, marker, ch1, ch2, ...chN, ...derived]] }
```

#### Cognitive Signals Payload (`/bci/ccsignals/ws`):
```json
{
  "signal": {
    "timestamp_ms": 1234567890,
    "signals": {
      "focus": 0.72,
      "stress": 0.3,
      "engagement": 0.6,
      "relaxation": 0.5,
      "frustration": 0.15
    },
    "device_type": "cyton"
  }
}
```

> **Key field for pong:** `signals.stress` drives the AI difficulty adaptation. The `StressMeter.tsx` component (currently hardcoded to `0.24`) should bind to this value.

#### Motor Imagery Payload (`/mi/ws`) — for paddle control:
```json
{
  "type": "prediction",
  "prediction": 0,
  "label": "Left Hand",
  "confidence": 85.3,
  "command": "strafe_left",
  "status": "MOVING",
  "timestamp": 1234567890.123
}
```

Relevant MI commands for Pong paddle control:

| `command` value | Pong action |
|----------------|-------------|
| `strafe_left` / `move_up` | Paddle up |
| `strafe_right` / `move_down` | Paddle down |
| `idle` | No movement |

---

### 4.3 Required: GPT Taunt Endpoint

Per AGENTS.md: *"AI opponent generates ragebaiting trash-talk speech bubbles via GPT during key game events."*

The Node.js pong server should implement:

```
POST /api/taunt  (internal, no client access)
```

Or more likely, the taunt generation is internal to the server and emitted via `pong:taunt` Socket.IO event:

- **Rate limit:** Max 1 request per 3 seconds per session (per AGENTS.md)
- **OpenAI API key:** Server-side only, never exposed to client
- **Trigger events:** Score, near-miss, streak, calibration fail
- **Payload to GPT:** Current score, stress level, game event context
- **Response to client:** `{ text: "Your neurons are filing for divorce.", stress: 0.6, timestamp: 12345 }`

---

### 4.4 Required: Calibration Endpoints

Per AGENTS.md: *"Calibration must converge in ≤3 trials per direction"*

The Python backend already has MI calibration routes:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/mi/calibration/start?user_id=` | Start calibration session |
| `POST` | `/mi/calibration/trial/start?label=` | Start a calibration trial |
| `POST` | `/mi/calibration/trial/end` | End current trial |
| `POST` | `/mi/calibration/end` | End calibration session |
| `GET` | `/mi/calibration/stats` | Get calibration progress |

The `CalibrationWizard.tsx` component needs to be wired to these endpoints.

Additionally, the `@ragemachine/bci-shared` package provides:
- `getSeparationScore(left[], right[])` — σ-based quality metric
- `canTrustCalibration(left[], right[], threshold=1)` — quality gate
- `saveCalibrationProfile()` / `loadCalibrationProfile()` — localStorage

---

## 5. Python Backend File Structure (relevant to Pong)

These files must exist and be compatible:

```
backend/
├── app.py                                # Mount eeg_router at /bci, mi_router at /mi
├── eeg/
│   ├── routes.py                         # /bci/* REST + WS endpoints
│   ├── models.py                         # EEGChunk, Session, EmbeddingConfig
│   └── services/
│       ├── stream_service.py             # EEGStreamer (board → WS broadcast)
│       ├── command_centre_service.py     # Derives stress, focus, etc.
│       └── streaming/
│           ├── board_manager.py          # BrainFlow board setup
│           ├── data_processor.py         # Raw data formatting
│           ├── device_registry.py        # Cyton & Muse configs
│           └── websocket_broadcaster.py  # Fan-out to WS clients
├── mi/
│   ├── routes.py                         # /mi/* REST + WS endpoints
│   └── services/
│       ├── mi_controller.py              # EEGNet prediction → command
│       ├── stream_service.py             # MI calibration data
│       └── calibration_manager.py        # Calibration dataset
└── shared/
    ├── websocket_server.py               # Generic WS handler
    └── config/
        └── app_config.py                 # Board/hardware constants
```

---

## 6. Frontend Files That Need Backend Wiring

These frontend files currently have stubs that need real backend connections:

| File | Current State | Needs |
|------|--------------|-------|
| `components/StressMeter.tsx` | Hardcoded `stress = 0.24` | Bind to `/bci/ccsignals/ws` → `signals.stress` |
| `components/TauntBubble.tsx` | Hardcoded `"RAGE! Make me hit it."` | Bind to `pong:taunt` Socket.IO event |
| `components/CalibrationWizard.tsx` | Stub/demo | Wire to `/mi/calibration/*` REST endpoints |
| `components/DebugOverlay.tsx` | `thonkConnected: false` | Bind to `/bci/status` REST endpoint |
| `components/GameCanvas.tsx` | Local game loop only | Add Socket.IO client for server-authoritative state |
| `game/gameLoop.ts` | Client-side physics | Either replace with server state sync or keep client-predicted |

---

## 7. Data Flow Diagram (Target Architecture)

```
┌─────────────────────────────────────────────┐
│          Python Backend (:8000)              │
│                                             │
│  /bci/ws         → EEG samples              │
│  /bci/ccsignals/ws → stress, focus signals   │
│  /mi/ws          → motor imagery predictions │
│  /mi/calibration/* → calibration endpoints   │
│  /bci/start|stop → streaming control         │
└──────────┬──────────────────────────────────┘
           │ Native WebSocket + HTTP
           ▼
┌─────────────────────────────────────────────────────┐
│               Frontend (:5173)                       │
│                                                     │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │ BCI Hooks             │  │  Pong Game            │ │
│  │ useBCIStream()        │  │                      │ │
│  │ useCommandCentreSignals│ │  StressMeter ←────┐  │ │
│  │ useMotorImagery()     │──│→ GameCanvas        │  │ │
│  └──────────────────────┘  │  TauntBubble ←──┐  │  │ │
│                             │  CalibWizard    │  │  │ │
│                             └───────┬────┬───┘  │  │ │
│                                     │    │      │  │ │
└─────────────────────────────────────┼────┼──────┼──┘ │
                                      │    │      │    │
                          Socket.IO   │    │      │    │
                                      ▼    │      │    │
                    ┌─────────────────────────────────┐│
                    │   Node.js Pong Server (:4002)    ││
                    │                                  ││
                    │   pong:join                       ││
                    │   pong:session-config             ││
                    │   pong:paddle-input               ││
                    │   pong:state    ─────────────────┘│
                    │   pong:taunt   ──────────────────┘│
                    │   pong:stress  ───────────────────┘
                    │   pong:score                      │
                    │   pong:difficulty                 │
                    │   pong:error                      │
                    │                                  │
                    │   Internal: OpenAI API (taunts)  │
                    └─────────────────────────────────┘
```

---

## 8. Existing Frontend Hooks Available for Reuse

These hooks already exist in `frontend/src/hooks/` and connect to the Python backend. The Pong feature can import and use them directly:

| Hook | Connects To | Returns |
|------|------------|---------|
| `useBCIStream()` | `/bci/ws` + `/bci/start` | EEG sample stream |
| `useCommandCentreSignals()` | `/bci/ccsignals/ws` | `{ stress, focus, engagement, ... }` |
| `useMotorImagery()` | `/mi/ws` | MI predictions (`{ command, confidence }`) |
| `useClassificationModelEmbeddings()` | `/bci/classification/embeddings/*` | Embedding vectors |

---

## 9. Quick Checklist for Backend Compatibility

### Python Backend (`backend/`)
- [ ] `app.py` mounts `eeg_router` at `/bci` prefix
- [ ] `app.py` mounts `mi_router` at `/mi` prefix
- [ ] `/bci/ws` WebSocket streams EEG samples as `{ "samples": [...] }`
- [ ] `/bci/ccsignals/ws` WebSocket streams cognitive signals with `signals.stress`
- [ ] `/mi/ws` WebSocket accepts `{ "action": "start" }` and streams predictions
- [ ] `/mi/calibration/start`, `/trial/start`, `/trial/end`, `/end`, `/stats` all functional
- [ ] CORS allows `http://localhost:5173`

### Node.js Pong Server (`server/pong/`) — TO BE CREATED
- [ ] `server/pong/` directory exists with `package.json`, `tsconfig.json`, `src/`
- [ ] Socket.IO server listens on port `4002` (or `PONG_PORT`)
- [ ] All `pong:*` events defined in `contracts.ts` with Zod schemas
- [ ] `pong:join` handler validates and emits `pong:session-config`
- [ ] Game state broadcast loop emits `pong:state` at ~60fps
- [ ] Taunt integration via OpenAI API (rate-limited to 1/3s)
- [ ] Stress level forwarding from Python backend `/bci/ccsignals/ws`
- [ ] CORS allows `http://localhost:5173`

### Shared Package
- [ ] `@ragemachine/bci-shared` is built and linked
- [ ] Pong feature imports BCI types for EEG mode
