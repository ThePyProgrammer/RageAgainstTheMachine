# Backend Compatibility Guide — Combat3D

> How the backend must be structured so the **Combat3D** frontend (`frontend/src/combat3d/` + `frontend/apps/combat3d/`) works end-to-end.

---

## 1. Server Overview

The Combat3D game relies on **two independent servers** running simultaneously:

| Server | Stack | Default Port | Env Var | Purpose |
|--------|-------|-------------|---------|---------|
| **Python backend** | FastAPI + Uvicorn | `8000` | `VITE_BACKEND_URL` | BCI signal processing (EEG, MI, PPG, Ocular) |
| **Node.js combat3d server** | Socket.IO + http | `4001` | `VITE_COMBAT3D_WS_URL` / `COMBAT3D_PORT` | Game session management, telemetry relay, taunts |

There is **no inter-server communication**. The frontend connects to both independently.

---

## 2. Node.js Combat3D Server

### 2.1 Expected Location

```
server/combat3d/
├── package.json          # Package name: @rage/combat3d-server
├── tsconfig.json
└── src/
    ├── index.ts          # HTTP + Socket.IO server entry point
    ├── contracts.ts      # Zod schemas + WS_EVENTS constant map
    └── mocks.ts          # Synthetic telemetry & taunt generators
```

### 2.2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMBAT3D_PORT` | `4001` | Port the Socket.IO server listens on |

### 2.3 Transport

- **Protocol:** Socket.IO (not raw WebSocket)
- **Path:** `/socket.io` (default)
- **Transports:** `["websocket"]` only (no long-polling)
- **CORS:** Must allow the Vite dev server origin (typically `http://localhost:5173`)

The frontend connects via:
```typescript
io(url, {
  autoConnect: false,
  path: "/socket.io",
  reconnectionDelay: 500,
  transports: ["websocket"],
});
```

### 2.4 Socket.IO Event Contract

All events use the `combat3d:` namespace prefix.

#### Client → Server Events

| Constant | Event String | Payload Schema | Description |
|----------|-------------|---------------|-------------|
| `WS_EVENTS.CONNECT` | `combat3d:connect` | `serverJoinSchema` | Client requests to join/create a session |

**`serverJoinSchema` / `joinSessionSchema`:**
```typescript
{
  sessionId: string;   // UUID v4
  mode: "classifier" | "features";
}
```

#### Server → Client Events

| Constant | Event String | Payload Schema | Description |
|----------|-------------|---------------|-------------|
| `WS_EVENTS.SESSION_CONFIG` | `combat3d:session-config` | `serverSessionSchema` | Session accepted, returns seed |
| `WS_EVENTS.TELEMETRY` | `combat3d:telemetry` | `serverTelemetrySchema` | BCI telemetry stream (~60ms interval) |
| `WS_EVENTS.TAUNT` | `combat3d:taunt` | `serverTauntSchema` | AI trash-talk (every 12th tick) |
| `WS_EVENTS.ERROR` | `combat3d:error` | `serverErrorSchema` | Validation/join errors |

**`serverSessionSchema` / `joinResponseSchema`:**
```typescript
{
  sessionId: string;       // UUID
  acceptedAt: number;      // Unix ms timestamp
  activeMode: "classifier" | "features";
  seed: number;            // int, >= 0, used for deterministic game RNG
}
```

**`serverTelemetrySchema` / `telemetrySchema`:**
```typescript
{
  timeMs: number;
  mode: "classifier" | "features";
  kind: "classifier" | "features";
  confidence: number;                    // 0–1
  payload: Record<string, number>;       // See below
  sessionId: string;                     // UUID
}
```

For `kind: "classifier"`, payload keys must be:
```json
{ "left": 0.3, "right": 0.5, "throttle_up": 0.1, "throttle_down": 0.05, "fire": 0.0, "idle": 0.05 }
```

For `kind: "features"`, payload keys must be `f0`, `f1`, `f2`, ... (sorted lexicographically):
```json
{ "f0": 0.12, "f1": -0.34, "f2": 0.56 }
```

> The frontend sorts keys matching `/^f/` to build the feature vector. Non-`f`-prefixed keys are silently ignored for features mode.

**`serverTauntSchema` / `tauntSchema`:**
```typescript
{
  sessionId: string;   // UUID
  tone: string;        // min length 2 (server) / 1 (frontend)
  stress: number;      // 0–1
  timeMs: number;
}
```

**`serverErrorSchema`:**
```typescript
{
  sessionId?: string;  // UUID, optional
  reason: string;      // min length 2
}
```

### 2.5 Server Behavior Requirements

1. **On `combat3d:connect`:** Validate the join payload against `serverJoinSchema`. Generate a deterministic seed from the `sessionId`. Emit `combat3d:session-config` back.
2. **Telemetry loop:** Start a ~60ms (`setInterval`) tick per session. Each tick emits `combat3d:telemetry` with either mock or real BCI data.
3. **Taunt loop:** Every 12th tick, emit `combat3d:taunt` with a taunt string and current stress level.
4. **On `disconnect`:** Clear the session's interval timer, free resources.
5. **HTTP fallback:** The server should return 404 for any HTTP requests (no REST endpoints).

### 2.6 Mock Data (for development)

The `mocks.ts` file provides:
- `createMockTelemetry(session)` — generates synthetic classifier or feature payloads
- `createMockTaunt(session)` — cycles through a queue of hardcoded taunt strings with computed stress
- A `sessionSchema` for internal session state: `{ sessionId, mode, tick, seed }`

---

## 3. Python Backend (BCI Signal Processing)

### 3.1 Expected Location

```
backend/
├── app.py                                # FastAPI entry point
├── pyproject.toml
├── .env / .env.sample
├── eeg/
│   ├── __init__.py
│   ├── models.py                         # Pydantic: EEGChunk, Session, EmbeddingConfig
│   ├── routes.py                         # /bci/* routes
│   └── services/
│       ├── __init__.py
│       ├── stream_service.py             # EEGStreamer (BrainFlow board → WebSocket)
│       ├── command_centre_service.py     # Cognitive signal derivation (focus, stress, etc.)
│       ├── embedding_service.py          # LaBraM embedding processor
│       └── streaming/
│           ├── board_manager.py
│           ├── data_processor.py
│           ├── device_registry.py
│           ├── session_manager.py
│           └── websocket_broadcaster.py
├── mi/
│   ├── initialization.py
│   ├── routes.py                         # /mi/* routes
│   ├── services/
│   │   ├── mi_controller.py             # EEGNet prediction → command mapping
│   │   ├── mi_processor.py
│   │   ├── stream_service.py
│   │   ├── calibration_manager.py
│   │   └── fine_tuner.py
│   └── config/
├── ppg/                                  # Pulse detection (DISABLED in app.py)
│   ├── routes.py
│   ├── models.py
│   ├── controller.py
│   └── services/
├── ocular/                               # Pupillometry (DISABLED in app.py)
│   ├── routes.py
│   ├── models.py
│   ├── controller.py
│   └── services/
└── shared/
    ├── __init__.py
    ├── websocket_server.py               # Generic WebSocket handler
    └── config/
        ├── app_config.py                 # Board/hardware constants
        └── logging.py
```

### 3.2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BOARD_SERIAL_PORT` | None | Serial port for EEG board (e.g., `COM3`) |
| `HF_REPO` | None | HuggingFace repo for model uploads |
| `HF_TOKEN` | None | HuggingFace API token |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### 3.3 REST Endpoints Used by Combat3D (indirectly)

Combat3D itself doesn't call the Python backend directly, but the BCI control pipeline (`bci/controlPipeline.ts`) processes data that **originates** from these Python endpoints, forwarded via the Node.js server:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/bci/start?device_type=` | Start EEG streaming |
| `POST` | `/bci/stop` | Stop EEG streaming |
| `GET` | `/bci/status` | Check streaming status |
| `GET` | `/bci/details` | Detailed session info |
| `GET` | `/bci/devices` | List available BCI devices |
| `POST` | `/bci/classification/embeddings/configure` | Enable/disable LaBraM embeddings |
| `GET` | `/bci/classification/embeddings/latest` | Get latest embedding |
| `GET` | `/bci/classification/embeddings/history?n=` | Embedding history |

### 3.4 WebSocket Endpoints Used by Frontend

| Path | Protocol | Payload Direction | Description |
|------|----------|-------------------|-------------|
| `/bci/ws` | Native WS | Server → Client | EEG samples: `{ "samples": [[...]] }` |
| `/bci/ccsignals/ws` | Native WS | Server → Client | Cognitive signals: `{ "signal": { "timestamp_ms", "signals": {...}, "raw": {...}, "device_type" } }` |
| `/mi/ws` | Native WS | Bidirectional | MI predictions. Client sends: `{ "action": "start"\|"stop" }`. Server sends: `{ "type": "prediction", "prediction", "label", "confidence", "command", "status", "timestamp" }` |

### 3.5 Cognitive Signal Payload Shape (`/bci/ccsignals/ws`)

This is particularly relevant for stress-based difficulty adaptation:

```json
{
  "signal": {
    "timestamp_ms": 1234567890,
    "signals": {
      "focus": 0.72,
      "alertness": 0.55,
      "drowsiness": 0.1,
      "stress": 0.3,
      "workload": 0.4,
      "engagement": 0.6,
      "relaxation": 0.5,
      "flow": 0.45,
      "frustration": 0.15
    },
    "raw": { "focus_ratio": 0.8, "drowsiness_ratio": 0.2 },
    "device_type": "cyton"
  }
}
```

### 3.6 MI Prediction Payload Shape (`/mi/ws`)

Used for motor imagery control:

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

---

## 4. Shared Package: `@ragemachine/bci-shared`

### 4.1 Expected Location

```
frontend/packages/bci-shared/
├── package.json              # Name: @ragemachine/bci-shared
└── src/
    ├── index.ts              # Re-exports all modules
    ├── types.ts              # Core BCI types
    ├── calibration.ts        # Calibration profile storage + quality gate
    └── filters/
        ├── math.ts           # clamp, clamp01, mapRange
        ├── ema.ts            # EMAFilter
        ├── hysteresis.ts     # BinaryHysteresis
        └── deadzone.ts       # applyDeadzone
```

### 4.2 Key Types

```typescript
type BCIMode = "classifier" | "features";

type ClassifierLabel = "left" | "right" | "throttle_up" | "throttle_down" | "fire" | "idle";

type BCIStreamPacket =
  | {
      kind: "classifier";
      label: ClassifierLabel;
      probabilities: Record<ClassifierLabel, number>;
      confidence: number;
      sessionId: string;
      timestamp: number;
    }
  | {
      kind: "features";
      features: readonly number[];
      confidence: number;
      sessionId: string;
      timestamp: number;
    };

interface BCIDecodedControls {
  throttle: number;    // -1 to 1
  turn: number;        // -1 to 1
  fire: boolean;
  confidence: number;  // 0 to 1
  source: BCIMode;
  timestamp: number;
}
```

---

## 5. Data Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │      Python Backend (:8000)          │
                    │  /bci/ws  /bci/ccsignals/ws  /mi/ws │
                    └────────────────┬────────────────────┘
                                     │ (EEG/MI data)
                                     │ NOT directly consumed by combat3d
                                     │ (used by command-centre UI, calibration)
                                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    Frontend (Vite :5173)                      │
│                                                              │
│  ┌─────────────────────┐     ┌──────────────────────────┐   │
│  │  BCI Hooks/UI        │     │  Combat3D Game            │   │
│  │  useBCIStream        │     │  ┌──────────────────┐     │   │
│  │  useMotorImagery     │     │  │  controlPipeline  │     │   │
│  │  useCCSignals        │     │  │  (BCI → controls) │     │   │
│  └─────────────────────┘     │  └────────┬─────────┘     │   │
│                               │           │               │   │
│                               │  ┌────────▼─────────┐     │   │
│                               │  │  Game Engine       │     │   │
│                               │  │  (deterministic    │     │   │
│                               │  │   physics + AI)    │     │   │
│                               │  └──────────────────┘     │   │
│                               └──────────┬───────────────┘   │
│                                          │ Socket.IO          │
└──────────────────────────────────────────┼───────────────────┘
                                           │
                              ┌─────────────▼──────────────┐
                              │  Node.js Server (:4001)     │
                              │  Socket.IO events:          │
                              │  combat3d:connect           │
                              │  combat3d:session-config    │
                              │  combat3d:telemetry         │
                              │  combat3d:taunt             │
                              │  combat3d:error             │
                              └────────────────────────────┘
```

---

## 6. Quick Checklist for Backend Compatibility

- [ ] `server/combat3d/src/index.ts` exists and starts a Socket.IO server on port `4001`
- [ ] `server/combat3d/src/contracts.ts` exports `WS_EVENTS` with all five event strings
- [ ] All Zod schemas match the shapes listed in §2.4
- [ ] `combat3d:connect` handler validates join payload and emits `combat3d:session-config`
- [ ] Telemetry loop emits `combat3d:telemetry` at ~60ms intervals
- [ ] Taunt loop emits `combat3d:taunt` every 12th tick
- [ ] Disconnect handler clears session intervals
- [ ] CORS allows `http://localhost:5173` (Vite dev server)
- [ ] Python backend `app.py` mounts `eeg_router` at `/bci` and `mi_router` at `/mi`
- [ ] All Python WebSocket endpoints (`/bci/ws`, `/bci/ccsignals/ws`, `/mi/ws`) are functional
- [ ] `@ragemachine/bci-shared` package is built and linked in the workspace
