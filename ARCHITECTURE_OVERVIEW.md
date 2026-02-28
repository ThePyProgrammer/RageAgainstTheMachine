# ARCHITECTURE_OVERVIEW.md

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  React UI (Vite)                                    │    │
│  │  - Menu, scores, overlays                           │    │
│  │  - Tailwind CSS components                          │    │
│  │  - Calibration wizard UI                            │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Canvas Game Loop (refs + RAF)                      │    │
│  │  - Renders at 60 FPS                                │    │
│  │  - Paddle, ball, AI opponent                        │    │
│  │  - Speech bubbles overlay                           │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Input Pipeline                                     │    │
│  │  - Thonk signal → filtered → command                │    │
│  │  - Calibration-aware normalization                  │    │
│  │  - Debounce & hysteresis                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │ Socket.IO                         │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────────┐
│                    Node.js Server                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Socket.IO Handler                                  │    │
│  │  - Session management                               │    │
│  │  - Event routing                                    │    │
│  │  - Calibration state storage                        │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Game State Manager                                 │    │
│  │  - Authoritative game state                         │    │
│  │  - AI opponent logic                                │    │
│  │  - Collision detection                              │    │
│  │  - Score tracking                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Stress Metric Calculator                           │    │
│  │  - Lightweight proxy from input variance            │    │
│  │  - Exponential moving average                       │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  GPT Taunt Generator                                │    │
│  │  - OpenAI API client                                │    │
│  │  - Event-triggered prompts                          │    │
│  │  - Rate limiting                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                   ┌───────┴────────┐
                   │  OpenAI API    │
                   └────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Thonk (External)                           │
│  - Muse/OpenBCI EEG hardware                                │
│  - Signal bridge → 4 signals (up/down/left/right)           │
│  - WebSocket server (assumed running)                        │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Initialization
```
User → Server: connect
Server → User: session_created {sessionId, calibrationRequired}
```

### 2. Calibration Flow
```
User → Server: start_calibration
Server → User: calibration_step {step: "baseline" | "left" | "right", trial: number}
Thonk → User: thonk_signal {up, down, left, right}
User → Server: calibration_sample {signals[], timestamp}
Server → User: calibration_complete {params: CalibrationParams, quality: number}
```

### 3. Game Loop (60 FPS client-side)
```
Thonk → User: thonk_signal (30-60 Hz)
User (Input Pipeline): signal → normalized → command {direction: "LEFT"|"RIGHT"|"NEUTRAL"}
User (Game Loop): command → paddle velocity → render
User → Server: input_command {direction, timestamp} (every 16ms)
Server (Game State): update ball, AI, collisions
Server → User: game_state {ball, paddles, score} (every 16ms)
User (Game Loop): reconcile server state → render
```

### 4. Stress & Taunt Flow
```
Server (Stress Metric): calculate from input variance
Server (Event Detector): detect game events (score, rally, etc.)
Server → GPT API: generate_taunt {event, stress, context}
GPT API → Server: taunt text
Server → User: taunt_message {text, duration}
User (React): render speech bubble overlay
```

## Package Responsibilities

### `/packages/shared`
- TypeScript types for all entities (GameState, Paddle, Ball, etc.)
- Zod schemas for WebSocket events
- Shared constants (game physics, thresholds)
- Event contract definitions

### `/packages/game`
- Pure game engine (no I/O)
- Physics: ball movement, paddle movement, collision detection
- Deterministic functions (testable)
- No dependency on browser or Node.js APIs

### `/packages/thonk`
- Placeholder adapter interface
- Mock Thonk client for development
- Type definitions for Thonk messages

### `/apps/web`
- React UI components (Tailwind CSS)
- Canvas rendering loop
- Input pipeline (signal processing)
- WebSocket client
- Calibration wizard

### `/apps/server`
- Socket.IO server
- Session management (in-memory Map)
- Game state authority
- Stress metric calculation
- GPT taunt orchestration
- Optional persistence stubs

## Latency Budget

| Component | Target | Max |
|-----------|--------|-----|
| Thonk → Browser | 10ms | 20ms |
| Input Pipeline | 5ms | 10ms |
| Browser → Server (WS) | 10ms | 20ms |
| Server Processing | 3ms | 5ms |
| Server → Browser (WS) | 10ms | 20ms |
| Render Frame | 10ms | 16.67ms |
| **End-to-End (Thonk → Screen)** | **48ms** | **91.67ms** |

## State Synchronization

- **Client-side prediction**: Client renders paddle immediately based on input
- **Server reconciliation**: Server sends authoritative ball/AI state every frame
- **No rollback**: Client trusts server for ball position (no client-side physics for ball)
- **Optimistic UI**: Paddle movement is instant; ball/score updates have 16-32ms delay

## Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Thonk disconnect | No signals for 2s | Show "Reconnecting..." overlay, pause game |
| Calibration failure | Quality score < 1.0 | Prompt re-calibration or keyboard fallback |
| WebSocket disconnect | Socket.IO event | Auto-reconnect with exponential backoff |
| Server crash | Client timeout | Show error, preserve calibration in localStorage |
| GPT API timeout | > 5s wait | Skip taunt, log error |
| Frame drop | Missed RAF | No action (occasional drops tolerated) |

## Technology Choices

### Frontend: React + Vite
**Rationale**: Vite offers fast HMR for rapid iteration. React for UI chrome only; game loop uses refs to avoid re-render overhead.

**Alternatives considered**:
- Next.js: Overkill (no SSR needed for single-page game)
- Vanilla TS: Harder to manage UI state for menus/overlays

### Backend: Node.js + Socket.IO
**Rationale**: Socket.IO simplifies WebSocket with auto-reconnect, rooms, and TypeScript support.

**Alternatives considered**:
- Native `ws`: More boilerplate for reconnection logic
- Go/Rust: Unnecessary complexity; latency bottleneck is network, not CPU
