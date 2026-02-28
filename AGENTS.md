  # AGENTS.md

  ## Project: Rage Against The Machine

  A live brain-computer interface Pong game where a human controls a paddle using Muse/OpenBCI EEG via an existing Thonk signal
  bridge. The AI opponent adapts difficulty based on inferred stress and generates ragebaiting trash-talk speech bubbles via GPT
  during key game events.

  ## Core Technology Stack

  - **Language**: TypeScript everywhere
  - **Frontend**: React + Vite + Tailwind CSS
  - **Backend**: Node.js + TypeScript + Socket.IO
  - **Game Rendering**: Canvas + requestAnimationFrame (no React re-render per frame)
  - **Realtime Transport**: Socket.IO
  - **AI Taunts**: OpenAI API (abstracted behind provider interface)

  ## Monorepo Structure

  /apps/web              # React + Vite frontend
  /apps/server           # Node.js + TypeScript backend
  /packages/shared       # Shared types, Zod schemas, event contracts
  /packages/thonk        # Placeholder Thonk adapter (no real hardware)
  /packages/game         # Pure TS game engine (deterministic physics)

  ## Dev Guardrails (Enforceable Rules)

  ### Code Quality
  - All modules must export TypeScript types/interfaces
  - No `any` types except in explicit adapter boundaries
  - All WebSocket events must have Zod schemas in `/packages/shared`
  - All game physics must be deterministic and testable without canvas

  ### Performance
  - Game loop must run at stable 60 FPS (16.67ms budget)
  - Input pipeline latency: thonk signal → rendered paddle movement < 50ms (p95)
  - WebSocket message latency < 20ms (p95)
  - Canvas rendering operations must not block event loop

  ### State Management
  - Game state lives in refs, NOT React state
  - React renders only for UI chrome (scores, menus, overlays)
  - No prop drilling; use Context only for session/calibration state
  - Server maintains single source of truth for game state

  ### Calibration
  - Calibration must converge in ≤3 trials per direction
  - Must produce quality score; reject if separation < 1.0σ
  - Store calibration per user session ID
  - Graceful fallback to keyboard if calibration fails

  ### Testing
  - All game engine functions must have unit tests
  - All WebSocket contracts must have integration tests
  - Calibration algorithm must have property-based tests
  - No visual regression tests (manual only)

  ### Security
  - No PII stored (session IDs are ephemeral UUIDs)
  - OpenAI API key server-side only
  - Rate limit GPT taunt requests (max 1/3s per session)
  - Validate all client→server messages with Zod

  ## Prohibited Patterns

  - ❌ React state for game loop variables (position, velocity)
  - ❌ Heavy ML training for calibration (must be closed-form)
  - ❌ Inline Tailwind classes longer than 12 tokens (extract to components)
  - ❌ WebSocket events without TypeScript types
  - ❌ Blocking operations in game loop
  - ❌ CSS frameworks other than Tailwind
  - ❌ Medical-grade stress detection claims

  ## Success Criteria

  1. Game runs at stable 60 FPS on mid-range laptop
  2. Calibration completes in < 30 seconds total
  3. Paddle responds to EEG within 50ms (perceptually instant)
  4. Taunts are contextual and entertaining
  5. AI difficulty adapts smoothly (not jarring)
  6. Code passes type-check with strict mode