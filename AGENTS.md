# AGENTS.md

## Project: Rage Against The Machine

A live brain-computer interface game platform where a human plays against an adaptive AI opponent.
The frontend renders game/UI, while the backend handles EEG streaming, cognitive signal derivation,
and AI opponent personality responses (taunt text + speech + difficulty updates).

## Core Technology Stack

- Language: TypeScript (frontend), Python (backend)
- Frontend: React + Vite + Tailwind CSS
- Backend: FastAPI + WebSockets
- EEG Streaming: BrainFlow + Muse LSL adapters
- AI Opponent: OpenAI Responses API + OpenAI Audio Speech API (server-side only)

## Repository Structure

- `/frontend` - React app, visualizations, and game pages/hooks
- `/backend` - FastAPI services, EEG pipelines, websocket APIs, opponent logic
- `/docs` - technical notes and background references

## Dev Guardrails

### Code Quality
- Keep backend request/websocket payloads validated with Pydantic models.
- Avoid `any` in frontend except explicit integration boundaries.
- Keep opponent output contract stable and structured.
- Keep all difficulty math deterministic and testable.

### Real-time and Performance
- Avoid blocking work in websocket handlers.
- Opponent processing should keep event-to-response latency low (<1s target in local demo).
- Keep EEG-derived command-centre metrics backend-authoritative.
- Use rate limiting for taunt/speech generation to avoid event spam.

### Security
- No API keys in frontend.
- `OPENAI_API_KEY` must remain backend-only.
- Validate all client->server websocket payloads.
- No persistent PII storage for player identity in hackathon mode.

### Product Constraints
- Calibration is a separate workstream and not owned by opponent module tasks.
- Mock vs real EEG input should stay configurable.
- Difficulty policy must weight stress/frustration more than score.

## AI Opponent Personality Contract

### Input (frontend -> backend)
`game_event` payload contains:
- event type: `player_score | ai_score | near_score`
- current score
- current difficulty (0..1)
- optional near-score context

### Backend context
- latest command-centre metrics: stress, frustration, focus, alertness (0..1)
- backend-derived prior difficulty

### Output (backend -> frontend)
`opponent_update` payload includes:
- `taunt_text`
- `difficulty`: previous/model_target/final
- `speech.audio_base64` (MP3)
- metric snapshot and latency metadata

## Prohibited Patterns

- React state for high-frequency game-loop physics values
- Unvalidated websocket payloads
- Opponent responses without bounded difficulty values
- Backend calls to OpenAI without fallback handling
- Frontend use of raw OpenAI keys

## Success Criteria

1. Frontend receives opponent update bundles for game events over websocket.
2. Difficulty adjusts smoothly with stress-dominant weighting.
3. Taunts remain playful and safe.
4. Speech payload is playable by frontend.
5. System remains usable when OpenAI calls fail (fallback mode).
