# Rage Against The Machine

**Rage Against The Machine** is a thought-controlled game platform built around one core belief:
people who cannot reliably use their hands should still be able to compete, not just spectate.

Imagine a player with ALS using a controller that does not require hands.

Imagine someone with a spinal cord injury playing a real match.

Imagine a child with cerebral palsy moving a paddle using motor imagery alone.

That is what this project stands for.

## Hackathon Pitch

Brain-computer interfaces are real, but they are still hard to build in practice: signal-processing complexity, ML pipeline overhead, and long setup cycles.

For this hackathon, we built a working brain-controlled game loop end to end:

- a live Pong experience controlled by EEG intent,
- a Muse-compatible path (consumer headband form factor, around a $250 device class),
- real-time cognitive signal derivation (stress, frustration, focus, alertness),
- an adaptive AI opponent that changes difficulty from those signals,
- and live voice taunts generated server-side with OpenAI.

Codex was our primary coding partner across the stack: backend services, frontend game/UI wiring, BCI integration, and opponent orchestration.

## Open-Source Foundations and Prior Work

This project is built on open software and prior community work. We did not build in a vacuum.
In particular, we adapted an OpenBCI-oriented software path to run with a Muse headband form factor for practical public-facing demos.

Key foundations and inspirations:

- [Thonk](https://github.com/aether-raid/thonk): EEG streaming and OpenBCI system patterns used as a practical reference point.
- [NeuralFlight](https://github.com/dronefreak/NeuralFlight): motor imagery control ideas and model workflow inspiration.
- [muse-lsl](https://github.com/alexandrebarachant/muse-lsl): Muse-to-LSL streaming ecosystem that makes consumer-headset integration possible.
- [brain-arcade](https://github.com/itayinbarr/brain-arcade): game-oriented neurofeedback/opponent design inspiration.
- [BrainFlow](https://github.com/brainflow-dev/brainflow): hardware interface layer for EEG device streaming.
- [MNE-Python](https://mne.tools/stable/index.html): EEG data handling and preprocessing ecosystem.
- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/): benchmark dataset used in the model training/evaluation path.

This submission extends and adapts those foundations for a hackathon-ready, accessibility-first gameplay experience with real-time AI opponent interaction.

## What We Built

### 1) Thought-to-control gameplay
- Motor imagery predictions are streamed over WebSocket (`/mi/ws`).
- Left/right imagery maps to in-game movement (`strafe_left` / `strafe_right`).
- The Pong loop runs at 60 FPS in the browser and supports EEG + keyboard/hybrid control modes.

### 2) Real-time EEG streaming and signal services
- FastAPI backend streams EEG packets over `/bci/ws`.
- Device routing supports OpenBCI Cyton and Muse v1 paths.
- Muse control flow runs at approximately 250 Hz in practice (Muse v1 nominal 256 Hz stream).
- Command-centre cognitive signals are broadcast over `/bci/ccsignals/ws`.

### 3) Adaptive AI opponent with personality
- Game events are sent to `/opponent/ws` as structured payloads.
- Backend computes stress-weighted difficulty updates with bounded outputs.
- OpenAI generates taunt text + MP3 speech (`audio_base64`) with rule-based fallback when unavailable.

### 4) Practical model track
- EEGNet/EEGNetResidual training and tuning scripts live under `ml/`.
- Muse calibration fine-tuning script: `ml/tune_eegnet_muse.py`.
- LaBraM probing/fine-tuning/curriculum experiments are documented in `docs/eeg/main.tex` and `ml/labram-for-eegmmidb.ipynb`.

## Why This Matters

Most game interfaces assume motor ability.
Rage Against The Machine explores a different future: cognition-native control as an accessibility primitive, not a novelty.

This is not just EEG visualization. It is a playable loop with:
- intent decoding,
- feedback adaptation,
- and opponent behavior linked to live brain-state features.

## Architecture At A Glance

- `frontend/`
  React + Vite + Tailwind app with modules for Pong, Breakout, Combat3D, EEG dashboard, MI panel, and command-centre telemetry.

- `backend/`
  FastAPI services for EEG streaming, MI endpoints, and opponent WebSocket orchestration. Includes OpenAI-backed taunts/speech with fallback mode.

- `ml/`
  Model training/fine-tuning utilities (EEGNet + LaBraM experimentation and Muse calibration workflows).

- `docs/`
  Research and technical documentation.

## Quick Start (Local)

### Prerequisites
- Python 3.12+ for backend
- Node.js 18+ for frontend
- Optional: Muse LSL stream or OpenBCI device for live EEG

### 1) Backend (FastAPI)

```bash
cd backend
uv sync
```

Create `backend/.env` (or copy from `.env.sample`) and set at least:

```env
OPENAI_API_KEY=your_key_here
```

Run backend:

```bash
uv run app.py
```

Backend default URL: `http://localhost:8000`

### 2) Frontend (React/Vite)

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
VITE_BACKEND_URL=http://localhost:8000
```

Run frontend:

```bash
npm run dev
```

Frontend default URL: `http://localhost:5173`

## Demo Flow (Judge-Friendly)

1. Start backend and frontend.
2. Open `/pong`.
3. Start EEG stream from the UI (`/eeg`) using a supported device path.
4. Start MI streaming (`/mi`) and verify predictions.
5. Play Pong with EEG or hybrid controls.
6. Trigger score events and observe:
   - opponent taunts,
   - voice output,
   - difficulty updates based on live metrics.

## Key Endpoints

- `POST /bci/start`
- `POST /bci/stop`
- `GET /bci/status`
- `WS /bci/ws`
- `WS /bci/ccsignals/ws`
- `WS /mi/ws`
- `WS /opponent/ws`

## Accessibility Vision

Rage Against The Machine is a prototype of a broader claim:
controllers should adapt to humans, not force humans to adapt to controllers.

If you can think a command, you should be able to play.

## Team and Build Process

We are a four-person team and built this in hackathon time by combining:
- BCI/EEG domain understanding,
- rapid product iteration,
- and Codex-assisted full-stack execution.

What used to require a larger specialized team can now be prototyped fast enough to test with users.

## Repository Pointers

- [frontend](./frontend)
- [backend](./backend)
- [ml](./ml)
- [docs](./docs)
- [ARCHITECTURE_OVERVIEW.md](./ARCHITECTURE_OVERVIEW.md)
- [EVENT_CONTRACTS.md](./EVENT_CONTRACTS.md)
