import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import { createOpponentInput } from "./ai/difficulty";
import { CombatView } from "./render/CombatView";
import { createDefaultControlPipeline } from "./bci/controlPipeline";
import {
  FRAME_MS,
  createCombatConfig,
  createStateFromSeed,
  stepSimulation,
} from "./engine";
import { InputQueue } from "./engine/inputQueue";
import { SeededRNG } from "./engine/seededRng";
import { DebugPanel } from "./debug/DebugPanel";
import { TauntBubble } from "./ui/TauntBubble";
import { EEGStreamModal } from "./ui/EEGStreamModal";
import type { BCIStreamPacket } from "@ragemachine/bci-shared";
import { createCombat3DSocket } from "./net/socket";
import { normalizeTaunt, type TauntPacket, type TelemetryPacket } from "./net/contracts";
import type { CombatState, DynamicObstacle, InputSample } from "./engine/types";
import type { BarrierBreakEvent } from "./engine/barriers";
import { barriersFromObstacles, type Barrier } from "./engine/barriers";
import { generateObstacles } from "./render/Combat3DScene";
import { API_ENDPOINTS } from "@/config/api";
import { CalibrationWizard } from "@/features/pong/components/CalibrationWizard";

const WS_URL = import.meta.env.VITE_COMBAT3D_WS_URL || "ws://localhost:4001";

const mapTelemetry = (packet: TelemetryPacket): BCIStreamPacket => {
  const base = {
    confidence: packet.confidence,
    timestamp: packet.timeMs,
    sessionId: packet.sessionId,
  };

  if (packet.kind === "features") {
    const ordered = Object.keys(packet.payload)
      .filter((key) => key.startsWith("f"))
      .sort()
      .map((key) => packet.payload[key] ?? 0);
    return {
      ...base,
      kind: "features",
      features: ordered,
    };
  }

  return {
    ...base,
    kind: "classifier",
    label: "idle",
    probabilities: {
      left: packet.payload.left ?? 0,
      right: packet.payload.right ?? 0,
      throttle_up: packet.payload.throttle_up ?? 0,
      throttle_down: packet.payload.throttle_down ?? 0,
      fire: packet.payload.fire ?? 0,
      idle: packet.payload.idle ?? 0,
    },
  };
};

/* ── Keyboard held-key state ────────────────────────────────────────── */

const createSessionId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i += 1) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }

  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
};

interface ActiveTaunt {
  text: string;
  audioBase64: string;
  eventKey: number;
}

const TRACKED_KEYS = new Set([
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "w", "W", "a", "A", "s", "S", "d", "D", " ",
  "r", "R",
]);

interface KeyState {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  fire: boolean;
  reload: boolean;
}

const createKeyState = (): KeyState => ({
  up: false, down: false, left: false, right: false, fire: false, reload: false,
});

/** Sample held-key state into a deterministic InputSample at the given sim time. */
const sampleKeyboard = (keys: KeyState, simTimeMs: number): InputSample => {
  let throttle = 0;
  let turn = 0;
  if (keys.up) throttle += 1;
  if (keys.down) throttle -= 1;
  if (keys.left) turn -= 0.8;
  if (keys.right) turn += 0.8;
  return { timestamp: simTimeMs, throttle, turn, fire: keys.fire, reload: keys.reload };
};

/** Returns true if any movement/fire key is currently held. */
const hasActiveInput = (keys: KeyState): boolean =>
  keys.up || keys.down || keys.left || keys.right || keys.fire || keys.reload;

export interface DebugState {
  queueLen: number;
  rttMs: number;
  lastInput: { throttle: number; turn: number; fire: boolean; source: string };
  playerPos: { x: number; y: number; yaw: number };
  projectileCount: number;
  lastEvent: string;
  tick: number;
}

const INITIAL_DEBUG: DebugState = {
  queueLen: 0,
  rttMs: 0,
  lastInput: { throttle: 0, turn: 0, fire: false, source: "none" },
  playerPos: { x: 0, y: 0, yaw: 0 },
  projectileCount: 0,
  lastEvent: "—",
  tick: 0,
};

type EegHemisphere = "left" | "right";

interface EegStreamHudState {
  waveSamples: number[];
  leftPower: number;
  rightPower: number;
  confidence: number;
  packetRateHz: number;
  activeHemisphere: EegHemisphere;
  mode: TelemetryPacket["mode"];
}

interface EegRateWindow {
  windowStartMs: number;
  packetCount: number;
  packetRateHz: number;
}

const EEG_WAVE_POINTS = 42;
const EEG_UI_COMMIT_INTERVAL_MS = 90;

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));
const clampSignal = (value: number): number => Math.max(-1, Math.min(1, value));

const createInitialWave = (): number[] =>
  Array.from(
    { length: EEG_WAVE_POINTS },
    (_, i) => Math.sin((i / EEG_WAVE_POINTS) * Math.PI * 3.5) * 0.16,
  );

const createInitialEegHudState = (): EegStreamHudState => ({
  waveSamples: createInitialWave(),
  leftPower: 0.62,
  rightPower: 0.2,
  confidence: 0,
  packetRateHz: 0,
  activeHemisphere: "left",
  mode: "classifier",
});

/* ── EEG calibration types & helpers ──────────────────────────────── */

type CombatScreen = "menu" | "calibration" | "game" | "paused";
type CalibrationStep = "left" | "right" | "fine_tuning" | "complete" | "error";

const LEFT_TRIAL_MS = 7000;
const RIGHT_TRIAL_MS = 7000;
const PROGRESS_TICK_MS = 120;

const sleep = (ms: number) =>
  new Promise<void>((resolve) => { window.setTimeout(resolve, ms); });

const parseErrorMessage = (payload: unknown, fallback: string): string => {
  if (typeof payload === "object" && payload !== null) {
    for (const key of ["detail", "error", "message"]) {
      const val = Reflect.get(payload, key);
      if (typeof val === "string" && val.length > 0) return val;
    }
  }
  return fallback;
};

async function postJson<T>(url: string, body?: Record<string, unknown>): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await res.json().catch(() => undefined);
  if (!res.ok) throw new Error(parseErrorMessage(payload, `Request failed (${res.status})`));
  return payload as T;
}

async function ensureEegStreamRunning(): Promise<void> {
  const statusRes = await fetch(API_ENDPOINTS.BCI_STATUS);
  if (statusRes.ok) {
    const data = (await statusRes.json().catch(() => undefined)) as
      | { is_streaming?: boolean }
      | undefined;
    if (data?.is_streaming) return;
  }
  const startRes = await fetch(API_ENDPOINTS.BCI_START, { method: "POST" });
  const startPayload = await startRes.json().catch(() => undefined);
  if (!startRes.ok) {
    const msg = parseErrorMessage(startPayload, `Failed to start EEG (${startRes.status})`);
    if (!msg.toLowerCase().includes("already_running")) throw new Error(msg);
  }
}

export const App = () => {
  const [activeTaunt, setActiveTaunt] = useState<ActiveTaunt | null>(null);
  const [screen, setScreen] = useState<CombatScreen>("menu");
  const [debug, setDebug] = useState<DebugState>(INITIAL_DEBUG);
  const [barrierBreaks, setBarrierBreaks] = useState<BarrierBreakEvent[]>([]);
  const [score, setScore] = useState({ player: 0, enemy: 0 });
  const [dynamicObs, setDynamicObs] = useState<DynamicObstacle[]>([]);
  const [eegMode, setEegMode] = useState(false);
  const [tipVisible, setTipVisible] = useState(true);
  const initialEegHudState = useMemo(() => createInitialEegHudState(), []);
  const [eegHud, setEegHud] = useState<EegStreamHudState>(initialEegHudState);

  /* Calibration state */
  const [calibrationStep, setCalibrationStep] = useState<CalibrationStep>("left");
  const [calibrationInstruction, setCalibrationInstruction] = useState("");
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationQuality, setCalibrationQuality] = useState<number | undefined>(undefined);
  const [calibrationError, setCalibrationError] = useState<string | undefined>(undefined);
  const [calibrationRunning, setCalibrationRunning] = useState(false);

  const stateRef = useRef<CombatState>(createStateFromSeed(0x2f2f));
  const config = useMemo(() => createCombatConfig(0x2f2f), []);
  const playerQueueRef = useRef(new InputQueue());
  const enemyQueueRef = useRef(new InputQueue());
  const rngRef = useRef(new SeededRNG(config.seed));
  const pipelineRef = useRef(createDefaultControlPipeline());
  const socketRef = useRef<ReturnType<typeof createCombat3DSocket> | null>(null);
  const keyStateRef = useRef<KeyState>(createKeyState());
  const prevScoreRef = useRef({ player: 0, enemy: 0 });
  const lastEventRef = useRef("—");
  const tauntEventKeyRef = useRef(0);

  /* EEG / MI refs */
  const screenRef = useRef<CombatScreen>(screen);
  screenRef.current = screen;
  const eegModeRef = useRef(false);
  const eegTurnRef = useRef(0);
  const miSocketRef = useRef<WebSocket | null>(null);
  const calibrationRunIdRef = useRef(0);
  const eegHudStateRef = useRef<EegStreamHudState>(initialEegHudState);
  const eegWaveRef = useRef<number[]>(initialEegHudState.waveSamples.slice());
  const eegRateWindowRef = useRef<EegRateWindow>({
    windowStartMs: performance.now(),
    packetCount: 0,
    packetRateHz: 0,
  });
  const lastEegHudCommitRef = useRef(0);

  // Barrier collision state — initialized from the same deterministic obstacle generation
  const barriersRef = useRef<Barrier[]>(
    barriersFromObstacles(generateObstacles(0x2f2f, 20)),
  );
  /** Tracks how many dynamic obstacles have already been registered as barriers. */
  const dynBarrierCountRef = useRef(0);

  const lastFrameMsRef = useRef(performance.now());
  const accumulatorMsRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  /* ── MI WebSocket helpers ─────────────────────────────────────────── */

  const publishEegHud = useCallback(
    (
      waveSample: number,
      patch: Partial<Omit<EegStreamHudState, "waveSamples">> = {},
      forceCommit = false,
    ) => {
      const nextWave = eegWaveRef.current;
      nextWave.push(clampSignal(waveSample));
      if (nextWave.length > EEG_WAVE_POINTS) {
        nextWave.shift();
      }

      const nextState: EegStreamHudState = {
        ...eegHudStateRef.current,
        ...patch,
        waveSamples: [...nextWave],
      };
      eegHudStateRef.current = nextState;

      const now = performance.now();
      if (forceCommit || now - lastEegHudCommitRef.current >= EEG_UI_COMMIT_INTERVAL_MS) {
        lastEegHudCommitRef.current = now;
        setEegHud(nextState);
      }
    },
    [],
  );

  const teardownMiSocket = useCallback(() => {
    const ws = miSocketRef.current;
    miSocketRef.current = null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stop" }));
    }
    if (ws && ws.readyState !== WebSocket.CLOSED) ws.close();
    eegTurnRef.current = 0;
    publishEegHud(
      -0.24,
      {
        activeHemisphere: "left",
        leftPower: 0.62,
        rightPower: 0.18,
      },
      true,
    );
  }, [publishEegHud]);

  const connectAndStartMiStreaming = useCallback(async () => {
    teardownMiSocket();
    await new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(API_ENDPOINTS.MI_WS);
      miSocketRef.current = ws;
      let settled = false;
      const tid = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        ws.close();
        reject(new Error("MI websocket start timed out."));
      }, 8000);

      ws.onopen = () => {
        ws.send(JSON.stringify({ action: "start", interval_ms: 1000, reset: true }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data) as {
          status?: string; type?: string; error?: string; command?: string;
        };
        if (data.error) {
          if (!settled) { settled = true; window.clearTimeout(tid); reject(new Error(data.error)); }
          return;
        }
        if (data.status === "started") {
          if (!settled) { settled = true; window.clearTimeout(tid); resolve(); }
          return;
        }
        if (data.type === "prediction") {
          const cmd = String(data.command ?? "").toLowerCase();
          if (cmd === "strafe_left" || cmd === "left") eegTurnRef.current = -0.8;
          else if (cmd === "strafe_right" || cmd === "right") eegTurnRef.current = 0.8;
          else eegTurnRef.current = 0;

          const hemisphere: EegHemisphere = eegTurnRef.current > 0 ? "right" : "left";
          const signed = hemisphere === "right" ? 1 : -1;
          const pulse = 0.58 + Math.abs(Math.sin(Date.now() / 120)) * 0.36;
          publishEegHud(
            signed * pulse,
            {
              activeHemisphere: hemisphere,
              leftPower: hemisphere === "left" ? 0.84 : 0.18,
              rightPower: hemisphere === "right" ? 0.84 : 0.18,
            },
            true,
          );
        }
      };

      ws.onerror = () => {
        if (!settled) { settled = true; window.clearTimeout(tid); reject(new Error("MI websocket failed.")); }
      };
      ws.onclose = () => {
        eegTurnRef.current = 0;
        publishEegHud(
          -0.24,
          {
            activeHemisphere: "left",
            leftPower: 0.62,
            rightPower: 0.18,
          },
          true,
        );
        if (!settled) { settled = true; window.clearTimeout(tid); reject(new Error("MI websocket closed.")); }
      };
    });
  }, [publishEegHud, teardownMiSocket]);

  const waitWithProgress = useCallback(async (durationMs: number, start: number, end: number) => {
    const began = performance.now();
    for (;;) {
      const ratio = Math.min(1, (performance.now() - began) / durationMs);
      setCalibrationProgress(start + (end - start) * ratio);
      if (ratio >= 1) break;
      await sleep(PROGRESS_TICK_MS);
    }
  }, []);

  const runCalibrationRound = useCallback(
    async (runId: number) => {
      const userId = `combat_${Date.now().toString(36)}`;
      let sessionOpen = false;
      const isActive = () => calibrationRunIdRef.current === runId;

      setCalibrationRunning(true);
      setCalibrationError(undefined);
      setCalibrationQuality(undefined);
      setCalibrationProgress(0.02);

      try {
        await ensureEegStreamRunning();
        await postJson<{ save_dir: string }>(API_ENDPOINTS.MI_CALIBRATION_START, { user_id: userId });
        sessionOpen = true;
        if (!isActive()) return;

        setCalibrationStep("left");
        setCalibrationInstruction("Look left and hold focus.");
        await postJson(API_ENDPOINTS.MI_CALIBRATION_TRIAL_START, { label: 0 });
        await waitWithProgress(LEFT_TRIAL_MS, 0.05, 0.45);
        await postJson(API_ENDPOINTS.MI_CALIBRATION_TRIAL_END);
        if (!isActive()) return;

        setCalibrationStep("right");
        setCalibrationInstruction("Look right and hold focus.");
        await postJson(API_ENDPOINTS.MI_CALIBRATION_TRIAL_START, { label: 1 });
        await waitWithProgress(RIGHT_TRIAL_MS, 0.45, 0.85);
        await postJson(API_ENDPOINTS.MI_CALIBRATION_TRIAL_END);
        if (!isActive()) return;

        const calEnd = await postJson<{ session_dir: string }>(API_ENDPOINTS.MI_CALIBRATION_END);
        sessionOpen = false;

        setCalibrationStep("fine_tuning");
        setCalibrationInstruction("Fine-tuning lightweight classifier...");
        setCalibrationProgress(0.9);

        await postJson(API_ENDPOINTS.MI_FINETUNE_PREPARE, {
          user_id: userId,
          session_dir: calEnd.session_dir,
        });
        if (!isActive()) return;

        const runRes = await postJson<{ summary?: { best_val_acc?: number } }>(
          API_ENDPOINTS.MI_FINETUNE_RUN,
          { n_epochs: 12, batch_size: 8, val_split: 0.25 },
        );
        if (!isActive()) return;

        await postJson(API_ENDPOINTS.MI_FINETUNE_SAVE, { user_id: userId });

        const bestVal = Number(runRes.summary?.best_val_acc ?? 0);
        const quality = Number.isFinite(bestVal) ? Math.max(0, Math.min(1, bestVal)) : 0;
        setCalibrationQuality(quality);

        setCalibrationStep("complete");
        setCalibrationInstruction("Calibration complete. Starting EEG control...");
        setCalibrationProgress(1);

        await connectAndStartMiStreaming();
        if (!isActive()) { teardownMiSocket(); return; }

        eegModeRef.current = true;
        setEegMode(true);
        setScreen("game");
        setCalibrationRunning(false);
      } catch (err) {
        if (!isActive()) return;
        setCalibrationStep("error");
        setCalibrationRunning(false);
        setCalibrationError(err instanceof Error ? err.message : "Calibration failed.");
        setCalibrationInstruction("Unable to calibrate EEG signal. Check headset contact.");
      } finally {
        if (sessionOpen) await postJson(API_ENDPOINTS.MI_CALIBRATION_END).catch(() => {});
      }
    },
    [connectAndStartMiStreaming, teardownMiSocket, waitWithProgress],
  );

  const startEEGCalibration = useCallback(() => {
    calibrationRunIdRef.current += 1;
    const runId = calibrationRunIdRef.current;
    setScreen("calibration");
    setCalibrationStep("left");
    setCalibrationInstruction("Look left to begin calibration.");
    setCalibrationProgress(0);
    setCalibrationError(undefined);
    setCalibrationQuality(undefined);
    void runCalibrationRound(runId);
  }, [runCalibrationRound]);

  const handleManualPlay = useCallback(() => {
    eegModeRef.current = false;
    setEegMode(false);
    setScreen("game");
  }, []);

  // Clean up MI socket on unmount
  useEffect(() => () => { teardownMiSocket(); }, [teardownMiSocket]);

  // Pause (P) and quit (Esc) shortcuts
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && (screenRef.current === "game" || screenRef.current === "paused")) {
        event.preventDefault();
        teardownMiSocket();
        setScreen("menu");
        return;
      }
      if (event.key.toLowerCase() === "p" && (screenRef.current === "game" || screenRef.current === "paused")) {
        event.preventDefault();
        setScreen((prev) => (prev === "paused" ? "game" : "paused"));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [teardownMiSocket]);

  useEffect(() => {
    const socket = createCombat3DSocket({
      url: WS_URL,
      onTelemetry: (packet: TelemetryPacket) => {
        const sharedPacket = mapTelemetry(packet);
        const controls = pipelineRef.current.onPacket(sharedPacket);

        const input: InputSample = {
          timestamp: sharedPacket.timestamp,
          throttle: controls.throttle,
          turn: controls.turn,
          fire: controls.fire,
        };
        playerQueueRef.current.push(input);

        const rateWindow = eegRateWindowRef.current;
        rateWindow.packetCount += 1;
        const now = performance.now();
        const elapsed = now - rateWindow.windowStartMs;
        if (elapsed >= 1000) {
          rateWindow.packetRateHz = (rateWindow.packetCount * 1000) / elapsed;
          rateWindow.packetCount = 0;
          rateWindow.windowStartMs = now;
        }

        const confidence = clamp01(packet.confidence);
        const priorHemisphere = eegHudStateRef.current.activeHemisphere;
        let activeHemisphere: EegHemisphere = priorHemisphere;
        let leftPower = eegHudStateRef.current.leftPower;
        let rightPower = eegHudStateRef.current.rightPower;
        let waveSample = 0;

        if (packet.kind === "classifier") {
          leftPower = clamp01(packet.payload.left ?? 0);
          rightPower = clamp01(packet.payload.right ?? 0);
          const dominance = rightPower - leftPower;
          const osc = Math.sin(packet.timeMs / 120) * (0.2 + confidence * 0.5);
          waveSample = clampSignal(dominance * 0.95 + osc * 0.6);

          if (!eegModeRef.current) {
            activeHemisphere = rightPower > leftPower + 0.04 ? "right" : "left";
          }
        } else {
          const values = Object.values(packet.payload);
          const energy = values.length > 0
            ? Math.min(
              1,
              values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length / 8,
            )
            : 0;
          const direction = priorHemisphere === "right" ? 1 : -1;
          waveSample = clampSignal(
            Math.sin(packet.timeMs / 140) * (0.22 + energy * 0.75) * direction,
          );
          leftPower = priorHemisphere === "left" ? 0.62 : 0.22;
          rightPower = priorHemisphere === "right" ? 0.62 : 0.22;
        }

        if (eegModeRef.current) {
          activeHemisphere = eegTurnRef.current > 0 ? "right" : "left";
          leftPower = activeHemisphere === "left"
            ? Math.max(leftPower, 0.68)
            : Math.min(leftPower, 0.26);
          rightPower = activeHemisphere === "right"
            ? Math.max(rightPower, 0.68)
            : Math.min(rightPower, 0.26);
        }

        publishEegHud(waveSample, {
          leftPower,
          rightPower,
          confidence,
          packetRateHz: rateWindow.packetRateHz,
          activeHemisphere,
          mode: packet.mode,
        });

        setDebug((current) => ({ ...current, queueLen: playerQueueRef.current.length + enemyQueueRef.current.length }));
      },
      onSession: () => {
        // Socket connected — could trigger UI state in future
      },
      onTaunt: (payload: TauntPacket) => {
        const taunt = normalizeTaunt(payload);
        tauntEventKeyRef.current += 1;
        setActiveTaunt({
          text: taunt.text,
          audioBase64: taunt.audioBase64,
          eventKey: tauntEventKeyRef.current,
        });
      },
    });

    socket.connect(createSessionId(), "features");
    socketRef.current = socket;

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, [publishEegHud]);

  useEffect(() => {
    let alive = true;

    const loop = (nowMs: number) => {
      if (!alive) {
        return;
      }

      // Only step simulation when the game is active
      if (screenRef.current !== "game") {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      const frameDelta = Math.min(250, nowMs - lastFrameMsRef.current);
      lastFrameMsRef.current = nowMs;
      accumulatorMsRef.current += frameDelta;

      while (accumulatorMsRef.current >= FRAME_MS) {
        const currentState = stateRef.current;
        const simTimeMs = currentState.simTimeMs + FRAME_MS;

        // ── Keyboard + optional EEG turn ──
        const kbInput = sampleKeyboard(keyStateRef.current, simTimeMs);
        const mergedInput: InputSample = eegModeRef.current && eegTurnRef.current !== 0
          ? { ...kbInput, turn: eegTurnRef.current }
          : kbInput;
        if (hasActiveInput(keyStateRef.current) || (eegModeRef.current && eegTurnRef.current !== 0)) {
          playerQueueRef.current.push(mergedInput);
        }

        const deterministicEnemy = createOpponentInput(
          { x: currentState.player.x, y: currentState.player.y },
          currentState.enemy,
          currentState.difficulty,
          rngRef.current,
          simTimeMs,
        );
        enemyQueueRef.current.push(deterministicEnemy);

        const playerInput = playerQueueRef.current.drainUpTo(simTimeMs);
        const enemyInput = enemyQueueRef.current.drainUpTo(simTimeMs);
        const result = stepSimulation(
          currentState,
          {
            config,
            playerInput,
            enemyInput,
            dtMs: FRAME_MS,
            barriers: barriersRef.current,
          },
          rngRef.current,
        );

        stateRef.current = result.state;

        // Register newly-spawned dynamic obstacles as collision barriers
        const dynObs = result.state.dynamicObstacles;
        if (dynObs.length > dynBarrierCountRef.current) {
          const newObs = dynObs.slice(dynBarrierCountRef.current);
          for (const o of newObs) {
            barriersRef.current.push({
              id: o.id,
              x: o.x,
              z: o.z,
              halfW: o.width / 2,
              halfD: o.depth / 2,
              hp: 3,
            });
          }
          dynBarrierCountRef.current = dynObs.length;
          setDynamicObs([...dynObs]);
        }

        // Emit barrier break events to React state for rendering
        if (result.newBreakEvents.length > 0) {
          setBarrierBreaks((prev) => [...prev, ...result.newBreakEvents]);
        }

        // Track score change events
        const s = result.state.score;
        if (s.player !== prevScoreRef.current.player) {
          lastEventRef.current = `PlayerKill @${result.state.tick}`;
          setScore({ ...s });
        } else if (s.enemy !== prevScoreRef.current.enemy) {
          lastEventRef.current = `EnemyKill @${result.state.tick}`;
          setScore({ ...s });
        }
        prevScoreRef.current = s;

        if (stateRef.current.tick % 6 === 0) {
          setDebug({
            queueLen: playerQueueRef.current.length + enemyQueueRef.current.length,
            rttMs: 0,
            lastInput: {
              throttle: playerInput.throttle,
              turn: playerInput.turn,
              fire: playerInput.fire,
              source: hasActiveInput(keyStateRef.current) ? "keyboard" : eegModeRef.current ? "eeg" : "bci",
            },
            playerPos: {
              x: stateRef.current.player.x,
              y: stateRef.current.player.y,
              yaw: stateRef.current.player.yaw,
            },
            projectileCount: stateRef.current.projectiles.length,
            lastEvent: lastEventRef.current,
            tick: stateRef.current.tick,
          });
        }

        accumulatorMsRef.current -= FRAME_MS;
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    const applyKey = (event: KeyboardEvent, value: boolean) => {
      if (!TRACKED_KEYS.has(event.key)) return;
      event.preventDefault();
      const keys = keyStateRef.current;
      const k = event.key.toLowerCase();
      if (k === "arrowup" || k === "w") keys.up = value;
      else if (k === "arrowdown" || k === "s") keys.down = value;
      else if (k === "arrowleft" || k === "a") keys.left = value;
      else if (k === "arrowright" || k === "d") {
        keys.right = value;
        keys.reload = value; // D / ArrowRight doubles as reload (hold 2s)
      }
      else if (k === "r") keys.reload = value;
      else if (k === " ") keys.fire = value;
    };

    const onKeyDown = (event: KeyboardEvent) => applyKey(event, true);
    const onKeyUp = (event: KeyboardEvent) => applyKey(event, false);

    rafRef.current = requestAnimationFrame(loop);
    window.addEventListener("keydown", onKeyDown, { passive: false });
    window.addEventListener("keyup", onKeyUp);

    return () => {
      alive = false;
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [config]);

  return (
    <div className="relative h-screen overflow-hidden bg-neutral-950">
      {/* 5px inset arena boundary */}
      <div className="absolute inset-[5px] overflow-hidden rounded-sm border border-cyan-800/30">
        {screen === "calibration" ? (
          <CalibrationWizard
            step={calibrationStep}
            instruction={calibrationInstruction}
            progress={calibrationProgress}
            quality={calibrationQuality}
            errorMessage={calibrationError}
            running={calibrationRunning}
            onRetry={startEEGCalibration}
          />
        ) : (
          <>
            <CombatView
              stateRef={stateRef as RefObject<CombatState>}
              barrierBreaks={barrierBreaks}
              dynamicObstacles={dynamicObs}
            />

            {/* ── Scoreboard HUD ── */}
            {(screen === "game" || screen === "paused") && (
              <div className="pointer-events-none absolute top-4 left-1/2 z-20 -translate-x-1/2 select-none">
                <div className="flex items-center gap-4 rounded-xl border border-white/15 bg-black/60 px-6 py-2 backdrop-blur-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-blue-300/80">You</span>
                    <span className="min-w-[2ch] text-center text-2xl font-bold tabular-nums text-blue-400">{score.player}</span>
                  </div>
                  <span className="text-lg font-light text-white/30">:</span>
                  <div className="flex items-center gap-2">
                    <span className="min-w-[2ch] text-center text-2xl font-bold tabular-nums text-red-400">{score.enemy}</span>
                    <span className="text-xs font-semibold uppercase tracking-wider text-red-300/80">AI</span>
                  </div>
                </div>
              </div>
            )}

            {(screen === "game" || screen === "paused") && <DebugPanel debug={debug} />}

            {screen === "game" && (
              <EEGStreamModal
                waveSamples={eegHud.waveSamples}
                leftPower={eegHud.leftPower}
                rightPower={eegHud.rightPower}
                confidence={eegHud.confidence}
                packetRateHz={eegHud.packetRateHz}
                activeHemisphere={eegHud.activeHemisphere}
                mode={eegHud.mode}
              />
            )}

            {screen === "game" && (
              <TauntBubble
                text={activeTaunt?.text ?? ""}
                audioBase64={activeTaunt?.audioBase64}
                eventKey={activeTaunt?.eventKey ?? 0}
                onDismiss={() => setActiveTaunt(null)}
              />
            )}

            {/* BCI control tip — EEG mode only, minimizable */}
            {screen === "game" && eegMode && tipVisible && (
              <div className="absolute top-12 right-4 z-30 flex items-start gap-2 rounded-lg border border-cyan-700/40 bg-black/70 px-3 py-2 text-xs text-cyan-200 backdrop-blur-sm">
                <div>
                  <span className="mb-0.5 block font-semibold">BCI Controls</span>
                  Look left &rarr; Rotate CCW &nbsp;|&nbsp; Look right &rarr; Rotate CW
                </div>
                <button
                  type="button"
                  onClick={() => setTipVisible(false)}
                  className="mt-0.5 text-cyan-400 hover:text-white"
                  aria-label="Dismiss tip"
                >
                  &#x2715;
                </button>
              </div>
            )}

            {/* Paused overlay */}
            {screen === "paused" && (
              <div className="pointer-events-none absolute inset-0 z-[25] flex items-center justify-center">
                <span className="text-3xl font-bold tracking-widest text-white/70 uppercase select-none">Paused</span>
              </div>
            )}

            {/* Game control buttons — pause and quit */}
            {(screen === "game" || screen === "paused") && (
              <div className="absolute top-2 right-2 z-40 flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => { teardownMiSocket(); setScreen("menu"); }}
                  style={{ fontSize: 18 }}
                  className="rounded px-2 py-0.5 text-white/60 hover:text-white bg-black/50 hover:bg-black/80 border border-white/20 leading-none"
                  title="Quit to menu (Esc)"
                >
                  ✕
                </button>
                <button
                  type="button"
                  onClick={() => setScreen((prev) => (prev === "paused" ? "game" : "paused"))}
                  style={{ fontSize: 18 }}
                  className="rounded px-2 py-0.5 text-white/60 hover:text-white bg-black/50 hover:bg-black/80 border border-white/20 leading-none"
                  title={screen === "paused" ? "Resume (P)" : "Pause (P)"}
                >
                  {screen === "paused" ? "▶" : "⏸"}
                </button>
              </div>
            )}

            {/* Menu overlay — Manual + EEG play buttons */}
            {screen === "menu" && (
              <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                <div className="flex flex-col items-center gap-4">
                  <h1 className="mb-2 text-3xl font-bold tracking-tight text-white">Combat 3D</h1>
                  <button
                    type="button"
                    onClick={handleManualPlay}
                    className="w-48 rounded-xl border border-emerald-400/60 bg-emerald-600 px-6 py-3 text-lg font-bold uppercase tracking-wider text-white shadow-lg shadow-emerald-900/40 transition hover:bg-emerald-500 hover:shadow-emerald-700/60 active:scale-95"
                  >
                    Manual Play
                  </button>
                  <button
                    type="button"
                    onClick={startEEGCalibration}
                    className="w-48 rounded-xl border border-cyan-400/60 bg-cyan-600 px-6 py-3 text-lg font-bold uppercase tracking-wider text-white shadow-lg shadow-cyan-900/40 transition hover:bg-cyan-500 hover:shadow-cyan-700/60 active:scale-95"
                  >
                    EEG Play
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default App;
