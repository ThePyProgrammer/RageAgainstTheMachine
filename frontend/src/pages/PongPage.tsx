import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_ENDPOINTS } from "@/config/api";
import { useDevice } from "@/contexts/DeviceContext";
import { CalibrationWizard } from "@/features/pong/components/CalibrationWizard";
import { DebugOverlay } from "@/features/pong/components/DebugOverlay";
import { GameCanvas } from "@/features/pong/components/GameCanvas";
import { KeyboardHints } from "@/features/pong/components/KeyboardHints";
import { MenuScreen } from "@/features/pong/components/MenuScreen";
import { ScoreBoard } from "@/features/pong/components/ScoreBoard";
import { SettingsOverlay } from "@/features/pong/components/SettingsOverlay";
import { StressMeter } from "@/features/pong/components/StressMeter";
import { TauntBubble } from "@/features/pong/components/TauntBubble";
import type { LoopDebugPayload, OpponentLoopEventPayload } from "@/features/pong/game/gameLoop";
import { usePongSettings } from "@/features/pong/state/usePongSettings";
import { deriveScoreAccentColor, resolveUiColorToken } from "@/features/pong/types/pongSettings";
import type { DebugStats, GameScreen, RuntimeState } from "@/features/pong/types/pongRuntime";
import { useAIOpponent } from "@/hooks/useAIOpponent";
import { useBCIStream } from "@/hooks/useBCIStream";
import type { OpponentInputMode } from "@/hooks/useAIOpponent";
import { EEGStreamModal } from "@/combat3d/ui/EEGStreamModal";

type CalibrationStep = "left" | "right" | "fine_tuning" | "complete" | "error";
type BallControlMode = "paddle" | "ball";
type PaddleCommand = "none" | "left" | "right";
type CaptureKeyState = "none" | "left" | "right";
type ActiveTaunt = { text: string; timestamp: number };
type SentOpponentEvent = {
  sentAtMs: number;
  event: OpponentLoopEventPayload["event"];
  score: string;
};
type CapturedKeyboardEegRow = {
  timestampMs: number;
  keyPressed: CaptureKeyState;
  channelValuesUv: number[];
};
type KeyboardMovementState = {
  left: boolean;
  right: boolean;
  up: boolean;
  down: boolean;
};
type MovementStateKey = keyof KeyboardMovementState;
type EegHemisphere = "left" | "right";

interface EegPaneState {
  waveSamples: number[];
  leftPower: number;
  rightPower: number;
  confidence: number;
  packetRateHz: number;
  activeHemisphere: EegHemisphere;
}

const LEFT_TRIAL_MS = 7000;
const RIGHT_TRIAL_MS = 7000;
const PROGRESS_TICK_MS = 120;
const DEFAULT_STRESS_LEVEL = 0.24;
const INITIAL_OPPONENT_DIFFICULTY = 0.5;
const OPPONENT_DEBUG = import.meta.env.DEV;
const EEG_WS_INTERVAL_MS = 120;
const EEG_COMMAND_HOLD_MS = 900;
const EEG_MIN_CONFIDENCE = 56;
const KEY_TIMELINE_RETENTION = 2000;

const createRuntimeState = (): RuntimeState => ({
  width: 960,
  height: 540,
  ball: { x: 480, y: 270, radius: 10 },
  topPaddle: { x: 425, y: 20, width: 110, height: 14 },
  bottomPaddle: { x: 425, y: 506, width: 110, height: 14 },
  playerScore: 0,
  aiScore: 0,
});

const INITIAL_DEBUG: DebugStats = {
  fps: 0,
  latencyMs: 0,
  thonkConnected: false,
  calibrationQuality: undefined,
  ballX: undefined,
  ballY: undefined,
  ballVX: undefined,
  ballVY: undefined,
  deltaMs: undefined,
  collisionNormals: undefined,
  positionClampedPerSecond: undefined,
  collisionResolvedPerSecond: undefined,
};

const sleep = (ms: number) =>
  new Promise<void>((resolve) => {
    window.setTimeout(resolve, ms);
  });

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));
const clampSignal = (value: number): number => Math.max(-1, Math.min(1, value));
const EEG_WAVE_POINTS= 42;
const EEG_STREAM_STALE_MS = 1800;

const createDefaultWave = (phase = 0): number[] =>
  Array.from(
    { length: EEG_WAVE_POINTS },
    (_, index) => Math.sin(phase + (index / EEG_WAVE_POINTS) * Math.PI * 3.5) * 0.16,
  );

const createInitialEegPaneState = (): EegPaneState => ({
  waveSamples: createDefaultWave(),
  leftPower: 0.58,
  rightPower: 0.2,
  confidence: 0,
  packetRateHz: 0,
  activeHemisphere: "left",
});

const mapCommandToPaddle = (command: unknown): PaddleCommand => {
  const normalized = String(command ?? "").toLowerCase();
  if (normalized === "strafe_left" || normalized === "left") {
    return "left";
  }
  if (normalized === "strafe_right" || normalized === "right") {
    return "right";
  }
  return "none";
};

const parseErrorMessage = (payload: unknown, fallback: string): string => {
  if (typeof payload === "object" && payload !== null) {
    const detail = Reflect.get(payload, "detail");
    if (typeof detail === "string" && detail.length > 0) {
      return detail;
    }
    const error = Reflect.get(payload, "error");
    if (typeof error === "string" && error.length > 0) {
      return error;
    }
    const message = Reflect.get(payload, "message");
    if (typeof message === "string" && message.length > 0) {
      return message;
    }
  }
  return fallback;
};

const toMovementStateKey = (key: string): MovementStateKey | null => {
  const normalized = key.toLowerCase();
  if (normalized === "arrowleft" || normalized === "a") {
    return "left";
  }
  if (normalized === "arrowright" || normalized === "d") {
    return "right";
  }
  if (normalized === "arrowup" || normalized === "w") {
    return "up";
  }
  if (normalized === "arrowdown" || normalized === "s") {
    return "down";
  }
  return null;
};

const resolveCaptureDirection = (movementState: KeyboardMovementState): CaptureKeyState => {
  const leftPressed = movementState.left || movementState.up;
  const rightPressed = movementState.right || movementState.down;
  if (leftPressed === rightPressed) {
    return "none";
  }
  return leftPressed ? "left" : "right";
};

async function postJson<TResponse>(
  url: string,
  body?: Record<string, unknown>,
): Promise<TResponse> {
  const response = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  const payload = await response.json().catch(() => undefined);
  if (!response.ok) {
    throw new Error(parseErrorMessage(payload, `Request failed with status ${response.status}`));
  }
  return payload as TResponse;
}

async function ensureEegStreamRunning(): Promise<void> {
  const statusResponse = await fetch(API_ENDPOINTS.BCI_STATUS);
  if (statusResponse.ok) {
    const statusPayload = (await statusResponse.json().catch(() => undefined)) as
      | { is_streaming?: boolean }
      | undefined;
    if (statusPayload?.is_streaming) {
      return;
    }
  }

  const startResponse = await fetch(API_ENDPOINTS.BCI_START, { method: "POST" });
  const startPayload = await startResponse.json().catch(() => undefined);
  if (!startResponse.ok) {
    const message = parseErrorMessage(
      startPayload,
      `Failed to start EEG stream (${startResponse.status})`,
    );
    if (!message.toLowerCase().includes("already_running")) {
      throw new Error(message);
    }
  }
}

export default function PongPage() {
  const { settings, updateSettings, resetSettings } = usePongSettings();
  const { latestUpdate, latestFeedbackUpdate, sendGameEvent, playLatestAudio } = useAIOpponent();
  const { displayData, sampleCount, isStreaming } = useBCIStream();
  const { deviceConfig, deviceType } = useDevice();

  const [screen, setScreen] = useState<GameScreen>("menu");
  const [ballControlMode, setBallControlMode] = useState<BallControlMode>("paddle");
  const [inputMode, setInputMode] = useState<OpponentInputMode>("keyboard_paddle");
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [calibrationStep, setCalibrationStep] = useState<CalibrationStep>("left");
  const [calibrationInstruction, setCalibrationInstruction] = useState(
    "Look left to begin calibration.",
  );
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationQuality, setCalibrationQuality] = useState<number | undefined>(undefined);
  const [calibrationError, setCalibrationError] = useState<string | undefined>(undefined);
  const [calibrationRunning, setCalibrationRunning] = useState(false);
  const [debug, setDebug] = useState<DebugStats>(INITIAL_DEBUG);
  const [scores, setScores] = useState({ player: 0, ai: 0 });
  const [eegCommand, setEegCommand] = useState<PaddleCommand>("none");
  const [stressLevel, setStressLevel] = useState(DEFAULT_STRESS_LEVEL);
  const [activeTaunt, setActiveTaunt] = useState<ActiveTaunt | null>(null);
  const [isKeyboardEegCapturing, setIsKeyboardEegCapturing] = useState(false);
  const [capturedKeyboardEegSamples, setCapturedKeyboardEegSamples] = useState(0);
  const [captureNotice, setCaptureNotice] = useState<string | null>(null);
  const [captureBusy, setCaptureBusy] = useState(false);
  const [eegPane, setEegPane] = useState<EegPaneState>(() => createInitialEegPaneState());

  const debugRef = useRef<DebugStats>(INITIAL_DEBUG);
  const runtimeRef = useRef<RuntimeState>(createRuntimeState());
  const pausedRef = useRef(false);
  const miSocketRef = useRef<WebSocket | null>(null);
  const eegHoldTimerRef = useRef<number | null>(null);
  const eegCommandHoldUntilRef = useRef(0);
  const calibrationRunIdRef = useRef(0);
  const opponentDifficultyRef = useRef(INITIAL_OPPONENT_DIFFICULTY);
  const opponentEventCounterRef = useRef(0);
  const processedUpdateKeyRef = useRef<string | null>(null);
  const processedFeedbackUpdateKeyRef = useRef<string | null>(null);
  const sentOpponentEventsRef = useRef<Map<string, SentOpponentEvent>>(new Map());
  const keyboardEegRowsRef = useRef<CapturedKeyboardEegRow[]>([]);
  const lastCapturedSampleCountRef = useRef(0);
  const movementStateRef = useRef<KeyboardMovementState>({
    left: false,
    right: false,
    up: false,
    down: false,
  });
  const captureDirectionRef = useRef<CaptureKeyState>("none");
  const captureDirectionTimelineRef = useRef<
    Array<{ timestampMs: number; keyPressed: CaptureKeyState }>
  >([]);
  const captureDirectionIndexRef = useRef(0);
  const isCapturingRef = useRef(false);
  const eegLastPacketAtRef = useRef(0);
  const eegPacketRateWindowRef = useRef({
    windowStartMs: performance.now(),
    packetCount: 0,
    packetRateHz: 0,
  });
  const eegWavePhaseRef = useRef(0);

  const logOpponent = useCallback((label: string, payload: Record<string, unknown>) => {
    if (!OPPONENT_DEBUG) {
      return;
    }
    console.debug(`[pong-opponent] ${label}`, payload);
  }, []);

  const publishDebug = useCallback((next: Partial<DebugStats>) => {
    debugRef.current = {
      ...debugRef.current,
      ...next,
    };
  }, []);

  const pushEegSample = useCallback(
    (
      sample: number,
      patch: Partial<Omit<EegPaneState, "waveSamples">> = {},
    ) => {
      setEegPane((current) => ({
        ...current,
        ...patch,
        waveSamples: [
          ...current.waveSamples.slice(-(EEG_WAVE_POINTS - 1)),
          clampSignal(sample),
        ],
      }));
    },
    [],
  );

  const resetEegPane = useCallback(() => {
    eegLastPacketAtRef.current = 0;
    eegPacketRateWindowRef.current = {
      windowStartMs: performance.now(),
      packetCount: 0,
      packetRateHz: 0,
    };
    eegWavePhaseRef.current = 0;
    setEegPane(createInitialEegPaneState());
  }, []);

  const resetOpponentFeedback = useCallback(() => {
    opponentDifficultyRef.current = INITIAL_OPPONENT_DIFFICULTY;
    setStressLevel(DEFAULT_STRESS_LEVEL);
    setActiveTaunt(null);
    processedUpdateKeyRef.current = null;
    processedFeedbackUpdateKeyRef.current = null;
  }, []);

  const downloadCaptureCsv = useCallback(
    (rows: CapturedKeyboardEegRow[]) => {
      if (rows.length === 0) {
        setCaptureNotice("Capture stopped. No EEG samples were recorded.");
        return;
      }

      const header = [
        "timestamp_ms",
        "key_pressed",
        ...deviceConfig.channelNames.map((channelName) => `ch${channelName}_raw_uv`),
      ];
      const csvRows = rows.map((row) =>
        [
          String(row.timestampMs),
          row.keyPressed,
          ...row.channelValuesUv.map((value) => value.toFixed(6)),
        ].join(","),
      );
      const csvContent = [header.join(","), ...csvRows].join("\n");
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const dateStamp = new Date().toISOString().replace(/[:.]/g, "-");
      link.href = url;
      link.download = `pong_keyboard_eeg_capture_${deviceType}_${dateStamp}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setCaptureNotice(`Downloaded ${rows.length} EEG samples.`);
    },
    [deviceConfig.channelNames, deviceType],
  );

  const stopCaptureAndDownload = useCallback(() => {
    if (!isCapturingRef.current) {
      return;
    }

    const rows = keyboardEegRowsRef.current;
    isCapturingRef.current = false;
    setIsKeyboardEegCapturing(false);
    setCapturedKeyboardEegSamples(rows.length);
    keyboardEegRowsRef.current = [];
    lastCapturedSampleCountRef.current = sampleCount;
    captureDirectionTimelineRef.current = [];
    captureDirectionIndexRef.current = 0;
    downloadCaptureCsv(rows);
  }, [downloadCaptureCsv, sampleCount]);

  const startCapture = useCallback(async () => {
    if (isCapturingRef.current || captureBusy) {
      return;
    }

    setCaptureBusy(true);
    setCaptureNotice(null);

    try {
      await ensureEegStreamRunning();
      keyboardEegRowsRef.current = [];
      lastCapturedSampleCountRef.current = sampleCount;
      captureDirectionTimelineRef.current = [
        {
          timestampMs: Date.now(),
          keyPressed: captureDirectionRef.current,
        },
      ];
      captureDirectionIndexRef.current = 0;
      setCapturedKeyboardEegSamples(0);
      isCapturingRef.current = true;
      setIsKeyboardEegCapturing(true);
    } catch (error) {
      setCaptureNotice(
        error instanceof Error ? error.message : "Unable to start EEG capture.",
      );
    } finally {
      setCaptureBusy(false);
    }
  }, [captureBusy, sampleCount]);

  const teardownMiSocket = useCallback(
    (sendStop: boolean) => {
      const ws = miSocketRef.current;
      miSocketRef.current = null;
      if (eegHoldTimerRef.current !== null) {
        window.clearTimeout(eegHoldTimerRef.current);
        eegHoldTimerRef.current = null;
      }
      eegCommandHoldUntilRef.current = 0;
      if (ws && ws.readyState === WebSocket.OPEN && sendStop) {
        ws.send(JSON.stringify({ action: "stop" }));
      }
      if (ws && ws.readyState !== WebSocket.CLOSED) {
        ws.close();
      }
      setEegCommand("none");
      resetEegPane();
      publishDebug({ thonkConnected: false });
    },
    [publishDebug, resetEegPane],
  );

  const applyEegCommand = useCallback((command: PaddleCommand) => {
    setEegCommand(command);
    const holdUntil = performance.now() + EEG_COMMAND_HOLD_MS;
    eegCommandHoldUntilRef.current = holdUntil;
    if (eegHoldTimerRef.current !== null) {
      window.clearTimeout(eegHoldTimerRef.current);
    }
    eegHoldTimerRef.current = window.setTimeout(() => {
      if (performance.now() >= eegCommandHoldUntilRef.current) {
        setEegCommand("none");
      }
      eegHoldTimerRef.current = null;
    }, EEG_COMMAND_HOLD_MS + 40);
  }, []);

  const resetRuntime = useCallback(() => {
    Object.assign(runtimeRef.current, createRuntimeState());
    debugRef.current = {
      ...INITIAL_DEBUG,
      thonkConnected: debugRef.current.thonkConnected,
      calibrationQuality: debugRef.current.calibrationQuality,
    };
    setDebug(debugRef.current);
    setScores({ player: 0, ai: 0 });
  }, []);

  useEffect(() => {
    if (!latestUpdate) {
      return;
    }

    const updateKey = `${latestUpdate.event_id}:${latestUpdate.timestamp_ms}`;
    if (processedUpdateKeyRef.current === updateKey) {
      return;
    }
    processedUpdateKeyRef.current = updateKey;

    opponentDifficultyRef.current = latestUpdate.difficulty.final;
    setStressLevel(latestUpdate.metrics.stress);

    const sentMeta = sentOpponentEventsRef.current.get(latestUpdate.event_id);
    const now = performance.now();
    const rttMs = sentMeta ? now - sentMeta.sentAtMs : null;
    sentOpponentEventsRef.current.delete(latestUpdate.event_id);

    logOpponent("recv_update", {
      eventId: latestUpdate.event_id,
      event: sentMeta?.event ?? "unknown",
      score: sentMeta?.score ?? "unknown",
      provider: latestUpdate.meta.provider,
      serverLatencyMs: latestUpdate.meta.latency_ms,
      metricsAgeMs: latestUpdate.meta.metrics_age_ms,
      clientRttMs: rttMs !== null ? Number(rttMs.toFixed(2)) : null,
      tauntChars: latestUpdate.taunt_text.length,
      audioBytesEstimate: latestUpdate.speech.audio_base64
        ? Math.floor((latestUpdate.speech.audio_base64.length * 3) / 4)
        : 0,
    });
  }, [latestUpdate, logOpponent]);

  useEffect(() => {
    if (!latestFeedbackUpdate) {
      return;
    }

    const updateKey = `${latestFeedbackUpdate.event_id}:${latestFeedbackUpdate.timestamp_ms}`;
    if (processedFeedbackUpdateKeyRef.current === updateKey) {
      return;
    }
    processedFeedbackUpdateKeyRef.current = updateKey;

    if (latestFeedbackUpdate.taunt_text.trim()) {
      setActiveTaunt({
        text: latestFeedbackUpdate.taunt_text,
        timestamp: latestFeedbackUpdate.timestamp_ms,
      });
    }

    if (!latestFeedbackUpdate.speech.audio_base64) {
      logOpponent("audio_playback_skipped", {
        eventId: latestFeedbackUpdate.event_id,
        reason: "empty_audio_payload",
      });
      return;
    }

    void (async () => {
      const played = await playLatestAudio(latestFeedbackUpdate.event_id);
      logOpponent("audio_playback", {
        eventId: latestFeedbackUpdate.event_id,
        played,
        provider: latestFeedbackUpdate.meta.provider,
      });
    })();
  }, [latestFeedbackUpdate, logOpponent, playLatestAudio]);

  const handleOpponentEvent = useCallback(
    (payload: OpponentLoopEventPayload) => {
      if (screen !== "game") {
        return;
      }

      opponentEventCounterRef.current += 1;
      const eventId = `pong-${Date.now()}-${opponentEventCounterRef.current}`;
      const scoreText = `${payload.score.player}-${payload.score.ai}`;
      sentOpponentEventsRef.current.set(eventId, {
        sentAtMs: performance.now(),
        event: payload.event,
        score: scoreText,
      });
      if (sentOpponentEventsRef.current.size > 256) {
        const oldestKey = sentOpponentEventsRef.current.keys().next().value;
        if (oldestKey) {
          sentOpponentEventsRef.current.delete(oldestKey);
        }
      }

      logOpponent("send_event", {
        eventId,
        event: payload.event,
        score: scoreText,
        inputMode,
        difficulty: Number(opponentDifficultyRef.current.toFixed(3)),
        nearSide: payload.event_context?.near_side ?? null,
        proximity: payload.event_context?.proximity ?? null,
      });

      sendGameEvent({
        event_id: eventId,
        game_mode: "pong",
        input_mode: inputMode,
        event: payload.event,
        score: payload.score,
        current_difficulty: opponentDifficultyRef.current,
        event_context: payload.event_context,
      });
    },
    [inputMode, logOpponent, screen, sendGameEvent],
  );

  const startKeyboard = useCallback(
    (mode: BallControlMode) => {
      teardownMiSocket(true);
      resetRuntime();
      resetOpponentFeedback();
      setCaptureNotice(null);
      setCapturedKeyboardEegSamples(0);
      pausedRef.current = false;
      setBallControlMode(mode);
      setInputMode(mode === "ball" ? "keyboard_ball" : "keyboard_paddle");
      setScreen("game");
    },
    [resetOpponentFeedback, resetRuntime, teardownMiSocket],
  );

  const connectAndStartMiStreaming = useCallback(async () => {
    teardownMiSocket(false);

    await new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(API_ENDPOINTS.MI_WS);
      miSocketRef.current = ws;
      let settled = false;

      const timeoutId = window.setTimeout(() => {
        if (settled) {
          return;
        }
        settled = true;
        ws.close();
        reject(new Error("MI websocket start timed out."));
      }, 8000);

      ws.onopen = () => {
        ws.send(JSON.stringify({ action: "start", interval_ms: EEG_WS_INTERVAL_MS, reset: true }));
      };

      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data) as {
          status?: string;
          type?: string;
          error?: string;
          command?: string;
          confidence?: number;
        };

        if (payload.error) {
          if (!settled) {
            settled = true;
            window.clearTimeout(timeoutId);
            reject(new Error(payload.error));
          }
          return;
        }

        if (payload.status === "started") {
          publishDebug({ thonkConnected: true });
          if (!settled) {
            settled = true;
            window.clearTimeout(timeoutId);
            resolve();
          }
          return;
        }

        if (payload.type === "prediction") {
          const nextCommand = mapCommandToPaddle(payload.command);
          const confidence = Number(payload.confidence ?? 0);

          if (nextCommand !== "none") {
            if (Number.isFinite(confidence) && confidence > 0 && confidence < EEG_MIN_CONFIDENCE) {
              return;
            }
            applyEegCommand(nextCommand);
            return;
          }

          if (performance.now() >= eegCommandHoldUntilRef.current) {
            setEegCommand("none");
          }
        }
      };

      ws.onerror = () => {
        publishDebug({ thonkConnected: false });
        if (!settled) {
          settled = true;
          window.clearTimeout(timeoutId);
          reject(new Error("MI websocket connection failed."));
        }
      };

      ws.onclose = () => {
        publishDebug({ thonkConnected: false });
        setEegCommand("none");
        if (!settled) {
          settled = true;
          window.clearTimeout(timeoutId);
          reject(new Error("MI websocket closed before streaming started."));
        }
      };
    });
  }, [publishDebug, teardownMiSocket]);

  const waitWithProgress = useCallback(async (durationMs: number, start: number, end: number) => {
    const startedAt = performance.now();
    while (true) {
      const elapsed = performance.now() - startedAt;
      const ratio = Math.min(1, elapsed / durationMs);
      setCalibrationProgress(start + (end - start) * ratio);
      if (ratio >= 1) {
        break;
      }
      await sleep(PROGRESS_TICK_MS);
    }
  }, []);

  const runCalibrationRound = useCallback(
    async (runId: number) => {
      const userId = `pong_${Date.now().toString(36)}`;
      let calibrationSessionOpen = false;
      let calibrationSessionDir: string | undefined;
      const isActive = () => calibrationRunIdRef.current === runId;

      setCalibrationRunning(true);
      setCalibrationError(undefined);
      setCalibrationQuality(undefined);
      setCalibrationProgress(0.02);

      try {
        await ensureEegStreamRunning();

        await postJson<{ save_dir: string }>(API_ENDPOINTS.MI_CALIBRATION_START, {
          user_id: userId,
        });
        calibrationSessionOpen = true;
        if (!isActive()) {
          return;
        }

        setCalibrationStep("left");
        setCalibrationInstruction("Look left and hold focus.");
        await postJson(API_ENDPOINTS.MI_CALIBRATION_TRIAL_START, { label: 0 });
        await waitWithProgress(LEFT_TRIAL_MS, 0.05, 0.45);
        await postJson(API_ENDPOINTS.MI_CALIBRATION_TRIAL_END);
        if (!isActive()) {
          return;
        }

        setCalibrationStep("right");
        setCalibrationInstruction("Look right and hold focus.");
        await postJson(API_ENDPOINTS.MI_CALIBRATION_TRIAL_START, { label: 1 });
        await waitWithProgress(RIGHT_TRIAL_MS, 0.45, 0.85);
        await postJson(API_ENDPOINTS.MI_CALIBRATION_TRIAL_END);
        if (!isActive()) {
          return;
        }

        const calibrationEnd = await postJson<{ session_dir: string }>(API_ENDPOINTS.MI_CALIBRATION_END);
        calibrationSessionDir = calibrationEnd.session_dir;
        calibrationSessionOpen = false;

        setCalibrationStep("fine_tuning");
        setCalibrationInstruction("Fine-tuning lightweight left/right classifier...");
        setCalibrationProgress(0.9);

        await postJson(API_ENDPOINTS.MI_FINETUNE_PREPARE, {
          user_id: userId,
          session_dir: calibrationSessionDir,
        });
        if (!isActive()) {
          return;
        }

        const runResponse = await postJson<{
          summary?: { best_val_acc?: number };
        }>(API_ENDPOINTS.MI_FINETUNE_RUN, {
          n_epochs: 12,
          batch_size: 8,
          val_split: 0.25,
        });
        if (!isActive()) {
          return;
        }

        await postJson(API_ENDPOINTS.MI_FINETUNE_SAVE, { user_id: userId });

        const bestValAcc = Number(runResponse.summary?.best_val_acc ?? 0);
        const boundedQuality = Number.isFinite(bestValAcc)
          ? Math.max(0, Math.min(1, bestValAcc))
          : 0;
        setCalibrationQuality(boundedQuality);
        publishDebug({ calibrationQuality: boundedQuality });

        setCalibrationStep("complete");
        setCalibrationInstruction("Calibration complete. Starting EEG control...");
        setCalibrationProgress(1);

        await connectAndStartMiStreaming();
        if (!isActive()) {
          teardownMiSocket(true);
          return;
        }

        resetRuntime();
        resetOpponentFeedback();
        pausedRef.current = false;
        setBallControlMode("paddle");
        setInputMode("eeg");
        setScreen("game");
        setCalibrationRunning(false);
      } catch (error) {
        if (!isActive()) {
          return;
        }
        setCalibrationStep("error");
        setCalibrationRunning(false);
        setCalibrationError(
          error instanceof Error ? error.message : "Calibration failed. Please retry.",
        );
        setCalibrationInstruction(
          "Unable to calibrate EEG signal. Check headset contact and stream health.",
        );
      } finally {
        if (calibrationSessionOpen) {
          await postJson(API_ENDPOINTS.MI_CALIBRATION_END).catch(() => undefined);
        }
      }
    },
    [connectAndStartMiStreaming, publishDebug, resetOpponentFeedback, resetRuntime, teardownMiSocket, waitWithProgress],
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

  const updateScoresFromRuntime = useCallback((state: RuntimeState) => {
    setScores((previous) => {
      if (previous.player === state.playerScore && previous.ai === state.aiScore) {
        return previous;
      }
      return { player: state.playerScore, ai: state.aiScore };
    });
  }, []);

  const saveSettings = useCallback(
    (next: typeof settings) => {
      updateSettings(next);
    },
    [settings, updateSettings],
  );

  const expireTaunt = useCallback((timestamp: number) => {
    setActiveTaunt((current) => {
      if (!current || current.timestamp !== timestamp) {
        return current;
      }
      return null;
    });
  }, []);

  useEffect(() => {
    const updateMovementState = (event: KeyboardEvent, isPressed: boolean) => {
      const movementKey = toMovementStateKey(event.key);
      if (!movementKey) {
        return;
      }

      if (movementStateRef.current[movementKey] === isPressed) {
        return;
      }

      movementStateRef.current[movementKey] = isPressed;
      const nextDirection = resolveCaptureDirection(movementStateRef.current);
      if (nextDirection === captureDirectionRef.current) {
        return;
      }

      captureDirectionRef.current = nextDirection;
      if (!isCapturingRef.current) {
        return;
      }

      captureDirectionTimelineRef.current.push({
        timestampMs: Date.now(),
        keyPressed: nextDirection,
      });

      if (captureDirectionTimelineRef.current.length <= KEY_TIMELINE_RETENTION * 2) {
        return;
      }

      const removeCount = captureDirectionTimelineRef.current.length - KEY_TIMELINE_RETENTION;
      captureDirectionTimelineRef.current.splice(0, removeCount);
      captureDirectionIndexRef.current = Math.max(0, captureDirectionIndexRef.current - removeCount);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      updateMovementState(event, true);
    };

    const onKeyUp = (event: KeyboardEvent) => {
      updateMovementState(event, false);
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (key === "escape") {
        if (overlayOpen) {
          setOverlayOpen(false);
          return;
        }
        if (screen === "calibration") {
          calibrationRunIdRef.current += 1;
          setCalibrationRunning(false);
        }
        if (screen === "game" || screen === "paused") {
          teardownMiSocket(true);
        }
        setScreen("menu");
        setActiveTaunt(null);
        pausedRef.current = false;
        return;
      }

      if (key === "c") {
        setOverlayOpen(true);
        return;
      }

      if (screen === "menu") {
        if (key === "k") startKeyboard("paddle");
        if (key === "b") startKeyboard("ball");
        if (key === "e") startEEGCalibration();
        return;
      }

      if (screen === "calibration") {
        if (!calibrationRunning && key === "r") {
          startEEGCalibration();
        }
        return;
      }

      if (screen === "game" || screen === "paused") {
        if (key === "p") {
          pausedRef.current = !pausedRef.current;
          setScreen(pausedRef.current ? "paused" : "game");
          return;
        }
        if (key === "o") {
          setOverlayOpen(true);
          return;
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [calibrationRunning, overlayOpen, screen, startEEGCalibration, startKeyboard, teardownMiSocket]);

  useEffect(() => {
    if (!isKeyboardEegCapturing) {
      return;
    }

    const pendingSampleCount = sampleCount - lastCapturedSampleCountRef.current;
    if (pendingSampleCount <= 0) {
      return;
    }

    const availableSampleCount = Math.min(pendingSampleCount, displayData.length);
    if (availableSampleCount <= 0) {
      lastCapturedSampleCountRef.current = sampleCount;
      return;
    }

    const droppedSampleCount = pendingSampleCount - availableSampleCount;
    const latestPoints = displayData.slice(-availableSampleCount);
    const directionTimeline = captureDirectionTimelineRef.current;
    let directionIndex = captureDirectionIndexRef.current;

    const capturedRows = latestPoints.map((point) => {
      const timestampMs =
        typeof point.timestampMs === "number" && Number.isFinite(point.timestampMs)
          ? point.timestampMs
          : Date.now();

      while (
        directionIndex + 1 < directionTimeline.length &&
        directionTimeline[directionIndex + 1].timestampMs <= timestampMs
      ) {
        directionIndex += 1;
      }

      const keyPressed = directionTimeline[directionIndex]?.keyPressed ?? captureDirectionRef.current;
      const channelValuesUv = deviceConfig.channelNames.map((channelName) => {
        const rawValue = point[`ch${channelName}`];
        return typeof rawValue === "number" && Number.isFinite(rawValue) ? rawValue : 0;
      });

      return {
        timestampMs,
        keyPressed,
        channelValuesUv,
      };
    });

    captureDirectionIndexRef.current = directionIndex;
    keyboardEegRowsRef.current.push(...capturedRows);
    setCapturedKeyboardEegSamples((previous) => previous + capturedRows.length);
    lastCapturedSampleCountRef.current = sampleCount;

    if (droppedSampleCount > 0) {
      setCaptureNotice(`Dropped ${droppedSampleCount} EEG samples from the capture window.`);
    }
  }, [deviceConfig.channelNames, displayData, isKeyboardEegCapturing, sampleCount]);

  useEffect(() => {
    const isManualKeyboardMode = inputMode === "keyboard_paddle";
    const inGameSession = screen === "game" || screen === "paused";
    if (!isKeyboardEegCapturing || (isManualKeyboardMode && inGameSession)) {
      return;
    }
    stopCaptureAndDownload();
  }, [inputMode, isKeyboardEegCapturing, screen, stopCaptureAndDownload]);

  useEffect(() => {
    if (screen !== "game" && screen !== "paused") {
      return;
    }
    const interval = window.setInterval(() => {
      setDebug({ ...debugRef.current });
    }, 120);
    return () => clearInterval(interval);
  }, [screen]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (screen !== "game" && screen !== "paused") {
        return;
      }

      const now = performance.now();
      if (now - eegLastPacketAtRef.current <= EEG_STREAM_STALE_MS) {
        return;
      }

      eegWavePhaseRef.current += 0.34;
      const phase = eegWavePhaseRef.current;
      const sample = Math.sin(phase) * 0.21 + Math.sin(phase * 0.45) * 0.09;

      pushEegSample(sample, {
        leftPower: 0.58,
        rightPower: 0.2,
        confidence: 0,
        packetRateHz: 0,
        activeHemisphere: "left",
      });
    }, 120);

    return () => window.clearInterval(interval);
  }, [pushEegSample, screen]);

  useEffect(() => {
    return () => {
      teardownMiSocket(true);
    };
  }, [teardownMiSocket]);

  const scoreboardAccent = useMemo(
    () => resolveUiColorToken(deriveScoreAccentColor(settings.theme)),
    [settings.theme],
  );

  const handleFps = useCallback(
    (fps: number) => {
      publishDebug({
        fps,
        latencyMs: 1000 / (fps || 1),
      });
    },
    [publishDebug],
  );

  const handleDebugMetrics = useCallback(
    (payload: LoopDebugPayload) => {
      publishDebug({
        ballX: payload.ballX,
        ballY: payload.ballY,
        ballVX: payload.ballVX,
        ballVY: payload.ballVY,
        deltaMs: payload.deltaMs,
        collisionNormals: payload.collisionNormals,
        collisionResolvedPerSecond: payload.collisionResolvedPerSecond,
        positionClampedPerSecond: payload.positionClampedPerSecond,
      });
    },
    [publishDebug],
  );

  const showCaptureControls =
    (screen === "game" || screen === "paused") && inputMode === "keyboard_paddle";

  return (
    <div className="relative min-h-[calc(100vh-4rem)] bg-black text-white">
      {screen === "menu" && (
        <MenuScreen
          hasSavedCalibration={false}
          onStartKeyboard={() => startKeyboard("paddle")}
          onStartBallMode={() => startKeyboard("ball")}
          onStartEEG={startEEGCalibration}
          onStartWithSavedCalibration={startEEGCalibration}
          onCustomize={() => setOverlayOpen(true)}
        />
      )}

      {screen === "calibration" && (
        <CalibrationWizard
          step={calibrationStep}
          instruction={calibrationInstruction}
          progress={calibrationProgress}
          quality={calibrationQuality}
          errorMessage={calibrationError}
          running={calibrationRunning}
          onRetry={startEEGCalibration}
          onContinue={() => setScreen("menu")}
        />
      )}

      {(screen === "game" || screen === "paused") && (
        <section className="relative w-full h-full">
          <GameCanvas
            runtimeState={runtimeRef.current}
            settings={settings}
            onFps={handleFps}
            onDebugMetrics={handleDebugMetrics}
            onRuntimeUpdate={updateScoresFromRuntime}
            onOpponentEvent={handleOpponentEvent}
            isPausedRef={pausedRef}
            controlMode={ballControlMode}
            eegCommand={ballControlMode === "paddle" ? eegCommand : "none"}
            difficultyRef={opponentDifficultyRef}
          />
          <ScoreBoard
            playerScore={scores.player}
            aiScore={scores.ai}
            accentColor={scoreboardAccent}
          />
          <StressMeter stressLevel={stressLevel} />
          <EEGStreamModal
            waveSamples={eegPane.waveSamples}
            leftPower={eegPane.leftPower}
            rightPower={eegPane.rightPower}
            confidence={eegPane.confidence}
            packetRateHz={eegPane.packetRateHz}
            activeHemisphere={eegPane.activeHemisphere}
            mode="classifier"
          />
          {activeTaunt && (
            <TauntBubble
              text={activeTaunt.text}
              durationMs={3000}
              timestamp={activeTaunt.timestamp}
              onExpire={() => expireTaunt(activeTaunt.timestamp)}
            />
          )}
          <DebugOverlay {...debug} />
          <KeyboardHints mode="game" />
          {showCaptureControls && (
            <div className="fixed right-3 top-3 z-30 rounded border border-cyan-400/40 bg-black/90 p-3 text-xs text-cyan-100 shadow-[0_0_18px_rgba(34,211,238,0.2)]">
              <p className="font-mono text-[11px] uppercase tracking-wide text-cyan-200">
                Manual EEG Capture
              </p>
              <p className="mt-1 font-mono tabular-nums text-cyan-100">
                {capturedKeyboardEegSamples} samples
              </p>
              <p className="font-mono text-[11px] text-cyan-300/90">
                Stream: {isStreaming ? "on" : "off"}
              </p>
              <button
                type="button"
                onClick={
                  isKeyboardEegCapturing
                    ? stopCaptureAndDownload
                    : () => {
                        void startCapture();
                      }
                }
                disabled={captureBusy}
                className="mt-2 w-full rounded border border-cyan-300/60 px-2 py-1.5 font-semibold text-cyan-100 transition hover:border-cyan-200 hover:bg-cyan-400/10 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isKeyboardEegCapturing
                  ? "Stop Recording & Download CSV"
                  : captureBusy
                    ? "Starting..."
                    : "Start Recording EEG CSV"}
              </button>
              {captureNotice && (
                <p className="mt-2 max-w-60 text-[11px] text-cyan-200/80">{captureNotice}</p>
              )}
            </div>
          )}
          <div className="absolute top-2 right-2 z-40 flex items-center gap-1">
            <button
              type="button"
              onClick={() => {
                teardownMiSocket(true);
                setScreen("menu");
                pausedRef.current = false;
              }}
              style={{ fontSize: 18 }}
              className="rounded px-2 py-0.5 text-white/60 hover:text-white bg-black/50 hover:bg-black/80 border border-white/20 leading-none"
              title="Quit to menu (Esc)"
            >
              ✕
            </button>
            <button
              type="button"
              onClick={() => {
                pausedRef.current = !pausedRef.current;
                setScreen(pausedRef.current ? "paused" : "game");
              }}
              style={{ fontSize: 18 }}
              className="rounded px-2 py-0.5 text-white/60 hover:text-white bg-black/50 hover:bg-black/80 border border-white/20 leading-none"
              title={screen === "paused" ? "Resume (P)" : "Pause (P)"}
            >
              {screen === "paused" ? "▶" : "⏸"}
            </button>
          </div>
        </section>
      )}

      {(screen === "menu" || screen === "calibration") && (
        <KeyboardHints mode={screen === "menu" ? "menu" : "game"} />
      )}

      {overlayOpen && (
        <SettingsOverlay
          open={overlayOpen}
          settings={settings}
          onClose={() => setOverlayOpen(false)}
          onChange={saveSettings}
          onReset={resetSettings}
        />
      )}
    </div>
  );
}
