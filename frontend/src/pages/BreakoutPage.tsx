import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_ENDPOINTS } from "@/config/api";
import { EEGStreamModal } from "@/combat3d/ui/EEGStreamModal";
import { CalibrationWizard } from "@/features/pong/components/CalibrationWizard";
import { SettingsOverlay } from "@/features/pong/components/SettingsOverlay";
import {
  createInitialRuntimeState,
  type LoopDebugPayload,
} from "@/features/breakout/game/gameLoop";
import { DebugOverlay } from "@/features/breakout/components/DebugOverlay";
import { GameCanvas } from "@/features/breakout/components/GameCanvas";
import { KeyboardHints } from "@/features/breakout/components/KeyboardHints";
import { MenuScreen } from "@/features/breakout/components/MenuScreen";
import { ScoreBoard } from "@/features/breakout/components/ScoreBoard";
import { usePongSettings } from "@/features/pong/state/usePongSettings";
import {
  deriveScoreAccentColor,
  resolveUiColorToken,
} from "@/features/pong/types/pongSettings";
import type {
  DebugStats,
  GameScreen,
  RuntimeState,
} from "@/features/breakout/types/breakoutRuntime";

type CalibrationStep = "left" | "right" | "fine_tuning" | "complete" | "error";
type PaddleCommand = "none" | "left" | "right";
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
const EEG_WAVE_POINTS = 42;
const EEG_DEFAULT_TICK_MS = 120;
const EEG_STREAM_STALE_MS = 1800;

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
  bricksRemaining: undefined,
  speedMultiplier: undefined,
};

const sleep = (ms: number) =>
  new Promise<void>((resolve) => {
    window.setTimeout(resolve, ms);
  });

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));
const clampSignal = (value: number): number => Math.max(-1, Math.min(1, value));

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

export default function BreakoutPage() {
  const { settings, updateSettings, resetSettings } = usePongSettings();

  const runtimeRef = useRef<RuntimeState>(createInitialRuntimeState());
  const debugRef = useRef<DebugStats>(INITIAL_DEBUG);
  const pausedRef = useRef(false);
  const miSocketRef = useRef<WebSocket | null>(null);
  const calibrationRunIdRef = useRef(0);
  const eegLastPacketAtRef = useRef(0);
  const eegPacketRateWindowRef = useRef({
    windowStartMs: performance.now(),
    packetCount: 0,
    packetRateHz: 0,
  });
  const eegWavePhaseRef = useRef(0);

  const [screen, setScreen] = useState<GameScreen>("menu");
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
  const [hud, setHud] = useState(() => ({
    score: runtimeRef.current.score,
    lives: runtimeRef.current.lives,
    level: runtimeRef.current.level,
  }));
  const [sessionToken, setSessionToken] = useState(0);
  const [gameOver, setGameOver] = useState(runtimeRef.current.gameOver);
  const [eegCommand, setEegCommand] = useState<PaddleCommand>("none");
  const [eegPane, setEegPane] = useState<EegPaneState>(() => createInitialEegPaneState());

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

  const teardownMiSocket = useCallback(
    (sendStop: boolean) => {
      const ws = miSocketRef.current;
      miSocketRef.current = null;
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

  const resetRuntime = useCallback(() => {
    Object.assign(runtimeRef.current, createInitialRuntimeState());
    debugRef.current = {
      ...INITIAL_DEBUG,
      thonkConnected: debugRef.current.thonkConnected,
      calibrationQuality: debugRef.current.calibrationQuality,
    };
    setDebug(debugRef.current);
    setHud({
      score: runtimeRef.current.score,
      lives: runtimeRef.current.lives,
      level: runtimeRef.current.level,
    });
    setGameOver(runtimeRef.current.gameOver);
    setSessionToken((current) => current + 1);
  }, []);

  const startKeyboard = useCallback(() => {
    teardownMiSocket(true);
    resetRuntime();
    pausedRef.current = false;
    setScreen("game");
  }, [resetRuntime, teardownMiSocket]);

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
        ws.send(JSON.stringify({ action: "start", interval_ms: 1000, reset: true }));
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
          const command = mapCommandToPaddle(payload.command);
          setEegCommand(command);

          const now = performance.now();
          eegLastPacketAtRef.current = now;

          const packetRateWindow = eegPacketRateWindowRef.current;
          packetRateWindow.packetCount += 1;
          const elapsed = now - packetRateWindow.windowStartMs;
          if (elapsed >= 1000) {
            packetRateWindow.packetRateHz = (packetRateWindow.packetCount * 1000) / elapsed;
            packetRateWindow.packetCount = 0;
            packetRateWindow.windowStartMs = now;
          }

          const rawConfidence = Number(payload.confidence ?? 0);
          const confidence = Number.isFinite(rawConfidence)
            ? clamp01(rawConfidence > 1 ? rawConfidence / 100 : rawConfidence)
            : 0;
          const activeHemisphere: EegHemisphere = command === "right" ? "right" : "left";
          const leftPower = activeHemisphere === "left"
            ? 0.56 + confidence * 0.36
            : Math.max(0.12, 0.26 - confidence * 0.08);
          const rightPower = activeHemisphere === "right"
            ? 0.56 + confidence * 0.36
            : Math.max(0.12, 0.26 - confidence * 0.08);
          const direction = activeHemisphere === "right" ? 1 : -1;
          const sample = clampSignal(
            direction * (0.36 + confidence * 0.46)
              + Math.sin(now / 95) * (0.15 + confidence * 0.32),
          );

          pushEegSample(sample, {
            leftPower,
            rightPower,
            confidence,
            packetRateHz: packetRateWindow.packetRateHz,
            activeHemisphere,
          });
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
  }, [publishDebug, pushEegSample, teardownMiSocket]);

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
      const userId = `breakout_${Date.now().toString(36)}`;
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
        pausedRef.current = false;
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
    [connectAndStartMiStreaming, publishDebug, resetRuntime, teardownMiSocket, waitWithProgress],
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

  const updateHudFromRuntime = useCallback((state: RuntimeState) => {
    setHud((previous) => {
      if (
        previous.score === state.score &&
        previous.lives === state.lives &&
        previous.level === state.level
      ) {
        return previous;
      }
      return {
        score: state.score,
        lives: state.lives,
        level: state.level,
      };
    });
    setGameOver(state.gameOver);
  }, []);

  const handleRestartGame = useCallback(() => {
    resetRuntime();
    pausedRef.current = false;
    setScreen("game");
  }, [resetRuntime]);

  const saveSettings = useCallback(
    (next: typeof settings) => {
      updateSettings(next);
    },
    [updateSettings],
  );

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
        pausedRef.current = false;
        return;
      }

      if (key === "c") {
        setOverlayOpen(true);
        return;
      }

      if (screen === "menu") {
        if (key === "k") {
          startKeyboard();
        }
        if (key === "e") {
          startEEGCalibration();
        }
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
        if (key === "r" && gameOver) {
          handleRestartGame();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    calibrationRunning,
    gameOver,
    handleRestartGame,
    overlayOpen,
    screen,
    startEEGCalibration,
    startKeyboard,
    teardownMiSocket,
  ]);

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
    }, EEG_DEFAULT_TICK_MS);

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
        bricksRemaining: payload.bricksRemaining,
        speedMultiplier: payload.speedMultiplier,
      });
    },
    [publishDebug],
  );

  return (
    <div className="relative min-h-[calc(100vh-4rem)] bg-black text-white">
      {screen === "menu" && (
        <MenuScreen
          hasSavedCalibration={false}
          onStartKeyboard={startKeyboard}
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
            key={sessionToken}
            runtimeState={runtimeRef.current}
            settings={settings}
            onFps={handleFps}
            onDebugMetrics={handleDebugMetrics}
            onRuntimeUpdate={updateHudFromRuntime}
            isPausedRef={pausedRef}
            eegCommand={eegCommand}
          />
          <ScoreBoard
            score={hud.score}
            lives={hud.lives}
            level={hud.level}
            accentColor={scoreboardAccent}
          />
          <EEGStreamModal
            waveSamples={eegPane.waveSamples}
            leftPower={eegPane.leftPower}
            rightPower={eegPane.rightPower}
            confidence={eegPane.confidence}
            packetRateHz={eegPane.packetRateHz}
            activeHemisphere={eegPane.activeHemisphere}
            mode="classifier"
          />
          {gameOver && (
            <div className="absolute inset-0 z-30 flex items-center justify-center pointer-events-none">
              <div className="rounded border border-red-400/60 bg-black/80 px-6 py-5 text-center shadow-[0_0_20px_rgba(248,113,113,0.35)]">
                <p className="text-xl font-semibold text-red-300">Game Over</p>
                <p className="mt-2 text-sm text-zinc-300">
                  Press R to restart or Esc to return to menu.
                </p>
              </div>
            </div>
          )}
          <DebugOverlay {...debug} />
          <KeyboardHints mode="game" />
          <div className="absolute top-2 right-2 z-40 flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                teardownMiSocket(true);
                setScreen("menu");
                pausedRef.current = false;
              }}
              className="rounded px-2 py-1 text-xs text-white/70 hover:text-white bg-black/50 hover:bg-black/80 border border-white/20 leading-none"
              title="Quit to menu (Esc)"
            >
              Quit
            </button>
            <button
              type="button"
              onClick={() => {
                pausedRef.current = !pausedRef.current;
                setScreen(pausedRef.current ? "paused" : "game");
              }}
              className="rounded px-2 py-1 text-xs text-white/70 hover:text-white bg-black/50 hover:bg-black/80 border border-white/20 leading-none"
              title={screen === "paused" ? "Resume (P)" : "Pause (P)"}
            >
              {screen === "paused" ? "Resume" : "Pause"}
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
