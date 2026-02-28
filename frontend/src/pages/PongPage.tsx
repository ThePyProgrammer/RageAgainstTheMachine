import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_ENDPOINTS } from "@/config/api";
import { CalibrationWizard } from "@/features/pong/components/CalibrationWizard";
import { DebugOverlay } from "@/features/pong/components/DebugOverlay";
import { GameCanvas } from "@/features/pong/components/GameCanvas";
import { KeyboardHints } from "@/features/pong/components/KeyboardHints";
import { MenuScreen } from "@/features/pong/components/MenuScreen";
import { ScoreBoard } from "@/features/pong/components/ScoreBoard";
import { SettingsOverlay } from "@/features/pong/components/SettingsOverlay";
import type { LoopDebugPayload } from "@/features/pong/game/gameLoop";
import { usePongSettings } from "@/features/pong/state/usePongSettings";
import { deriveScoreAccentColor, resolveUiColorToken } from "@/features/pong/types/pongSettings";
import type { DebugStats, GameScreen, RuntimeState } from "@/features/pong/types/pongRuntime";

type CalibrationStep = "left" | "right" | "fine_tuning" | "complete" | "error";
type BallControlMode = "paddle" | "ball";
type PaddleCommand = "none" | "left" | "right";

const LEFT_TRIAL_MS = 7000;
const RIGHT_TRIAL_MS = 7000;
const PROGRESS_TICK_MS = 120;

const createRuntimeState = (): RuntimeState => ({
  width: 960,
  height: 540,
  ball: { x: 480, y: 270, radius: 10 },
  topPaddle: { x: 425, y: 20, width: 110, height: 14 },      // Player at top
  bottomPaddle: { x: 425, y: 506, width: 110, height: 14 },  // AI at bottom (540-14-20)
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

const mapCommandToPaddle = (command: unknown): PaddleCommand => {
  const normalized = String(command ?? "").toLowerCase();
  if (normalized === "strafe_left" || normalized === "left") {
    return "left";  // EEG left -> paddle left
  }
  if (normalized === "strafe_right" || normalized === "right") {
    return "right"; // EEG right -> paddle right
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

export default function PongPage() {
  const { settings, updateSettings, resetSettings } = usePongSettings();
  const [screen, setScreen] = useState<GameScreen>("menu");
  const [ballControlMode, setBallControlMode] = useState<BallControlMode>("paddle");
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [calibrationStep, setCalibrationStep] = useState<CalibrationStep>("left");
  const [calibrationInstruction, setCalibrationInstruction] = useState("Look left to begin calibration.");
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationQuality, setCalibrationQuality] = useState<number | undefined>(undefined);
  const [calibrationError, setCalibrationError] = useState<string | undefined>(undefined);
  const [calibrationRunning, setCalibrationRunning] = useState(false);
  const [debug, setDebug] = useState<DebugStats>(INITIAL_DEBUG);
  const [scores, setScores] = useState({ player: 0, ai: 0 });
  const [eegCommand, setEegCommand] = useState<PaddleCommand>("none");

  const debugRef = useRef<DebugStats>(INITIAL_DEBUG);
  const runtimeRef = useRef<RuntimeState>(createRuntimeState());
  const pausedRef = useRef(false);
  const miSocketRef = useRef<WebSocket | null>(null);
  const calibrationRunIdRef = useRef(0);

  const publishDebug = useCallback((next: Partial<DebugStats>) => {
    debugRef.current = {
      ...debugRef.current,
      ...next,
    };
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
      publishDebug({ thonkConnected: false });
    },
    [publishDebug],
  );

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

  const startKeyboard = useCallback(
    (mode: BallControlMode) => {
      teardownMiSocket(true);
      resetRuntime();
      pausedRef.current = false;
      setBallControlMode(mode);
      setScreen("game");
    },
    [resetRuntime, teardownMiSocket],
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
        ws.send(JSON.stringify({ action: "start", interval_ms: 1000, reset: true }));
      };

      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data) as {
          status?: string;
          type?: string;
          error?: string;
          command?: string;
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
          setEegCommand(mapCommandToPaddle(payload.command));
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

  const runCalibrationRound = useCallback(async (runId: number) => {
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

      const calibrationEnd = await postJson<{ session_dir: string }>(
        API_ENDPOINTS.MI_CALIBRATION_END,
      );
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
      setBallControlMode("paddle");
      setScreen("game");
      setCalibrationRunning(false);
    } catch (error) {
      if (!isActive()) {
        return;
      }
      setCalibrationStep("error");
      setCalibrationRunning(false);
      setCalibrationError(
        error instanceof Error
          ? error.message
          : "Calibration failed. Please retry.",
      );
      setCalibrationInstruction(
        "Unable to calibrate EEG signal. Check headset contact and stream health.",
      );
    } finally {
      if (calibrationSessionOpen) {
        await postJson(API_ENDPOINTS.MI_CALIBRATION_END).catch(() => undefined);
      }
    }
  }, [connectAndStartMiStreaming, publishDebug, resetRuntime, teardownMiSocket, waitWithProgress]);

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
  }, [
    calibrationRunning,
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
            isPausedRef={pausedRef}
            controlMode={ballControlMode}
            eegCommand={ballControlMode === "paddle" ? eegCommand : "none"}
          />
          <ScoreBoard
            playerScore={scores.player}
            aiScore={scores.ai}
            accentColor={scoreboardAccent}
          />
          <DebugOverlay {...debug} />
          <KeyboardHints mode="game" />
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
