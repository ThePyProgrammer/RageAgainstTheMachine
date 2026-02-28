import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MenuScreen } from "@/features/pong/components/MenuScreen";
import { GameCanvas } from "@/features/pong/components/GameCanvas";
import { ScoreBoard } from "@/features/pong/components/ScoreBoard";
import { DebugOverlay } from "@/features/pong/components/DebugOverlay";
import { KeyboardHints } from "@/features/pong/components/KeyboardHints";
import { CalibrationWizard } from "@/features/pong/components/CalibrationWizard";
import { SettingsOverlay } from "@/features/pong/components/SettingsOverlay";
import type { LoopDebugPayload } from "@/features/pong/game/gameLoop";
import { resolveUiColorToken, deriveScoreAccentColor } from "@/features/pong/types/pongSettings";
import { usePongSettings } from "@/features/pong/state/usePongSettings";
import type {
  DebugStats,
  GameScreen,
} from "@/features/pong/types/pongRuntime";
import type { RuntimeState } from "@/features/pong/types/pongRuntime";

type CalibrationStep = "baseline" | "left" | "right" | "complete";
type BallControlMode = "paddle" | "ball";

const createRuntimeState = (): RuntimeState => ({
  width: 960,
  height: 540,
  ball: { x: 480, y: 270, radius: 10 },
  leftPaddle: { x: 20, y: 215, width: 14, height: 110 },
  rightPaddle: { x: 926, y: 215, width: 14, height: 110 },
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

const CALIBRATION_KEY = "pong_saved_calibration_v1";

export default function PongPage() {
  const { settings, updateSettings, resetSettings } = usePongSettings();
  const [screen, setScreen] = useState<GameScreen>("menu");
  const [ballControlMode, setBallControlMode] = useState<BallControlMode>("paddle");
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [hasSavedCalibration, setHasSavedCalibration] = useState(false);
  const [calibrationStep, setCalibrationStep] = useState<CalibrationStep>("baseline");
  const [quality, setQuality] = useState(0);
  const [debug, setDebug] = useState<DebugStats>(INITIAL_DEBUG);
  const [scores, setScores] = useState({ player: 0, ai: 0 });
  const debugRef = useRef<DebugStats>(INITIAL_DEBUG);
  const runtimeRef = useRef<RuntimeState>(createRuntimeState());
  const pausedRef = useRef(false);

  const resetRuntime = useCallback(() => {
    Object.assign(runtimeRef.current, createRuntimeState());
    debugRef.current = INITIAL_DEBUG;
    setDebug(INITIAL_DEBUG);
    setScores({ player: 0, ai: 0 });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    setHasSavedCalibration(localStorage.getItem(CALIBRATION_KEY) === "1");
  }, []);

  const openCalibration = useCallback(() => {
    setScreen("calibration");
    setCalibrationStep("baseline");
  }, []);

  const startKeyboard = useCallback((mode: BallControlMode) => {
    resetRuntime();
    pausedRef.current = false;
    setBallControlMode(mode);
    setScreen("game");
  }, [resetRuntime]);

  const startEEGCalibration = useCallback(() => {
    openCalibration();
    setHasSavedCalibration(false);
  }, [openCalibration]);

  const playWithSavedCalibration = useCallback(() => {
    setBallControlMode("paddle");
    resetRuntime();
    pausedRef.current = false;
    setScreen("game");
  }, [resetRuntime]);

  const updateScoresFromRuntime = useCallback((state: RuntimeState) => {
    setScores((previous) => {
      if (previous.player === state.playerScore && previous.ai === state.aiScore) {
        return previous;
      }
      return {
        player: state.playerScore,
        ai: state.aiScore,
      };
    });
  }, []);

  const continueCalibration = useCallback(() => {
    if (calibrationStep === "baseline") {
      setQuality(1.2);
      setCalibrationStep("left");
      return;
    }
    if (calibrationStep === "left") {
      setQuality(1.15);
      setCalibrationStep("right");
      return;
    }
    if (calibrationStep === "right") {
      setQuality(1.08);
      setCalibrationStep("complete");
      return;
    }
    if (calibrationStep === "complete") {
      if (typeof window !== "undefined") {
        localStorage.setItem(CALIBRATION_KEY, "1");
      }
      setHasSavedCalibration(true);
      setScreen("game");
    }
  }, [calibrationStep]);

  const retryCalibration = useCallback(() => {
    setCalibrationStep("baseline");
    setQuality(0);
  }, []);

  const saveSettings = useCallback(
    (next: typeof settings) => {
      updateSettings(next);
    },
    [updateSettings],
  );

  const publishDebug = useCallback((next: Partial<DebugStats>) => {
    debugRef.current = {
      ...debugRef.current,
      ...next,
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
        if (key === "enter" || key === " ") continueCalibration();
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
  }, [continueCalibration, overlayOpen, screen, startEEGCalibration, startKeyboard]);

  useEffect(() => {
    if (screen !== "game" && screen !== "paused") {
      return;
    }

    const interval = window.setInterval(() => {
      setDebug({
        ...debugRef.current,
      });
    }, 120);

    return () => {
      clearInterval(interval);
    };
  }, [screen]);

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
          hasSavedCalibration={hasSavedCalibration}
          onStartKeyboard={() => startKeyboard("paddle")}
          onStartBallMode={() => startKeyboard("ball")}
          onStartEEG={startEEGCalibration}
          onStartWithSavedCalibration={playWithSavedCalibration}
          onCustomize={() => setOverlayOpen(true)}
        />
      )}

      {screen === "calibration" && (
        <CalibrationWizard
          step={calibrationStep}
          trial={calibrationStep === "left" || calibrationStep === "right" ? 1 : 0}
          instruction={
            calibrationStep === "baseline"
              ? "Press Enter to continue to left trial."
              : calibrationStep === "left"
                ? "Press Enter to continue to right trial."
                : calibrationStep === "right"
                  ? "Press Enter to complete."
                  : "Calibration accepted. Continue to game."
          }
          progress={
            calibrationStep === "baseline"
              ? 0.25
              : calibrationStep === "left"
                ? 0.5
                : calibrationStep === "right"
                  ? 0.75
                  : 1
          }
          quality={quality}
          onRetry={retryCalibration}
          onContinue={continueCalibration}
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
