import { useEffect, useRef } from "react";
import { createBreakoutLoop } from "@/features/breakout/game/gameLoop";
import type { LoopDebugPayload } from "@/features/breakout/game/gameLoop";
import type {
  GameInputState,
  RuntimeState,
} from "@/features/breakout/types/breakoutRuntime";
import type { UiSettings } from "@/features/pong/types/pongSettings";

type GameCanvasProps = {
  runtimeState: RuntimeState;
  settings: UiSettings;
  onFps: (fps: number) => void;
  onRuntimeUpdate?: (state: RuntimeState) => void;
  onDebugMetrics?: (payload: LoopDebugPayload) => void;
  isPausedRef?: { current: boolean };
  eegCommand?: "none" | "left" | "right";
};

export const GameCanvas = ({
  runtimeState,
  settings,
  onFps,
  onRuntimeUpdate,
  onDebugMetrics,
  isPausedRef,
  eegCommand = "none",
}: GameCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const runtimeRef = useRef(runtimeState);
  const settingsRef = useRef(settings);
  const onFpsRef = useRef(onFps);
  const onRuntimeUpdateRef = useRef(onRuntimeUpdate);
  const onDebugMetricsRef = useRef(onDebugMetrics);
  const eegCommandRef = useRef<"none" | "left" | "right">(eegCommand);
  const inputRef = useRef<GameInputState>({
    left: false,
    right: false,
  });

  useEffect(() => {
    runtimeRef.current = runtimeState;
  }, [runtimeState]);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    onFpsRef.current = onFps;
  }, [onFps]);

  useEffect(() => {
    onRuntimeUpdateRef.current = onRuntimeUpdate;
  }, [onRuntimeUpdate]);

  useEffect(() => {
    onDebugMetricsRef.current = onDebugMetrics;
  }, [onDebugMetrics]);

  useEffect(() => {
    eegCommandRef.current = eegCommand;
  }, [eegCommand]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (event.key === "ArrowLeft" || key === "a") {
        inputRef.current.left = true;
      }
      if (event.key === "ArrowRight" || key === "d") {
        inputRef.current.right = true;
      }
    };

    const onKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (event.key === "ArrowLeft" || key === "a") {
        inputRef.current.left = false;
      }
      if (event.key === "ArrowRight" || key === "d") {
        inputRef.current.right = false;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    ctxRef.current = context;

    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = runtimeState.width * dpr;
    canvas.height = runtimeState.height * dpr;
    canvas.style.width = `${runtimeState.width}px`;
    canvas.style.height = `${runtimeState.height}px`;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);

    const loopCtx = ctxRef.current;
    if (!loopCtx) {
      return;
    }

    const stop = createBreakoutLoop({
      ctx: loopCtx,
      runtimeState: runtimeRef.current,
      settingsRef,
      inputRef,
      onFps: (value) => onFpsRef.current?.(value),
      onRuntimeUpdate: (value) => onRuntimeUpdateRef.current?.(value),
      onDebugMetrics: (value) => onDebugMetricsRef.current?.(value),
      isPausedRef,
      eegCommandRef,
    });

    return stop;
  }, [isPausedRef, runtimeState.width, runtimeState.height]);

  return (
    <div className="w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
};
