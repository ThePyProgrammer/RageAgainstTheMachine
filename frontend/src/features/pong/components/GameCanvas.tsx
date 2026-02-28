import { useEffect, useRef } from "react";
import { createPongLoop } from "@/features/pong/game/gameLoop";
import type { LoopDebugPayload, OpponentLoopEventPayload } from "@/features/pong/game/gameLoop";
import type { RuntimeState, GameInputState } from "@/features/pong/types/pongRuntime";
import type { UiSettings } from "@/features/pong/types/pongSettings";

type GameCanvasProps = {
  runtimeState: RuntimeState;
  settings: UiSettings;
  onFps: (fps: number) => void;
  onRuntimeUpdate?: (state: RuntimeState) => void;
  onOpponentEvent?: (event: OpponentLoopEventPayload) => void;
  onDebugMetrics?: (payload: LoopDebugPayload) => void;
  isPausedRef?: { current: boolean };
  controlMode?: "paddle" | "ball";
  eegCommand?: "none" | "left" | "right";
  difficultyRef?: { current: number };
};

export const GameCanvas = ({
  runtimeState,
  settings,
  onFps,
  onRuntimeUpdate,
  onOpponentEvent,
  onDebugMetrics,
  isPausedRef,
  controlMode = "paddle",
  eegCommand = "none",
  difficultyRef,
}: GameCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const runtimeRef = useRef(runtimeState);
  const settingsRef = useRef(settings);
  const onFpsRef = useRef(onFps);
  const onRuntimeUpdateRef = useRef(onRuntimeUpdate);
  const onOpponentEventRef = useRef(onOpponentEvent);
  const onDebugMetricsRef = useRef(onDebugMetrics);
  const controlModeRef = useRef(controlMode);
  const eegCommandRef = useRef<"none" | "left" | "right">(eegCommand);
  const inputRef = useRef<GameInputState>({
    up: false,
    down: false,
    left: false,
    right: false,
    pointerX: 0,
    pointerY: 0,
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
    onOpponentEventRef.current = onOpponentEvent;
  }, [onOpponentEvent]);

  useEffect(() => {
    onDebugMetricsRef.current = onDebugMetrics;
  }, [onDebugMetrics]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (event.key === "ArrowUp" || key === "w") {
        inputRef.current.up = true;
      }
      if (event.key === "ArrowDown" || key === "s") {
        inputRef.current.down = true;
      }
      if (event.key === "ArrowLeft" || key === "a") {
        inputRef.current.left = true;
      }
      if (event.key === "ArrowRight" || key === "d") {
        inputRef.current.right = true;
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (event.key === "ArrowUp" || key === "w") {
        inputRef.current.up = false;
      }
      if (event.key === "ArrowDown" || key === "s") {
        inputRef.current.down = false;
      }
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
    controlModeRef.current = controlMode;
  }, [controlMode]);

  useEffect(() => {
    eegCommandRef.current = eegCommand;
  }, [eegCommand]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const onPointerMove = (event: PointerEvent) => {
      if (controlMode !== "ball") {
        return;
      }
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        return;
      }

      const nx = (event.clientX - rect.left) / rect.width;
      const ny = (event.clientY - rect.top) / rect.height;
      inputRef.current.pointerX = nx * 2 - 1;
      inputRef.current.pointerY = ny * 2 - 1;
    };

    const onPointerLeave = () => {
      inputRef.current.pointerX = 0;
      inputRef.current.pointerY = 0;
    };

    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerleave", onPointerLeave);
    canvas.addEventListener("pointercancel", onPointerLeave);
    return () => {
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerleave", onPointerLeave);
      canvas.removeEventListener("pointercancel", onPointerLeave);
    };
  }, [controlMode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;
    ctxRef.current = context;

    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = runtimeState.width * dpr;
    canvas.height = runtimeState.height * dpr;
    canvas.style.width = `${runtimeState.width}px`;
    canvas.style.height = `${runtimeState.height}px`;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);

    const loopCtx = ctxRef.current;
    if (!loopCtx) return;

    const stop = createPongLoop({
      ctx: loopCtx,
      runtimeState: runtimeRef.current,
      settingsRef,
      inputRef,
      onFps: (value) => onFpsRef.current?.(value),
      onRuntimeUpdate: (value) => onRuntimeUpdateRef.current?.(value),
      onOpponentEvent: (value) => onOpponentEventRef.current?.(value),
      onDebugMetrics: (value) => onDebugMetricsRef.current?.(value),
      isPausedRef,
      controlModeRef,
      eegCommandRef,
      difficultyRef,
    });

    return stop;
  }, [difficultyRef, isPausedRef, runtimeState.width, runtimeState.height]);

  return (
    <div className="w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
};
