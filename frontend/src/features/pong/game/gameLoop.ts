import { renderFrame } from "@/features/pong/game/renderer";
import type { GameInputState, RuntimeState } from "@/features/pong/types/pongRuntime";
import type { UiSettings } from "@/features/pong/types/pongSettings";

export interface LoopConfig {
  ctx: CanvasRenderingContext2D;
  runtimeState: RuntimeState;
  settingsRef: { current: UiSettings };
  inputRef: { current: GameInputState };
  onFps: (fps: number) => void;
  onRuntimeUpdate?: (state: RuntimeState) => void;
  onDebugMetrics?: (payload: LoopDebugPayload) => void;
  isPausedRef?: { current: boolean };
  controlMode?: "paddle" | "ball";
  controlModeRef?: { current: "paddle" | "ball" };
}

export const FRAME_MS = 1000 / 60;
export const PADDLE_SPEED = 260;
const PONG_INITIAL_SPEED = 0.5; // PONG INITIAL SPEED: 50% of current tuned speed
export const BALL_SPEED_X = 320 * PONG_INITIAL_SPEED;
export const BALL_SPEED_Y = 220 * PONG_INITIAL_SPEED;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));
const normalize = (value: number, max: number): number => clamp(value, -max, max);

export type CollisionNormal = { x: number; y: number };

export interface LoopDebugPayload {
  ballX: number;
  ballY: number;
  ballVX: number;
  ballVY: number;
  deltaMs: number;
  collisionNormals: CollisionNormal[];
  collisionResolvedPerFrame: number;
  positionClampedPerFrame: number;
  collisionResolvedPerSecond: number;
  positionClampedPerSecond: number;
}

interface StepMetrics {
  collisionNormals: CollisionNormal[];
  collisionResolvedCount: number;
  positionClampedCount: number;
}

type ScoreSide = "player" | "ai" | null;

interface StepResult {
  ballVX: number;
  ballVY: number;
  scoreSide: ScoreSide;
  metrics: StepMetrics;
}

const resetBall = (runtimeState: RuntimeState, previousBallVY: number, direction: number): {
  ballVX: number;
  ballVY: number;
} => {
  runtimeState.ball.x = runtimeState.width / 2;
  runtimeState.ball.y = runtimeState.height / 2;

  return {
    ballVX: BALL_SPEED_X * direction,
    ballVY: BALL_SPEED_Y * Math.sign(previousBallVY || 1),
  };
};

export const stepPongPhysics = (
  runtimeState: RuntimeState,
  inputState: GameInputState,
  controlMode: "paddle" | "ball",
  currentBallVX: number,
  currentBallVY: number,
): StepResult => {
  let ballVX = currentBallVX;
  let ballVY = currentBallVY;
  let scoreSide: ScoreSide = null;
  let positionClampedCount = 0;
  let collisionResolvedCount = 0;
  const collisionNormals: CollisionNormal[] = [];

  if (controlMode === "paddle") {
    runtimeState.leftPaddle.y +=
      ((inputState.down ? 1 : 0) - (inputState.up ? 1 : 0)) * PADDLE_SPEED * (FRAME_MS / 1000);
  }

  const rightTarget = runtimeState.ball.y - runtimeState.rightPaddle.height / 2;
  const rightDelta = rightTarget - runtimeState.rightPaddle.y;
  const rightUnclampedY = runtimeState.rightPaddle.y + clamp(rightDelta, -180, 180) * 0.03;
  const rightNextY = clamp(
    rightUnclampedY,
    0,
    runtimeState.height - runtimeState.rightPaddle.height,
  );
  if (rightNextY !== rightUnclampedY) {
    positionClampedCount += 1;
  }
  if (rightNextY !== runtimeState.rightPaddle.y) {
    runtimeState.rightPaddle.y = rightNextY;
  }

  const leftNextY = clamp(
    runtimeState.leftPaddle.y,
    0,
    runtimeState.height - runtimeState.leftPaddle.height,
  );
  if (leftNextY !== runtimeState.leftPaddle.y) {
    positionClampedCount += 1;
  }
  if (leftNextY !== runtimeState.leftPaddle.y) {
    runtimeState.leftPaddle.y = leftNextY;
  }

  if (controlMode === "ball") {
    const keyX = (inputState.right ? 1 : 0) - (inputState.left ? 1 : 0);
    const keyY = (inputState.down ? 1 : 0) - (inputState.up ? 1 : 0);
    const px = normalize(Number(inputState.pointerX ?? 0), 1);
    const py = normalize(Number(inputState.pointerY ?? 0), 1);
    const dx = keyX || px;
    const dy = keyY || py;

    if (dx !== 0 || dy !== 0) {
      runtimeState.ball.x += dx * BALL_SPEED_X * (FRAME_MS / 1000);
      runtimeState.ball.y += dy * BALL_SPEED_Y * (FRAME_MS / 1000);
    } else {
      runtimeState.ball.x += ballVX * (FRAME_MS / 1000);
      runtimeState.ball.y += ballVY * (FRAME_MS / 1000);
    }
  } else {
    runtimeState.ball.x += ballVX * (FRAME_MS / 1000);
    runtimeState.ball.y += ballVY * (FRAME_MS / 1000);
  }

  if (runtimeState.ball.y - runtimeState.ball.radius <= 0) {
    ballVY *= -1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 0, y: 1 });
  } else if (runtimeState.ball.y + runtimeState.ball.radius >= runtimeState.height) {
    ballVY *= -1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 0, y: -1 });
  }

  const nextBallX = runtimeState.ball.x;
  const nextBallY = runtimeState.ball.y;
  const lp = runtimeState.leftPaddle;
  const rp = runtimeState.rightPaddle;
  const radius = runtimeState.ball.radius;

  if (
    ballVX < 0 &&
    nextBallX - radius <= lp.x + lp.width &&
    nextBallX >= lp.x + lp.width &&
    nextBallY >= lp.y &&
    nextBallY <= lp.y + lp.height
  ) {
    ballVX *= -1.06;
    runtimeState.ball.x = lp.x + lp.width + radius + 1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 1, y: 0 });
  }

  if (
    ballVX > 0 &&
    nextBallX + radius >= rp.x &&
    nextBallX - radius <= rp.x + rp.width &&
    nextBallY >= rp.y &&
    nextBallY <= rp.y + rp.height
  ) {
    ballVX *= -1.06;
    runtimeState.ball.x = rp.x - radius - 1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: -1, y: 0 });
  }

  if (runtimeState.ball.x < -radius) {
    runtimeState.aiScore += 1;
    scoreSide = "ai";
    ({ ballVX, ballVY } = resetBall(runtimeState, ballVY, -1));
  } else if (runtimeState.ball.x > runtimeState.width + radius) {
    runtimeState.playerScore += 1;
    scoreSide = "player";
    ({ ballVX, ballVY } = resetBall(runtimeState, ballVY, 1));
  }

  return {
    ballVX,
    ballVY,
    scoreSide,
    metrics: {
      collisionNormals,
      collisionResolvedCount,
      positionClampedCount,
    },
  };
};

export const createPongLoop = (cfg: LoopConfig) => {
  const {
    ctx,
    runtimeState,
    settingsRef,
    inputRef,
    onFps,
    onRuntimeUpdate,
    onDebugMetrics,
    isPausedRef,
    controlMode = "paddle",
    controlModeRef,
  } = cfg;

  const getControlMode = () => controlModeRef?.current ?? controlMode;
  let rafId = 0;
  let lastTs = performance.now();
  let accumulator = 0;
  let fpsWindowStart = performance.now();
  let frameCount = 0;

  let debugWindowStart = performance.now();
  let debugWindowCollisionEvents = 0;
  let debugWindowClampEvents = 0;
  let collisionResolvedPerSecond = 0;
  let positionClampedPerSecond = 0;

  let ballVX = BALL_SPEED_X;
  let ballVY = BALL_SPEED_Y;

  const tick = (now: number) => {
    const delta = now - lastTs;
    lastTs = now;
    accumulator += delta;

    if (isPausedRef?.current) {
      onDebugMetrics?.({
        ballX: runtimeState.ball.x,
        ballY: runtimeState.ball.y,
        ballVX,
        ballVY,
        deltaMs: delta,
        collisionNormals: [],
        collisionResolvedPerFrame: 0,
        positionClampedPerFrame: 0,
        collisionResolvedPerSecond,
        positionClampedPerSecond,
      });

      renderFrame(ctx, runtimeState, settingsRef.current);
      frameCount += 1;
      const fpsElapsed = now - fpsWindowStart;
      if (fpsElapsed >= 500) {
        onFps((frameCount * 1000) / fpsElapsed);
        frameCount = 0;
        fpsWindowStart = now;
      }

      rafId = requestAnimationFrame(tick);
      return;
    }

    let collisionResolvedThisFrame = 0;
    let clampedThisFrame = 0;
    const collisionNormalsThisFrame: CollisionNormal[] = [];

    while (accumulator >= FRAME_MS) {
      const stepResult = stepPongPhysics(
        runtimeState,
        inputRef.current,
        getControlMode(),
        ballVX,
        ballVY,
      );

      ballVX = stepResult.ballVX;
      ballVY = stepResult.ballVY;
      collisionResolvedThisFrame += stepResult.metrics.collisionResolvedCount;
      clampedThisFrame += stepResult.metrics.positionClampedCount;
      debugWindowCollisionEvents += stepResult.metrics.collisionResolvedCount;
      debugWindowClampEvents += stepResult.metrics.positionClampedCount;
      collisionNormalsThisFrame.push(...stepResult.metrics.collisionNormals);

      if (stepResult.scoreSide) {
        onRuntimeUpdate?.(runtimeState);
      }

      accumulator -= FRAME_MS;
    }

    const debugWindowElapsed = now - debugWindowStart;
    if (debugWindowElapsed >= 1000) {
      const elapsedSeconds = Math.max(debugWindowElapsed / 1000, 1);
      collisionResolvedPerSecond = debugWindowCollisionEvents / elapsedSeconds;
      positionClampedPerSecond = debugWindowClampEvents / elapsedSeconds;
      debugWindowCollisionEvents = 0;
      debugWindowClampEvents = 0;
      debugWindowStart = now;
    }

    onDebugMetrics?.({
      ballX: runtimeState.ball.x,
      ballY: runtimeState.ball.y,
      ballVX,
      ballVY,
      deltaMs: delta,
      collisionNormals: collisionNormalsThisFrame,
      collisionResolvedPerFrame: collisionResolvedThisFrame,
      positionClampedPerFrame: clampedThisFrame,
      collisionResolvedPerSecond,
      positionClampedPerSecond,
    });

    renderFrame(ctx, runtimeState, settingsRef.current);
    frameCount += 1;
    const fpsElapsed = now - fpsWindowStart;
    if (fpsElapsed >= 500) {
      onFps((frameCount * 1000) / fpsElapsed);
      frameCount = 0;
      fpsWindowStart = now;
    }

    rafId = requestAnimationFrame(tick);
  };

  rafId = requestAnimationFrame(tick);

  return () => {
    if (rafId) {
      cancelAnimationFrame(rafId);
    }
  };
};
