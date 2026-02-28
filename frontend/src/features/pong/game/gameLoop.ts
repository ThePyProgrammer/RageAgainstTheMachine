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
  eegCommandRef?: { current: "none" | "left" | "right" };
}

export const FRAME_MS = 1000 / 60;
export const PADDLE_SPEED = 260;
const PONG_INITIAL_SPEED = 0.5; // PONG INITIAL SPEED: 50% of current tuned speed
export const BALL_SPEED_X = 220 * PONG_INITIAL_SPEED;  // Secondary (lateral movement)
export const BALL_SPEED_Y = 320 * PONG_INITIAL_SPEED;  // Primary (vertical movement)

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

const resetBall = (runtimeState: RuntimeState, previousBallVX: number, direction: number): {
  ballVX: number;
  ballVY: number;
} => {
  runtimeState.ball.x = runtimeState.width / 2;
  runtimeState.ball.y = runtimeState.height / 2;

  // Ball travels vertically: direction controls Y, X is secondary
  return {
    ballVX: BALL_SPEED_X * Math.sign(previousBallVX || 1),
    ballVY: BALL_SPEED_Y * direction,
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

  // Player paddle movement: horizontal (left/right keys affect topPaddle.x)
  // Also allow up/down keys to move paddle left/right for convenience
  if (controlMode === "paddle") {
    const moveLeft = inputState.left || inputState.up;
    const moveRight = inputState.right || inputState.down;
    runtimeState.topPaddle.x +=
      ((moveRight ? 1 : 0) - (moveLeft ? 1 : 0)) * PADDLE_SPEED * (FRAME_MS / 1000);
  }

  // AI tracks ball.x and moves bottomPaddle.x
  const bottomTarget = runtimeState.ball.x - runtimeState.bottomPaddle.width / 2;
  const bottomDelta = bottomTarget - runtimeState.bottomPaddle.x;
  const bottomUnclampedX = runtimeState.bottomPaddle.x + clamp(bottomDelta, -180, 180) * 0.03;
  const bottomNextX = clamp(
    bottomUnclampedX,
    0,
    runtimeState.width - runtimeState.bottomPaddle.width,
  );
  if (bottomNextX !== bottomUnclampedX) {
    positionClampedCount += 1;
  }
  if (bottomNextX !== runtimeState.bottomPaddle.x) {
    runtimeState.bottomPaddle.x = bottomNextX;
  }

  // Clamp player paddle X position
  const topNextX = clamp(
    runtimeState.topPaddle.x,
    0,
    runtimeState.width - runtimeState.topPaddle.width,
  );
  if (topNextX !== runtimeState.topPaddle.x) {
    positionClampedCount += 1;
  }
  if (topNextX !== runtimeState.topPaddle.x) {
    runtimeState.topPaddle.x = topNextX;
  }

  // Ball movement
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

  // Wall bouncing: ball bounces off LEFT and RIGHT walls
  const radius = runtimeState.ball.radius;
  if (runtimeState.ball.x - radius <= 0) {
    ballVX *= -1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 1, y: 0 });
  } else if (runtimeState.ball.x + radius >= runtimeState.width) {
    ballVX *= -1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: -1, y: 0 });
  }

  const nextBallX = runtimeState.ball.x;
  const nextBallY = runtimeState.ball.y;
  const tp = runtimeState.topPaddle;
  const bp = runtimeState.bottomPaddle;

  // Top paddle collision: ball moving up (ballVY < 0), hits bottom edge of top paddle
  if (
    ballVY < 0 &&
    nextBallY - radius <= tp.y + tp.height &&
    nextBallY + radius >= tp.y &&
    nextBallX >= tp.x &&
    nextBallX <= tp.x + tp.width
  ) {
    ballVY *= -1.06;
    runtimeState.ball.y = tp.y + tp.height + radius + 1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 0, y: 1 });
  }

  // Bottom paddle collision: ball moving down (ballVY > 0), hits top edge of bottom paddle
  if (
    ballVY > 0 &&
    nextBallY + radius >= bp.y &&
    nextBallY - radius <= bp.y + bp.height &&
    nextBallX >= bp.x &&
    nextBallX <= bp.x + bp.width
  ) {
    ballVY *= -1.06;
    runtimeState.ball.y = bp.y - radius - 1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 0, y: -1 });
  }

  // Scoring: ball exits top (AI scores) or bottom (player scores)
  if (runtimeState.ball.y < -radius) {
    runtimeState.aiScore += 1;
    scoreSide = "ai";
    ({ ballVX, ballVY } = resetBall(runtimeState, ballVX, -1));
  } else if (runtimeState.ball.y > runtimeState.height + radius) {
    runtimeState.playerScore += 1;
    scoreSide = "player";
    ({ ballVX, ballVY } = resetBall(runtimeState, ballVX, 1));
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
    eegCommandRef,
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
        {
          ...inputRef.current,
          left:
            inputRef.current.left || eegCommandRef?.current === "left",
          right:
            inputRef.current.right || eegCommandRef?.current === "right",
        },
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
