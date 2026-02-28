import { renderFrame } from "@/features/breakout/game/renderer";
import type {
  GameInputState,
  RuntimeBrick,
  RuntimeState,
} from "@/features/breakout/types/breakoutRuntime";
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
  eegCommandRef?: { current: "none" | "left" | "right" };
}

export const FRAME_MS = 1000 / 60;
const MAX_FRAME_DELTA_MS = 50;
const MAX_STEPS_PER_TICK = 6;
const BRICK_SCORE = 100;
const BALL_LAUNCH_ANGLE = Math.PI / 7;
const MIN_HORIZONTAL_VELOCITY_RATIO = 0.2;

export const DEFAULT_WIDTH = 960;
export const DEFAULT_HEIGHT = 540;
export const DEFAULT_LIVES = 5;
export const PADDLE_SPEED = 600;
export const BASE_BALL_SPEED = 450;

export const START_BLOCK_COUNT = 8;
export const BLOCKS_PER_LEVEL_STEP = 3;
export const MAX_BLOCK_COUNT = 20;
export const LEVEL_SPEED_STEP = 0.1;
export const MAX_SPEED_BONUS = 0.5;

const PADDLE_WIDTH = 132;
const PADDLE_HEIGHT = 14;
const PADDLE_MARGIN_BOTTOM = 30;
const BALL_RADIUS = 10;
const BRICK_TOP_OFFSET = 72;
const BRICK_SIDE_MARGIN = 48;
const BRICK_HEIGHT = 22;
const BRICK_GAP = 8;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const intersectsBallRect = (
  ball: RuntimeState["ball"],
  rect: Pick<RuntimeBrick, "x" | "y" | "width" | "height">,
): boolean => {
  const nearestX = clamp(ball.x, rect.x, rect.x + rect.width);
  const nearestY = clamp(ball.y, rect.y, rect.y + rect.height);
  const dx = ball.x - nearestX;
  const dy = ball.y - nearestY;
  return dx * dx + dy * dy <= ball.radius * ball.radius;
};

const createLevelBricks = (fieldWidth: number, count: number): RuntimeBrick[] => {
  const normalizedCount = Math.max(1, Math.floor(count));
  const columns = Math.min(8, Math.max(4, Math.ceil(normalizedCount / 4)));
  const rows = Math.ceil(normalizedCount / columns);
  const usableWidth = fieldWidth - BRICK_SIDE_MARGIN * 2;
  const brickWidth = Math.max(24, (usableWidth - BRICK_GAP * (columns - 1)) / columns);
  const bricks: RuntimeBrick[] = [];

  for (let index = 0; index < normalizedCount; index += 1) {
    const row = Math.floor(index / columns);
    const col = index % columns;

    if (row >= rows) {
      break;
    }

    const x = BRICK_SIDE_MARGIN + col * (brickWidth + BRICK_GAP);
    const y = BRICK_TOP_OFFSET + row * (BRICK_HEIGHT + BRICK_GAP);
    bricks.push({
      id: index + 1,
      x,
      y,
      width: brickWidth,
      height: BRICK_HEIGHT,
      active: true,
    });
  }

  return bricks;
};

export const getBlocksForLevel = (level: number): number => {
  const normalizedLevel = Math.max(1, Math.floor(level));
  const rawCount = START_BLOCK_COUNT + (normalizedLevel - 1) * BLOCKS_PER_LEVEL_STEP;
  return Math.min(rawCount, MAX_BLOCK_COUNT);
};

export const getBallSpeedMultiplier = (level: number): number => {
  const normalizedLevel = Math.max(1, Math.floor(level));
  const bonus = Math.min((normalizedLevel - 1) * LEVEL_SPEED_STEP, MAX_SPEED_BONUS);
  return 1 + bonus;
};

export const getBallSpeedForLevel = (level: number): number =>
  BASE_BALL_SPEED * getBallSpeedMultiplier(level);

export const createBricksForLevel = (fieldWidth: number, level: number): RuntimeBrick[] =>
  createLevelBricks(fieldWidth, getBlocksForLevel(level));

type Velocity = {
  ballVX: number;
  ballVY: number;
};

const resetBallAndPaddle = (
  runtimeState: RuntimeState,
  serveDirection: number,
): Velocity => {
  runtimeState.paddle.x = (runtimeState.width - runtimeState.paddle.width) / 2;
  runtimeState.paddle.y = runtimeState.height - PADDLE_MARGIN_BOTTOM - runtimeState.paddle.height;

  runtimeState.ball.x = runtimeState.width / 2;
  runtimeState.ball.y = runtimeState.paddle.y - runtimeState.ball.radius - 2;

  const speed = getBallSpeedForLevel(runtimeState.level);
  const direction = serveDirection >= 0 ? 1 : -1;
  const vx = direction * speed * Math.sin(BALL_LAUNCH_ANGLE);
  const vy = -Math.abs(speed * Math.cos(BALL_LAUNCH_ANGLE));
  return { ballVX: vx, ballVY: vy };
};

export const createInitialRuntimeState = (): RuntimeState => {
  const level = 1;
  const bricks = createBricksForLevel(DEFAULT_WIDTH, level);
  const paddleY = DEFAULT_HEIGHT - PADDLE_MARGIN_BOTTOM - PADDLE_HEIGHT;
  return {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    paddle: {
      x: (DEFAULT_WIDTH - PADDLE_WIDTH) / 2,
      y: paddleY,
      width: PADDLE_WIDTH,
      height: PADDLE_HEIGHT,
    },
    ball: {
      x: DEFAULT_WIDTH / 2,
      y: paddleY - BALL_RADIUS - 2,
      radius: BALL_RADIUS,
    },
    bricks,
    bricksRemaining: bricks.length,
    score: 0,
    lives: DEFAULT_LIVES,
    level,
    gameOver: false,
  };
};

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
  bricksRemaining: number;
  speedMultiplier: number;
}

interface StepMetrics {
  collisionNormals: CollisionNormal[];
  collisionResolvedCount: number;
  positionClampedCount: number;
}

interface StepResult {
  ballVX: number;
  ballVY: number;
  metrics: StepMetrics;
  didLoseLife: boolean;
  didLevelUp: boolean;
  didGameOver: boolean;
  stateChanged: boolean;
}

export const stepBreakoutPhysics = (
  runtimeState: RuntimeState,
  inputState: GameInputState,
  currentBallVX: number,
  currentBallVY: number,
): StepResult => {
  if (runtimeState.gameOver) {
    return {
      ballVX: 0,
      ballVY: 0,
      metrics: {
        collisionNormals: [],
        collisionResolvedCount: 0,
        positionClampedCount: 0,
      },
      didLoseLife: false,
      didLevelUp: false,
      didGameOver: true,
      stateChanged: false,
    };
  }

  let ballVX = currentBallVX;
  let ballVY = currentBallVY;
  let collisionResolvedCount = 0;
  let positionClampedCount = 0;
  const collisionNormals: CollisionNormal[] = [];

  const moveLeft = inputState.left;
  const moveRight = inputState.right;
  const paddleUnclampedX =
    runtimeState.paddle.x +
    ((moveRight ? 1 : 0) - (moveLeft ? 1 : 0)) *
      PADDLE_SPEED *
      (FRAME_MS / 1000);
  const paddleX = clamp(
    paddleUnclampedX,
    0,
    runtimeState.width - runtimeState.paddle.width,
  );
  if (paddleX !== paddleUnclampedX) {
    positionClampedCount += 1;
  }
  runtimeState.paddle.x = paddleX;

  runtimeState.ball.x += ballVX * (FRAME_MS / 1000);
  runtimeState.ball.y += ballVY * (FRAME_MS / 1000);

  const radius = runtimeState.ball.radius;
  if (runtimeState.ball.x - radius <= 0) {
    runtimeState.ball.x = radius;
    ballVX = Math.abs(ballVX);
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 1, y: 0 });
  } else if (runtimeState.ball.x + radius >= runtimeState.width) {
    runtimeState.ball.x = runtimeState.width - radius;
    ballVX = -Math.abs(ballVX);
    collisionResolvedCount += 1;
    collisionNormals.push({ x: -1, y: 0 });
  }

  if (runtimeState.ball.y - radius <= 0) {
    runtimeState.ball.y = radius;
    ballVY = Math.abs(ballVY);
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 0, y: 1 });
  }

  const paddle = runtimeState.paddle;
  if (
    ballVY > 0 &&
    intersectsBallRect(runtimeState.ball, paddle)
  ) {
    const paddleCenter = paddle.x + paddle.width / 2;
    const hitOffset = clamp((runtimeState.ball.x - paddleCenter) / (paddle.width / 2), -1, 1);
    const bounceAngle = hitOffset * (Math.PI / 3);
    const speed = getBallSpeedForLevel(runtimeState.level);

    ballVX = speed * Math.sin(bounceAngle);
    if (Math.abs(ballVX) < speed * MIN_HORIZONTAL_VELOCITY_RATIO) {
      const sign = Math.sign(ballVX || currentBallVX || 1);
      ballVX = sign * speed * MIN_HORIZONTAL_VELOCITY_RATIO;
    }
    ballVY = -Math.abs(speed * Math.cos(bounceAngle));
    runtimeState.ball.y = paddle.y - radius - 0.5;

    collisionResolvedCount += 1;
    collisionNormals.push({ x: 0, y: -1 });
  }

  let destroyedBrick = false;
  for (let index = 0; index < runtimeState.bricks.length; index += 1) {
    const brick = runtimeState.bricks[index];
    if (!brick.active || !intersectsBallRect(runtimeState.ball, brick)) {
      continue;
    }

    brick.active = false;
    runtimeState.bricksRemaining = Math.max(0, runtimeState.bricksRemaining - 1);
    runtimeState.score += BRICK_SCORE;
    destroyedBrick = true;

    const brickCenterX = brick.x + brick.width / 2;
    const brickCenterY = brick.y + brick.height / 2;
    const deltaX = runtimeState.ball.x - brickCenterX;
    const deltaY = runtimeState.ball.y - brickCenterY;
    const overlapX = brick.width / 2 + radius - Math.abs(deltaX);
    const overlapY = brick.height / 2 + radius - Math.abs(deltaY);

    if (overlapX < overlapY) {
      if (deltaX < 0) {
        runtimeState.ball.x -= overlapX;
        ballVX = -Math.abs(ballVX);
        collisionNormals.push({ x: -1, y: 0 });
      } else {
        runtimeState.ball.x += overlapX;
        ballVX = Math.abs(ballVX);
        collisionNormals.push({ x: 1, y: 0 });
      }
    } else {
      if (deltaY < 0) {
        runtimeState.ball.y -= overlapY;
        ballVY = -Math.abs(ballVY);
        collisionNormals.push({ x: 0, y: -1 });
      } else {
        runtimeState.ball.y += overlapY;
        ballVY = Math.abs(ballVY);
        collisionNormals.push({ x: 0, y: 1 });
      }
    }

    collisionResolvedCount += 1;
    break;
  }

  let didLoseLife = false;
  let didLevelUp = false;
  let didGameOver = false;
  let stateChanged = destroyedBrick;

  if (runtimeState.ball.y - radius > runtimeState.height) {
    runtimeState.lives = Math.max(0, runtimeState.lives - 1);
    didLoseLife = true;
    stateChanged = true;
    if (runtimeState.lives <= 0) {
      runtimeState.gameOver = true;
      didGameOver = true;
      ballVX = 0;
      ballVY = 0;
    }
  } else if (runtimeState.bricksRemaining <= 0) {
    runtimeState.level += 1;
    const nextBricks = createBricksForLevel(runtimeState.width, runtimeState.level);
    runtimeState.bricks = nextBricks;
    runtimeState.bricksRemaining = nextBricks.length;
    didLevelUp = true;
    stateChanged = true;
  }

  return {
    ballVX,
    ballVY,
    metrics: {
      collisionNormals,
      collisionResolvedCount,
      positionClampedCount,
    },
    didLoseLife,
    didLevelUp,
    didGameOver,
    stateChanged,
  };
};

export const createBreakoutLoop = (cfg: LoopConfig) => {
  const {
    ctx,
    runtimeState,
    settingsRef,
    inputRef,
    onFps,
    onRuntimeUpdate,
    onDebugMetrics,
    isPausedRef,
    eegCommandRef,
  } = cfg;

  let serveDirection = 1;
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

  let initialVelocity = resetBallAndPaddle(runtimeState, serveDirection);
  let ballVX = initialVelocity.ballVX;
  let ballVY = initialVelocity.ballVY;
  onRuntimeUpdate?.(runtimeState);

  const tick = (now: number) => {
    const rawDelta = now - lastTs;
    const delta = Math.min(rawDelta, MAX_FRAME_DELTA_MS);
    lastTs = now;
    accumulator = Math.min(accumulator + delta, FRAME_MS * MAX_STEPS_PER_TICK);

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
        bricksRemaining: runtimeState.bricksRemaining,
        speedMultiplier: getBallSpeedMultiplier(runtimeState.level),
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

    let stepsThisTick = 0;
    while (accumulator >= FRAME_MS && stepsThisTick < MAX_STEPS_PER_TICK) {
      const stepResult = stepBreakoutPhysics(
        runtimeState,
        {
          left: inputRef.current.left || eegCommandRef?.current === "left",
          right: inputRef.current.right || eegCommandRef?.current === "right",
        },
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

      if (stepResult.didLoseLife && !stepResult.didGameOver) {
        serveDirection *= -1;
        initialVelocity = resetBallAndPaddle(runtimeState, serveDirection);
        ballVX = initialVelocity.ballVX;
        ballVY = initialVelocity.ballVY;
      } else if (stepResult.didLevelUp) {
        serveDirection *= -1;
        initialVelocity = resetBallAndPaddle(runtimeState, serveDirection);
        ballVX = initialVelocity.ballVX;
        ballVY = initialVelocity.ballVY;
      }

      if (stepResult.stateChanged) {
        onRuntimeUpdate?.(runtimeState);
      }

      accumulator -= FRAME_MS;
      stepsThisTick += 1;
    }

    if (accumulator >= FRAME_MS) {
      accumulator = 0;
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
      bricksRemaining: runtimeState.bricksRemaining,
      speedMultiplier: getBallSpeedMultiplier(runtimeState.level),
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
