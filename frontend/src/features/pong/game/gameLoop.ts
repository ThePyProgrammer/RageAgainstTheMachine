import { renderFrame } from "@/features/pong/game/renderer";
import type { GameInputState, RuntimeState } from "@/features/pong/types/pongRuntime";
import type { UiSettings } from "@/features/pong/types/pongSettings";

export type OpponentLoopEventPayload = {
  event: "player_score" | "ai_score" | "near_score";
  score: { player: number; ai: number };
  event_context?: { near_side: "player_goal" | "ai_goal"; proximity: number };
};

export interface LoopConfig {
  ctx: CanvasRenderingContext2D;
  runtimeState: RuntimeState;
  settingsRef: { current: UiSettings };
  inputRef: { current: GameInputState };
  onFps: (fps: number) => void;
  onRuntimeUpdate?: (state: RuntimeState) => void;
  onOpponentEvent?: (event: OpponentLoopEventPayload) => void;
  onDebugMetrics?: (payload: LoopDebugPayload) => void;
  isPausedRef?: { current: boolean };
  controlMode?: "paddle" | "ball";
  controlModeRef?: { current: "paddle" | "ball" };
  eegCommandRef?: { current: "none" | "left" | "right" };
  difficultyRef?: { current: number };
}

export const FRAME_MS = 1000 / 60;
export const PADDLE_SPEED = 260;
const PONG_INITIAL_SPEED = 0.5;
const DEFAULT_DIFFICULTY = 0.5;
const NEAR_ZONE_RATIO = 0.18;
const NEAR_SCORE_COOLDOWN_MS = 1200;
const MAX_FRAME_DELTA_MS = 50;
const MAX_STEPS_PER_TICK = 6;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));
const clamp01 = (value: number): number => clamp(value, 0, 1);
const normalize = (value: number, max: number): number => clamp(value, -max, max);
const lerp = (start: number, end: number, t: number): number => start + (end - start) * t;

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
type NearGoalSide = "player_goal" | "ai_goal";

interface StepResult {
  ballVX: number;
  ballVY: number;
  scoreSide: ScoreSide;
  metrics: StepMetrics;
}

type NearThreatState = {
  nearSide: NearGoalSide;
  minDistanceToGoal: number;
};

interface PongDifficultyTuning {
  curve: number;
  ballSpeedX: number;
  ballSpeedY: number;
  ballMaxSpeed: number;
  bounceGain: number;
  aiErrorAmplitude: number;
  aiDeadZone: number;
  aiResponseGain: number;
  aiMaxStepPerSecond: number;
}

const difficultyCurve = (difficulty: number): number => {
  const d = clamp01(difficulty);
  if (d <= 0.8) {
    return 0.78 * (d / 0.8) ** 1.15;
  }
  return 0.78 + 0.22 * ((d - 0.8) / 0.2) ** 2.6;
};

export const getPongDifficultyTuning = (difficulty: number): PongDifficultyTuning => {
  const curve = difficultyCurve(difficulty);
  return {
    curve,
    ballSpeedX: lerp(120, 760, curve) * PONG_INITIAL_SPEED,
    ballSpeedY: lerp(180, 1120, curve) * PONG_INITIAL_SPEED,
    ballMaxSpeed: lerp(320, 1950, curve) * PONG_INITIAL_SPEED,
    bounceGain: lerp(1.01, 1.10, curve),
    aiErrorAmplitude: lerp(95, 4, curve),
    aiDeadZone: lerp(95, 6, curve),
    aiResponseGain: lerp(0.014, 0.18, curve),
    aiMaxStepPerSecond: lerp(90, 440, curve),
  };
};

const DEFAULT_TUNING = getPongDifficultyTuning(DEFAULT_DIFFICULTY);
export const BALL_SPEED_X = DEFAULT_TUNING.ballSpeedX;
export const BALL_SPEED_Y = DEFAULT_TUNING.ballSpeedY;

const distanceToGoalLine = (runtimeState: RuntimeState, nearSide: NearGoalSide): number => {
  if (nearSide === "player_goal") {
    return Math.max(0, runtimeState.ball.y - runtimeState.ball.radius);
  }
  return Math.max(0, runtimeState.height - (runtimeState.ball.y + runtimeState.ball.radius));
};

const resetBall = (
  runtimeState: RuntimeState,
  previousBallVX: number,
  direction: number,
  tuning: PongDifficultyTuning,
): {
  ballVX: number;
  ballVY: number;
} => {
  runtimeState.ball.x = runtimeState.width / 2;
  runtimeState.ball.y = runtimeState.height / 2;

  return {
    ballVX: tuning.ballSpeedX * Math.sign(previousBallVX || 1),
    ballVY: tuning.ballSpeedY * direction,
  };
};

export const stepPongPhysics = (
  runtimeState: RuntimeState,
  inputState: GameInputState,
  controlMode: "paddle" | "ball",
  currentBallVX: number,
  currentBallVY: number,
  difficulty: number = DEFAULT_DIFFICULTY,
): StepResult => {
  const tuning = getPongDifficultyTuning(difficulty);
  let ballVX = currentBallVX;
  let ballVY = currentBallVY;
  let scoreSide: ScoreSide = null;
  let positionClampedCount = 0;
  let collisionResolvedCount = 0;
  const collisionNormals: CollisionNormal[] = [];

  // Player paddle movement is horizontal (left/right).
  if (controlMode === "paddle") {
    const moveLeft = inputState.left || inputState.up;
    const moveRight = inputState.right || inputState.down;
    runtimeState.topPaddle.x +=
      ((moveRight ? 1 : 0) - (moveLeft ? 1 : 0)) * PADDLE_SPEED * (FRAME_MS / 1000);
  }

  const phase =
    runtimeState.ball.x * 0.013 + runtimeState.ball.y * 0.017 + runtimeState.aiScore * 0.11;
  const trackingBias = Math.sin(phase) * tuning.aiErrorAmplitude;
  const bottomTarget = runtimeState.ball.x - runtimeState.bottomPaddle.width / 2 + trackingBias;
  const bottomDelta = bottomTarget - runtimeState.bottomPaddle.x;
  const deadZoneAdjustedDelta =
    Math.abs(bottomDelta) <= tuning.aiDeadZone
      ? 0
      : bottomDelta - Math.sign(bottomDelta) * tuning.aiDeadZone;
  const bottomUnclampedX =
    runtimeState.bottomPaddle.x +
    clamp(
      deadZoneAdjustedDelta * tuning.aiResponseGain,
      -(tuning.aiMaxStepPerSecond * (FRAME_MS / 1000)),
      tuning.aiMaxStepPerSecond * (FRAME_MS / 1000),
    );
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

  if (controlMode === "ball") {
    const keyX = (inputState.right ? 1 : 0) - (inputState.left ? 1 : 0);
    const keyY = (inputState.down ? 1 : 0) - (inputState.up ? 1 : 0);
    const px = normalize(Number(inputState.pointerX ?? 0), 1);
    const py = normalize(Number(inputState.pointerY ?? 0), 1);
    const dx = keyX || px;
    const dy = keyY || py;

    if (dx !== 0 || dy !== 0) {
      runtimeState.ball.x += dx * tuning.ballSpeedX * (FRAME_MS / 1000);
      runtimeState.ball.y += dy * tuning.ballSpeedY * (FRAME_MS / 1000);
    } else {
      runtimeState.ball.x += ballVX * (FRAME_MS / 1000);
      runtimeState.ball.y += ballVY * (FRAME_MS / 1000);
    }
  } else {
    runtimeState.ball.x += ballVX * (FRAME_MS / 1000);
    runtimeState.ball.y += ballVY * (FRAME_MS / 1000);
  }

  const radius = runtimeState.ball.radius;
  if (runtimeState.ball.x - radius <= 0) {
    ballVX *= -tuning.bounceGain;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 1, y: 0 });
  } else if (runtimeState.ball.x + radius >= runtimeState.width) {
    ballVX *= -tuning.bounceGain;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: -1, y: 0 });
  }

  const nextBallX = runtimeState.ball.x;
  const nextBallY = runtimeState.ball.y;
  const tp = runtimeState.topPaddle;
  const bp = runtimeState.bottomPaddle;

  if (
    ballVY < 0 &&
    nextBallY - radius <= tp.y + tp.height &&
    nextBallY + radius >= tp.y &&
    nextBallX >= tp.x &&
    nextBallX <= tp.x + tp.width
  ) {
    ballVY *= -tuning.bounceGain;
    runtimeState.ball.y = tp.y + tp.height + radius + 1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 0, y: 1 });
  }

  if (
    ballVY > 0 &&
    nextBallY + radius >= bp.y &&
    nextBallY - radius <= bp.y + bp.height &&
    nextBallX >= bp.x &&
    nextBallX <= bp.x + bp.width
  ) {
    ballVY *= -tuning.bounceGain;
    runtimeState.ball.y = bp.y - radius - 1;
    collisionResolvedCount += 1;
    collisionNormals.push({ x: 0, y: -1 });
  }

  const maxXSpeed = tuning.ballMaxSpeed * 0.92;
  ballVX = clamp(ballVX, -maxXSpeed, maxXSpeed);
  ballVY = clamp(ballVY, -tuning.ballMaxSpeed, tuning.ballMaxSpeed);

  if (runtimeState.ball.y < -radius) {
    runtimeState.aiScore += 1;
    scoreSide = "ai";
    ({ ballVX, ballVY } = resetBall(runtimeState, ballVX, -1, tuning));
  } else if (runtimeState.ball.y > runtimeState.height + radius) {
    runtimeState.playerScore += 1;
    scoreSide = "player";
    ({ ballVX, ballVY } = resetBall(runtimeState, ballVX, 1, tuning));
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
    onOpponentEvent,
    onDebugMetrics,
    isPausedRef,
    controlMode = "paddle",
    controlModeRef,
    eegCommandRef,
    difficultyRef,
  } = cfg;

  const getControlMode = () => controlModeRef?.current ?? controlMode;
  const getDifficulty = () => clamp01(difficultyRef?.current ?? DEFAULT_DIFFICULTY);
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

  let ballVX = getPongDifficultyTuning(getDifficulty()).ballSpeedX;
  let ballVY = getPongDifficultyTuning(getDifficulty()).ballSpeedY;
  let nearThreat: NearThreatState | null = null;
  let nearScoreCooldownUntilMs = 0;

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
      const stepNowMs = now - accumulator;
      const stepResult = stepPongPhysics(
        runtimeState,
        {
          ...inputRef.current,
          left: inputRef.current.left || eegCommandRef?.current === "left",
          right: inputRef.current.right || eegCommandRef?.current === "right",
        },
        getControlMode(),
        ballVX,
        ballVY,
        getDifficulty(),
      );

      ballVX = stepResult.ballVX;
      ballVY = stepResult.ballVY;
      collisionResolvedThisFrame += stepResult.metrics.collisionResolvedCount;
      clampedThisFrame += stepResult.metrics.positionClampedCount;
      debugWindowCollisionEvents += stepResult.metrics.collisionResolvedCount;
      debugWindowClampEvents += stepResult.metrics.positionClampedCount;
      collisionNormalsThisFrame.push(...stepResult.metrics.collisionNormals);

      if (stepResult.scoreSide) {
        onOpponentEvent?.({
          event: stepResult.scoreSide === "player" ? "player_score" : "ai_score",
          score: {
            player: runtimeState.playerScore,
            ai: runtimeState.aiScore,
          },
        });
        nearThreat = null;
        nearScoreCooldownUntilMs = stepNowMs + NEAR_SCORE_COOLDOWN_MS;
        onRuntimeUpdate?.(runtimeState);
      } else {
        if (!nearThreat && stepNowMs >= nearScoreCooldownUntilMs) {
          const nearZoneHeight = runtimeState.height * NEAR_ZONE_RATIO;
          if (ballVY < 0 && runtimeState.ball.y <= nearZoneHeight) {
            nearThreat = {
              nearSide: "player_goal",
              minDistanceToGoal: distanceToGoalLine(runtimeState, "player_goal"),
            };
          } else if (ballVY > 0 && runtimeState.ball.y >= runtimeState.height - nearZoneHeight) {
            nearThreat = {
              nearSide: "ai_goal",
              minDistanceToGoal: distanceToGoalLine(runtimeState, "ai_goal"),
            };
          }
        }

        if (nearThreat) {
          const nearZoneHeight = runtimeState.height * NEAR_ZONE_RATIO;
          nearThreat.minDistanceToGoal = Math.min(
            nearThreat.minDistanceToGoal,
            distanceToGoalLine(runtimeState, nearThreat.nearSide),
          );

          const hasEscaped =
            (nearThreat.nearSide === "player_goal" && ballVY > 0) ||
            (nearThreat.nearSide === "ai_goal" && ballVY < 0);

          if (hasEscaped) {
            const proximity = clamp(1 - nearThreat.minDistanceToGoal / nearZoneHeight, 0, 1);
            onOpponentEvent?.({
              event: "near_score",
              score: {
                player: runtimeState.playerScore,
                ai: runtimeState.aiScore,
              },
              event_context: {
                near_side: nearThreat.nearSide,
                proximity,
              },
            });
            nearThreat = null;
            nearScoreCooldownUntilMs = stepNowMs + NEAR_SCORE_COOLDOWN_MS;
          }
        }
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
