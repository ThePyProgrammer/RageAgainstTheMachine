import { describe, expect, it } from "vitest";
import {
  BALL_SPEED_X,
  BALL_SPEED_Y,
  getPongDifficultyTuning,
  stepPongPhysics,
} from "../src/features/pong/game/gameLoop";
import type { GameInputState, RuntimeState } from "../src/features/pong/types/pongRuntime";

const createRuntimeState = (overrides?: Partial<RuntimeState>): RuntimeState => ({
  width: 960,
  height: 540,
  ball: { x: 480, y: 270, radius: 10 },
  topPaddle: { x: 425, y: 20, width: 110, height: 14 },      // Player at top
  bottomPaddle: { x: 425, y: 506, width: 110, height: 14 },  // AI at bottom
  playerScore: 0,
  aiScore: 0,
  ...(overrides ?? {}),
});

const neutralInput = (): GameInputState => ({
  up: false,
  down: false,
  left: false,
  right: false,
  pointerX: 0,
  pointerY: 0,
});

describe("pong physics step", () => {
  it("maps difficulty to steeper high-end speed scaling", () => {
    const low = getPongDifficultyTuning(0);
    const mid = getPongDifficultyTuning(0.5);
    const high = getPongDifficultyTuning(0.8);
    const peak = getPongDifficultyTuning(1);

    expect(low.ballSpeedX).toBeLessThan(mid.ballSpeedX);
    expect(mid.ballSpeedX).toBeLessThan(high.ballSpeedX);
    expect(high.ballSpeedX).toBeLessThan(peak.ballSpeedX);

    const highToPeakDelta = peak.ballSpeedX - high.ballSpeedX;
    const midToHighDelta = high.ballSpeedX - mid.ballSpeedX;
    expect(highToPeakDelta).toBeGreaterThan(midToHighDelta * 0.45);
    expect(peak.aiErrorAmplitude).toBeLessThan(high.aiErrorAmplitude);
  });

  it("crosses the field midline within deterministic ticks", () => {
    const state = createRuntimeState({
      ball: { x: 100, y: 60, radius: 10 },
    });
    const input = neutralInput();
    let ballVX = BALL_SPEED_X;
    let ballVY = BALL_SPEED_Y;
    let crossedMidline = false;

    for (let i = 0; i < 240; i += 1) {
      const step = stepPongPhysics(state, input, "paddle", ballVX, ballVY);
      ballVX = step.ballVX;
      ballVY = step.ballVY;
      if (state.ball.x >= state.width / 2) {
        crossedMidline = true;
        break;
      }
    }

    expect(crossedMidline).toBe(true);
  });

  it("inverts ball velocity only on wall/paddle contacts", () => {
    // Test left wall collision (ball bounces off left wall)
    const leftWallState = createRuntimeState({
      ball: { x: 1, y: 270, radius: 10 },
    });
    const input = neutralInput();
    const leftWall = stepPongPhysics(leftWallState, input, "paddle", -170, BALL_SPEED_Y);
    expect(leftWall.ballVX).toBeGreaterThan(0);
    expect(leftWall.metrics.collisionResolvedCount).toBeGreaterThan(0);
    expect(leftWall.metrics.collisionNormals.some((normal) => normal.x === 1 && normal.y === 0)).toBe(true);

    // Test top paddle collision (ball moving up hits bottom edge of top paddle)
    const paddleState = createRuntimeState({
      ball: { x: 480, y: 44, radius: 10 },
      topPaddle: { x: 425, y: 20, width: 110, height: 14 },
    });
    const paddleHit = stepPongPhysics(paddleState, input, "paddle", BALL_SPEED_X, -BALL_SPEED_Y);
    expect(paddleHit.ballVY).toBeGreaterThan(0);
    expect(paddleHit.metrics.collisionResolvedCount).toBeGreaterThan(0);
    expect(paddleHit.metrics.collisionNormals.some((normal) => normal.x === 0 && normal.y === 1)).toBe(
      true,
    );
  });

  it("expands ball bounding box beyond threshold across time (no micro-bounce)", () => {
    // Use a more square field since ball now travels vertically (faster Y, slower X)
    const state = createRuntimeState({
      width: 2400,
      height: 4800,
      ball: { x: 1200, y: 2400, radius: 10 },
      topPaddle: { x: 1145, y: 20, width: 110, height: 14 },
      bottomPaddle: { x: 1145, y: 4766, width: 110, height: 14 },
    });

    let minX = state.ball.x;
    let maxX = state.ball.x;
    let minY = state.ball.y;
    let maxY = state.ball.y;
    let ballVX = BALL_SPEED_X;
    let ballVY = BALL_SPEED_Y;
    const input = neutralInput();

    // Run more iterations to ensure ball covers sufficient area
    for (let i = 0; i < 3000; i += 1) {
      const step = stepPongPhysics(state, input, "paddle", ballVX, ballVY);
      ballVX = step.ballVX;
      ballVY = step.ballVY;
      minX = Math.min(minX, state.ball.x);
      maxX = Math.max(maxX, state.ball.x);
      minY = Math.min(minY, state.ball.y);
      maxY = Math.max(maxY, state.ball.y);
    }

    expect(maxX - minX).toBeGreaterThan(state.width * 0.2);
    expect(maxY - minY).toBeGreaterThan(state.height * 0.2);
  });

  it("keeps physics deterministic across repeated runs", () => {
    const stateA = createRuntimeState({
      ball: { x: 300, y: 180, radius: 10 },
      topPaddle: { x: 245, y: 20, width: 110, height: 14 },
      bottomPaddle: { x: 245, y: 506, width: 110, height: 14 },
    });
    const stateB = createRuntimeState({
      ball: { x: 300, y: 180, radius: 10 },
      topPaddle: { x: 245, y: 20, width: 110, height: 14 },
      bottomPaddle: { x: 245, y: 506, width: 110, height: 14 },
    });
    const input = neutralInput();

    let stateABallVX = BALL_SPEED_X;
    let stateABallVY = BALL_SPEED_Y;
    let stateBBallVX = BALL_SPEED_X;
    let stateBBallVY = BALL_SPEED_Y;

    for (let i = 0; i < 240; i += 1) {
      const stepA = stepPongPhysics(stateA, input, "paddle", stateABallVX, stateABallVY);
      const stepB = stepPongPhysics(stateB, input, "paddle", stateBBallVX, stateBBallVY);

      stateABallVX = stepA.ballVX;
      stateABallVY = stepA.ballVY;
      stateBBallVX = stepB.ballVX;
      stateBBallVY = stepB.ballVY;

      expect(stepA.ballVX).toBe(stepB.ballVX);
      expect(stepA.ballVY).toBe(stepB.ballVY);
      expect(stateA.ball.x).toBe(stateB.ball.x);
      expect(stateA.ball.y).toBe(stateB.ball.y);
      expect(stateA.playerScore).toBe(stateB.playerScore);
      expect(stateA.aiScore).toBe(stateB.aiScore);
    }
  });
});
