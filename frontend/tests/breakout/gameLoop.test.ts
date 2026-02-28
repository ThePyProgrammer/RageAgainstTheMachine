import { describe, expect, it } from "vitest";
import {
  createInitialRuntimeState,
  getBallSpeedMultiplier,
  getBlocksForLevel,
  stepBreakoutPhysics,
} from "../../src/features/breakout/game/gameLoop";
import type { GameInputState } from "../../src/features/breakout/types/breakoutRuntime";

const neutralInput = (): GameInputState => ({
  left: false,
  right: false,
});

describe("breakout progression rules", () => {
  it("starts with 12 blocks and caps at 32 blocks per level", () => {
    expect(getBlocksForLevel(1)).toBe(12);
    expect(getBlocksForLevel(2)).toBe(16);
    expect(getBlocksForLevel(3)).toBe(20);
    expect(getBlocksForLevel(6)).toBe(32);
    expect(getBlocksForLevel(20)).toBe(32);
  });

  it("increases speed by +5% per level and caps at +40%", () => {
    expect(getBallSpeedMultiplier(1)).toBeCloseTo(1, 6);
    expect(getBallSpeedMultiplier(2)).toBeCloseTo(1.05, 6);
    expect(getBallSpeedMultiplier(5)).toBeCloseTo(1.2, 6);
    expect(getBallSpeedMultiplier(9)).toBeCloseTo(1.4, 6);
    expect(getBallSpeedMultiplier(30)).toBeCloseTo(1.4, 6);
  });

  it("levels up and regenerates the next block count when the last brick is cleared", () => {
    const state = createInitialRuntimeState();
    state.level = 1;
    state.bricks = [
      {
        id: 1,
        x: 400,
        y: 120,
        width: 100,
        height: 20,
        active: true,
      },
    ];
    state.bricksRemaining = 1;
    state.ball.x = 450;
    state.ball.y = 130;
    state.ball.radius = 10;

    const step = stepBreakoutPhysics(state, neutralInput(), 0, -320);

    expect(step.didLevelUp).toBe(true);
    expect(state.level).toBe(2);
    expect(state.bricksRemaining).toBe(16);
    expect(state.bricks.length).toBe(16);
  });
});
