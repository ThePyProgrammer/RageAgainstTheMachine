import type { CombatDifficultyState, CombatState, InputSample } from "../engine/types";

export interface DifficultySignal {
  scoreDelta: number;
  stress: number;
  aggression: number;
}

export const updateDifficulty = (state: CombatState, playerScore: number, enemyScore: number): CombatDifficultyState => {
  const scoreDelta = playerScore - enemyScore;
  const stress = Math.min(1, Math.max(0, 0.55 + scoreDelta * -0.08 + state.difficulty.stress * 0.4));
  const aggression = Math.min(1, Math.max(0, state.difficulty.aggression + (playerScore > enemyScore ? -0.01 : 0.02)));

  return {
    stress,
    aggression,
    reactionMs: 220 + aggression * 180,
  };
};

export const createOpponentInput = (
  playerPos: { x: number; y: number },
  enemyState: { x: number; y: number; yaw: number },
  difficulty: CombatDifficultyState,
  rng: { nextSignedRange: () => number },
  timeMs: number,
): InputSample => {
  const aimX = playerPos.x - enemyState.x;
  const aimY = playerPos.y - enemyState.y;
  const yawToTarget = Math.atan2(aimY, aimX);
  const yawDelta = ((yawToTarget - enemyState.yaw + Math.PI) % (Math.PI * 2)) - Math.PI;

  const jitter = rng.nextSignedRange() * (0.4 + difficulty.aggression * 0.4);
  const turn = Math.max(-1, Math.min(1, yawDelta * 0.25 + jitter));
  const throttle = Math.min(1, Math.max(-1, Math.abs(aimX) > 8 || Math.abs(aimY) > 8 ? 0.35 : 0.02));
  const fire = Math.abs(yawDelta) < (1 - difficulty.reactionMs / 500);

  return {
    timestamp: timeMs,
    throttle,
    turn,
    fire,
  };
};
