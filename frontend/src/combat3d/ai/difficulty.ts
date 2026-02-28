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

/** Distance² threshold below which enemy is in shooting range */
const SHOOT_RANGE_SQ = 40 * 40; // 40 units

export const createOpponentInput = (
  playerPos: { x: number; y: number },
  enemyState: { x: number; y: number; yaw: number },
  difficulty: CombatDifficultyState,
  rng: { nextSignedRange: () => number },
  timeMs: number,
): InputSample => {
  const aimX = playerPos.x - enemyState.x;
  const aimY = playerPos.y - enemyState.y;
  const distSq = aimX * aimX + aimY * aimY;
  const yawToTarget = Math.atan2(aimY, aimX);
  const yawDelta = ((yawToTarget - enemyState.yaw + Math.PI) % (Math.PI * 2)) - Math.PI;

  const jitter = rng.nextSignedRange() * (0.4 + difficulty.aggression * 0.4);
  const turn = Math.max(-1, Math.min(1, yawDelta * 0.25 + jitter));

  // Always chase the player — more aggressive throttle
  const throttle = Math.min(1, Math.max(-1, 0.55 + difficulty.aggression * 0.3));

  // Shoot immediately when in range AND roughly aimed at the player
  const aimed = Math.abs(yawDelta) < 0.5;
  const inRange = distSq < SHOOT_RANGE_SQ;
  const fire = aimed && inRange;

  return {
    timestamp: timeMs,
    throttle,
    turn,
    fire,
  };
};
