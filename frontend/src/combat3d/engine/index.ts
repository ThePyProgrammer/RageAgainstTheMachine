import { SeededRNG } from "./seededRng";
import type { CombatState, InputSample, SimulationConfig } from "./types";
import { advanceSimulation } from "./physics";
import type { Barrier, BarrierBreakEvent } from "./barriers";

export const FRAME_MS = 1000 / 60;

export interface SimulationFrameArgs {
  readonly config: SimulationConfig;
  readonly playerInput: InputSample;
  readonly enemyInput: InputSample;
  readonly dtMs: number;
  readonly barriers?: Barrier[];
}

export interface DeterministicStepResult {
  readonly state: CombatState;
  readonly nextTickTimeMs: number;
  readonly newBreakEvents: BarrierBreakEvent[];
}

export const createStateFromSeed = (seed: number): CombatState => {
  const rng = new SeededRNG(seed);
  const state: CombatState = {
    tick: 0,
    simTimeMs: 0,
    player: {
      x: -20,
      y: -10,
      yaw: 0,
      vx: 0,
      vy: 0,
      cooldownMs: 0,
      speedBoostMs: 0,
      hp: 5,
      maxHp: 5,
      ammo: 15,
      maxAmmo: 15,
      reloadMs: 0,
      cumulativeYaw: 0,
    },
    enemy: {
      x: 20,
      y: 10,
      yaw: Math.PI,
      vx: 0,
      vy: 0,
      cooldownMs: 0,
      speedBoostMs: 0,
      hp: 5,
      maxHp: 5,
      ammo: 999,
      maxAmmo: 999,
      reloadMs: 0,
      cumulativeYaw: 0,
    },
    projectiles: [],
    score: {
      player: 0,
      enemy: 0,
    },
    difficulty: {
      stress: rng.nextFloat(),
      aggression: rng.nextFloat() * 0.4,
      reactionMs: 220,
    },
    lastObstacleSpawnMs: 0,
    nextDynamicObstacleId: 1000,
    dynamicObstacles: [],
  };

  return state;
};

export const createCombatConfig = (seed: number): SimulationConfig => ({
  worldSize: 120,
  turnSpeed: 2.0,
  thrust: 25,
  bulletSpeed: 45,
  bulletTTL: 1600,
  seed,
});

export const stepSimulation = (
  state: CombatState,
  args: SimulationFrameArgs,
  rng: SeededRNG,
): DeterministicStepResult => {
  const dt = args.dtMs / 1000;
  const playerInput = args.playerInput;
  const enemyInput = args.enemyInput;

  const advanceResult = advanceSimulation(
    state,
    playerInput,
    enemyInput,
    args.config,
    dt,
    args.barriers,
  );
  const nextState = advanceResult.state;

  const randomShift = rng.nextSignedRange() * 0.001;
  const nextDifficulty = {
    ...nextState.difficulty,
    stress: Math.min(1, Math.max(0, nextState.difficulty.stress + randomShift)),
  };

  return {
    state: {
      ...nextState,
      difficulty: nextDifficulty,
    },
    nextTickTimeMs: state.simTimeMs + args.dtMs,
    newBreakEvents: advanceResult.newBreakEvents,
  };
};
