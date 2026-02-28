export interface InputSample {
  timestamp: number;
  throttle: number;
  turn: number;
  fire: boolean;
}

export interface CombatVehicleState {
  x: number;
  y: number;
  yaw: number;
  vx: number;
  vy: number;
  cooldownMs: number;
  speedBoostMs: number;
}

export interface Projectile {
  x: number;
  y: number;
  vx: number;
  vy: number;
  ttlMs: number;
  owner: "player" | "enemy";
}

export interface CombatDifficultyState {
  stress: number;
  aggression: number;
  reactionMs: number;
}

export interface CombatState {
  tick: number;
  simTimeMs: number;
  player: CombatVehicleState;
  enemy: CombatVehicleState;
  projectiles: Projectile[];
  score: {
    player: number;
    enemy: number;
  };
  difficulty: CombatDifficultyState;
}

export interface SimulationConfig {
  worldSize: number;
  turnSpeed: number;
  thrust: number;
  bulletSpeed: number;
  bulletTTL: number;
  seed: number;
}

export type TickResult = CombatState;
