export interface InputSample {
  timestamp: number;
  throttle: number;
  turn: number;
  fire: boolean;
  reload?: boolean;
}

export interface CombatVehicleState {
  x: number;
  y: number;
  yaw: number;
  vx: number;
  vy: number;
  cooldownMs: number;
  speedBoostMs: number;
  hp: number;
  maxHp: number;
  ammo: number;
  maxAmmo: number;
  reloadMs: number; // >0 means currently reloading
  /** Cumulative yaw change since last reset (for 720° teleport). */
  cumulativeYaw: number;
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

/** A small obstacle spawned dynamically during low-HP phases. */
export interface DynamicObstacle {
  id: number;
  x: number;
  z: number;
  width: number;
  height: number;
  depth: number;
  color: string;
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
  /** Set on the tick a vehicle was killed & respawned */
  lastKillEvent?: { victim: "player" | "enemy"; tick: number };
  /** Sim-time of the last dynamically-spawned obstacle (ms). */
  lastObstacleSpawnMs: number;
  /** Counter for unique dynamic-obstacle IDs. */
  nextDynamicObstacleId: number;
  /** Obstacles spawned at runtime during low-HP phases. */
  dynamicObstacles: DynamicObstacle[];
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
