import { clamp } from "@ragemachine/bci-shared";
import type { CombatState, InputSample, SimulationConfig } from "./types";

const WORLD_HALF = 50;

export const createInitialState = (seed: number): CombatState => ({
  tick: 0,
  simTimeMs: 0,
  player: {
    x: -22,
    y: 0,
    yaw: 0,
    vx: 0,
    vy: 0,
    cooldownMs: 0,
    speedBoostMs: 0,
  },
  enemy: {
    x: 22,
    y: 0,
    yaw: Math.PI,
    vx: 0,
    vy: 0,
    cooldownMs: 0,
    speedBoostMs: 0,
  },
  projectiles: [],
  score: {
    player: 0,
    enemy: 0,
  },
  difficulty: {
    stress: 0.55,
    aggression: 0.4,
    reactionMs: 220,
  },
});

const integrateVehicle = (state: CombatState["player"], input: InputSample, config: SimulationConfig, dt: number): CombatState["player"] => {
  const yaw = state.yaw + input.turn * config.turnSpeed * dt;
  const throttle = clamp(input.throttle, -1, 1);
  const ax = Math.cos(yaw) * throttle * config.thrust;
  const ay = Math.sin(yaw) * throttle * config.thrust;
  const vx = state.vx + ax * dt;
  const vy = state.vy + ay * dt;
  const drag = 0.94;
  const nextX = clamp(state.x + vx * dt, -WORLD_HALF, WORLD_HALF);
  const nextY = clamp(state.y + vy * dt, -WORLD_HALF, WORLD_HALF);

  return {
    ...state,
    x: nextX,
    y: nextY,
    yaw,
    vx: vx * drag,
    vy: vy * drag,
  };
};

const spawnProjectile = (vehicle: CombatState["player"], owner: "player" | "enemy", config: SimulationConfig) => ({
  x: vehicle.x,
  y: vehicle.y,
  vx: Math.cos(vehicle.yaw) * config.bulletSpeed,
  vy: Math.sin(vehicle.yaw) * config.bulletSpeed,
  ttlMs: config.bulletTTL,
  owner,
});

const updateProjectiles = (projectiles: CombatState["projectiles"], state: CombatState, config: SimulationConfig, dt: number): CombatState["projectiles"] => {
  const updated = projectiles
    .map((projectile) => ({
      ...projectile,
      x: projectile.x + projectile.vx * dt,
      y: projectile.y + projectile.vy * dt,
      ttlMs: projectile.ttlMs - dt * 1000,
    }))
    .filter((projectile) => projectile.ttlMs > 0);

  if (updated.length === 0) {
    return updated;
  }

  const filtered = updated.filter((projectile) => {
    const enemy = projectile.owner === "player" ? state.enemy : state.player;
    const distX = enemy.x - projectile.x;
    const distY = enemy.y - projectile.y;
    const distSq = distX * distX + distY * distY;
    return distSq > 2;
  });

  return filtered;
};

export const advanceSimulation = (
  current: CombatState,
  playerInput: InputSample,
  enemyInput: InputSample,
  config: SimulationConfig,
  dt: number,
): CombatState => {
  const nextPlayer = integrateVehicle(current.player, playerInput, config, dt);
  const nextEnemy = integrateVehicle(current.enemy, enemyInput, config, dt);

  const nextProjectiles: CombatState["projectiles"] = [...current.projectiles];

  if (playerInput.fire && current.player.cooldownMs <= 0) {
    nextProjectiles.push(spawnProjectile(nextPlayer, "player", config));
  }

  if (enemyInput.fire && current.enemy.cooldownMs <= 0) {
    nextProjectiles.push(spawnProjectile(nextEnemy, "enemy", config));
  }

  const movedProjectiles = updateProjectiles(
    nextProjectiles.map((projectile) => ({
      ...projectile,
    })),
    { ...current, player: nextPlayer, enemy: nextEnemy },
    config,
    dt,
  );

  const playerCooldown = Math.max(0, (playerInput.fire ? 220 : current.player.cooldownMs) - dt * 1000);
  const enemyCooldown = Math.max(0, (enemyInput.fire ? 220 : current.enemy.cooldownMs) - dt * 1000);

  const nextDifficulty = {
    ...current.difficulty,
    stress: clamp(current.difficulty.stress + playerInput.throttle * 0.001 + enemyInput.throttle * 0.001, 0, 1),
    aggression: clamp(current.difficulty.aggression + (playerInput.fire ? 0.001 : -0.0005), 0, 1),
    reactionMs: 180 + current.difficulty.aggression * 120,
  };

  return {
    tick: current.tick + 1,
    simTimeMs: current.simTimeMs + dt * 1000,
    player: {
      ...nextPlayer,
      cooldownMs: playerCooldown,
      speedBoostMs: 0,
    },
    enemy: {
      ...nextEnemy,
      cooldownMs: enemyCooldown,
      speedBoostMs: 0,
    },
    projectiles: movedProjectiles,
    score: current.score,
    difficulty: nextDifficulty,
  };
};
