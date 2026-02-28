import { clamp } from "@ragemachine/bci-shared";
import type { CombatState, InputSample, SimulationConfig } from "./types";
import { resolveBarrierCollision, resolveProjectileBarrierHits, type Barrier, type BarrierBreakEvent } from "./barriers";

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

interface ProjectileResult {
  projectiles: CombatState["projectiles"];
  playerHits: number;
  enemyHits: number;
}

const HIT_RADIUS_SQ = 4; // distance² threshold for hit detection

const updateProjectiles = (
  projectiles: CombatState["projectiles"],
  state: CombatState,
  config: SimulationConfig,
  dt: number,
): ProjectileResult => {
  let playerHits = 0;
  let enemyHits = 0;

  const updated = projectiles
    .map((projectile) => ({
      ...projectile,
      x: projectile.x + projectile.vx * dt,
      y: projectile.y + projectile.vy * dt,
      ttlMs: projectile.ttlMs - dt * 1000,
    }))
    .filter((projectile) => projectile.ttlMs > 0);

  if (updated.length === 0) {
    return { projectiles: updated, playerHits: 0, enemyHits: 0 };
  }

  const surviving = updated.filter((projectile) => {
    const target = projectile.owner === "player" ? state.enemy : state.player;
    const distX = target.x - projectile.x;
    const distY = target.y - projectile.y;
    const distSq = distX * distX + distY * distY;
    if (distSq <= HIT_RADIUS_SQ) {
      if (projectile.owner === "player") playerHits++;
      else enemyHits++;
      return false; // remove projectile on hit
    }
    return true;
  });

  return { projectiles: surviving, playerHits, enemyHits };
};

export interface AdvanceResult {
  state: CombatState;
  newBreakEvents: BarrierBreakEvent[];
}

export const advanceSimulation = (
  current: CombatState,
  playerInput: InputSample,
  enemyInput: InputSample,
  config: SimulationConfig,
  dt: number,
  barriers?: Barrier[],
): AdvanceResult => {
  let nextPlayer = integrateVehicle(current.player, playerInput, config, dt);
  let nextEnemy = integrateVehicle(current.enemy, enemyInput, config, dt);

  // ── Barrier collision (blocks vehicles, no damage) ──
  const breakEvents: BarrierBreakEvent[] = [];
  if (barriers && barriers.length > 0) {
    const pResolved = resolveBarrierCollision(nextPlayer.x, nextPlayer.y, barriers);
    nextPlayer = { ...nextPlayer, x: pResolved.x, y: pResolved.y };
    const eResolved = resolveBarrierCollision(nextEnemy.x, nextEnemy.y, barriers);
    nextEnemy = { ...nextEnemy, x: eResolved.x, y: eResolved.y };
  }

  const nextProjectiles: CombatState["projectiles"] = [...current.projectiles];

  if (playerInput.fire && current.player.cooldownMs <= 0) {
    nextProjectiles.push(spawnProjectile(nextPlayer, "player", config));
  }

  if (enemyInput.fire && current.enemy.cooldownMs <= 0) {
    nextProjectiles.push(spawnProjectile(nextEnemy, "enemy", config));
  }

  const projectileResult = updateProjectiles(
    nextProjectiles.map((projectile) => ({
      ...projectile,
    })),
    { ...current, player: nextPlayer, enemy: nextEnemy },
    config,
    dt,
  );

  // ── Projectile-barrier hits (damages barriers, consumes bullet) ──
  let finalProjectiles = projectileResult.projectiles;
  if (barriers && barriers.length > 0) {
    const simTimeMs = current.simTimeMs + dt * 1000;
    finalProjectiles = resolveProjectileBarrierHits(finalProjectiles, barriers, simTimeMs, breakEvents);
  }

  const playerCooldown = Math.max(0, (playerInput.fire ? 220 : current.player.cooldownMs) - dt * 1000);
  const enemyCooldown = Math.max(0, (enemyInput.fire ? 220 : current.enemy.cooldownMs) - dt * 1000);

  const nextDifficulty = {
    ...current.difficulty,
    stress: clamp(current.difficulty.stress + playerInput.throttle * 0.001 + enemyInput.throttle * 0.001, 0, 1),
    aggression: clamp(current.difficulty.aggression + (playerInput.fire ? 0.001 : -0.0005), 0, 1),
    reactionMs: 180 + current.difficulty.aggression * 120,
  };

  return {
    state: {
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
      projectiles: finalProjectiles,
      score: {
        player: current.score.player + projectileResult.playerHits,
        enemy: current.score.enemy + projectileResult.enemyHits,
      },
      difficulty: nextDifficulty,
    },
    newBreakEvents: breakEvents,
  };
};
