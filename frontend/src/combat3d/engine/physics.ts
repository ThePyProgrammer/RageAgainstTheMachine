import { clamp } from "@ragemachine/bci-shared";
import type { CombatState, DynamicObstacle, InputSample, SimulationConfig } from "./types";
import { resolveBarrierCollision, resolveProjectileBarrierHits, type Barrier, type BarrierBreakEvent } from "./barriers";

const WORLD_HALF = 50;

/** Gameplay constants */
const PLAYER_MAX_HP = 5;
const ENEMY_MAX_HP = 5;
const PLAYER_MAX_AMMO = 15;
const ENEMY_MAX_AMMO = 999; // enemy has unlimited ammo
const PLAYER_FIRE_COOLDOWN_MS = 333;  // 3 shots/sec
const ENEMY_FIRE_COOLDOWN_MS = 500;   // 2 shots/sec
const RELOAD_DURATION_MS = 2000;      // 2s hold to reload
const ENEMY_SPEED_MULT = 1.05;        // 5% faster than player

/** Low-HP mechanic constants */
const LOW_HP_THRESHOLD = 3;           // below this HP -> speed boost + obstacle gen
const CRITICAL_HP = 1;               // at this HP -> faster obstacle gen
const SPEED_BOOST_PER_LOST_HP = 0.10; // compounding 10% per HP lost below threshold
const OBSTACLE_SPAWN_INTERVAL_LOW_MS = 20_000;  // every 20s when HP < 3
const OBSTACLE_SPAWN_INTERVAL_CRIT_MS = 10_000; // every 10s when HP = 1 (anyone)
const TELEPORT_YAW_THRESHOLD = 4 * Math.PI; // 720° = 2 full turns

/** Player spawn position (used for teleport reset). */
const PLAYER_SPAWN = { x: -22, y: 0, yaw: 0 };

/** Small-obstacle RNG colours. */
const DYN_OBS_COLORS = ["#553322", "#334455", "#225544", "#554422"];

const createVehicle = (
  x: number, y: number, yaw: number, maxHp: number, maxAmmo: number,
): CombatState["player"] => ({
  x, y, yaw, vx: 0, vy: 0,
  cooldownMs: 0, speedBoostMs: 0,
  hp: maxHp, maxHp, ammo: maxAmmo, maxAmmo, reloadMs: 0,
  cumulativeYaw: 0,
});

export const createInitialState = (_seed: number): CombatState => ({
  tick: 0,
  simTimeMs: 0,
  player: createVehicle(-22, 0, 0, PLAYER_MAX_HP, PLAYER_MAX_AMMO),
  enemy: createVehicle(22, 0, Math.PI, ENEMY_MAX_HP, ENEMY_MAX_AMMO),
  projectiles: [],
  score: { player: 0, enemy: 0 },
  difficulty: { stress: 0.55, aggression: 0.4, reactionMs: 220 },
  lastObstacleSpawnMs: 0,
  nextDynamicObstacleId: 1000, // start above static obstacle IDs
  dynamicObstacles: [],
});

const integrateVehicle = (
  state: CombatState["player"], input: InputSample, config: SimulationConfig,
  dt: number, speedMult = 1, turnMult = 1,
): CombatState["player"] => {
  const yawDelta = input.turn * config.turnSpeed * turnMult * dt;
  const yaw = state.yaw + yawDelta;
  const throttle = clamp(input.throttle, -1, 1);
  const thrust = config.thrust * speedMult;
  const ax = Math.cos(yaw) * throttle * thrust;
  const ay = Math.sin(yaw) * throttle * thrust;
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
    cumulativeYaw: state.cumulativeYaw + Math.abs(yawDelta),
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
  // ── Player low-HP speed boost (compounding 10% per HP lost below 3) ──
  const playerHpLost = Math.max(0, LOW_HP_THRESHOLD - current.player.hp);
  const playerSpeedMult = playerHpLost > 0 ? Math.pow(1 + SPEED_BOOST_PER_LOST_HP, playerHpLost) : 1;
  const playerTurnMult = playerSpeedMult; // same boost for rotation

  let nextPlayer = integrateVehicle(current.player, playerInput, config, dt, playerSpeedMult, playerTurnMult);
  let nextEnemy = integrateVehicle(current.enemy, enemyInput, config, dt, ENEMY_SPEED_MULT);

  // ── 720° teleport check for player ──
  if (nextPlayer.cumulativeYaw >= TELEPORT_YAW_THRESHOLD) {
    nextPlayer = {
      ...nextPlayer,
      x: PLAYER_SPAWN.x,
      y: PLAYER_SPAWN.y,
      yaw: PLAYER_SPAWN.yaw,
      vx: 0,
      vy: 0,
      cumulativeYaw: 0, // reset after teleport
    };
  }

  // ── Barrier collision (blocks vehicles, no damage) ──
  const breakEvents: BarrierBreakEvent[] = [];
  if (barriers && barriers.length > 0) {
    const pResolved = resolveBarrierCollision(nextPlayer.x, nextPlayer.y, barriers);
    nextPlayer = { ...nextPlayer, x: pResolved.x, y: pResolved.y };
    const eResolved = resolveBarrierCollision(nextEnemy.x, nextEnemy.y, barriers);
    nextEnemy = { ...nextEnemy, x: eResolved.x, y: eResolved.y };
  }

  const dtMs = dt * 1000;

  // ── Reload logic ──
  // Player: reload starts when reload input is true and ammo < max
  let playerReloadMs = current.player.reloadMs;
  if (playerInput.reload && current.player.ammo < current.player.maxAmmo) {
    playerReloadMs += dtMs;
  } else if (!playerInput.reload) {
    playerReloadMs = 0; // must hold continuously
  }
  let playerAmmo = current.player.ammo;
  if (playerReloadMs >= RELOAD_DURATION_MS) {
    playerAmmo = current.player.maxAmmo;
    playerReloadMs = 0;
  }

  // Enemy: instant reload (unlimited ammo)
  const enemyAmmo = current.enemy.ammo;

  // ── Projectile spawning with ammo + cooldown ──
  const nextProjectiles: CombatState["projectiles"] = [...current.projectiles];

  const playerCanFire = playerInput.fire
    && current.player.cooldownMs <= 0
    && playerAmmo > 0
    && playerReloadMs === 0
    && current.player.hp > 0;
  if (playerCanFire) {
    nextProjectiles.push(spawnProjectile(nextPlayer, "player", config));
    playerAmmo -= 1;
  }

  const enemyCanFire = enemyInput.fire
    && current.enemy.cooldownMs <= 0
    && enemyAmmo > 0
    && current.enemy.hp > 0;
  if (enemyCanFire) {
    nextProjectiles.push(spawnProjectile(nextEnemy, "enemy", config));
  }

  const projectileResult = updateProjectiles(
    nextProjectiles.map((projectile) => ({
      ...projectile,
    })),
    { ...current, player: nextPlayer, enemy: nextEnemy },
    dt,
  );

  // ── Projectile-barrier hits (damages barriers, consumes bullet) ──
  let finalProjectiles = projectileResult.projectiles;
  if (barriers && barriers.length > 0) {
    const simTimeMs = current.simTimeMs + dtMs;
    finalProjectiles = resolveProjectileBarrierHits(finalProjectiles, barriers, simTimeMs, breakEvents);
  }

  // ── Apply HP damage from hits ──
  let playerHp = current.player.hp - projectileResult.enemyHits; // enemy bullets hit player
  let enemyHp = current.enemy.hp - projectileResult.playerHits;  // player bullets hit enemy

  // ── Kill detection & respawn ──
  let scorePlayer = current.score.player;
  let scoreEnemy = current.score.enemy;
  let killEvent: CombatState["lastKillEvent"];

  if (enemyHp <= 0) {
    scorePlayer += 1;
    // Respawn enemy
    nextEnemy = { ...nextEnemy, ...createVehicle(22, 0, Math.PI, ENEMY_MAX_HP, ENEMY_MAX_AMMO) };
    enemyHp = ENEMY_MAX_HP;
    killEvent = { victim: "enemy", tick: current.tick + 1 };
  }
  if (playerHp <= 0) {
    scoreEnemy += 1;
    // Respawn player
    nextPlayer = { ...nextPlayer, ...createVehicle(-22, 0, 0, PLAYER_MAX_HP, PLAYER_MAX_AMMO) };
    playerHp = PLAYER_MAX_HP;
    playerAmmo = PLAYER_MAX_AMMO;
    playerReloadMs = 0;
    killEvent = { victim: "player", tick: current.tick + 1 };
  }

  const playerCooldown = Math.max(0, (playerCanFire ? PLAYER_FIRE_COOLDOWN_MS : current.player.cooldownMs) - dtMs);
  const enemyCooldown = Math.max(0, (enemyCanFire ? ENEMY_FIRE_COOLDOWN_MS : current.enemy.cooldownMs) - dtMs);

  const nextDifficulty = {
    ...current.difficulty,
    stress: clamp(current.difficulty.stress + playerInput.throttle * 0.001 + enemyInput.throttle * 0.001, 0, 1),
    aggression: clamp(current.difficulty.aggression + (playerInput.fire ? 0.001 : -0.0005), 0, 1),
    reactionMs: 180 + current.difficulty.aggression * 120,
  };

  // ── Dynamic obstacle spawning on low HP ──
  const nextSimTimeMs = current.simTimeMs + dtMs;
  const anyoneCritical = playerHp <= CRITICAL_HP || enemyHp <= CRITICAL_HP;
  const playerLow = playerHp > 0 && playerHp < LOW_HP_THRESHOLD;

  let spawnIntervalMs = 0;
  if (anyoneCritical) {
    spawnIntervalMs = OBSTACLE_SPAWN_INTERVAL_CRIT_MS; // 10s
  } else if (playerLow) {
    spawnIntervalMs = OBSTACLE_SPAWN_INTERVAL_LOW_MS; // 20s
  }

  let lastSpawnMs = current.lastObstacleSpawnMs;
  let nextDynId = current.nextDynamicObstacleId;
  const newDynamicObstacles: DynamicObstacle[] = [];

  if (spawnIntervalMs > 0 && nextSimTimeMs - lastSpawnMs >= spawnIntervalMs) {
    // Spawn a small obstacle at a pseudo-random position based on sim time
    const seed = (current.tick * 31337 + nextDynId * 7919) >>> 0;
    const rx = ((seed & 0xffff) / 0xffff) * 2 - 1;        // -1..1
    const rz = (((seed >>> 16) & 0xffff) / 0xffff) * 2 - 1;
    const px = rx * (WORLD_HALF - 10);
    const pz = rz * (WORLD_HALF - 10);
    // Small obstacle: width/depth < 1.5 world units (~15 pixels in 3D equiv.)
    const w = 0.6 + ((seed & 0xff) / 0xff) * 0.8;   // 0.6..1.4
    const h = 0.8 + ((seed >>> 8 & 0xff) / 0xff) * 1.0; // 0.8..1.8
    const d = 0.6 + ((seed >>> 4 & 0xff) / 0xff) * 0.8;
    const color = DYN_OBS_COLORS[nextDynId % DYN_OBS_COLORS.length];

    newDynamicObstacles.push({ id: nextDynId, x: px, z: pz, width: w, height: h, depth: d, color });
    lastSpawnMs = nextSimTimeMs;
    nextDynId += 1;
  }

  const allDynamicObstacles = newDynamicObstacles.length > 0
    ? [...current.dynamicObstacles, ...newDynamicObstacles]
    : current.dynamicObstacles;

  return {
    state: {
      tick: current.tick + 1,
      simTimeMs: nextSimTimeMs,
      player: {
        ...nextPlayer,
        cooldownMs: playerCooldown,
        speedBoostMs: 0,
        hp: playerHp,
        ammo: playerAmmo,
        reloadMs: playerReloadMs,
        maxHp: PLAYER_MAX_HP,
        maxAmmo: PLAYER_MAX_AMMO,
      },
      enemy: {
        ...nextEnemy,
        cooldownMs: enemyCooldown,
        speedBoostMs: 0,
        hp: enemyHp,
        ammo: enemyAmmo,
        reloadMs: 0,
        maxHp: ENEMY_MAX_HP,
        maxAmmo: ENEMY_MAX_AMMO,
      },
      projectiles: finalProjectiles,
      score: { player: scorePlayer, enemy: scoreEnemy },
      difficulty: nextDifficulty,
      lastKillEvent: killEvent,
      lastObstacleSpawnMs: lastSpawnMs,
      nextDynamicObstacleId: nextDynId,
      dynamicObstacles: allDynamicObstacles,
    },
    newBreakEvents: breakEvents,
  };
};
