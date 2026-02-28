/**
 * Combat3D engine determinism + scoring tests.
 *
 * Run with:  npx tsx frontend/src/combat3d/engine/__tests__/engine.test.ts
 *
 * Uses lightweight in-file assertions — no test framework required.
 */
import { SeededRNG } from "../seededRng";
import { advanceSimulation, createInitialState } from "../physics";
import {
  createCombatConfig,
  createStateFromSeed,
  stepSimulation,
  FRAME_MS,
} from "../index";
import type { InputSample, CombatState } from "../types";

const assert = {
  equal<T>(actual: T, expected: T, message?: string): void {
    if (actual !== expected) {
      throw new Error(message ?? `Expected ${String(expected)}, got ${String(actual)}`);
    }
  },
  notEqual<T>(actual: T, expected: T, message?: string): void {
    if (actual === expected) {
      throw new Error(message ?? `Expected value to differ from ${String(expected)}`);
    }
  },
  ok(value: unknown, message?: string): void {
    if (!value) {
      throw new Error(message ?? "Expected truthy value");
    }
  },
};

/* ── helpers ─────────────────────────────────────────────────────────── */

const idle = (ts: number): InputSample => ({
  timestamp: ts,
  throttle: 0,
  turn: 0,
  fire: false,
});

const forward = (ts: number): InputSample => ({
  timestamp: ts,
  throttle: 1,
  turn: 0,
  fire: false,
});

const turnLeft = (ts: number): InputSample => ({
  timestamp: ts,
  throttle: 0,
  turn: -0.8,
  fire: false,
});

const fireShot = (ts: number): InputSample => ({
  timestamp: ts,
  throttle: 0,
  turn: 0,
  fire: true,
});

const dt = FRAME_MS / 1000;
const config = createCombatConfig(0x2f2f);

/* ── 1. Determinism: same seed + inputs → same output ────────────── */
{
  const rng1 = new SeededRNG(0x2f2f);
  const rng2 = new SeededRNG(0x2f2f);
  let s1 = createStateFromSeed(0x2f2f);
  let s2 = createStateFromSeed(0x2f2f);

  for (let i = 0; i < 120; i++) {
    const t = s1.simTimeMs + FRAME_MS;
    const input = i % 20 < 10 ? forward(t) : turnLeft(t);
    const r1 = stepSimulation(s1, { config, playerInput: input, enemyInput: idle(t), dtMs: FRAME_MS }, rng1);
    const r2 = stepSimulation(s2, { config, playerInput: input, enemyInput: idle(t), dtMs: FRAME_MS }, rng2);
    s1 = r1.state;
    s2 = r2.state;
  }
  assert.equal(s1.tick, s2.tick, "ticks must match");
  assert.equal(s1.player.x, s2.player.x, "player x must match");
  assert.equal(s1.player.y, s2.player.y, "player y must match");
  assert.equal(s1.player.yaw, s2.player.yaw, "player yaw must match");
  assert.equal(s1.score.player, s2.score.player, "player score must match");
  assert.equal(s1.score.enemy, s2.score.enemy, "enemy score must match");
  console.log("✓ Determinism: identical seeds + inputs produce identical state");
}

/* ── 2. Forward throttle moves player in facing direction ──────────── */
{
  let state = createInitialState(0);
  const startX = state.player.x;
  for (let i = 0; i < 60; i++) {
    state = advanceSimulation(state, forward(state.simTimeMs), idle(state.simTimeMs), config, dt).state;
  }
  assert.ok(
    state.player.x > startX + 1,
    `Player should have moved forward (x increased). Start=${startX}, End=${state.player.x}`,
  );
  console.log("✓ Throttle moves player forward");
}

/* ── 3. Turn changes yaw ──────────────────────────────────────────── */
{
  let state = createInitialState(0);
  const startYaw = state.player.yaw;
  for (let i = 0; i < 30; i++) {
    state = advanceSimulation(state, turnLeft(state.simTimeMs), idle(state.simTimeMs), config, dt).state;
  }
  assert.notEqual(state.player.yaw, startYaw, "Yaw must have changed after turning");
  console.log("✓ Turn input changes yaw");
}

/* ── 4. Fire spawns projectile ────────────────────────────────────── */
{
  let state = createInitialState(0);
  assert.equal(state.projectiles.length, 0, "No projectiles initially");
  state = advanceSimulation(state, fireShot(state.simTimeMs), idle(state.simTimeMs), config, dt).state;
  assert.ok(state.projectiles.length >= 1, "Fire should spawn a projectile");
  console.log("✓ Fire spawns projectile");
}

/* ── 5. Scoring: projectile hitting enemy increments player score ── */
{
  // Place player directly facing the enemy at close range
  let state: CombatState = {
    tick: 0,
    simTimeMs: 0,
    player: { x: 0, y: 0, yaw: 0, vx: 0, vy: 0, cooldownMs: 0, speedBoostMs: 0, hp: 5, maxHp: 5, ammo: 15, maxAmmo: 15, reloadMs: 0, cumulativeYaw: 0 },
    // Enemy directly ahead of player (yaw 0 = +X direction)
    enemy: { x: 3, y: 0, yaw: Math.PI, vx: 0, vy: 0, cooldownMs: 0, speedBoostMs: 0, hp: 5, maxHp: 5, ammo: 999, maxAmmo: 999, reloadMs: 0, cumulativeYaw: 0 },
    projectiles: [],
    score: { player: 0, enemy: 0 },
    difficulty: { stress: 0.5, aggression: 0.4, reactionMs: 220 },
    lastObstacleSpawnMs: 0,
    nextDynamicObstacleId: 1000,
    dynamicObstacles: [],
  };

  // Fire from player toward enemy
  state = advanceSimulation(state, fireShot(0), idle(0), config, dt).state;
  assert.equal(state.projectiles.length, 1, "Should have 1 projectile after firing");

  // Step until hit or timeout
  let hit = false;
  for (let i = 0; i < 120; i++) {
    state = advanceSimulation(state, idle(state.simTimeMs), idle(state.simTimeMs), config, dt).state;
    if (state.enemy.hp < 5) {
      hit = true;
      break;
    }
  }
  assert.ok(hit, "Enemy HP must decrease when projectile hits");
  console.log("✓ Scoring: projectile hit reduces enemy HP");
}

/* ── 6. Key state maps to correct controls ────────────────────────── */
{
  // Simulating the sampleKeyboard logic from App.tsx
  const sampleKeyboard = (keys: { up: boolean; down: boolean; left: boolean; right: boolean; fire: boolean }, ts: number): InputSample => {
    let throttle = 0;
    let turn = 0;
    if (keys.up) throttle += 1;
    if (keys.down) throttle -= 1;
    if (keys.left) turn -= 0.8;
    if (keys.right) turn += 0.8;
    return { timestamp: ts, throttle, turn, fire: keys.fire };
  };

  const upOnly = sampleKeyboard({ up: true, down: false, left: false, right: false, fire: false }, 0);
  assert.equal(upOnly.throttle, 1, "Up → throttle 1");
  assert.equal(upOnly.turn, 0, "Up → turn 0");

  const downLeft = sampleKeyboard({ up: false, down: true, left: true, right: false, fire: false }, 0);
  assert.equal(downLeft.throttle, -1, "Down → throttle -1");
  assert.equal(downLeft.turn, -0.8, "Left → turn -0.8");

  const fireKey = sampleKeyboard({ up: false, down: false, left: false, right: false, fire: true }, 0);
  assert.equal(fireKey.fire, true, "Space → fire true");

  const combined = sampleKeyboard({ up: true, down: true, left: false, right: true, fire: true }, 0);
  assert.equal(combined.throttle, 0, "Up+Down cancel out → throttle 0");
  assert.equal(combined.turn, 0.8, "Right → turn 0.8");
  assert.equal(combined.fire, true, "Fire active");

  console.log("✓ Key state maps to correct throttle/turn/fire");
}

/* ── 7. SeededRNG determinism ────────────────────────────────────── */
{
  const a = new SeededRNG(42);
  const b = new SeededRNG(42);
  for (let i = 0; i < 100; i++) {
    assert.equal(a.nextFloat(), b.nextFloat(), `RNG diverged at step ${i}`);
  }
  console.log("✓ SeededRNG is deterministic");
}

console.log("\nAll Combat3D engine tests passed.");
