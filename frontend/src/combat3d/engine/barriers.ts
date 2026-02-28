import type { Projectile } from "./types";

export interface Barrier {
  id: number;
  x: number;
  z: number;
  halfW: number;
  halfD: number;
  hp: number;
}

export interface BarrierBreakEvent {
  id: number;
  x: number;
  z: number;
  halfW: number;
  halfD: number;
  timeMs: number;
}

interface ObstacleLike {
  x: number;
  z: number;
  width: number;
  depth: number;
}

export const barriersFromObstacles = (
  obstacles: ReadonlyArray<ObstacleLike>,
): Barrier[] =>
  obstacles.map((obstacle, index) => ({
    id: index,
    x: obstacle.x,
    z: obstacle.z,
    halfW: obstacle.width / 2,
    halfD: obstacle.depth / 2,
    hp: 3,
  }));

const VEHICLE_RADIUS = 1;
const BULLET_HIT_RADIUS_SQ = 1.5 * 1.5;

export const resolveBarrierCollision = (
  vx: number,
  vy: number,
  barriers: Barrier[],
): { x: number; y: number } => {
  let x = vx;
  let y = vy;

  for (const barrier of barriers) {
    if (barrier.hp <= 0) {
      continue;
    }

    const closestX = Math.max(barrier.x - barrier.halfW, Math.min(x, barrier.x + barrier.halfW));
    const closestZ = Math.max(barrier.z - barrier.halfD, Math.min(y, barrier.z + barrier.halfD));
    const dx = x - closestX;
    const dz = y - closestZ;
    const distSq = dx * dx + dz * dz;

    if (distSq >= VEHICLE_RADIUS * VEHICLE_RADIUS) {
      continue;
    }

    const dist = Math.sqrt(distSq) || 0.001;
    const overlap = VEHICLE_RADIUS - dist;
    x += (dx / dist) * overlap;
    y += (dz / dist) * overlap;
  }

  return { x, y };
};

export const resolveProjectileBarrierHits = (
  projectiles: ReadonlyArray<Projectile>,
  barriers: Barrier[],
  simTimeMs: number,
  breakEvents: BarrierBreakEvent[],
): Projectile[] => {
  const surviving: Projectile[] = [];

  for (const projectile of projectiles) {
    let absorbed = false;

    for (const barrier of barriers) {
      if (barrier.hp <= 0) {
        continue;
      }

      const closestX = Math.max(
        barrier.x - barrier.halfW,
        Math.min(projectile.x, barrier.x + barrier.halfW),
      );
      const closestZ = Math.max(
        barrier.z - barrier.halfD,
        Math.min(projectile.y, barrier.z + barrier.halfD),
      );
      const dx = projectile.x - closestX;
      const dz = projectile.y - closestZ;
      const distSq = dx * dx + dz * dz;

      if (distSq >= BULLET_HIT_RADIUS_SQ) {
        continue;
      }

      barrier.hp -= 1;
      absorbed = true;

      if (barrier.hp <= 0) {
        breakEvents.push({
          id: barrier.id,
          x: barrier.x,
          z: barrier.z,
          halfW: barrier.halfW,
          halfD: barrier.halfD,
          timeMs: simTimeMs,
        });
      }
      break;
    }

    if (!absorbed) {
      surviving.push(projectile);
    }
  }

  return surviving;
};
