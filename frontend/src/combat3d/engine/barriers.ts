/**
 * Barrier / obstacle collision and destruction system.
 *
 * Obstacles start intact. Vehicles are blocked by barriers but do NOT damage
 * them. Only projectile hits reduce HP (starts at 3). When HP reaches 0 a
 * BarrierBreakEvent is emitted and the obstacle shatters (animated in the
 * renderer). After a 2-second dissolve the obstacle is removed from collision.
 */

export interface Barrier {
  id: number;
  x: number;
  z: number; // world uses (x, z) on the ground plane; in CombatState z maps to y
  halfW: number;
  halfD: number;
  hp: number; // starts at 3
}

export interface BarrierBreakEvent {
  id: number;
  x: number;
  z: number;
  halfW: number;
  halfD: number;
  timeMs: number; // sim time of the break
}

/** Build the barrier list from the obstacle generation output. */
export const barriersFromObstacles = (
  obstacles: ReadonlyArray<{
    x: number;
    z: number;
    width: number;
    depth: number;
  }>,
): Barrier[] =>
  obstacles.map((obs, i) => ({
    id: i,
    x: obs.x,
    z: obs.z,
    halfW: obs.width / 2,
    halfD: obs.depth / 2,
    hp: 3,
  }));

const VEHICLE_RADIUS = 1.0;

/**
 * Test & resolve collision between a point-circle vehicle and an axis-aligned
 * rectangular barrier. Returns the pushed-out position if colliding, or the
 * original position if not.
 *
 * Vehicle contact does NOT damage the barrier — only projectile hits do.
 */
export const resolveBarrierCollision = (
  vx: number,
  vy: number,
  barriers: Barrier[],
): { x: number; y: number } => {
  let x = vx;
  let y = vy;

  for (const b of barriers) {
    if (b.hp <= 0) continue; // already broken

    // Closest point on AABB to vehicle centre
    const closestX = Math.max(b.x - b.halfW, Math.min(x, b.x + b.halfW));
    const closestZ = Math.max(b.z - b.halfD, Math.min(y, b.z + b.halfD));

    const dx = x - closestX;
    const dz = y - closestZ;
    const distSq = dx * dx + dz * dz;

    if (distSq < VEHICLE_RADIUS * VEHICLE_RADIUS) {
      // Collision! Push out along shortest axis
      const dist = Math.sqrt(distSq) || 0.001;
      const overlap = VEHICLE_RADIUS - dist;
      x += (dx / dist) * overlap;
      y += (dz / dist) * overlap;
    }
  }

  return { x, y };
};

const BULLET_HIT_RADIUS_SQ = 1.5 * 1.5; // projectile-vs-barrier hit threshold²

/**
 * Test projectile positions against barriers. Any projectile that overlaps a
 * live barrier damages it (HP -= 1) and is consumed. When a barrier's HP
 * reaches 0 a BarrierBreakEvent is emitted.
 *
 * Returns the subset of projectiles that survived (did not hit any barrier).
 */
export const resolveProjectileBarrierHits = (
  projectiles: ReadonlyArray<{ x: number; y: number; vx: number; vy: number; ttlMs: number; owner: string }>,
  barriers: Barrier[],
  simTimeMs: number,
  breakEvents: BarrierBreakEvent[],
): typeof projectiles extends ReadonlyArray<infer T> ? T[] : never => {
  const surviving: any[] = [];

  for (const p of projectiles) {
    let absorbed = false;

    for (const b of barriers) {
      if (b.hp <= 0) continue;

      // Closest point on AABB to projectile centre
      const closestX = Math.max(b.x - b.halfW, Math.min(p.x, b.x + b.halfW));
      const closestZ = Math.max(b.z - b.halfD, Math.min(p.y, b.z + b.halfD));

      const dx = p.x - closestX;
      const dz = p.y - closestZ;
      const distSq = dx * dx + dz * dz;

      if (distSq < BULLET_HIT_RADIUS_SQ) {
        b.hp -= 1;
        absorbed = true;

        if (b.hp <= 0) {
          breakEvents.push({
            id: b.id,
            x: b.x,
            z: b.z,
            halfW: b.halfW,
            halfD: b.halfD,
            timeMs: simTimeMs,
          });
        }
        break; // one projectile damages at most one barrier
      }
    }

    if (!absorbed) {
      surviving.push(p);
    }
  }

  return surviving;
};
