import { useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { RefObject } from "react";
import { useRef, useMemo, useState } from "react";
import type { Mesh, InstancedMesh as ThreeInstancedMesh } from "three";
import * as THREE from "three";
import type { CombatState } from "../engine/types";
import type { BarrierBreakEvent } from "../engine/barriers";
import { SeededRNG } from "../engine/seededRng";

interface CombatSceneProps {
  stateRef: RefObject<CombatState>;
  barrierBreaks: BarrierBreakEvent[];
}

/* ── Deterministic obstacle generation ─────────────────────────────── */

export interface Obstacle {
  x: number;
  z: number;
  width: number;
  height: number;
  depth: number;
  color: string;
}

const OBSTACLE_COLORS = ["#2a5a3a", "#3a3a5a", "#5a3a2a", "#4a4a2a", "#2a4a5a", "#5a2a4a"];
const WORLD_HALF = 50;

export const generateObstacles = (seed: number, count: number): Obstacle[] => {
  const rng = new SeededRNG(seed);
  const obstacles: Obstacle[] = [];
  for (let i = 0; i < count; i++) {
    const x = rng.nextSignedRange() * (WORLD_HALF - 8);
    const z = rng.nextSignedRange() * (WORLD_HALF - 8);
    // avoid spawning too close to the player/enemy start positions
    if (Math.abs(x) < 12 && Math.abs(z) < 12) continue;
    const width = 1.5 + rng.nextFloat() * 3;
    const height = 1.5 + rng.nextFloat() * 4;
    const depth = 1.5 + rng.nextFloat() * 3;
    const color = OBSTACLE_COLORS[rng.nextInt(OBSTACLE_COLORS.length)];
    obstacles.push({ x, z, width, height, depth, color });
  }
  return obstacles;
};

/* ── Sub-components ────────────────────────────────────────────────── */

const Bullet = ({ bullet }: { bullet: { x: number; y: number; owner: string } }) => {
  const color = bullet.owner === "player" ? "#00ccff" : "#ff4400";
  return (
    <mesh position={[bullet.x, 0.3, bullet.y]}>
      <sphereGeometry args={[0.25, 12, 12]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} />
    </mesh>
  );
};

const ObstacleMesh = ({ obs, broken }: { obs: Obstacle; broken: boolean }) => {
  if (broken) return null; // removed after fragmentation completes
  return (
    <mesh position={[obs.x, obs.height / 2, obs.z]}>
      <boxGeometry args={[obs.width, obs.height, obs.depth]} />
      <meshStandardMaterial color={obs.color} roughness={0.7} />
    </mesh>
  );
};

/* ── Fragment shard for breakable obstacles ─────────────────────────── */

interface Fragment {
  px: number;
  py: number;
  pz: number;
  vx: number;
  vy: number;
  vz: number;
  rx: number;
  ry: number;
  rz: number;
  rvx: number;
  rvy: number;
  rvz: number;
  scale: number;
  color: string;
}

const FRAGMENT_COUNT = 12;
const FRAGMENT_LIFE_MS = 2000;

const generateFragments = (evt: BarrierBreakEvent, obs: Obstacle | undefined): Fragment[] => {
  const rng = new SeededRNG(evt.id * 7919 + 31);
  const color = obs?.color ?? "#555";
  const frags: Fragment[] = [];
  for (let i = 0; i < FRAGMENT_COUNT; i++) {
    const angle = rng.nextFloat() * Math.PI * 2;
    const speed = 2 + rng.nextFloat() * 6;
    frags.push({
      px: evt.x + (rng.nextSignedRange() * (evt.halfW * 0.6)),
      py: 0.5 + rng.nextFloat() * 1.5,
      pz: evt.z + (rng.nextSignedRange() * (evt.halfD * 0.6)),
      vx: Math.cos(angle) * speed,
      vy: 3 + rng.nextFloat() * 5,
      vz: Math.sin(angle) * speed,
      rx: rng.nextFloat() * Math.PI * 2,
      ry: rng.nextFloat() * Math.PI * 2,
      rz: rng.nextFloat() * Math.PI * 2,
      rvx: (rng.nextFloat() - 0.5) * 10,
      rvy: (rng.nextFloat() - 0.5) * 10,
      rvz: (rng.nextFloat() - 0.5) * 10,
      scale: 0.15 + rng.nextFloat() * 0.35,
      color,
    });
  }
  return frags;
};

/** Animated shard cloud for a single broken barrier */
const FragmentCloud = ({
  evt,
  obs,
  onDone,
}: {
  evt: BarrierBreakEvent;
  obs: Obstacle | undefined;
  onDone: () => void;
}) => {
  const meshRef = useRef<ThreeInstancedMesh>(null);
  const fragsRef = useRef<Fragment[]>(generateFragments(evt, obs));
  const elapsedRef = useRef(0);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const matRef = useRef<THREE.MeshStandardMaterial>(null);

  useFrame((_, delta) => {
    elapsedRef.current += delta * 1000;
    const t = elapsedRef.current / FRAGMENT_LIFE_MS;
    if (t >= 1) {
      onDone();
      return;
    }

    const frags = fragsRef.current;
    const im = meshRef.current;
    if (!im) return;

    const gravity = -9.8;
    for (let i = 0; i < frags.length; i++) {
      const f = frags[i];
      f.px += f.vx * delta;
      f.py += f.vy * delta;
      f.pz += f.vz * delta;
      f.vy += gravity * delta;
      f.rx += f.rvx * delta;
      f.ry += f.rvy * delta;
      f.rz += f.rvz * delta;

      const fadeScale = f.scale * (1 - t);
      dummy.position.set(f.px, Math.max(0, f.py), f.pz);
      dummy.rotation.set(f.rx, f.ry, f.rz);
      dummy.scale.setScalar(fadeScale);
      dummy.updateMatrix();
      im.setMatrixAt(i, dummy.matrix);
    }
    im.instanceMatrix.needsUpdate = true;

    if (matRef.current) {
      matRef.current.opacity = 1 - t;
    }
  });

  const color = obs?.color ?? "#555";

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, FRAGMENT_COUNT]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        ref={matRef}
        color={color}
        emissive={color}
        emissiveIntensity={0.5}
        transparent
        opacity={1}
        roughness={0.6}
      />
    </instancedMesh>
  );
};

const BoundaryWalls = () => {
  const wallHeight = 3;
  const wallThickness = 0.5;
  const extent = WORLD_HALF;
  const wallColor = "#1a3344";

  return (
    <group>
      {/* North */}
      <mesh position={[0, wallHeight / 2, -extent]}>
        <boxGeometry args={[extent * 2, wallHeight, wallThickness]} />
        <meshStandardMaterial color={wallColor} transparent opacity={0.6} />
      </mesh>
      {/* South */}
      <mesh position={[0, wallHeight / 2, extent]}>
        <boxGeometry args={[extent * 2, wallHeight, wallThickness]} />
        <meshStandardMaterial color={wallColor} transparent opacity={0.6} />
      </mesh>
      {/* West */}
      <mesh position={[-extent, wallHeight / 2, 0]}>
        <boxGeometry args={[wallThickness, wallHeight, extent * 2]} />
        <meshStandardMaterial color={wallColor} transparent opacity={0.6} />
      </mesh>
      {/* East */}
      <mesh position={[extent, wallHeight / 2, 0]}>
        <boxGeometry args={[wallThickness, wallHeight, extent * 2]} />
        <meshStandardMaterial color={wallColor} transparent opacity={0.6} />
      </mesh>
    </group>
  );
};

const GroundGrid = () => {
  const gridHelper = useMemo(() => {
    const grid = new THREE.GridHelper(100, 40, "#224444", "#112222");
    grid.position.y = -0.24;
    return grid;
  }, []);
  return <primitive object={gridHelper} />;
};

/* ── Main scene ────────────────────────────────────────────────────── */

export const Combat3DScene = ({ stateRef, barrierBreaks }: CombatSceneProps) => {
  const playerRef = useRef<Mesh>(null);
  const enemyRef = useRef<Mesh>(null);

  // Generate obstacles deterministically from seed 0x2f2f
  const obstacles = useMemo(() => generateObstacles(0x2f2f, 20), []);

  // Track which barrier IDs have been broken
  const brokenIds = useMemo(() => {
    const set = new Set<number>();
    for (const evt of barrierBreaks) set.add(evt.id);
    return set;
  }, [barrierBreaks]);

  // Track which fragment clouds are still animating
  const [activeFragments, setActiveFragments] = useState<BarrierBreakEvent[]>([]);

  // When new break events arrive, start fragment animations
  const processedRef = useRef(new Set<number>());
  useMemo(() => {
    const newEvents: BarrierBreakEvent[] = [];
    for (const evt of barrierBreaks) {
      if (!processedRef.current.has(evt.id)) {
        processedRef.current.add(evt.id);
        newEvents.push(evt);
      }
    }
    if (newEvents.length > 0) {
      setActiveFragments((prev) => [...prev, ...newEvents]);
    }
  }, [barrierBreaks]);

  const removeFragment = (id: number) => {
    setActiveFragments((prev) => prev.filter((f) => f.id !== id));
  };

  useFrame(() => {
    if (!stateRef.current) {
      return;
    }

    const state = stateRef.current;
    if (playerRef.current) {
      playerRef.current.position.set(state.player.x, 0.35, state.player.y);
      playerRef.current.rotation.y = state.player.yaw + Math.PI / 2;
    }

    if (enemyRef.current) {
      enemyRef.current.position.set(state.enemy.x, 0.35, state.enemy.y);
      enemyRef.current.rotation.y = state.enemy.yaw + Math.PI / 2;
    }
  });

  return (
    <group>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[30, 40, 20]} intensity={0.8} castShadow />
      <directionalLight position={[-20, 20, -10]} intensity={0.3} color="#4488ff" />
      <hemisphereLight args={["#334466", "#112211", 0.4]} />

      {/* Ground plane */}
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.25, 0]}>
        <planeGeometry args={[120, 120]} />
        <meshStandardMaterial color="#1a2a1a" roughness={0.9} />
      </mesh>

      {/* Grid overlay */}
      <GroundGrid />

      {/* Boundary walls */}
      <BoundaryWalls />

      {/* Obstacles (hidden when broken) */}
      {obstacles.map((obs, i) => (
        <ObstacleMesh key={`obs-${i}`} obs={obs} broken={brokenIds.has(i)} />
      ))}

      {/* Fragment clouds for broken barriers */}
      {activeFragments.map((evt) => (
        <FragmentCloud
          key={`frag-${evt.id}`}
          evt={evt}
          obs={obstacles[evt.id]}
          onDone={() => removeFragment(evt.id)}
        />
      ))}

      {/* Player ship (blue cone) */}
      <mesh ref={playerRef}>
        <coneGeometry args={[0.9, 2, 16]} />
        <meshStandardMaterial color="#4d9dff" emissive="#1144aa" emissiveIntensity={0.3} />
      </mesh>

      {/* Enemy ship (red cone) */}
      <mesh ref={enemyRef}>
        <coneGeometry args={[0.9, 2, 16]} />
        <meshStandardMaterial color="#ff4d4d" emissive="#aa1111" emissiveIntensity={0.3} />
      </mesh>

      <OrbitControls enabled={false} />

      {/* Projectiles */}
      {stateRef.current?.projectiles.map((bullet, index) => (
        <Bullet key={`${bullet.owner}-${index}`} bullet={bullet} />
      ))}
    </group>
  );
};
