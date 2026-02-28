import { useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { RefObject } from "react";
import { useRef } from "react";
import type { Mesh } from "three";
import type { CombatState } from "../engine/types";

interface CombatSceneProps {
  stateRef: RefObject<CombatState>;
}

const Bullet = ({ bullet }: { bullet: { x: number; y: number } }) => {
  return (
    <mesh position={[bullet.x, 0.1, bullet.y]}>
      <sphereGeometry args={[0.2, 12, 12]} />
      <meshStandardMaterial color="#ffcc00" />
    </mesh>
  );
};

export const Combat3DScene = ({ stateRef }: CombatSceneProps) => {
  const playerRef = useRef<Mesh>(null);
  const enemyRef = useRef<Mesh>(null);

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
      <ambientLight intensity={0.5} />
      <directionalLight position={[20, 20, 10]} intensity={0.7} />
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.25, 0]}>
        <planeGeometry args={[120, 120]} />
        <meshStandardMaterial color="#102" />
      </mesh>
      <mesh ref={playerRef}>
        <coneGeometry args={[0.9, 2, 16]} />
        <meshStandardMaterial color="#4d9dff" />
      </mesh>
      <mesh ref={enemyRef}>
        <coneGeometry args={[0.9, 2, 16]} />
        <meshStandardMaterial color="#ff4d4d" />
      </mesh>
      <OrbitControls enabled={false} />
      {stateRef.current?.projectiles.map((bullet, index) => (
        <Bullet key={`${bullet.owner}-${index}`} bullet={bullet} />
      ))}
    </group>
  );
};
