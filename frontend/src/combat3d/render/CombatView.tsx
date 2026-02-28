import { Canvas } from "@react-three/fiber";
import type { RefObject } from "react";
import type { CombatState } from "../engine/types";
import { Combat3DScene } from "./Combat3DScene";

interface CombatViewProps {
  stateRef: RefObject<CombatState>;
}

export const CombatView = ({ stateRef }: CombatViewProps) => {
  return (
    <div className="h-full w-full">
      <Canvas camera={{ fov: 55, position: [0, 28, 26] }}>
        <Combat3DScene stateRef={stateRef} />
      </Canvas>
    </div>
  );
};
