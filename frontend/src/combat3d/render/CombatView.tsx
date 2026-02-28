import { Canvas } from "@react-three/fiber";
import type { RefObject } from "react";
import type { CombatState, DynamicObstacle } from "../engine/types";
import type { BarrierBreakEvent } from "../engine/barriers";
import { Combat3DScene } from "./Combat3DScene";

interface CombatViewProps {
  stateRef: RefObject<CombatState>;
  barrierBreaks: BarrierBreakEvent[];
  dynamicObstacles?: DynamicObstacle[];
}

export const CombatView = ({ stateRef, barrierBreaks, dynamicObstacles }: CombatViewProps) => {
  return (
    <div className="h-full w-full">
      <Canvas camera={{ fov: 55, position: [0, 28, 26] }}>
        <Combat3DScene stateRef={stateRef} barrierBreaks={barrierBreaks} dynamicObstacles={dynamicObstacles} />
      </Canvas>
    </div>
  );
};
