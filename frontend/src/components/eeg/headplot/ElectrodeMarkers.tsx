import { Html } from "@react-three/drei";
import * as THREE from "three";
import { CHANNEL_COLORS, STANDARD_ELECTRODES } from "@/config/eeg";
import type { ElectrodeMapping } from "@/types/eeg";

type Props = {
  isConfigMode: boolean;
  mapping: ElectrodeMapping;
  onElectrodeClick: (electrodeId: string) => void;
  onRemoveMapping: (electrodeId: string) => void;
};

const ElectrodeMarkers = ({
  isConfigMode,
  mapping,
  onElectrodeClick,
  onRemoveMapping,
}: Props) => {
  const HEAD_SCALE = { x: 0.85, y: 1.1, z: 1.05 };
  const RADIUS = 1.05;

  return (
    <group>
      {STANDARD_ELECTRODES.map((electrode) => {
        const x = electrode.x * RADIUS * HEAD_SCALE.x;
        const y = electrode.y * RADIUS * HEAD_SCALE.y;
        const z = electrode.z * RADIUS * HEAD_SCALE.z;

        const positionVector = new THREE.Vector3(x, y, z);
        const normal = positionVector.clone().normalize();
        const assignedChannel = mapping[electrode.id];

        if (!isConfigMode && !assignedChannel) return null;

        return (
          <group key={electrode.id} position={[x, y, z]}>
            <mesh
              onClick={() => onElectrodeClick(electrode.id)}
              onPointerEnter={() => (document.body.style.cursor = "pointer")}
              onPointerLeave={() => (document.body.style.cursor = "auto")}
            >
              <sphereGeometry args={[0.03, 16, 16]} />
              <meshStandardMaterial
                color={
                  assignedChannel
                    ? CHANNEL_COLORS[parseInt(assignedChannel) - 1]
                    : "#fbbf24"
                }
                emissive="#f59e0b"
                emissiveIntensity={assignedChannel ? 0.8 : 0.5}
                metalness={0.5}
                roughness={0.2}
              />
            </mesh>

            <Html
              position={[normal.x * 0.16, normal.y * 0.16, normal.z * 0.16]}
              center
              distanceFactor={6}
              style={{ pointerEvents: "auto", userSelect: "none" }}
            >
              <div
                onClick={() => onElectrodeClick(electrode.id)}
                onMouseEnter={() => (document.body.style.cursor = "pointer")}
                onMouseLeave={() => (document.body.style.cursor = "auto")}
                className={`relative flex flex-col items-center justify-center text-center text-[9px] font-bold px-2.5 py-1.5 rounded-md border shadow-sm whitespace-nowrap cursor-pointer transition-opacity opacity-70 hover:opacity-30 ${
                  assignedChannel
                    ? "border-green-500 text-white"
                    : "border-zinc-300 text-zinc-900"
                }`}
                style={{
                  backgroundColor: assignedChannel
                    ? CHANNEL_COLORS[parseInt(assignedChannel) - 1]
                    : "#f8fafc",
                }}
              >
                {isConfigMode && assignedChannel && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveMapping(electrode.id);
                    }}
                    className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-red-500 text-white text-xs font-bold flex items-center justify-center shadow"
                    title="Remove mapping"
                  >
                    ×
                  </button>
                )}
                <div className="leading-tight">
                  <div>{electrode.id}</div>
                  {assignedChannel && (
                    <div className="text-[8px] opacity-90">
                      {assignedChannel}
                    </div>
                  )}
                </div>
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
};

export default ElectrodeMarkers;
