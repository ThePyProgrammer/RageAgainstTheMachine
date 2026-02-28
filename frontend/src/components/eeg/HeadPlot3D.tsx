import { Suspense, useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { Button } from "@/components/ui/button";
import { Settings, Eye, EyeOff } from "lucide-react";
import BrainModel from "@/components/eeg/headplot/BrainModel";
import HeadModel from "@/components/eeg/headplot/HeadModel";
import ElectrodeMarkers from "@/components/eeg/headplot/ElectrodeMarkers";
import ColorLegend from "@/components/eeg/headplot/ColorLegend";
import { useElectrodeMappingContext } from "@/contexts/ElectrodeMappingContext";
import { useBCIStream } from "@/hooks/useBCIStream";

export default function HeadPlot() {
  const [isConfigMode, setIsConfigMode] = useState(false);
  const [showHead, setShowHead] = useState(true);
  const {
    activeChannel,
    setActiveChannel,
    mapping,
    handleElectrodeClick,
    handleRemoveMapping,
  } = useElectrodeMappingContext();
  const { displayData, status } = useBCIStream();

  // Scene offsets
  const downOffset = 0.14; // ~20px in current camera setup
  const brainYOffset = 0.7;
  const brainZOffset = -0.1;

  const controlsRef = useRef<OrbitControlsImpl | null>(null);

  // Extract latest electrode data from displayData
  const getLatestElectrodeData = () => {
    if (displayData.length === 0) return {};
    const latestPoint = displayData[displayData.length - 1];
    return latestPoint;
  };

  useEffect(() => {
    const targetY = (showHead ? 0 : brainYOffset) - downOffset;
    const targetZ = showHead ? 0 : brainZOffset;
    if (controlsRef.current) {
      controlsRef.current.target.set(0, targetY, targetZ);
      controlsRef.current.update?.();
    }
  }, [showHead, downOffset, brainYOffset, brainZOffset]);

  return (
    <div className="h-full w-full bg-white text-zinc-900 flex flex-col">
      {/* Header Controls */}
      <header className="h-16 border-b border-zinc-200 px-4 flex items-center justify-between bg-white shrink-0 z-30">
        <h1 className="text-lg font-bold tracking-tight">Head Plot</h1>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="lg"
            onClick={() => setShowHead((v) => !v)}
            className="w-32 bg-white text-zinc-900 border border-zinc-300 hover:bg-zinc-100 cursor-pointer"
            aria-label={showHead ? "Hide head model" : "Show head model"}
          >
            {showHead ? <EyeOff size={16} /> : <Eye size={16} />}
            {showHead ? "Hide Head" : "Show Head"}
          </Button>

          <Button
            variant="outline"
            size="lg"
            onClick={() => setIsConfigMode(!isConfigMode)}
            className="bg-white text-zinc-900 border border-zinc-300 hover:bg-zinc-100 cursor-pointer"
          >
            <Settings size={18} />
            {isConfigMode ? "Done" : "Configure"}
          </Button>
        </div>
      </header>

      {/* Canvas Area */}
      <div className="flex-1 overflow-hidden flex gap-4">
        {isConfigMode && (
          <div className="w-36 bg-white border-r border-zinc-200 p-3 space-y-2 overflow-y-auto shrink-0">
            <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
              Select Channel
            </h3>
            {["1", "2", "3", "4", "5", "6", "7", "8"].map((ch) => (
              <Button
                key={ch}
                variant={activeChannel === ch ? "default" : "outline"}
                size="sm"
                onClick={() => setActiveChannel(ch)}
                className={`w-full justify-start font-semibold transition-all cursor-pointer text-xs ${
                  activeChannel === ch
                    ? "bg-zinc-900 hover:bg-zinc-800 text-white"
                    : "bg-white hover:bg-zinc-100 text-zinc-900 border-zinc-300"
                }`}
              >
                Ch {ch}
                {Object.values(mapping).includes(ch) && (
                  <span className="ml-auto text-xs">✓</span>
                )}
              </Button>
            ))}
          </div>
        )}

        <div
          className="flex-1 border-zinc-200 relative"
          style={{
            background: "radial-gradient(circle, #d4d4d8 0%, #ffffff 100%)",
          }}
        >
          <ColorLegend />
          <Canvas
            camera={{ position: [0, 0, 5], fov: 50 }}
            shadows
            gl={{ antialias: true }}
          >
            <ambientLight intensity={0.3} />
            <directionalLight
              position={[3, 4, 5]}
              intensity={2.5}
              castShadow
              shadow-mapSize={[2048, 2048]}
            />
            <directionalLight position={[-3, 2, -2]} intensity={1} />
            <directionalLight position={[0, 0, 5]} intensity={0.6} />
            <spotLight
              position={[0, 5, 0]}
              angle={0.5}
              intensity={1.2}
              castShadow
            />
            <hemisphereLight args={["#ffffff", "#555555", 0.6]} />
            <Environment preset="city" />

            <Suspense fallback={null}>
              <group position={[0, -downOffset, 0]}>
                <BrainModel
                  electrodeData={getLatestElectrodeData()}
                  mapping={mapping}
                  status={status}
                />
                {showHead && <HeadModel />}
                <ElectrodeMarkers
                  isConfigMode={isConfigMode}
                  mapping={mapping}
                  onElectrodeClick={(id) =>
                    handleElectrodeClick(id, isConfigMode)
                  }
                  onRemoveMapping={handleRemoveMapping}
                />
              </group>
            </Suspense>

            <OrbitControls ref={controlsRef} enablePan={true} />
          </Canvas>
        </div>
      </div>
    </div>
  );
}

export { HeadPlot };
