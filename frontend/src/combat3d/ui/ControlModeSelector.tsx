import type { Combat3DControlMode } from "../bci/useBCIDiscreteInput";

interface ControlModeSelectorProps {
  mode: Combat3DControlMode;
  onModeChange: (mode: Combat3DControlMode) => void;
}

export const ControlModeSelector = ({
  mode,
  onModeChange,
}: ControlModeSelectorProps) => (
  <div className="absolute top-3 left-3 z-20 flex items-center gap-2 rounded-lg border border-white/15 bg-black/70 backdrop-blur-sm px-4 py-2">
    <span className="text-[10px] uppercase tracking-wider text-white/40 mr-2">
      Controls
    </span>
    <button
      type="button"
      onClick={() => onModeChange("manual")}
      className={`rounded px-3 py-1 text-xs font-semibold transition ${
        mode === "manual"
          ? "bg-white text-black"
          : "text-white/50 hover:text-white"
      }`}
    >
      Manual
    </button>
    <button
      type="button"
      onClick={() => onModeChange("bci_hybrid")}
      className={`rounded px-3 py-1 text-xs font-semibold transition ${
        mode === "bci_hybrid"
          ? "bg-emerald-500 text-black"
          : "text-white/50 hover:text-white"
      }`}
    >
      BCI Hybrid
    </button>
  </div>
);
