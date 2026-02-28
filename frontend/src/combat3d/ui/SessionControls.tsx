import { useState } from "react";

interface SessionControlsProps {
  onStart: (mode: "classifier" | "features") => void;
  disabled: boolean;
}

export const SessionControls = ({ onStart, disabled }: SessionControlsProps) => {
  const [mode, setMode] = useState<"features" | "classifier">("features");

  return (
    <div className="absolute left-3 bottom-3 z-10 rounded border border-white/40 bg-black/60 p-3">
      <div className="mb-2">BCI mode</div>
      <div className="flex gap-2">
        <button
          className="rounded border px-2 py-1"
          type="button"
          onClick={() => setMode("features")}
        >
          Features
        </button>
        <button
          className="rounded border px-2 py-1"
          type="button"
          onClick={() => setMode("classifier")}
        >
          Classifier
        </button>
      </div>
      <button
        className="mt-2 rounded bg-emerald-600 px-3 py-1"
        type="button"
        onClick={() => onStart(mode)}
        disabled={disabled}
      >
        Start session
      </button>
    </div>
  );
};
