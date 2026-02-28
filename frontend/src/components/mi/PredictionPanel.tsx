import { Brain, Radio, ArrowUpCircle } from "lucide-react";

interface PredictionPanelProps {
  prediction: string;
  confidence: number;
  command: string;
}

export function PredictionPanel({
  prediction,
  confidence,
  command,
}: PredictionPanelProps) {
  const direction = command.toLowerCase();
  const displayConfidence = Number.isFinite(confidence)
    ? confidence.toFixed(1)
    : "0.0";

  return (
    <div className="flex-1 bg-white border border-zinc-200 rounded-sm p-0 flex flex-col shadow-sm">
      <div className="h-10 border-b border-zinc-100 flex items-center px-4">
        <span className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-zinc-500">
          <Brain className="w-3 h-3" /> Prediction
        </span>
      </div>

      <div className="p-6 flex flex-col gap-6 items-center justify-center flex-1">
        {/* Visual Indicator */}
        <div className="relative w-32 h-32 rounded-full border border-zinc-100 flex items-center justify-center bg-zinc-50">
          <div className="absolute inset-0 border border-zinc-200 rounded-full scale-75 opacity-50" />
          {/* Dynamic Rotation based on command */}
          <ArrowUpCircle
            className={`w-12 h-12 text-zinc-800 transition-transform duration-300 ${
              direction === "strafe_left"
                ? "-rotate-90"
                : direction === "strafe_right"
                  ? "rotate-90"
                  : direction === "forward"
                    ? "rotate-0"
                    : direction === "hover"
                      ? "rotate-0"
                      : "rotate-0"
            } ${direction === "hover" ? "opacity-30" : "opacity-100"}`}
          />
        </div>

        <div className="text-center">
          <div className="text-2xl font-light text-zinc-900 tracking-tight">
            {prediction || "Idle"}
          </div>
          <div className="text-[10px] text-zinc-400 font-mono uppercase tracking-widest mt-1">
            Motor Imagery
          </div>
        </div>
      </div>

      <div className="border-t border-zinc-100 bg-zinc-50/50">
        <div className="p-4 flex flex-col items-center justify-center gap-1">
          <div className="flex items-center gap-1.5 text-zinc-400 mb-0.5">
            <Radio className="w-3 h-3" />
            <span className="text-[10px] uppercase font-bold tracking-wider">
              Confidence
            </span>
          </div>
          <div className="font-mono text-xl text-zinc-700">
            {displayConfidence}
            <span className="text-sm text-zinc-400 ml-0.5">%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
