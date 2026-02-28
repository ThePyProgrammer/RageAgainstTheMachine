import { ControlInterface } from "../components/mi/ControlInterface";
import { TelemetryPanel } from "../components/mi/TelemetryPanel";
import { PredictionPanel } from "../components/mi/PredictionPanel";
import { useMotorImagery } from "../hooks/useMotorImagery";
import { Alert, AlertDescription } from "../components/ui/alert";
import { AlertCircle } from "lucide-react";

export default function MotorImageryPage() {
  const {
    stats,
    isConnected,
    isStreaming,
    startStreaming,
    stopStreaming,
    trail,
    trailNow,
    streamingError,
  } = useMotorImagery();

  return (
    <div className="min-h-screen w-full bg-white flex flex-col font-sans text-zinc-900 border border-zinc-200 shadow-sm">
      {/* Main Content */}
      <div className="flex-1 min-h-0 overflow-auto p-4 bg-zinc-50/50">
        <div className="mx-auto w-full max-w-6xl mb-4 flex items-center justify-between">
          <div className="text-xs uppercase tracking-widest text-zinc-500 font-semibold">
            Motor Imagery Stream
          </div>
          <div className="flex items-center gap-3">
            <span
              className={`text-[10px] font-mono uppercase tracking-wider ${isConnected ? "text-emerald-600" : "text-zinc-400"}`}
            >
              {isConnected ? "Connected" : "Disconnected"}
            </span>
            <button
              type="button"
              onClick={() =>
                isStreaming ? stopStreaming() : startStreaming(1000)
              }
              className={`h-8 px-3 text-xs font-semibold uppercase tracking-wider border rounded-sm transition cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 ${isStreaming ? "bg-zinc-900 text-white border-zinc-900 hover:bg-zinc-800" : "bg-white text-zinc-900 border-zinc-200 hover:bg-zinc-50"}`}
              disabled={!isConnected}
            >
              {isStreaming ? "Stop" : "Start"}
            </button>
          </div>
        </div>

        {/* Error Alert */}
        {streamingError && (
          <div className="mx-auto w-full max-w-6xl mb-4 animate-in fade-in slide-in-from-top-2 duration-300">
            <Alert
              variant="destructive"
              className="border-red-200 bg-red-50/80 backdrop-blur-sm"
            >
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-red-600 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <AlertDescription className="text-sm font-medium text-red-900 leading-relaxed">
                    {streamingError}
                  </AlertDescription>
                </div>
              </div>
            </Alert>
          </div>
        )}

        <div className="mx-auto w-full max-w-6xl h-full grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-4">
          {/* Control Interface */}
          <ControlInterface
            x={stats.x}
            y={stats.y}
            trail={trail}
            trailNow={trailNow}
          />

          {/* Side Panels */}
          <div className="space-y-4 flex flex-col h-full">
            {/* Telemetry (Actual) */}
            <TelemetryPanel
              x={stats.x}
              y={stats.y}
              speed={stats.speed}
              command={stats.command}
              status={stats.status}
            />

            {/* Prediction */}
            <PredictionPanel
              prediction={stats.prediction}
              confidence={stats.confidence}
              command={stats.command}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
