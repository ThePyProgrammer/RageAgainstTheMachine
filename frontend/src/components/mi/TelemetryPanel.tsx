import { Activity, Navigation, Terminal } from "lucide-react";

interface TelemetryPanelProps {
  x: number;
  y: number;
  speed: number;
  command: string;
  status: string;
}

export function TelemetryPanel({
  x,
  y,
  speed,
  command,
  status,
}: TelemetryPanelProps) {
  return (
    <div className="bg-white border border-zinc-200 rounded-sm shadow-sm">
      <div className="h-10 border-b border-zinc-100 flex items-center px-4">
        <span className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-zinc-500">
          <Activity className="w-3 h-3" /> Telemetry
        </span>
      </div>
      <div className="p-4 grid grid-cols-2 gap-4">
        {/* Status */}
        <div className="col-span-2 bg-zinc-50 rounded border border-zinc-100 p-3 flex items-center justify-between">
          <span className="text-xs font-mono font-bold text-zinc-400 uppercase">
            Status
          </span>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                status === "MOVING"
                  ? "bg-emerald-500"
                  : status === "STAGNANT" || status === "HOVERING"
                    ? "bg-amber-500"
                    : "bg-zinc-300"
              }`}
            />
            <span className="font-mono text-zinc-900 text-sm uppercase">
              {status}
            </span>
          </div>
        </div>

        {/* Position */}
        <div className="col-span-2 bg-zinc-50 rounded border border-zinc-100 p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono font-bold text-zinc-400 uppercase">
              Position
            </span>
            <Navigation className="w-3 h-3 text-zinc-300" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col">
              <span className="text-[10px] text-zinc-400 font-mono">
                X-AXIS
              </span>
              <span className="font-mono text-zinc-900">{x.toFixed(2)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-zinc-400 font-mono">
                Y-AXIS
              </span>
              <span className="font-mono text-zinc-900">{y.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Speed */}
        <div className="bg-zinc-50 rounded border border-zinc-100 p-3 flex flex-col justify-between">
          <span className="text-xs font-mono font-bold text-zinc-400 uppercase">
            Speed
          </span>
          <div className="flex items-end gap-1">
            <span className="font-mono text-zinc-900 text-lg">
              {speed.toFixed(1)}
            </span>
            <span className="text-[10px] text-zinc-400 font-mono mb-1">
              m/s
            </span>
          </div>
        </div>

        {/* Command */}
        <div className="bg-zinc-50 rounded border border-zinc-100 p-3 flex flex-col justify-between">
          <span className="text-xs font-mono font-bold text-zinc-400 uppercase">
            Command
          </span>
          <div className="flex items-center gap-2">
            <Terminal className="w-3 h-3 text-zinc-400" />
            <span className="font-mono text-zinc-900 text-sm">{command}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
