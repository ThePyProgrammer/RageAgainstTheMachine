import { Card, CardContent } from "@/components/ui/card";
import { Activity } from "lucide-react";
import { PulseWaveformChart } from "./PulseWaveformChart";

interface PulseBPMDisplayProps {
  bpm: number;
  spo2: number;
  rr: number;
  bvp: number;
}

export function PulseBPMDisplay({ bpm, spo2, rr, bvp }: PulseBPMDisplayProps) {
  return (
    <div className="absolute bottom-6 left-6 flex flex-col gap-4">
      {/* Charts Row */}
      <div className="flex">
        <Card className="bg-white/95 backdrop-blur-sm shadow-xl border-0 w-[400px] h-[100px]">
          <CardContent className="p-3 h-full flex flex-col justify-center">
            <div className="flex items-center gap-2 mb-1">
              <Activity size={16} className="text-red-500" />
              <span className="text-slate-700 text-xs font-semibold">
                Blood Volume Pulse
              </span>
            </div>
            <div className="flex-1 w-full relative">
              <PulseWaveformChart
                dataPoint={bvp}
                height={60}
                color="rgb(239, 68, 68)"
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Metrics Row */}
      <div className="flex gap-4">
        <Card className="bg-white/95 backdrop-blur-sm shadow-xl border-0">
          <CardContent className="p-5">
            <div className="flex items-center gap-2 mb-2">
              <Activity size={20} className="text-red-500" />
              <span className="text-slate-700 text-sm font-semibold">
                Heart Rate
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-6xl font-bold text-red-500 tracking-tight">
                {bpm > 0 ? bpm : "--"}
              </span>
              <span className="text-xl text-slate-500 font-semibold mb-1">
                BPM
              </span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white/95 backdrop-blur-sm shadow-xl border-0">
          <CardContent className="p-5">
            <div className="flex items-center gap-2 mb-2">
              <Activity size={20} className="text-blue-500" />
              <span className="text-slate-700 text-sm font-semibold">SpO2</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-6xl font-bold text-blue-500 tracking-tight">
                {spo2 > 0 ? spo2 : "--"}
              </span>
              <span className="text-xl text-slate-500 font-semibold mb-1">
                %
              </span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white/95 backdrop-blur-sm shadow-xl border-0">
          <CardContent className="p-5">
            <div className="flex items-center gap-2 mb-2">
              <Activity size={20} className="text-green-500" />
              <span className="text-slate-700 text-sm font-semibold">
                Resp. Rate
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-6xl font-bold text-green-500 tracking-tight">
                {rr > 0 ? rr : "--"}
              </span>
              <span className="text-xl text-slate-500 font-semibold mb-1">
                rpm
              </span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
