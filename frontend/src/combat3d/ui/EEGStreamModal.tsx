import { useMemo } from "react";

type EegHemisphere = "left" | "right";
type EegMode = "classifier" | "features";

interface EEGStreamModalProps {
  waveSamples: number[];
  leftPower: number;
  rightPower: number;
  confidence: number;
  packetRateHz: number;
  activeHemisphere: EegHemisphere;
  mode: EegMode;
}

interface PowerBarProps {
  label: string;
  value: number;
  active: boolean;
}

const WIDTH = 238;
const HEIGHT = 56;
const PADDING = 4;

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

const PowerBar = ({ label, value, active }: PowerBarProps) => {
  const pct = Math.round(clamp01(value) * 100);

  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2 text-[10px] font-semibold ${active ? "text-cyan-200" : "text-slate-400"}`}>{label}</span>
      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-slate-700/70">
        <div
          className={`h-full rounded-full transition-all duration-150 ${active ? "bg-cyan-300 shadow-[0_0_10px_rgba(103,232,249,0.8)]" : "bg-slate-400/65"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-7 text-right font-mono text-[10px] text-cyan-100/80">{pct}</span>
    </div>
  );
};

export const EEGStreamModal = ({
  waveSamples,
  leftPower,
  rightPower,
  confidence,
  packetRateHz,
  activeHemisphere,
  mode,
}: EEGStreamModalProps) => {
  const centerY = HEIGHT / 2;
  const amplitude = HEIGHT * 0.3;
  const innerWidth = WIDTH - PADDING * 2;
  const sampleCount = Math.max(waveSamples.length, 2);

  const wavePath = useMemo(() => {
    if (waveSamples.length === 0) {
      return `M ${PADDING} ${centerY} L ${WIDTH - PADDING} ${centerY}`;
    }

    return waveSamples
      .map((sample, index) => {
        const x = PADDING + (index / (sampleCount - 1)) * innerWidth;
        const y = centerY - sample * amplitude;
        return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  }, [waveSamples, centerY, sampleCount, innerWidth, amplitude]);

  const areaPath = `${wavePath} L ${WIDTH - PADDING} ${centerY} L ${PADDING} ${centerY} Z`;
  const confidencePct = Math.round(clamp01(confidence) * 100);
  const leftActive = activeHemisphere === "left";
  const rightActive = activeHemisphere === "right";

  return (
    <div className="pointer-events-none absolute bottom-4 left-4 z-30 select-none">
      <div className="w-[280px] rounded-xl border border-cyan-300/25 bg-slate-950/80 p-2.5 shadow-[0_0_30px_rgba(6,182,212,0.18)] backdrop-blur-md">
        <div className="mb-1.5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-300 shadow-[0_0_10px_rgba(34,211,238,0.95)] animate-pulse" />
            <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-100/90">EEG Live</span>
          </div>
          <span className="font-mono text-[10px] text-cyan-100/70">{packetRateHz.toFixed(0)} Hz</span>
        </div>

        <div className="relative mb-2 h-14 overflow-hidden rounded-md border border-cyan-400/30 bg-gradient-to-r from-cyan-950/80 via-slate-950/95 to-cyan-950/55">
          <svg width={WIDTH} height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="absolute inset-0 h-full w-full">
            <defs>
              <linearGradient id="eeg-wave-line" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#67e8f9" stopOpacity="0.2" />
                <stop offset="50%" stopColor="#67e8f9" stopOpacity="1" />
                <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.45" />
              </linearGradient>
              <linearGradient id="eeg-wave-fill" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
              </linearGradient>
            </defs>

            <line x1={PADDING} y1={centerY} x2={WIDTH - PADDING} y2={centerY} stroke="#134e4a" strokeWidth={1} />
            <path d={areaPath} fill="url(#eeg-wave-fill)" />
            <path d={wavePath} fill="none" stroke="url(#eeg-wave-line)" strokeWidth={2} strokeLinecap="round" />
          </svg>

          <div className="absolute right-2 top-1.5 rounded bg-black/45 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-cyan-100/90">
            {leftActive ? "Left Active" : "Right Active"}
          </div>
        </div>

        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <svg viewBox="0 0 120 80" className="h-10 w-16">
              <ellipse
                cx={42}
                cy={34}
                rx={29}
                ry={23}
                fill={leftActive ? "#67e8f9" : "#334155"}
                fillOpacity={leftActive ? 0.95 : 0.85}
                stroke={leftActive ? "#a5f3fc" : "#64748b"}
                strokeWidth={2}
              />
              <ellipse
                cx={78}
                cy={34}
                rx={29}
                ry={23}
                fill={rightActive ? "#67e8f9" : "#334155"}
                fillOpacity={rightActive ? 0.95 : 0.85}
                stroke={rightActive ? "#a5f3fc" : "#64748b"}
                strokeWidth={2}
              />
              <path d="M60 15 L60 56" stroke="#0f172a" strokeWidth={2.5} />
              <path d="M60 57 L60 71" stroke="#0f172a" strokeWidth={3} strokeLinecap="round" />
            </svg>
            <div className="space-y-1">
              <PowerBar label="L" value={leftPower} active={leftActive} />
              <PowerBar label="R" value={rightPower} active={rightActive} />
            </div>
          </div>

          <div className="text-right">
            <div className="font-mono text-[11px] text-cyan-100/90">{confidencePct}%</div>
            <div className="text-[9px] uppercase tracking-wider text-cyan-100/60">Confidence</div>
            <div className="mt-0.5 text-[9px] uppercase tracking-wider text-cyan-100/45">{mode}</div>
          </div>
        </div>
      </div>
    </div>
  );
};
