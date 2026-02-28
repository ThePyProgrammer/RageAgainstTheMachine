import { Target } from "lucide-react";

interface ControlInterfaceProps {
  x: number;
  y: number;
  trail: { x: number; y: number; timestamp: number }[];
  trailNow: number;
}

export function ControlInterface({
  x,
  y,
  trail,
  trailNow,
}: ControlInterfaceProps) {
  return (
    <div className="flex flex-col h-full bg-white border border-zinc-200 rounded-sm overflow-hidden relative shadow-sm">
      <div className="h-10 border-b border-zinc-100 flex items-center justify-between px-4 bg-white">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-zinc-500">
          <Target className="w-3 h-3" /> Control Interface
        </div>
      </div>

      <div className="flex-1 relative bg-zinc-50/30 overflow-hidden">
        {/* Technical Grid */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              "linear-gradient(to right, #e4e4e7 1px, transparent 1px), linear-gradient(to bottom, #e4e4e7 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        {/* Trail path - connected lines */}
        <svg
          className="absolute inset-0 pointer-events-none"
          style={{ zIndex: 4 }}
        >
          {trail.length > 1 &&
            trail.map((point, index) => {
              if (index === 0) return null;
              const prevPoint = trail[index - 1];
              const age = trailNow - point.timestamp;
              const opacity = Math.max(0, 0.3 * (1 - age / 6000));

              // Convert normalized coords to screen coords (relative to center)
              const x1 = `calc(50% + ${prevPoint.x * 200}px)`;
              const y1 = `calc(50% + ${-prevPoint.y * 200}px)`;
              const x2 = `calc(50% + ${point.x * 200}px)`;
              const y2 = `calc(50% + ${-point.y * 200}px)`;

              return (
                <line
                  key={`${point.timestamp}-${index}-line`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="#18181b"
                  strokeWidth="2"
                  strokeOpacity={opacity}
                  strokeLinecap="round"
                />
              );
            })}
        </svg>

        {/* Trail dots */}
        {trail.map((point, index) => {
          const age = trailNow - point.timestamp;
          const opacity = Math.max(0, 0.6 * (1 - age / 6000));
          const size = 4 + Math.max(0, 1 - age / 6000) * 4;
          return (
            <div
              key={`${point.timestamp}-${index}`}
              className="mi-trail-dot"
              style={{
                width: `${size}px`,
                height: `${size}px`,
                opacity,
                transform: `translate(-50%, -50%) translate(${point.x * 200}px, ${-point.y * 200}px)`,
              }}
            />
          );
        })}

        {/* Center Dot */}
        <div
          className="mi-drone-dot shadow-lg border border-blue-400/30"
          style={{
            transform: `translate(-50%, -50%) translate(${x * 200}px, ${-y * 200}px)`,
          }}
        >
          <div className="absolute inset-0 bg-linear-to-br from-blue-500 to-blue-700 rounded-full" />
          <div className="absolute -inset-1.5 border border-blue-300/40 rounded-full animate-pulse opacity-40" />
          <div
            className="absolute -inset-3 border border-blue-500/20 rounded-full animate-pulse opacity-20"
            style={{ animationDelay: "0.1s" }}
          />
        </div>

        {/* Intent Line */}
        <div className="mi-intent-line" />
      </div>
      <style>{`
                .mi-drone-dot {
                    position: absolute;
                    width: 14px;
                    height: 14px;
                    left: 50%;
                    top: 50%;
                    transition: transform 0.8s cubic-bezier(0.25, 0.1, 0.25, 1);
                    will-change: transform;
                    z-index: 10;
                }

                .mi-trail-dot {
                    position: absolute;
                    left: 50%;
                    top: 50%;
                    background: #18181b;
                    border-radius: 999px;
                    transform: translate(-50%, -50%);
                    z-index: 5;
                    filter: blur(0.2px);
                }

                .mi-intent-line {
                    position: absolute;
                    left: 20%;
                    top: 50%;
                    width: 60%;
                    height: 1px;
                    background: linear-gradient(90deg, transparent, #e4e4e7, transparent);
                    opacity: 0.5;
                }
                
                .mi-direction-label {
                    position: absolute;
                    right: 24px;
                    top: 24px;
                    font-size: 10px;
                    letter-spacing: 0.2em;
                    font-weight: 600;
                }
            `}</style>
    </div>
  );
}
