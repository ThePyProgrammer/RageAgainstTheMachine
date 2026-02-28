import { useState } from "react";
import type { DebugState } from "../App";

interface DebugPanelProps {
  debug: DebugState;
}

export const DebugPanel = ({ debug }: DebugPanelProps) => {
  const [visible, setVisible] = useState(true);

  return (
    <>
      {/* Toggle button — always visible */}
      <button
        type="button"
        className="absolute top-3 right-3 z-20 rounded border border-sky-300/40 bg-black/70 px-2 py-1 text-xs text-sky-300"
        onClick={() => setVisible((v) => !v)}
      >
        {visible ? "Hide Debug" : "Show Debug"}
      </button>

      {visible && (
        <div className="absolute top-10 right-3 z-10 rounded border border-sky-300/40 bg-black/80 p-3 font-mono text-xs text-sky-200 select-none">
          <div className="mb-1 font-bold text-sky-400">Debug HUD</div>
          <div>Tick: {debug.tick}</div>
          <div className="mt-1 font-bold text-sky-400">Control Frame</div>
          <div>
            Throttle: {debug.lastInput.throttle.toFixed(2)} | Turn:{" "}
            {debug.lastInput.turn.toFixed(2)} | Fire:{" "}
            {debug.lastInput.fire ? "YES" : "no"}
          </div>
          <div>Source: {debug.lastInput.source}</div>
          <div className="mt-1 font-bold text-sky-400">Player Pose</div>
          <div>
            X: {debug.playerPos.x.toFixed(1)} Y: {debug.playerPos.y.toFixed(1)}{" "}
            Yaw: {((debug.playerPos.yaw * 180) / Math.PI).toFixed(0)}°
          </div>
          <div className="mt-1 font-bold text-sky-400">World</div>
          <div>Projectiles: {debug.projectileCount}</div>
          <div>Queue: {debug.queueLen}</div>
          <div className="mt-1 font-bold text-sky-400">Events</div>
          <div>Last: {debug.lastEvent}</div>
        </div>
      )}
    </>
  );
};
