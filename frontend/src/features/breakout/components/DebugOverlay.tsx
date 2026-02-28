import type { DebugStats } from "@/features/breakout/types/breakoutRuntime";

type DebugOverlayProps = DebugStats;

const formatNormalList = (normals: DebugStats["collisionNormals"]) =>
  normals?.map((normal) => `(${normal.x},${normal.y})`).join(" ") || "none";

export const DebugOverlay = ({
  fps,
  latencyMs,
  thonkConnected,
  calibrationQuality,
  ballX,
  ballY,
  ballVX,
  ballVY,
  deltaMs,
  collisionNormals,
  collisionResolvedPerSecond,
  positionClampedPerSecond,
  bricksRemaining,
  speedMultiplier,
}: DebugOverlayProps) => {
  const speedBonusPct =
    typeof speedMultiplier === "number" ? Math.max(0, (speedMultiplier - 1) * 100) : 0;

  return (
    <div className="fixed bottom-3 left-3 z-20 rounded bg-black/90 border border-emerald-400/40 p-3 text-emerald-300 font-mono text-xs shadow-[0_0_16px_rgba(16,185,129,0.3)] space-y-1">
      <p>FPS: {fps.toFixed(1)}</p>
      <p>Latency: {latencyMs.toFixed(0)}ms</p>
      <p>Latency dt: {typeof deltaMs === "number" ? `${deltaMs.toFixed(1)}ms` : "n/a"}</p>
      <p>Thonk: {thonkConnected ? "on" : "off"}</p>
      {typeof calibrationQuality === "number" && <p>Cal: {calibrationQuality.toFixed(2)}</p>}
      <p>Ball: {ballX?.toFixed(1) ?? "n/a"}/{ballY?.toFixed(1) ?? "n/a"}</p>
      <p>Ball v: {ballVX?.toFixed(1) ?? "n/a"}/{ballVY?.toFixed(1) ?? "n/a"}</p>
      <p>Bricks: {bricksRemaining ?? "n/a"}</p>
      <p>Speed bonus: +{speedBonusPct.toFixed(0)}%</p>
      <p>Collisions/sec: {(collisionResolvedPerSecond ?? 0).toFixed(2)}</p>
      <p>Clamps/sec: {(positionClampedPerSecond ?? 0).toFixed(2)}</p>
      <p>Normals: {formatNormalList(collisionNormals)}</p>
    </div>
  );
};
