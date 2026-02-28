import type { HybridInputDebug } from "../bci/useHybridInput";

interface InputDebugHUDProps {
  debug: HybridInputDebug;
  visible: boolean;
}

const Row = ({ label, value }: { label: string; value: string }) => (
  <div className="flex justify-between gap-4">
    <span className="text-white/40">{label}</span>
    <span className="text-white/80">{value}</span>
  </div>
);

export const InputDebugHUD = ({ debug, visible }: InputDebugHUDProps) => {
  if (!visible) return null;

  const { mode, bci, keysHeld, lastSample } = debug;

  return (
    <div className="absolute bottom-12 left-3 z-20 min-w-[280px] rounded-lg border border-white/10 bg-black/80 backdrop-blur-sm p-4 font-mono text-[11px] text-white/70 space-y-1 select-none">
      <div className="text-white/40 uppercase text-[9px] tracking-widest mb-2">
        Hybrid Input Debug
      </div>

      <Row label="Mode" value={mode} />
      <Row
        label="BCI WS"
        value={bci.connected ? "● Connected" : "○ Disconnected"}
      />
      <Row label="BCI Rotation" value={bci.rotation ?? "—"} />
      <Row label="BCI Shoot" value={bci.shoot ? "FIRE" : "—"} />
      <Row label="BCI Raw" value={bci.lastRawCommand} />
      <Row
        label="BCI Last"
        value={
          bci.lastCommandTs ? `${Date.now() - bci.lastCommandTs}ms ago` : "—"
        }
      />

      <div className="border-t border-white/10 pt-1 mt-1" />
      <Row
        label="Keys held"
        value={keysHeld.length > 0 ? keysHeld.join(", ") : "—"}
      />

      {lastSample && (
        <>
          <div className="border-t border-white/10 pt-1 mt-1" />
          <Row label="Throttle" value={lastSample.throttle.toFixed(2)} />
          <Row label="Turn" value={lastSample.turn.toFixed(2)} />
          <Row label="Fire" value={lastSample.fire ? "YES" : "no"} />
        </>
      )}
    </div>
  );
};
