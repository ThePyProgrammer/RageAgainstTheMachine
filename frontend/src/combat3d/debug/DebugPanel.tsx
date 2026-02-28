interface DebugState {
  queueLen: number;
  rttMs: number;
}

interface DebugPanelProps {
  debug: DebugState;
}

export const DebugPanel = ({ debug }: DebugPanelProps) => {
  return (
    <div className="absolute right-3 top-3 z-10 rounded border border-sky-300/40 bg-black/70 p-3 text-xs">
      <div>Queue {debug.queueLen}</div>
      <div>RTT {debug.rttMs}ms</div>
    </div>
  );
};
