type StressMeterProps = {
  stressLevel: number;
};

export const StressMeter = ({ stressLevel }: StressMeterProps) => {
  const clamped = Math.max(0, Math.min(1, stressLevel));
  const pct = Math.round(clamped * 100);
  const color =
    clamped > 0.7 ? "bg-red-500" : clamped > 0.4 ? "bg-amber-400" : "bg-emerald-400";

  return (
    <div className="fixed top-3 right-3 z-20 rounded border border-zinc-700 bg-black/75 px-4 py-3 text-white min-w-[150px]">
      <p className="text-xs uppercase tracking-wide text-zinc-300">Stress</p>
      <p className="text-2xl font-bold">{pct}%</p>
      <div className="mt-2 h-2 rounded bg-zinc-800 overflow-hidden">
        <div className={`${color} h-full`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};
