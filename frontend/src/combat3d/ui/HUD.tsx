import type { CombatState } from "../engine/types";

interface HudProps {
  state: CombatState | null;
  status: string;
  taunt?: string;
}

export const HUD = ({ state, status, taunt }: HudProps) => {
  const score = state ? `${state.score.player} : ${state.score.enemy}` : "0 : 0";

  return (
    <div className="pointer-events-none absolute top-3 left-3 z-10">
      <div className="rounded-lg border border-cyan-400/40 bg-black/60 p-3">
        <div>Combat3D</div>
        <div>Score {score}</div>
        <div>Tick {state?.tick ?? 0}</div>
        <div>Mode {status}</div>
        {typeof state?.difficulty.stress === "number" ? (
          <div>Stress {state.difficulty.stress.toFixed(2)}</div>
        ) : null}
        <div className="mt-2 text-cyan-200">{taunt}</div>
      </div>
    </div>
  );
};
