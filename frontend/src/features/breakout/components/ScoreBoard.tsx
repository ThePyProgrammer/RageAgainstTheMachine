import type { FunctionComponent } from "react";

type ScoreBoardProps = {
  score: number;
  lives: number;
  level: number;
  accentColor: string;
};

export const ScoreBoard: FunctionComponent<ScoreBoardProps> = ({
  score,
  lives,
  level,
  accentColor,
}) => {
  return (
    <div className="pointer-events-none fixed top-3 left-1/2 -translate-x-1/2 z-20">
      <div className="rounded-full border border-zinc-600 bg-black/70 px-6 py-2 text-white text-xl font-semibold shadow-[0_0_24px] font-mono">
        <span className="text-zinc-400 mr-2">Score</span>
        <span style={{ color: accentColor }}>{score}</span>
        <span className="text-zinc-500 mx-4">|</span>
        <span className="text-zinc-400 mr-2">Lives</span>
        <span style={{ color: accentColor }}>{lives}</span>
        <span className="text-zinc-500 mx-4">|</span>
        <span className="text-zinc-400 mr-2">Level</span>
        <span style={{ color: accentColor }}>{level}</span>
      </div>
    </div>
  );
};
