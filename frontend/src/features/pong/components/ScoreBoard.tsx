import type { FunctionComponent } from "react";

type ScoreBoardProps = {
  playerScore: number;
  aiScore: number;
  accentColor: string;
};

export const ScoreBoard: FunctionComponent<ScoreBoardProps> = ({
  playerScore,
  aiScore,
  accentColor,
}) => {
  return (
    <div className="pointer-events-none fixed top-3 left-1/2 -translate-x-1/2 z-20">
      <div className="rounded-full border border-zinc-600 bg-black/70 px-6 py-2 text-white text-xl font-semibold shadow-[0_0_24px] font-mono">
        <span className="text-zinc-400 mr-4">P1</span>
        <span style={{ color: accentColor }}>{playerScore}</span>
        <span className="text-zinc-500 mx-4">|</span>
        <span style={{ color: accentColor }}>{aiScore}</span>
        <span className="text-zinc-400 ml-4">AI</span>
      </div>
    </div>
  );
};
