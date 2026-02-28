import { useEffect, useState } from "react";

type TauntBubbleProps = {
  text: string;
  durationMs: number;
  timestamp: number;
  onExpire: () => void;
};

export const TauntBubble = ({
  text,
  durationMs,
  timestamp,
  onExpire,
}: TauntBubbleProps) => {
  const [opacity, setOpacity] = useState(1);

  useEffect(() => {
    const elapsed = Date.now() - timestamp;
    const remaining = durationMs - elapsed;
    if (remaining <= 0) {
      onExpire();
      return;
    }

    const fadeStart = durationMs * 0.8;
    if (elapsed >= fadeStart) {
      const fade = 1 - (elapsed - fadeStart) / (durationMs - fadeStart);
      setOpacity(fade);
    }

    const timer = setTimeout(onExpire, remaining);
    return () => clearTimeout(timer);
  }, [durationMs, onExpire, timestamp]);

  return (
    <div
      className="pointer-events-none fixed top-16 right-4 z-20 w-[min(26rem,calc(100vw-2rem))] border-2 border-cyan-400 bg-black/80 px-4 py-3 rounded-lg text-white shadow-[0_0_30px] font-semibold text-base leading-snug"
      style={{ opacity }}
    >
      {text}
    </div>
  );
};
