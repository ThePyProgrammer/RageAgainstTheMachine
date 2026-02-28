import { useEffect, useState } from "react";

interface TauntBubbleProps {
  text: string;
  onDismiss?: () => void;
}

/**
 * Speech-bubble overlay for opponent taunts.
 * Only renders when `text` is non-empty. Auto-dismisses after 6 seconds.
 */
export const TauntBubble = ({ text, onDismiss }: TauntBubbleProps) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!text) {
      setVisible(false);
      return;
    }

    setVisible(true);
    const timer = setTimeout(() => {
      setVisible(false);
      onDismiss?.();
    }, 6_000);

    return () => clearTimeout(timer);
  }, [text, onDismiss]);

  if (!visible || !text) return null;

  return (
    <div className="pointer-events-none absolute right-6 top-1/3 z-20 max-w-[320px] animate-[fadeInScale_0.25s_ease-out]">
      {/* Bubble */}
      <div className="pointer-events-auto relative rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-black/80 to-cyan-950/60 px-5 py-3 text-sm leading-relaxed text-cyan-100 shadow-xl shadow-cyan-900/30 backdrop-blur-md">
        {text}

        {/* Tail */}
        <div className="absolute -left-2 top-6 h-4 w-4 rotate-45 border-l border-b border-cyan-400/30 bg-black/80" />

        {/* Dismiss */}
        <button
          type="button"
          className="pointer-events-auto absolute -top-2 -right-2 flex h-5 w-5 items-center justify-center rounded-full bg-black/70 text-[10px] text-white/50 transition hover:text-white"
          onClick={() => {
            setVisible(false);
            onDismiss?.();
          }}
        >
          ✕
        </button>
      </div>
    </div>
  );
};
