import { useEffect, useRef, useState } from "react";

interface TauntBubbleProps {
  text: string;
  audioBase64?: string;
  eventKey: number;
  onDismiss?: () => void;
  durationMs?: number;
}

const decodeAudioBase64ToUrl = (audioBase64: string): string | null => {
  if (!audioBase64) {
    return null;
  }

  try {
    const binary = window.atob(audioBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "audio/mpeg" });
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error("Failed to decode taunt audio payload", error);
    return null;
  }
};

/**
 * Speech-bubble overlay for opponent taunts.
 * Text visibility and optional speech playback are triggered from the same event payload.
 */
export const TauntBubble = ({
  text,
  audioBase64 = "",
  eventKey,
  onDismiss,
  durationMs = 6_000,
}: TauntBubbleProps) => {
  const [visible, setVisible] = useState(false);
  const audioUrlRef = useRef<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (!text) {
      setVisible(false);
      return;
    }

    setVisible(true);

    if (audioBase64) {
      const url = decodeAudioBase64ToUrl(audioBase64);
      if (url) {
        if (audioUrlRef.current) {
          URL.revokeObjectURL(audioUrlRef.current);
        }
        audioUrlRef.current = url;
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.play().catch((error) => {
          console.warn("Taunt audio playback was blocked or failed", error);
        });
      }
    }

    const timer = window.setTimeout(() => {
      setVisible(false);
      onDismiss?.();
    }, durationMs);

    return () => {
      window.clearTimeout(timer);
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
        audioUrlRef.current = null;
      }
    };
  }, [audioBase64, durationMs, eventKey, onDismiss, text]);

  if (!visible || !text) {
    return null;
  }

  return (
    <div className="pointer-events-none absolute right-6 top-1/3 z-20 max-w-[320px] animate-[fadeInScale_0.25s_ease-out]">
      <div className="pointer-events-auto relative rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-black/80 to-cyan-950/60 px-5 py-3 text-sm leading-relaxed text-cyan-100 shadow-xl shadow-cyan-900/30 backdrop-blur-md">
        {text}
        <div className="absolute -left-2 top-6 h-4 w-4 rotate-45 border-l border-b border-cyan-400/30 bg-black/80" />
        <button
          type="button"
          className="pointer-events-auto absolute -top-2 -right-2 flex h-5 w-5 items-center justify-center rounded-full bg-black/70 text-[10px] text-white/50 transition hover:text-white"
          onClick={() => {
            setVisible(false);
            onDismiss?.();
          }}
        >
          x
        </button>
      </div>
    </div>
  );
};
