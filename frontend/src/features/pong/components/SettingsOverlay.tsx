import { useEffect } from "react";
import { AvatarCustomizer } from "@/features/pong/components/AvatarCustomizer";
import { ThemeCustomizer } from "@/features/pong/components/ThemeCustomizer";
import type { UiSettings } from "@/features/pong/types/pongSettings";

type SettingsOverlayProps = {
  open: boolean;
  settings: UiSettings;
  onClose: () => void;
  onChange: (next: UiSettings) => void;
  onReset: () => void;
};

export const SettingsOverlay = ({
  open,
  settings,
  onClose,
  onChange,
  onReset,
}: SettingsOverlayProps) => {
  useEffect(() => {
    const onEsc = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onEsc);
    return () => window.removeEventListener("keydown", onEsc);
  }, [onClose]);

  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* backdrop */}
      <button
        type="button"
        aria-label="Dismiss settings"
        className="absolute inset-0 bg-black/80 backdrop-blur-md"
        onClick={onClose}
      />

      {/* panel */}
      <section
        role="dialog"
        aria-modal="true"
        className="relative w-[min(860px,95vw)] max-h-[88vh] overflow-y-auto rounded-2xl border border-white/10 bg-gradient-to-b from-[#111118] to-[#08080c] shadow-2xl p-8 text-white"
      >
        {/* close */}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close settings"
          className="absolute top-4 right-4 flex h-8 w-8 items-center justify-center rounded-full border border-white/20 text-white/50 transition hover:border-white/50 hover:text-white"
        >
          ✕
        </button>

        <h2 className="text-lg font-bold uppercase tracking-[0.15em] text-white/90 mb-8">
          Pong Settings
        </h2>

        <div className="grid gap-10 md:grid-cols-2">
          <AvatarCustomizer settings={settings} onChange={onChange} />
          <ThemeCustomizer settings={settings} onChange={onChange} />
        </div>

        <div className="mt-10 flex justify-end border-t border-white/5 pt-6">
          <button
            type="button"
            className="rounded-lg border border-white/10 px-6 py-2 text-xs font-semibold uppercase tracking-wider text-white/40 transition hover:border-red-400/40 hover:text-red-400"
            onClick={onReset}
          >
            Reset defaults
          </button>
        </div>
      </section>
    </div>
  );
};
