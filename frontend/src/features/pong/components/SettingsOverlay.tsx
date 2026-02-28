import { useEffect } from "react";
import { X } from "lucide-react";
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
    <div className="fixed inset-0 z-30">
      <button
        type="button"
        aria-label="Dismiss settings"
        className="absolute inset-0 bg-black/70"
        onClick={onClose}
      />
      <section
        role="dialog"
        aria-modal="true"
        className="absolute inset-0 m-auto h-fit w-[min(920px,95%)] max-h-[90vh] overflow-auto rounded-lg border border-zinc-700 bg-zinc-900/95 p-5 text-zinc-100"
      >
        <header className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Pong settings</h2>
          <button
            type="button"
            onClick={onClose}
            className="h-8 w-8 rounded border border-zinc-600"
            aria-label="Close settings"
          >
            <X size={16} />
          </button>
        </header>

        <div className="grid gap-5 md:grid-cols-2">
          <AvatarCustomizer settings={settings} onChange={onChange} />
          <ThemeCustomizer settings={settings} onChange={onChange} />
        </div>

        <div className="mt-5 flex justify-end">
          <button
            type="button"
            className="rounded border border-zinc-500 px-3 py-1.5 text-xs"
            onClick={onReset}
          >
            Reset defaults
          </button>
        </div>
      </section>
    </div>
  );
};
