import {
  type AvatarSettings,
  type PatternType,
  type ShapeType,
  type UiSettings,
} from "@/features/pong/types/pongSettings";
import { PatternPreview } from "@/features/pong/components/PatternPreview";
import { UI_COLOR_TOKEN_TO_CSS, resolveUiColorToken } from "@/features/pong/types/pongSettings";

const shapeOptions: ShapeType[] = ["circle", "oval", "triangle", "square", "hexagon"];
const patternOptions: PatternType[] = [
  "solid",
  "moire",
  "spiral",
  "dots",
  "rings",
  "stripes",
];
const tokenOptions: Array<keyof typeof UI_COLOR_TOKEN_TO_CSS> = [
  "neonRed",
  "neonOrange",
  "neonYellow",
  "neonBlue",
  "neonViolet",
  "neonCyan",
  "neonGreen",
  "grayscale",
  "white",
];

type AvatarCustomizerProps = {
  settings: UiSettings;
  onChange: (next: UiSettings) => void;
};

const setAvatar = (settings: UiSettings, patch: Partial<AvatarSettings>): UiSettings => ({
  ...settings,
  avatar: {
    ...settings.avatar,
    ...patch,
  },
});

/** Pill button toggling selection state */
const PillBtn = ({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`rounded-full px-4 py-1.5 text-[11px] font-semibold uppercase tracking-wide transition-all duration-150 border ${
      active
        ? "bg-white text-black border-white shadow-[0_0_12px_rgba(255,255,255,0.4)]"
        : "bg-transparent text-white/50 border-white/15 hover:border-white/40 hover:text-white/80"
    }`}
  >
    {label}
  </button>
);

export const AvatarCustomizer = ({ settings, onChange }: AvatarCustomizerProps) => {
  const primaryCss = resolveUiColorToken(settings.avatar.primaryColorToken);

  return (
    <section className="space-y-6">
      <h3 className="text-[11px] font-bold uppercase tracking-[0.2em] text-white/40 mb-3">
        Avatar
      </h3>

      {/* ── live preview strip ────────────────────────────── */}
      <PatternPreview settings={settings} />

      {/* ── paddle shape ──────────────────────────────────── */}
      <div>
        <p className="text-xs text-white/50 mb-2">Paddle shape</p>
        <div className="flex gap-2 flex-wrap">
          {shapeOptions.map((shape) => (
            <PillBtn
              key={`paddle-${shape}`}
              label={shape}
              active={settings.avatar.paddleShape === shape}
              onClick={() => onChange(setAvatar(settings, { paddleShape: shape }))}
            />
          ))}
        </div>
      </div>

      {/* ── ball shape ────────────────────────────────────── */}
      <div>
        <p className="text-xs text-white/50 mb-2">Ball shape</p>
        <div className="flex gap-2 flex-wrap">
          {shapeOptions.map((shape) => (
            <PillBtn
              key={`ball-${shape}`}
              label={shape}
              active={settings.avatar.ballShape === shape}
              onClick={() => onChange(setAvatar(settings, { ballShape: shape }))}
            />
          ))}
        </div>
      </div>

      {/* ── pattern ───────────────────────────────────────── */}
      <div>
        <p className="text-xs text-white/50 mb-2">Pattern</p>
        <div className="grid grid-cols-3 gap-2">
          {patternOptions.map((pattern) => (
            <button
              key={pattern}
              type="button"
              className={`rounded-lg border px-3 py-2.5 text-xs font-semibold uppercase tracking-wide transition-all duration-150 ${
                settings.avatar.pattern === pattern
                  ? "bg-white/15 border-white text-white shadow-[0_0_10px_rgba(255,255,255,0.2)]"
                  : "bg-white/[0.03] border-white/15 text-white/50 hover:border-white/40 hover:text-white/80"
              }`}
              onClick={() => onChange(setAvatar(settings, { pattern }))}
            >
              {pattern}
            </button>
          ))}
        </div>
      </div>

      {/* ── color tokens ──────────────────────────────────── */}
      <div>
        <p className="text-xs text-white/50 mb-2">Primary / secondary color</p>
        <div className="flex gap-3">
          {/* primary selector */}
          <div className="relative">
            <select
              value={settings.avatar.primaryColorToken}
              onChange={(e) =>
                onChange(setAvatar(settings, { primaryColorToken: e.target.value }))
              }
              className="w-full appearance-none rounded-lg border border-white/20 bg-white/5 py-2 pl-10 pr-8 text-sm text-white/90 outline-none transition focus:border-white/50"
            >
              {tokenOptions.map((token) => (
                <option key={`primary-${token}`} value={token}>
                  {token}
                </option>
              ))}
            </select>
            <span
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full"
              style={{ background: primaryCss, boxShadow: `0 0 6px ${primaryCss}` }}
            />
          </div>

          {/* secondary selector */}
          <div className="relative">
            <select
              value={settings.avatar.secondaryColorToken}
              onChange={(e) =>
                onChange(setAvatar(settings, { secondaryColorToken: e.target.value }))
              }
              className="w-full appearance-none rounded-lg border border-white/20 bg-white/5 py-2 pl-10 pr-8 text-sm text-white/90 outline-none transition focus:border-white/50"
            >
              <option value="">none</option>
              {tokenOptions.map((token) => (
                <option key={`secondary-${token}`} value={token}>
                  {token}
                </option>
              ))}
            </select>
            <span
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full"
              style={{
                background: resolveUiColorToken(settings.avatar.secondaryColorToken ?? "white"),
                boxShadow: `0 0 6px ${resolveUiColorToken(settings.avatar.secondaryColorToken ?? "white")}`,
              }}
            />
          </div>
        </div>

        {/* swatch strip */}
        <div className="mt-3 flex gap-1.5">
          {tokenOptions.map((token) => (
            <button
              key={token}
              type="button"
              onClick={() => onChange(setAvatar(settings, { primaryColorToken: token }))}
              className={`h-5 w-5 rounded-sm border transition ${
                settings.avatar.primaryColorToken === token
                  ? "border-white ring-1 ring-white/60 ring-offset-1 ring-offset-[#111118]"
                  : "border-white/15 hover:border-white/50"
              }`}
              style={{
                background: UI_COLOR_TOKEN_TO_CSS[token],
                boxShadow:
                  settings.avatar.primaryColorToken === token
                    ? `0 0 8px ${UI_COLOR_TOKEN_TO_CSS[token]}`
                    : undefined,
              }}
            />
          ))}
        </div>
      </div>

      {/* ── randomize seed ────────────────────────────────── */}
      <button
        type="button"
        className="rounded-lg border border-white/20 px-5 py-2 text-xs font-semibold uppercase tracking-wider text-white/60 transition hover:border-white/50 hover:text-white"
        onClick={() =>
          onChange(
            setAvatar(settings, {
              patternSeed: settings.avatar.patternSeed + 1 + (Date.now() % 999),
            }),
          )
        }
      >
        Randomize seed
      </button>
    </section>
  );
};
