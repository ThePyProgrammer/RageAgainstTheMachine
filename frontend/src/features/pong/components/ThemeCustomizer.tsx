import {
  UI_COLOR_TOKEN_TO_CSS,
  type LineColor,
  type UiSettings,
  type ColorPalette,
  resolveUiColorToken,
} from "@/features/pong/types/pongSettings";

type ThemeCustomizerProps = {
  settings: UiSettings;
  onChange: (next: UiSettings) => void;
};

const paletteOptions: ColorPalette[] = [
  "neonRed",
  "neonOrange",
  "neonYellow",
  "neonBlue",
  "neonViolet",
  "neonCyan",
  "neonGreen",
  "grayscale",
];

const lineOptions: LineColor[] = [
  "white",
  "neonRed",
  "neonOrange",
  "neonYellow",
  "neonBlue",
  "neonViolet",
  "neonCyan",
  "neonGreen",
];

export const ThemeCustomizer = ({ settings, onChange }: ThemeCustomizerProps) => {
  const updateTheme = (patch: Partial<UiSettings["theme"]>) => {
    onChange({
      ...settings,
      theme: {
        ...settings.theme,
        ...patch,
      },
    });
  };

  const paletteCss = resolveUiColorToken(settings.theme.palette);
  const lineCss = resolveUiColorToken(settings.theme.lineColor);

  return (
    <section className="space-y-6">
      <h3 className="text-[11px] font-bold uppercase tracking-[0.2em] text-white/40 mb-3">
        Theme
      </h3>

      {/* ── Palette ───────────────────────────────────────── */}
      <div>
        <p className="text-xs text-white/50 mb-2">Palette</p>
        <div className="relative">
          <select
            value={settings.theme.palette}
            onChange={(e) =>
              updateTheme({ palette: e.target.value as ColorPalette })
            }
            className="w-full appearance-none rounded-lg border border-white/20 bg-white/5 py-2.5 pl-10 pr-8 text-sm text-white/90 outline-none transition focus:border-white/50 cursor-pointer"
          >
            {paletteOptions.map((token) => (
              <option key={token} value={token}>
                {token}
              </option>
            ))}
          </select>
          <span
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full"
            style={{ background: paletteCss, boxShadow: `0 0 8px ${paletteCss}` }}
          />
          <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-white/30 text-xs">
            ▼
          </span>
        </div>

        {/* swatch row */}
        <div className="mt-3 flex gap-1.5">
          {paletteOptions.map((token) => {
            const active = settings.theme.palette === token;
            const css = UI_COLOR_TOKEN_TO_CSS[token];
            return (
              <button
                key={token}
                type="button"
                aria-label={`palette-${token}`}
                onClick={() => updateTheme({ palette: token })}
                className={`h-7 w-7 rounded-md border transition ${
                  active
                    ? "border-white ring-1 ring-white/60 ring-offset-1 ring-offset-[#111118]"
                    : "border-white/15 hover:border-white/50"
                }`}
                style={{
                  background: css,
                  boxShadow: active ? `0 0 10px ${css}` : undefined,
                }}
              />
            );
          })}
        </div>
      </div>

      {/* ── Line color ────────────────────────────────────── */}
      <div>
        <p className="text-xs text-white/50 mb-2">Line color</p>
        <div className="relative">
          <select
            value={settings.theme.lineColor}
            onChange={(e) =>
              updateTheme({ lineColor: e.target.value as LineColor })
            }
            className="w-full appearance-none rounded-lg border border-white/20 bg-white/5 py-2.5 pl-10 pr-8 text-sm text-white/90 outline-none transition focus:border-white/50 cursor-pointer"
          >
            {lineOptions.map((token) => (
              <option key={token} value={token}>
                {token}
              </option>
            ))}
          </select>
          <span
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full"
            style={{ background: lineCss, boxShadow: `0 0 8px ${lineCss}` }}
          />
          <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-white/30 text-xs">
            ▼
          </span>
        </div>

        {/* swatch row */}
        <div className="mt-3 flex gap-1.5">
          {lineOptions.map((token) => {
            const active = settings.theme.lineColor === token;
            const css = resolveUiColorToken(token);
            return (
              <button
                key={token}
                type="button"
                aria-label={`line-${token}`}
                onClick={() => updateTheme({ lineColor: token })}
                className={`h-7 w-7 rounded-md border transition ${
                  active
                    ? "border-white ring-1 ring-white/60 ring-offset-1 ring-offset-[#111118]"
                    : "border-white/15 hover:border-white/50"
                }`}
                style={{
                  background: css,
                  boxShadow: active ? `0 0 10px ${css}` : undefined,
                }}
              />
            );
          })}
        </div>
      </div>

      {/* ── Glow slider ───────────────────────────────────── */}
      <div>
        <p className="text-xs text-white/50 mb-2">
          Glow intensity{" "}
          <span className="text-white/30 ml-1">
            {Math.round(settings.theme.glowIntensity * 100)}%
          </span>
        </p>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={settings.theme.glowIntensity}
          onChange={(e) => updateTheme({ glowIntensity: Number(e.target.value) })}
          className="w-full accent-white h-1.5 rounded-full appearance-none bg-white/10 cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_8px_white]"
        />
      </div>
    </section>
  );
};
