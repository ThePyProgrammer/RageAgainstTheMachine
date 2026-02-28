import { UI_COLOR_TOKEN_TO_CSS, type LineColor, type UiSettings, type ColorPalette, resolveUiColorToken } from "@/features/pong/types/pongSettings";

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

  return (
    <section className="space-y-3">
      <h3 className="font-semibold text-sm uppercase tracking-wide text-zinc-200">Theme</h3>
      <div>
        <p className="text-xs text-zinc-400 mb-1">Palette</p>
        <div className="flex flex-wrap gap-2">
          {paletteOptions.map((token) => {
            const active = settings.theme.palette === token;
            return (
              <button
                key={token}
                type="button"
                aria-label={`palette-${token}`}
                className={`h-8 w-8 rounded border ${active ? "border-white" : "border-zinc-500"}`}
                style={{ background: UI_COLOR_TOKEN_TO_CSS[token] }}
                onClick={() => updateTheme({ palette: token })}
              />
            );
          })}
        </div>
      </div>

      <div>
        <p className="text-xs text-zinc-400 mb-1">Line</p>
        <div className="flex flex-wrap gap-2">
          {lineOptions.map((token) => {
            const active = settings.theme.lineColor === token;
            return (
              <button
                key={token}
                type="button"
                aria-label={`line-${token}`}
                className={`h-8 w-8 rounded border ${active ? "border-white" : "border-zinc-500"}`}
                style={{ background: resolveUiColorToken(token) }}
                onClick={() => updateTheme({ lineColor: token })}
              />
            );
          })}
        </div>
      </div>

      <div>
        <label htmlFor="glow" className="text-xs text-zinc-400">
          Glow
        </label>
        <input
          id="glow"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={settings.theme.glowIntensity}
          onChange={(e) => updateTheme({ glowIntensity: Number(e.target.value) })}
          className="w-full mt-1"
        />
      </div>
    </section>
  );
};
