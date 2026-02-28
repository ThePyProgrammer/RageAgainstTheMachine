import {
  type AvatarSettings,
  type PatternType,
  type ShapeType,
  type UiSettings,
} from "@/features/pong/types/pongSettings";
import { PatternPreview } from "@/features/pong/components/PatternPreview";
import { UI_COLOR_TOKEN_TO_CSS } from "@/features/pong/types/pongSettings";

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

export const AvatarCustomizer = ({ settings, onChange }: AvatarCustomizerProps) => {
  const colorize = (color: keyof typeof UI_COLOR_TOKEN_TO_CSS) =>
    UI_COLOR_TOKEN_TO_CSS[color];

  return (
    <section className="space-y-4">
      <h3 className="font-semibold text-sm uppercase tracking-wide text-zinc-200">Avatar</h3>
      <PatternPreview settings={settings} />

      <div>
        <p className="text-xs text-zinc-400 mb-1">Paddle shape</p>
        <div className="flex gap-2 flex-wrap">
          {shapeOptions.map((shape) => (
            <button
              type="button"
              key={`paddle-${shape}`}
              className={`rounded border px-3 py-1 text-xs ${settings.avatar.paddleShape === shape ? "bg-white/20 border-white" : "border-zinc-500"}`}
              onClick={() => onChange(setAvatar(settings, { paddleShape: shape }))}
            >
              {shape}
            </button>
          ))}
        </div>
      </div>

      <div>
        <p className="text-xs text-zinc-400 mb-1">Ball shape</p>
        <div className="flex gap-2 flex-wrap">
          {shapeOptions.map((shape) => (
            <button
              type="button"
              key={`ball-${shape}`}
              className={`rounded border px-3 py-1 text-xs ${settings.avatar.ballShape === shape ? "bg-white/20 border-white" : "border-zinc-500"}`}
              onClick={() => onChange(setAvatar(settings, { ballShape: shape }))}
            >
              {shape}
            </button>
          ))}
        </div>
      </div>

      <div>
        <p className="text-xs text-zinc-400 mb-1">Pattern</p>
        <div className="grid grid-cols-2 gap-2">
          {patternOptions.map((pattern) => (
            <button
              key={pattern}
              type="button"
              className={`rounded border px-3 py-2 text-xs ${settings.avatar.pattern === pattern ? "bg-white/20 border-white" : "border-zinc-500"}`}
              onClick={() => onChange(setAvatar(settings, { pattern }))}
            >
              {pattern}
            </button>
          ))}
        </div>
      </div>

      <div>
        <p className="text-xs text-zinc-400 mb-1">Primary / secondary color token</p>
        <div className="flex gap-2">
          <select
            value={settings.avatar.primaryColorToken}
            onChange={(e) => onChange(setAvatar(settings, { primaryColorToken: e.target.value }))}
          >
            {tokenOptions.map((token) => (
              <option key={`primary-${token}`} value={token}>
                {token}
              </option>
            ))}
          </select>
          <select
            value={settings.avatar.secondaryColorToken}
            onChange={(e) => onChange(setAvatar(settings, { secondaryColorToken: e.target.value }))}
          >
            <option value="">none</option>
            {tokenOptions.map((token) => (
              <option key={`secondary-${token}`} value={token}>
                {token}
              </option>
            ))}
          </select>
        </div>
        <div className="mt-2 flex gap-2">
          {tokenOptions.map((token) => (
            <span
              key={token}
              className="h-4 w-4 rounded border border-zinc-500 inline-block"
              style={{ background: colorize(token) }}
            />
          ))}
        </div>
      </div>

      <button
        type="button"
        className="rounded border border-cyan-500 px-3 py-1 text-xs"
        onClick={() =>
          onChange(setAvatar(settings, { patternSeed: settings.avatar.patternSeed + 1 + Date.now() % 999 }))
        }
      >
        Randomize seed
      </button>
    </section>
  );
};
