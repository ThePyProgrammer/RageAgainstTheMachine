import type { UiSettings } from "@/features/pong/types/pongSettings";

const SETTINGS_KEY = "pong_ui_settings_v1";

const defaultThemeSettings = () => ({
  palette: "neonBlue" as const,
  lineColor: "neonCyan" as const,
  scoreAccentColor: "neonGreen" as const,
  glowIntensity: 0.35,
});

const defaultAvatarSettings = () => ({
  paddleShape: "square" as const,
  ballShape: "circle" as const,
  pattern: "dots" as const,
  patternSeed: 42,
  primaryColorToken: "neonBlue",
  secondaryColorToken: "white",
});

export const defaultSettings = (): UiSettings => ({
  version: 1,
  theme: defaultThemeSettings(),
  avatar: defaultAvatarSettings(),
});

export const loadSettings = (): UiSettings => {
  if (typeof window === "undefined") {
    return defaultSettings();
  }

  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) {
      return defaultSettings();
    }

    const value = JSON.parse(raw) as unknown;
    if (!value || typeof value !== "object") {
      return defaultSettings();
    }

    const maybeSettings = value as {
      version?: unknown;
      theme?: unknown;
      avatar?: unknown;
    };

    if (maybeSettings.version !== 1) {
      return defaultSettings();
    }

    const loadedTheme = maybeSettings.theme as
      | UiSettings["theme"]
      | undefined;
    const loadedAvatar = maybeSettings.avatar as
      | UiSettings["avatar"]
      | undefined;

    if (!loadedTheme || !loadedAvatar) {
      return defaultSettings();
    }

    const versioned: UiSettings = {
      version: 1,
      theme: {
        ...defaultThemeSettings(),
        ...loadedTheme,
      },
      avatar: {
        ...defaultAvatarSettings(),
        ...loadedAvatar,
      },
    };

    return versioned;
  } catch {
    return defaultSettings();
  }
};

export const saveSettings = (settings: UiSettings): void => {
  if (typeof window === "undefined") {
    return;
  }
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
};
