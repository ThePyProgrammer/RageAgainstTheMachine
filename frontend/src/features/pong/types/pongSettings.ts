export type ColorPalette =
  | "neonRed"
  | "neonOrange"
  | "neonYellow"
  | "neonBlue"
  | "neonViolet"
  | "neonCyan"
  | "neonGreen"
  | "grayscale";

export type LineColor =
  | "white"
  | "neonRed"
  | "neonOrange"
  | "neonYellow"
  | "neonBlue"
  | "neonViolet"
  | "neonCyan"
  | "neonGreen";

export type PatternType =
  | "solid"
  | "moire"
  | "spiral"
  | "dots"
  | "rings"
  | "stripes";

export type ShapeType = "circle" | "oval" | "triangle" | "square" | "hexagon";

export interface UiThemeSettings {
  palette: ColorPalette;
  lineColor: LineColor;
  scoreAccentColor?: LineColor;
  glowIntensity: number;
}

export interface AvatarSettings {
  paddleShape: ShapeType;
  ballShape: ShapeType;
  pattern: PatternType;
  patternSeed: number;
  primaryColorToken: string;
  secondaryColorToken?: string;
}

export interface UiSettings {
  version: 1;
  theme: UiThemeSettings;
  avatar: AvatarSettings;
}

export const UI_COLOR_TOKEN_TO_CSS: Record<ColorPalette | LineColor, string> = {
  white: "rgb(255, 255, 255)",
  neonRed: "rgb(248, 113, 113)",
  neonOrange: "rgb(251, 146, 60)",
  neonYellow: "rgb(250, 204, 21)",
  neonBlue: "rgb(59, 130, 246)",
  neonViolet: "rgb(167, 139, 250)",
  neonCyan: "rgb(34, 211, 238)",
  neonGreen: "rgb(74, 222, 128)",
  grayscale: "rgb(228, 228, 231)",
};

export const resolveUiColorToken = (token: string): string =>
  UI_COLOR_TOKEN_TO_CSS[token as keyof typeof UI_COLOR_TOKEN_TO_CSS] ??
  token;

export const deriveScoreAccentColor = (theme: UiThemeSettings): LineColor =>
  theme.scoreAccentColor ?? theme.lineColor;
