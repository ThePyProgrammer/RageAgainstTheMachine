import { drawShapePath } from "@/features/pong/game/shapes";
import { getCachedPattern } from "@/features/pong/game/patterns";
import {
  deriveScoreAccentColor,
  resolveUiColorToken,
  type UiSettings,
} from "@/features/pong/types/pongSettings";
import type { RuntimeState } from "@/features/pong/types/pongRuntime";

const buildBoundaryGradient = (ctx: CanvasRenderingContext2D) => {
  const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
  gradient.addColorStop(0, "rgb(3, 7, 18)");
  gradient.addColorStop(1, "rgb(7, 13, 30)");
  return gradient;
};

const drawHorizontalDashedLine = (
  ctx: CanvasRenderingContext2D,
  x1: number,
  x2: number,
  y: number,
  color: string,
) => {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 10]);
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x2, y);
  ctx.stroke();
  ctx.restore();
};

const drawBoundary = (ctx: CanvasRenderingContext2D, color: string, height: number) => {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.6;
  ctx.lineWidth = 2;
  ctx.beginPath();
  // Draw left and right boundaries (ball bounces off these walls)
  ctx.moveTo(0, 0);
  ctx.lineTo(0, height);
  ctx.moveTo(ctx.canvas.width, 0);
  ctx.lineTo(ctx.canvas.width, height);
  ctx.stroke();
  ctx.restore();
};

export const renderFrame = (
  ctx: CanvasRenderingContext2D,
  runtimeState: RuntimeState,
  settings: UiSettings,
) => {
  const { width, height, ball, topPaddle, bottomPaddle } = runtimeState;
  const palette = resolveUiColorToken(settings.theme.palette);
  const lineColor = resolveUiColorToken(settings.theme.lineColor);
  const scoreAccent = resolveUiColorToken(
    deriveScoreAccentColor(settings.theme),
  );
  const glow = settings.theme.glowIntensity;

  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = buildBoundaryGradient(ctx);
  ctx.fillRect(0, 0, width, height);

  ctx.shadowColor = lineColor;
  ctx.shadowBlur = 12 * glow;
  drawBoundary(ctx, palette, height);
  drawHorizontalDashedLine(ctx, 0, width, height / 2, lineColor);
  ctx.restore();

  const primary = resolveUiColorToken(settings.avatar.primaryColorToken);
  const secondary = resolveUiColorToken(settings.avatar.secondaryColorToken ?? primary);
  const secondaryToken = settings.avatar.secondaryColorToken ?? settings.avatar.primaryColorToken;
  const pattern = getCachedPattern(ctx, {
    pattern: settings.avatar.pattern,
    seed: settings.avatar.patternSeed,
    primaryToken: settings.avatar.primaryColorToken,
    secondaryToken,
  });

  const paintShape = (
    shape: "circle" | "oval" | "triangle" | "square" | "hexagon",
    x: number,
    y: number,
    w: number,
    h: number,
    strokeColor?: string,
  ) => {
    ctx.save();
    if (settings.avatar.pattern === "solid") {
      ctx.fillStyle = primary;
    } else {
      ctx.fillStyle = pattern;
    }
    drawShapePath(ctx, shape, x, y, w, h);
    ctx.fill();
    ctx.strokeStyle = strokeColor ?? scoreAccent;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.75 + glow * 0.25;
    ctx.stroke();
    ctx.restore();
  };

  paintShape(
    settings.avatar.paddleShape,
    topPaddle.x,
    topPaddle.y,
    topPaddle.width,
    topPaddle.height,
    lineColor,
  );
  paintShape(
    settings.avatar.paddleShape,
    bottomPaddle.x,
    bottomPaddle.y,
    bottomPaddle.width,
    bottomPaddle.height,
    lineColor,
  );

  const diam = ball.radius * 2;
  paintShape(
    settings.avatar.ballShape,
    ball.x - ball.radius,
    ball.y - ball.radius,
    diam,
    diam,
    secondary,
  );
};
