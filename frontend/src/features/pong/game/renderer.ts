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

const drawCenteredDashedLine = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y1: number,
  y2: number,
  color: string,
) => {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 10]);
  ctx.beginPath();
  ctx.moveTo(x, y1);
  ctx.lineTo(x, y2);
  ctx.stroke();
  ctx.restore();
};

const drawBoundary = (ctx: CanvasRenderingContext2D, color: string, width: number) => {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.6;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(width, 0);
  ctx.moveTo(0, ctx.canvas.height);
  ctx.lineTo(width, ctx.canvas.height);
  ctx.stroke();
  ctx.restore();
};

export const renderFrame = (
  ctx: CanvasRenderingContext2D,
  runtimeState: RuntimeState,
  settings: UiSettings,
) => {
  const { width, height, ball, leftPaddle, rightPaddle } = runtimeState;
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
  drawBoundary(ctx, palette, width);
  drawCenteredDashedLine(ctx, width / 2, 0, height, lineColor);
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
    leftPaddle.x,
    leftPaddle.y,
    leftPaddle.width,
    leftPaddle.height,
    lineColor,
  );
  paintShape(
    settings.avatar.paddleShape,
    rightPaddle.x,
    rightPaddle.y,
    rightPaddle.width,
    rightPaddle.height,
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
