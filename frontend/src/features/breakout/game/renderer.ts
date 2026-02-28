import { drawShapePath } from "@/features/pong/game/shapes";
import { getCachedPattern } from "@/features/pong/game/patterns";
import {
  deriveScoreAccentColor,
  resolveUiColorToken,
  type UiSettings,
} from "@/features/pong/types/pongSettings";
import type { RuntimeState } from "@/features/breakout/types/breakoutRuntime";

const buildBoundaryGradient = (ctx: CanvasRenderingContext2D) => {
  const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
  gradient.addColorStop(0, "rgb(3, 7, 18)");
  gradient.addColorStop(1, "rgb(7, 13, 30)");
  return gradient;
};

const drawBoundary = (
  ctx: CanvasRenderingContext2D,
  color: string,
  width: number,
  height: number,
) => {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, width, height);
  ctx.restore();
};

export const renderFrame = (
  ctx: CanvasRenderingContext2D,
  runtimeState: RuntimeState,
  settings: UiSettings,
) => {
  const { width, height, ball, paddle, bricks } = runtimeState;
  const palette = resolveUiColorToken(settings.theme.palette);
  const lineColor = resolveUiColorToken(settings.theme.lineColor);
  const scoreAccent = resolveUiColorToken(deriveScoreAccentColor(settings.theme));
  const glow = settings.theme.glowIntensity;

  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = buildBoundaryGradient(ctx);
  ctx.fillRect(0, 0, width, height);
  drawBoundary(ctx, palette, width, height);
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

  ctx.save();
  ctx.shadowColor = lineColor;
  ctx.shadowBlur = 10 * glow;
  for (const brick of bricks) {
    if (!brick.active) {
      continue;
    }

    ctx.save();
    ctx.fillStyle = settings.avatar.pattern === "solid" ? primary : pattern;
    ctx.globalAlpha = 0.9;
    ctx.fillRect(brick.x, brick.y, brick.width, brick.height);
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.78;
    ctx.strokeRect(brick.x, brick.y, brick.width, brick.height);
    ctx.restore();
  }
  ctx.restore();

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
    paddle.x,
    paddle.y,
    paddle.width,
    paddle.height,
    lineColor,
  );

  const diameter = ball.radius * 2;
  paintShape(
    settings.avatar.ballShape,
    ball.x - ball.radius,
    ball.y - ball.radius,
    diameter,
    diameter,
    secondary,
  );
};
