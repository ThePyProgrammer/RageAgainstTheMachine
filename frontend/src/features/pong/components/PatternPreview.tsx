import { useEffect, useRef } from "react";
import { drawShapePath } from "@/features/pong/game/shapes";
import { getCachedPattern } from "@/features/pong/game/patterns";
import {
  resolveUiColorToken,
  type UiSettings,
  type ShapeType,
} from "@/features/pong/types/pongSettings";

type PatternPreviewProps = {
  settings: UiSettings;
};

export const PatternPreview = ({ settings }: PatternPreviewProps) => {
  const leftCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const ballCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lineCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const render = (canvas: HTMLCanvasElement | null, shape: ShapeType, size = 90) => {
      if (!canvas) {
        return;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      canvas.style.width = `${size}px`;
      canvas.style.height = `${size}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);

      // High-contrast dark background
      ctx.fillStyle = "#050508";
      ctx.fillRect(0, 0, size, size);

      const pattern = getCachedPattern(ctx, {
        pattern: settings.avatar.pattern,
        seed: settings.avatar.patternSeed,
        primaryToken: settings.avatar.primaryColorToken,
        secondaryToken: settings.avatar.secondaryColorToken,
      });

      const centerW = size * 0.5;
      const centerH = size * 0.5;
      const x = (size - centerW) / 2;
      const y = (size - centerH) / 2;

      // Draw shape with glow for contrast
      const lineColor = resolveUiColorToken(settings.theme.lineColor);
      ctx.save();
      ctx.shadowColor = lineColor;
      ctx.shadowBlur = 8 * settings.theme.glowIntensity;

      ctx.fillStyle = pattern;
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 1.5;
      drawShapePath(ctx, shape, x, y, centerW, centerH);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    };

    render(leftCanvasRef.current, settings.avatar.paddleShape, 90);
    render(ballCanvasRef.current, settings.avatar.ballShape, 68);
    render(lineCanvasRef.current, "square", 70);
  }, [settings]);

  return (
    <div className="grid gap-4 sm:grid-cols-3">
      <div className="rounded-xl border border-white/10 bg-black/40 p-3 text-center">
        <p className="text-[10px] uppercase tracking-wider text-white/40 mb-2">
          Paddle fill
        </p>
        <canvas ref={leftCanvasRef} className="mx-auto rounded-lg" />
      </div>
      <div className="rounded-xl border border-white/10 bg-black/40 p-3 text-center">
        <p className="text-[10px] uppercase tracking-wider text-white/40 mb-2">
          Ball fill
        </p>
        <canvas ref={ballCanvasRef} className="mx-auto rounded-lg" />
      </div>
      <div className="rounded-xl border border-white/10 bg-black/40 p-3 text-center">
        <p className="text-[10px] uppercase tracking-wider text-white/40 mb-2">
          Line sample
        </p>
        <canvas ref={lineCanvasRef} className="mx-auto rounded-lg" />
      </div>
    </div>
  );
};
