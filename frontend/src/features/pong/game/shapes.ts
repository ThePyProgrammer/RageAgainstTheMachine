import type { ShapeType } from "@/features/pong/types/pongSettings";

export const drawShapePath = (
  ctx: CanvasRenderingContext2D,
  shape: ShapeType,
  x: number,
  y: number,
  width: number,
  height: number,
) => {
  ctx.beginPath();

  switch (shape) {
    case "circle": {
      const radius = Math.min(width, height) / 2;
      ctx.ellipse(x + width / 2, y + height / 2, radius, radius, 0, 0, Math.PI * 2);
      break;
    }
    case "oval":
      ctx.ellipse(x + width / 2, y + height / 2, width / 2, height / 2, 0, 0, Math.PI * 2);
      break;
    case "square":
      ctx.rect(x, y, width, height);
      break;
    case "triangle": {
      const apex = y + height * 0.1;
      const base = y + height * 0.88;
      ctx.moveTo(x + width / 2, apex);
      ctx.lineTo(x + width, base);
      ctx.lineTo(x, base);
      break;
    }
    case "hexagon": {
      const cx = x + width / 2;
      const cy = y + height / 2;
      const rx = width / 2;
      const ry = height / 2;
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i - Math.PI / 2;
        const px = cx + rx * Math.cos(angle);
        const py = cy + ry * Math.sin(angle);
        if (i === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      break;
    }
  }

  ctx.closePath();
};
