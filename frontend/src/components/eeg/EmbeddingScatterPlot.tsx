import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface EmbeddingPoint {
  reduced: {
    embeddings_2d: [number, number];
  };
}

interface EmbeddingScatterPlotProps {
  embeddings: EmbeddingPoint[];
  latest?: EmbeddingPoint | null;
}

export function EmbeddingScatterPlot({
  embeddings,
  latest,
}: EmbeddingScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [trailLength, setTrailLength] = useState(5);
  const [showLines, setShowLines] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (embeddings.length === 0) {
      // Draw placeholder text
      ctx.fillStyle = "#999";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        "Waiting for embeddings...",
        canvas.width / 2,
        canvas.height / 2,
      );
      return;
    }

    // Find bounds for scaling
    const xValues = embeddings.map((e) => e.reduced.embeddings_2d[0]);
    const yValues = embeddings.map((e) => e.reduced.embeddings_2d[1]);

    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    const padding = 40;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;

    // Scale function
    const scaleX = (x: number) => {
      if (xMax === xMin) return canvas.width / 2;
      return padding + ((x - xMin) / (xMax - xMin)) * width;
    };
    const scaleY = (y: number) => {
      if (yMax === yMin) return canvas.height / 2;
      return canvas.height - padding - ((y - yMin) / (yMax - yMin)) * height;
    };

    // Draw axes
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = "#666";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("t-SNE 1", canvas.width / 2, canvas.height - 10);
    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("t-SNE 2", 0, 0);
    ctx.restore();

    const totalPoints = embeddings.length;

    // Draw historical points (older = more transparent)
    embeddings.forEach((emb, idx) => {
      const x = scaleX(emb.reduced.embeddings_2d[0]);
      const y = scaleY(emb.reduced.embeddings_2d[1]);

      // Check if this point is in the trail (last N points excluding current)
      const isInTrail =
        idx >= totalPoints - trailLength && idx < totalPoints - 1;

      if (isInTrail) {
        // Trail points: red with gradient based on recency
        const trailIdx = idx - (totalPoints - trailLength);
        const trailAlpha = 0.4 + (trailIdx / (trailLength - 1)) * 0.4;
        ctx.fillStyle = `rgba(239, 68, 68, ${trailAlpha})`; // red with varying opacity

        // Draw larger points for trail
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
      } else {
        // Regular historical points
        const alpha = 0.3 + (idx / embeddings.length) * 0.5;
        ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`; // blue with varying opacity

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    // Draw connecting line for trail points only using smooth curves
    if (showLines && totalPoints > 1) {
      const trailStartIdx = Math.max(0, totalPoints - trailLength);
      const trailPoints = embeddings.slice(trailStartIdx);

      if (trailPoints.length > 1) {
        ctx.strokeStyle = "rgba(239, 68, 68, 0.2)"; // low alpha for transparency
        ctx.lineWidth = 2;
        ctx.setLineDash([]);

        ctx.beginPath();

        // Draw smooth spline through trail points
        const scaledPoints = trailPoints.map((emb) => ({
          x: scaleX(emb.reduced.embeddings_2d[0]),
          y: scaleY(emb.reduced.embeddings_2d[1]),
        }));

        // Start at first point
        ctx.moveTo(scaledPoints[0].x, scaledPoints[0].y);

        // Use quadratic curves for smooth spline
        for (let i = 1; i < scaledPoints.length; i++) {
          if (i === scaledPoints.length - 1) {
            // Last point: draw straight to it
            ctx.lineTo(scaledPoints[i].x, scaledPoints[i].y);
          } else {
            // Create smooth curve using control point between current and next
            const xc = (scaledPoints[i].x + scaledPoints[i + 1].x) / 2;
            const yc = (scaledPoints[i].y + scaledPoints[i + 1].y) / 2;
            ctx.quadraticCurveTo(scaledPoints[i].x, scaledPoints[i].y, xc, yc);
          }
        }

        ctx.stroke();
      }
    }

    // Draw latest point (highlighted)
    if (latest) {
      const x = scaleX(latest.reduced.embeddings_2d[0]);
      const y = scaleY(latest.reduced.embeddings_2d[1]);

      // Outer glow
      ctx.fillStyle = "rgba(239, 68, 68, 0.3)";
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, 2 * Math.PI);
      ctx.fill();

      // Inner point
      ctx.fillStyle = "#ef4444"; // red
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();

      // Draw coordinates
      ctx.fillStyle = "#333";
      ctx.font = "11px monospace";
      ctx.textAlign = "left";
      const coords = `(${latest.reduced.embeddings_2d[0].toFixed(2)}, ${latest.reduced.embeddings_2d[1].toFixed(2)})`;
      ctx.fillText(coords, x + 12, y - 5);
    }

    // Draw legend
    const legendX = padding + 10;
    const legendY = padding + 10;
    const legendWidth = 140;
    const legendHeight = 75;

    // Legend background
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
    ctx.strokeStyle = "#d0d0d0";
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

    // Legend items
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";

    // Historical points
    ctx.fillStyle = "rgba(59, 130, 246, 0.7)";
    ctx.beginPath();
    ctx.arc(legendX + 10, legendY + 15, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#333";
    ctx.fillText("Historical", legendX + 20, legendY + 19);

    // Trail points
    ctx.fillStyle = "rgba(239, 68, 68, 0.7)";
    ctx.beginPath();
    ctx.arc(legendX + 10, legendY + 35, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#333";
    ctx.fillText(`Trail (last ${trailLength})`, legendX + 20, legendY + 39);

    // Latest point
    ctx.fillStyle = "#ef4444";
    ctx.beginPath();
    ctx.arc(legendX + 10, legendY + 55, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#333";
    ctx.fillText("Current", legendX + 20, legendY + 59);
  }, [embeddings, latest, trailLength, showLines]);

  return (
    <Card className="h-full flex flex-col bg-white">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Embeddings Plot</CardTitle>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="show-lines"
                checked={showLines}
                onChange={(e) => setShowLines(e.target.checked)}
                className="w-4 h-4 rounded cursor-pointer accent-blue-600"
              />
              <label
                htmlFor="show-lines"
                className="text-xs text-muted-foreground whitespace-nowrap cursor-pointer"
              >
                Show Trail Line
              </label>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-muted-foreground whitespace-nowrap">
                Trail: {trailLength}
              </label>
              <input
                type="range"
                min="3"
                max="10"
                value={trailLength}
                onChange={(e) => setTrailLength(parseInt(e.target.value))}
                className="w-24 h-2 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex items-center justify-center ">
        <canvas
          ref={canvasRef}
          width={1000}
          height={800}
          className="border rounded bg-white"
        />
      </CardContent>
    </Card>
  );
}
