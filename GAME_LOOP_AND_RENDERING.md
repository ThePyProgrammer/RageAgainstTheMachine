# GAME_LOOP_AND_RENDERING.md

Canvas-based game rendering with requestAnimationFrame. No React re-renders per frame.

Location: `/apps/web/src/game/`

## Architecture

```
┌─────────────────────────────────────────────────┐
│  React Component (GameCanvas.tsx)               │
│  - Canvas ref                                   │
│  - Game state ref (not React state!)           │
│  - useEffect → start loop                       │
│  - Cleanup → stop loop                          │
└─────────────────────────────────────────────────┘
          │
          ├─ Refs (no re-render)
          │  - canvasRef: HTMLCanvasElement
          │  - gameStateRef: GameState
          │  - animationIdRef: number
          │
          ▼
┌─────────────────────────────────────────────────┐
│  Game Loop (gameLoop.ts)                        │
│  - requestAnimationFrame                        │
│  - Delta time calculation                       │
│  - Fixed timestep (60 Hz)                       │
│  - Render pipeline                              │
└─────────────────────────────────────────────────┘
          │
          ├─ Calls
          │
          ▼
┌─────────────────────────────────────────────────┐
│  Renderer (renderer.ts)                         │
│  - Clear canvas                                 │
│  - Draw field                                   │
│  - Draw paddles                                 │
│  - Draw ball                                    │
│  - Draw speech bubbles (overlay)                │
└─────────────────────────────────────────────────┘
```

## GameCanvas Component

File: `/apps/web/src/game/GameCanvas.tsx`

```typescript
import React, { useRef, useEffect } from 'react';
import { GameState } from '@packages/game';
import { startGameLoop, stopGameLoop } from './gameLoop';
import { CanvasRenderer } from './renderer';

interface GameCanvasProps {
  gameState: GameState; // Passed from parent (updated via WebSocket)
  onInputCommand: (command: 'UP' | 'DOWN' | 'NEUTRAL') => void;
  taunts: Array<{ text: string; timestamp: number; duration: number }>;
}

export const GameCanvas: React.FC<GameCanvasProps> = ({
  gameState,
  onInputCommand,
  taunts,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameStateRef = useRef<GameState>(gameState);
  const tauntsRef = useRef(taunts);
  const rendererRef = useRef<CanvasRenderer | null>(null);

  // Update refs when props change (no re-render triggers)
  useEffect(() => {
    gameStateRef.current = gameState;
  }, [gameState]);

  useEffect(() => {
    tauntsRef.current = taunts;
  }, [taunts]);

  // Initialize renderer and start loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas logical size
    canvas.width = 800;
    canvas.height = 600;

    // Initialize renderer
    rendererRef.current = new CanvasRenderer(ctx);

    // Start game loop
    const loopId = startGameLoop({
      getGameState: () => gameStateRef.current,
      getTaunts: () => tauntsRef.current,
      renderer: rendererRef.current,
      onInputCommand,
    });

    // Cleanup
    return () => {
      stopGameLoop(loopId);
    };
  }, [onInputCommand]);

  return (
    <canvas
      ref={canvasRef}
      className="border-2 border-gray-800 rounded-lg shadow-xl"
      style={{ width: '100%', height: 'auto' }}
    />
  );
};
```

## Game Loop

File: `/apps/web/src/game/gameLoop.ts`

```typescript
import { GameState } from '@packages/game';
import { CanvasRenderer } from './renderer';

interface GameLoopConfig {
  getGameState: () => GameState;
  getTaunts: () => Array<{ text: string; timestamp: number; duration: number }>;
  renderer: CanvasRenderer;
  onInputCommand: (command: 'UP' | 'DOWN' | 'NEUTRAL') => void;
}

let lastFrameTime = 0;
const TARGET_FPS = 60;
const FRAME_DURATION = 1000 / TARGET_FPS; // 16.67ms

export function startGameLoop(config: GameLoopConfig): number {
  const { getGameState, getTaunts, renderer, onInputCommand } = config;

  function loop(timestamp: number) {
    // Delta time calculation
    const deltaTime = timestamp - lastFrameTime;

    // Fixed timestep (skip frame if too soon)
    if (deltaTime < FRAME_DURATION) {
      return requestAnimationFrame(loop);
    }

    lastFrameTime = timestamp;

    // Get current state (from ref, not React state)
    const gameState = getGameState();
    const taunts = getTaunts();

    // Render frame
    renderer.render(gameState, taunts, timestamp);

    // Continue loop
    return requestAnimationFrame(loop);
  }

  lastFrameTime = performance.now();
  return requestAnimationFrame(loop);
}

export function stopGameLoop(loopId: number): void {
  cancelAnimationFrame(loopId);
}
```

## Renderer

File: `/apps/web/src/game/renderer.ts`

```typescript
import { GameState, GAME_CONSTANTS as C } from '@packages/game';

export class CanvasRenderer {
  private ctx: CanvasRenderingContext2D;

  constructor(ctx: CanvasRenderingContext2D) {
    this.ctx = ctx;
  }

  /**
   * Main render function
   */
  render(
    gameState: GameState,
    taunts: Array<{ text: string; timestamp: number; duration: number }>,
    timestamp: number
  ): void {
    this.clear();
    this.drawField();
    this.drawPaddle(gameState.playerPaddle);
    this.drawPaddle(gameState.aiPaddle);
    this.drawBall(gameState.ball);
    this.drawCenterLine();
    this.drawActiveTaunts(taunts, timestamp);
  }

  /**
   * Clear canvas
   */
  private clear(): void {
    this.ctx.fillStyle = '#0a0a0a'; // Near-black background
    this.ctx.fillRect(0, 0, C.FIELD_WIDTH, C.FIELD_HEIGHT);
  }

  /**
   * Draw field background
   */
  private drawField(): void {
    // Optional: draw subtle grid or gradient
  }

  /**
   * Draw center dashed line
   */
  private drawCenterLine(): void {
    this.ctx.strokeStyle = '#333';
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([10, 10]);
    this.ctx.beginPath();
    this.ctx.moveTo(C.FIELD_WIDTH / 2, 0);
    this.ctx.lineTo(C.FIELD_WIDTH / 2, C.FIELD_HEIGHT);
    this.ctx.stroke();
    this.ctx.setLineDash([]); // Reset
  }

  /**
   * Draw paddle
   */
  private drawPaddle(paddle: { position: { x: number; y: number }; width: number; height: number }): void {
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(
      paddle.position.x,
      paddle.position.y,
      paddle.width,
      paddle.height
    );
  }

  /**
   * Draw ball
   */
  private drawBall(ball: { position: { x: number; y: number }; radius: number }): void {
    this.ctx.fillStyle = '#ff4444'; // Red ball
    this.ctx.beginPath();
    this.ctx.arc(ball.position.x, ball.position.y, ball.radius, 0, Math.PI * 2);
    this.ctx.fill();
  }

  /**
   * Draw active taunts (speech bubbles)
   */
  private drawActiveTaunts(
    taunts: Array<{ text: string; timestamp: number; duration: number }>,
    currentTime: number
  ): void {
    const activeTaunts = taunts.filter(
      (taunt) => currentTime - taunt.timestamp < taunt.duration
    );

    activeTaunts.forEach((taunt, index) => {
      const age = currentTime - taunt.timestamp;
      const progress = age / taunt.duration;

      // Fade out in last 20%
      const alpha = progress > 0.8 ? (1 - progress) / 0.2 : 1;

      this.drawSpeechBubble(
        taunt.text,
        C.FIELD_WIDTH * 0.5, // Center X
        50 + index * 60, // Stacked vertically
        alpha
      );
    });
  }

  /**
   * Draw speech bubble
   */
  private drawSpeechBubble(text: string, x: number, y: number, alpha: number): void {
    this.ctx.save();
    this.ctx.globalAlpha = alpha;

    // Measure text
    this.ctx.font = 'bold 20px monospace';
    const metrics = this.ctx.measureText(text);
    const padding = 12;
    const width = metrics.width + padding * 2;
    const height = 40;

    // Draw bubble background
    this.ctx.fillStyle = '#1a1a1a';
    this.ctx.strokeStyle = '#ff4444';
    this.ctx.lineWidth = 2;
    this.roundRect(x - width / 2, y, width, height, 8);
    this.ctx.fill();
    this.ctx.stroke();

    // Draw text
    this.ctx.fillStyle = '#ffffff';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(text, x, y + height / 2);

    this.ctx.restore();
  }

  /**
   * Helper: draw rounded rectangle
   */
  private roundRect(x: number, y: number, width: number, height: number, radius: number): void {
    this.ctx.beginPath();
    this.ctx.moveTo(x + radius, y);
    this.ctx.lineTo(x + width - radius, y);
    this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    this.ctx.lineTo(x + width, y + height - radius);
    this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    this.ctx.lineTo(x + radius, y + height);
    this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    this.ctx.lineTo(x, y + radius);
    this.ctx.quadraticCurveTo(x, y, x + radius, y);
    this.ctx.closePath();
  }
}
```

## Performance Optimization

### Avoid React Re-Renders
- Game state stored in `useRef` (not `useState`)
- Props changes update refs, not state
- Canvas ref never changes (stable)

### Canvas Optimization
- Use `fillRect` for paddles (faster than paths)
- Cache font measurements where possible
- Minimize `save()`/`restore()` calls
- Use integer coordinates (avoid subpixel rendering)

### Frame Rate Stability
- Fixed timestep (16.67ms)
- Skip frames if deltaTime < target (avoid double-render)
- Use `performance.now()` for high-res timestamps

## Edge Cases

| Scenario | Handling |
|----------|----------|
| Canvas ref null | Skip rendering, retry next frame |
| WebSocket lag | Client renders last known state (no jitter) |
| Frame drop | Continue loop, no special handling |
| Resize window | Canvas maintains logical size (CSS scales) |
| Hidden tab | `requestAnimationFrame` pauses automatically |

## Testing

### Manual Tests
- [ ] Game renders at 60 FPS (check DevTools performance)
- [ ] No jank during paddle movement
- [ ] Speech bubbles fade smoothly
- [ ] No memory leaks (long session)
- [ ] Resizing window maintains aspect ratio

### Automated Tests
- [ ] Renderer draws all elements without crash
- [ ] Taunt filtering removes expired taunts
- [ ] Loop cleanup cancels `requestAnimationFrame`

## Latency Target

- **Goal**: Thonk signal → screen update < 50ms (p95)
- **Breakdown**:
  - Thonk → Browser: 10ms
  - Input pipeline: 5ms
  - Render frame: 10ms
  - Browser compositing: 5ms
  - **Total**: 30ms (headroom: 20ms)
