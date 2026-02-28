export type GameScreen = "menu" | "calibration" | "game" | "paused";

export interface DebugStats {
  fps: number;
  latencyMs: number;
  thonkConnected: boolean;
  calibrationQuality?: number;
  ballX?: number;
  ballY?: number;
  ballVX?: number;
  ballVY?: number;
  deltaMs?: number;
  collisionNormals?: Array<{ x: number; y: number }>;
  positionClampedPerSecond?: number;
  collisionResolvedPerSecond?: number;
  bricksRemaining?: number;
  speedMultiplier?: number;
}

export interface RuntimePaddle {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RuntimeBall {
  x: number;
  y: number;
  radius: number;
}

export interface RuntimeBrick {
  id: number;
  x: number;
  y: number;
  width: number;
  height: number;
  active: boolean;
}

export interface RuntimeState {
  width: number;
  height: number;
  paddle: RuntimePaddle;
  ball: RuntimeBall;
  bricks: RuntimeBrick[];
  bricksRemaining: number;
  score: number;
  lives: number;
  level: number;
  gameOver: boolean;
}

export type GameInputState = {
  left: boolean;
  right: boolean;
};
