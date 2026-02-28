export type GameMode = "keyboard" | "eeg";
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
}

export interface Vec2 {
  x: number;
  y: number;
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

export interface RuntimeState {
  width: number;
  height: number;
  ball: RuntimeBall;
  leftPaddle: RuntimePaddle;
  rightPaddle: RuntimePaddle;
  playerScore: number;
  aiScore: number;
}

export type GameInputState = {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  pointerX?: number;
  pointerY?: number;
};
