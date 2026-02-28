GAME_ENGINE.md

Pure TypeScript game engine with deterministic physics. No I/O, no side effects.

Location: `/packages/game/src/`

## Responsibilities

- Ball physics (position, velocity, acceleration)
- Paddle movement (velocity, constraints)
- Collision detection (ball-paddle, ball-wall)
- Score updates
- Game state transitions
- Constants (field dimensions, velocities, etc.)

## Constants

File: `/packages/game/src/constants.ts`

```typescript
export const GAME_CONSTANTS = {
// Field dimensions (canvas logical coordinates)
FIELD_WIDTH: 800,
FIELD_HEIGHT: 600,

// Paddle
PADDLE_WIDTH: 12,
PADDLE_HEIGHT: 80,
PADDLE_OFFSET_X: 20, // Distance from edge
PADDLE_MAX_SPEED: 400, // pixels/second
PADDLE_ACCELERATION: 1200, // pixels/second^2
PADDLE_FRICTION: 0.85, // velocity decay per frame when no input

// Ball
BALL_RADIUS: 8,
BALL_INITIAL_SPEED: 300, // pixels/second
BALL_SPEED_INCREMENT: 20, // added per paddle hit
BALL_MAX_SPEED: 600,

// Physics
FRAME_RATE: 60, // Hz
DT: 1 / 60, // seconds per frame

// Collision
PADDLE_RESTITUTION: 1.1, // Bounce energy multiplier
WALL_RESTITUTION: 1.0, // Perfect elastic bounce

// Scoring
WINNING_SCORE: 11,
} as const;

Type Definitions

File: /packages/game/src/types.ts

export interface Vec2 {
x: number;
y: number;
}

export interface Paddle {
position: Vec2; // Top-left corner
velocity: Vec2;
width: number;
height: number;
side: 'left' | 'right';
}

export interface Ball {
position: Vec2; // Center
velocity: Vec2;
radius: number;
}

export interface Score {
player: number;
ai: number;
}

export interface GameState {
ball: Ball;
playerPaddle: Paddle;
aiPaddle: Paddle;
score: Score;
status: 'waiting' | 'playing' | 'paused' | 'game_over';
winner: 'player' | 'ai' | null;
timestamp: number; // ms since epoch
}

export type PaddleCommand = 'UP' | 'DOWN' | 'NEUTRAL';

export interface CollisionResult {
occurred: boolean;
type: 'paddle' | 'wall_top' | 'wall_bottom' | 'goal_left' | 'goal_right' | null;
newVelocity?: Vec2;
}

Core Functions

File: /packages/game/src/engine.ts

import { GameState, Paddle, Ball, PaddleCommand, CollisionResult, Vec2 } from './types';
import { GAME_CONSTANTS as C } from './constants';

/**
 * Create initial game state
*/
export function createInitialState(): GameState {
return {
  ball: {
    position: { x: C.FIELD_WIDTH / 2, y: C.FIELD_HEIGHT / 2 },
    velocity: {
      x: C.BALL_INITIAL_SPEED * (Math.random() > 0.5 ? 1 : -1),
      y: C.BALL_INITIAL_SPEED * (Math.random() * 0.6 - 0.3), // Angle variance
    },
    radius: C.BALL_RADIUS,
  },
  playerPaddle: {
    position: { x: C.PADDLE_OFFSET_X, y: C.FIELD_HEIGHT / 2 - C.PADDLE_HEIGHT / 2 },
    velocity: { x: 0, y: 0 },
    width: C.PADDLE_WIDTH,
    height: C.PADDLE_HEIGHT,
    side: 'left',
  },
  aiPaddle: {
    position: {
      x: C.FIELD_WIDTH - C.PADDLE_OFFSET_X - C.PADDLE_WIDTH,
      y: C.FIELD_HEIGHT / 2 - C.PADDLE_HEIGHT / 2,
    },
    velocity: { x: 0, y: 0 },
    width: C.PADDLE_WIDTH,
    height: C.PADDLE_HEIGHT,
    side: 'right',
  },
  score: { player: 0, ai: 0 },
  status: 'waiting',
  winner: null,
  timestamp: Date.now(),
};
}

/**
 * Update paddle velocity based on command
*/
export function updatePaddleVelocity(
paddle: Paddle,
command: PaddleCommand,
dt: number
): Paddle {
let newVy = paddle.velocity.y;

if (command === 'UP') {
  newVy -= C.PADDLE_ACCELERATION * dt;
} else if (command === 'DOWN') {
  newVy += C.PADDLE_ACCELERATION * dt;
} else {
  // Apply friction
  newVy *= C.PADDLE_FRICTION;
}

// Clamp to max speed
newVy = Math.max(-C.PADDLE_MAX_SPEED, Math.min(C.PADDLE_MAX_SPEED, newVy));

return {
  ...paddle,
  velocity: { x: 0, y: newVy },
};
}

/**
 * Update paddle position (with bounds checking)
*/
export function updatePaddlePosition(paddle: Paddle, dt: number): Paddle {
let newY = paddle.position.y + paddle.velocity.y * dt;

// Clamp to field bounds
newY = Math.max(0, Math.min(C.FIELD_HEIGHT - paddle.height, newY));

return {
  ...paddle,
  position: { ...paddle.position, y: newY },
};
}

/**
 * Update ball position
*/
export function updateBallPosition(ball: Ball, dt: number): Ball {
return {
  ...ball,
  position: {
    x: ball.position.x + ball.velocity.x * dt,
    y: ball.position.y + ball.velocity.y * dt,
  },
};
}

/**
 * Check ball-paddle collision
*/
export function checkPaddleCollision(ball: Ball, paddle: Paddle): CollisionResult {
const ballLeft = ball.position.x - ball.radius;
const ballRight = ball.position.x + ball.radius;
const ballTop = ball.position.y - ball.radius;
const ballBottom = ball.position.y + ball.radius;

const paddleLeft = paddle.position.x;
const paddleRight = paddle.position.x + paddle.width;
const paddleTop = paddle.position.y;
const paddleBottom = paddle.position.y + paddle.height;

const overlapsX = ballRight >= paddleLeft && ballLeft <= paddleRight;
const overlapsY = ballBottom >= paddleTop && ballTop <= paddleBottom;

if (overlapsX && overlapsY) {
  // Calculate new velocity (reverse X, add spin based on paddle velocity)
  const newVx = -ball.velocity.x * C.PADDLE_RESTITUTION;
  const spin = paddle.velocity.y * 0.3; // Transfer 30% of paddle velocity
  const newVy = ball.velocity.y + spin;

  // Clamp ball speed
  const speed = Math.sqrt(newVx * newVx + newVy * newVy);
  const clampedSpeed = Math.min(speed, C.BALL_MAX_SPEED);
  const scale = clampedSpeed / speed;

  return {
    occurred: true,
    type: 'paddle',
    newVelocity: { x: newVx * scale, y: newVy * scale },
  };
}

return { occurred: false, type: null };
}

/**
 * Check ball-wall collision (top/bottom)
*/
export function checkWallCollision(ball: Ball): CollisionResult {
const ballTop = ball.position.y - ball.radius;
const ballBottom = ball.position.y + ball.radius;

if (ballTop <= 0 || ballBottom >= C.FIELD_HEIGHT) {
  return {
    occurred: true,
    type: ballTop <= 0 ? 'wall_top' : 'wall_bottom',
    newVelocity: { x: ball.velocity.x, y: -ball.velocity.y * C.WALL_RESTITUTION },
  };
}

return { occurred: false, type: null };
}

/**
 * Check ball-goal collision (left/right edges)
*/
export function checkGoalCollision(ball: Ball): CollisionResult {
const ballLeft = ball.position.x - ball.radius;
const ballRight = ball.position.x + ball.radius;

if (ballLeft <= 0) {
  return { occurred: true, type: 'goal_left' }; // AI scores
}
if (ballRight >= C.FIELD_WIDTH) {
  return { occurred: true, type: 'goal_right' }; // Player scores
}

return { occurred: false, type: null };
}

/**
 * Reset ball to center (after goal)
*/
export function resetBall(): Ball {
const angle = (Math.random() * Math.PI / 3) - (Math.PI / 6); // ±30 degrees
const direction = Math.random() > 0.5 ? 1 : -1;

return {
  position: { x: C.FIELD_WIDTH / 2, y: C.FIELD_HEIGHT / 2 },
  velocity: {
    x: C.BALL_INITIAL_SPEED * Math.cos(angle) * direction,
    y: C.BALL_INITIAL_SPEED * Math.sin(angle),
  },
  radius: C.BALL_RADIUS,
};
}

/**
 * Main game tick (deterministic)
*/
export function tick(
state: GameState,
playerCommand: PaddleCommand,
aiCommand: PaddleCommand,
dt: number = C.DT
): GameState {
if (state.status !== 'playing') {
  return state;
}

// Update paddles
let playerPaddle = updatePaddleVelocity(state.playerPaddle, playerCommand, dt);
playerPaddle = updatePaddlePosition(playerPaddle, dt);

let aiPaddle = updatePaddleVelocity(state.aiPaddle, aiCommand, dt);
aiPaddle = updatePaddlePosition(aiPaddle, dt);

// Update ball
let ball = updateBallPosition(state.ball, dt);

// Check collisions
const paddleCollision = checkPaddleCollision(ball, playerPaddle) ||
                        checkPaddleCollision(ball, aiPaddle);
const wallCollision = checkWallCollision(ball);
const goalCollision = checkGoalCollision(ball);

if (paddleCollision.occurred && paddleCollision.newVelocity) {
  ball = { ...ball, velocity: paddleCollision.newVelocity };
}

if (wallCollision.occurred && wallCollision.newVelocity) {
  ball = { ...ball, velocity: wallCollision.newVelocity };
}

let score = state.score;
let winner = state.winner;
let status = state.status;

if (goalCollision.occurred) {
  if (goalCollision.type === 'goal_left') {
    score = { ...score, ai: score.ai + 1 };
  } else if (goalCollision.type === 'goal_right') {
    score = { ...score, player: score.player + 1 };
  }

  // Check win condition
  if (score.player >= C.WINNING_SCORE) {
    winner = 'player';
    status = 'game_over';
  } else if (score.ai >= C.WINNING_SCORE) {
    winner = 'ai';
    status = 'game_over';
  } else {
    ball = resetBall();
  }
}

return {
  ball,
  playerPaddle,
  aiPaddle,
  score,
  status,
  winner,
  timestamp: Date.now(),
};
}

AI Opponent Logic

File: /packages/game/src/ai.ts

import { Paddle, Ball, PaddleCommand } from './types';
import { GAME_CONSTANTS as C } from './constants';

export interface AIDifficulty {
reactionDelay: number; // seconds (simulated delay)
errorMargin: number; // pixels (random offset)
maxSpeed: number; // fraction of PADDLE_MAX_SPEED (0.0 - 1.0)
predictionDepth: number; // how many frames ahead to predict ball
}

/**
 * Calculate AI paddle command based on difficulty
*/
export function calculateAICommand(
aiPaddle: Paddle,
ball: Ball,
difficulty: AIDifficulty,
dt: number
): PaddleCommand {
// Predict ball position
const predictedY = predictBallY(ball, difficulty.predictionDepth, dt);

// Add error margin (random jitter)
const targetY = predictedY + (Math.random() - 0.5) * difficulty.errorMargin;

// Calculate paddle center
const paddleCenterY = aiPaddle.position.y + aiPaddle.height / 2;

// Dead zone (don't move if close enough)
const deadZone = 5;
if (Math.abs(targetY - paddleCenterY) < deadZone) {
  return 'NEUTRAL';
}

// Move towards target
if (targetY < paddleCenterY) {
  return 'UP';
} else {
  return 'DOWN';
}
}

/**
 * Predict ball Y position after N frames (simple linear prediction)
*/
function predictBallY(ball: Ball, frames: number, dt: number): number {
let y = ball.position.y;
let vy = ball.velocity.y;

for (let i = 0; i < frames; i++) {
  y += vy * dt;

  // Bounce off top/bottom
  if (y - ball.radius <= 0 || y + ball.radius >= C.FIELD_HEIGHT) {
    vy = -vy;
  }
}

return y;
}

/**
 * Adjust difficulty based on stress level
*/
export function adaptDifficulty(
baselineDifficulty: AIDifficulty,
stressLevel: number // 0.0 - 1.0
): AIDifficulty {
// High stress → easier AI (more error, slower)
// Low stress → harder AI (less error, faster)
const stressFactor = 1 - stressLevel; // Invert (0 = stressed, 1 = calm)

return {
  reactionDelay: baselineDifficulty.reactionDelay * (1 + stressLevel * 0.5),
  errorMargin: baselineDifficulty.errorMargin * (1 + stressLevel * 2),
  maxSpeed: baselineDifficulty.maxSpeed * (0.5 + stressFactor * 0.5), // 50%-100%
  predictionDepth: Math.floor(baselineDifficulty.predictionDepth * stressFactor),
};
}

Testing Checklist

- Ball bounces correctly off paddles (velocity reverses)
- Ball bounces correctly off walls (Y velocity reverses)
- Paddle cannot move outside field bounds
- Paddle velocity clamps to max speed
- Goal detection triggers score increment
- Win condition triggers game_over status
- Ball resets to center after goal
- AI prediction handles wall bounces
- Difficulty adaptation clamps to valid ranges
- All functions are pure (no side effects)
- Deterministic: same inputs → same outputs

Invariants (Property-Based Tests)

// Example invariants to test
describe('Game Engine Invariants', () => {
test('Paddle Y always within [0, FIELD_HEIGHT - PADDLE_HEIGHT]', () => {
  // Property: ∀ state, paddle.position.y ∈ [0, FIELD_HEIGHT - PADDLE_HEIGHT]
});

test('Ball speed never exceeds BALL_MAX_SPEED', () => {
  // Property: ∀ state, √(vx² + vy²) ≤ BALL_MAX_SPEED
});

test('Score never decreases', () => {
  // Property: ∀ tick, newScore.player ≥ oldScore.player
});

test('Game ends when score reaches WINNING_SCORE', () => {
  // Property: score ≥ WINNING_SCORE ⟹ status = 'game_over'
});
});