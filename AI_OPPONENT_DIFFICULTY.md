# AI_OPPONENT_DIFFICULTY.md

Adaptive AI difficulty based on player stress level.

Location: `/packages/game/src/ai.ts` (shared) + `/apps/server/src/game/aiManager.ts`

## Difficulty Levels

Defined in `/packages/game/src/ai.ts`:

```typescript
export interface AIDifficulty {
  reactionDelay: number; // seconds (simulated input lag)
  errorMargin: number; // pixels (random target offset)
  maxSpeed: number; // fraction of PADDLE_MAX_SPEED (0.0 - 1.0)
  predictionDepth: number; // frames ahead to predict ball
}

export const DIFFICULTY_PRESETS = {
  EASY: {
    reactionDelay: 0.3,
    errorMargin: 60,
    maxSpeed: 0.6,
    predictionDepth: 5,
  },
  MEDIUM: {
    reactionDelay: 0.15,
    errorMargin: 30,
    maxSpeed: 0.8,
    predictionDepth: 10,
  },
  HARD: {
    reactionDelay: 0.05,
    errorMargin: 10,
    maxSpeed: 1.0,
    predictionDepth: 20,
  },
  IMPOSSIBLE: {
    reactionDelay: 0.0,
    errorMargin: 0,
    maxSpeed: 1.0,
    predictionDepth: 60,
  },
} as const;
```

## Stress-Based Adaptation

File: `/apps/server/src/game/aiManager.ts`

```typescript
import { AIDifficulty, DIFFICULTY_PRESETS, adaptDifficulty } from '@packages/game';

export class AIManager {
  private baselineDifficulty: AIDifficulty = DIFFICULTY_PRESETS.MEDIUM;
  private currentDifficulty: AIDifficulty = DIFFICULTY_PRESETS.MEDIUM;

  /**
   * Update difficulty based on stress
   * Called every second or when stress changes significantly
   */
  updateDifficulty(stressLevel: number): void {
    this.currentDifficulty = adaptDifficulty(this.baselineDifficulty, stressLevel);
  }

  /**
   * Get current difficulty
   */
  getDifficulty(): AIDifficulty {
    return this.currentDifficulty;
  }

  /**
   * Set baseline difficulty (user choice or adaptive baseline)
   */
  setBaseline(difficulty: AIDifficulty): void {
    this.baselineDifficulty = difficulty;
  }
}
```

## Adaptation Function

Defined in `/packages/game/src/ai.ts`:

```typescript
/**
 * Adapt difficulty based on stress level
 * High stress (1.0) → easier AI
 * Low stress (0.0) → harder AI
 */
export function adaptDifficulty(
  baseline: AIDifficulty,
  stressLevel: number // 0.0 - 1.0
): AIDifficulty {
  // Invert stress: 0.0 = stressed, 1.0 = calm
  const calmFactor = 1 - stressLevel;

  return {
    // Slower reaction when stressed
    reactionDelay: baseline.reactionDelay * (1 + stressLevel * 0.8),

    // More errors when stressed
    errorMargin: baseline.errorMargin * (1 + stressLevel * 1.5),

    // Slower paddle when stressed (50% - 100% of baseline)
    maxSpeed: baseline.maxSpeed * (0.5 + calmFactor * 0.5),

    // Shallower prediction when stressed
    predictionDepth: Math.floor(baseline.predictionDepth * (0.5 + calmFactor * 0.5)),
  };
}
```

## Adaptation Curve

| Stress Level | AI Behavior |
|--------------|-------------|
| 0.0 (Calm) | Baseline difficulty (no change) |
| 0.3 | Slightly slower, 20% more errors |
| 0.5 | Noticeably easier, 50% more errors |
| 0.7 | Significantly easier, 75% slower |
| 1.0 (Stressed) | Very easy, 50% speed, 150% more errors |

## Smoothing

Apply exponential moving average to prevent jarring difficulty changes:

```typescript
export class AIManager {
  private targetDifficulty: AIDifficulty = DIFFICULTY_PRESETS.MEDIUM;
  private currentDifficulty: AIDifficulty = DIFFICULTY_PRESETS.MEDIUM;

  updateDifficulty(stressLevel: number): void {
    this.targetDifficulty = adaptDifficulty(this.baselineDifficulty, stressLevel);

    // Smooth transition (alpha = 0.1)
    this.currentDifficulty = {
      reactionDelay: lerp(this.currentDifficulty.reactionDelay, this.targetDifficulty.reactionDelay, 0.1),
      errorMargin: lerp(this.currentDifficulty.errorMargin, this.targetDifficulty.errorMargin, 0.1),
      maxSpeed: lerp(this.currentDifficulty.maxSpeed, this.targetDifficulty.maxSpeed, 0.1),
      predictionDepth: Math.floor(lerp(this.currentDifficulty.predictionDepth, this.targetDifficulty.predictionDepth, 0.1)),
    };
  }
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
```

## AI Command Generation

File: `/apps/server/src/game/gameLoop.ts`

```typescript
import { calculateAICommand } from '@packages/game';
import { AIManager } from './aiManager';

export class ServerGameLoop {
  private aiManager = new AIManager();
  private aiReactionBuffer: Array<{ command: PaddleCommand; timestamp: number }> = [];

  tick(gameState: GameState, stressLevel: number): GameState {
    // Update AI difficulty based on stress
    this.aiManager.updateDifficulty(stressLevel);
    const difficulty = this.aiManager.getDifficulty();

    // Calculate AI command (with prediction)
    const rawCommand = calculateAICommand(
      gameState.aiPaddle,
      gameState.ball,
      difficulty,
      GAME_CONSTANTS.DT
    );

    // Apply reaction delay
    this.aiReactionBuffer.push({ command: rawCommand, timestamp: Date.now() });
    const delayMs = difficulty.reactionDelay * 1000;
    const delayedCommand = this.aiReactionBuffer.find(
      (entry) => Date.now() - entry.timestamp >= delayMs
    )?.command ?? 'NEUTRAL';

    // Tick game engine
    const playerCommand = /* from input pipeline */;
    return tick(gameState, playerCommand, delayedCommand);
  }
}
```

## User Control (Optional)

Allow user to set baseline difficulty:

```typescript
// Client UI
<select onChange={(e) => setBaseline(e.target.value)}>
  <option value="EASY">Easy</option>
  <option value="MEDIUM">Medium</option>
  <option value="HARD">Hard</option>
  <option value="ADAPTIVE">Adaptive (stress-based)</option>
</select>

// Server
socket.on('set_difficulty', (level: keyof typeof DIFFICULTY_PRESETS) => {
  aiManager.setBaseline(DIFFICULTY_PRESETS[level]);
});
```

## Edge Cases

| Case | Handling |
|------|----------|
| Stress data missing | Use baseline difficulty (no adaptation) |
| Stress spikes rapidly | Smoothing prevents jarring changes |
| User too calm | Difficulty increases smoothly to baseline |
| User too stressed | Difficulty floors at 50% speed minimum |

## Testing

### Unit Tests
- [ ] `adaptDifficulty` clamps to valid ranges
- [ ] Smoothing converges to target over time
- [ ] Reaction delay buffer works correctly

### Integration Tests
- [ ] AI gets easier when stress = 1.0
- [ ] AI returns to baseline when stress = 0.0
- [ ] Smooth transitions over 5-10 seconds

### Playtesting
- [ ] Adaptive difficulty feels natural
- [ ] No "rubber-banding" perception
- [ ] AI remains beatable at all stress levels
