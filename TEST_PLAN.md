# TEST_PLAN.md

## Unit Tests

### `/packages/game`
- [ ] `createInitialState` returns valid state
- [ ] `updatePaddleVelocity` clamps to max speed
- [ ] `updatePaddlePosition` respects bounds
- [ ] `checkPaddleCollision` detects overlaps
- [ ] `checkWallCollision` reverses Y velocity
- [ ] `checkGoalCollision` detects left/right edges
- [ ] `tick` advances game deterministically
- [ ] AI `calculateAICommand` targets ball correctly
- [ ] `adaptDifficulty` scales parameters correctly

### `/apps/server/src/calibration`
- [ ] `computeActivation` aggregates signals
- [ ] `mean` and `std` compute correctly
- [ ] `filterOutliers` removes > 3σ
- [ ] `Calibrator.compute` returns valid params
- [ ] Quality score formula matches spec
- [ ] Phase transitions follow sequence

### `/apps/server/src/stress`
- [ ] `computeVariance` calculates std
- [ ] `computeSwitchRate` counts transitions
- [ ] `computeStress` normalizes to [0, 1]
- [ ] EMA smoothing dampens spikes
- [ ] Event detection fires at thresholds

### `/apps/server/src/input`
- [ ] `detectCommandRaw` respects hysteresis
- [ ] `applyDebounce` requires k-of-n
- [ ] `filterOutliers` clamps to ±3σ

## Integration Tests

### Calibration Flow
- [ ] Client → Server: `start_calibration`
- [ ] Server → Client: `calibration_step` (baseline, left, right)
- [ ] Client sends samples for 3s baseline
- [ ] Server advances to left trial 1
- [ ] Client sends samples for 1.5s
- [ ] Server computes params after all trials
- [ ] Quality score > 1.0 → success

### Game Loop
- [ ] Client → Server: `start_game`
- [ ] Server broadcasts `game_state` at 60 Hz
- [ ] Client sends `input_command` at 60 Hz
- [ ] Ball position updates smoothly
- [ ] Paddle responds to input < 50ms
- [ ] Score increments on goal
- [ ] Game ends at winning score

### Taunt System
- [ ] Event triggers GPT request
- [ ] Rate limit enforces 1 per 3s
- [ ] Fallback on GPT timeout
- [ ] Client displays taunt for duration
- [ ] Taunt fades at 80% duration

## Property-Based Tests

```typescript
import fc from 'fast-check';

test('Paddle Y always in bounds', () => {
  fc.assert(
    fc.property(fc.record({ command: fc.constantFrom('UP', 'DOWN', 'NEUTRAL'), paddle: fc.anything() }), (input) => {
      const paddle = updatePaddlePosition(input.paddle, 0.016);
      return paddle.position.y >= 0 && paddle.position.y <= C.FIELD_HEIGHT - C.PADDLE_HEIGHT;
    })
  );
});

test('Stress metric in [0, 1]', () => {
  fc.assert(
    fc.property(fc.array(fc.record({ activation: fc.float(), command: fc.constantFrom('UP', 'DOWN', 'NEUTRAL'), timestamp: fc.nat() })), (samples) => {
      const calc = new StressCalculator();
      samples.forEach((s) => calc.addSample(s));
      const stress = calc.computeStress();
      return stress >= 0 && stress <= 1;
    })
  );
});
```

## End-to-End Tests

### Manual Playtest
- [ ] Start calibration
- [ ] Complete calibration in < 30s
- [ ] Game runs smoothly at 60 FPS
- [ ] Paddle responds to EEG/keyboard
- [ ] AI difficulty changes with stress
- [ ] Taunts appear contextually
- [ ] Game ends correctly

### Performance Benchmarks
- [ ] Game loop: 60 FPS sustained for 5 minutes
- [ ] Input latency: p95 < 50ms
- [ ] WebSocket latency: p95 < 20ms
- [ ] Stress computation: < 5ms per update
- [ ] Calibration: completes in < 30s

## Test Coverage Goals

| Package | Target |
|---------|--------|
| `/packages/game` | 90% |
| `/packages/shared` | 100% (schemas) |
| `/apps/server` | 70% |
| `/apps/web` | 60% (UI less critical) |

## CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 20
      - run: npm ci
      - run: npm run test
      - run: npm run lint
      - run: npm run typecheck
```

## Test Commands

```json
{
  "scripts": {
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "test:ui": "vitest --ui",
    "lint": "eslint .",
    "typecheck": "tsc --noEmit"
  }
}
```
