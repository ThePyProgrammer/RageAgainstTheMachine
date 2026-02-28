# STRESS_METRIC.md

Lightweight stress proxy derived from EEG input variance. Not medical-grade; purely for gameplay adaptation.

Location: `/apps/server/src/stress/`

## Objectives

- **Lightweight**: No heavy ML, real-time computation
- **Responsive**: Update every 1-2 seconds
- **Proxy metric**: Rough estimate, not clinical accuracy
- **Gameplay use**: Adapt AI difficulty and trigger taunts

## Stress Proxy Algorithm

### Hypothesis
High stress correlates with:
1. **High variance** in input commands (jittery, erratic)
2. **Rapid switching** between commands (indecision)
3. **Sustained high activation** (muscle tension)

### Metric Components

File: `/apps/server/src/stress/stressCalculator.ts`

```typescript
export interface StressMetricConfig {
  windowSizeMs: number; // Rolling window duration
  varianceWeight: number; // 0.0 - 1.0
  switchRateWeight: number; // 0.0 - 1.0
  activationWeight: number; // 0.0 - 1.0
}

export const DEFAULT_CONFIG: StressMetricConfig = {
  windowSizeMs: 2000, // 2 second window
  varianceWeight: 0.4,
  switchRateWeight: 0.4,
  activationWeight: 0.2,
};

export interface StressSample {
  activation: number; // Raw activation score
  command: 'UP' | 'DOWN' | 'NEUTRAL';
  timestamp: number;
}

export class StressCalculator {
  private samples: StressSample[] = [];
  private config: StressMetricConfig;
  private currentStress: number = 0.0; // 0.0 - 1.0

  constructor(config: StressMetricConfig = DEFAULT_CONFIG) {
    this.config = config;
  }

  /**
   * Add sample to rolling window
   */
  addSample(sample: StressSample): void {
    this.samples.push(sample);

    // Remove samples outside window
    const cutoff = Date.now() - this.config.windowSizeMs;
    this.samples = this.samples.filter((s) => s.timestamp >= cutoff);
  }

  /**
   * Compute stress level (0.0 - 1.0)
   */
  computeStress(): number {
    if (this.samples.length < 10) {
      // Insufficient data
      return this.currentStress; // Return last known
    }

    const variance = this.computeVariance();
    const switchRate = this.computeSwitchRate();
    const activationLevel = this.computeActivationLevel();

    // Weighted sum
    const rawStress =
      variance * this.config.varianceWeight +
      switchRate * this.config.switchRateWeight +
      activationLevel * this.config.activationWeight;

    // Normalize to 0.0 - 1.0
    const normalized = Math.max(0, Math.min(1, rawStress));

    // Exponential moving average (smoothing)
    const alpha = 0.3; // Smoothing factor
    this.currentStress = alpha * normalized + (1 - alpha) * this.currentStress;

    return this.currentStress;
  }

  /**
   * Variance of activation scores
   */
  private computeVariance(): number {
    const activations = this.samples.map((s) => s.activation);
    const mean = activations.reduce((a, b) => a + b, 0) / activations.length;
    const variance =
      activations.reduce((acc, v) => acc + (v - mean) ** 2, 0) / activations.length;

    // Normalize: assume typical std ~0.5, stress if std > 1.0
    const std = Math.sqrt(variance);
    return Math.min(1, std / 1.0);
  }

  /**
   * Rate of command switches per second
   */
  private computeSwitchRate(): number {
    let switches = 0;
    for (let i = 1; i < this.samples.length; i++) {
      if (this.samples[i].command !== this.samples[i - 1].command) {
        switches++;
      }
    }

    const durationSec = this.config.windowSizeMs / 1000;
    const switchesPerSec = switches / durationSec;

    // Normalize: assume typical switch rate ~2/s, stress if > 5/s
    return Math.min(1, switchesPerSec / 5);
  }

  /**
   * Average absolute activation level
   */
  private computeActivationLevel(): number {
    const absActivations = this.samples.map((s) => Math.abs(s.activation));
    const mean = absActivations.reduce((a, b) => a + b, 0) / absActivations.length;

    // Normalize: assume typical mean ~0.5, stress if > 1.5
    return Math.min(1, mean / 1.5);
  }

  /**
   * Get current stress level (cached)
   */
  getStress(): number {
    return this.currentStress;
  }
}
```

## Stress Events

Detect stress-related events for taunt triggers.

File: `/apps/server/src/stress/stressEvents.ts`

```typescript
export interface StressEvent {
  type: 'stress_spike' | 'calm_streak';
  timestamp: number;
  stressLevel: number;
}

export class StressEventDetector {
  private lastStress: number = 0.5; // Baseline
  private calmStartTime: number | null = null;

  /**
   * Detect events based on stress change
   */
  detectEvents(currentStress: number): StressEvent[] {
    const events: StressEvent[] = [];
    const timestamp = Date.now();

    // Stress spike: sudden increase > 0.3
    if (currentStress - this.lastStress > 0.3 && currentStress > 0.7) {
      events.push({ type: 'stress_spike', timestamp, stressLevel: currentStress });
    }

    // Calm streak: stress < 0.3 for > 10 seconds
    if (currentStress < 0.3) {
      if (this.calmStartTime === null) {
        this.calmStartTime = timestamp;
      } else if (timestamp - this.calmStartTime > 10000) {
        events.push({ type: 'calm_streak', timestamp, stressLevel: currentStress });
        this.calmStartTime = null; // Reset
      }
    } else {
      this.calmStartTime = null;
    }

    this.lastStress = currentStress;
    return events;
  }
}
```

## Integration with Game Loop

File: `/apps/server/src/game/gameManager.ts`

```typescript
import { StressCalculator, StressSample } from '../stress/stressCalculator';
import { StressEventDetector } from '../stress/stressEvents';

export class GameManager {
  private stressCalculator = new StressCalculator();
  private stressEventDetector = new StressEventDetector();

  /**
   * Called every time an input command is received
   */
  onInputCommand(command: 'UP' | 'DOWN' | 'NEUTRAL', activation: number): void {
    // Add to stress calculator
    this.stressCalculator.addSample({
      activation,
      command,
      timestamp: Date.now(),
    });

    // Compute stress (throttled to once per second)
    // (In practice, use a timer to avoid recomputing every frame)
  }

  /**
   * Called every game tick (or every second for stress)
   */
  updateStress(): { stressLevel: number; events: StressEvent[] } {
    const stressLevel = this.stressCalculator.computeStress();
    const events = this.stressEventDetector.detectEvents(stressLevel);
    return { stressLevel, events };
  }
}
```

## Calibration Impact

Stress metric is **independent** of calibration thresholds. It operates on raw activation variance, not normalized z-scores.

## Visualization (Debug UI)

Display stress level in debug overlay:

```typescript
// Client-side component
<div className="fixed top-4 right-4 bg-black/80 text-white p-4 rounded">
  <div>Stress: {(stressLevel * 100).toFixed(0)}%</div>
  <div className="w-32 h-2 bg-gray-700 rounded mt-2">
    <div
      className="h-full bg-red-500 rounded transition-all"
      style={{ width: `${stressLevel * 100}%` }}
    />
  </div>
</div>
```

## Edge Cases

| Case | Handling |
|------|----------|
| Insufficient data | Return last known stress |
| Flatline (no variance) | Stress → 0.0 (calm) |
| High jitter | Stress → 1.0 (stressed) |
| Network lag | Samples may arrive out of order; sort by timestamp |
| Disconnect | Stress decays to baseline (0.5) |

## Testing

### Unit Tests
- [ ] Variance calculation matches expected values
- [ ] Switch rate counts transitions correctly
- [ ] Activation level normalizes to [0, 1]
- [ ] Exponential smoothing dampens spikes
- [ ] Event detection fires at correct thresholds

### Integration Tests
- [ ] Stress updates every second during gameplay
- [ ] Events trigger taunts correctly
- [ ] High stress → easier AI difficulty

### Property-Based Tests
```typescript
test('Stress always in range [0, 1]', () => {
  forAll(fc.array(fc.record({ activation: fc.float(), command: fc.constantFrom('UP', 'DOWN', 'NEUTRAL'), timestamp: fc.nat() })), (samples) => {
    const calc = new StressCalculator();
    samples.forEach((s) => calc.addSample(s));
    const stress = calc.computeStress();
    return stress >= 0 && stress <= 1;
  });
});
```

## Disclaimer

**Not Medical-Grade**: This metric is a gameplay heuristic, not a diagnostic tool. Users should not rely on it for health monitoring.

Display in UI (footer):
> "Stress metric is for entertainment purposes only and not a medical device."
