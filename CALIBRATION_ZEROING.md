# CALIBRATION_ZEROING.md

Ultra-fast per-user calibration to tune detection thresholds in 2-3 trials per direction.

Location: `/apps/server/src/calibration/`

## Objectives

1. **Speed**: Converge in < 30 seconds total
2. **Robustness**: Handle noise, artifacts, short attention lapses
3. **Per-user**: Adapt to individual EEG baselines
4. **Quality score**: Detect unusable signals, prompt re-calibration or fallback

## Calibration Protocol

### Phase 1: Baseline Capture (2-4 seconds)
- **Instruction**: "Relax and think neutral thoughts."
- **Duration**: 3 seconds
- **Goal**: Establish per-user mean (μ₀) and std (σ₀) of activation score

### Phase 2: Directional Capture (LEFT → RIGHT)
Each direction gets 2-3 trials:
- **Trial 1**: "Think UP" (1.5s)
- **Trial 2**: "Think UP" (1.5s)
- **Trial 3**: (optional, if variance high) "Think UP" (1.5s)

Repeat for DOWN.

### Total Time
- Baseline: 3s
- UP: 2 trials × 1.5s = 3s
- DOWN: 2 trials × 1.5s = 3s
- Transitions + instructions: ~5s
- **Total**: ~14 seconds (up to 20s with optional trial 3)

## Algorithm

File: `/apps/server/src/calibration/calibrator.ts`

```typescript
import { ThonkSignal } from '@packages/shared';

export interface CalibrationParams {
  baselineMean: number;
  baselineStd: number;
  thresholdLeft: number; // z-score threshold for UP
  thresholdRight: number; // z-score threshold for DOWN
  thresholdEnter: number; // Entry threshold (higher)
  thresholdExit: number; // Exit threshold (lower, for hysteresis)
  quality: number; // Separation score
}

export interface CalibrationStep {
  phase: 'baseline' | 'left' | 'right';
  trial: number; // 1-indexed
  samples: number[]; // Activation scores
}

/**
 * Compute activation from signal (same as client-side)
 */
function computeActivation(signal: ThonkSignal): number {
  const scoreUp = signal.up + signal.left;
  const scoreDown = signal.down + signal.right;
  return scoreUp - scoreDown;
}

/**
 * Statistical helpers
 */
function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function std(values: number[], meanVal?: number): number {
  const m = meanVal ?? mean(values);
  const variance = values.reduce((acc, v) => acc + (v - m) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Remove outliers beyond ±3σ
 */
function filterOutliers(values: number[]): number[] {
  const m = mean(values);
  const s = std(values, m);
  return values.filter((v) => Math.abs(v - m) <= 3 * s);
}

/**
 * Main calibration class
 */
export class Calibrator {
  private steps: CalibrationStep[] = [];

  /**
   * Add baseline samples
   */
  addBaselineSamples(signals: ThonkSignal[]): void {
    const activations = signals.map(computeActivation);
    this.steps.push({
      phase: 'baseline',
      trial: 1,
      samples: activations,
    });
  }

  /**
   * Add directional samples (UP or DOWN)
   */
  addDirectionalSamples(direction: 'left' | 'right', trial: number, signals: ThonkSignal[]): void {
    const activations = signals.map(computeActivation);
    this.steps.push({
      phase: direction,
      trial,
      samples: activations,
    });
  }

  /**
   * Compute calibration parameters
   */
  compute(): CalibrationParams {
    // Extract baseline
    const baselineStep = this.steps.find((s) => s.phase === 'baseline');
    if (!baselineStep) throw new Error('Missing baseline step');

    const baselineSamples = filterOutliers(baselineStep.samples);
    const baselineMean = mean(baselineSamples);
    const baselineStd = std(baselineSamples, baselineMean);

    // Extract UP trials
    const upSteps = this.steps.filter((s) => s.phase === 'left');
    const upSamples = filterOutliers(upSteps.flatMap((s) => s.samples));
    const upMean = mean(upSamples);

    // Extract DOWN trials
    const downSteps = this.steps.filter((s) => s.phase === 'right');
    const downSamples = filterOutliers(downSteps.flatMap((s) => s.samples));
    const downMean = mean(downSamples);

    // Compute z-scores for direction means
    const zUp = (upMean - baselineMean) / baselineStd;
    const zDown = (downMean - baselineMean) / baselineStd;

    // Set thresholds midway between baseline and direction means
    const thresholdLeft = (0 + zUp) / 2; // Midpoint
    const thresholdRight = (0 + zDown) / 2;

    // Hysteresis thresholds
    const margin = 0.3; // Safety margin
    const thresholdEnter = Math.abs(thresholdLeft) + margin;
    const thresholdExit = Math.abs(thresholdLeft) - margin;

    // Quality score: separation between UP and DOWN
    const pooledStd = Math.sqrt(
      (std(upSamples) ** 2 + std(downSamples) ** 2) / 2
    );
    const quality = Math.abs(upMean - downMean) / pooledStd;

    return {
      baselineMean,
      baselineStd,
      thresholdLeft,
      thresholdRight,
      thresholdEnter,
      thresholdExit,
      quality,
    };
  }

  /**
   * Check if calibration is complete
   */
  isComplete(): boolean {
    const hasBaseline = this.steps.some((s) => s.phase === 'baseline');
    const upTrials = this.steps.filter((s) => s.phase === 'left').length;
    const downTrials = this.steps.filter((s) => s.phase === 'right').length;
    return hasBaseline && upTrials >= 2 && downTrials >= 2;
  }

  /**
   * Reset calibration
   */
  reset(): void {
    this.steps = [];
  }
}
```

## Calibration Service

File: `/apps/server/src/calibration/calibrationService.ts`

```typescript
import { Calibrator, CalibrationParams } from './calibrator';
import { ThonkSignal } from '@packages/shared';

export interface CalibrationSession {
  sessionId: string;
  calibrator: Calibrator;
  currentPhase: 'baseline' | 'left' | 'right';
  currentTrial: number;
  startTime: number;
  samples: ThonkSignal[];
}

const BASELINE_DURATION_MS = 3000;
const TRIAL_DURATION_MS = 1500;
const SAMPLE_RATE_HZ = 30; // Expected Thonk sample rate

export class CalibrationService {
  private sessions = new Map<string, CalibrationSession>();

  /**
   * Start calibration for a session
   */
  startCalibration(sessionId: string): void {
    const session: CalibrationSession = {
      sessionId,
      calibrator: new Calibrator(),
      currentPhase: 'baseline',
      currentTrial: 1,
      startTime: Date.now(),
      samples: [],
    };
    this.sessions.set(sessionId, session);
  }

  /**
   * Add sample to current phase
   */
  addSample(sessionId: string, signal: ThonkSignal): void {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error('Session not found');

    session.samples.push(signal);
  }

  /**
   * Check if current phase is complete
   */
  isPhaseComplete(sessionId: string): boolean {
    const session = this.sessions.get(sessionId);
    if (!session) return false;

    const elapsed = Date.now() - session.startTime;
    const duration =
      session.currentPhase === 'baseline' ? BASELINE_DURATION_MS : TRIAL_DURATION_MS;

    return elapsed >= duration;
  }

  /**
   * Advance to next phase
   */
  advancePhase(sessionId: string): { phase: string; trial: number; complete: boolean } {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error('Session not found');

    // Commit current phase samples
    if (session.currentPhase === 'baseline') {
      session.calibrator.addBaselineSamples(session.samples);
    } else {
      session.calibrator.addDirectionalSamples(
        session.currentPhase,
        session.currentTrial,
        session.samples
      );
    }

    // Reset for next phase
    session.samples = [];
    session.startTime = Date.now();

    // Determine next phase
    if (session.currentPhase === 'baseline') {
      session.currentPhase = 'left';
      session.currentTrial = 1;
    } else if (session.currentPhase === 'left') {
      if (session.currentTrial < 2) {
        session.currentTrial++;
      } else {
        session.currentPhase = 'right';
        session.currentTrial = 1;
      }
    } else if (session.currentPhase === 'right') {
      if (session.currentTrial < 2) {
        session.currentTrial++;
      } else {
        // Calibration complete
        return { phase: 'complete', trial: 0, complete: true };
      }
    }

    return {
      phase: session.currentPhase,
      trial: session.currentTrial,
      complete: false,
    };
  }

  /**
   * Finalize calibration and return parameters
   */
  finalize(sessionId: string): { params: CalibrationParams; status: 'success' | 'retry' | 'failed' } {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error('Session not found');

    const params = session.calibrator.compute();

    // Quality check
    let status: 'success' | 'retry' | 'failed';
    if (params.quality >= 1.5) {
      status = 'success';
    } else if (params.quality >= 1.0) {
      status = 'retry'; // Marginal, suggest retry
    } else {
      status = 'failed'; // Unusable
    }

    return { params, status };
  }

  /**
   * Get current session state
   */
  getSession(sessionId: string): CalibrationSession | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * Clean up session
   */
  endCalibration(sessionId: string): void {
    this.sessions.delete(sessionId);
  }
}
```

## Calibration Flow (Server-Side)

File: `/apps/server/src/handlers/calibrationHandler.ts`

```typescript
import { Socket } from 'socket.io';
import { CalibrationService } from '../calibration/calibrationService';
import { ServerEvents, ClientEvents } from '@packages/shared';

export function setupCalibrationHandlers(
  socket: Socket,
  calibrationService: CalibrationService,
  sessionId: string
) {
  socket.on(ClientEvents.START_CALIBRATION, () => {
    calibrationService.startCalibration(sessionId);

    // Send first step
    socket.emit(ServerEvents.CALIBRATION_STEP, {
      step: 'baseline',
      trial: 1,
      instruction: 'Relax and think neutral thoughts.',
      durationMs: 3000,
    });

    // Monitor for phase completion
    const interval = setInterval(() => {
      if (calibrationService.isPhaseComplete(sessionId)) {
        const next = calibrationService.advancePhase(sessionId);

        if (next.complete) {
          clearInterval(interval);

          // Finalize
          const { params, status } = calibrationService.finalize(sessionId);

          socket.emit(ServerEvents.CALIBRATION_COMPLETE, {
            params,
            quality: params.quality,
            status,
            message:
              status === 'success'
                ? 'Calibration successful!'
                : status === 'retry'
                ? 'Signal quality marginal. Retry recommended.'
                : 'Calibration failed. Try again or use keyboard fallback.',
          });

          calibrationService.endCalibration(sessionId);
        } else {
          // Send next step
          const instruction =
            next.phase === 'left'
              ? 'Think about moving the paddle UP.'
              : 'Think about moving the paddle DOWN.';

          socket.emit(ServerEvents.CALIBRATION_STEP, {
            step: next.phase,
            trial: next.trial,
            instruction,
            durationMs: 1500,
          });
        }
      }
    }, 100); // Check every 100ms
  });

  socket.on(ClientEvents.CALIBRATION_SAMPLE, (data) => {
    calibrationService.addSample(sessionId, data.signals);
  });
}
```

## Quality Score Interpretation

| Quality Score | Interpretation | Action |
|---------------|----------------|--------|
| ≥ 2.0 | Excellent separation | Proceed |
| 1.5 - 2.0 | Good separation | Proceed |
| 1.0 - 1.5 | Marginal | Suggest retry |
| < 1.0 | Poor separation | Fail, offer keyboard fallback |

**Formula**: `quality = |μ_up - μ_down| / √((σ_up² + σ_down²) / 2)`

## Artifact Handling

| Artifact | Detection | Mitigation |
|----------|-----------|------------|
| Extreme spike | > 3σ from mean | Remove outliers before mean/std |
| Flatline | All signals = 0 for >0.5s | Warn user, extend trial |
| Missing packets | Gap > 100ms | Interpolate or extend trial |
| Drift | Baseline shifts mid-calibration | Re-run baseline if detected |

## Debounce & Hysteresis Details

### Hysteresis
- **Enter threshold**: Higher (e.g., 1.2σ) to avoid false positives
- **Exit threshold**: Lower (e.g., 0.5σ) to avoid jitter

Example:
```
State: NEUTRAL
z-score rises above 1.2 → Enter UP
State: UP
z-score drops to 0.8 → Still UP (above exit threshold 0.5)
z-score drops to 0.4 → Exit UP → NEUTRAL
```

### Debounce (k-of-n)
- **k = 3**: Require 3 frames
- **n = 5**: Out of last 5 frames
- Prevents single-frame noise from changing command

## Persistence

Store calibration params per session:

```typescript
// In-memory (session-scoped)
const sessionParams = new Map<string, CalibrationParams>();

// Optional: persist to localStorage (client-side)
localStorage.setItem(`calibration_${userId}`, JSON.stringify(params));

// Optional: persist to database (server-side)
// await db.calibrations.upsert({ userId, params });
```

## Edge Cases

| Case | Handling |
|------|----------|
| User blinks during baseline | Outlier filtering removes spikes |
| User moves head | Extend trial if variance > threshold |
| User doesn't understand instruction | Show visual cue (arrow, color) |
| Calibration timeout (>60s) | Abort, offer keyboard fallback |
| Network lag during calibration | Buffer samples, extend trial |
| User disconnects mid-calibration | Discard session, start fresh |

## Testing

### Unit Tests
- [ ] Outlier filtering removes > 3σ values
- [ ] Mean/std calculations match expected values
- [ ] Threshold midpoint calculation correct
- [ ] Quality score formula validated
- [ ] Phase transitions in correct order

### Integration Tests
- [ ] Full calibration flow completes in < 30s
- [ ] Quality score computed correctly
- [ ] Failed calibration triggers retry prompt
- [ ] Params stored and retrieved correctly

### Property-Based Tests
```typescript
test('Quality score always non-negative', () => {
  forAll(
    { upMean: fc.float(), downMean: fc.float(), upStd: fc.float(0.1, 5), downStd: fc.float(0.1, 5) },
    ({ upMean, downMean, upStd, downStd }) => {
      const quality = Math.abs(upMean - downMean) / Math.sqrt((upStd ** 2 + downStd ** 2) / 2);
      return quality >= 0;
    }
  );
});
```

## User Experience

### UI Flow
1. **Start Calibration** button
2. **Progress bar** (baseline → trial 1/2 → trial 1/2)
3. **Instruction text** (large, center screen)
4. **Visual cue** (arrow UP/DOWN, color change)
5. **Quality result** (checkmark or retry prompt)

### Accessibility
- Screen reader support for instructions
- High-contrast visual cues
- Keyboard fallback always available

### Error Messages
- "Signal quality too low. Try again or use keyboard."
- "Calibration incomplete. Please follow instructions."
- "Connection lost. Reconnecting..."
