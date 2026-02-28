# INPUT_PIPELINE.md

Transforms raw Thonk EEG signals into discrete paddle commands with calibration-aware normalization.

Location: `/apps/web/src/input/`

## Architecture

```
Thonk (WebSocket)
  │ {up, down, left, right}
  ▼
Signal Receiver (thonkClient.ts)
  │ Raw signals buffered
  ▼
Signal Processor (signalProcessor.ts)
  │ Normalize with baseline
  │ Compute activation score
  │ Apply z-score transform
  ▼
Command Detector (commandDetector.ts)
  │ Threshold detection
  │ Hysteresis
  │ Debounce (k-of-n)
  ▼
Command Output
  │ {direction: "LEFT"|"RIGHT"|"NEUTRAL", timestamp}
  ▼
WebSocket → Server
```

## Control Mapping (Vertical Axis)

Pong paddles move **vertically** (Y-axis). Mapping:

| Thonk Signal | Aggregate | Paddle Command |
|--------------|-----------|----------------|
| up + left    | LEFT      | UP (paddle moves up) |
| down + right | RIGHT     | DOWN (paddle moves down) |

**Rationale**: "LEFT/RIGHT" naming in Thonk maps to "UP/DOWN" in Pong. Internally, we use `PaddleCommand = "UP" | "DOWN" | "NEUTRAL"` for clarity.

## Signal Aggregation

File: `/apps/web/src/input/signalProcessor.ts`

```typescript
export interface ThonkSignal {
  up: number; // 0.0 - 1.0
  down: number;
  left: number;
  right: number;
  timestamp: number; // ms since epoch
}

export interface CalibrationParams {
  baselineMean: number;
  baselineStd: number;
  thresholdLeft: number; // z-score threshold for UP
  thresholdRight: number; // z-score threshold for DOWN
  thresholdEnter: number; // z-score to enter state
  thresholdExit: number; // z-score to exit state (hysteresis)
}

/**
 * Compute activation score from raw signals
 * Activation > 0 → UP, < 0 → DOWN, ~0 → NEUTRAL
 */
export function computeActivation(signal: ThonkSignal): number {
  // Aggregate opposing directions
  const scoreUp = signal.up + signal.left; // 0.0 - 2.0
  const scoreDown = signal.down + signal.right; // 0.0 - 2.0

  // Activation: difference (range -2.0 to +2.0)
  return scoreUp - scoreDown;
}

/**
 * Normalize activation with baseline
 * Returns z-score
 */
export function normalizeActivation(
  activation: number,
  params: CalibrationParams
): number {
  if (params.baselineStd === 0) return 0; // Avoid division by zero
  return (activation - params.baselineMean) / params.baselineStd;
}

/**
 * Filter outliers (spike rejection)
 * Clamp activation to ±3σ to ignore extreme spikes
 */
export function filterOutliers(zScore: number, maxZ: number = 3.0): number {
  return Math.max(-maxZ, Math.min(maxZ, zScore));
}
```

## Command Detection

File: `/apps/web/src/input/commandDetector.ts`

```typescript
import { CalibrationParams } from './signalProcessor';

export type PaddleCommand = 'UP' | 'DOWN' | 'NEUTRAL';

export interface CommandDetectorState {
  currentCommand: PaddleCommand;
  buffer: PaddleCommand[]; // Ring buffer for k-of-n debounce
  bufferIndex: number;
}

const DEBOUNCE_K = 3; // Require 3 consecutive frames
const DEBOUNCE_N = 5; // Out of 5 total frames

/**
 * Detect command from z-score with hysteresis
 */
export function detectCommandRaw(
  zScore: number,
  params: CalibrationParams,
  currentCommand: PaddleCommand
): PaddleCommand {
  const { thresholdLeft, thresholdRight, thresholdEnter, thresholdExit } = params;

  // State machine with hysteresis
  if (currentCommand === 'UP') {
    // Exit UP if z-score drops below exit threshold
    if (zScore < thresholdExit) {
      return 'NEUTRAL';
    }
    return 'UP';
  }

  if (currentCommand === 'DOWN') {
    // Exit DOWN if z-score rises above negative exit threshold
    if (zScore > -thresholdExit) {
      return 'NEUTRAL';
    }
    return 'DOWN';
  }

  // NEUTRAL state: check for entry
  if (zScore > thresholdEnter && zScore > thresholdLeft) {
    return 'UP';
  }
  if (zScore < -thresholdEnter && zScore < -thresholdRight) {
    return 'DOWN';
  }

  return 'NEUTRAL';
}

/**
 * Apply k-of-n debounce filter
 * Command changes only if k out of last n frames agree
 */
export function applyDebounce(
  newCommand: PaddleCommand,
  state: CommandDetectorState
): { command: PaddleCommand; state: CommandDetectorState } {
  // Add to ring buffer
  const buffer = [...state.buffer];
  buffer[state.bufferIndex] = newCommand;
  const nextIndex = (state.bufferIndex + 1) % DEBOUNCE_N;

  // Count occurrences
  const counts = { UP: 0, DOWN: 0, NEUTRAL: 0 };
  buffer.forEach((cmd) => counts[cmd]++);

  // Determine command
  let finalCommand = state.currentCommand;

  if (counts.UP >= DEBOUNCE_K) {
    finalCommand = 'UP';
  } else if (counts.DOWN >= DEBOUNCE_K) {
    finalCommand = 'DOWN';
  } else if (counts.NEUTRAL >= DEBOUNCE_K) {
    finalCommand = 'NEUTRAL';
  }

  return {
    command: finalCommand,
    state: {
      currentCommand: finalCommand,
      buffer,
      bufferIndex: nextIndex,
    },
  };
}

/**
 * Initialize detector state
 */
export function initCommandDetector(): CommandDetectorState {
  return {
    currentCommand: 'NEUTRAL',
    buffer: Array(DEBOUNCE_N).fill('NEUTRAL'),
    bufferIndex: 0,
  };
}
```

## Thonk Client (Placeholder)

File: `/packages/thonk/src/client.ts`

```typescript
import { ThonkSignal } from '@apps/web/input/signalProcessor';

export type ThonkMessageHandler = (signal: ThonkSignal) => void;

/**
 * Placeholder Thonk WebSocket client
 * In production, replace with actual Thonk integration
 */
export class ThonkClient {
  private ws: WebSocket | null = null;
  private handlers: ThonkMessageHandler[] = [];

  constructor(private url: string) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('[Thonk] Connected');
        resolve();
      };

      this.ws.onerror = (err) => {
        console.error('[Thonk] Error:', err);
        reject(err);
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const signal: ThonkSignal = {
            up: data.up ?? 0,
            down: data.down ?? 0,
            left: data.left ?? 0,
            right: data.right ?? 0,
            timestamp: Date.now(),
          };
          this.handlers.forEach((handler) => handler(signal));
        } catch (err) {
          console.error('[Thonk] Parse error:', err);
        }
      };

      this.ws.onclose = () => {
        console.log('[Thonk] Disconnected');
      };
    });
  }

  onSignal(handler: ThonkMessageHandler): void {
    this.handlers.push(handler);
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  /**
   * Mock signal generator for testing
   */
  static createMock(frequency: number = 30): ThonkClient {
    const mock = new ThonkClient('mock://thonk');
    mock.connect = async () => {
      setInterval(() => {
        const signal: ThonkSignal = {
          up: Math.random() * 0.5,
          down: Math.random() * 0.5,
          left: Math.random() * 0.5,
          right: Math.random() * 0.5,
          timestamp: Date.now(),
        };
        mock.handlers.forEach((h) => h(signal));
      }, 1000 / frequency);
    };
    return mock;
  }
}
```

## Integration Hook

File: `/apps/web/src/input/useInputPipeline.ts`

```typescript
import { useEffect, useRef, useState } from 'react';
import { ThonkClient } from '@packages/thonk';
import {
  computeActivation,
  normalizeActivation,
  filterOutliers,
  CalibrationParams,
  ThonkSignal,
} from './signalProcessor';
import {
  detectCommandRaw,
  applyDebounce,
  initCommandDetector,
  PaddleCommand,
} from './commandDetector';

export function useInputPipeline(
  thonkUrl: string,
  calibrationParams: CalibrationParams | null,
  onCommand: (command: PaddleCommand) => void
) {
  const detectorStateRef = useRef(initCommandDetector());
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!calibrationParams) return; // Wait for calibration

    const client = new ThonkClient(thonkUrl);

    client.connect().then(() => {
      setConnected(true);

      client.onSignal((signal: ThonkSignal) => {
        // Compute activation
        const activation = computeActivation(signal);

        // Normalize
        const zScore = normalizeActivation(activation, calibrationParams);

        // Filter outliers
        const filtered = filterOutliers(zScore);

        // Detect command (with hysteresis)
        const rawCommand = detectCommandRaw(
          filtered,
          calibrationParams,
          detectorStateRef.current.currentCommand
        );

        // Apply debounce
        const { command, state } = applyDebounce(rawCommand, detectorStateRef.current);
        detectorStateRef.current = state;

        // Emit command
        onCommand(command);
      });
    });

    return () => {
      client.disconnect();
      setConnected(false);
    };
  }, [thonkUrl, calibrationParams, onCommand]);

  return { connected };
}
```

## Constants

Default thresholds (overridden by calibration):

```typescript
export const DEFAULT_THRESHOLDS = {
  baselineMean: 0.0,
  baselineStd: 1.0,
  thresholdLeft: 1.0, // z-score
  thresholdRight: -1.0, // z-score
  thresholdEnter: 1.2, // Enter state at higher threshold
  thresholdExit: 0.5, // Exit state at lower threshold (hysteresis)
};
```

## Edge Cases

| Scenario | Handling |
|----------|----------|
| Flatline (all signals 0) | Activation = 0 → NEUTRAL |
| Extreme spike (> 3σ) | Clamp to ±3σ |
| Missing Thonk packets | Last command persists (no timeout) |
| Rapid jitter | k-of-n debounce filters noise |
| Calibration incomplete | Reject commands, show warning |
| Thonk disconnect | Stop emitting commands, show overlay |

## Testing

### Unit Tests
- [ ] `computeActivation` sums signals correctly
- [ ] `normalizeActivation` returns valid z-scores
- [ ] `filterOutliers` clamps to ±3σ
- [ ] `detectCommandRaw` respects hysteresis
- [ ] `applyDebounce` requires k-of-n agreement

### Integration Tests
- [ ] Mock Thonk signals produce expected commands
- [ ] Calibration params applied correctly
- [ ] Pipeline latency < 10ms (benchmark)

### Property-Based Tests
```typescript
// Example: activation must be in range
test('Activation range', () => {
  forAll(
    { up: fc.float(0, 1), down: fc.float(0, 1), left: fc.float(0, 1), right: fc.float(0, 1) },
    (signal) => {
      const activation = computeActivation(signal);
      return activation >= -2.0 && activation <= 2.0;
    }
  );
});
```

## Latency Target

- **Goal**: Thonk signal → command output < 10ms
- **Measurement**: Log timestamps at each stage
- **Optimization**: Avoid array allocations in hot path
