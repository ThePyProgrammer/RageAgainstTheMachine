import { applyDeadzone, EMAFilter, BinaryHysteresis, canTrustCalibration, clamp } from "@ragemachine/bci-shared";
import type { BCIDecodedControls, BCIMode, BCIStreamPacket } from "@ragemachine/bci-shared";

interface PipelineConfig {
  mode: BCIMode;
  confidenceFloor: number;
  decayMsToZero: number;
}

const DEFAULT_HOLD_DECAY_MS = 180;

const safe = (value: number): number => (Number.isFinite(value) ? value : 0);

const featuresToControl = (features: readonly number[], modeConfidence: number): BCIDecodedControls => ({
  throttle: clamp(safe(features[0] ?? 0), -1, 1),
  turn: clamp(safe(features[1] ?? 0), -1, 1),
  fire: safe(features[2] ?? 0) > 0.55,
  confidence: modeConfidence,
  source: "features",
  timestamp: 0,
});

const classifierToControl = (packet: Extract<BCIStreamPacket, { kind: "classifier" }>): BCIDecodedControls => {
  const entries = Object.entries(packet.probabilities);
  const sorted = [...entries].sort((left, right) => right[1] - left[1]);
  const winner = sorted[0]?.[0] ?? "idle";
  const isTurn = winner === "left" || winner === "right";
  const isThrottle = winner === "throttle_up" || winner === "throttle_down";

  return {
    throttle: isThrottle ? (winner === "throttle_up" ? 1 : winner === "throttle_down" ? -1 : 0) : 0,
    turn: isTurn ? (winner === "left" ? -1 : winner === "right" ? 1 : 0) : 0,
    fire: winner === "fire" && packet.confidence > 0.45,
    confidence: packet.confidence,
    source: "classifier",
    timestamp: packet.timestamp,
  };
};

export class ControlPipeline {
  private readonly throttleFilter = new EMAFilter(0.2);
  private readonly turnFilter = new EMAFilter(0.2);
  private readonly fireGate = new BinaryHysteresis({ enterThreshold: 0.6, exitThreshold: 0.4 });
  private lastGood: BCIDecodedControls = {
    throttle: 0,
    turn: 0,
    fire: false,
    confidence: 0,
    source: "features",
    timestamp: 0,
  };
  private lastPacketAtMs = 0;
  private mode: BCIMode;

  constructor(
    private readonly config: PipelineConfig,
    mode: BCIMode,
  ) {
    this.mode = mode;
  }

  private clampAndFilter(raw: BCIDecodedControls, nowMs: number): BCIDecodedControls {
    const smoothedThrottle = this.throttleFilter.next(applyDeadzone(raw.throttle, 0.12));
    const smoothedTurn = this.turnFilter.next(applyDeadzone(raw.turn, 0.1));
    const smoothedFire = this.fireGate.update(raw.confidence) && raw.fire;
    const fire = smoothedFire;

    return {
      throttle: clamp(smoothedThrottle, -1, 1),
      turn: clamp(smoothedTurn, -1, 1),
      fire,
      confidence: raw.confidence,
      source: raw.source,
      timestamp: nowMs,
    };
  }

  onPacket(packet: BCIStreamPacket): BCIDecodedControls {
    if (packet.confidence < this.config.confidenceFloor) {
      return this.holdOrDecay(packet.timestamp, packet.kind);
    }

    const mapped =
      this.mode === "features"
        ? featuresToControl(packet.kind === "features" ? packet.features : [], packet.confidence)
        : classifierToControl(packet);
    const filtered = this.clampAndFilter(mapped, packet.timestamp);

    this.lastGood = filtered;
    this.lastPacketAtMs = packet.timestamp;

    return filtered;
  }

  holdOrDecay(nowMs: number, source: BCIMode): BCIDecodedControls {
    const elapsed = Math.max(0, nowMs - this.lastPacketAtMs);
    const decayFactor = Math.exp(-elapsed / this.config.decayMsToZero);

    if (this.lastPacketAtMs === 0 || elapsed > this.config.decayMsToZero) {
      return {
        throttle: 0,
        turn: 0,
        fire: false,
        confidence: 0,
        source,
        timestamp: nowMs,
      };
    }

    return {
      ...this.lastGood,
      throttle: this.lastGood.throttle * decayFactor,
      turn: this.lastGood.turn * decayFactor,
      fire: false,
      confidence: Math.max(0, this.lastGood.confidence * decayFactor),
      source,
      timestamp: nowMs,
    };
  }

  updateMode(nextMode: BCIMode): void {
    this.mode = nextMode;
  }

  get latestControl(): BCIDecodedControls {
    return this.lastGood;
  }
}

export const createDefaultControlPipeline = () =>
  new ControlPipeline(
    {
      mode: "features",
      confidenceFloor: 0.38,
      decayMsToZero: DEFAULT_HOLD_DECAY_MS,
    },
    "features",
  );

export const canCalibrate = (left: number[], right: number[]): boolean => canTrustCalibration(left, right, 1);
