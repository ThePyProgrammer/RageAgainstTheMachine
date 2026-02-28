export type BCIMode = "classifier" | "features";

export type ClassifierLabel = "left" | "right" | "throttle_up" | "throttle_down" | "fire" | "idle";

export interface SessionPayload {
  readonly sessionId: string;
  readonly timestamp: number;
}

export interface ClassifierTelemetry extends SessionPayload {
  readonly kind: "classifier";
  readonly label: ClassifierLabel;
  readonly probabilities: Record<ClassifierLabel, number>;
  readonly confidence: number;
}

export interface FeatureTelemetry extends SessionPayload {
  readonly kind: "features";
  readonly features: readonly number[];
  readonly confidence: number;
}

export type BCIStreamPacket = ClassifierTelemetry | FeatureTelemetry;

export interface BCIModeChoice {
  readonly requested: BCIMode;
  readonly active: BCIMode;
}

export interface BCIDecodedControls {
  readonly throttle: number;
  readonly turn: number;
  readonly fire: boolean;
  readonly confidence: number;
  readonly source: BCIMode;
  readonly timestamp: number;
}

export interface CalibrationProfile {
  readonly sessionId: string;
  readonly mean: readonly number[];
  readonly variance: readonly number[];
  readonly updatedAt: number;
}

export interface InputSample {
  readonly sessionId: string;
  readonly timestamp: number;
  readonly controls: BCIDecodedControls;
}

export interface InputSampleSeries {
  readonly samples: readonly InputSample[];
  readonly createdAt: number;
}
