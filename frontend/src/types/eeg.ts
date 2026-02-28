export type EEGDataPoint = {
  time: number;
  [key: string]: number;
};

export type StreamStatus = "disconnected" | "connected";

export type BlinkDetectionState = {
  detected: boolean;
  lastDetectedAtMs: number | null;
  blinkCount: number;
};

export type BlinkDetectionSettings = {
  selectedChannels: string[];
  requiredChannels: number;
  warmupSamples: number;
  baselineAlpha: number;
  zScoreThreshold: number;
  minAmplitudeUv: number;
  minDeviationUv: number;
  cooldownMs: number;
  flashMs: number;
};

export type ChannelRange = {
  min: number;
  max: number;
  railed: boolean;
  railedWarn: boolean;
  rmsUv: number;
  dcOffsetPercent: number;
};

export type CommandCentreSignalKey =
  | "focus"
  | "alertness"
  | "drowsiness"
  | "stress"
  | "workload"
  | "engagement"
  | "relaxation"
  | "flow"
  | "frustration";

export type CommandCentreSignals = Record<CommandCentreSignalKey, number>;

export type CommandCentreSignalPacket = {
  timestampMs: number;
  signals: CommandCentreSignals;
  deviceType?: string;
  raw?: Record<string, number>;
};

// HeadPlot Model
export type ElectrodeId = string;
export type ChannelId = string;

export type ElectrodeMapping = Record<ElectrodeId, ChannelId>;
