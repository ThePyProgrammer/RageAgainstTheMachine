export type EEGDataPoint = {
  time: number;
  [key: string]: number;
};

export type StreamStatus = "disconnected" | "connected";

export type ChannelRange = {
  min: number;
  max: number;
  railed: boolean;
  railedWarn: boolean;
  rmsUv: number;
  dcOffsetPercent: number;
};

// HeadPlot Model
export type ElectrodeId = string;
export type ChannelId = string;

export type ElectrodeMapping = Record<ElectrodeId, ChannelId>;
