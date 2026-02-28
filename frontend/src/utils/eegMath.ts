import type { ChannelRange } from "@/types/eeg";
import {
  ADS1299_MAX_UV,
  NEAR_RAILED_THRESHOLD_PERCENT,
  RAILED_THRESHOLD_PERCENT,
} from "@/config/eeg";

// OpenBCI Cyton uses 24-bit ADC with ±200mV = ±200,000 µV range
// Railed = signal hits hardware limits (saturation)
const RAIL_THRESHOLD_UV = ADS1299_MAX_UV;
const THRESHOLD_WARN = NEAR_RAILED_THRESHOLD_PERCENT * 100;
const THRESHOLD_RAILED = RAILED_THRESHOLD_PERCENT * 100;

export const calculateChannelStats = (values: number[]): ChannelRange => {
  if (values.length === 0) {
    return {
      min: 0,
      max: 0,
      railed: false,
      railedWarn: false,
      rmsUv: 0,
      dcOffsetPercent: 0,
    };
  }

  let min = values[0];
  let max = values[0];
  let sum = 0;

  // To calculate Percentage, we need the biggest absolute deviation from 0
  let maxAbs = Math.abs(values[0]);

  for (let i = 0; i < values.length; i++) {
    const val = values[i];
    const absVal = Math.abs(val);

    if (val < min) min = val;
    if (val > max) max = val;
    if (absVal > maxAbs) maxAbs = absVal;

    sum += val;
  }

  // 1. Calculate DC Offset %
  const dcOffsetPercent = (maxAbs / RAIL_THRESHOLD_UV) * 100;

  // 2. Determine Railed Status
  const railedWarn = dcOffsetPercent >= THRESHOLD_WARN;
  const railed = dcOffsetPercent >= THRESHOLD_RAILED;

  // If signal is a flatline (max === min), it's also railed/disconnected
  const isFlatline = max === min || (max === 0 && min === 0);

  // 3. Calculate RMS (Standard Deviation approach)
  const mean = sum / values.length;
  let sumSqDiff = 0;
  for (let i = 0; i < values.length; i++) {
    sumSqDiff += Math.pow(values[i] - mean, 2);
  }
  const rmsUv = Math.sqrt(sumSqDiff / values.length);

  return {
    min,
    max,
    railed: railed || isFlatline,
    railedWarn,
    rmsUv,
    dcOffsetPercent,
  };
};
