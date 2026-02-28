import { useState, useEffect } from "react";
import { CHANNEL_NAMES } from "@/config/eeg";
import type { ChannelRange, EEGDataPoint } from "@/types/eeg";
import { calculateChannelStats } from "@/utils/eegMath";

const STATS_WINDOW_POINTS = 1250; // 5s at 250Hz

export const useChannelStats = (
  dataBufferRef: React.MutableRefObject<EEGDataPoint[]>,
) => {
  const [channelStats, setChannelStats] = useState<
    Record<string, ChannelRange>
  >({});

  useEffect(() => {
    let lastProcessedLength = -1;

    // Run calculation every 250ms to keep UI responsive without blocking the main thread
    const interval = setInterval(() => {
      const buffer = dataBufferRef.current;
      if (buffer.length === 0 || buffer.length === lastProcessedLength) return;
      lastProcessedLength = buffer.length;

      const recentBuffer =
        buffer.length > STATS_WINDOW_POINTS
          ? buffer.slice(-STATS_WINDOW_POINTS)
          : buffer;

      const newStats: Record<string, ChannelRange> = {};

      CHANNEL_NAMES.forEach((ch) => {
        const chKey = `ch${ch}`;
        const chFilteredKey = `fch${ch}`;
        const values = recentBuffer.map(
          (p) => (p[chFilteredKey] as number) ?? (p[chKey] as number) ?? 0,
        );

        newStats[chKey] = calculateChannelStats(values);
      });

      setChannelStats(newStats);
    }, 250);

    return () => clearInterval(interval);
  }, [dataBufferRef]);

  return channelStats;
};
