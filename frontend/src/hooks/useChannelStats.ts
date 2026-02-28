import { useState, useEffect } from "react";
import { CHANNEL_NAMES } from "@/config/eeg";
import type { ChannelRange, EEGDataPoint } from "@/types/eeg";
import { calculateChannelStats } from "@/utils/eegMath";

export const useChannelStats = (
  dataBufferRef: React.MutableRefObject<EEGDataPoint[]>,
) => {
  const [channelStats, setChannelStats] = useState<
    Record<string, ChannelRange>
  >({});

  useEffect(() => {
    // Run calculation every 250ms to keep UI responsive without blocking the main thread
    const interval = setInterval(() => {
      const buffer = dataBufferRef.current;
      if (buffer.length === 0) return;

      const newStats: Record<string, ChannelRange> = {};


      CHANNEL_NAMES.forEach((ch) => {
        const chKey = `ch${ch}`;
        const chFilteredKey = `fch${ch}`;
        const values = buffer.map(
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
