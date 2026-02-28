import {
  createElement,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import { MAX_POINTS } from "@/config/eeg";
import { API_ENDPOINTS, buildStartUrl } from "@/config/api";
import { useDevice } from "@/contexts/DeviceContext";
import type {
  BlinkDetectionSettings,
  BlinkDetectionState,
  EEGDataPoint,
  StreamStatus,
} from "@/types/eeg";
import { useChannelStats } from "@/hooks/useChannelStats";

const BASE_COLS = 4; // sample_index, ts_unix_ms, ts_formatted, marker
const DISPLAY_MAX_POINTS = 10_000;
const BLINK_SETTINGS_STORAGE_KEY_PREFIX = "ratm.blink-settings";
const DEFAULT_BLINK_BASELINE_ALPHA = 0.02;
const DEFAULT_BLINK_COOLDOWN_MS = 250;
const DEFAULT_BLINK_FLASH_MS = 160;
const DEFAULT_BLINK_Z_SCORE_THRESHOLD = 4.0;
const DEFAULT_BLINK_MIN_AMPLITUDE_UV = 50;
const DEFAULT_BLINK_MIN_DEVIATION_UV = 7;
const DEFAULT_BLINK_WARMUP_SECONDS = 1;

const BLINK_ALPHA_MIN = 0.001;
const BLINK_ALPHA_MAX = 0.3;
const BLINK_Z_SCORE_MIN = 0.5;
const BLINK_Z_SCORE_MAX = 10;
const BLINK_AMPLITUDE_MIN_UV = 0;
const BLINK_AMPLITUDE_MAX_UV = 500;
const BLINK_DEVIATION_MIN_UV = 0.1;
const BLINK_DEVIATION_MAX_UV = 100;
const BLINK_COOLDOWN_MIN_MS = 0;
const BLINK_COOLDOWN_MAX_MS = 5000;
const BLINK_FLASH_MIN_MS = 50;
const BLINK_FLASH_MAX_MS = 3000;
const BLINK_WARMUP_MAX_SECONDS = 10;

type BlinkChannelStats = {
  meanAbsUv: number;
  deviationUv: number;
};

type BlinkRuntimeSettings = BlinkDetectionSettings & {
  selectedChannelSet: Set<string>;
};

const clamp = (value: number, min: number, max: number): number => {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
};

const getDefaultBlinkChannels = (channelNames: readonly string[]): string[] => {
  const frontalChannels = channelNames.filter((channelName) =>
    /^(af|fp|f)/i.test(channelName),
  );
  if (frontalChannels.length > 0) {
    return frontalChannels;
  }
  return channelNames.slice(0, Math.min(2, channelNames.length));
};

const createDefaultBlinkSettings = (
  channelNames: readonly string[],
  samplingRate: number,
): BlinkDetectionSettings => {
  const selectedChannels = getDefaultBlinkChannels(channelNames);
  return {
    selectedChannels,
    requiredChannels: Math.min(2, selectedChannels.length),
    warmupSamples: Math.round(DEFAULT_BLINK_WARMUP_SECONDS * samplingRate),
    baselineAlpha: DEFAULT_BLINK_BASELINE_ALPHA,
    zScoreThreshold: DEFAULT_BLINK_Z_SCORE_THRESHOLD,
    minAmplitudeUv: DEFAULT_BLINK_MIN_AMPLITUDE_UV,
    minDeviationUv: DEFAULT_BLINK_MIN_DEVIATION_UV,
    cooldownMs: DEFAULT_BLINK_COOLDOWN_MS,
    flashMs: DEFAULT_BLINK_FLASH_MS,
  };
};

const normalizeBlinkSettings = (
  settings: BlinkDetectionSettings,
  channelNames: readonly string[],
  samplingRate: number,
): BlinkDetectionSettings => {
  const allowed = new Set(channelNames);
  const deduped = Array.from(new Set(settings.selectedChannels));
  const selectedChannels = channelNames.filter((channelName) =>
    deduped.includes(channelName),
  );
  const fallback =
    selectedChannels.length > 0
      ? selectedChannels
      : channelNames.length > 0
        ? [channelNames[0]]
        : [];

  const maxWarmupSamples = Math.round(samplingRate * BLINK_WARMUP_MAX_SECONDS);
  const requiredMax = fallback.length;
  const requiredChannels =
    requiredMax > 0
      ? Math.round(clamp(settings.requiredChannels, 1, requiredMax))
      : 0;

  return {
    selectedChannels: fallback.filter((channelName) => allowed.has(channelName)),
    requiredChannels,
    warmupSamples: Math.round(clamp(settings.warmupSamples, 0, maxWarmupSamples)),
    baselineAlpha: clamp(settings.baselineAlpha, BLINK_ALPHA_MIN, BLINK_ALPHA_MAX),
    zScoreThreshold: clamp(
      settings.zScoreThreshold,
      BLINK_Z_SCORE_MIN,
      BLINK_Z_SCORE_MAX,
    ),
    minAmplitudeUv: clamp(
      settings.minAmplitudeUv,
      BLINK_AMPLITUDE_MIN_UV,
      BLINK_AMPLITUDE_MAX_UV,
    ),
    minDeviationUv: clamp(
      settings.minDeviationUv,
      BLINK_DEVIATION_MIN_UV,
      BLINK_DEVIATION_MAX_UV,
    ),
    cooldownMs: Math.round(
      clamp(settings.cooldownMs, BLINK_COOLDOWN_MIN_MS, BLINK_COOLDOWN_MAX_MS),
    ),
    flashMs: Math.round(
      clamp(settings.flashMs, BLINK_FLASH_MIN_MS, BLINK_FLASH_MAX_MS),
    ),
  };
};

const toRuntimeBlinkSettings = (
  settings: BlinkDetectionSettings,
): BlinkRuntimeSettings => ({
  ...settings,
  selectedChannelSet: new Set(settings.selectedChannels),
});

const getBlinkSettingsStorageKey = (deviceType: string): string =>
  `${BLINK_SETTINGS_STORAGE_KEY_PREFIX}:${deviceType}`;

const loadBlinkSettingsFromStorage = (
  deviceType: string,
  channelNames: readonly string[],
  samplingRate: number,
): BlinkDetectionSettings | null => {
  if (typeof window === "undefined") {
    return null;
  }

  const defaults = createDefaultBlinkSettings(channelNames, samplingRate);
  const raw = window.localStorage.getItem(getBlinkSettingsStorageKey(deviceType));
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    const parsedSettings = parsed as Partial<BlinkDetectionSettings>;
    const selectedChannels = Array.isArray(parsedSettings.selectedChannels)
      ? parsedSettings.selectedChannels.filter(
          (channel): channel is string => typeof channel === "string",
        )
      : defaults.selectedChannels;

    return normalizeBlinkSettings(
      {
        ...defaults,
        ...parsedSettings,
        selectedChannels,
      },
      channelNames,
      samplingRate,
    );
  } catch {
    return null;
  }
};

type BCIStreamContextValue = {
  displayData: EEGDataPoint[];
  status: StreamStatus;
  sampleCount: number;
  channelRanges: ReturnType<typeof useChannelStats>;
  executeStreamAction: (action: "start" | "stop") => Promise<void>;
  errorMessage: string | null;
  clearError: () => void;
  isStreaming: boolean;
  blink: BlinkDetectionState;
  blinkSettings: BlinkDetectionSettings;
  updateBlinkSettings: (
    updates: Partial<Omit<BlinkDetectionSettings, "selectedChannels">>,
  ) => void;
  toggleBlinkDetectionChannel: (channelName: string) => void;
  resetBlinkSettings: () => void;
  registerStopEmbeddings: (callback: () => Promise<void>) => void;
};

const BCIStreamContext = createContext<BCIStreamContextValue | undefined>(
  undefined,
);

const useProvideBCIStream = (): BCIStreamContextValue => {
  const { deviceType, deviceConfig } = useDevice();
  const [displayData, setDisplayData] = useState<EEGDataPoint[]>([]);
  const [status, setStatus] = useState<StreamStatus>("disconnected");
  const [sampleCount, setSampleCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [blink, setBlink] = useState<BlinkDetectionState>({
    detected: false,
    lastDetectedAtMs: null,
    blinkCount: 0,
  });

  const dataBufferRef = useRef<EEGDataPoint[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const blinkResetTimerRef = useRef<number | null>(null);
  const timeOffsetRef = useRef(0);
  const connectRef = useRef<() => void>(() => {});
  const stopEmbeddingsCallbackRef = useRef<(() => Promise<void>) | null>(null);
  const dataVersionRef = useRef(0);
  const renderedVersionRef = useRef(-1);
  const blinkStatsRef = useRef<Record<string, BlinkChannelStats>>({});
  const blinkWarmupSamplesRef = useRef(0);
  const lastBlinkAtMsRef = useRef(-Infinity);
  const channelRanges = useChannelStats(dataBufferRef);

  // Derive column layout from device config
  const eegCols = deviceConfig.channelNames.length;
  const accelCols = deviceConfig.accelChannels;
  const analogCols = deviceConfig.analogChannels;
  const auxCols = deviceConfig.auxChannels;
  const filteredCols = eegCols;
  const samplingRate = deviceConfig.samplingRate;
  const maxUv = deviceConfig.maxUv;
  const initialBlinkSettings = useMemo(
    () => createDefaultBlinkSettings(deviceConfig.channelNames, samplingRate),
    [deviceConfig.channelNames, samplingRate],
  );
  const [blinkSettings, setBlinkSettings] =
    useState<BlinkDetectionSettings>(initialBlinkSettings);
  const blinkSettingsRef = useRef<BlinkRuntimeSettings>(
    toRuntimeBlinkSettings(initialBlinkSettings),
  );
  const skipNextBlinkSettingsPersistRef = useRef(true);

  useEffect(() => {
    blinkSettingsRef.current = toRuntimeBlinkSettings(blinkSettings);
  }, [blinkSettings]);

  const processIncomingData = useCallback(
    (samples: (number | string)[][]) => {
      const newPoints: EEGDataPoint[] = [];
      let detectedBlinks = 0;
      let latestBlinkAtMs: number | null = null;
      const blinkConfig = blinkSettingsRef.current;

      for (const sample of samples) {
        if (!sample || sample.length < BASE_COLS + eegCols) continue;

        const sampleTimestampMs =
          typeof sample[1] === "number" ? (sample[1] as number) : Date.now();
        const point: EEGDataPoint = {
          time: timeOffsetRef.current,
          timestampMs: sampleTimestampMs,
        };
        let blinkCandidateChannels = 0;

        // Determine where optional columns start
        const rawEnd = BASE_COLS + eegCols;
        const auxEnd = rawEnd + auxCols;
        const accelEnd = auxEnd + accelCols;
        const analogEnd = accelEnd + analogCols;
        const hasExtra = sample.length >= analogEnd;

        const filteredIdx = hasExtra ? analogEnd : -1;
        const railedIdx =
          filteredIdx >= 0 &&
          sample.length >= filteredIdx + filteredCols + eegCols
            ? filteredIdx + filteredCols
            : -1;
        const percentIdx = railedIdx >= 0 ? railedIdx + eegCols : -1;
        const uvrmsIdx = percentIdx >= 0 ? percentIdx + eegCols : -1;

        deviceConfig.channelNames.forEach((chName: string, chIdx: number) => {
          const rawVal = sample[BASE_COLS + chIdx];
          const rawUv = typeof rawVal === "number" ? (rawVal as number) : 0;
          point[`ch${chName}`] = rawUv;

          const filtVal =
            filteredIdx >= 0 ? sample[filteredIdx + chIdx] : undefined;
          const filteredUv =
            typeof filtVal === "number" ? (filtVal as number) : rawUv;
          point[`fch${chName}`] = filteredUv;

          const railedFlag =
            railedIdx >= 0 ? sample[railedIdx + chIdx] : undefined;
          const percentVal =
            percentIdx >= 0 ? sample[percentIdx + chIdx] : undefined;
          const rmsVal = uvrmsIdx >= 0 ? sample[uvrmsIdx + chIdx] : undefined;

          const computedPercent = Math.min(
            100,
            Math.max(0, (Math.abs(rawUv) / maxUv) * 100),
          );
          const percent =
            typeof percentVal === "number"
              ? (percentVal as number)
              : computedPercent;
          const railedStrict =
            typeof railedFlag === "number"
              ? (railedFlag as number)
              : percent >= 90
                ? 1
                : 0;
          const uvrms = typeof rmsVal === "number" ? (rmsVal as number) : 0;

          point[`dcOffsetPercent_${chName}`] = percent;
          point[`railedStrict_${chName}`] = railedStrict;
          point[`uvrms_${chName}`] = uvrms;

          if (blinkConfig.selectedChannelSet.has(chName)) {
            const absFilteredUv = Math.abs(filteredUv);
            const existingStats = blinkStatsRef.current[chName];

            if (existingStats) {
              const dynamicDeviation = Math.max(
                existingStats.deviationUv,
                blinkConfig.minDeviationUv,
              );
              const zScore =
                (absFilteredUv - existingStats.meanAbsUv) / dynamicDeviation;

              if (
                absFilteredUv >= blinkConfig.minAmplitudeUv &&
                zScore >= blinkConfig.zScoreThreshold
              ) {
                blinkCandidateChannels += 1;
              }

              const meanDelta = absFilteredUv - existingStats.meanAbsUv;
              const deviationDelta = Math.abs(meanDelta) - existingStats.deviationUv;
              existingStats.meanAbsUv += blinkConfig.baselineAlpha * meanDelta;
              existingStats.deviationUv +=
                blinkConfig.baselineAlpha * deviationDelta;
            } else {
              blinkStatsRef.current[chName] = {
                meanAbsUv: absFilteredUv,
                deviationUv: blinkConfig.minDeviationUv,
              };
            }
          }
        });

        blinkWarmupSamplesRef.current += 1;
        const hasWarmupData =
          blinkWarmupSamplesRef.current >= blinkConfig.warmupSamples;
        const cooldownElapsed =
          sampleTimestampMs - lastBlinkAtMsRef.current >= blinkConfig.cooldownMs;
        const canDetectBlink =
          blinkConfig.requiredChannels > 0 &&
          blinkConfig.selectedChannels.length > 0;
        if (
          canDetectBlink &&
          hasWarmupData &&
          cooldownElapsed &&
          blinkCandidateChannels >= blinkConfig.requiredChannels
        ) {
          detectedBlinks += 1;
          latestBlinkAtMs = sampleTimestampMs;
          lastBlinkAtMsRef.current = sampleTimestampMs;
        }

        timeOffsetRef.current += 1 / samplingRate;
        newPoints.push(point);
      }

      if (newPoints.length > 0) {
        const buffer = dataBufferRef.current;
        buffer.push(...newPoints);
        if (buffer.length > MAX_POINTS) {
          buffer.splice(0, buffer.length - MAX_POINTS);
        }
        dataVersionRef.current += 1;
        setSampleCount((prev) => prev + newPoints.length);
      }

      if (detectedBlinks > 0) {
        setBlink((prev) => ({
          detected: true,
          lastDetectedAtMs: latestBlinkAtMs,
          blinkCount: prev.blinkCount + detectedBlinks,
        }));
        if (blinkResetTimerRef.current) {
          window.clearTimeout(blinkResetTimerRef.current);
        }
        blinkResetTimerRef.current = window.setTimeout(() => {
          setBlink((prev) => {
            if (!prev.detected) {
              return prev;
            }
            return { ...prev, detected: false };
          });
          blinkResetTimerRef.current = null;
        }, blinkConfig.flashMs);
      }
    },
    [
      accelCols,
      analogCols,
      auxCols,
      deviceConfig.channelNames,
      eegCols,
      filteredCols,
      maxUv,
      samplingRate,
    ],
  );

  const updateBlinkSettings = useCallback(
    (updates: Partial<Omit<BlinkDetectionSettings, "selectedChannels">>) => {
      setBlinkSettings((previous) =>
        normalizeBlinkSettings(
          { ...previous, ...updates },
          deviceConfig.channelNames,
          samplingRate,
        ),
      );
    },
    [deviceConfig.channelNames, samplingRate],
  );

  const toggleBlinkDetectionChannel = useCallback(
    (channelName: string) => {
      if (!deviceConfig.channelNames.includes(channelName)) {
        return;
      }

      setBlinkSettings((previous) => {
        const isSelected = previous.selectedChannels.includes(channelName);
        const nextSelected = isSelected
          ? previous.selectedChannels.filter((name) => name !== channelName)
          : [...previous.selectedChannels, channelName];

        const selectedChannels =
          nextSelected.length > 0 ? nextSelected : [channelName];

        return normalizeBlinkSettings(
          { ...previous, selectedChannels },
          deviceConfig.channelNames,
          samplingRate,
        );
      });
    },
    [deviceConfig.channelNames, samplingRate],
  );

  const resetBlinkSettings = useCallback(() => {
    setBlinkSettings(createDefaultBlinkSettings(deviceConfig.channelNames, samplingRate));
  }, [deviceConfig.channelNames, samplingRate]);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(API_ENDPOINTS.BCI_WS);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      setErrorMessage(null);
    };
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.samples) {
          processIncomingData(msg.samples);
          setStatus("connected");
          setIsStreaming(true);
          setErrorMessage(null);
        } else if (msg.error) {
          setErrorMessage(msg.error);
          setIsStreaming(false);
        }
      } catch (e) {
        console.error("Error parsing WebSocket message:", e);
      }
    };
    ws.onclose = () => {
      setStatus("disconnected");
      wsRef.current = null;
      reconnectTimer.current = window.setTimeout(() => connectRef.current(), 2000);
    };
  }, [processIncomingData]);

  const executeStreamAction = async (action: "start" | "stop") => {
    try {
      const endpoint =
        action === "start" ? buildStartUrl(deviceType) : API_ENDPOINTS.BCI_STOP;
      const response = await fetch(endpoint, { method: "POST" });

      // Handle 400 errors gracefully (e.g., trying to stop when not running)
      if (!response.ok && response.status === 400) {
        setErrorMessage(null);
        setIsStreaming(false);
        return;
      }

      const data = await response.json();

      if (data.error) {
        setErrorMessage(data.error);
        setIsStreaming(false);
      } else if (data.status === "stopped" || action === "stop") {
        setStatus("connected");
        setErrorMessage(null);
        setIsStreaming(false);
        if (stopEmbeddingsCallbackRef.current) {
          await stopEmbeddingsCallbackRef.current();
        }
      } else if (data.status === "streaming" || action === "start") {
        setStatus("connected");
        setErrorMessage(null);
      }
    } catch (e) {
      console.error(e);
      setErrorMessage("Failed to communicate with the server");
      setIsStreaming(false);
    }
  };

  // Clear stream buffers when device type changes
  useEffect(() => {
    const persistedBlinkSettings = loadBlinkSettingsFromStorage(
      deviceType,
      deviceConfig.channelNames,
      samplingRate,
    );
    const defaultBlinkSettings = createDefaultBlinkSettings(
      deviceConfig.channelNames,
      samplingRate,
    );
    const activeBlinkSettings = persistedBlinkSettings ?? defaultBlinkSettings;

    dataBufferRef.current = [];
    timeOffsetRef.current = 0;
    blinkStatsRef.current = {};
    blinkWarmupSamplesRef.current = 0;
    lastBlinkAtMsRef.current = -Infinity;
    blinkSettingsRef.current = toRuntimeBlinkSettings(activeBlinkSettings);
    skipNextBlinkSettingsPersistRef.current = true;
    if (blinkResetTimerRef.current) {
      window.clearTimeout(blinkResetTimerRef.current);
      blinkResetTimerRef.current = null;
    }
    dataVersionRef.current += 1;
    renderedVersionRef.current = -1;
    const frame = window.requestAnimationFrame(() => {
      setDisplayData([]);
      setSampleCount(0);
      setBlinkSettings(activeBlinkSettings);
      setBlink({
        detected: false,
        lastDetectedAtMs: null,
        blinkCount: 0,
      });
    });
    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [deviceConfig.channelNames, deviceType, samplingRate]);

  useEffect(() => {
    if (skipNextBlinkSettingsPersistRef.current) {
      skipNextBlinkSettingsPersistRef.current = false;
      return;
    }
    if (typeof window === "undefined") {
      return;
    }
    try {
      window.localStorage.setItem(
        getBlinkSettingsStorageKey(deviceType),
        JSON.stringify(blinkSettings),
      );
    } catch {
      // Ignore storage errors; detection still works with in-memory settings.
    }
  }, [blinkSettings, deviceType]);

  useEffect(() => {
    connectRef.current = connectWebSocket;
  }, [connectWebSocket]);

  useEffect(() => {
    let lastDrawTime = 0;
    const FRAME_INTERVAL = 33; // ~30 FPS

    const draw = (timestamp: number) => {
      const elapsed = timestamp - lastDrawTime;
      if (elapsed > FRAME_INTERVAL) {
        if (dataVersionRef.current !== renderedVersionRef.current) {
          const buffer = dataBufferRef.current;
          const start = Math.max(0, buffer.length - DISPLAY_MAX_POINTS);
          setDisplayData(buffer.slice(start));
          renderedVersionRef.current = dataVersionRef.current;
        }
        lastDrawTime = timestamp - (elapsed % FRAME_INTERVAL);
      }
      animationFrameRef.current = requestAnimationFrame(draw);
    };

    animationFrameRef.current = requestAnimationFrame(draw);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (blinkResetTimerRef.current) clearTimeout(blinkResetTimerRef.current);
    };
  }, [connectWebSocket]);

  const clearError = useCallback(() => {
    setErrorMessage(null);
  }, []);

  const registerStopEmbeddings = useCallback(
    (callback: () => Promise<void>) => {
      stopEmbeddingsCallbackRef.current = callback;
    },
    [],
  );

  return {
    displayData,
    status,
    sampleCount,
    channelRanges,
    executeStreamAction,
    errorMessage,
    clearError,
    isStreaming,
    blink,
    blinkSettings,
    updateBlinkSettings,
    toggleBlinkDetectionChannel,
    resetBlinkSettings,
    registerStopEmbeddings,
  };
};

export function BCIStreamProvider({ children }: { children: ReactNode }) {
  const value = useProvideBCIStream();
  return createElement(BCIStreamContext.Provider, { value }, children);
}

export const useBCIStream = () => {
  const context = useContext(BCIStreamContext);
  if (!context) {
    throw new Error("useBCIStream must be used within a BCIStreamProvider");
  }
  return context;
};
