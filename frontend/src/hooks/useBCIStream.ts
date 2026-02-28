import { useState, useRef, useCallback, useEffect } from "react";
import { MAX_POINTS } from "@/config/eeg";
import { API_ENDPOINTS, buildStartUrl } from "@/config/api";
import { useDevice } from "@/contexts/DeviceContext";
import type { EEGDataPoint, StreamStatus } from "@/types/eeg";
import { useChannelStats } from "@/hooks/useChannelStats";

const BASE_COLS = 4; // sample_index, ts_unix_ms, ts_formatted, marker

export const useBCIStream = () => {
  const { deviceType, deviceConfig } = useDevice();
  const [displayData, setDisplayData] = useState<EEGDataPoint[]>([]);
  const [status, setStatus] = useState<StreamStatus>("disconnected");
  const [sampleCount, setSampleCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const dataBufferRef = useRef<EEGDataPoint[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const timeOffsetRef = useRef(0);
  const connectRef = useRef<() => void>(() => { });
  const stopEmbeddingsCallbackRef = useRef<(() => Promise<void>) | null>(null);
  const channelRanges = useChannelStats(dataBufferRef);

  // Derive column layout from device config
  const eegCols = deviceConfig.channelNames.length;
  const accelCols = deviceConfig.accelChannels;
  const analogCols = deviceConfig.analogChannels;
  const auxCols = deviceConfig.auxChannels;
  const filteredCols = eegCols;
  const samplingRate = deviceConfig.samplingRate;
  const maxUv = deviceConfig.maxUv;

  const processIncomingData = useCallback((samples: (number | string)[][]) => {
    const newPoints: EEGDataPoint[] = [];

    for (const sample of samples) {
      if (!sample || sample.length < BASE_COLS + eegCols) continue;

      const point: EEGDataPoint = { time: timeOffsetRef.current };

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
      });

      timeOffsetRef.current += 1 / samplingRate;
      newPoints.push(point);
    }

    if (newPoints.length > 0) {
      dataBufferRef.current = [...dataBufferRef.current, ...newPoints].slice(
        -MAX_POINTS,
      );
      setSampleCount((prev) => prev + newPoints.length);
    }
  }, [eegCols, auxCols, accelCols, analogCols, filteredCols, samplingRate, maxUv, deviceConfig.channelNames]);

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
      reconnectTimer.current = window.setTimeout(
        () => connectRef.current(),
        2000,
      );
    };
  }, [processIncomingData]);

  const executeStreamAction = async (action: "start" | "stop") => {
    try {
      const endpoint =
        action === "start"
          ? buildStartUrl(deviceType)
          : API_ENDPOINTS.BCI_STOP;
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
        // Stop embeddings when stream stops
        if (stopEmbeddingsCallbackRef.current) {
          await stopEmbeddingsCallbackRef.current();
        }
      } else if (data.status === "streaming" || action === "start") {
        setStatus("connected");
        setErrorMessage(null);
        // isStreaming will be set to true when samples start arriving
      }
    } catch (e) {
      console.error(e);
      setErrorMessage("Failed to communicate with the server");
      setIsStreaming(false);
    }
  };

  // Clear data buffer when device type changes
  useEffect(() => {
    dataBufferRef.current = [];
    setDisplayData([]);
    setSampleCount(0);
    timeOffsetRef.current = 0;
  }, [deviceType]);

  useEffect(() => {
    connectRef.current = connectWebSocket;
  }, [connectWebSocket]);

  useEffect(() => {
    let lastDrawTime = 0;
    const FRAME_INTERVAL = 16; // ~60 FPS to match display refresh

    const draw = (timestamp: number) => {
      const elapsed = timestamp - lastDrawTime;
      if (elapsed > FRAME_INTERVAL) {
        if (dataBufferRef.current.length > 0) {
          setDisplayData([...dataBufferRef.current]);
        }
        lastDrawTime = timestamp - (elapsed % FRAME_INTERVAL);
      }
      animationFrameRef.current = requestAnimationFrame(draw);
    };
    animationFrameRef.current = requestAnimationFrame(draw);
    return () => {
      if (animationFrameRef.current)
        cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
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
    registerStopEmbeddings,
  };
};
