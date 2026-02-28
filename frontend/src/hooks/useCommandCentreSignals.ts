import { useEffect, useRef, useState } from "react";
import { API_ENDPOINTS } from "@/config/api";
import type {
  CommandCentreSignalKey,
  CommandCentreSignalPacket,
  CommandCentreSignals,
  StreamStatus,
} from "@/types/eeg";

const SIGNAL_KEYS: CommandCentreSignalKey[] = [
  "focus",
  "alertness",
  "drowsiness",
  "stress",
  "workload",
  "engagement",
  "relaxation",
  "flow",
  "frustration",
];

const DEFAULT_SIGNALS: CommandCentreSignals = {
  focus: 0,
  alertness: 0,
  drowsiness: 0,
  stress: 0,
  workload: 0,
  engagement: 0,
  relaxation: 0,
  flow: 0,
  frustration: 0,
};

export type UseCommandCentreSignalsResult = {
  signals: CommandCentreSignals;
  previousSignals: CommandCentreSignals;
  status: StreamStatus;
  errorMessage: string | null;
  lastUpdated: number | null;
  deviceType: string | null;
};

const normalizeSignals = (input: Partial<Record<CommandCentreSignalKey, unknown>>) => {
  const normalized = { ...DEFAULT_SIGNALS };

  SIGNAL_KEYS.forEach((key) => {
    const value = Number(input[key]);
    if (Number.isFinite(value)) {
      normalized[key] = Math.min(1, Math.max(0, value));
    }
  });

  return normalized;
};

const parseSignalPacket = (message: unknown): CommandCentreSignalPacket | null => {
  if (!message || typeof message !== "object") return null;
  const data = message as Record<string, unknown>;
  const signal = data.signal;
  if (!signal || typeof signal !== "object") return null;
  const packet = signal as Record<string, unknown>;
  if (!packet.signals || typeof packet.signals !== "object") return null;

  const timestampMs = Number(packet.timestamp_ms);
  return {
    timestampMs: Number.isFinite(timestampMs) ? timestampMs : Date.now(),
    signals: normalizeSignals(
      packet.signals as Partial<Record<CommandCentreSignalKey, unknown>>,
    ),
    deviceType:
      typeof packet.device_type === "string" ? packet.device_type : undefined,
    raw:
      packet.raw && typeof packet.raw === "object"
        ? (packet.raw as Record<string, number>)
        : undefined,
  };
};

export const useCommandCentreSignals = (): UseCommandCentreSignalsResult => {
  const [signals, setSignals] = useState<CommandCentreSignals>(DEFAULT_SIGNALS);
  const [previousSignals, setPreviousSignals] =
    useState<CommandCentreSignals>(DEFAULT_SIGNALS);
  const [status, setStatus] = useState<StreamStatus>("disconnected");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [deviceType, setDeviceType] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const currentSignalsRef = useRef<CommandCentreSignals>(DEFAULT_SIGNALS);
  const connectRef = useRef<() => void>(() => {});

  useEffect(() => {
    const connect = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) return;

      const ws = new WebSocket(API_ENDPOINTS.BCI_CC_SIGNALS_WS);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("connected");
        setErrorMessage(null);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as unknown;

          if (
            message &&
            typeof message === "object" &&
            "error" in (message as Record<string, unknown>)
          ) {
            const error = String((message as Record<string, unknown>).error);
            setErrorMessage(error);
            return;
          }

          const packet = parseSignalPacket(message);
          if (!packet) return;

          setPreviousSignals(currentSignalsRef.current);
          currentSignalsRef.current = packet.signals;
          setSignals(packet.signals);
          setLastUpdated(packet.timestampMs);
          if (packet.deviceType) {
            setDeviceType(packet.deviceType);
          }
          setErrorMessage(null);
        } catch (error) {
          console.error("Failed to parse command-centre websocket message", error);
        }
      };

      ws.onclose = () => {
        setStatus("disconnected");
        wsRef.current = null;
        reconnectTimerRef.current = window.setTimeout(
          () => connectRef.current(),
          2000,
        );
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connectRef.current = connect;
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
    };
  }, []);

  return {
    signals,
    previousSignals,
    status,
    errorMessage,
    lastUpdated,
    deviceType,
  };
};
