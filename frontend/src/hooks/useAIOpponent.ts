import { useCallback, useEffect, useRef, useState } from "react";

import { API_ENDPOINTS } from "@/config/api";

export type OpponentGameMode = "pong" | "combat";
export type OpponentEventType = "player_score" | "ai_score" | "near_score";
export type OpponentNearSide = "player_goal" | "ai_goal";
export type OpponentStreamStatus = "connected" | "disconnected";

export type OpponentGameEvent = {
  event_id: string;
  game_mode: OpponentGameMode;
  event: OpponentEventType;
  score: { player: number; ai: number };
  current_difficulty: number;
  event_context?: { near_side?: OpponentNearSide; proximity?: number };
  timestamp_ms?: number;
};

export type OpponentUpdate = {
  type: "opponent_update";
  event_id: string;
  taunt_text: string;
  difficulty: { previous: number; model_target: number; final: number };
  speech: { mime_type: "audio/mpeg"; audio_base64: string };
  metrics: {
    stress: number;
    frustration: number;
    focus: number;
    alertness: number;
  };
  meta: {
    provider: "responses_speech" | "rule_based";
    latency_ms: number;
    metrics_age_ms: number;
  };
  timestamp_ms: number;
};

export type OpponentError = {
  type: "error";
  code:
    | "INVALID_EVENT"
    | "OPENAI_ERROR"
    | "METRICS_UNAVAILABLE"
    | "RATE_LIMIT";
  message: string;
  recoverable: boolean;
};

export type UseAIOpponentResult = {
  status: OpponentStreamStatus;
  latestUpdate: OpponentUpdate | null;
  lastError: OpponentError | null;
  sendGameEvent: (event: OpponentGameEvent) => void;
  playLatestAudio: () => Promise<boolean>;
  clearError: () => void;
};

const decodeAudioBase64ToUrl = (audioBase64: string): string | null => {
  if (!audioBase64) {
    return null;
  }

  try {
    const binary = window.atob(audioBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "audio/mpeg" });
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error("Failed to decode opponent audio payload", error);
    return null;
  }
};

export const useAIOpponent = (): UseAIOpponentResult => {
  const [status, setStatus] = useState<OpponentStreamStatus>("disconnected");
  const [latestUpdate, setLatestUpdate] = useState<OpponentUpdate | null>(null);
  const [lastError, setLastError] = useState<OpponentError | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const connectRef = useRef<() => void>(() => {});
  const latestAudioUrlRef = useRef<string | null>(null);

  const cleanupAudioUrl = useCallback(() => {
    if (latestAudioUrlRef.current) {
      URL.revokeObjectURL(latestAudioUrlRef.current);
      latestAudioUrlRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(API_ENDPOINTS.OPPONENT_WS);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      setLastError(null);
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as OpponentUpdate | OpponentError;
        if (!payload || typeof payload !== "object" || !("type" in payload)) {
          return;
        }

        if (payload.type === "error") {
          setLastError(payload);
          return;
        }

        if (payload.type === "opponent_update") {
          setLatestUpdate(payload);
          setLastError(null);

          cleanupAudioUrl();
          const url = decodeAudioBase64ToUrl(payload.speech.audio_base64);
          latestAudioUrlRef.current = url;
        }
      } catch (error) {
        console.error("Failed parsing opponent websocket payload", error);
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
  }, [cleanupAudioUrl]);

  useEffect(() => {
    connectRef.current = connect;
    connect();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      cleanupAudioUrl();
    };
  }, [cleanupAudioUrl, connect]);

  const sendGameEvent = useCallback((event: OpponentGameEvent) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setLastError({
        type: "error",
        code: "INVALID_EVENT",
        message: "Opponent websocket is not connected.",
        recoverable: true,
      });
      return;
    }

    ws.send(
      JSON.stringify({
        type: "game_event",
        ...event,
        timestamp_ms: event.timestamp_ms ?? Date.now(),
      }),
    );
  }, []);

  const playLatestAudio = useCallback(async (): Promise<boolean> => {
    const audioUrl = latestAudioUrlRef.current;
    if (!audioUrl) {
      return false;
    }

    try {
      const audio = new Audio(audioUrl);
      await audio.play();
      return true;
    } catch (error) {
      console.error("Failed to play opponent audio", error);
      return false;
    }
  }, []);

  const clearError = useCallback(() => {
    setLastError(null);
  }, []);

  return {
    status,
    latestUpdate,
    lastError,
    sendGameEvent,
    playLatestAudio,
    clearError,
  };
};

