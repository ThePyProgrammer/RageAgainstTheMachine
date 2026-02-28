import { useCallback, useEffect, useRef, useState } from "react";

import { API_ENDPOINTS } from "@/config/api";

export type OpponentGameMode = "pong" | "combat";
export type OpponentInputMode = "eeg" | "keyboard_paddle" | "keyboard_ball";
export type OpponentEventType = "player_score" | "ai_score" | "near_score";
export type OpponentNearSide = "player_goal" | "ai_goal";
export type OpponentStreamStatus = "connected" | "disconnected";

export type OpponentGameEvent = {
  event_id: string;
  game_mode: OpponentGameMode;
  input_mode?: OpponentInputMode;
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
  latestFeedbackUpdate: OpponentUpdate | null;
  lastError: OpponentError | null;
  sendGameEvent: (event: OpponentGameEvent) => void;
  playLatestAudio: (eventId?: string) => Promise<boolean>;
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
  const [latestFeedbackUpdate, setLatestFeedbackUpdate] = useState<OpponentUpdate | null>(null);
  const [lastError, setLastError] = useState<OpponentError | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const connectRef = useRef<() => void>(() => {});
  const shouldReconnectRef = useRef(true);
  const latestAudioEventIdRef = useRef<string | null>(null);
  const cachedAudioUrlsRef = useRef<Map<string, string>>(new Map());
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);

  const clearCachedAudio = useCallback(() => {
    for (const url of cachedAudioUrlsRef.current.values()) {
      URL.revokeObjectURL(url);
    }
    cachedAudioUrlsRef.current.clear();
    latestAudioEventIdRef.current = null;
  }, []);

  const cacheAudioForEvent = useCallback((eventId: string, audioBase64: string) => {
    const url = decodeAudioBase64ToUrl(audioBase64);
    if (!url) {
      return;
    }

    const existingUrl = cachedAudioUrlsRef.current.get(eventId);
    if (existingUrl) {
      URL.revokeObjectURL(existingUrl);
    }
    cachedAudioUrlsRef.current.set(eventId, url);
    latestAudioEventIdRef.current = eventId;

    while (cachedAudioUrlsRef.current.size > 24) {
      const oldestEventId = cachedAudioUrlsRef.current.keys().next().value as
        | string
        | undefined;
      if (!oldestEventId) {
        break;
      }
      const oldestUrl = cachedAudioUrlsRef.current.get(oldestEventId);
      if (oldestUrl) {
        URL.revokeObjectURL(oldestUrl);
      }
      cachedAudioUrlsRef.current.delete(oldestEventId);
      if (latestAudioEventIdRef.current === oldestEventId) {
        latestAudioEventIdRef.current = null;
      }
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
          if (payload.taunt_text.trim() || payload.speech.audio_base64) {
            setLatestFeedbackUpdate(payload);
          }
          setLastError(null);

          if (payload.speech.audio_base64) {
            cacheAudioForEvent(payload.event_id, payload.speech.audio_base64);
          }
        }
      } catch (error) {
        console.error("Failed parsing opponent websocket payload", error);
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      wsRef.current = null;
      if (!shouldReconnectRef.current) {
        return;
      }
      reconnectTimerRef.current = window.setTimeout(
        () => connectRef.current(),
        2000,
      );
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [cacheAudioForEvent]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connectRef.current = connect;
    connect();

    return () => {
      shouldReconnectRef.current = false;
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      if (activeAudioRef.current) {
        activeAudioRef.current.pause();
        activeAudioRef.current = null;
      }
      clearCachedAudio();
    };
  }, [clearCachedAudio, connect]);

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

  const playLatestAudio = useCallback(async (eventId?: string): Promise<boolean> => {
    const targetEventId = eventId ?? latestAudioEventIdRef.current;
    if (!targetEventId) {
      return false;
    }

    const audioUrl = cachedAudioUrlsRef.current.get(targetEventId);
    if (!audioUrl) {
      return false;
    }

    try {
      if (activeAudioRef.current) {
        activeAudioRef.current.pause();
      }
      const audio = new Audio(audioUrl);
      audio.preload = "auto";
      activeAudioRef.current = audio;
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
    latestFeedbackUpdate,
    lastError,
    sendGameEvent,
    playLatestAudio,
    clearError,
  };
};
