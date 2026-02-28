import { useState, useEffect, useRef } from "react";
import { API_ENDPOINTS } from "@/config/api";

export interface MotorImageryStats {
  x: number;
  y: number;
  confidence: number;
  latency: number;
  speed: number;
  command: string;
  status: "LANDED" | "MOVING" | "HOVERING";
  prediction: string;
}

export interface MotorImageryTrailPoint {
  x: number;
  y: number;
  timestamp: number;
}

export const useMotorImagery = () => {
  const [stats, setStats] = useState<MotorImageryStats>({
    x: 0,
    y: 0,
    confidence: 0,
    latency: 0,
    speed: 0.0,
    command: "HOVER",
    status: "LANDED",
    prediction: "Idle",
  });

  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingError, setStreamingError] = useState<string | null>(null);
  const [trail, setTrail] = useState<MotorImageryTrailPoint[]>([]);
  const [trailNow, setTrailNow] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoStreamIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );
  const positionRef = useRef({ x: 0, y: 0 });
  const moduleUnavailable = useRef(false);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(API_ENDPOINTS.MI_WS);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("[MI Frontend] Connected to MI Backend");
        setStats((prev) => ({ ...prev, status: "LANDED" }));
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.error) {
            console.error("[MI Frontend] Error from backend:", data.error);

            // Set error message for display
            setStreamingError(data.error);

            if (
              data.error.includes("not initialized") ||
              data.status === "unavailable"
            ) {
              console.error(
                "[MI Frontend] MI module unavailable - stopping reconnection attempts",
              );
              moduleUnavailable.current = true;
            }

            // If EEG stream not running, keep trying but show error
            if (data.status === "eeg_not_running") {
              setIsStreaming(false);
            }
            return;
          }
          if (data.status === "started") {
            console.log("[MI Frontend] Streaming started");
            setIsStreaming(true);
            setStreamingError(null); // Clear any previous errors
            // Start auto-streaming every 3 seconds
            if (autoStreamIntervalRef.current) {
              clearInterval(autoStreamIntervalRef.current);
            }
            autoStreamIntervalRef.current = setInterval(() => {}, 3000);
            return;
          }
          if (data.status === "stopped") {
            console.log("[MI Frontend] Streaming stopped");
            setIsStreaming(false);
            if (autoStreamIntervalRef.current) {
              clearInterval(autoStreamIntervalRef.current);
              autoStreamIntervalRef.current = null;
            }
            return;
          }
          if (data.type === "prediction") {
            const command = String(data.command || "hover");
            const commandLower = command.toLowerCase();
            const confidence = Math.max(
              0,
              Math.min(100, Number(data.confidence) || 0),
            );
            const normalized = confidence / 100;

            // Smooth movement with confidence-based scaling
            const movementScale =
              commandLower === "hover" ? 0 : 0.06 + 0.18 * normalized;
            const speed = commandLower === "hover" ? 0 : 0.5 + 1.5 * normalized;

            // Calculate deltas: left/right (X-axis), forward (Y-axis)
            const deltaX =
              commandLower === "strafe_right"
                ? movementScale
                : commandLower === "strafe_left"
                  ? -movementScale
                  : 0;
            const deltaY = commandLower === "forward" ? movementScale : 0;

            const now = Date.now();
            const nextX = Math.max(
              -1,
              Math.min(1, positionRef.current.x + deltaX),
            );
            const nextY = Math.max(
              -1,
              Math.min(1, positionRef.current.y + deltaY),
            );
            positionRef.current = { x: nextX, y: nextY };

            setStats((prev) => ({
              ...prev,
              prediction: data.label || "Idle",
              confidence,
              command: command.toUpperCase(),
              status:
                data.status ||
                (commandLower === "hover" ? "HOVERING" : "MOVING"),
              x: nextX,
              y: nextY,
              speed,
            }));

            setTrail((prev) => {
              const trimmed = prev.filter(
                (point) => now - point.timestamp <= 6000,
              );
              return [...trimmed, { x: nextX, y: nextY, timestamp: now }];
            });
            setTrailNow(now);
          }
        } catch (e) {
          console.error("Error parsing MI data:", e);
        }
      };

      ws.onclose = () => {
        console.log("[MI Frontend] MI Backend Disconnected");
        setStats((prev) => ({ ...prev, status: "LANDED", speed: 0 }));
        setIsConnected(false);
        setIsStreaming(false);
        positionRef.current = { x: 0, y: 0 };
        setTrail([]);
        setTrailNow(0);

        // Only retry if module is available
        if (!moduleUnavailable.current) {
          console.log("[MI Frontend] Reconnecting in 2s...");
          retryTimeoutRef.current = setTimeout(connect, 2000);
        } else {
          console.log(
            "[MI Frontend] Not reconnecting - module unavailable. Please restart backend.",
          );
        }
      };

      ws.onerror = (err) => {
        console.error("[MI Frontend] WebSocket Error:", err);
        ws.close();
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      if (autoStreamIntervalRef.current) {
        clearInterval(autoStreamIntervalRef.current);
      }
    };
  }, []);

  const startStreaming = (intervalMs = 1000) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.log("[MI Frontend] Cannot start - WebSocket not connected");
      return;
    }
    console.log(
      `[MI Frontend] Sending START command with interval=${intervalMs}ms`,
    );
    wsRef.current.send(
      JSON.stringify({ action: "start", interval_ms: intervalMs, reset: true }),
    );
  };

  const stopStreaming = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.log("[MI Frontend] Cannot stop - WebSocket not connected");
      return;
    }
    console.log("[MI Frontend] Sending STOP command");
    wsRef.current.send(JSON.stringify({ action: "stop" }));
  };

  return {
    stats,
    isConnected,
    isStreaming,
    streamingError,
    trail,
    trailNow,
    startStreaming,
    stopStreaming,
  };
};
