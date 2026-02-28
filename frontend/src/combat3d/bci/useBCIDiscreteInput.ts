import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Discrete BCI commands produced by a simple EEG pipeline
 * (e.g. Muse headband with blink / left-gaze / right-gaze detection).
 *
 * - `left`  → rotate counter-clockwise
 * - `right` → rotate clockwise
 * - `blink` → shoot / fire
 * - `none`  → idle
 */
export type BCIDiscreteCommand = "left" | "right" | "blink" | "none";

/** The two control modes available in Combat3D */
export type Combat3DControlMode = "manual" | "bci_hybrid";

/**
 * Rotation direction derived from an EEG signal.
 * - `"ccw"` = counter-clockwise (left signal)
 * - `"cw"`  = clockwise (right signal)
 * - `null`  = no rotation signal active
 */
export type RotationSignal = "ccw" | "cw" | null;

export interface BCIDiscreteState {
  /** Current rotation signal from EEG left/right */
  rotation: RotationSignal;
  /** True when a blink (shoot) was detected and not yet consumed */
  shoot: boolean;
  /** Whether the BCI websocket is connected */
  connected: boolean;
  /** Latest raw command for debug */
  lastRawCommand: BCIDiscreteCommand;
  /** Timestamp of last received command */
  lastCommandTs: number;
}

const WS_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_BCI_WS_URL) ||
  "ws://localhost:8000/ws/bci";
const RECONNECT_INTERVAL_MS = 3_000;
const SHOOT_HOLD_MS = 200;

/**
 * Hook that connects to the backend BCI websocket and provides
 * rotation + shoot signals for the Combat3D game.
 *
 * Backend payload expected:
 * ```json
 * { "command": "left" | "right" | "blink" | "none", "ts": <epoch_ms> }
 * ```
 */
export function useBCIDiscreteInput(enabled: boolean) {
  const [state, setState] = useState<BCIDiscreteState>({
    rotation: null,
    shoot: false,
    connected: false,
    lastRawCommand: "none",
    lastCommandTs: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shootTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Call after consuming the shoot flag */
  const consumeShoot = useCallback(() => {
    setState((prev) => (prev.shoot ? { ...prev, shoot: false } : prev));
  }, []);

  useEffect(() => {
    if (!enabled) {
      wsRef.current?.close();
      wsRef.current = null;
      setState((p) => ({ ...p, connected: false, rotation: null }));
      return;
    }

    function connect() {
      if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setState((p) => ({ ...p, connected: true }));
      };

      ws.onclose = () => {
        setState((p) => ({ ...p, connected: false, rotation: null }));
        reconnectRef.current = setTimeout(connect, RECONNECT_INTERVAL_MS);
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as {
            command: BCIDiscreteCommand;
            ts?: number;
          };
          const cmd = data.command;
          const ts = data.ts ?? Date.now();

          setState((prev) => {
            const next: BCIDiscreteState = {
              ...prev,
              lastRawCommand: cmd,
              lastCommandTs: ts,
            };

            if (cmd === "left") {
              next.rotation = "ccw";
            } else if (cmd === "right") {
              next.rotation = "cw";
            } else if (cmd === "blink") {
              next.shoot = true;
              if (shootTimerRef.current) clearTimeout(shootTimerRef.current);
              shootTimerRef.current = setTimeout(() => {
                setState((s) => ({ ...s, shoot: false }));
              }, SHOOT_HOLD_MS);
            } else {
              next.rotation = null;
            }

            return next;
          });
        } catch {
          // ignore malformed payloads
        }
      };
    }

    connect();

    return () => {
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (shootTimerRef.current) clearTimeout(shootTimerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled]);

  return { state, consumeShoot };
}
