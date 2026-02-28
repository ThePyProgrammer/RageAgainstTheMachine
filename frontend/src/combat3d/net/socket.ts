import { io, type Socket } from "socket.io-client";
import {
  EV_CONNECT,
  EV_SERVER_CONFIG,
  EV_TAUNT,
  EV_TELEMETRY,
  type JoinSession,
  joinSessionSchema,
  joinResponseSchema,
  telemetrySchema,
  tauntSchema,
  type TelemetryPacket,
  type TauntPacket,
  type JoinResponse,
} from "./contracts";
import { z } from "zod";
import type { BCIMode } from "@ragemachine/bci-shared";

interface CombatSocketConfig {
  url: string;
  onTelemetry: (payload: TelemetryPacket) => void;
  onTaunt: (payload: TauntPacket) => void;
  onSession: (payload: JoinResponse) => void;
}

export interface SocketSession {
  socket: Socket;
  connect: (sessionId: string, mode: BCIMode) => void;
  disconnect: () => void;
}

const validateOrThrow = <T>(value: T, schema: z.ZodType<T>): T => schema.parse(value);

export const createCombat3DSocket = (config: CombatSocketConfig): SocketSession => {
  const socket = io(config.url, {
    autoConnect: false,
    path: "/socket.io",
    reconnectionDelay: 500,
    transports: ["websocket"],
  });

  socket.on("connect_error", (err) => {
    console.error("Combat3D socket connect error", err.message);
  });

  socket.on(EV_SERVER_CONFIG, (raw) => {
    const payload = validateOrThrow(raw, joinResponseSchema);
    config.onSession(payload);
  });

  socket.on(EV_TELEMETRY, (raw) => {
    const payload = validateOrThrow(raw, telemetrySchema);
    config.onTelemetry(payload);
  });

  socket.on(EV_TAUNT, (raw) => {
    const payload = validateOrThrow(raw, tauntSchema);
    config.onTaunt(payload);
  });

  const connect = (sessionId: string, mode: BCIMode): void => {
    if (!socket.connected) {
      socket.connect();
    }

    const request: JoinSession = {
      sessionId,
      mode: mode === "classifier" ? "classifier" : "features",
    };
    const payload = validateOrThrow(request, joinSessionSchema);

    socket.emit(EV_CONNECT, payload);
  };

  const disconnect = (): void => {
    socket.removeAllListeners();
    socket.disconnect();
  };

  return {
    socket,
    connect,
    disconnect,
  };
};
