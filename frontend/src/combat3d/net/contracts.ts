import { z } from "zod";

export const joinSessionSchema = z.object({
  sessionId: z.string().uuid(),
  mode: z.enum(["classifier", "features"]),
});

export const joinResponseSchema = z.object({
  sessionId: z.string().uuid(),
  acceptedAt: z.number(),
  activeMode: z.enum(["classifier", "features"]),
  seed: z.number().int().nonnegative(),
});

export const telemetrySchema = z.object({
  timeMs: z.number(),
  mode: z.enum(["classifier", "features"]),
  kind: z.enum(["classifier", "features"]),
  confidence: z.number().min(0).max(1),
  payload: z.record(z.number()),
  sessionId: z.string().uuid(),
});

export const tauntSchema = z.object({
  sessionId: z.string().uuid(),
  tone: z.string().min(1),
  stress: z.number().min(0).max(1),
  timeMs: z.number(),
});

export type JoinSession = z.infer<typeof joinSessionSchema>;
export type JoinResponse = z.infer<typeof joinResponseSchema>;
export type TelemetryPacket = z.infer<typeof telemetrySchema>;
export type TauntPacket = z.infer<typeof tauntSchema>;

export const WS_DEFAULT_PATH = "/combat3d";
export const EV_CONNECT = "combat3d:connect" as const;
export const EV_TELEMETRY = "combat3d:telemetry" as const;
export const EV_TAUNT = "combat3d:taunt" as const;
export const EV_SERVER_CONFIG = "combat3d:session-config" as const;
export const EV_ERROR = "combat3d:error" as const;
