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

const legacyTauntSchema = z.object({
  sessionId: z.string().uuid(),
  tone: z.string().min(1),
  stress: z.number().min(0).max(1),
  timeMs: z.number(),
});

const opponentSpeechSchema = z.object({
  mime_type: z.literal("audio/mpeg").optional(),
  audio_base64: z.string(),
});

const opponentUpdateTauntSchema = z.object({
  type: z.literal("opponent_update").optional(),
  event_id: z.string().min(1).optional(),
  taunt_text: z.string().min(1),
  speech: opponentSpeechSchema.optional(),
  stress: z.number().min(0).max(1).optional(),
  timestamp_ms: z.number().optional(),
  timeMs: z.number().optional(),
  sessionId: z.string().uuid().optional(),
});

export const tauntSchema = z.union([legacyTauntSchema, opponentUpdateTauntSchema]);

export type JoinSession = z.infer<typeof joinSessionSchema>;
export type JoinResponse = z.infer<typeof joinResponseSchema>;
export type TelemetryPacket = z.infer<typeof telemetrySchema>;
export type TauntPacket = z.infer<typeof tauntSchema>;

export interface NormalizedTaunt {
  text: string;
  audioBase64: string;
  stress: number | null;
  timeMs: number;
}

export const normalizeTaunt = (packet: TauntPacket): NormalizedTaunt => {
  if ("tone" in packet) {
    return {
      text: packet.tone,
      audioBase64: "",
      stress: packet.stress,
      timeMs: packet.timeMs,
    };
  }

  return {
    text: packet.taunt_text,
    audioBase64: packet.speech?.audio_base64 ?? "",
    stress: packet.stress ?? null,
    timeMs: packet.timestamp_ms ?? packet.timeMs ?? Date.now(),
  };
};

export const WS_DEFAULT_PATH = "/combat3d";
export const EV_CONNECT = "combat3d:connect" as const;
export const EV_TELEMETRY = "combat3d:telemetry" as const;
export const EV_TAUNT = "combat3d:taunt" as const;
export const EV_SERVER_CONFIG = "combat3d:session-config" as const;
export const EV_ERROR = "combat3d:error" as const;
