import { z } from "zod";

export const calibrationProfileSchema = z.object({
  sessionId: z.string().min(1),
  mean: z.array(z.number()),
  variance: z.array(z.number()),
  updatedAt: z.number(),
});

export type CalibrationProfilePayload = z.infer<typeof calibrationProfileSchema>;

export interface StorageAdapter {
  read(key: string): string | null;
  write(key: string, value: string): void;
  clear(key: string): void;
}

export const localStorageAdapter: StorageAdapter = {
  read: (key) => {
    if (typeof localStorage === "undefined") {
      return null;
    }

    return localStorage.getItem(key);
  },
  write: (key, value) => {
    if (typeof localStorage === "undefined") {
      return;
    }

    localStorage.setItem(key, value);
  },
  clear: (key) => {
    if (typeof localStorage === "undefined") {
      return;
    }

    localStorage.removeItem(key);
  },
};

export const serializeProfile = (profile: CalibrationProfilePayload): string => JSON.stringify(profile);

export const deserializeProfile = (value: string): CalibrationProfilePayload =>
  calibrationProfileSchema.parse(JSON.parse(value));

export const buildProfileKey = (sessionId: string): string => `bci-calibration-${sessionId}`;

export const getSeparationScore = (left: number[], right: number[]): number => {
  if (left.length === 0 || left.length !== right.length) {
    return 0;
  }

  const diffs = left.map((value, index) => value - right[index]);
  const mean = diffs.reduce((sum, value) => sum + value, 0) / diffs.length;
  const variance =
    diffs.reduce((sum, value) => {
      const delta = value - mean;
      return sum + delta * delta;
    }, 0) / diffs.length;

  const std = Math.sqrt(Math.max(variance, Number.EPSILON));
  return Math.abs(mean) / std;
};

export const canTrustCalibration = (left: number[], right: number[], threshold = 1): boolean =>
  getSeparationScore(left, right) >= threshold;

export const saveCalibrationProfile = (
  profile: CalibrationProfilePayload,
  storage: StorageAdapter = localStorageAdapter,
): void => {
  storage.write(buildProfileKey(profile.sessionId), serializeProfile(profile));
};

export const loadCalibrationProfile = (
  sessionId: string,
  storage: StorageAdapter = localStorageAdapter,
): CalibrationProfilePayload | null => {
  const key = buildProfileKey(sessionId);
  const raw = storage.read(key);

  if (!raw) {
    return null;
  }

  return deserializeProfile(raw);
};
