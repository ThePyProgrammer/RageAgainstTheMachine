import type { PatternType } from "@/features/pong/types/pongSettings";
import { resolveUiColorToken } from "@/features/pong/types/pongSettings";

interface PatternOptions {
  pattern: PatternType;
  seed: number;
  primaryToken: string;
  secondaryToken?: string;
}

const clamp = (v: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, v));

const mulberry32 = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

export const createPatternTile = ({
  pattern,
  seed,
  primaryToken,
  secondaryToken,
}: PatternOptions): HTMLCanvasElement => {
  const size = 64;
  const tile = document.createElement("canvas");
  tile.width = size;
  tile.height = size;
  const ctx = tile.getContext("2d");
  if (!ctx) {
    return tile;
  }

  const primary = resolveUiColorToken(primaryToken);
  const secondary = resolveUiColorToken(secondaryToken ?? primaryToken);
  const rand = mulberry32(seed);
  const cx = size / 2;
  const cy = size / 2;
  ctx.clearRect(0, 0, size, size);

  if (pattern === "solid") {
    ctx.fillStyle = primary;
    ctx.fillRect(0, 0, size, size);
    return tile;
  }

  if (pattern === "dots") {
    ctx.fillStyle = secondary;
    for (let i = 0; i < 72; i++) {
      const x = rand() * size;
      const y = rand() * size;
      const radius = 1.5 + rand() * 2.5;
      ctx.globalAlpha = 0.35 + rand() * 0.45;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    return tile;
  }

  if (pattern === "rings") {
    const stroke = secondary;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    const rings = 6;
    for (let i = 1; i <= rings; i++) {
      const r = (size / (rings + 1)) * i;
      ctx.globalAlpha = clamp(1 - i / (rings + 2), 0.18, 0.9);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }
    return tile;
  }

  if (pattern === "stripes") {
    ctx.strokeStyle = secondary;
    ctx.lineWidth = 3;
    const step = 10;
    for (let x = -size; x < size * 2; x += step) {
      const offset = Math.floor((rand() - 0.5) * 2);
      ctx.beginPath();
      ctx.moveTo(x + offset, size);
      ctx.lineTo(x + size + offset, 0);
      ctx.stroke();
    }
    return tile;
  }

  if (pattern === "spiral") {
    ctx.strokeStyle = secondary;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    const turns = 4;
    const points = 120;
    for (let i = 0; i < points; i++) {
      const t = i / points;
      const angle = t * Math.PI * 2 * turns + seed * 0.02;
      const radius = t * size;
      const x = cx + Math.cos(angle) * radius * 0.45;
      const y = cy + Math.sin(angle) * radius * 0.45;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    return tile;
  }

  // moire fallback
  ctx.fillStyle = secondary;
  ctx.globalAlpha = 0.15;
  for (let r = 4; r < size; r += 6) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = r % 12 === 0 ? primary : secondary;
    ctx.lineWidth = r % 2 === 0 ? 1 : 1.2;
    ctx.globalAlpha = r % 12 === 0 ? 0.35 : 0.18;
    ctx.stroke();
  }
  return tile;
};

type CacheKey = string;
const MAX_PATTERNS = 10;
const patternCache = new Map<CacheKey, CanvasPattern>();
const lru = new Set<CacheKey>();

const touch = (key: CacheKey) => {
  lru.delete(key);
  lru.add(key);
};

export const getCachedPattern = (
  ctx: CanvasRenderingContext2D,
  opts: PatternOptions,
): CanvasPattern => {
  const key = `${opts.pattern}|${opts.seed}|${opts.primaryToken}|${opts.secondaryToken ?? ""}`;
  const existing = patternCache.get(key);
  if (existing) {
    touch(key);
    return existing;
  }

  while (lru.size >= MAX_PATTERNS) {
    const oldest = lru.values().next().value;
    if (oldest) {
      lru.delete(oldest);
      patternCache.delete(oldest);
    }
  }

  const tile = createPatternTile(opts);
  const pattern = ctx.createPattern(tile, "repeat");
  if (!pattern) {
    const fallback = ctx.createPattern(tile, "repeat");
    if (fallback) {
      patternCache.set(key, fallback);
      touch(key);
      return fallback;
    }
    throw new Error("CanvasPattern creation failed");
  }

  patternCache.set(key, pattern);
  touch(key);
  return pattern;
};
