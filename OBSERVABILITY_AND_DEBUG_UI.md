# OBSERVABILITY_AND_DEBUG_UI.md

Minimal logging and debug overlay for development.

## Client-Side Debug Overlay

File: `/apps/web/src/hooks/useDebugMetrics.ts`

```typescript
import { useEffect, useRef, useState } from 'react';

export function useDebugMetrics() {
  const [fps, setFps] = useState(60);
  const [latency, setLatency] = useState(0);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());

  useEffect(() => {
    const interval = setInterval(() => {
      const now = performance.now();
      const elapsed = now - lastTimeRef.current;
      const currentFps = (frameCountRef.current / elapsed) * 1000;
      setFps(currentFps);
      frameCountRef.current = 0;
      lastTimeRef.current = now;
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const recordFrame = () => {
    frameCountRef.current++;
  };

  const recordLatency = (lat: number) => {
    setLatency(lat);
  };

  return { fps, latency, recordFrame, recordLatency };
}
```

## Server-Side Logging

File: `/apps/server/src/utils/logger.ts`

```typescript
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

class Logger {
  private level: LogLevel = LogLevel.INFO;

  setLevel(level: LogLevel): void {
    this.level = level;
  }

  debug(message: string, ...args: unknown[]): void {
    if (this.level <= LogLevel.DEBUG) {
      console.log(`[DEBUG] ${message}`, ...args);
    }
  }

  info(message: string, ...args: unknown[]): void {
    if (this.level <= LogLevel.INFO) {
      console.log(`[INFO] ${message}`, ...args);
    }
  }

  warn(message: string, ...args: unknown[]): void {
    if (this.level <= LogLevel.WARN) {
      console.warn(`[WARN] ${message}`, ...args);
    }
  }

  error(message: string, ...args: unknown[]): void {
    if (this.level <= LogLevel.ERROR) {
      console.error(`[ERROR] ${message}`, ...args);
    }
  }
}

export const logger = new Logger();
```

## Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| FPS | 60 | < 55 |
| Input latency | < 50ms | > 100ms |
| WebSocket latency | < 20ms | > 50ms |
| Calibration quality | > 1.5 | < 1.0 |
| Stress computation time | < 5ms | > 10ms |
| GPT API latency | < 2s | > 5s |

## Error Tracking

```typescript
// Client-side
window.addEventListener('error', (event) => {
  console.error('[Global Error]', event.error);
  // Optional: send to error tracking service
});

// Server-side
process.on('uncaughtException', (err) => {
  logger.error('Uncaught exception:', err);
  process.exit(1);
});
```
