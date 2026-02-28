import type { InputSample } from "./types";

export class InputQueue {
  private queue: InputSample[] = [];

  push(input: InputSample): void {
    this.queue.push(input);
    this.queue.sort((left, right) => left.timestamp - right.timestamp);
  }

  drainUpTo(timeMs: number): InputSample {
    if (this.queue.length === 0) {
      return {
        timestamp: timeMs,
        throttle: 0,
        turn: 0,
        fire: false,
      };
    }

    let last = this.queue[0];
    while (this.queue.length > 0 && this.queue[0].timestamp <= timeMs) {
      last = this.queue.shift()!;
    }

    if (!last) {
      const tail = this.queue[this.queue.length - 1];
      return tail ?? { timestamp: timeMs, throttle: 0, turn: 0, fire: false };
    }

    return {
      ...last,
      timestamp: timeMs,
    };
  }

  flushBefore(timeMs: number): void {
    this.queue = this.queue.filter((sample) => sample.timestamp > timeMs);
  }

  clear(): void {
    this.queue = [];
  }

  get length(): number {
    return this.queue.length;
  }
}
