export interface EMAState {
  value: number;
  ready: boolean;
}

export class EMAFilter {
  private alpha: number;
  private state: EMAState;

  constructor(alpha = 0.2, initial = 0) {
    if (alpha <= 0 || alpha > 1) {
      throw new Error("EMA alpha must be between 0 (exclusive) and 1 (inclusive)");
    }

    this.alpha = alpha;
    this.state = { value: initial, ready: false };
  }

  reset(seedValue = 0): void {
    this.state = { value: seedValue, ready: false };
  }

  next(sample: number): number {
    if (!this.state.ready) {
      this.state = { value: sample, ready: true };
      return sample;
    }

    this.state = {
      value: this.alpha * sample + (1 - this.alpha) * this.state.value,
      ready: true,
    };
    return this.state.value;
  }

  get current(): number {
    return this.state.value;
  }
}
