export interface HysteresisConfig {
  readonly enterThreshold: number;
  readonly exitThreshold: number;
}

export class BinaryHysteresis {
  private state = false;

  constructor(private readonly config: HysteresisConfig) {
    if (config.exitThreshold >= config.enterThreshold) {
      throw new Error("exitThreshold must be lower than enterThreshold");
    }
  }

  update(level: number): boolean {
    if (!this.state && level >= this.config.enterThreshold) {
      this.state = true;
    } else if (this.state && level <= this.config.exitThreshold) {
      this.state = false;
    }

    return this.state;
  }

  get value(): boolean {
    return this.state;
  }
}
