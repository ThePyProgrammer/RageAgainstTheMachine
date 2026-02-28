export interface DeadzoneOptions {
  readonly threshold: number;
}

export const applyDeadzone = (value: number, threshold = 0.15): number => {
  if (Math.abs(value) <= threshold) {
    return 0;
  }

  const sign = value >= 0 ? 1 : -1;
  return sign * ((Math.abs(value) - threshold) / (1 - threshold));
};
