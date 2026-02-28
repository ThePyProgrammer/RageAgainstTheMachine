export const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

export const clamp01 = (value: number): number => clamp(value, 0, 1);

export const mapRange = (value: number, fromMin: number, fromMax: number, toMin: number, toMax: number): number => {
  if (fromMax === fromMin) {
    return toMin;
  }

  const ratio = (value - fromMin) / (fromMax - fromMin);
  return toMin + ratio * (toMax - toMin);
};
