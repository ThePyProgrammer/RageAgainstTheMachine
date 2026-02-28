import * as THREE from "three";
import { BRAIN_GRADIENT_COLORS, STANDARD_ELECTRODES } from "@/config/eeg";

// Voltage steps corresponding to gradient colors (non-linear)
const VOLTAGE_STEPS = [-500, 0, 500, 1000, 1500, 2000, 4000, 9000];

// Reusable color objects to avoid GC pressure
const colorTemp = new THREE.Color();
const colorStart = new THREE.Color();
const colorEnd = new THREE.Color();

// Maps a raw EEG microvolt value to a color with proper blending
export const getEEGColor = (value: number): THREE.Color => {
  // Handle out of bounds
  if (value <= VOLTAGE_STEPS[0]) return colorTemp.set(BRAIN_GRADIENT_COLORS[0]);
  if (value >= VOLTAGE_STEPS[VOLTAGE_STEPS.length - 1]) {
    return colorTemp.set(
      BRAIN_GRADIENT_COLORS[BRAIN_GRADIENT_COLORS.length - 1],
    );
  }

  // Find which two colors to blend between
  let i = 0;
  while (i < VOLTAGE_STEPS.length - 1 && value > VOLTAGE_STEPS[i + 1]) {
    i++;
  }

  // Calculate interpolation factor (0 to 1) between step i and step i+1
  const lowStep = VOLTAGE_STEPS[i];
  const highStep = VOLTAGE_STEPS[i + 1];
  const t = (value - lowStep) / (highStep - lowStep);

  // Lerp between the two colors
  colorStart.set(BRAIN_GRADIENT_COLORS[i]);
  colorEnd.set(BRAIN_GRADIENT_COLORS[i + 1]);
  colorTemp.copy(colorStart).lerp(colorEnd, t);

  return colorTemp;
};

// Pre-calculates inverse distance weights from each vertex to each electrode with distance thresholding
export const calculateVertexWeights = (
  positions: Float32Array,
  brainScale: number = 0.26,
  centerOffset: THREE.Vector3 = new THREE.Vector3(),
  positionOffset: THREE.Vector3 = new THREE.Vector3(),
  rotationY: number = 0,
  radiusInfluence: number = 4.0, // Higher power = more localized "dots"
): number[][] => {
  const vertexCount = positions.length / 3;
  const weights: number[][] = [];
  const vPos = new THREE.Vector3();

  // Threshold: Vertices further than this from an electrode receive 0 color from it
  const MAX_DISTANCE = 0.5;

  for (let i = 0; i < vertexCount; i++) {
    // Get local vertex position
    vPos.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);

    // Apply the same transformations as the brain model:
    // 1. Subtract center (centering)
    vPos.sub(centerOffset);

    // 2. Apply scale
    vPos.multiplyScalar(brainScale);

    // 3. Apply rotation around Y axis
    if (rotationY !== 0) {
      const cos = Math.cos(rotationY);
      const sin = Math.sin(rotationY);
      const x = vPos.x * cos - vPos.z * sin;
      const z = vPos.x * sin + vPos.z * cos;
      vPos.x = x;
      vPos.z = z;
    }

    // 4. Apply position offset
    vPos.add(positionOffset);

    const vertexWeights = STANDARD_ELECTRODES.map((electrode) => {
      const ePos = new THREE.Vector3(electrode.x, electrode.y, electrode.z);

      const dist = vPos.distanceTo(ePos);

      // Cutoff logic: If the vertex is too far, it's not affected by this sensor
      if (dist > MAX_DISTANCE) return 0;

      // IDW Formula with higher power for localization
      // Adding 0.05 prevents "division by zero" spikes
      return 1 / Math.pow(dist + 0.05, radiusInfluence);
    });

    weights.push(vertexWeights);
  }

  return weights;
};

// Interpolates EEG values across the brain surface using electrode data
export const interpolateVertexValue = (
  vertexWeights: number[],
  electrodeData: Record<string, number>,
): number => {
  let totalWeight = 0;
  let weightedSum = 0;

  for (let i = 0; i < STANDARD_ELECTRODES.length; i++) {
    const weight = vertexWeights[i];
    if (weight === 0) continue; // Ignore electrodes outside the MAX_DISTANCE

    const electrode = STANDARD_ELECTRODES[i];
    const value = electrodeData[electrode.id] ?? 0;

    weightedSum += value * weight;
    totalWeight += weight;
  }

  // If totalWeight is 0, the vertex is far from all sensors
  // Returning 0 ensures it renders the "Baseline" color (usually Blue in your scale)
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
};

// Updates a moving average buffer for a single electrode's history
export const updateHistoryBuffer = (
  buffer: number[],
  newValue: number,
  maxSize: number = 10,
): number => {
  buffer.push(newValue);
  if (buffer.length > maxSize) {
    buffer.shift();
  }

  // Return the average
  return buffer.reduce((a, b) => a + b, 0) / buffer.length;
};
