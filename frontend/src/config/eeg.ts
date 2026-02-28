// EEG Configuration
export const CHANNEL_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"] as const;

export const CHANNEL_COLORS = [
  "#6b7280", // 1 - Grey
  "#9333ea", // 2 - Purple
  "#3b82f6", // 3 - Blue
  "#10b981", // 4 - Green
  "#eab308", // 5 - Yellow
  "#ef4444", // 6 - Red
  "#dc2626", // 7 - Dark Red
  "#7f1d1d", // 8 - Brown
] as const;

// Color scale from scientific EEG visualization
export const BRAIN_GRADIENT_COLORS = [
  "#ff00ff", // -500: Magenta (Min)
  "#0000ff", // 0: Blue
  "#0096ff", // 500: Cyan
  "#006464", // 1000: Dark Cyan
  "#00ff96", // 1500: Green-Cyan
  "#00ff00", // 2000: Green
  "#9b9600", // 4000: Yellow-Brown
  "#ff0000", // 9000: Red (Max)
] as const;

// ADS1299 Hardware Limits (Cyton Board)
export const ADS1299_VREF = 4.5;
export const ADS1299_GAIN = 24;
export const ADS1299_MAX_UV = (ADS1299_VREF / ADS1299_GAIN) * 1_000_000;

// Signal Quality Thresholds (per OpenBCI)
export const RAILED_THRESHOLD_PERCENT = 0.9;
export const NEAR_RAILED_THRESHOLD_PERCENT = 0.75;

export const MAX_POINTS = 50000;
export const SAMPLING_RATE = 250;
export const VERT_SCALES = [50, 100, 200, 500, 1000, 2000, 5000] as const;
export const WINDOW_SECONDS = [2, 5, 10, 20, 30] as const;

export const STANDARD_ELECTRODES = [
  // --- CENTRAL (Top) ---
  { id: "Cz", x: 0.0, y: 1.52, z: 0.0 },
  { id: "C3", x: -0.91, y: 0.95, z: 0.0 },
  { id: "C4", x: 0.91, y: 0.95, z: 0.0 },

  // --- FRONTAL (Forehead) ---
  { id: "Fz", x: 0.0, y: 1.15, z: 0.76 },
  { id: "F3", x: -0.64, y: 0.86, z: 0.6 },
  { id: "F4", x: 0.64, y: 0.86, z: 0.6 },
  // F7/F8 are lower and wider
  { id: "F7", x: -0.76, y: 0.6, z: 0.48 },
  { id: "F8", x: 0.76, y: 0.6, z: 0.48 },

  // --- FRONTAL POLE (Eyebrows) ---
  { id: "Fp1", x: -0.37, y: 0.71, z: 0.84 },
  { id: "Fp2", x: 0.37, y: 0.71, z: 0.84 },

  // --- TEMPORAL (Ears) ---
  { id: "T7", x: -0.87, y: 0.35, z: 0.0 }, // Left Ear
  { id: "T8", x: 0.87, y: 0.35, z: 0.0 }, // Right Ear

  // --- PARIETAL (Top Back) ---
  { id: "Pz", x: 0.0, y: 1.05, z: -0.95 },
  { id: "P3", x: -0.7, y: 0.9, z: -0.76 },
  { id: "P4", x: 0.7, y: 0.9, z: -0.76 },
  // P7/P8 (also called T5/T6) are lower back sides
  { id: "P7", x: -0.95, y: 0.67, z: -0.43 },
  { id: "P8", x: 0.95, y: 0.67, z: -0.43 },

  // --- OCCIPITAL (Back Base of Skull) ---
  { id: "Oz", x: 0.0, y: 0.45, z: -1.02 },
  { id: "O1", x: -0.31, y: 0.68, z: -1.0 },
  { id: "O2", x: 0.31, y: 0.68, z: -1.0 },
];
