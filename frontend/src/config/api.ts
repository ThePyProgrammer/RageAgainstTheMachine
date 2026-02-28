// API Configuration
export const API_BASE_URL = import.meta.env.VITE_BACKEND_URL;
export const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");

// Backend API Endpoints
export const API_ENDPOINTS = {
  BCI_START: `${API_BASE_URL}/bci/start`,
  BCI_STOP: `${API_BASE_URL}/bci/stop`,
  BCI_STATUS: `${API_BASE_URL}/bci/status`,
  BCI_DETAILS: `${API_BASE_URL}/bci/details`,
  BCI_DEVICES: `${API_BASE_URL}/bci/devices`,
  BCI_WS: `${WS_BASE_URL}/bci/ws`,
  BCI_CC_SIGNALS_WS: `${WS_BASE_URL}/bci/ccsignals/ws`,

  // Model Selection & Embeddings
  MODEL_CONFIGURE: `${API_BASE_URL}/bci/classification/embeddings/configure`,
  MODEL_LATEST: `${API_BASE_URL}/bci/classification/embeddings/latest`,
  MODEL_HISTORY: `${API_BASE_URL}/bci/classification/embeddings/history`,

  // Webcam - Pulse Detection (rPPG)
  PULSE_WS: `${WS_BASE_URL}/ppg/ws/ppg`,

  // Webcam - Pupillometry
  PUPILLOMETRY_WS: `${WS_BASE_URL}/ocular/ws/pupillometry`,

  // Motor Imagery
  MI_CONFIGURE: `${API_BASE_URL}/bci/motor_imagery/configure`,
  MI_LATEST: `${API_BASE_URL}/bci/motor_imagery/latest`,
  MI_WS: `${WS_BASE_URL}/mi/ws`,

  // AI Opponent
  OPPONENT_WS: `${WS_BASE_URL}/opponent/ws`,
} as const;

// Helper to build parameterized start URL
export const buildStartUrl = (deviceType?: string): string => {
  const base = API_ENDPOINTS.BCI_START;
  if (deviceType) {
    return `${base}?device_type=${encodeURIComponent(deviceType)}`;
  }
  return base;
};
