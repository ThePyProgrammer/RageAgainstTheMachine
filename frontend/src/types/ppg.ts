export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PulseData {
  bpm: number;
  timestamp: number;
  faceDetected: boolean;
}

export interface PulseDetectionResponse {
  faceDetected?: boolean;
  face?: Box;
  forehead?: Box;
  bpm?: number;
  faceLocked?: boolean;
  calibrating?: boolean;
  calibrationProgress?: number;
  calibrationTarget?: number;
  waitTime?: number;
  bvp?: number;
  spo2?: number;
  rr?: number;
}

export interface CameraConfig {
  width: number;
  height: number;
}
