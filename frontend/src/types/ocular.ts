export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PupilData {
  diameter: number;
  center_x: number;
  center_y: number;
  outline_confidence: number;
  confidence?: number;
  major_axis: number;
  minor_axis: number;
  angle: number;
  circumference?: number;
  iris_center_x?: number;
  iris_center_y?: number;
  iris_radius?: number;
}

export interface EyeData {
  detected: boolean;
  region?: Box;
  pupil?: PupilData;
}

export interface PupillometryResponse {
  faceDetected?: boolean;
  face?: Box;
  leftEye?: EyeData;
  rightEye?: EyeData;
  averagePupilDiameter?: number;
  calibrating?: boolean;
  calibrationProgress?: number;
  calibrationTarget?: number;
}
