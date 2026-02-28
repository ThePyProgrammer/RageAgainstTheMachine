from typing import Optional, Tuple
import time

import cv2
import numpy as np

from shared.config.logging import get_logger

from ppg.models import PulseDetectionResponse, Box
from ppg.services.bpm.face_detection import FaceDetector
from ppg.services.bpm.bpm_calculator import SignalProcessor

logger = get_logger(__name__)


class PulseProcessor:
    def __init__(self, bpm_limits: Tuple[int, int] = (50, 160), buffer_size: int = 300):
        """
        Initialize the pulse processor.

        Args:
            bpm_limits: Valid heart rate range (min_bpm, max_bpm)
            buffer_size: Number of samples to buffer for BPM calculation
        """
        self.face_detector = FaceDetector()
        self.signal_processor = SignalProcessor(
            bpm_limits=bpm_limits, buffer_size=buffer_size
        )
        self.t0 = time.time()
        self.last_face_detected_time = 0.0
        self.face_timeout = 2.0  # Reset if no face for 2 seconds
        self.last_detected_face: Optional[Tuple[int, int, int, int]] = None

    def _decode_frame(self, frame_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode frame bytes into an OpenCV image.

        Args:
            frame_bytes: Raw frame bytes

        Returns:
            Decoded image or None if decoding fails
        """
        if not frame_bytes:
            return None

        np_arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return img

    def process_frame(self, frame_bytes: bytes) -> PulseDetectionResponse:
        """
        Process a webcam frame to detect face and calculate BPM.

        Args:
            frame_bytes: Raw frame bytes from webcam

        Returns:
            PulseDetectionResponse with face/forehead boxes, BPM, and calibration status
        """
        current_time = time.time() - self.t0

        # Decode frame
        img = self._decode_frame(frame_bytes)
        if img is None:
            return PulseDetectionResponse(faceDetected=False)

        # Detect faces
        faces = self.face_detector.detect_faces(img)

        # Check if face was lost
        if len(faces) == 0:
            self.last_detected_face = None

            # If face was locked and now lost, check timeout
            if self.face_detector.is_face_locked():
                if current_time - self.last_face_detected_time > self.face_timeout:
                    # Face lost for too long, reset everything
                    self.reset()

            return PulseDetectionResponse(
                faceDetected=False, faceLocked=self.face_detector.is_face_locked()
            )

        # Update last face detected time
        self.last_face_detected_time = current_time

        # Use the first detected face
        x, y, w, h = map(int, faces[0])
        face_box = (x, y, w, h)
        self.last_detected_face = face_box

        # If face is locked, proceed with data collection
        if self.face_detector.is_face_locked():
            # Get forehead ROI
            fx, fy, fw, fh = self.face_detector.get_forehead_roi(face_box)

            # Extract RGB mean intensity from forehead
            # Now returns Tuple[float, float, float]
            rgb_intensity = self.face_detector.extract_forehead_intensity(
                img, (fx, fy, fw, fh)
            )

            # Add sample to buffer
            self.signal_processor.add_sample(rgb_intensity, current_time, frame=img)

            # Get calibration status
            is_calibrating = not self.signal_processor.is_calibration_complete()
            cal_progress, cal_target = self.signal_processor.get_calibration_progress()
            wait_time = self.signal_processor.get_wait_time()

            # Calculate Metrics (BVP, BPM, SpO2, RR)
            current_bvp, current_bpm, current_spo2, current_rr = (
                self.signal_processor.calculate_metrics()
            )

            return PulseDetectionResponse(
                faceDetected=True,
                face=Box(x=x, y=y, w=w, h=h),
                forehead=Box(x=fx, y=fy, w=fw, h=fh),
                bpm=current_bpm if current_bpm is not None else None,
                bvp=current_bvp if current_bvp is not None else None,
                spo2=current_spo2 if current_spo2 is not None else None,
                rr=current_rr if current_rr is not None else None,
                faceLocked=True,
                calibrating=is_calibrating,
                calibrationProgress=cal_progress,
                calibrationTarget=cal_target,
                waitTime=wait_time if is_calibrating else None,
            )
        else:
            # Face detection phase - not locked yet
            # Show both face and forehead boxes
            fx, fy, fw, fh = self.face_detector.get_forehead_roi(face_box)

            return PulseDetectionResponse(
                faceDetected=True,
                face=Box(x=x, y=y, w=w, h=h),
                forehead=Box(x=fx, y=fy, w=fw, h=fh),
                faceLocked=False,
            )

    def lock_current_face(self) -> bool:
        """
        Lock the currently detected face and start calibration.

        Returns:
            True if face was locked successfully, False if no face detected
        """
        if self.last_detected_face is None:
            return False

        # Lock the face``
        self.face_detector.lock_face(self.last_detected_face)

        # Reset buffer to start fresh calibration
        self.signal_processor.reset()

        return True

    def reset(self) -> None:
        """Reset all buffers, face lock, and start time."""
        self.signal_processor.reset()
        self.face_detector.unlock_face()
        self.t0 = time.time()
        self.last_face_detected_time = 0.0
        self.last_detected_face = None

    def warmup(self) -> None:
        """Run a dummy frame to warm up the model."""
        try:
            # Create a dummy image (black)
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.face_detector.detect_faces(dummy_img)
            # No need to process signal as we don't have a face
        except Exception as e:
            logger.warning("PulseProcessor warmup failed: %s", e, exc_info=True)
