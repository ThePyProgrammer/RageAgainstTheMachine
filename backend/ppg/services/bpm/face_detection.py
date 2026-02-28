"""
Face detection module for pulse detection.
Handles face detection and forehead ROI extraction.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class FaceDetector:
    """
    Detects faces in webcam frames and extracts forehead regions.
    """

    def __init__(self, cascade_path: Optional[Path] = None):
        """
        Initialize face detector with Haar cascade.

        Args:
            cascade_path: Path to Haar cascade XML. If None, uses default.
        """
        if cascade_path is None:
            cascade_dir = (
                Path(__file__).parent.parent.parent.parent / "shared" / "haarcascades"
            )
            cascade_path = cascade_dir / "haarcascade_frontalface_alt.xml"

        self.cascade_path = cascade_path

        # Verify file exists
        if not self.cascade_path.exists():
            raise FileNotFoundError(
                f"Haar cascade file not found at: {self.cascade_path}\n"
                f"Current file: {__file__}\n"
                f"Calculated path: {cascade_path}"
            )

        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))

        # Detection parameters
        self.scale_factor = 1.3
        self.min_neighbors = 4
        self.min_size = (50, 50)

        self.forehead_x_center = 0.5
        self.forehead_y_center = 0.18
        self.forehead_width_ratio = 0.20
        self.forehead_height_ratio = 0.20

        # Face locking state
        self.face_locked = False
        self.locked_face_box: Optional[Tuple[int, int, int, int]] = None

        # Face tracking for stability
        self.last_center = np.array([0, 0])
        self.last_face: Optional[Tuple[int, int, int, int]] = None
        self.shift_threshold = 4

    def detect_faces(self, img: np.ndarray) -> list:
        """
        Detect faces in an image.

        Args:
            img: Input image (BGR format)

        Returns:
            List of detected face bounding boxes [(x, y, w, h), ...]
        """

        if self.face_locked and self.locked_face_box is not None:
            return [self.locked_face_box]

        # Run detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detected = list(
            self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        )

        if len(detected) == 0:
            # No faces found
            # If locked, return the last known locked position (preserve lock)
            if self.face_locked and self.locked_face_box is not None:
                return [self.locked_face_box]

            # Otherwise return last valid face if available (stabilization)
            if self.last_face is not None:
                return [self.last_face]
            return []

        # If locked, find the face closest to the locked position
        if self.face_locked and self.locked_face_box is not None:
            lx, ly, lw, lh = self.locked_face_box
            locked_center = np.array([lx + 0.5 * lw, ly + 0.5 * lh])

            # Find closest face
            closest_face = None
            min_dist = float("inf")

            for face in detected:
                x, y, w, h = face
                center = np.array([x + 0.5 * w, y + 0.5 * h])
                dist = np.linalg.norm(center - locked_center)

                if dist < min_dist:
                    min_dist = dist
                    closest_face = face

            # Update locked position if we found a corresponding face
            # Use same stabilization logic as specific below
            if closest_face is not None:
                x, y, w, h = closest_face
                center = np.array([x + 0.5 * w, y + 0.5 * h])

                self.locked_face_box = closest_face
                return [closest_face]
            else:
                return [self.locked_face_box]

        # Normal mode (not locked) - find largest face
        detected.sort(key=lambda a: a[-1] * a[-2])
        largest_face = detected[-1]

        # Check if face moved significantly (shift > threshold pixels)
        x, y, w, h = largest_face
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        # Only update if moved enough or first detection
        if shift > self.shift_threshold or np.all(self.last_center == 0):
            self.last_center = center
            self.last_face = largest_face
            return [largest_face]
        else:
            # Return previous face position if shift too small (stabilizes detection)
            if self.last_face is not None:
                return [self.last_face]
            return [largest_face]

    def lock_face(self, face: Tuple[int, int, int, int]) -> None:
        """
        Lock onto a detected face to stabilize detection.

        Args:
            face: Face bounding box (x, y, w, h)
        """
        self.face_locked = True
        self.locked_face_box = face

    def unlock_face(self) -> None:
        """Unlock face detection to resume searching for faces."""
        self.face_locked = False
        self.locked_face_box = None
        self.last_center = np.array([0, 0])
        self.last_face = None

    def is_face_locked(self) -> bool:
        """Check if face is currently locked."""
        return self.face_locked

    def get_forehead_roi(
        self, face: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate forehead ROI within a detected face.

        Args:
            face: Face bounding box (x, y, w, h)

        Returns:
            Forehead bounding box (x, y, w, h)
        """
        x, y, w, h = face

        # Calculate forehead dimensions - centered approach
        fx = int(x + w * self.forehead_x_center - (w * self.forehead_width_ratio / 2.0))
        fy = int(
            y + h * self.forehead_y_center - (h * self.forehead_height_ratio / 2.0)
        )
        fw = int(w * self.forehead_width_ratio)
        fh = int(h * self.forehead_height_ratio)

        return fx, fy, fw, fh

    def extract_forehead_intensity(
        self, img: np.ndarray, forehead: Tuple[int, int, int, int]
    ) -> Tuple[float, float, float]:
        """
        Extract mean intensity from forehead region.
        Averages all RGB channels

        Args:
            img: Input image
            forehead: Forehead ROI (x, y, w, h)

        Returns:
            Mean intensity value
        """
        x, y, w, h = forehead

        # Ensure coordinates are within image bounds
        h_img, w_img = img.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        # Extract forehead region and calculate mean of all channels
        subframe = img[y : y + h, x : x + w, :]
        v1 = np.mean(subframe[:, :, 0])  # Blue
        v2 = np.mean(subframe[:, :, 1])  # Green
        v3 = np.mean(subframe[:, :, 2])  # Red

        return (v3, v2, v1)  # Return (R, G, B) tuple. OpenCV uses BGR order.
