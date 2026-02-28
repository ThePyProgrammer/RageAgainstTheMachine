import cv2
import numpy as np
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageOps

from tensorflow.keras.models import load_model
from scipy.ndimage import center_of_mass, label, sum as area

from ocular.models import PupillometryResponse, EyeData, PupilData

logger = logging.getLogger(__name__)

# Iris diameter constant in millimeters (for scale reference)
IRIS_DIAMETER_MM = 11.7


class PupillometryService:
    def __init__(self, model_threshold: float = 0.5):
        self.threshold = model_threshold
        model_path = self._get_model_path()
        logger.info(f"Loading meye model from {model_path}")
        self.model = load_model(model_path, compile=False)

        model_input = (
            self.model.input[0]
            if isinstance(self.model.input, list)
            else self.model.input
        )
        self.input_shape = model_input.shape[1:3]  # (height, width)

        # Load Haar Cascade classifiers
        cascade_dir = (
            Path(__file__).parent.parent.parent.parent / "shared" / "haarcascades"
        )
        face_cascade_path = cascade_dir / "haarcascade_frontalface_alt.xml"
        eye_cascade_path = cascade_dir / "haarcascade_eye.xml"

        self.face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
        self.eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))

        if self.face_cascade.empty():
            raise FileNotFoundError(f"Face cascade not found at {face_cascade_path}")
        if self.eye_cascade.empty():
            raise FileNotFoundError(f"Eye cascade not found at {eye_cascade_path}")

        logger.info(f"Meye model loaded. Input shape: {self.input_shape}")
        logger.info(f"Haar Cascades loaded from {cascade_dir}")

    def _get_model_path(self) -> str:
        """Download and cache the meye model."""
        # Store model in external/meye/models directory
        model_dir = (
            Path(__file__).parent.parent.parent.parent / "external" / "meye" / "models"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "meye-2022-01-24.h5"

        if not model_path.exists():
            logger.info(f"Downloading meye model to {model_path}...")
            url = "https://github.com/fabiocarrara/meye/releases/download/v0.1.1/meye-2022-01-24.h5"
            urllib.request.urlretrieve(url, model_path)
            logger.info("Meye model downloaded successfully!")

        return str(model_path)

    def process_frame(self, frame_bytes: bytes) -> PupillometryResponse:
        """
        Process a frame and detect pupils using meye deep learning model.

        Args:
            frame_bytes: Raw frame bytes from webcam

        Returns:
            PupillometryResponse with pupil data
        """
        try:
            # Decode image bytes to numpy array
            nparr = np.frombuffer(frame_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.error("Failed to decode image")
                return PupillometryResponse(faceDetected=False)

            # Process both eyes
            left_eye_data = self._process_eye(image, "left")
            right_eye_data = self._process_eye(image, "right")

            # Calculate average pupil diameter
            diameters = []
            if right_eye_data and right_eye_data.pupil:
                diameters.append(right_eye_data.pupil.diameter)
            if left_eye_data and left_eye_data.pupil:
                diameters.append(left_eye_data.pupil.diameter)

            avg_diameter = sum(diameters) / len(diameters) if diameters else None

            # Face detected if at least one eye was detected
            face_detected = bool(left_eye_data or right_eye_data)

            return PupillometryResponse(
                faceDetected=face_detected,
                leftEye=left_eye_data,
                rightEye=right_eye_data,
                averagePupilDiameter=avg_diameter,
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return PupillometryResponse(faceDetected=False)

    def _process_eye(self, image: np.ndarray, eye_label: str) -> Optional[EyeData]:
        """
        Process a single eye region.

        Args:
            image: Input image (BGR)
            eye_label: "left" or "right"

        Returns:
            EyeData object with pupil information
        """
        try:
            # Extract eye region from image
            eye_region = self._extract_eye_region(image, eye_label)

            if eye_region is None:
                return None

            # Preprocess for model
            preprocessed = self._preprocess_eye(eye_region)

            # Run inference
            predictions = self.model.predict(preprocessed, verbose=0)
            pupil_map, tags = predictions

            # Extract eye and blink confidence
            is_eye, is_blink = tags.squeeze()

            # If not an eye or blink detected, return None
            if is_eye < 0.5:
                return None

            # Compute pupil metrics
            (pupil_y, pupil_x), pupil_area = self._compute_metrics(pupil_map)

            if pupil_area == 0:
                return None

            # Convert pupil area to diameter in mm
            pupil_diameter_mm = self._area_to_diameter_mm(pupil_area, eye_region.shape)

            # Scale coordinates back to original image
            scale_x = eye_region.shape[1] / self.input_shape[1]
            scale_y = eye_region.shape[0] / self.input_shape[0]
            pupil_x_scaled = pupil_x * scale_x
            pupil_y_scaled = pupil_y * scale_y

            logger.info(
                f"👁️ {eye_label}: center=({pupil_x_scaled:.1f}, {pupil_y_scaled:.1f}), "
                f"area={pupil_area:.0f}px², d={pupil_diameter_mm:.2f}mm, "
                f"eye_conf={is_eye:.2f}, blink={is_blink:.2f}"
            )

            # Create pupil data
            pupil_data = PupilData(
                diameter=pupil_diameter_mm,
                center_x=float(pupil_x_scaled),
                center_y=float(pupil_y_scaled),
                outline_confidence=float(is_eye),
                confidence=float(is_eye),
                major_axis=pupil_diameter_mm,
                minor_axis=pupil_diameter_mm,
                angle=0.0,
                iris_center_x=float(pupil_x_scaled),
                iris_center_y=float(pupil_y_scaled),
                iris_radius=float(
                    pupil_diameter_mm / 2 * (IRIS_DIAMETER_MM / pupil_diameter_mm)
                ),
            )

            return EyeData(detected=True, pupil=pupil_data)

        except Exception as e:
            logger.error(f"Error processing {eye_label} eye: {e}")
            return None

    def _detect_face_and_eyes(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect face and extract left/right eye regions using Haar Cascades.

        Returns:
            Tuple of (left_eye_region, right_eye_region) as numpy arrays, or None if not detected
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            logger.debug("No face detected")
            return None, None

        # Use the first (largest) face
        (fx, fy, fw, fh) = faces[0]
        face_roi_gray = gray[fy : fy + fh, fx : fx + fw]
        face_roi_color = image[fy : fy + fh, fx : fx + fw]

        # Detect eyes within the face region
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
        )

        if len(eyes) < 2:
            logger.debug(f"Only {len(eyes)} eye(s) detected, need 2")
            return None, None

        # Sort eyes by x-coordinate (left to right)
        eyes_sorted = sorted(eyes, key=lambda e: e[0])

        # Split face into left and right halves to determine which eye is which
        face_center_x = fw // 2

        left_eye = None
        right_eye = None

        for ex, ey, ew, eh in eyes_sorted:
            eye_center_x = ex + ew // 2

            # Eye on left half of face = right eye (from camera perspective)
            # Eye on right half of face = left eye (from camera perspective)
            if eye_center_x < face_center_x and right_eye is None:
                # Extract eye region in original image coordinates
                right_eye = face_roi_color[ey : ey + eh, ex : ex + ew]
            elif eye_center_x >= face_center_x and left_eye is None:
                # Extract eye region in original image coordinates
                left_eye = face_roi_color[ey : ey + eh, ex : ex + ew]

        return left_eye, right_eye

    def _extract_eye_region(
        self, image: np.ndarray, eye_label: str
    ) -> Optional[np.ndarray]:
        """
        Extract eye region from full image using Haar Cascade detection.

        Args:
            image: Input image (BGR)
            eye_label: "left" or "right"

        Returns:
            Eye region as numpy array, or None if not detected
        """
        left_eye, right_eye = self._detect_face_and_eyes(image)

        if eye_label == "left":
            return left_eye
        else:
            return right_eye

    def _preprocess_eye(self, eye_region: np.ndarray) -> np.ndarray:
        """
        Preprocess eye region for model input.
        Converts to grayscale, resizes to model input shape, normalizes.
        """
        # Convert to PIL Image
        eye_pil = Image.fromarray(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))

        # Convert to grayscale
        eye_gray = ImageOps.grayscale(eye_pil)

        # Resize to model input shape
        eye_resized = eye_gray.resize(self.input_shape)

        # Convert to numpy and normalize
        eye_array = np.array(eye_resized).astype(np.float32) / 255.0

        # Add batch and channel dimensions: (1, H, W, 1)
        eye_array = eye_array[None, :, :, None]

        return eye_array

    def _compute_metrics(self, pupil_map: np.ndarray, nms: bool = True) -> tuple:
        """
        Compute pupil center and area from segmentation map.
        Based on meye's utils.compute_metrics function.
        """
        p = pupil_map.squeeze()

        # Threshold
        p = p > self.threshold

        if nms:
            # Perform non-maximum suppression: keep only largest area
            s = np.ones((3, 3))  # connectivity structure
            p = self._nms_on_area(p, s)

        center = center_of_mass(p)
        pupil_area = p.sum()

        return center, pupil_area

    def _nms_on_area(self, x: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Non-maximum suppression on area.
        Based on meye's utils.nms_on_area function.
        """
        labels, num_labels = label(x, structure=s)

        if num_labels > 1:
            indexes = np.arange(1, num_labels + 1)
            areas = area(x, labels, indexes)

            biggest = max(zip(areas, indexes))[1]
            x[labels != biggest] = 0

        return x

    def _area_to_diameter_mm(
        self, pupil_area_px: float, eye_region_shape: tuple
    ) -> float:
        """
        Convert pupil area in pixels to diameter in millimeters.
        Uses iris diameter as reference scale.

        Assumes:
        - Iris diameter is constant at ~11.7mm
        - Eye region is roughly the size of the iris
        """
        if pupil_area_px == 0:
            return 0.0

        # Calculate pupil radius from area
        pupil_radius_px = np.sqrt(pupil_area_px / np.pi)

        # Estimate scale: assume eye region width corresponds to ~iris diameter
        # This is a rough approximation
        eye_width_px = eye_region_shape[1]
        mm_per_pixel = IRIS_DIAMETER_MM / eye_width_px

        # Convert to mm
        pupil_diameter_mm = pupil_radius_px * 2 * mm_per_pixel

        return float(pupil_diameter_mm)

    def warmup(self):
        """Warm up the model with a dummy frame."""
        try:
            logger.info("Warming up meye model...")
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            _, dummy_bytes = cv2.imencode(".jpg", dummy_img)
            self.process_frame(dummy_bytes.tobytes())
            logger.info("Meye model warmup complete")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
