from ppg.models import PulseDetectionResponse
from ppg.services.bpm.pulse_processor import PulseProcessor

# Initialize the pulse processor service
_pulse_processor = PulseProcessor()


def detect_pulse_frame(frame_bytes: bytes) -> PulseDetectionResponse:
    return _pulse_processor.process_frame(frame_bytes)


def lock_face_and_start_calibration() -> bool:
    return _pulse_processor.lock_current_face()


def reset_pulse_detection() -> None:
    _pulse_processor.reset()


def initialize() -> None:
    """Initialize and warm up the service."""
    _pulse_processor.warmup()
