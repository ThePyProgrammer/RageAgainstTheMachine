from ocular.models import PupillometryResponse
from ocular.services.pupillometry.pupillometry_service import PupillometryService

_pupillometry_service = PupillometryService()

_pupillometry_service.warmup()


def detect_pupillometry_frame(frame_bytes: bytes) -> PupillometryResponse:
    """Process a frame for pupillometry detection."""
    return _pupillometry_service.process_frame(frame_bytes)
