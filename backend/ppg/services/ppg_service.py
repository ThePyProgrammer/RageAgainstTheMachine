"""
PPG Pipeline for pulse detection using services from bpm folder.
Handles the complete pulse detection workflow.
"""

import asyncio
import time
from typing import Any, Dict

from shared.config.logging import get_logger

from ppg.services.bpm.pulse_processor import PulseProcessor
from ppg.models import PulseDetectionResponse

logger = get_logger(__name__)


class PPGPipeline:
    """
    Pipeline for processing pulse detection frames through WebSocket.
    Coordinates pulse_processor, face_detection, and bpm_calculator services.
    """

    def __init__(self):
        # Initialize pulse processor (which uses face_detection and bpm_calculator internally)
        self.pulse_processor = PulseProcessor()

        # Performance tracking
        self.processing = False
        self.frames_dropped = 0
        self.processing_times = []

    async def process(self, data: Any) -> Dict[str, Any]:
        """
        Process incoming WebSocket data (commands or frame data).

        Args:
            data: Can be:
                - Dict with 'type' for commands (lock, reset)
                - bytes/bytearray for raw frame data

        Returns:
            Response dict with type and data
        """
        try:
            # Handle JSON commands
            if isinstance(data, dict):
                command_type = data.get("type")
                logger.debug("Received command: %s", command_type)

                if command_type == "lock":
                    success = await asyncio.to_thread(self._lock_face)
                    return {"type": "lock_response", "success": success}

                elif command_type == "reset":
                    self._reset()
                    return {"type": "reset_response", "success": True}

                else:
                    return {
                        "type": "error",
                        "message": f"Unknown command: {command_type}",
                    }

            # Handle binary frame data
            elif isinstance(data, (bytes, bytearray)):
                return await self._process_frame(data)

            else:
                return {
                    "type": "error",
                    "message": f"Unsupported data type: {type(data)}",
                }

        except Exception as e:
            self.processing = False
            return {"type": "error", "message": f"Processing error: {str(e)}"}

    async def _process_frame(self, frame_bytes: bytes) -> Dict[str, Any]:
        """Process a video frame through the pulse detection pipeline."""

        # Frame skipping: drop if still processing previous frame
        if self.processing:
            self.frames_dropped += 1
            return {
                "type": "dropped",
                "message": "Frame dropped - backend busy",
                "data": {"framesDropped": self.frames_dropped},
            }

        self.processing = True
        start_time = time.time()

        try:
            # Process frame through pulse_processor
            # This calls face_detection and bpm_calculator internally
            result: PulseDetectionResponse = await asyncio.to_thread(
                self.pulse_processor.process_frame, frame_bytes
            )

            # Track processing time
            processing_time_ms = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time_ms)

            # Keep last 100 processing times for stats
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]

            # Build response data
            response_data = {
                "faceDetected": result.faceDetected,
                "faceLocked": result.faceLocked,
                "calibrating": result.calibrating,
                "calibrationProgress": result.calibrationProgress,
                "calibrationTarget": result.calibrationTarget,
                "waitTime": result.waitTime,
                "bpm": result.bpm,
                "bvp": result.bvp,
                "spo2": result.spo2,
                "rr": result.rr,
            }

            # Add face box if present
            if result.face:
                response_data["face"] = {
                    "x": result.face.x,
                    "y": result.face.y,
                    "w": result.face.w,
                    "h": result.face.h,
                }

            # Add forehead box if present
            if result.forehead:
                response_data["forehead"] = {
                    "x": result.forehead.x,
                    "y": result.forehead.y,
                    "w": result.forehead.w,
                    "h": result.forehead.h,
                }

            # Calculate average processing time
            avg_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times
                else 0
            )

            return {
                "type": "pulse",
                "data": response_data,
                "meta": {
                    "processing_time_ms": round(processing_time_ms, 2),
                    "avg_processing_time_ms": round(avg_time, 2),
                    "frames_dropped": self.frames_dropped,
                },
            }

        finally:
            self.processing = False

    def _lock_face(self) -> bool:
        """Lock the current face and start calibration."""
        return self.pulse_processor.lock_current_face()

    def _reset(self) -> None:
        """Reset pulse detection and clear all buffers."""
        self.pulse_processor.reset()
        self.frames_dropped = 0
        self.processing_times.clear()
