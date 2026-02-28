from fastapi import APIRouter, UploadFile, File, WebSocket
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import asyncio
import logging

from ppg.controller import (
    detect_pulse_frame,
    lock_face_and_start_calibration,
    reset_pulse_detection,
)
from ppg.models import PulseDetectionResponse
from ppg.services.ppg_service import PPGPipeline

router = APIRouter(
    prefix="/ppg",
    tags=["ppg"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger("ppg.routes")


def _json_ok(payload):
    return JSONResponse(status_code=200, content=jsonable_encoder(payload))


def _json_error(exc: Exception):
    logger.error("PPG route error", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


@router.post("/ppg/detect")
async def detect_pulse(frame: UploadFile = File(...)):
    try:
        data = await frame.read()
        result: PulseDetectionResponse = detect_pulse_frame(data)
        return _json_ok(result)
    except Exception as e:
        return _json_error(e)


@router.post("/ppg/lock")
async def lock_face():
    """Lock the current face position and start calibration."""
    try:
        result = lock_face_and_start_calibration()
        return _json_ok({"success": result})
    except Exception as e:
        return _json_error(e)


@router.post("/ppg/reset")
async def reset_detection():
    """Reset face detection and calibration."""
    try:
        reset_pulse_detection()
        return _json_ok({"success": True})
    except Exception as e:
        return _json_error(e)


@router.websocket("/ws/ppg")
async def pulse_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pulse detection.
    Uses a latest-frame-only policy to prevent backlog and latency.
    """
    await websocket.accept()

    pipeline = PPGPipeline()

    # State for the optimized loop
    queue = asyncio.Queue()  # For commands (lock, reset) - process ALL of them
    latest_frame = {
        "data": None,
        "event": asyncio.Event(),
    }  # For frames - process ONLY NEWEST

    async def receive_loop():
        """Continuously receive data and update state."""
        try:
            while True:
                # Receive message
                message = await websocket.receive()

                # Parse message data
                data = None
                if "json" in message:
                    data = message["json"]
                elif "bytes" in message:
                    data = message["bytes"]
                elif "text" in message:
                    import json

                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        data = message["text"]

                # Handle Data
                if isinstance(data, dict) and "type" in data:
                    # It's a command -> Put in queue
                    await queue.put(data)
                elif isinstance(data, (bytes, bytearray)):
                    # It's a video frame -> Update latest frame
                    latest_frame["data"] = data
                    latest_frame["event"].set()

        except Exception as e:
            logger.info("Receive loop ended: %s", e)
            # Signal processor to stop
            await queue.put(None)
            latest_frame["event"].set()

    async def process_loop():
        """Process commands and latest frames."""
        try:
            while True:
                # 1. PRIORITY: Process all pending commands first
                while not queue.empty():
                    command = await queue.get()
                    if command is None:  # Signal to stop
                        return

                    # Process command
                    processed = await pipeline.process(command)
                    try:
                        await websocket.send_json(processed)
                    except Exception:
                        logger.error("Error sending command response", exc_info=True)
                        return

                # 2. Process latest frame if available
                # Wait for a frame OR a new command
                # We use a timeout so we can keep checking the queue periodically
                # even if no frames come (though frames should be frequent)
                try:
                    await asyncio.wait_for(latest_frame["event"].wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # Check if it was for a frame or we should just loop back to check queue
                if latest_frame["data"] is not None:
                    # Grab frame and clear for next
                    frame_data = latest_frame["data"]
                    latest_frame["data"] = None
                    latest_frame["event"].clear()

                    # Process frame
                    processed = await pipeline.process(frame_data)

                    try:
                        await websocket.send_json(processed)
                    except Exception:
                        logger.error("Error sending frame response", exc_info=True)
                        return

        except Exception:
            logger.error("Process loop error", exc_info=True)

    # Run loops concurrently
    receiver = asyncio.create_task(receive_loop())
    processor = asyncio.create_task(process_loop())

    # Wait for either to finish (likely receiver due to disconnect)
    done, pending = await asyncio.wait(
        [receiver, processor], return_when=asyncio.FIRST_COMPLETED
    )

    # Cancel pending tasks
    for task in pending:
        task.cancel()

    logger.info("Pulse WebSocket connection closed")
