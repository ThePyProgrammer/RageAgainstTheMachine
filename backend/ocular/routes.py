from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import logging

from ocular.controller import detect_pupillometry_frame
from ocular.models import PupillometryResponse

router = APIRouter(
    prefix="/ocular",
    tags=["ocular"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger("ocular.routes")
logger.setLevel(logging.INFO)


@router.websocket("/ws/pupillometry")
async def pupillometry_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pupillometry detection.
    """
    await websocket.accept()
    logger.info("DEBUG: /ocular/ws/pupillometry accepted connection")

    try:
        while True:
            # Receive message from WebSocket
            message = await websocket.receive()

            # Parse message data
            frame_bytes = None
            if "bytes" in message:
                frame_bytes = message["bytes"]
            elif "text" in message:
                # Handle text/json messages if any
                pass

            if frame_bytes:
                # Process frame in thread pool to avoid blocking async event loop
                result: PupillometryResponse = await asyncio.to_thread(
                    detect_pupillometry_frame, frame_bytes
                )

                # Send result back
                response = {"type": "pupillometry", "data": result.model_dump()}
                await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket closed with error: {e}")
    finally:
        pass
