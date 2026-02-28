from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import asyncio
import logging

from eeg.services.stream_service import (
    get_active_streamer,
    get_active_device_type,
    set_active_device_type,
    get_streamer_for_device,
)
from eeg.services.streaming.device_registry import get_device_config, get_available_devices
from eeg.services.embedding_service import embeddingProcessor
from eeg.models import EmbeddingConfig


# Helper for consistent JSON responses
def json_response(status_code: int, content: dict):
    return JSONResponse(status_code=status_code, content=content)


router = APIRouter(
    prefix="/bci",
    tags=["bci", "streaming"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger("eeg.routes")

# ======================================================
# REST Endpoints
# ======================================================


@router.get("/devices")
async def list_devices():
    """List all available BCI devices."""
    try:
        devices = get_available_devices()
        active = get_active_device_type()
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({
                "devices": devices,
                "active_device": active,
            }),
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"error": str(e)})
        )


@router.get("/devices/{device_type}")
async def get_device_details(device_type: str):
    """Get detailed configuration for a specific device."""
    try:
        config = get_device_config(device_type)
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder(config),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"error": str(e)})
        )


@router.post("/start")
async def start_stream(device_type: str = Query(default=None)):
    """Start EEG streaming from hardware.

    Args:
        device_type: Optional device type to use (e.g. 'cyton', 'muse_v1').
                     If provided, switches the active device before starting.
    """
    try:
        # Switch device if requested
        if device_type is not None:
            try:
                get_device_config(device_type)  # validate
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            set_active_device_type(device_type)

        streamer = get_active_streamer()
        ok, reason = streamer.start()
        if not ok:
            raise HTTPException(status_code=400, detail=f"Failed to start: {reason}")

        # Wait a bit to see if initialization succeeds
        await asyncio.sleep(0.5)

        # Check if stream is still running after initialization
        if not streamer.is_running:
            return JSONResponse(
                status_code=500,
                content=jsonable_encoder(
                    {
                        "error": "Failed to start stream. Please check if your BCI board is powered on and properly connected."
                    }
                ),
            )

        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({
                "status": "streaming",
                "reason": reason,
                "device_type": get_active_device_type(),
            }),
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "BOARD_NOT_READY" in error_msg or "BOARD_NOT_CREATED" in error_msg:
            error_msg = "Unable to connect to the BCI board. Please ensure your board is powered on and properly connected via the dongle."
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"error": error_msg})
        )


@router.post("/stop")
async def stop_stream():
    """Stop EEG streaming."""
    try:
        streamer = get_active_streamer()
        ok, reason = streamer.stop()
        if not ok:
            raise HTTPException(status_code=400, detail=f"Failed to stop: {reason}")
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({"status": "stopped", "reason": reason}),
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"error": str(e)})
        )


@router.get("/status")
async def get_status():
    """Get current streaming status."""
    try:
        streamer = get_active_streamer()
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({
                "is_streaming": streamer.is_running,
                "device_type": get_active_device_type(),
            }),
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"error": str(e)})
        )


@router.get("/details")
async def get_stream_details():
    """Get detailed streaming session information."""
    try:
        streamer = get_active_streamer()
        details = streamer.get_session_details()
        return JSONResponse(status_code=200, content=jsonable_encoder(details))
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"error": str(e)})
        )


# ======================================================
# Classification Endpoints
# ======================================================
@router.post("/classification/embeddings/configure")
async def configure_embeddings(config: EmbeddingConfig):
    """Configure embedding processing."""
    try:
        streamer = get_active_streamer()
        if config.enabled:
            embeddingProcessor.load_model(config.checkpoint_path)
            embeddingProcessor.channel_names = config.channel_names
            embeddingProcessor.channel_mapping = config.channel_mapping
            embeddingProcessor.channel_indices = (
                embeddingProcessor._compute_channel_indices()
            )
            embeddingProcessor.num_channels = len(embeddingProcessor.channel_names)
            embeddingProcessor.enable()

            # Connect to streamer
            streamer.embedding_processor = embeddingProcessor
            streamer.enable_embeddings = True
            logger.info(
                "Embedding processor connected with %s channels",
                len(embeddingProcessor.channel_names),
            )
            logger.debug("Channel mapping: %s", embeddingProcessor.channel_mapping)
        else:
            embeddingProcessor.disable()
            streamer.enable_embeddings = False
            logger.info("Embedding processor disconnected from streamer")

        return json_response(200, {"status": "configured", "enabled": config.enabled})
    except Exception as e:
        logger.error("Failed to configure embeddings", exc_info=True)
        return json_response(500, {"error": str(e)})


@router.get("/classification/embeddings/latest")
async def get_latest_embedding():
    """Get the most recent embedding from the stream."""
    try:
        embedding = embeddingProcessor.get_latest_embedding()
        if embedding is None:
            return json_response(404, {"error": "No embeddings available"})

        return json_response(
            200,
            {
                "elapsed_time": embeddingProcessor.elapsed_time,
                "raw": {"shape": embedding["raw"]["shape"]},
                "reduced": {
                    "embeddings_2d": embedding["reduced"]["embeddings_2d"],
                    "shape": embedding["reduced"]["shape"],
                },
            },
        )
    except Exception as e:
        return json_response(500, {"error": str(e)})


@router.get("/classification/embeddings/history")
async def get_embedding_history(n: int = None):
    """
    Get embedding history.

    Args:
        n: Number of recent embeddings to return (None = all)
    """
    try:
        history = embeddingProcessor.get_embedding_history(n)
        if not history:
            return json_response(404, {"error": "No embedding history available"})

        summary = [
            {
                "raw": {"shape": emb["raw"]["shape"]},
                "reduced": {
                    "embeddings_2d": emb["reduced"]["embeddings_2d"],
                    "shape": emb["reduced"]["shape"],
                },
            }
            for emb in history
        ]

        return json_response(200, {"count": len(summary), "embeddings": summary})
    except Exception as e:
        return json_response(500, {"error": str(e)})


# ======================================================
# WebSocket Endpoints
# ======================================================


@router.websocket("/ws")
async def stream_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time EEG data streaming."""
    await websocket.accept()
    streamer = get_active_streamer()
    streamer.register_client(websocket)

    try:
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect as exc:
                logger.info("EEG stream client disconnected: code=%s", exc.code)
                break
    except Exception as e:
        logger.error("EEG stream WebSocket error", exc_info=True)
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1011)
    finally:
        streamer.unregister_client(websocket)
        try:
            await websocket.close()
        except Exception:
            pass
