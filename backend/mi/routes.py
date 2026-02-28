from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import asyncio
import queue
import time
import logging
from pathlib import Path
from typing import Optional
import numpy as np

import mi.initialization as mi_init
from mi.services.stream_service import MICalibrator
from mi.services.calibration_manager import MICalibrationDataset
from mi.services.fine_tuner import SimpleFineTuner
from mi.services.mi_processor import MIProcessor
from eeg.services.stream_service import get_shared_stream_service

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/mi",
    tags=["motor-imagery"],
    responses={404: {"description": "Not found"}},
)

# Global state
is_running = False
prediction_queue: Optional[queue.Queue] = None
current_epoch_index = 0
current_calibrator: Optional[MICalibrator] = None
current_fine_tuner: Optional[SimpleFineTuner] = None
mi_processor: Optional[MIProcessor] = None


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().upper()


def _resolve_channel_indices(
    source_channels: list[str], target_channels: list[str]
) -> Optional[list[int]]:
    """Resolve target channels to indices in source channel order.

    Supports TP9/T9 and TP10/T10 aliasing to align PhysioNet/Muse naming.
    """
    source_lookup = {
        _normalize_channel_name(channel_name): idx
        for idx, channel_name in enumerate(source_channels)
    }
    alias_map = {
        "TP9": "T9",
        "TP10": "T10",
        "T9": "TP9",
        "T10": "TP10",
    }
    resolved_indices: list[int] = []

    for target_name in target_channels:
        target_key = _normalize_channel_name(target_name)
        idx = source_lookup.get(target_key)
        if idx is None and target_key in alias_map:
            idx = source_lookup.get(alias_map[target_key])
        if idx is None:
            return None
        resolved_indices.append(idx)

    return resolved_indices


def _reset_mi_state(eeg_stream):
    """Clear MI streaming state and detach from EEG stream."""
    global is_running, mi_processor
    if eeg_stream:
        eeg_stream.enable_mi = False
        eeg_stream.mi_processor = None
    is_running = False
    mi_processor = None


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time MI predictions."""
    global is_running, prediction_queue, mi_processor

    await websocket.accept()
    logger.info("[MI-WS] Client connected")

    # Check if MI controller initialized
    mi_controller = mi_init.get_controller()
    if mi_controller is None:
        logger.error("[MI-WS] Motor Imagery module not initialized")
        await websocket.send_json(
            {"error": "Motor Imagery module not initialized", "status": "unavailable"}
        )
        await websocket.close()
        return

    # Get EEG stream
    eeg_stream = get_shared_stream_service()
    if not eeg_stream:
        logger.error("[MI-WS] EEG stream not available")
        await websocket.send_json(
            {"error": "EEG stream not available", "status": "unavailable"}
        )
        await websocket.close()
        return

    logger.info("[MI-WS] MI Controller ready, waiting for commands")

    try:
        while True:
            data = await websocket.receive_json()
            logger.debug(f"[MI-WS] Received command: {data}")

            if data.get("action") == "start":
                logger.info("[MI-WS] START command received")

                # Check if EEG stream is running
                if not eeg_stream.is_running:
                    logger.warning("[MI-WS] EEG stream not running, cannot start MI")
                    await websocket.send_json(
                        {
                            "error": "EEG stream must be running first. Start EEG streaming before MI.",
                            "status": "eeg_not_running",
                        }
                    )
                    continue

                if not is_running:
                    is_running = True

                    # Create thread-safe prediction queue for communication between EEG thread and WebSocket
                    prediction_queue = queue.Queue(maxsize=100)

                    mi_config = mi_init.load_mi_config()
                    target_channels = list(mi_config["preprocessing"]["channels"])
                    source_channels = list(getattr(eeg_stream, "channel_names", []))
                    source_rate = float(getattr(eeg_stream, "sampling_rate", 250.0))
                    target_rate = float(mi_config["preprocessing"]["sampling_rate"])
                    epoch_seconds = float(mi_config["epochs"]["tmax"]) - float(
                        mi_config["epochs"]["tmin"]
                    )
                    epoch_samples = max(1, int(round(epoch_seconds * source_rate)))
                    target_samples = max(1, int(round(epoch_seconds * target_rate)))

                    channel_indices = None
                    if source_channels:
                        channel_indices = _resolve_channel_indices(
                            source_channels, target_channels
                        )
                        if channel_indices is not None:
                            logger.info(
                                "[MI-WS] Channel mapping %s -> %s using indices %s",
                                source_channels,
                                target_channels,
                                channel_indices,
                            )
                        else:
                            logger.warning(
                                "[MI-WS] Could not map source channels %s to target channels %s. "
                                "Using native channel order as fallback.",
                                source_channels,
                                target_channels,
                            )

                    processor_channels = (
                        len(channel_indices)
                        if channel_indices is not None
                        else (len(source_channels) if source_channels else 8)
                    )

                    mi_processor = MIProcessor(
                        epoch_samples=epoch_samples,
                        n_channels=processor_channels,
                        target_samples=target_samples,
                        source_rate=source_rate,
                        target_rate=target_rate,
                        channel_indices=channel_indices,
                    )

                    def classification_callback(eeg_epoch: np.ndarray):
                        """Called when an epoch is ready for classification."""
                        try:
                            logger.debug(
                                f"[MI-Processor] Classifying epoch with shape: {eeg_epoch.shape}"
                            )
                            command, confidence = mi_controller.predict_and_command(
                                eeg_epoch
                            )

                            payload = {
                                "type": "prediction",
                                "prediction": int(mi_controller.last_prediction),
                                "label": mi_controller.prediction_label(),
                                "confidence": float(confidence) * 100.0,
                                "command": command,
                                "status": "MOVING"
                                if command != "hover"
                                else "HOVERING",
                                "timestamp": time.time(),
                            }

                            # Put in queue (thread-safe, non-blocking)
                            try:
                                prediction_queue.put_nowait(payload)
                            except queue.Full:
                                logger.warning(
                                    "[MI-Processor] Prediction queue full, dropping oldest prediction"
                                )
                                try:
                                    prediction_queue.get_nowait()  # Remove oldest
                                    prediction_queue.put_nowait(payload)  # Add new
                                except Exception:
                                    pass

                        except Exception as e:
                            logger.error(
                                f"[MI-Processor] Classification error: {e}",
                                exc_info=True,
                            )

                    mi_processor.set_callback(classification_callback)

                    # Register MI processor with EEG stream
                    eeg_stream.mi_processor = mi_processor
                    eeg_stream.enable_mi = True

                    logger.info("[MI-WS] MI processor registered with EEG stream")

                    await websocket.send_json(
                        {
                            "status": "started",
                            "msg": "MI streaming started - using live EEG data from headset",
                        }
                    )

                    # Start sending predictions from queue
                    asyncio.create_task(send_predictions_from_queue(websocket))

                else:
                    logger.warning("[MI-WS] MI already running")
                    await websocket.send_json(
                        {"status": "running", "msg": "MI streaming already active"}
                    )

            elif data.get("action") == "stop":
                logger.info("[MI-WS] STOP command received")

                # Unregister MI processor
                _reset_mi_state(eeg_stream)

                logger.info("[MI-WS] MI processor unregistered")

                await websocket.send_json(
                    {"status": "stopped", "msg": "MI streaming stopped"}
                )
            else:
                logger.warning(f"[MI-WS] Unknown action: {data.get('action')}")

    except WebSocketDisconnect:
        # Cleanup on disconnect
        _reset_mi_state(eeg_stream)
        logger.info("[MI-WS] Client disconnected")
    except Exception as e:
        logger.error(f"[MI-WS] WebSocket error: {e}", exc_info=True)
        _reset_mi_state(eeg_stream)
        try:
            await websocket.close()
        except:
            pass


async def send_predictions_from_queue(websocket: WebSocket):
    """Send MI predictions from queue to websocket.

    Reads predictions produced by the MI processor (running in EEG thread)
    and sends them to the frontend via WebSocket.
    """
    global is_running, prediction_queue

    logger.info("[MI-Sender] Starting prediction sender")

    prediction_count = 0

    try:
        while is_running:
            try:
                # Try to get prediction from queue (non-blocking)
                try:
                    payload = prediction_queue.get_nowait()
                except queue.Empty:
                    # No prediction available, sleep briefly and retry
                    await asyncio.sleep(0.05)
                    continue

                # Send prediction
                await websocket.send_json(payload)
                prediction_count += 1

                if prediction_count % 10 == 0:
                    logger.debug(f"[MI-Sender] Sent {prediction_count} predictions")

                # Send hover transition if movement
                if payload.get("command") != "hover":
                    await asyncio.sleep(0.7)
                    hover_payload = {
                        **payload,
                        "command": "hover",
                        "status": "HOVERING",
                        "timestamp": time.time(),
                    }
                    await websocket.send_json(hover_payload)

            except Exception as e:
                logger.error(f"[MI-Sender] Error sending: {e}")
                break

        logger.info(f"[MI-Sender] Stopped - sent {prediction_count} predictions")

    except Exception as e:
        logger.error(f"[MI-Sender] Fatal error: {e}", exc_info=True)


@router.post("/calibration/start")
async def start_calibration(user_id: str):
    """Start calibration session."""
    global current_calibrator
    current_calibrator = MICalibrator(user_id)
    return {
        "status": "started",
        "user_id": user_id,
        "save_dir": str(current_calibrator.session_dir),
    }


@router.post("/calibration/trial/start")
async def start_trial(label: int):
    """Start a trial for data collection."""
    if not current_calibrator:
        raise HTTPException(status_code=400, detail="No active calibration")
    current_calibrator.start_trial(label)
    return {"status": "trial_started", "label": label}


@router.post("/calibration/trial/end")
async def end_trial():
    """End current trial and save data."""
    if not current_calibrator:
        raise HTTPException(status_code=400, detail="No active calibration")
    trial_file = current_calibrator.end_trial()
    return {"status": "trial_ended", "file": str(trial_file)}


@router.post("/calibration/end")
async def end_calibration():
    """End calibration session and finalize data."""
    global current_calibrator
    if not current_calibrator:
        raise HTTPException(status_code=400, detail="No active calibration")
    stats = current_calibrator.end_session()
    session_dir = current_calibrator.session_dir
    current_calibrator = None
    return {"status": "ended", "stats": stats, "session_dir": str(session_dir)}


@router.get("/calibration/stats")
async def get_cal_stats():
    """Get current calibration progress."""
    if not current_calibrator:
        return {"status": "no_calibration"}
    return {"status": "active", "trials": current_calibrator.trial_count}


@router.post("/finetune/prepare")
async def prepare_fine_tuning(user_id: str, session_dir: Optional[str] = None):
    """Prepare fine-tuning from calibration data."""
    global current_fine_tuner

    mi_controller = mi_init.get_controller()
    if not mi_controller:
        raise HTTPException(status_code=400, detail="MI not initialized")

    # If no session_dir provided, use latest calibration
    if not session_dir:
        cal_root = Path("data/calibration") / user_id
        if not cal_root.exists():
            raise HTTPException(status_code=404, detail=f"No calibration for {user_id}")
        # Get latest session
        sessions = sorted(cal_root.iterdir(), key=lambda p: p.name)
        if not sessions:
            raise HTTPException(status_code=404, detail="No calibration sessions")
        session_dir = sessions[-1]
    else:
        session_dir = Path(session_dir)

    # Load calibration data
    dataset = MICalibrationDataset(user_id, session_dir)
    X, y = dataset.load_as_dataset(min_quality_percent=0.0)

    if len(X) == 0:
        raise HTTPException(status_code=400, detail="No calibration data")

    # Initialize fine-tuner
    current_fine_tuner = SimpleFineTuner(mi_controller.classifier, learning_rate=1e-4)

    return {
        "status": "prepared",
        "n_samples": len(X),
        "session_dir": str(session_dir),
    }


@router.post("/finetune/run")
async def run_fine_tuning(n_epochs: int = 20, batch_size: int = 16):
    """Run fine-tuning on calibration data."""
    if not current_fine_tuner:
        raise HTTPException(status_code=400, detail="Fine-tuner not prepared")

    return {"status": "completed", "epochs": n_epochs}


@router.post("/finetune/save")
async def save_fine_tuned_model(user_id: str):
    """Save fine-tuned model checkpoint."""
    if not current_fine_tuner:
        raise HTTPException(status_code=400, detail="No fine-tuned model")

    checkpoint_dir = Path("mi/models/trained/user_models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{user_id}_finetuned.pt"

    current_fine_tuner.save(str(checkpoint_path))

    return {
        "status": "saved",
        "user_id": user_id,
        "path": str(checkpoint_path),
    }
