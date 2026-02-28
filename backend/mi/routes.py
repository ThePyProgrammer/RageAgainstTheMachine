import asyncio
import logging
import queue
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

import mi.initialization as mi_init
from eeg.services.stream_service import get_shared_stream_service
from mi.services.calibration_manager import MICalibrationDataset
from mi.services.fine_tuner import SimpleFineTuner
from mi.services.fine_tuning import LightweightFineTuningService
from mi.services.mi_processor import MIProcessor
from mi.services.stream_service import MICalibrator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/mi",
    tags=["motor-imagery"],
    responses={404: {"description": "Not found"}},
)


class MIWsCommand(BaseModel):
    action: str = Field(pattern="^(start|stop)$")
    interval_ms: Optional[int] = Field(default=None, ge=100, le=10000)
    reset: Optional[bool] = None


class CalibrationStartRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)


class TrialStartRequest(BaseModel):
    label: int = Field(ge=0, le=1)


class FineTunePrepareRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    session_dir: Optional[str] = None


class FineTuneRunRequest(BaseModel):
    n_epochs: int = Field(default=12, ge=1, le=200)
    batch_size: int = Field(default=8, ge=1, le=256)
    val_split: float = Field(default=0.2, gt=0.0, lt=0.5)


class FineTuneSaveRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)


# Global MI streaming state
is_running = False
prediction_queue: Optional[queue.Queue] = None
mi_processor: Optional[MIProcessor] = None

# Global calibration/fine-tuning state
current_calibrator: Optional[MICalibrator] = None
calibration_processor: Optional[MIProcessor] = None
current_fine_tuner: Optional[SimpleFineTuner] = None
current_fine_tuning_service: Optional[LightweightFineTuningService] = None
prepared_calibration_X: Optional[np.ndarray] = None
prepared_calibration_y: Optional[np.ndarray] = None
prepared_session_dir: Optional[Path] = None

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


def _reset_mi_state(eeg_stream) -> None:
    """Clear MI streaming state and detach from EEG stream."""
    global is_running, mi_processor
    if eeg_stream:
        eeg_stream.enable_mi = False
        eeg_stream.mi_processor = None
    is_running = False
    mi_processor = None


def _reset_calibration_state(eeg_stream) -> None:
    """Clear calibration capture state and detach from EEG stream."""
    global calibration_processor
    if eeg_stream:
        eeg_stream.enable_calibration = False
        eeg_stream.calibration_processor = None
    calibration_processor = None


def _build_mi_epoch_processor(callback) -> MIProcessor:
    processor = MIProcessor(
        epoch_samples=750,  # 3 seconds @ 250Hz live stream rate
        n_channels=8,  # Cyton stream channels
        target_samples=480,  # 3 seconds @ 160Hz model rate
        source_rate=250.0,
        target_rate=160.0,
    )
    processor.set_callback(callback)
    return processor


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time MI predictions."""
    global is_running, prediction_queue, mi_processor

    await websocket.accept()
    logger.info("[MI-WS] Client connected")

    mi_controller = mi_init.get_controller()
    if mi_controller is None:
        logger.error("[MI-WS] Motor Imagery module not initialized")
        await websocket.send_json(
            {"error": "Motor Imagery module not initialized", "status": "unavailable"}
        )
        await websocket.close()
        return

    eeg_stream = get_shared_stream_service()
    if not eeg_stream:
        logger.error("[MI-WS] EEG stream not available")
        await websocket.send_json(
            {"error": "EEG stream not available", "status": "unavailable"}
        )
        await websocket.close()
        return

    sender_task: Optional[asyncio.Task] = None
    logger.info("[MI-WS] MI controller ready")

    try:
        while True:
            incoming = await websocket.receive_json()
            try:
                command = MIWsCommand.model_validate(incoming)
            except ValidationError as exc:
                await websocket.send_json(
                    {
                        "error": f"Invalid websocket command payload: {exc.errors()}",
                        "status": "invalid_payload",
                    }
                )
                continue

            if command.action == "start":
                logger.info("[MI-WS] START command received")

                if not eeg_stream.is_running:
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
                    continue

                is_running = True
                prediction_queue = queue.Queue(maxsize=100)

                def classification_callback(eeg_epoch: np.ndarray):
                    """Classify epoch and queue prediction payload."""
                    try:
                        move_command, confidence = mi_controller.predict_and_command(
                            eeg_epoch
                        )
                        payload = {
                            "type": "prediction",
                            "prediction": int(mi_controller.last_prediction),
                            "label": mi_controller.prediction_label(),
                            "confidence": float(confidence) * 100.0,
                            "command": move_command,
                            "status": "MOVING"
                            if move_command != "hover"
                            else "HOVERING",
                            "timestamp": time.time(),
                        }
                        try:
                            prediction_queue.put_nowait(payload)
                        except queue.Full:
                            try:
                                prediction_queue.get_nowait()
                                prediction_queue.put_nowait(payload)
                            except Exception:
                                logger.warning(
                                    "[MI-Processor] Dropped prediction due to queue pressure"
                                )
                    except Exception as exc:
                        logger.error(
                            "[MI-Processor] Classification error: %s",
                            exc,
                            exc_info=True,
                        )

                mi_processor = _build_mi_epoch_processor(classification_callback)
                eeg_stream.mi_processor = mi_processor
                eeg_stream.enable_mi = True

                await websocket.send_json(
                    {
                        "status": "started",
                        "msg": "MI streaming started - using live EEG data from headset",
                    }
                )
                if sender_task is None or sender_task.done():
                    sender_task = asyncio.create_task(send_predictions_from_queue(websocket))
            else:
                logger.info("[MI-WS] STOP command received")
                _reset_mi_state(eeg_stream)
                await websocket.send_json(
                    {"status": "stopped", "msg": "MI streaming stopped"}
                )

    except WebSocketDisconnect:
        logger.info("[MI-WS] Client disconnected")
    except Exception as exc:
        logger.error("[MI-WS] WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        _reset_mi_state(eeg_stream)
        if sender_task:
            sender_task.cancel()


async def send_predictions_from_queue(websocket: WebSocket):
    """Send queued MI predictions to the websocket client."""
    global is_running, prediction_queue

    prediction_count = 0
    logger.info("[MI-Sender] Starting prediction sender")
    try:
        while is_running:
            try:
                if prediction_queue is None:
                    await asyncio.sleep(0.05)
                    continue
                payload = prediction_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            except Exception as exc:
                logger.error("[MI-Sender] Queue error: %s", exc, exc_info=True)
                break

            try:
                await websocket.send_json(payload)
                prediction_count += 1
            except Exception as exc:
                logger.error("[MI-Sender] Send error: %s", exc)
                break

            if payload.get("command") != "hover":
                # Emit a short hover transition for UI smoothing
                await asyncio.sleep(0.7)
                hover_payload = {
                    **payload,
                    "command": "hover",
                    "status": "HOVERING",
                    "timestamp": time.time(),
                }
                try:
                    await websocket.send_json(hover_payload)
                except Exception:
                    break

        logger.info("[MI-Sender] Stopped - sent %s predictions", prediction_count)
    except Exception as exc:
        logger.error("[MI-Sender] Fatal sender error: %s", exc, exc_info=True)


@router.post("/calibration/start")
async def start_calibration(payload: CalibrationStartRequest):
    """Start a live calibration capture session."""
    global current_calibrator, calibration_processor

    eeg_stream = get_shared_stream_service()
    if not eeg_stream or not eeg_stream.is_running:
        raise HTTPException(
            status_code=400,
            detail="EEG stream must be running before starting calibration.",
        )

    _reset_calibration_state(eeg_stream)
    current_calibrator = MICalibrator(payload.user_id)
    calibration_processor = _build_mi_epoch_processor(current_calibrator.add_epoch)
    eeg_stream.calibration_processor = calibration_processor
    eeg_stream.enable_calibration = True

    logger.info(
        "[MI-Cal] Calibration session started for user=%s dir=%s",
        payload.user_id,
        current_calibrator.session_dir,
    )

    return {
        "status": "started",
        "user_id": payload.user_id,
        "save_dir": str(current_calibrator.session_dir),
        "instruction": "Look left",
    }


@router.post("/calibration/trial/start")
async def start_trial(payload: TrialStartRequest):
    """Start collecting calibration epochs for one label."""
    if not current_calibrator:
        raise HTTPException(status_code=400, detail="No active calibration session.")

    current_calibrator.start_trial(payload.label)
    return {
        "status": "trial_started",
        "label": payload.label,
        "label_name": current_calibrator.label_name(payload.label),
    }


@router.post("/calibration/trial/end")
async def end_trial():
    """End current calibration trial and persist its epochs."""
    if not current_calibrator:
        raise HTTPException(status_code=400, detail="No active calibration session.")

    try:
        trial_file = current_calibrator.end_trial()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    latest_trial = (
        current_calibrator.session_info["trials"][-1]
        if current_calibrator.session_info["trials"]
        else None
    )
    return {
        "status": "trial_ended",
        "file": str(trial_file) if trial_file else None,
        "trial": latest_trial,
    }


@router.post("/calibration/end")
async def end_calibration():
    """End calibration session and persist metadata."""
    global current_calibrator
    if not current_calibrator:
        raise HTTPException(status_code=400, detail="No active calibration session.")

    eeg_stream = get_shared_stream_service()
    stats = current_calibrator.end_session()
    session_dir = current_calibrator.session_dir
    current_calibrator = None
    _reset_calibration_state(eeg_stream)

    return {"status": "ended", "stats": stats, "session_dir": str(session_dir)}


@router.get("/calibration/stats")
async def get_calibration_stats():
    """Get current calibration progress."""
    if not current_calibrator:
        return {"status": "no_calibration"}
    return {
        "status": "active",
        "trials": current_calibrator.trial_count,
        "is_collecting": current_calibrator.is_collecting,
    }


@router.post("/finetune/prepare")
async def prepare_fine_tuning(payload: FineTunePrepareRequest):
    """Prepare fine-tuning from saved calibration data."""
    global current_fine_tuner, current_fine_tuning_service
    global prepared_calibration_X, prepared_calibration_y, prepared_session_dir

    mi_controller = mi_init.get_controller()
    if not mi_controller:
        raise HTTPException(status_code=400, detail="MI module is not initialized.")

    if payload.session_dir:
        session_dir = Path(payload.session_dir)
    else:
        cal_root = Path("data/calibration") / payload.user_id
        if not cal_root.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No calibration sessions found for user {payload.user_id}.",
            )
        sessions = sorted(cal_root.iterdir(), key=lambda p: p.name)
        if not sessions:
            raise HTTPException(status_code=404, detail="No calibration sessions found.")
        session_dir = sessions[-1]

    dataset = MICalibrationDataset(payload.user_id, session_dir)
    X, y = dataset.load_as_dataset(min_quality_percent=0.0)
    if len(X) == 0:
        raise HTTPException(status_code=400, detail="No calibration data available.")

    current_fine_tuner = SimpleFineTuner(mi_controller.classifier, learning_rate=1e-4)
    current_fine_tuning_service = LightweightFineTuningService(current_fine_tuner)

    try:
        counts = current_fine_tuning_service.validate_dataset(X, y, min_samples_per_class=2)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prepared_calibration_X = X
    prepared_calibration_y = y
    prepared_session_dir = session_dir

    return {
        "status": "prepared",
        "session_dir": str(session_dir),
        "n_samples": int(len(y)),
        "n_left": counts["left_count"],
        "n_right": counts["right_count"],
    }


@router.post("/finetune/run")
async def run_fine_tuning(payload: FineTuneRunRequest):
    """Run lightweight fine-tuning on prepared calibration data."""
    if (
        not current_fine_tuning_service
        or prepared_calibration_X is None
        or prepared_calibration_y is None
    ):
        raise HTTPException(
            status_code=400, detail="Fine-tuner not prepared. Call /finetune/prepare first."
        )

    try:
        summary = await asyncio.to_thread(
            current_fine_tuning_service.run,
            prepared_calibration_X,
            prepared_calibration_y,
            payload.n_epochs,
            payload.batch_size,
            payload.val_split,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("[FineTuning] Training failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Fine-tuning failed.") from exc

    return {
        "status": "completed",
        "summary": {
            "n_samples": summary.n_samples,
            "n_left": summary.n_left,
            "n_right": summary.n_right,
            "n_epochs": summary.n_epochs,
            "batch_size": summary.batch_size,
            "final_loss": summary.final_loss,
            "final_acc": summary.final_acc,
            "final_val_loss": summary.final_val_loss,
            "final_val_acc": summary.final_val_acc,
            "best_val_acc": summary.best_val_acc,
        },
    }


@router.post("/finetune/save")
async def save_fine_tuned_model(payload: FineTuneSaveRequest):
    """Persist and optionally upload the fine-tuned left/right classifier."""
    if not current_fine_tuning_service:
        raise HTTPException(
            status_code=400, detail="No fine-tuned model available. Call /finetune/run first."
        )

    artifacts = current_fine_tuning_service.save_and_optionally_upload(payload.user_id)
    summary = current_fine_tuning_service.last_summary

    return {
        "status": "saved",
        "user_id": payload.user_id,
        "path": artifacts["path"],
        "uploaded": artifacts["uploaded"],
        "remote_path": artifacts["remote_path"],
        "summary": {
            "best_val_acc": summary.best_val_acc if summary else None,
            "final_val_acc": summary.final_val_acc if summary else None,
        },
        "session_dir": str(prepared_session_dir) if prepared_session_dir else None,
    }
