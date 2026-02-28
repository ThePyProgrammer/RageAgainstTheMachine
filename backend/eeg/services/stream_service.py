import threading
import time
from datetime import datetime

import numpy as np

from shared.config.app_config import (
    EEG_DRY_RUN,
    ADS1299_MAX_UV,
    RAILED_THRESHOLD_PERCENT,
    HF_REPO_ID,
    HF_TOKEN,
)
from shared.config.logging import get_logger

from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes
from eeg.services.streaming.board_manager import BoardManager
from eeg.services.streaming.data_processor import DataProcessor
from eeg.services.streaming.session_manager import SessionManager
from eeg.services.streaming.websocket_broadcaster import WebSocketBroadcaster
from shared.storage.hf_store import upload_to_hf

logger = get_logger(__name__)


class EEGStreamer:
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.session_start_time = None

        self.board_manager = BoardManager()
        self.data_processor = DataProcessor()
        self.session_manager = None
        self.ws_broadcaster = WebSocketBroadcaster()
        self.eeg_buffer = None

        # Embedding processor for LaBraM
        self.embedding_processor = None
        self.enable_embeddings = False
        self.embedding_interval = (
            1600  # Process embeddings every 1600 samples (6.4s at 250Hz)
        )

        # MI processor for motor imagery classification
        self.mi_processor = None
        self.enable_mi = False

    def start(self):
        if self.is_running:
            return False, "already_running"
        self.stop_event.clear()
        self.session_start_time = datetime.now()
        if EEG_DRY_RUN:
            self.is_running = True
            return True, "dry_run"
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        self.is_running = True
        return True, "started"

    def stop(self):
        if not self.is_running:
            return False, "not_running"
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.is_running = False
        return True, "stopped"

    def _stream_loop(self):
        try:
            self.board_manager.initialize()
            self.board_manager.board.config_board("d")
            time.sleep(0.5)

            channel_info = self.board_manager.get_channel_info()
            self.session_manager = SessionManager(self.session_start_time)
            self.session_manager.create_file(self.data_processor.build_header())

            sample_counter = 0
            sampling_rate = channel_info["sampling_rate"]
            eeg_channels = channel_info["eeg_channels"]
            n_channels = len(eeg_channels)

            self.eeg_buffer = np.zeros((n_channels, 1250))
            startup_samples = 0

            logger.info("[Stream] Warming up filters...")

            while not self.stop_event.is_set():
                time.sleep(0.033)

                data = self.board_manager.get_data()
                if data.size == 0:
                    continue

                new_points_count = data.shape[1]
                raw_data = data.copy()
                raw_eeg = raw_data[eeg_channels]

                # RAW-based percent/railed (pre-filter)
                percent_matrix = (np.abs(raw_eeg) / ADS1299_MAX_UV) * 100.0
                railed_threshold = ADS1299_MAX_UV * RAILED_THRESHOLD_PERCENT
                is_railed_matrix_strict = (np.abs(raw_eeg) > railed_threshold) | (
                    raw_eeg == 0
                )

                # Maintain buffer of RAW for filtering window
                self.eeg_buffer = np.hstack(
                    (self.eeg_buffer[:, new_points_count:], raw_eeg)
                )

                if startup_samples < 1250:
                    startup_samples += new_points_count
                    continue

                filter_window = self.eeg_buffer.copy()
                for i in range(n_channels):
                    channel_rail_count = np.sum(is_railed_matrix_strict[i, :])
                    if channel_rail_count > (new_points_count * 0.5):
                        filter_window[i, -new_points_count:] = 0.0
                        continue

                    # Filtering
                    DataFilter.detrend(
                        filter_window[i], DetrendOperations.CONSTANT.value
                    )  # removes dc drift
                    DataFilter.remove_environmental_noise(
                        filter_window[i], sampling_rate, NoiseTypes.FIFTY.value
                    )  # removes electriclaa interference at 50Hz
                    DataFilter.perform_bandpass(
                        filter_window[i],
                        sampling_rate,
                        1.0,
                        50.0,
                        2,
                        FilterTypes.BUTTERWORTH.value,
                        0,
                    )  # removes frequencies outside 1-50Hz since thats where brain activity is

                filtered_chunk = filter_window[:, -new_points_count:]

                # Feed data to embedding processor if enabled
                if self.enable_embeddings and self.embedding_processor:
                    self.embedding_processor.add_samples(filtered_chunk)

                # Feed data to MI processor if enabled
                if self.enable_mi and self.mi_processor:
                    self.mi_processor.add_samples(filtered_chunk)

                # Save RAW only
                rows = self.data_processor.process_batch(
                    raw_data, channel_info, sample_counter
                )
                sample_counter += len(rows)

                # Build broadcast rows with derived metrics appended (not persisted)
                window_len = min(sampling_rate, filter_window.shape[1])
                filtered_last_sec = filter_window[:, -window_len:]
                uvrms_vals = np.sqrt(
                    np.mean(
                        (
                            filtered_last_sec
                            - filtered_last_sec.mean(axis=1, keepdims=True)
                        )
                        ** 2,
                        axis=1,
                    )
                )
                uvrms_vals = [round(float(x), 2) for x in uvrms_vals]

                broadcast_rows = []
                for idx, row in enumerate(rows):
                    railed_flags = is_railed_matrix_strict[:, idx].astype(int).tolist()
                    percents = np.round(percent_matrix[:, idx], 2).tolist()
                    filtered_vals = filtered_chunk[:, idx].tolist()
                    broadcast_rows.append(
                        row + filtered_vals + railed_flags + percents + uvrms_vals
                    )

                self.session_manager.append_rows(rows)
                self.ws_broadcaster.broadcast(broadcast_rows)

        except Exception as exc:
            error_msg = str(exc)
            logger.error("[EEGStreamer] Error: %s", exc, exc_info=True)
            # Broadcast error to all connected clients
            self.ws_broadcaster.broadcast_error(error_msg)
        finally:
            try:
                self.board_manager.stop()
            except Exception as stop_exc:
                logger.error(
                    "[EEGStreamer] Error stopping board: %s", stop_exc, exc_info=True
                )
            if self.session_manager:
                self.session_manager.log_end()
                if self.session_manager.file_path and HF_REPO_ID:
                    try:
                        remote_path = upload_to_hf(
                            self.session_manager.file_path, HF_REPO_ID, HF_TOKEN
                        )
                        logger.info("[Stream] Uploaded to HF: %s", remote_path)
                    except Exception as exc:
                        logger.error(
                            "[Stream] HF upload failed: %s", exc, exc_info=True
                        )
            self.is_running = False

    def register_client(self, websocket):
        self.ws_broadcaster.register_client(websocket)

    def unregister_client(self, websocket):
        self.ws_broadcaster.unregister_client(websocket)


streamer = EEGStreamer()


def get_shared_stream_service():
    return streamer
