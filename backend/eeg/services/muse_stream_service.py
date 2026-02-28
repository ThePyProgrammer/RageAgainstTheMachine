"""
Muse LSL Stream Service.

Reads EEG data from a Muse headband via an existing LSL stream
(produced by muselsl, BlueMuse, or similar). Provides the same public
interface as EEGStreamer so it can be swapped in transparently.
"""

import threading
import time
from datetime import datetime

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes

from eeg.services.streaming.device_registry import get_device_config
from eeg.services.streaming.session_manager import SessionManager
from eeg.services.streaming.websocket_broadcaster import WebSocketBroadcaster
from shared.config.app_config import HF_REPO_ID, HF_TOKEN
from shared.config.logging import get_logger
from shared.storage.hf_store import upload_to_hf

logger = get_logger(__name__)

DEVICE_CFG = get_device_config("muse_v1")


class MuseStreamer:
    """LSL-based EEG streamer for the Muse headband."""

    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.session_start_time = None

        self.session_manager = None
        self.ws_broadcaster = WebSocketBroadcaster()
        self.eeg_buffer = None
        self.inlet = None

        # Embedding processor (same interface as EEGStreamer)
        self.embedding_processor = None
        self.enable_embeddings = False

        # MI processor
        self.mi_processor = None
        self.enable_mi = False

        # Device specs
        self.n_eeg = DEVICE_CFG["eeg_channel_count"]
        self.n_aux = DEVICE_CFG["aux_channel_count"]
        self.sampling_rate = DEVICE_CFG["sampling_rate"]
        self.max_uv = DEVICE_CFG["max_uv"]
        self.railed_threshold = DEVICE_CFG["railed_threshold_percent"]
        self.channel_names = DEVICE_CFG["eeg_channel_names"]

    # ------------------------------------------------------------------
    # Public interface (matches EEGStreamer)
    # ------------------------------------------------------------------

    def start(self):
        if self.is_running:
            return False, "already_running"
        self.stop_event.clear()
        self.session_start_time = datetime.now()
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

    def register_client(self, websocket):
        self.ws_broadcaster.register_client(websocket)

    def unregister_client(self, websocket):
        self.ws_broadcaster.unregister_client(websocket)

    # ------------------------------------------------------------------
    # Internal streaming loop
    # ------------------------------------------------------------------

    def _resolve_lsl_stream(self):
        """Resolve the Muse LSL stream. Raises if not found."""
        logger.info("[MuseStreamer] Resolving LSL stream (type=EEG)...")
        streams = resolve_byprop(
            "type",
            DEVICE_CFG["lsl_stream_type"],
            timeout=10.0,
        )
        if not streams:
            raise RuntimeError(
                "No Muse LSL stream found. "
                "Start one with: muselsl stream"
            )
        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        logger.info(
            "[MuseStreamer] Connected to LSL stream: %s",
            streams[0].name(),
        )

    def _build_header(self):
        """Build CSV header for Muse data."""
        header = ["sample_index", "ts_unix_ms", "ts_formatted", "marker"]
        header += [f"eeg_{name}" for name in self.channel_names]
        if self.n_aux > 0:
            header += [f"aux_{name}" for name in DEVICE_CFG.get("aux_channel_names", [])]
        return header

    def _stream_loop(self):
        try:
            self._resolve_lsl_stream()
            self.session_manager = SessionManager(self.session_start_time)
            self.session_manager.create_file(self._build_header())

            sample_counter = 0
            buffer_len = self.sampling_rate * 5  # 5-second filter buffer
            self.eeg_buffer = np.zeros((self.n_eeg, buffer_len))
            startup_samples = 0
            railed_threshold_uv = self.max_uv * self.railed_threshold

            logger.info("[MuseStreamer] Streaming started, warming up filters...")

            while not self.stop_event.is_set():
                # Pull a chunk from LSL (non-blocking with short timeout)
                samples, timestamps = self.inlet.pull_chunk(
                    timeout=0.05, max_samples=64
                )
                if not timestamps:
                    continue

                samples = np.array(samples)   # (num_samples, n_channels)
                timestamps = np.array(timestamps)

                # Muse LSL sends EEG channels (+ possibly AUX as last col)
                n_cols = samples.shape[1]
                eeg_cols = min(n_cols, self.n_eeg)
                raw_eeg = samples[:, :eeg_cols].T  # (n_eeg, num_samples)
                aux_data = samples[:, eeg_cols:].T if n_cols > eeg_cols else np.empty((0, samples.shape[0]))

                new_points_count = raw_eeg.shape[1]

                # Railed detection (pre-filter, on raw data)
                percent_matrix = (np.abs(raw_eeg) / self.max_uv) * 100.0
                is_railed_matrix_strict = (np.abs(raw_eeg) > railed_threshold_uv) | (raw_eeg == 0)

                # Maintain rolling filter buffer
                self.eeg_buffer = np.hstack(
                    (self.eeg_buffer[:, new_points_count:], raw_eeg)
                )

                if startup_samples < buffer_len:
                    startup_samples += new_points_count
                    continue

                # Apply filters on the buffer copy
                filter_window = self.eeg_buffer.copy()
                for i in range(self.n_eeg):
                    channel_rail_count = np.sum(is_railed_matrix_strict[i, :])
                    if channel_rail_count > (new_points_count * 0.5):
                        filter_window[i, -new_points_count:] = 0.0
                        continue

                    DataFilter.detrend(
                        filter_window[i], DetrendOperations.CONSTANT.value
                    )
                    DataFilter.remove_environmental_noise(
                        filter_window[i], self.sampling_rate, NoiseTypes.FIFTY.value
                    )
                    DataFilter.perform_bandpass(
                        filter_window[i],
                        self.sampling_rate,
                        1.0,
                        50.0,
                        2,
                        FilterTypes.BUTTERWORTH.value,
                        0,
                    )

                filtered_chunk = filter_window[:, -new_points_count:]

                # Feed to embedding / MI processors
                if self.enable_embeddings and self.embedding_processor:
                    self.embedding_processor.add_samples(filtered_chunk)
                if self.enable_mi and self.mi_processor:
                    self.mi_processor.add_samples(filtered_chunk)

                # Build CSV rows (raw only)
                rows = []
                for j in range(new_points_count):
                    ts = float(timestamps[j])
                    ts_ms = int(ts * 1000)
                    ts_fmt = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]

                    row = [sample_counter, ts_ms, ts_fmt, 0]
                    row.extend(raw_eeg[:, j].tolist())
                    if aux_data.shape[0] > 0:
                        row.extend(aux_data[:, j].tolist())
                    rows.append(row)
                    sample_counter += 1

                # Compute µVrms (last 1s of filtered data)
                window_len = min(self.sampling_rate, filter_window.shape[1])
                filtered_last_sec = filter_window[:, -window_len:]
                uvrms_vals = np.sqrt(
                    np.mean(
                        (filtered_last_sec - filtered_last_sec.mean(axis=1, keepdims=True)) ** 2,
                        axis=1,
                    )
                )
                uvrms_vals = [round(float(x), 2) for x in uvrms_vals]

                # Build broadcast rows:
                # base(4) + raw_eeg(n_eeg) + aux(n_aux) + filtered(n_eeg) + railed(n_eeg) + percent(n_eeg) + uvrms(n_eeg)
                broadcast_rows = []
                for j, row in enumerate(rows):
                    railed_flags = is_railed_matrix_strict[:, j].astype(int).tolist()
                    percents = np.round(percent_matrix[:, j], 2).tolist()
                    filtered_vals = filtered_chunk[:, j].tolist()
                    broadcast_rows.append(
                        row + filtered_vals + railed_flags + percents + uvrms_vals
                    )

                self.session_manager.append_rows(rows)
                self.ws_broadcaster.broadcast(broadcast_rows)

        except Exception as exc:
            error_msg = str(exc)
            logger.error("[MuseStreamer] Error: %s", exc, exc_info=True)
            self.ws_broadcaster.broadcast_error(error_msg)
        finally:
            if self.inlet:
                try:
                    self.inlet.close_stream()
                except Exception:
                    pass
                self.inlet = None
            if self.session_manager:
                self.session_manager.log_end()
                if self.session_manager.file_path and HF_REPO_ID:
                    try:
                        remote_path = upload_to_hf(
                            self.session_manager.file_path, HF_REPO_ID, HF_TOKEN
                        )
                        logger.info("[MuseStreamer] Uploaded to HF: %s", remote_path)
                    except Exception as exc:
                        logger.error(
                            "[MuseStreamer] HF upload failed: %s", exc, exc_info=True
                        )
            self.is_running = False
