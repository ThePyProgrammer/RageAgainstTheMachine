from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from shared.config.app_config import BOARD_SERIAL_PORT, BOARD_TIMEOUT_SEC, DATA_DIR


class BoardManager:
    """Manages BrainFlow board connection and data retrieval."""

    def __init__(self):
        self.board = None
        self.board_id = BoardIds.CYTON_BOARD.value

    def initialize(self):
        """Initialize and start BrainFlow board."""
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        params.serial_port = BOARD_SERIAL_PORT
        params.timeout = BOARD_TIMEOUT_SEC
        params.file = DATA_DIR

        self.board = BoardShim(BoardIds.CYTON_BOARD, params)
        self.board.prepare_session()
        self.board.start_stream()
        return self.board

    def get_channel_info(self):
        """Get channel indices and sampling rate from board."""
        return {
            "sampling_rate": BoardShim.get_sampling_rate(self.board_id),
            "eeg_channels": BoardShim.get_eeg_channels(self.board_id),
            "accel_channels": BoardShim.get_accel_channels(self.board_id),
            "analog_channels": BoardShim.get_analog_channels(self.board_id),
            "ts_idx": BoardShim.get_timestamp_channel(self.board_id),
            "marker_idx": BoardShim.get_marker_channel(self.board_id),
        }

    def get_data(self):
        """Retrieve data from board."""
        if not self.board:
            raise RuntimeError("Board not initialized")
        return self.board.get_board_data()

    def stop(self):
        """Stop board streaming and release session."""
        if self.board:
            self.board.stop_stream()
            self.board.release_session()
