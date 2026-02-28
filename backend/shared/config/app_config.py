import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=False)

# Board config
BOARD_ID = 2
BOARD_SERIAL_PORT = os.getenv("BOARD_SERIAL_PORT")
BOARD_SAMPLING_RATE = 250
BOARD_EEG_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]

# ADS1299 Hardware Limits (Cyton Board)
ADS1299_VREF = 4.5
ADS1299_GAIN = 24
ADS1299_MAX_UV = (ADS1299_VREF / ADS1299_GAIN) * 1_000_000

# Signal Quality Thresholds (per OpenBCI)
RAILED_THRESHOLD_PERCENT = 0.90
NEAR_RAILED_THRESHOLD_PERCENT = 0.75

# Stream config
BOARD_TIMEOUT_SEC = 15
STREAM_INTERVAL_SEC = 0.001
DATA_DIR = "data"
STREAM_PREVIEW_ROWS = 0
RUN_MODELS = False
EEG_DRY_RUN = False

# Recording config
RECORDING_FORMAT = "EDF"
RECORDING_OUTPUT_DIR = "data/raw/eeg"

# Database config
HF_REPO_ID = os.getenv("HF_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")
