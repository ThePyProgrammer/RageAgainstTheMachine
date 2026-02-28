import csv
import os
from datetime import datetime

from shared.config.logging import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages session CSV files and logging."""

    def __init__(self, session_start_time):
        self.session_start_time = session_start_time
        self.file_path = None

    def create_file(self, header):
        """Create CSV file for session data."""
        session_timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        self.file_path = f"data/raw/eeg/session_{session_timestamp}.csv"
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        with open(self.file_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

        logger.info("[Stream] Writing to %s", self.file_path)
        return self.file_path

    def append_rows(self, rows):
        """Append rows to session CSV file."""
        if not self.file_path:
            raise RuntimeError("Session file not created")

        with open(self.file_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)

    def log_end(self):
        """Log session duration and file path."""
        duration = (datetime.now() - self.session_start_time).total_seconds()
        logger.info(
            "[Stream] Session ended. Duration: %.2fs, File: %s",
            duration,
            self.file_path,
        )
