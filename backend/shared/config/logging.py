import logging
import os

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int | None = None) -> None:
    if level is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = logging._nameToLevel.get(level_name, logging.INFO)

    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
