"""A2A Hub Utilities — Shared helpers and constants."""

from __future__ import annotations

import logging
import sys

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
DEFAULT_TASK_TTL = 300  # seconds
HEARTBEAT_INTERVAL = 30  # seconds
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 10


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
