"""A2A Hub Utilities — Shared helpers and constants."""

from __future__ import annotations

import logging
import sys

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
DEFAULT_TASK_TTL = 300  # seconds
HEARTBEAT_INTERVAL = 30  # seconds
HEARTBEAT_TIMEOUT = 60  # seconds — auto-deregister agents silent for this long
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 10
RATE_LIMIT_WINDOW = 10  # seconds — sliding window for rate limiting
RATE_LIMIT_MAX_MESSAGES = 100  # max messages per window per agent
CLEANUP_LOOP_INTERVAL = 10  # seconds — interval for the background cleanup loop
MAX_TASK_RETRIES = 2  # default max retries when an agent disconnects mid-task


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
