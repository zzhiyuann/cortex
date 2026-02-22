"""Unified error handling for Cortex ecosystem.

Provides structured error types, logging, and optional Telegram notification
for critical errors. All Cortex CLI commands should use these error types
so errors are catchable, loggable, and optionally reported.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

LOG_DIR = Path.home() / ".cortex"
ERROR_LOG = LOG_DIR / "errors.log"

logger = logging.getLogger("cortex")


class Severity(str, Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CortexError(Exception):
    """Base exception for all Cortex errors."""

    severity: Severity = Severity.ERROR

    def __init__(self, message: str, *, component: str = "cortex", detail: str = ""):
        super().__init__(message)
        self.component = component
        self.detail = detail
        self.timestamp = datetime.now().isoformat()


class ConfigError(CortexError):
    """Configuration-related errors (missing file, bad format, etc.)."""

    severity = Severity.WARNING


class ComponentError(CortexError):
    """A Cortex component failed (service crash, missing binary, etc.)."""

    severity = Severity.ERROR


class NetworkError(CortexError):
    """Network-related errors (Telegram API, WebSocket, etc.)."""

    severity = Severity.WARNING


def log_error(error: CortexError | Exception, *, component: str = "cortex"):
    """Log an error to the unified error log (~/.cortex/errors.log).

    Appends a structured JSON line for machine-readable error tracking.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "component": getattr(error, "component", component),
        "severity": getattr(error, "severity", Severity.ERROR).value,
        "message": str(error),
        "detail": getattr(error, "detail", ""),
        "traceback": traceback.format_exc() if traceback.format_exc().strip() != "NoneType: None" else "",
    }
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass

    # Also log to Python logger
    level = getattr(logging, entry["severity"].upper(), logging.ERROR)
    logger.log(level, "[%s] %s", entry["component"], entry["message"])


def notify_error(
    error: CortexError | Exception,
    *,
    component: str = "cortex",
    bot_token: str = "",
    chat_id: int = 0,
):
    """Send a critical error notification via Telegram.

    Only sends if bot_token and chat_id are provided. Fails silently
    so error reporting never causes secondary failures.
    """
    if not bot_token or not chat_id:
        return

    message = str(error)
    detail = getattr(error, "detail", "")
    severity = getattr(error, "severity", Severity.ERROR).value

    text = f"[cortex-error] {component} ({severity})\n{message}"
    if detail:
        text += f"\n{detail[:500]}"

    try:
        payload = json.dumps({"chat_id": chat_id, "text": text}).encode()
        req = Request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urlopen(req, timeout=10)
    except (URLError, OSError):
        pass


def get_recent_errors(limit: int = 20) -> list[dict]:
    """Read recent errors from the error log."""
    if not ERROR_LOG.exists():
        return []
    lines = ERROR_LOG.read_text().strip().split("\n")
    errors = []
    for line in lines[-limit:]:
        try:
            errors.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return errors


def clear_error_log():
    """Clear the error log."""
    if ERROR_LOG.exists():
        ERROR_LOG.write_text("")
