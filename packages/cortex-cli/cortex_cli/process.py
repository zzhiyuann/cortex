"""Unified process management for Cortex services.

Handles PID file management, process lifecycle, and log access
for all Cortex services (A2A Hub, Dispatcher, Agent).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

PID_DIR = Path.home() / ".cortex"
LOG_DIR = PID_DIR


def pid_file(name: str) -> Path:
    """Get path to PID file for a named service."""
    return PID_DIR / f"{name}.pid"


def log_file(name: str) -> Path:
    """Get path to log file for a named service."""
    return LOG_DIR / f"{name}.log"


def read_pid(name: str) -> int | None:
    """Read PID for a named service. Returns None if not running."""
    path = pid_file(name)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)  # Check if process is alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        path.unlink(missing_ok=True)
        return None


def write_pid(name: str, pid: int):
    """Write PID file for a named service."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    pid_file(name).write_text(str(pid))


def remove_pid(name: str):
    """Remove PID file for a named service."""
    pid_file(name).unlink(missing_ok=True)


def is_running(name: str) -> bool:
    """Check if a named service is running."""
    return read_pid(name) is not None


def stop_process(name: str, timeout: float = 5.0) -> bool:
    """Stop a running service by name. Returns True if it was running.

    Sends SIGTERM first, then SIGKILL after timeout.
    """
    pid = read_pid(name)
    if not pid:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.3)
            except ProcessLookupError:
                break
        else:
            # Force kill if still alive after timeout
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except ProcessLookupError:
        pass

    remove_pid(name)
    return True


def start_service(
    name: str,
    command: list[str],
    cwd: str | Path | None = None,
    env: dict | None = None,
) -> int | None:
    """Start a service as a background process.

    Returns the PID if started, None if already running.
    """
    pid = read_pid(name)
    if pid:
        return None  # Already running

    PID_DIR.mkdir(parents=True, exist_ok=True)
    log = log_file(name)

    with open(log, "a") as logf:
        proc = subprocess.Popen(
            command,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
            cwd=cwd,
            env=env,
        )

    write_pid(name, proc.pid)
    return proc.pid


def tail_log(name: str, lines: int = 50) -> list[str]:
    """Get the last N lines from a service's log file."""
    path = log_file(name)
    if not path.exists():
        return []
    content = path.read_text().strip()
    if not content:
        return []
    all_lines = content.split("\n")
    return all_lines[-lines:]


def log_size(name: str) -> int:
    """Get the size of a service's log file in bytes."""
    path = log_file(name)
    return path.stat().st_size if path.exists() else 0
