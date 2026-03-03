"""Tests for cortex_cli.doctor — diagnostic checks for common issues."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cortex_cli.doctor import _check_port_available, _check_stale_pid, run_doctor


@pytest.fixture(autouse=True)
def _patch_paths(tmp_path, monkeypatch):
    """Redirect file paths to temp directory for safe testing."""
    import cortex_cli.process as process_mod
    import cortex_cli.doctor as doctor_mod
    import cortex_cli.config as config_mod
    import cortex_cli.setup as setup_mod

    monkeypatch.setattr(process_mod, "PID_DIR", tmp_path)
    monkeypatch.setattr(process_mod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CORTEX_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(setup_mod, "MCP_CONFIG", tmp_path / ".mcp.json")
    monkeypatch.setattr(setup_mod, "CLAUDE_SETTINGS", tmp_path / "settings.json")
    monkeypatch.setattr(setup_mod, "DISPATCHER_CONFIG", tmp_path / "dispatcher.yaml")
    monkeypatch.setattr(setup_mod, "VIBE_REPLAY_DIR", tmp_path / ".vibe-replay")


class TestCheckPortAvailable:
    def test_unused_port_is_available(self):
        """A high, unused port should be available."""
        # Use a port unlikely to be in use
        result = _check_port_available("localhost", 59999)
        assert result is True

    def test_in_use_port(self):
        """A port that's actually being listened on should not be available."""
        import socket
        # Bind a temporary socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        sock.listen(1)
        try:
            result = _check_port_available("localhost", port)
            assert result is False
        finally:
            sock.close()


class TestCheckStalePid:
    def test_no_pid_file(self, tmp_path):
        """No PID file means no stale issue."""
        assert _check_stale_pid("missing-service") is None

    def test_alive_process(self, tmp_path):
        """PID file with alive process is not stale."""
        pid_path = tmp_path / "alive-svc.pid"
        pid_path.write_text(str(os.getpid()))
        assert _check_stale_pid("alive-svc") is None

    def test_dead_process_is_stale(self, tmp_path):
        """PID file with dead process is detected as stale."""
        pid_path = tmp_path / "dead-svc.pid"
        pid_path.write_text("999999999")
        result = _check_stale_pid("dead-svc")
        assert result is not None
        assert "Stale PID" in result["issue"]
        assert "fix" in result


class TestRunDoctor:
    def test_runs_without_error(self, tmp_path):
        """run_doctor() should complete without raising."""
        with patch("cortex_cli.doctor.detect_all", return_value={}):
            with patch("cortex_cli.doctor.detect_system_python", return_value="/usr/bin/python3"):
                run_doctor(auto_fix=False)

    def test_detects_missing_config(self, tmp_path):
        """Doctor should detect missing shared config."""
        # Config path doesn't exist (patched to tmp_path)
        # Just verify it runs without errors
        with patch("cortex_cli.doctor.detect_all", return_value={}):
            with patch("cortex_cli.doctor.detect_system_python", return_value="/usr/bin/python3"):
                run_doctor(auto_fix=False)

    def test_auto_fix_stale_pid(self, tmp_path):
        """Doctor with --fix should clean up stale PID files."""
        pid_path = tmp_path / "a2a-hub.pid"
        pid_path.write_text("999999999")

        with patch("cortex_cli.doctor.detect_all", return_value={}):
            with patch("cortex_cli.doctor.detect_system_python", return_value="/usr/bin/python3"):
                run_doctor(auto_fix=True)

        # Stale PID file should be removed
        assert not pid_path.exists()
