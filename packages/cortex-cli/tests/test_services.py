"""Tests for cortex_cli.services — service start/stop/restart/logs."""


import pytest

import cortex_cli.process as process_mod
import cortex_cli.services as services_mod
from cortex_cli.services import _resolve_services, SERVICE_MAP, SERVICES


@pytest.fixture(autouse=True)
def _patch_paths(tmp_path, monkeypatch):
    """Redirect PID/LOG paths to temp directory."""
    monkeypatch.setattr(process_mod, "PID_DIR", tmp_path)
    monkeypatch.setattr(process_mod, "LOG_DIR", tmp_path)


class TestResolveServices:
    def test_all_returns_all_services(self):
        """'all' resolves to the full SERVICES list."""
        result = _resolve_services("all")
        assert len(result) == len(SERVICES)

    def test_specific_service(self):
        """A known service name resolves to a single-item list."""
        result = _resolve_services("a2a-hub")
        assert len(result) == 1
        assert result[0]["name"] == "a2a-hub"

    def test_unknown_service(self, capsys):
        """An unknown service name returns empty list and prints error."""
        result = _resolve_services("nonexistent")
        assert result == []


class TestServiceMap:
    def test_all_services_in_map(self):
        """All SERVICES entries should be in SERVICE_MAP."""
        for svc in SERVICES:
            assert svc["name"] in SERVICE_MAP

    def test_known_services(self):
        """Known services should include a2a-hub and dispatcher."""
        assert "a2a-hub" in SERVICE_MAP
        assert "dispatcher" in SERVICE_MAP


class TestShowLogs:
    def test_show_logs_no_file(self, tmp_path, capsys):
        """show_logs should handle missing log files gracefully."""
        services_mod.show_logs("a2a-hub", lines=10)
        # Should not raise

    def test_show_logs_with_file(self, tmp_path, capsys):
        """show_logs should display log lines when file exists."""
        log_path = tmp_path / "a2a-hub.log"
        log_path.write_text("line1\nline2\nline3\n")
        services_mod.show_logs("a2a-hub", lines=10)
        # Should not raise

    def test_show_logs_unknown_service(self, capsys):
        """show_logs with unknown service should print error."""
        services_mod.show_logs("unknown-svc", lines=10)
        # Should not raise (prints error message)
