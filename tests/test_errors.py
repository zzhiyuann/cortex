"""Tests for cortex_cli.errors — error types, logging, reading, and clearing."""

import json

import pytest

import cortex_cli.errors as errors_mod
from cortex_cli.errors import (
    CortexError,
    ConfigError,
    ComponentError,
    NetworkError,
    Severity,
    log_error,
    get_recent_errors,
    clear_error_log,
    notify_error,
)


@pytest.fixture(autouse=True)
def _patch_error_paths(tmp_path, monkeypatch):
    """Redirect error log paths to a temp directory."""
    monkeypatch.setattr(errors_mod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(errors_mod, "ERROR_LOG", tmp_path / "errors.log")


# -- Error classes ---------------------------------------------------------


class TestErrorClasses:
    """Tests for CortexError and its subclasses."""

    def test_cortex_error_base(self):
        """CortexError stores message, component, detail, and timestamp."""
        err = CortexError("something broke", component="forge", detail="stack trace")
        assert str(err) == "something broke"
        assert err.component == "forge"
        assert err.detail == "stack trace"
        assert err.severity == Severity.ERROR
        assert err.timestamp  # should be non-empty ISO string

    def test_config_error_severity(self):
        """ConfigError has WARNING severity."""
        err = ConfigError("bad config")
        assert err.severity == Severity.WARNING

    def test_component_error_severity(self):
        """ComponentError has ERROR severity."""
        err = ComponentError("crashed")
        assert err.severity == Severity.ERROR

    def test_network_error_severity(self):
        """NetworkError has WARNING severity."""
        err = NetworkError("no connection")
        assert err.severity == Severity.WARNING

    def test_default_component_and_detail(self):
        """Default component is 'cortex' and detail is empty."""
        err = CortexError("msg")
        assert err.component == "cortex"
        assert err.detail == ""


# -- log_error -------------------------------------------------------------


class TestLogError:
    """Tests for log_error()."""

    def test_writes_json_line(self, tmp_path):
        """log_error() writes a structured JSONL entry to the error log."""
        err = CortexError("test failure", component="test-suite")
        log_error(err)

        log_path = tmp_path / "errors.log"
        assert log_path.exists()

        entry = json.loads(log_path.read_text().strip())
        assert entry["message"] == "test failure"
        assert entry["component"] == "test-suite"
        assert entry["severity"] == "error"
        assert "timestamp" in entry

    def test_logs_plain_exception(self, tmp_path):
        """log_error() handles non-CortexError exceptions."""
        err = ValueError("bad value")
        log_error(err, component="validator")

        entry = json.loads((tmp_path / "errors.log").read_text().strip())
        assert entry["message"] == "bad value"
        assert entry["component"] == "validator"

    def test_appends_multiple_entries(self, tmp_path):
        """Multiple log_error() calls append to the same file."""
        log_error(CortexError("first"))
        log_error(CortexError("second"))

        lines = [
            l for l in (tmp_path / "errors.log").read_text().strip().split("\n") if l
        ]
        assert len(lines) == 2

    def test_severity_from_subclass(self, tmp_path):
        """Severity in the log entry matches the error subclass."""
        log_error(ConfigError("config issue"))
        entry = json.loads((tmp_path / "errors.log").read_text().strip())
        assert entry["severity"] == "warning"


# -- get_recent_errors -----------------------------------------------------


class TestGetRecentErrors:
    """Tests for get_recent_errors()."""

    def test_returns_empty_when_no_log(self):
        """Returns [] when no error log exists."""
        assert get_recent_errors() == []

    def test_returns_logged_entries(self, tmp_path):
        """Returns entries that were previously logged."""
        log_error(CortexError("err-1"))
        log_error(CortexError("err-2"))

        errors = get_recent_errors()
        assert len(errors) == 2
        messages = [e["message"] for e in errors]
        assert "err-1" in messages
        assert "err-2" in messages

    def test_limit_parameter(self, tmp_path):
        """The limit parameter caps the number of returned entries."""
        for i in range(10):
            log_error(CortexError(f"err-{i}"))

        errors = get_recent_errors(limit=3)
        assert len(errors) == 3
        # Should return the *last* 3 entries
        assert errors[-1]["message"] == "err-9"

    def test_handles_malformed_lines(self, tmp_path):
        """Malformed JSON lines are skipped gracefully."""
        log_path = tmp_path / "errors.log"
        log_path.write_text('not json\n{"message":"valid"}\n')

        errors = get_recent_errors()
        assert len(errors) == 1
        assert errors[0]["message"] == "valid"


# -- clear_error_log -------------------------------------------------------


class TestClearErrorLog:
    """Tests for clear_error_log()."""

    def test_clears_existing_log(self, tmp_path):
        """clear_error_log() empties the error log file."""
        log_error(CortexError("will be cleared"))
        assert len(get_recent_errors()) == 1

        clear_error_log()
        assert get_recent_errors() == []
        # File should still exist but be empty
        assert (tmp_path / "errors.log").exists()
        assert (tmp_path / "errors.log").read_text() == ""

    def test_no_error_when_log_missing(self):
        """clear_error_log() does not raise when file doesn't exist."""
        clear_error_log()  # Should not raise


# -- notify_error ----------------------------------------------------------


class TestNotifyError:
    """Tests for notify_error()."""

    def test_noop_without_credentials(self):
        """notify_error() does nothing when bot_token or chat_id is missing."""
        # Should not raise or do anything
        notify_error(CortexError("error"), bot_token="", chat_id=0)
        notify_error(CortexError("error"), bot_token="some-token", chat_id=0)
        notify_error(CortexError("error"), bot_token="", chat_id=12345)
