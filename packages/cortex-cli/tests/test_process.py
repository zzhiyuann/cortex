"""Tests for cortex_cli.process — PID file management and process utilities."""

import os

import pytest

import cortex_cli.process as process_mod


@pytest.fixture(autouse=True)
def _patch_process_paths(tmp_path, monkeypatch):
    """Redirect PID_DIR and LOG_DIR to temp directory."""
    monkeypatch.setattr(process_mod, "PID_DIR", tmp_path)
    monkeypatch.setattr(process_mod, "LOG_DIR", tmp_path)


# -- pid_file / log_file --------------------------------------------------


class TestPathHelpers:
    """Tests for pid_file() and log_file()."""

    def test_pid_file_returns_correct_path(self, tmp_path):
        """pid_file() returns PID_DIR/<name>.pid."""
        assert process_mod.pid_file("myservice") == tmp_path / "myservice.pid"

    def test_log_file_returns_correct_path(self, tmp_path):
        """log_file() returns LOG_DIR/<name>.log."""
        assert process_mod.log_file("myservice") == tmp_path / "myservice.log"


# -- write_pid / read_pid -------------------------------------------------


class TestWriteAndReadPid:
    """Tests for write_pid() and read_pid()."""

    def test_write_creates_pid_file(self, tmp_path):
        """write_pid() creates a .pid file with the PID as text."""
        process_mod.write_pid("svc", 12345)
        content = (tmp_path / "svc.pid").read_text()
        assert content == "12345"

    def test_read_pid_returns_none_when_no_file(self):
        """read_pid() returns None when no PID file exists."""
        assert process_mod.read_pid("nonexistent") is None

    def test_read_pid_returns_none_for_dead_process(self, tmp_path):
        """read_pid() returns None and cleans up when the PID is dead."""
        pid_path = tmp_path / "dead.pid"
        pid_path.write_text("999999999")  # Very unlikely to be a real PID

        result = process_mod.read_pid("dead")
        assert result is None
        # Stale PID file should be removed
        assert not pid_path.exists()

    def test_read_pid_returns_pid_for_alive_process(self, tmp_path):
        """read_pid() returns the PID if the process is alive (use current PID)."""
        current_pid = os.getpid()
        process_mod.write_pid("alive", current_pid)

        result = process_mod.read_pid("alive")
        assert result == current_pid

    def test_read_pid_handles_invalid_content(self, tmp_path):
        """read_pid() returns None for non-integer PID file content."""
        (tmp_path / "bad.pid").write_text("garbage")
        assert process_mod.read_pid("bad") is None


# -- remove_pid ------------------------------------------------------------


class TestRemovePid:
    """Tests for remove_pid()."""

    def test_removes_existing_pid_file(self, tmp_path):
        """remove_pid() deletes the PID file."""
        process_mod.write_pid("target", 123)
        assert (tmp_path / "target.pid").exists()

        process_mod.remove_pid("target")
        assert not (tmp_path / "target.pid").exists()

    def test_no_error_when_file_missing(self):
        """remove_pid() does not raise when file doesn't exist."""
        process_mod.remove_pid("ghost")  # Should not raise


# -- is_running ------------------------------------------------------------


class TestIsRunning:
    """Tests for is_running()."""

    def test_false_when_no_pid_file(self):
        """is_running() is False when there's no PID file."""
        assert process_mod.is_running("no-such-service") is False

    def test_true_for_alive_process(self, tmp_path):
        """is_running() is True when the PID belongs to a live process."""
        process_mod.write_pid("self", os.getpid())
        assert process_mod.is_running("self") is True

    def test_false_for_dead_process(self, tmp_path):
        """is_running() is False when the PID is stale."""
        (tmp_path / "dead.pid").write_text("999999999")
        assert process_mod.is_running("dead") is False


# -- tail_log --------------------------------------------------------------


class TestTailLog:
    """Tests for tail_log()."""

    def test_returns_empty_when_no_log(self):
        """tail_log() returns [] when the log file doesn't exist."""
        assert process_mod.tail_log("no-service") == []

    def test_returns_last_n_lines(self, tmp_path):
        """tail_log() returns the last N lines from the log."""
        log = tmp_path / "svc.log"
        log.write_text("\n".join(f"line-{i}" for i in range(10)))

        result = process_mod.tail_log("svc", lines=3)
        assert len(result) == 3
        assert result == ["line-7", "line-8", "line-9"]

    def test_returns_all_if_fewer_than_n(self, tmp_path):
        """When log has fewer lines than requested, return all of them."""
        log = tmp_path / "short.log"
        log.write_text("only line")

        result = process_mod.tail_log("short", lines=50)
        assert result == ["only line"]

    def test_empty_log_file(self, tmp_path):
        """An empty log file returns []."""
        log = tmp_path / "empty.log"
        log.write_text("")
        assert process_mod.tail_log("empty") == []


# -- log_size --------------------------------------------------------------


class TestLogSize:
    """Tests for log_size()."""

    def test_returns_zero_when_no_log(self):
        """log_size() returns 0 when the log file doesn't exist."""
        assert process_mod.log_size("missing") == 0

    def test_returns_file_size_in_bytes(self, tmp_path):
        """log_size() returns the correct file size."""
        log = tmp_path / "sized.log"
        data = b"hello world\n"
        log.write_bytes(data)
        assert process_mod.log_size("sized") == len(data)
