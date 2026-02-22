"""Tests for cortex_cli.health — basic health check utility functions."""

from pathlib import Path

import pytest

from cortex_cli.health import _dir_size_mb, _check_telegram_connectivity, _get_component_version
from cortex_cli.detect import Component


class TestDirSizeMB:
    """Tests for _dir_size_mb()."""

    def test_empty_directory(self, tmp_path):
        """Empty directory has zero size."""
        assert _dir_size_mb(tmp_path) == 0.0

    def test_single_file(self, tmp_path):
        """Directory with one file reports its size in MB."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"x" * 1024)  # 1 KB
        result = _dir_size_mb(tmp_path)
        assert abs(result - 1024 / (1024 * 1024)) < 0.001

    def test_nested_files(self, tmp_path):
        """Nested files are counted recursively."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.txt").write_bytes(b"A" * 500)
        (sub / "b.txt").write_bytes(b"B" * 300)

        result = _dir_size_mb(tmp_path)
        expected = 800 / (1024 * 1024)
        assert abs(result - expected) < 0.001

    def test_nonexistent_directory(self, tmp_path):
        """Non-existent directory returns 0."""
        missing = tmp_path / "does_not_exist"
        assert _dir_size_mb(missing) == 0.0


class TestCheckTelegramConnectivity:
    """Tests for _check_telegram_connectivity()."""

    def test_no_token_returns_skip(self):
        """Empty bot token returns skip status."""
        status, detail = _check_telegram_connectivity("")
        assert status == "skip"
        assert "No bot token" in detail

    def test_invalid_token_returns_fail(self):
        """An obviously invalid token results in a fail status."""
        status, detail = _check_telegram_connectivity("invalid-token-12345")
        # Should fail because Telegram API rejects it or network error
        assert status in ("fail",)


class TestGetComponentVersion:
    """Tests for _get_component_version()."""

    def test_reads_version_from_init(self, tmp_path):
        """Extracts __version__ from a package's __init__.py."""
        pkg_dir = tmp_path / "my_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text('__version__ = "1.2.3"\n')

        comp = Component(name="my-pkg", project_dir=tmp_path)
        version = _get_component_version(comp)
        assert version == "1.2.3"

    def test_reads_version_from_pyproject(self, tmp_path):
        """Falls back to pyproject.toml if no __init__.py version."""
        (tmp_path / "pyproject.toml").write_text('version = "0.5.0"\n')
        comp = Component(name="demo", project_dir=tmp_path)
        version = _get_component_version(comp)
        assert version == "0.5.0"

    def test_no_project_dir_returns_empty(self):
        """Component with no project_dir returns empty string."""
        comp = Component(name="ghost", project_dir=None)
        assert _get_component_version(comp) == ""

    def test_no_version_found_returns_question_mark(self, tmp_path):
        """When neither __init__.py nor pyproject.toml have version, return '?'."""
        comp = Component(name="blank", project_dir=tmp_path)
        assert _get_component_version(comp) == "?"
