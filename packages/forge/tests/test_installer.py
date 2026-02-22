"""Tests for forge.installer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from forge.installer import install, uninstall
from forge.models import (
    InstallResult,
    InstallTarget,
    OutputType,
    Session,
    SessionState,
    ToolParam,
    ToolSpec,
)


def _make_session() -> Session:
    """Create a test session with generated code."""
    session = Session(description="test tool", output_type=OutputType.PYTHON)
    session.spec = ToolSpec(
        name="test_tool",
        description="A test tool.",
        params=[ToolParam(name="x", type_hint="str", description="input")],
        core_logic="return x",
    )
    session.generated_code = 'def test_tool(x: str) -> str:\n    return x\n'
    session.update_state(SessionState.SUCCEEDED)
    return session


class TestInstall:
    def test_no_spec_fails(self):
        session = Session(description="test")
        result = install(session, InstallTarget.LOCAL)
        assert not result.success
        assert "No tool spec" in result.error

    def test_no_code_fails(self):
        session = Session(description="test")
        session.spec = ToolSpec(name="t", description="t", core_logic="pass")
        result = install(session, InstallTarget.LOCAL)
        assert not result.success
        assert "No generated code" in result.error

    def test_local_install(self):
        session = _make_session()
        result = install(session, InstallTarget.LOCAL)
        assert result.success
        assert result.target == InstallTarget.LOCAL
        assert "test_tool" in result.install_path

    def test_cli_install(self):
        session = _make_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("forge.installer.CLI_BIN_DIR", Path(tmpdir)):
                result = install(session, InstallTarget.CLI)
                assert result.success
                assert result.target == InstallTarget.CLI
                # Check file was created
                script = Path(tmpdir) / "test-tool"
                assert script.exists()

    def test_mcp_install(self):
        session = _make_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mcp_config = tmp_path / ".mcp.json"
            mcp_tools = tmp_path / "mcp-tools"

            with (
                patch("forge.installer.MCP_TOOLS_DIR", mcp_tools),
                patch("forge.installer._load_mcp_config", return_value={"mcpServers": {}}),
                patch("forge.installer._save_mcp_config") as mock_save,
            ):
                result = install(session, InstallTarget.MCP)
                assert result.success
                assert result.target == InstallTarget.MCP
                # Check save was called
                assert mock_save.called


class TestUninstall:
    def test_uninstall_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake script
            script = Path(tmpdir) / "test-tool"
            script.write_text("#!/usr/bin/env python3\nprint('hi')\n")

            with patch("forge.installer.CLI_BIN_DIR", Path(tmpdir)):
                result = uninstall("test_tool", InstallTarget.CLI)
                assert result.success
                assert not script.exists()

    def test_uninstall_missing_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("forge.installer.CLI_BIN_DIR", Path(tmpdir)):
                result = uninstall("nonexistent", InstallTarget.CLI)
                assert result.success  # Idempotent
