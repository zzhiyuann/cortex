"""Installer — registers generated tools into the user's toolkit.

Supports three installation targets:
- MCP: Adds an entry to ~/.mcp.json and creates a server file
- CLI: Installs a script to ~/.local/bin/
- Local: Saves to ~/.forge/tools/ (always done; this handles additional targets)
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from forge.models import InstallResult, InstallTarget, Session


# MCP tools directory for forge-created tools
MCP_TOOLS_DIR = Path.home() / ".claude" / "mcp-tools" / "forge"

# CLI tools directory
CLI_BIN_DIR = Path.home() / ".local" / "bin"


def install(session: Session, target: InstallTarget) -> InstallResult:
    """Install a generated tool to the specified target.

    Args:
        session: A session with generated code and a ToolSpec.
        target: Where to install (mcp, cli, or local).

    Returns:
        InstallResult with success status and details.
    """
    if not session.spec:
        return InstallResult(
            success=False,
            target=target,
            error="No tool spec in session — nothing to install",
        )

    if not session.generated_code:
        return InstallResult(
            success=False,
            target=target,
            error="No generated code in session — generate first",
        )

    try:
        if target == InstallTarget.MCP:
            return _install_mcp(session)
        elif target == InstallTarget.CLI:
            return _install_cli(session)
        else:
            return _install_local(session)
    except Exception as e:
        return InstallResult(
            success=False,
            target=target,
            error=f"Installation failed: {e}",
        )


def uninstall(name: str, target: InstallTarget) -> InstallResult:
    """Uninstall a tool from the specified target.

    Args:
        name: The tool name.
        target: Where the tool is installed.

    Returns:
        InstallResult with success status.
    """
    try:
        if target == InstallTarget.MCP:
            return _uninstall_mcp(name)
        elif target == InstallTarget.CLI:
            return _uninstall_cli(name)
        else:
            return InstallResult(
                success=True,
                target=target,
                message="Local tools are managed by 'forge list/delete'",
            )
    except Exception as e:
        return InstallResult(
            success=False,
            target=target,
            error=f"Uninstallation failed: {e}",
        )


# ---------------------------------------------------------------------------
# MCP Installation
# ---------------------------------------------------------------------------

def _install_mcp(session: Session) -> InstallResult:
    """Install as an MCP server tool."""
    assert session.spec is not None

    MCP_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    # Write the MCP server file
    server_file = MCP_TOOLS_DIR / f"{session.spec.name}.py"
    server_file.write_text(session.generated_code, encoding="utf-8")

    # Update ~/.mcp.json
    mcp_config_path = Path.home() / ".mcp.json"
    config = _load_mcp_config(mcp_config_path)

    server_name = f"forge-{session.spec.name}"
    config.setdefault("mcpServers", {})[server_name] = {
        "command": "python3",
        "args": [str(server_file)],
    }

    _save_mcp_config(mcp_config_path, config)

    return InstallResult(
        success=True,
        install_path=str(server_file),
        target=InstallTarget.MCP,
        message=(
            f"Installed MCP tool '{session.spec.name}' as server '{server_name}'. "
            f"Server file: {server_file}. "
            f"Restart Claude Code to load the new tool."
        ),
    )


def _uninstall_mcp(name: str) -> InstallResult:
    """Remove an MCP server tool."""
    server_file = MCP_TOOLS_DIR / f"{name}.py"
    if server_file.exists():
        server_file.unlink()

    mcp_config_path = Path.home() / ".mcp.json"
    config = _load_mcp_config(mcp_config_path)

    server_name = f"forge-{name}"
    if server_name in config.get("mcpServers", {}):
        del config["mcpServers"][server_name]
        _save_mcp_config(mcp_config_path, config)

    return InstallResult(
        success=True,
        target=InstallTarget.MCP,
        message=f"Uninstalled MCP tool '{name}'",
    )


def _load_mcp_config(path: Path) -> dict:
    """Load the MCP config file, creating it if needed."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"mcpServers": {}}
    return {"mcpServers": {}}


def _save_mcp_config(path: Path, config: dict) -> None:
    """Save the MCP config file."""
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI Installation
# ---------------------------------------------------------------------------

def _install_cli(session: Session) -> InstallResult:
    """Install as a CLI command to ~/.local/bin/."""
    assert session.spec is not None

    CLI_BIN_DIR.mkdir(parents=True, exist_ok=True)

    # Command name: replace underscores with hyphens for CLI convention
    cmd_name = session.spec.name.replace("_", "-")
    script_path = CLI_BIN_DIR / cmd_name

    # Write the script with a shebang
    code = session.generated_code
    if not code.startswith("#!"):
        code = "#!/usr/bin/env python3\n" + code

    script_path.write_text(code, encoding="utf-8")

    # Make executable
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return InstallResult(
        success=True,
        install_path=str(script_path),
        target=InstallTarget.CLI,
        message=(
            f"Installed CLI tool '{cmd_name}' to {script_path}. "
            f"Ensure {CLI_BIN_DIR} is in your PATH."
        ),
    )


def _uninstall_cli(name: str) -> InstallResult:
    """Remove a CLI tool from ~/.local/bin/."""
    cmd_name = name.replace("_", "-")
    script_path = CLI_BIN_DIR / cmd_name

    if script_path.exists():
        script_path.unlink()
        return InstallResult(
            success=True,
            target=InstallTarget.CLI,
            message=f"Uninstalled CLI tool '{cmd_name}' from {script_path}",
        )

    return InstallResult(
        success=True,
        target=InstallTarget.CLI,
        message=f"CLI tool '{cmd_name}' not found in {CLI_BIN_DIR}",
    )


# ---------------------------------------------------------------------------
# Local Installation (just marks as installed)
# ---------------------------------------------------------------------------

def _install_local(session: Session) -> InstallResult:
    """Mark as locally installed (source code is in ~/.forge/tools/)."""
    assert session.spec is not None

    from forge.storage import TOOLS_DIR

    tool_path = TOOLS_DIR / session.spec.name / f"{session.spec.name}.py"

    return InstallResult(
        success=True,
        install_path=str(tool_path),
        target=InstallTarget.LOCAL,
        message=(
            f"Tool '{session.spec.name}' saved to {tool_path}. "
            f"Import with: from {session.spec.name} import {session.spec.name}"
        ),
    )
