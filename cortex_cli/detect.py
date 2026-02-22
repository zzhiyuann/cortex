"""Component detection — find installed Cortex components automatically."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Component:
    """A detected Cortex component."""

    name: str
    project_dir: Path | None = None
    venv_dir: Path | None = None
    cli_path: Path | None = None
    mcp_path: Path | None = None
    installed: bool = False
    version: str = ""
    extras: dict = field(default_factory=dict)


# Search locations for each component (relative to home or absolute patterns)
SEARCH_PATHS = [
    Path.home() / "projects",
    Path.home() / "code",
    Path.home() / "dev",
    Path.home() / "src",
    Path.home() / "Documents",
    Path.home() / "github",
]

COMPONENT_DEFS = {
    "dispatcher": {
        "dir_names": ["dispatcher", "agent-dispatcher", "cortex-dispatcher"],
        "cli_name": "dispatcher",
        "marker_file": "dispatcher/core.py",
    },
    "a2a-hub": {
        "dir_names": ["a2a-hub", "a2a_hub"],
        "cli_name": "a2a-hub",
        "mcp_cli": "a2a-hub",
        "mcp_args": ["bridge"],
        "marker_file": "a2a_hub/server.py",
    },
    "forge": {
        "dir_names": ["forge", "forge-agent"],
        "cli_name": "forge",
        "mcp_cli": "forge-mcp",
        "marker_file": "forge/engine.py",
    },
    "vibe-replay": {
        "dir_names": ["vibe-replay", "vibe_replay"],
        "cli_name": "vibe-replay",
        "marker_file": "vibe_replay/capture.py",
    },
}


def detect_all() -> dict[str, Component]:
    """Detect all Cortex components.

    Searches common project directories for each component,
    checks for venvs and CLI executables.
    """
    results = {}
    for name, defn in COMPONENT_DEFS.items():
        results[name] = _detect_component(name, defn)
    return results


def _detect_component(name: str, defn: dict) -> Component:
    """Detect a single component."""
    comp = Component(name=name)

    # Search for project directory
    for base in SEARCH_PATHS:
        if not base.exists():
            continue
        for dir_name in defn["dir_names"]:
            candidate = base / dir_name
            marker = candidate / defn["marker_file"]
            if marker.exists():
                comp.project_dir = candidate
                break
        if comp.project_dir:
            break

    if not comp.project_dir:
        return comp

    # Check for venv
    venv = comp.project_dir / ".venv"
    if venv.exists():
        comp.venv_dir = venv

        # Check for CLI executable in venv
        cli_path = venv / "bin" / defn["cli_name"]
        if cli_path.exists():
            comp.cli_path = cli_path
            comp.installed = True

        # Check for MCP executable
        mcp_cli = defn.get("mcp_cli")
        if mcp_cli:
            mcp_path = venv / "bin" / mcp_cli
            if mcp_path.exists():
                comp.mcp_path = mcp_path

    # Store MCP args if any
    if "mcp_args" in defn:
        comp.extras["mcp_args"] = defn["mcp_args"]

    return comp


def detect_system_python() -> str | None:
    """Find a suitable Python >= 3.11."""
    for name in ("python3.13", "python3.12", "python3.11", "python3"):
        path = shutil.which(name)
        if path:
            return path
    return None
