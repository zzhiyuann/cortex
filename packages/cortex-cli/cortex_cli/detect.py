"""Component detection — find installed Cortex components automatically.

Supports two modes:
1. Monorepo mode: components live under packages/ in the same repo
2. Filesystem scan: search common project directories (legacy fallback)
"""

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


# Search locations for filesystem scan fallback
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


def _find_monorepo_root() -> Path | None:
    """Find the monorepo root by walking up from this file's location.

    Looks for a directory containing a packages/ subdirectory with a root pyproject.toml.
    """
    current = Path(__file__).resolve().parent
    for _ in range(5):  # max 5 levels up
        packages_dir = current / "packages"
        if packages_dir.is_dir() and (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


def _detect_component_monorepo(name: str, defn: dict, monorepo_root: Path) -> Component:
    """Detect a component within the monorepo packages/ directory."""
    comp = Component(name=name)

    # Check each possible directory name under packages/
    for dir_name in defn["dir_names"]:
        candidate = monorepo_root / "packages" / dir_name
        marker = candidate / defn["marker_file"]
        if marker.exists():
            comp.project_dir = candidate
            break

    if not comp.project_dir:
        return comp

    # In monorepo mode, CLI executables live in the shared .venv/bin/
    venv = monorepo_root / ".venv"
    if venv.exists():
        comp.venv_dir = venv

        cli_path = venv / "bin" / defn["cli_name"]
        if cli_path.exists():
            comp.cli_path = cli_path
            comp.installed = True

        mcp_cli = defn.get("mcp_cli")
        if mcp_cli:
            mcp_path = venv / "bin" / mcp_cli
            if mcp_path.exists():
                comp.mcp_path = mcp_path

    # Also check system PATH as fallback
    if not comp.installed:
        system_cli = shutil.which(defn["cli_name"])
        if system_cli:
            comp.cli_path = Path(system_cli)
            comp.installed = True

    if "mcp_args" in defn:
        comp.extras["mcp_args"] = defn["mcp_args"]

    return comp


def detect_all() -> dict[str, Component]:
    """Detect all Cortex components.

    First tries monorepo mode (packages/ directory), then falls back
    to scanning common project directories.
    """
    monorepo_root = _find_monorepo_root()

    results = {}
    for name, defn in COMPONENT_DEFS.items():
        if monorepo_root:
            results[name] = _detect_component_monorepo(name, defn, monorepo_root)
        else:
            results[name] = _detect_component_filesystem(name, defn)
    return results


def _detect_component_filesystem(name: str, defn: dict) -> Component:
    """Detect a single component via filesystem scan (legacy mode)."""
    comp = Component(name=name)

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

    venv = comp.project_dir / ".venv"
    if venv.exists():
        comp.venv_dir = venv

        cli_path = venv / "bin" / defn["cli_name"]
        if cli_path.exists():
            comp.cli_path = cli_path
            comp.installed = True

        mcp_cli = defn.get("mcp_cli")
        if mcp_cli:
            mcp_path = venv / "bin" / mcp_cli
            if mcp_path.exists():
                comp.mcp_path = mcp_path

    if "mcp_args" in defn:
        comp.extras["mcp_args"] = defn["mcp_args"]

    return comp


def get_monorepo_root() -> Path | None:
    """Public accessor for the monorepo root path."""
    return _find_monorepo_root()


def detect_system_python() -> str | None:
    """Find a suitable Python >= 3.11."""
    for name in ("python3.13", "python3.12", "python3.11", "python3"):
        path = shutil.which(name)
        if path:
            return path
    return None
