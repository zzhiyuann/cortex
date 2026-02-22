"""Storage — persists tool metadata and source code to ~/.forge/tools/.

Each created tool gets its own directory containing source code, tests,
and a metadata JSON file for tracking and management.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from forge.models import OutputType, Session, ToolMetadata

FORGE_HOME = Path.home() / ".forge"
TOOLS_DIR = FORGE_HOME / "tools"


def ensure_dirs() -> None:
    """Create the forge storage directories if they don't exist."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)


def save_tool(session: Session) -> Path:
    """Save a tool's source code, tests, and metadata from a session.

    Args:
        session: A completed forge session with generated code.

    Returns:
        Path to the tool's directory.
    """
    ensure_dirs()

    if not session.spec:
        raise ValueError("Session has no tool spec — nothing to save")

    tool_dir = TOOLS_DIR / session.spec.name
    tool_dir.mkdir(parents=True, exist_ok=True)

    # Write source code
    ext = ".py"
    tool_file = tool_dir / f"{session.spec.name}{ext}"
    tool_file.write_text(session.generated_code, encoding="utf-8")

    # Write tests
    test_file = tool_dir / f"test_{session.spec.name}{ext}"
    test_file.write_text(session.generated_tests, encoding="utf-8")

    # Write metadata
    metadata = ToolMetadata(
        name=session.spec.name,
        display_name=session.spec.display_name,
        description=session.spec.description,
        output_type=session.output_type,
        session_id=session.id,
        tool_path=str(tool_file),
        test_path=str(test_file),
        installed=session.install_result is not None and session.install_result.success,
        install_path=(
            session.install_result.install_path
            if session.install_result
            else ""
        ),
        test_passed=bool(session.test_results and session.test_results[-1].passed),
        dependencies=session.spec.dependencies,
    )
    meta_file = tool_dir / "metadata.json"
    meta_file.write_text(
        metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return tool_dir


def load_tool(name: str) -> ToolMetadata | None:
    """Load metadata for a named tool.

    Args:
        name: The tool name (directory name under ~/.forge/tools/).

    Returns:
        ToolMetadata if found, None otherwise.
    """
    meta_file = TOOLS_DIR / name / "metadata.json"
    if not meta_file.exists():
        return None

    data = json.loads(meta_file.read_text(encoding="utf-8"))
    return ToolMetadata(**data)


def list_tools() -> list[ToolMetadata]:
    """List all saved tools.

    Returns:
        List of ToolMetadata for every tool in the forge tools directory.
    """
    ensure_dirs()
    tools: list[ToolMetadata] = []

    for tool_dir in sorted(TOOLS_DIR.iterdir()):
        if not tool_dir.is_dir():
            continue
        meta_file = tool_dir / "metadata.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                tools.append(ToolMetadata(**data))
            except (json.JSONDecodeError, Exception):
                continue

    return tools


def delete_tool(name: str) -> bool:
    """Delete a tool and all its files.

    Args:
        name: The tool name to delete.

    Returns:
        True if deleted, False if not found.
    """
    tool_dir = TOOLS_DIR / name
    if not tool_dir.exists():
        return False

    import shutil
    shutil.rmtree(tool_dir)
    return True


def get_tool_source(name: str) -> str | None:
    """Read the source code of a saved tool.

    Args:
        name: The tool name.

    Returns:
        Source code string, or None if not found.
    """
    tool_file = TOOLS_DIR / name / f"{name}.py"
    if tool_file.exists():
        return tool_file.read_text(encoding="utf-8")
    return None


def get_tool_tests(name: str) -> str | None:
    """Read the test code for a saved tool.

    Args:
        name: The tool name.

    Returns:
        Test code string, or None if not found.
    """
    test_file = TOOLS_DIR / name / f"test_{name}.py"
    if test_file.exists():
        return test_file.read_text(encoding="utf-8")
    return None
