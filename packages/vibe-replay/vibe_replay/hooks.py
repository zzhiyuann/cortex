"""Claude Code hook management for Vibe Replay.

Handles installation and removal of hooks in ~/.claude/settings.json
that capture session events for Vibe Replay.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any


SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
VIBE_REPLAY_DIR = Path.home() / ".vibe-replay"
CAPTURE_HOOK_PATH = VIBE_REPLAY_DIR / "capture-hook.py"
STOP_HOOK_PATH = VIBE_REPLAY_DIR / "stop-hook.py"

# Marker to identify Vibe Replay hooks
HOOK_MARKER = "vibe-replay"

CAPTURE_HOOK_SCRIPT = '''#!/usr/bin/env python3
"""Vibe Replay capture hook for Claude Code.

This script is called by Claude Code hooks (PostToolUse) and receives
event JSON on stdin. It appends the event to the current session's
event log.

Designed to be fast and non-blocking — never crashes.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    try:
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            return

        data = json.loads(stdin_data)

        session_id = data.get("session_id", "unknown")
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})
        tool_output = data.get("tool_result") or data.get("tool_output")

        # Determine event type
        code_tools = {"Edit", "Write", "NotebookEdit"}
        event_type = "code_change" if tool_name in code_tools else "tool_call"

        # Build summary
        summary = _summarize(tool_name, tool_input)

        # Extract files
        files = []
        for field in ("file_path", "path", "notebook_path"):
            val = tool_input.get(field)
            if val:
                files.append(val)

        # Extract code diff
        code_diff = None
        if tool_name == "Edit":
            old = tool_input.get("old_string", "")
            new = tool_input.get("new_string", "")
            if old or new:
                code_diff = f"--- old\\n+++ new\\n-{old}\\n+{new}"
        elif tool_name == "Write":
            content = tool_input.get("content", "")
            if len(content) > 3000:
                content = content[:3000] + "..."
            code_diff = f"+++ new file\\n{content}"

        # Build event
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "tool_name": tool_name,
            "summary": summary,
            "details": _safe_details(tool_input, tool_output),
            "code_diff": code_diff,
            "files_affected": files,
        }

        # Append to session event log
        sessions_dir = Path.home() / ".vibe-replay" / "sessions" / session_id
        sessions_dir.mkdir(parents=True, exist_ok=True)
        events_file = sessions_dir / "events.jsonl"

        with open(events_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\\n")

    except Exception:
        # Never crash — silently log errors
        try:
            err_path = Path.home() / ".vibe-replay" / "capture-errors.log"
            with open(err_path, "a") as f:
                import traceback
                f.write(f"{datetime.now().isoformat()} {traceback.format_exc()}\\n")
        except Exception:
            pass


def _summarize(tool_name, tool_input):
    """Generate a short summary."""
    if not tool_input:
        return f"Called {tool_name}"

    if tool_name == "Edit":
        fp = tool_input.get("file_path", "?")
        return f"Edited {Path(fp).name}"
    elif tool_name == "Write":
        fp = tool_input.get("file_path", "?")
        return f"Created/wrote {Path(fp).name}"
    elif tool_name == "Read":
        fp = tool_input.get("file_path", "?")
        return f"Read {Path(fp).name}"
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")[:100]
        return f"Ran: {cmd}"
    elif tool_name == "Glob":
        return f"Searched files: {tool_input.get('pattern', '?')}"
    elif tool_name == "Grep":
        return f"Searched content: {tool_input.get('pattern', '?')}"
    elif tool_name == "WebSearch":
        return f"Web search: {tool_input.get('query', '?')}"
    return f"Called {tool_name}"


def _safe_details(tool_input, tool_output):
    """Build a safe details dict with size limits."""
    details = {}
    if tool_input:
        inp = json.dumps(tool_input, default=str)
        if len(inp) > 10000:
            details["input"] = {"_truncated": True, "_size": len(inp)}
        else:
            details["input"] = tool_input

    if tool_output is not None:
        out = str(tool_output)
        if len(out) > 5000:
            details["output"] = out[:5000] + "..."
        else:
            details["output"] = tool_output

    return details


if __name__ == "__main__":
    main()
'''

STOP_HOOK_SCRIPT = '''#!/usr/bin/env python3
"""Vibe Replay stop hook for Claude Code.

Called when a Claude Code session ends. Finalizes session metadata.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    try:
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            return

        data = json.loads(stdin_data)
        session_id = data.get("session_id", "unknown")

        sessions_dir = Path.home() / ".vibe-replay" / "sessions" / session_id
        if not sessions_dir.exists():
            return

        # Add a session_end event
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "event_type": "session_end",
            "tool_name": None,
            "summary": "Session ended",
            "details": {},
            "code_diff": None,
            "files_affected": [],
        }

        events_file = sessions_dir / "events.jsonl"
        with open(events_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\\n")

        # Count events and build basic metadata
        event_count = 0
        first_ts = None
        last_ts = None
        files_seen = set()
        tools_used = {}

        with open(events_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    event_count += 1
                    ts = ev.get("timestamp", "")
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
                    for fp in ev.get("files_affected", []):
                        files_seen.add(fp)
                    tn = ev.get("tool_name")
                    if tn:
                        tools_used[tn] = tools_used.get(tn, 0) + 1
                except Exception:
                    continue

        metadata = {
            "session_id": session_id,
            "project": "",
            "project_path": "",
            "start_time": first_ts,
            "end_time": last_ts,
            "event_count": event_count,
            "summary": "",
            "tags": [],
            "files_modified": sorted(files_seen),
            "tools_used": tools_used,
        }

        meta_path = sessions_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    except Exception:
        try:
            err_path = Path.home() / ".vibe-replay" / "capture-errors.log"
            with open(err_path, "a") as f:
                import traceback
                f.write(f"{datetime.now().isoformat()} STOP: {traceback.format_exc()}\\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
'''


def _read_settings() -> dict[str, Any]:
    """Read the Claude Code settings file."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_settings(settings: dict[str, Any]) -> None:
    """Write the Claude Code settings file."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Back up existing settings
    if SETTINGS_PATH.exists():
        backup = SETTINGS_PATH.with_suffix(".json.bak")
        shutil.copy2(SETTINGS_PATH, backup)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def _write_hook_script(path: Path, content: str) -> None:
    """Write a hook script and make it executable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _is_vibe_replay_hook(hook: dict[str, Any]) -> bool:
    """Check if a hook entry belongs to Vibe Replay."""
    cmd = hook.get("command", "")
    return HOOK_MARKER in cmd or ".vibe-replay" in cmd


def install_hooks() -> dict[str, str]:
    """Install Vibe Replay hooks into Claude Code settings.

    Returns:
        Dict with status information.
    """
    # Write hook scripts
    _write_hook_script(CAPTURE_HOOK_PATH, CAPTURE_HOOK_SCRIPT)
    _write_hook_script(STOP_HOOK_PATH, STOP_HOOK_SCRIPT)

    # Read current settings
    settings = _read_settings()

    # Ensure hooks section exists
    if "hooks" not in settings:
        settings["hooks"] = {}

    hooks = settings["hooks"]

    # Add PostToolUse hook
    post_tool_use = hooks.get("PostToolUse", [])
    # Remove any existing vibe-replay hooks
    post_tool_use = [
        entry for entry in post_tool_use
        if not any(_is_vibe_replay_hook(h) for h in entry.get("hooks", []))
    ]
    # Add our hook
    post_tool_use.append(
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": f"python3 {CAPTURE_HOOK_PATH}  # vibe-replay",
                }
            ],
        }
    )
    hooks["PostToolUse"] = post_tool_use

    # Add Stop hook
    stop_hooks = hooks.get("Stop", [])
    stop_hooks = [
        entry for entry in stop_hooks
        if not any(_is_vibe_replay_hook(h) for h in entry.get("hooks", []))
    ]
    stop_hooks.append(
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": f"python3 {STOP_HOOK_PATH}  # vibe-replay",
                }
            ],
        }
    )
    hooks["Stop"] = stop_hooks

    # Write updated settings
    _write_settings(settings)

    return {
        "status": "installed",
        "capture_hook": str(CAPTURE_HOOK_PATH),
        "stop_hook": str(STOP_HOOK_PATH),
        "settings_path": str(SETTINGS_PATH),
    }


def uninstall_hooks() -> dict[str, str]:
    """Remove Vibe Replay hooks from Claude Code settings.

    Returns:
        Dict with status information.
    """
    settings = _read_settings()
    hooks = settings.get("hooks", {})

    removed = 0

    for hook_type in ("PostToolUse", "Stop", "PreToolUse", "Notification"):
        entries = hooks.get(hook_type, [])
        original_count = len(entries)
        entries = [
            entry for entry in entries
            if not any(_is_vibe_replay_hook(h) for h in entry.get("hooks", []))
        ]
        removed += original_count - len(entries)
        if entries:
            hooks[hook_type] = entries
        elif hook_type in hooks:
            # Remove empty hook type
            if not entries:
                hooks[hook_type] = entries

    settings["hooks"] = hooks
    _write_settings(settings)

    # Clean up hook scripts
    for path in (CAPTURE_HOOK_PATH, STOP_HOOK_PATH):
        if path.exists():
            path.unlink()

    return {
        "status": "uninstalled",
        "hooks_removed": removed,
        "settings_path": str(SETTINGS_PATH),
    }


def check_hooks() -> dict[str, Any]:
    """Check if Vibe Replay hooks are currently installed.

    Returns:
        Dict with installation status.
    """
    settings = _read_settings()
    hooks = settings.get("hooks", {})

    installed_hooks = []
    for hook_type, entries in hooks.items():
        for entry in entries:
            for hook in entry.get("hooks", []):
                if _is_vibe_replay_hook(hook):
                    installed_hooks.append(
                        {"type": hook_type, "command": hook.get("command", "")}
                    )

    capture_exists = CAPTURE_HOOK_PATH.exists()
    stop_exists = STOP_HOOK_PATH.exists()

    return {
        "installed": len(installed_hooks) > 0,
        "hooks": installed_hooks,
        "capture_script_exists": capture_exists,
        "stop_script_exists": stop_exists,
        "settings_path": str(SETTINGS_PATH),
    }
