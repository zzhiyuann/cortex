"""Event capture engine for Vibe Replay.

Captures events from Claude Code hooks and transforms them into
structured Event objects for storage and analysis.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Event, EventType, HookPayload


# Tools that produce code changes
CODE_CHANGE_TOOLS = {"Edit", "Write", "NotebookEdit"}

# Tools used for exploration
EXPLORATION_TOOLS = {"Read", "Glob", "Grep", "Bash", "WebFetch", "WebSearch"}

# Max size for details to avoid storing huge payloads
MAX_DETAIL_SIZE = 50_000


def parse_hook_stdin(stdin_data: str) -> HookPayload:
    """Parse JSON data received from Claude Code hook stdin.

    Args:
        stdin_data: Raw JSON string from stdin.

    Returns:
        Parsed HookPayload.
    """
    try:
        data = json.loads(stdin_data)
    except json.JSONDecodeError:
        return HookPayload(
            session_id="unknown",
            hook_type="error",
            message=f"Failed to parse hook input: {stdin_data[:200]}",
        )

    return HookPayload(
        session_id=data.get("session_id", "unknown"),
        tool_name=data.get("tool_name"),
        tool_input=data.get("tool_input"),
        tool_output=data.get("tool_result") or data.get("tool_output"),
        hook_type=data.get("hook_type", ""),
        message=data.get("message"),
        error=data.get("error"),
    )


def _truncate(text: str | None, max_len: int = 5000) -> str | None:
    """Truncate text to a reasonable size for storage."""
    if text is None:
        return None
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [truncated, {len(text)} chars total]"


def _extract_code_diff(tool_name: str, tool_input: dict[str, Any] | None) -> str | None:
    """Extract code diff information from tool input."""
    if not tool_input:
        return None

    if tool_name == "Edit":
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", "")
        if old or new:
            return f"--- old\n+++ new\n-{old}\n+{new}"

    elif tool_name == "Write":
        content = tool_input.get("content", "")
        return _truncate(f"+++ new file\n{content}", max_len=3000)

    return None


def _extract_files_affected(
    tool_name: str, tool_input: dict[str, Any] | None
) -> list[str]:
    """Extract list of files affected by a tool call."""
    if not tool_input:
        return []

    file_fields = ["file_path", "path", "notebook_path"]
    for field in file_fields:
        val = tool_input.get(field)
        if val:
            return [val]

    return []


def _summarize_tool_call(
    tool_name: str,
    tool_input: dict[str, Any] | None,
    tool_output: Any | None,
) -> str:
    """Generate a human-readable summary of a tool call."""
    if not tool_input:
        return f"Called {tool_name}"

    if tool_name == "Edit":
        fp = tool_input.get("file_path", "?")
        fname = Path(fp).name if fp != "?" else "?"
        return f"Edited {fname}"

    elif tool_name == "Write":
        fp = tool_input.get("file_path", "?")
        fname = Path(fp).name if fp != "?" else "?"
        return f"Created/wrote {fname}"

    elif tool_name == "Read":
        fp = tool_input.get("file_path", "?")
        fname = Path(fp).name if fp != "?" else "?"
        return f"Read {fname}"

    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        # Truncate long commands
        if len(cmd) > 100:
            cmd = cmd[:100] + "..."
        return f"Ran: {cmd}"

    elif tool_name == "Glob":
        pattern = tool_input.get("pattern", "?")
        return f"Searched for files: {pattern}"

    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "?")
        return f"Searched content: {pattern}"

    elif tool_name == "WebSearch":
        query = tool_input.get("query", "?")
        return f"Web search: {query}"

    elif tool_name == "WebFetch":
        url = tool_input.get("url", "?")
        return f"Fetched: {url}"

    elif tool_name == "Task":
        prompt = tool_input.get("prompt", "")
        if len(prompt) > 100:
            prompt = prompt[:100] + "..."
        return f"Delegated task: {prompt}"

    elif tool_name == "NotebookEdit":
        fp = tool_input.get("notebook_path", "?")
        fname = Path(fp).name if fp != "?" else "?"
        mode = tool_input.get("edit_mode", "replace")
        return f"Edited notebook {fname} ({mode})"

    return f"Called {tool_name}"


def _classify_event_type(tool_name: str) -> EventType:
    """Classify a tool call into an event type."""
    if tool_name in CODE_CHANGE_TOOLS:
        return EventType.CODE_CHANGE
    return EventType.TOOL_CALL


def _sanitize_details(details: dict[str, Any]) -> dict[str, Any]:
    """Sanitize details to keep storage reasonable."""
    sanitized = {}
    total_size = 0

    for key, value in details.items():
        if isinstance(value, str):
            val = _truncate(value, max_len=5000)
        else:
            val = value

        entry_json = json.dumps({key: val}, default=str)
        if total_size + len(entry_json) > MAX_DETAIL_SIZE:
            sanitized[key] = "[truncated — too large]"
            break
        total_size += len(entry_json)
        sanitized[key] = val

    return sanitized


def create_event_from_hook(payload: HookPayload) -> Event:
    """Transform a hook payload into a structured Event.

    Args:
        payload: Parsed hook payload from Claude Code.

    Returns:
        Structured Event ready for storage.
    """
    tool_name = payload.tool_name or ""
    tool_input = payload.tool_input or {}
    tool_output = payload.tool_output

    event_type = _classify_event_type(tool_name) if tool_name else EventType.TOOL_CALL

    # Handle special hook types
    if payload.hook_type == "Stop":
        event_type = EventType.SESSION_END
        summary = "Session ended"
    elif payload.hook_type == "Notification":
        event_type = EventType.NOTIFICATION
        summary = payload.message or "Notification"
    elif payload.error:
        event_type = EventType.ERROR
        summary = f"Error: {payload.error[:200]}"
    else:
        summary = _summarize_tool_call(tool_name, tool_input, tool_output)

    # Build details
    details: dict[str, Any] = {}
    if tool_input:
        details["input"] = tool_input
    if tool_output is not None:
        # Store a truncated version of the output
        if isinstance(tool_output, str):
            details["output"] = _truncate(tool_output, max_len=3000)
        elif isinstance(tool_output, dict):
            details["output"] = tool_output
        else:
            details["output"] = str(tool_output)[:3000]

    details = _sanitize_details(details)

    return Event(
        timestamp=datetime.now(),
        session_id=payload.session_id,
        event_type=event_type,
        tool_name=tool_name or None,
        summary=summary,
        details=details,
        code_diff=_extract_code_diff(tool_name, payload.tool_input),
        files_affected=_extract_files_affected(tool_name, payload.tool_input),
    )


def capture_from_stdin() -> Event | None:
    """Read hook data from stdin and create an Event.

    This is called by the capture hook script.

    Returns:
        Event if successfully parsed, None otherwise.
    """
    try:
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            return None

        payload = parse_hook_stdin(stdin_data)
        return create_event_from_hook(payload)

    except Exception as e:
        # Never crash the hook — log the error quietly
        err_path = Path.home() / ".vibe-replay" / "capture-errors.log"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with open(err_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} ERROR: {e}\n")
        return None
