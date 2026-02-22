"""Tests for the capture engine."""

import json
from datetime import datetime

from vibe_replay.capture import (
    create_event_from_hook,
    parse_hook_stdin,
    _extract_code_diff,
    _extract_files_affected,
    _summarize_tool_call,
)
from vibe_replay.models import EventType, HookPayload


class TestParseHookStdin:
    """Tests for parsing hook stdin data."""

    def test_valid_json(self):
        data = json.dumps({
            "session_id": "abc123",
            "tool_name": "Edit",
            "tool_input": {"file_path": "/tmp/test.py", "old_string": "a", "new_string": "b"},
        })
        payload = parse_hook_stdin(data)
        assert payload.session_id == "abc123"
        assert payload.tool_name == "Edit"
        assert payload.tool_input["file_path"] == "/tmp/test.py"

    def test_invalid_json(self):
        payload = parse_hook_stdin("not valid json {{{")
        assert payload.session_id == "unknown"
        assert payload.hook_type == "error"

    def test_empty_input(self):
        payload = parse_hook_stdin("{}")
        assert payload.session_id == "unknown"

    def test_tool_result_field(self):
        data = json.dumps({
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_result": "file1.py\nfile2.py",
        })
        payload = parse_hook_stdin(data)
        assert payload.tool_output == "file1.py\nfile2.py"


class TestCreateEventFromHook:
    """Tests for creating events from hook payloads."""

    def test_edit_event(self):
        payload = HookPayload(
            session_id="s1",
            tool_name="Edit",
            tool_input={
                "file_path": "/project/main.py",
                "old_string": "print('hello')",
                "new_string": "print('world')",
            },
        )
        event = create_event_from_hook(payload)
        assert event.event_type == EventType.CODE_CHANGE
        assert event.tool_name == "Edit"
        assert "main.py" in event.summary
        assert event.code_diff is not None
        assert "/project/main.py" in event.files_affected

    def test_bash_event(self):
        payload = HookPayload(
            session_id="s1",
            tool_name="Bash",
            tool_input={"command": "pytest tests/"},
        )
        event = create_event_from_hook(payload)
        assert event.event_type == EventType.TOOL_CALL
        assert "pytest" in event.summary

    def test_read_event(self):
        payload = HookPayload(
            session_id="s1",
            tool_name="Read",
            tool_input={"file_path": "/project/config.json"},
        )
        event = create_event_from_hook(payload)
        assert event.event_type == EventType.TOOL_CALL
        assert "config.json" in event.summary

    def test_stop_event(self):
        payload = HookPayload(
            session_id="s1",
            hook_type="Stop",
        )
        event = create_event_from_hook(payload)
        assert event.event_type == EventType.SESSION_END

    def test_error_event(self):
        payload = HookPayload(
            session_id="s1",
            error="Command failed with exit code 1",
        )
        event = create_event_from_hook(payload)
        assert event.event_type == EventType.ERROR

    def test_write_event_with_diff(self):
        payload = HookPayload(
            session_id="s1",
            tool_name="Write",
            tool_input={
                "file_path": "/project/new_file.py",
                "content": "print('new file')",
            },
        )
        event = create_event_from_hook(payload)
        assert event.event_type == EventType.CODE_CHANGE
        assert event.code_diff is not None
        assert "new_file.py" in event.summary


class TestHelpers:
    """Tests for helper functions."""

    def test_extract_code_diff_edit(self):
        diff = _extract_code_diff("Edit", {
            "old_string": "old code",
            "new_string": "new code",
        })
        assert diff is not None
        assert "old code" in diff
        assert "new code" in diff

    def test_extract_code_diff_no_input(self):
        diff = _extract_code_diff("Edit", None)
        assert diff is None

    def test_extract_files_edit(self):
        files = _extract_files_affected("Edit", {"file_path": "/a/b.py"})
        assert files == ["/a/b.py"]

    def test_extract_files_no_input(self):
        files = _extract_files_affected("Bash", None)
        assert files == []

    def test_summarize_glob(self):
        summary = _summarize_tool_call("Glob", {"pattern": "**/*.py"}, None)
        assert "**/*.py" in summary

    def test_summarize_grep(self):
        summary = _summarize_tool_call("Grep", {"pattern": "def main"}, None)
        assert "def main" in summary

    def test_summarize_unknown_tool(self):
        summary = _summarize_tool_call("UnknownTool", {}, None)
        assert "UnknownTool" in summary
