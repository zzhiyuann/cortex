"""Tests for AgentRunner._read_stream and progress loop behavior.

Focuses on:
- _read_stream returns promptly after result event (no stderr hang)
- Progress messages are cleaned up in all code paths
- Writing... indicator never leaks into final output
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from dispatcher.runner import AgentRunner
from dispatcher.session import Session


# -- Helpers --

class FakeProcess:
    """Simulates an asyncio subprocess with controllable stdout/stderr."""

    def __init__(self, stdout_lines: list[str], stderr_text: str = ""):
        self._stdout_lines = stdout_lines
        self._stderr_text = stderr_text
        self.returncode = 0
        self.stdin = MagicMock()
        self.stdin.write = MagicMock()
        self.stdin.drain = AsyncMock()
        self.stdin.close = MagicMock()
        # Build async stdout iterator
        self.stdout = self._make_stdout()
        self.stderr = self._make_stderr()
        self._killed = False

    async def _line_iter(self):
        for line in self._stdout_lines:
            yield (line + "\n").encode()

    def _make_stdout(self):
        return self._line_iter()

    def _make_stderr(self):
        """Stderr that only returns data after process is 'killed'."""
        async def read():
            # Simulate: stderr.read() blocks until process exits.
            # In tests, return immediately to avoid hanging.
            if self._killed:
                return self._stderr_text.encode()
            return self._stderr_text.encode()
        mock = MagicMock()
        mock.read = AsyncMock(side_effect=read)
        return mock

    def terminate(self):
        self._killed = True

    def kill(self):
        self._killed = True

    async def wait(self):
        return self.returncode


def make_assistant_event(text: str) -> str:
    """Build a stream-json assistant event with text content."""
    return json.dumps({
        "type": "assistant",
        "message": {
            "content": [{"type": "text", "text": text}]
        },
    })


def make_result_event(result: str) -> str:
    """Build a stream-json result event."""
    return json.dumps({"type": "result", "result": result})


def make_question_event(tool_use_id: str, question: str, options: list[dict]) -> str:
    """Build a stream-json assistant event with AskUserQuestion tool_use."""
    return json.dumps({
        "type": "assistant",
        "message": {
            "content": [{
                "type": "tool_use",
                "id": tool_use_id,
                "name": "AskUserQuestion",
                "input": {
                    "questions": [{
                        "question": question,
                        "options": options,
                        "multiSelect": False,
                    }]
                },
            }]
        },
    })


# -- Tests --

class TestReadStreamNoHang:
    """Verify _read_stream returns promptly after result event."""

    @pytest.mark.asyncio
    async def test_returns_result_immediately(self):
        """_read_stream should return as soon as result event is received."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "hi", "/tmp")

        proc = FakeProcess([
            make_assistant_event("Hello!"),
            make_result_event("Hello!"),
        ])

        start = time.monotonic()
        result = await runner._read_stream(proc, session)
        elapsed = time.monotonic() - start

        assert result == "Hello!"
        assert elapsed < 2, f"_read_stream took {elapsed:.1f}s, should be instant"

    @pytest.mark.asyncio
    async def test_no_result_uses_last_text(self):
        """When no result event, falls back to last assistant text."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "hi", "/tmp")

        proc = FakeProcess([
            make_assistant_event("Partial response"),
        ])

        result = await runner._read_stream(proc, session)
        assert result == "Partial response"

    @pytest.mark.asyncio
    async def test_no_result_no_text_uses_stderr(self):
        """When no result and no text, falls back to stderr."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "hi", "/tmp")

        proc = FakeProcess([], stderr_text="some error")

        result = await runner._read_stream(proc, session)
        assert "(stderr)" in result
        assert "some error" in result

    @pytest.mark.asyncio
    async def test_partial_output_updated_during_stream(self):
        """session.partial_output should be updated as assistant events arrive."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "hi", "/tmp")

        proc = FakeProcess([
            make_assistant_event("First chunk"),
            make_assistant_event("Second chunk"),
            make_result_event("Second chunk"),
        ])

        result = await runner._read_stream(proc, session)
        assert result == "Second chunk"
        # partial_output should contain accumulated text
        assert session.partial_output is not None
        assert "chunk" in session.partial_output

    @pytest.mark.asyncio
    async def test_result_has_no_writing_indicator(self):
        """Result text must never contain 'Writing...' — that's only for progress."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "hi", "/tmp")

        proc = FakeProcess([
            make_assistant_event("Hello! How can I help?"),
            make_result_event("Hello! How can I help?"),
        ])

        result = await runner._read_stream(proc, session)
        assert "Writing" not in result
        assert "\u270f" not in result  # pencil emoji


class TestBuildToolResult:
    """Verify tool_result JSON format for F7 question relay."""

    def test_with_answer(self):
        result = AgentRunner._build_tool_result(
            "toolu_123",
            [{"question": "Pick a color?"}],
            "Blue",
        )
        assert result["type"] == "user"
        content = result["message"]["content"][0]
        assert content["type"] == "tool_result"
        assert content["tool_use_id"] == "toolu_123"
        assert content["is_error"] is False
        parsed = json.loads(content["content"])
        assert parsed["questions"]["Pick a color?"] == "Blue"

    def test_timeout_answer(self):
        result = AgentRunner._build_tool_result(
            "toolu_456",
            [{"question": "Choose?"}],
            None,
        )
        content = result["message"]["content"][0]
        assert content["is_error"] is True
        parsed = json.loads(content["content"])
        assert "timed out" in parsed["questions"]["Choose?"].lower()


class TestProgressCleanup:
    """Verify progress messages are always cleaned up."""

    @pytest.mark.asyncio
    async def test_progress_loop_deletes_on_cancel(self):
        """When monitor.cancel() fires, finally block must delete progress msg."""
        from dispatcher.config import Config
        from dispatcher.core import Dispatcher

        # Minimal dispatcher with mocked Telegram
        import yaml, tempfile, os
        tmp = tempfile.mkdtemp()
        cfg_data = {
            "telegram": {"bot_token": "test", "chat_id": 12345},
            "agent": {"command": "echo", "args": [], "timeout": 30,
                      "max_turns": 5, "max_turns_chat": 3, "max_turns_followup": 3,
                      "max_concurrent": 2},
            "behavior": {"poll_timeout": 1, "progress_interval": 60,
                         "recent_window": 300},
            "projects": {},
            "data_dir": os.path.join(tmp, "data"),
        }
        os.makedirs(cfg_data["data_dir"], exist_ok=True)
        cfg_path = os.path.join(tmp, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_data, f)
        cfg = Config(cfg_path)

        d = Dispatcher(cfg)
        d.tg = MagicMock()
        d.tg.send = MagicMock(return_value=9999)
        d.tg.edit = MagicMock(return_value=True)
        d.tg.typing = MagicMock()
        d.tg.delete_message = MagicMock(return_value=True)
        d.tg.set_my_commands = MagicMock(return_value=True)

        # Create a session that simulates partial output
        session = Session(1, "test", "/tmp")
        session.status = "running"
        session.started = time.time()
        session.partial_output = "Some partial text"

        # Start progress loop, let it send a progress message, then cancel
        monitor = asyncio.create_task(d._progress_loop(1, session))
        await asyncio.sleep(4)  # Let it wake up and send progress msg

        # Now simulate task completion
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass

        # If a progress message was sent, it must have been deleted
        if d.tg.send.called:
            assert d.tg.delete_message.called, \
                "Progress message was sent but never deleted!"


class TestEndToEndNoWritingLeak:
    """Integration test: send message, verify final output has no 'Writing...'."""

    @pytest.mark.asyncio
    async def test_fast_response_no_writing_leak(self):
        """For a fast response (like 'hi'), user should never see 'Writing...'."""
        runner = AgentRunner(command="echo", args=[], timeout=10)

        session = Session(1, "hi", "/tmp")
        session.status = "running"
        session.started = time.time()

        # Simulate: agent responds instantly with simple text
        proc = FakeProcess([
            make_assistant_event("Hi! How can I help?"),
            make_result_event("Hi! How can I help?"),
        ])

        result = await runner._read_stream(proc, session)

        assert "Writing" not in result
        assert result == "Hi! How can I help?"
        # partial_output is for progress display only — result should be clean
        assert "Writing" not in (session.result or "")
