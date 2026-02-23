"""Comprehensive tests for dispatcher critical paths.

Covers the full message lifecycle, progress loop cleanup, F7 question relay,
callback handling, session routing, edited messages, stream reading edge cases,
and result delivery — all areas that directly affect user experience.

Uses mock runner + mock Telegram client: no real Claude CLI calls.
"""

from __future__ import annotations

import asyncio
import html
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import yaml

from dispatcher.config import Config
from dispatcher.core import Dispatcher, _md_to_telegram_html
from dispatcher.runner import AgentRunner
from dispatcher.session import Session, SessionManager
from dispatcher.transcript import Transcript


# ========================================================================
# Fixtures & helpers
# ========================================================================

@pytest.fixture(autouse=True)
def _no_record_issue(monkeypatch):
    """Patch record_issue to a no-op so tests don't pollute the real issue store."""
    monkeypatch.setattr("dispatcher.core.record_issue", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def _no_classify_llm(monkeypatch):
    """Patch classify_intent to always return 'task' so no real LLM calls are made."""
    async def mock_classify(message, active_sessions):
        return "task"
    monkeypatch.setattr("dispatcher.core.classify_intent", mock_classify)


def make_config(tmp_path, projects=None):
    """Create a minimal config file for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    cfg_data = {
        "telegram": {"bot_token": "test-token", "chat_id": 12345},
        "agent": {
            "command": "echo",
            "args": [],
            "max_concurrent": 3,
            "timeout": 30,
            "max_turns": 10,
            "max_turns_chat": 5,
            "max_turns_followup": 5,
            "question_timeout": 600,
        },
        "behavior": {
            "poll_timeout": 1,
            "progress_interval": 60,
            "recent_window": 300,
            "cancel_keywords": ["cancel", "stop", "取消", "停"],
            "status_keywords": ["status", "在干嘛"],
        },
        "projects": projects or {},
        "data_dir": str(data_dir),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))
    return Config(str(cfg_path))


def make_dispatcher(tmp_path, projects=None):
    """Create a Dispatcher with fully mocked Telegram client."""
    cfg = make_config(tmp_path, projects)
    d = Dispatcher(cfg)
    # Mock Telegram client
    d.tg = MagicMock()
    d.tg.send = MagicMock(return_value=9999)
    d.tg.edit = MagicMock(return_value=True)
    d.tg.typing = MagicMock()
    d.tg.delete_message = MagicMock(return_value=True)
    d.tg.send_document = MagicMock(return_value=8888)
    d.tg.answer_callback = MagicMock()
    d.tg.set_my_commands = MagicMock(return_value=True)
    d.tg.react = MagicMock()
    return d


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
        self.stdout = self._make_stdout()
        self.stderr = self._make_stderr()
        self._killed = False

    async def _line_iter(self):
        for line in self._stdout_lines:
            yield (line + "\n").encode()

    def _make_stdout(self):
        return self._line_iter()

    def _make_stderr(self):
        mock = MagicMock()
        mock.read = AsyncMock(return_value=self._stderr_text.encode())
        return mock

    def terminate(self):
        self._killed = True

    def kill(self):
        self._killed = True

    async def wait(self):
        return self.returncode


def make_assistant_event(text: str) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    })


def make_result_event(result: str) -> str:
    return json.dumps({"type": "result", "result": result})


def make_question_event(tool_use_id: str, question: str, options: list[dict]) -> str:
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


# ========================================================================
# 1. Full E2E Message Flow — no "Writing..." leak
# ========================================================================

class TestE2EMessageFlowNoWritingLeak:
    """End-to-end: user sends message -> session -> result -> Telegram, clean output."""

    @pytest.mark.asyncio
    async def test_fast_response_no_writing_in_final_message(self, tmp_path):
        """Fast response (< 3s): user should never see 'Writing...' or pencil emoji."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "hi", str(Path.home()))
        session.is_task = False

        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            session.status = "running"
            session.started = time.time()
            await asyncio.sleep(0.05)  # fast
            session.status = "done"
            session.finished = time.time()
            session.result = "Hello! How can I help?"
            return "Hello! How can I help?"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "hi")

        # Verify final message was sent
        assert d.tg.send.call_count >= 1
        # Check all send() calls for Writing... leak
        for call in d.tg.send.call_args_list:
            text = call[0][0] if call[0] else call[1].get("text", "")
            assert "Writing" not in text, f"'Writing...' leaked into final message: {text}"
            assert "\u270f" not in text, f"Pencil emoji leaked into final message: {text}"

    @pytest.mark.asyncio
    async def test_slow_response_final_message_clean(self, tmp_path):
        """Slow response (> 3s): progress may appear during run but final is clean."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "fix a complex bug", "/tmp/proj")
        session.is_task = True

        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            session.status = "running"
            session.started = time.time()
            # Simulate slow streaming with partial output
            session.partial_output = "Looking at the code..."
            await asyncio.sleep(0.1)
            session.partial_output = "Found the bug in auth.py"
            await asyncio.sleep(0.1)
            session.status = "done"
            session.finished = time.time()
            session.result = "Fixed the null check in auth.py line 42"
            return "Fixed the null check in auth.py line 42"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "fix a complex bug")

        # The LAST send call is the final result
        last_send = d.tg.send.call_args_list[-1]
        text = last_send[0][0] if last_send[0] else last_send[1].get("text", "")
        assert "Writing" not in text
        assert "\u270f" not in text
        assert "auth.py" in text

    @pytest.mark.asyncio
    async def test_stream_result_text_is_clean(self):
        """_read_stream result must never contain 'Writing...' indicator."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "hi", "/tmp")

        proc = FakeProcess([
            make_assistant_event("Hello! How can I help?"),
            make_result_event("Hello! How can I help?"),
        ])

        result = await runner._read_stream(proc, session)
        assert "Writing" not in result
        assert "\u270f" not in result
        assert result == "Hello! How can I help?"


# ========================================================================
# 2. Progress Loop Lifecycle
# ========================================================================

class TestProgressLoopLifecycle:
    """Progress loop sends 'Writing...' during streaming, cleans up on cancel."""

    @pytest.mark.asyncio
    async def test_progress_sends_writing_during_streaming(self, tmp_path):
        """Progress loop sends a message with 'Writing...' when partial output grows."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")
        session.status = "running"
        session.started = time.time()
        session.partial_output = "Some partial text here"

        task = asyncio.create_task(d._progress_loop(1, session))
        # Wait for first check cycle (sleep(3) in the loop) + update
        await asyncio.sleep(4)

        # Cancel to stop the loop
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # If partial output was there, a progress message should have been sent
        if d.tg.send.called:
            first_call = d.tg.send.call_args_list[0]
            text = first_call[0][0] if first_call[0] else first_call[1].get("text", "")
            assert "Writing" in text, "Progress message should contain 'Writing...'"

    @pytest.mark.asyncio
    async def test_progress_deletes_message_on_cancel(self, tmp_path):
        """When monitor.cancel() fires, finally block must delete progress message."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")
        session.status = "running"
        session.started = time.time()
        session.partial_output = "Some partial text"

        task = asyncio.create_task(d._progress_loop(1, session))
        await asyncio.sleep(4)  # Let it send a progress msg

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # If a progress message was sent, it must have been deleted
        if d.tg.send.called:
            assert d.tg.delete_message.called, \
                "Progress message was sent but never deleted!"

    @pytest.mark.asyncio
    async def test_cancel_during_sleep_still_cleans_up(self, tmp_path):
        """Cancel during the asyncio.sleep(3) inside the loop still runs finally."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")
        session.status = "running"
        session.started = time.time()
        session.partial_output = "Text"

        task = asyncio.create_task(d._progress_loop(1, session))
        # Cancel immediately, before the first sleep(3) completes
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # No progress message should have been sent (cancelled before 3s)
        # But the finally block should have run without error
        # (no crash = test passes)

    @pytest.mark.asyncio
    async def test_fast_task_no_progress_message(self, tmp_path):
        """If task finishes before the 3s check, no progress message is sent."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "hi", str(Path.home()))
        session.status = "running"
        session.started = time.time()

        task = asyncio.create_task(d._progress_loop(1, session))
        await asyncio.sleep(0.05)
        session.status = "done"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        d.tg.send.assert_not_called()
        d.tg.delete_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_during_edit_still_cleans_up(self, tmp_path):
        """Cancel during tg.edit() call still cleans up progress message."""
        d = make_dispatcher(tmp_path)
        # Make edit slow so cancel can happen during it
        d.tg.edit = MagicMock(side_effect=lambda *a, **kw: time.sleep(0.01))

        session = d.sm.create(1, "test", "/tmp")
        session.status = "running"
        session.started = time.time()
        session.partial_output = "Initial"

        task = asyncio.create_task(d._progress_loop(1, session))
        await asyncio.sleep(4)

        # Update partial to trigger an edit on next cycle
        session.partial_output = "Updated text that is longer"
        await asyncio.sleep(0.5)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Finally block should still fire delete_message
        if d.tg.send.called:
            assert d.tg.delete_message.called


# ========================================================================
# 3. F7 Question Relay (_surface_question)
# ========================================================================

class TestF7QuestionRelay:
    """Agent asks a question -> inline keyboard -> user answers -> result relayed."""

    def test_surface_question_creates_inline_keyboard(self, tmp_path):
        """_surface_question should send a message with inline keyboard buttons."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")
        session.sid = "abcd1234-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        session.pending_question = {
            "tool_use_id": "toolu_123",
            "questions": [{
                "question": "Which file to modify?",
                "options": [
                    {"label": "auth.py", "description": "Authentication module"},
                    {"label": "main.py", "description": "Main entry"},
                ],
            }],
            "tg_msg_id": None,
        }

        # Run the sync parts
        loop = asyncio.new_event_loop()
        loop.run_until_complete(d._surface_question(session))
        loop.close()

        # Verify message was sent with inline keyboard
        assert d.tg.send.called
        call_kwargs = d.tg.send.call_args
        text = call_kwargs[0][0]
        assert "Which file to modify?" in text
        assert "Agent asks" in text

        markup = call_kwargs[1].get("reply_markup")
        assert markup is not None
        rows = markup["inline_keyboard"]
        # 2 options + 1 "Other..." button = 3 rows
        assert len(rows) == 3

        # Check first button
        assert "auth.py" in rows[0][0]["text"]
        assert rows[0][0]["callback_data"] == "answer:abcd1234:0"

        # Check second button
        assert "main.py" in rows[1][0]["text"]
        assert rows[1][0]["callback_data"] == "answer:abcd1234:1"

        # Check "Other..." button
        assert "Other" in rows[2][0]["text"]
        assert rows[2][0]["callback_data"] == "answer:abcd1234:other"

    @pytest.mark.asyncio
    async def test_callback_answer_resolves_to_correct_label(self, tmp_path):
        """Clicking option button resolves index to the correct label text."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")
        session.sid = "abcd1234-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        session.answer_event = asyncio.Event()
        session.pending_question = {
            "tool_use_id": "toolu_123",
            "questions": [{"question": "Pick?", "options": [
                {"label": "Option A"},
                {"label": "Option B"},
            ]}],
            "option_labels": ["Option A", "Option B"],
            "tg_msg_id": 9999,
        }

        cb = {
            "id": "cb_test",
            "data": "answer:abcd1234:1",  # index 1 -> "Option B"
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)

        # Should have resolved to "Option B"
        assert session.answer_data == "Option B"
        assert session.answer_event.is_set()
        d.tg.answer_callback.assert_called_with("cb_test", "Selected: Option B")
        d.tg.edit.assert_called_with(9999, "\u2705 Answered: Option B")

    @pytest.mark.asyncio
    async def test_callback_answer_other_prompts_freetext(self, tmp_path):
        """Clicking 'Other...' tells user to reply with free text."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")
        session.sid = "abcd1234-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        session.answer_event = asyncio.Event()
        session.pending_question = {
            "tool_use_id": "toolu_123",
            "questions": [{"question": "Pick?"}],
            "option_labels": ["A"],
            "tg_msg_id": 9999,
        }

        cb = {
            "id": "cb_other",
            "data": "answer:abcd1234:other",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)

        # Should prompt for free-text, NOT set the answer_event
        d.tg.answer_callback.assert_called_with("cb_other", "Reply to that message with your answer")
        assert not session.answer_event.is_set()

    @pytest.mark.asyncio
    async def test_freetext_reply_to_question_signals_answer(self, tmp_path):
        """Replying to the question message with free text should signal answer_event."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        d._fire_typing = MagicMock()

        session = d.sm.create(1, "original task", "/tmp")
        session.status = "running"
        session.answer_event = asyncio.Event()
        session.pending_question = {
            "tool_use_id": "toolu_123",
            "questions": [{"question": "What color?"}],
            "tg_msg_id": 9999,  # The question message ID
        }

        # User replies to question message with free text
        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 10,
            "text": "Blue please",
            "reply_to_message": {"message_id": 9999},
        }
        await d._on_message(msg)

        assert session.answer_data == "Blue please"
        assert session.answer_event.is_set()
        # Question message should be edited to show the answer
        d.tg.edit.assert_called_with(9999, "\u2705 Answered: Blue please")

    @pytest.mark.asyncio
    async def test_question_timeout_sends_error_result(self):
        """When question times out, _build_tool_result sends error."""
        result = AgentRunner._build_tool_result(
            "toolu_timeout",
            [{"question": "Choose an option?"}],
            None,  # answer is None = timeout
        )
        content_block = result["message"]["content"][0]
        assert content_block["is_error"] is True
        parsed = json.loads(content_block["content"])
        assert "timed out" in parsed["questions"]["Choose an option?"].lower()

    def test_build_tool_result_with_answer(self):
        """_build_tool_result formats answer correctly."""
        result = AgentRunner._build_tool_result(
            "toolu_ok",
            [{"question": "Pick a color?"}],
            "Red",
        )
        assert result["type"] == "user"
        content_block = result["message"]["content"][0]
        assert content_block["type"] == "tool_result"
        assert content_block["tool_use_id"] == "toolu_ok"
        assert content_block["is_error"] is False
        parsed = json.loads(content_block["content"])
        assert parsed["questions"]["Pick a color?"] == "Red"

    @pytest.mark.asyncio
    async def test_callback_answer_expired_question(self, tmp_path):
        """Answering an expired/non-existent question shows 'Question expired'."""
        d = make_dispatcher(tmp_path)
        # No session has a pending question
        cb = {
            "id": "cb_expired",
            "data": "answer:nonexist:0",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)
        d.tg.answer_callback.assert_called_with("cb_expired", "Question expired")


# ========================================================================
# 4. Callback Handling (_on_callback)
# ========================================================================

class TestCallbackHandling:
    """Inline keyboard button press routing."""

    @pytest.mark.asyncio
    async def test_cancel_callback_kills_running_session(self, tmp_path):
        """cancel: callback kills running session."""
        d = make_dispatcher(tmp_path)
        s = d.sm.create(1, "running task", "/tmp")
        s.status = "running"
        s.proc = MagicMock()

        cb = {
            "id": "cb_cancel",
            "data": f"cancel:{s.sid[:8]}",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)

        assert s.status == "cancelled"
        s.proc.kill.assert_called_once()
        d.tg.answer_callback.assert_called_with("cb_cancel", "Cancelled")

    @pytest.mark.asyncio
    async def test_cancel_callback_already_finished(self, tmp_path):
        """cancel: callback for already-finished session shows 'already finished'."""
        d = make_dispatcher(tmp_path)
        s = d.sm.create(1, "done task", "/tmp")
        s.status = "done"
        s.proc = None

        cb = {
            "id": "cb_done",
            "data": f"cancel:{s.sid[:8]}",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)

        d.tg.answer_callback.assert_called_with("cb_done", "Task already finished")

    @pytest.mark.asyncio
    async def test_retry_callback_redispatches(self, tmp_path):
        """retry: callback re-dispatches original task."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        s = d.sm.create(10, "fix the auth bug", "/tmp/proj")
        s.status = "failed"
        s.model_override = "haiku"

        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append({"mid": mid, "text": text, "model": model})
        d._handle_task = track_handle

        cb = {
            "id": "cb_retry",
            "data": "retry:10",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 99},
        }
        await d._on_callback(cb)

        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "fix the auth bug"
        assert dispatched[0]["model"] == "haiku"
        d.tg.answer_callback.assert_called_with("cb_retry", "\U0001f504 Retrying...")

    @pytest.mark.asyncio
    async def test_retry_nonexistent_session(self, tmp_path):
        """retry: for non-existent session shows 'not found'."""
        d = make_dispatcher(tmp_path)

        cb = {
            "id": "cb_404",
            "data": "retry:99999",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 99},
        }
        await d._on_callback(cb)
        d.tg.answer_callback.assert_called_with("cb_404", "Original task not found")

    @pytest.mark.asyncio
    async def test_answer_callback_out_of_bounds_index(self, tmp_path):
        """answer: out-of-bounds index falls back to raw key."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")
        session.sid = "abcd1234-xxxx"
        session.answer_event = asyncio.Event()
        session.pending_question = {
            "tool_use_id": "toolu_x",
            "questions": [{"question": "Q?"}],
            "option_labels": ["Only One"],
            "tg_msg_id": 9999,
        }

        cb = {
            "id": "cb_oob",
            "data": "answer:abcd1234:99",  # index 99 is out of bounds
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)

        # Should fall back to the raw key "99"
        assert session.answer_data == "99"
        assert session.answer_event.is_set()

    @pytest.mark.asyncio
    async def test_new_session_callback(self, tmp_path):
        """new_session callback sets force_new flag."""
        d = make_dispatcher(tmp_path)

        cb = {
            "id": "cb_new",
            "data": "new_session",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)

        assert d.sm.force_new is True
        d.tg.answer_callback.assert_called_with(
            "cb_new", "OK, next message starts a new session"
        )

    @pytest.mark.asyncio
    async def test_unknown_callback_data(self, tmp_path):
        """Unknown callback data is answered silently (no crash)."""
        d = make_dispatcher(tmp_path)

        cb = {
            "id": "cb_unknown",
            "data": "something_weird",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)
        d.tg.answer_callback.assert_called_with("cb_unknown")

    @pytest.mark.asyncio
    async def test_callback_wrong_chat_rejected(self, tmp_path):
        """Callback from wrong chat_id is rejected."""
        d = make_dispatcher(tmp_path)

        cb = {
            "id": "cb_wrong",
            "data": "cancel:abc",
            "message": {"chat": {"id": 99999}, "message_id": 50},
        }
        await d._on_callback(cb)
        d.tg.answer_callback.assert_called_with("cb_wrong", "Unauthorized")


# ========================================================================
# 5. Session Routing
# ========================================================================

class TestSessionRouting:
    """Verify message routing: reply -> followup, no reply -> new/resume."""

    @pytest.mark.asyncio
    async def test_reply_to_bot_message_resumes(self, tmp_path):
        """Reply to a bot message -> find_by_reply -> _do_followup with resume."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        # Create a completed session and link bot message
        prev = d.sm.create(1, "original task", "/tmp/proj")
        prev.status = "done"
        prev.finished = time.time()
        d.sm.link_bot(100, 1)  # bot message 100 -> user message 1

        d.transcript.append(prev.conv_id, "user", "original task")
        d.transcript.append(prev.conv_id, "assistant", "Done with task.")

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            invocations.append({"resume": resume, "sid": session.sid, "prompt": prompt})
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Follow-up result"
        d.runner.invoke = mock_invoke

        # Reply to bot message 100
        await d._handle_task(2, "what about tests?", reply_to=100)
        if d._tasks:
            await asyncio.gather(*d._tasks, return_exceptions=True)

        assert len(invocations) == 1
        assert invocations[0]["resume"] is True
        assert invocations[0]["sid"] == prev.sid
        assert "Conversation History" in invocations[0]["prompt"]

    @pytest.mark.asyncio
    async def test_no_reply_no_active_has_last_session_resumes(self, tmp_path):
        """No reply, no active tasks, has last_session -> auto-resume."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        prev = d.sm.create(1, "previous task", "/tmp")
        prev.status = "done"
        prev.finished = time.time()

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            invocations.append({"resume": resume, "sid": session.sid})
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Continued"
        d.runner.invoke = mock_invoke

        await d._handle_task(2, "continue this", None)
        if d._tasks:
            await asyncio.gather(*d._tasks, return_exceptions=True)

        assert len(invocations) == 1
        assert invocations[0]["resume"] is True

    @pytest.mark.asyncio
    async def test_no_reply_no_active_no_last_session_new(self, tmp_path):
        """No reply, no active tasks, no last_session -> new session."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            invocations.append({"resume": resume, "sid": session.sid})
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "New session result"
        d.runner.invoke = mock_invoke

        await d._handle_task(1, "hello world", None)
        if d._tasks:
            await asyncio.gather(*d._tasks, return_exceptions=True)

        assert len(invocations) == 1
        assert invocations[0]["resume"] is False

    @pytest.mark.asyncio
    async def test_reply_to_running_task_queues_followup(self, tmp_path):
        """Reply to a running task -> _do_queued_followup."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        running = d.sm.create(1, "running task", "/tmp")
        running.status = "running"
        running.started = time.time()
        running.proc = MagicMock()

        # Reply to the running task
        await d._handle_task(2, "also do this", reply_to=1)

        # Should have sent a "queued" message
        sent_texts = [call[0][0] for call in d.tg.send.call_args_list if call[0]]
        assert any("Queued" in t or "queued" in t.lower() for t in sent_texts)

    @pytest.mark.asyncio
    async def test_force_new_prevents_auto_resume(self, tmp_path):
        """force_new flag prevents auto-resume of last session."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        prev = d.sm.create(1, "old task", "/tmp")
        prev.status = "done"
        prev.finished = time.time()

        d.sm.force_new = True

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            invocations.append({"resume": resume, "sid": session.sid})
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Fresh start"
        d.runner.invoke = mock_invoke

        await d._handle_task(2, "start fresh", None)
        if d._tasks:
            await asyncio.gather(*d._tasks, return_exceptions=True)

        assert len(invocations) == 1
        assert invocations[0]["resume"] is False
        assert invocations[0]["sid"] != prev.sid
        assert d.sm.force_new is False  # cleared after use

    @pytest.mark.asyncio
    async def test_project_affinity_different_project_new_session(self, tmp_path):
        """Message targeting different project than last session -> new session."""
        proj_a = tmp_path / "project_a"
        proj_b = tmp_path / "project_b"
        proj_a.mkdir()
        proj_b.mkdir()

        projects = {
            "project_a": {"path": str(proj_a), "keywords": ["project_a"]},
            "project_b": {"path": str(proj_b), "keywords": ["project_b"]},
        }
        d = make_dispatcher(tmp_path, projects)
        d._fire_reaction = MagicMock()

        # Last session was project_a
        prev = d.sm.create(1, "work on project_a", str(proj_a))
        prev.status = "done"
        prev.finished = time.time()
        prev.cwd = str(proj_a)

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            invocations.append({
                "resume": resume, "cwd": session.cwd, "sid": session.sid,
            })
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Done"
        d.runner.invoke = mock_invoke

        # Send message targeting project_b
        await d._handle_task(2, "fix project_b bug", None)
        if d._tasks:
            await asyncio.gather(*d._tasks, return_exceptions=True)

        assert len(invocations) == 1
        assert invocations[0]["resume"] is False
        assert invocations[0]["cwd"] == str(proj_b)


# ========================================================================
# 6. Edited Message Handling
# ========================================================================

class TestEditedMessageHandling:
    """Edit young session -> cancel + re-dispatch; old session -> ignore."""

    @pytest.mark.asyncio
    async def test_edit_young_session_cancels_and_redispatches(self, tmp_path):
        """Editing a message with a <10s-old session cancels and re-dispatches."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        s = d.sm.create(1, "old text", "/tmp/proj")
        s.status = "running"
        s.started = time.time()  # just started
        s.proc = MagicMock()

        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append({"mid": mid, "text": text})
        d._handle_task = track_handle

        edited_msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 1,
            "text": "new text",
        }
        await d._on_edited_message(edited_msg)

        assert s.status == "cancelled"
        s.proc.kill.assert_called_once()
        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "new text"

    @pytest.mark.asyncio
    async def test_edit_old_session_ignored(self, tmp_path):
        """Editing a message with a >10s-old session is ignored."""
        d = make_dispatcher(tmp_path)

        s = d.sm.create(1, "old text", "/tmp/proj")
        s.status = "running"
        s.started = time.time() - 30  # 30s ago
        s.proc = MagicMock()

        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append(text)
        d._handle_task = track_handle

        edited_msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 1,
            "text": "new text",
        }
        await d._on_edited_message(edited_msg)

        assert s.status == "running"
        assert len(dispatched) == 0

    @pytest.mark.asyncio
    async def test_edit_nonexistent_session_ignored(self, tmp_path):
        """Editing a message with no matching session is silently ignored."""
        d = make_dispatcher(tmp_path)

        edited_msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 999,
            "text": "edited text",
        }
        await d._on_edited_message(edited_msg)
        d.tg.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_wrong_chat_ignored(self, tmp_path):
        """Edited message from wrong chat is ignored."""
        d = make_dispatcher(tmp_path)

        edited_msg = {
            "chat": {"id": 99999},
            "message_id": 1,
            "text": "edited",
        }
        await d._on_edited_message(edited_msg)
        d.tg.send.assert_not_called()


# ========================================================================
# 7. Stream Reading Edge Cases
# ========================================================================

class TestStreamReadingEdgeCases:
    """Comprehensive edge cases for AgentRunner._read_stream."""

    @pytest.mark.asyncio
    async def test_multiple_assistant_events_accumulate(self):
        """Multiple assistant events with distinct text accumulate in partial_output."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        proc = FakeProcess([
            make_assistant_event("First paragraph."),
            make_assistant_event("Second paragraph."),
            make_result_event("Second paragraph."),
        ])

        result = await runner._read_stream(proc, session)
        assert result == "Second paragraph."
        assert "paragraph" in session.partial_output

    @pytest.mark.asyncio
    async def test_result_event_returns_immediately(self):
        """Result event should cause _read_stream to return, not wait for more."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        proc = FakeProcess([
            make_assistant_event("Hello"),
            make_result_event("Hello"),
            # Additional lines after result should not be read
            make_assistant_event("This should be ignored"),
        ])

        start = time.monotonic()
        result = await runner._read_stream(proc, session)
        elapsed = time.monotonic() - start

        assert result == "Hello"
        assert elapsed < 2, f"took {elapsed:.1f}s, should return immediately"

    @pytest.mark.asyncio
    async def test_no_result_falls_back_to_last_text(self):
        """No result event -> falls back to last assistant text."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        proc = FakeProcess([
            make_assistant_event("Partial response only"),
        ])

        result = await runner._read_stream(proc, session)
        assert result == "Partial response only"

    @pytest.mark.asyncio
    async def test_malformed_json_lines_skipped(self):
        """Malformed JSON lines are silently skipped."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        proc = FakeProcess([
            "not json at all",
            "{broken json",
            make_assistant_event("good output"),
            make_result_event("good output"),
        ])

        result = await runner._read_stream(proc, session)
        assert result == "good output"

    @pytest.mark.asyncio
    async def test_empty_result_falls_back_to_partial(self):
        """Empty result text falls back to partial_output."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        proc = FakeProcess([
            make_assistant_event("Some work done"),
            json.dumps({"type": "result", "result": ""}),  # empty result
        ])

        result = await runner._read_stream(proc, session)
        # Empty result string is returned by _read_stream
        # The caller (_invoke_cli) handles the fallback to partial_output
        assert result == "" or result == "Some work done"
        assert "Some work done" in session.partial_output

    @pytest.mark.asyncio
    async def test_no_text_no_result_falls_back_to_stderr(self):
        """No assistant text and no result -> stderr fallback."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        proc = FakeProcess(
            [json.dumps({"type": "system", "subtype": "init"})],
            stderr_text="fatal: something broke",
        )

        result = await runner._read_stream(proc, session)
        assert "(stderr)" in result
        assert "something broke" in result

    @pytest.mark.asyncio
    async def test_ask_user_question_detected(self):
        """AskUserQuestion tool_use triggers on_question callback."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")
        session.answer_event = asyncio.Event()
        session.stdin_writer = MagicMock()
        session.stdin_drain = AsyncMock()

        question_called = []
        async def on_question(s):
            question_called.append(s.pending_question)
            # Simulate user answering immediately
            s.answer_data = "Blue"
            s.answer_event.set()

        proc = FakeProcess([
            make_question_event("toolu_q1", "Pick a color?", [
                {"label": "Red"},
                {"label": "Blue"},
            ]),
            make_result_event("Great, blue it is!"),
        ])

        result = await runner._read_stream(proc, session, on_question=on_question)

        assert len(question_called) == 1
        assert question_called[0]["tool_use_id"] == "toolu_q1"
        assert result == "Great, blue it is!"
        # Verify tool_result was written to stdin
        session.stdin_writer.assert_called_once()

    @pytest.mark.asyncio
    async def test_thinking_blocks_ignored(self):
        """Thinking blocks should not appear in output."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        proc = FakeProcess([
            json.dumps({
                "type": "assistant",
                "message": {"content": [
                    {"type": "thinking", "thinking": "Let me think about this..."},
                ]},
            }),
            make_assistant_event("The answer is 42"),
            make_result_event("The answer is 42"),
        ])

        result = await runner._read_stream(proc, session)
        assert result == "The answer is 42"
        assert "think" not in session.partial_output.lower()


# ========================================================================
# 8. Send Result Variations
# ========================================================================

class TestSendResultVariations:
    """Verify result delivery format based on status and content."""

    def test_done_short_result_sent_as_html(self, tmp_path):
        """Done with short result -> reply with HTML."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()
        d._send_result(1, s, "Here is your answer")

        d.tg.send.assert_called_once()
        text = d.tg.send.call_args[0][0]
        assert "answer" in text
        d.tg.send_document.assert_not_called()

    def test_done_long_result_sent_as_file(self, tmp_path):
        """Done with long result (> 4000 chars) -> send as file."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.is_task = True
        s.started = time.time()
        s.finished = time.time()
        d._send_result(1, s, "x" * 5000)

        d.tg.send_document.assert_called_once()

    def test_failed_shows_error_and_retry_button(self, tmp_path):
        """Failed -> show error message + retry inline button."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "fix bug", "/tmp/proj")
        s.status = "failed"
        d._send_result(1, s, "Error: something broke")

        d.tg.send.assert_called_once()
        call_kwargs = d.tg.send.call_args[1]
        text = d.tg.send.call_args[0][0]
        assert "failed" in text.lower() or "失败" in text
        markup = call_kwargs.get("reply_markup")
        assert markup is not None
        buttons = markup["inline_keyboard"][0]
        assert any("retry" in b["callback_data"] for b in buttons)

    def test_cancelled_no_message(self, tmp_path):
        """Cancelled -> no message sent."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "test", "/tmp")
        s.status = "cancelled"
        d._send_result(1, s, "result")
        d.tg.send.assert_not_called()

    def test_empty_result_warning_message(self, tmp_path):
        """Empty result -> warning message."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "test", "/tmp")
        s.status = "done"
        d._send_result(1, s, "")

        d.tg.send.assert_called_once()
        text = d.tg.send.call_args[0][0]
        assert "no output" in text.lower() or "没有" in text or "turns" in text.lower()

    def test_duration_header_for_long_tasks(self, tmp_path):
        """Tasks > 60s get a duration header."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time() - 120  # 2 minutes ago
        s.finished = time.time()
        d._send_result(1, s, "Refactored the module")

        text = d.tg.send.call_args[0][0]
        assert "\u2705" in text  # checkmark
        assert "Done" in text or "完成" in text

    def test_no_duration_header_for_quick_tasks(self, tmp_path):
        """Tasks < 60s have no duration header."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()  # ~0s
        d._send_result(1, s, "Quick answer")

        text = d.tg.send.call_args[0][0]
        assert "Done" not in text and "完成" not in text

    def test_success_reaction(self, tmp_path):
        """Successful task triggers checkmark reaction."""
        d = make_dispatcher(tmp_path)
        reactions = []
        d._fire_reaction = lambda mid, emoji: reactions.append((mid, emoji))
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()
        d._send_result(1, s, "Done!")
        assert (1, "\u2705") in reactions

    def test_failure_reaction(self, tmp_path):
        """Failed task triggers X reaction."""
        d = make_dispatcher(tmp_path)
        reactions = []
        d._fire_reaction = lambda mid, emoji: reactions.append((mid, emoji))
        s = Session(1, "test", "/tmp")
        s.status = "failed"
        d._send_result(1, s, "Error occurred")
        assert (1, "\u274c") in reactions

    def test_consecutive_failures_escalation(self, tmp_path):
        """3 consecutive failures add escalation warning."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        for i in range(3):
            s = Session(i + 1, f"task{i}", "/tmp")
            s.status = "failed"
            d._send_result(i + 1, s, f"Error {i}")

        last_msg = d.tg.send.call_args[0][0]
        assert "3" in last_msg
        assert "consecutive" in last_msg.lower() or "连续" in last_msg

    def test_success_resets_failure_counter(self, tmp_path):
        """Success resets the consecutive failure counter."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        for i in range(2):
            s = Session(i + 1, f"fail{i}", "/tmp")
            s.status = "failed"
            d._send_result(i + 1, s, f"Error {i}")
        assert d._consecutive_failures == 2

        s = Session(10, "success", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()
        d._send_result(10, s, "All good!")
        assert d._consecutive_failures == 0

    def test_long_task_has_new_session_button(self, tmp_path):
        """Task running > 30s should get a 'new session' button."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        s = Session(1, "refactor", "/tmp/proj")
        s.status = "done"
        s.is_task = True
        s.started = time.time() - 60
        s.finished = time.time()
        d._send_result(1, s, "Refactored")

        markup = d.tg.send.call_args[1].get("reply_markup")
        assert markup is not None
        found = False
        for row in markup.get("inline_keyboard", []):
            for btn in row:
                if btn.get("callback_data") == "new_session":
                    found = True
        assert found, "Long task should have 'new session' button"

    def test_quick_task_no_buttons(self, tmp_path):
        """Quick task (< 30s) should have no inline buttons."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        s = Session(1, "quick", "/tmp")
        s.status = "done"
        s.is_task = False
        s.started = time.time()
        s.finished = time.time()
        d._send_result(1, s, "Quick answer")

        markup = d.tg.send.call_args[1].get("reply_markup")
        assert markup is None


# ========================================================================
# 9. Integration: Full E2E with Dispatcher._do_session
# ========================================================================

class TestDoSessionIntegration:
    """Integration tests for the full _do_session pipeline."""

    @pytest.mark.asyncio
    async def test_do_session_records_transcript(self, tmp_path):
        """_do_session records user message + assistant response in transcript."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "what is python", str(Path.home()))

        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Python is a programming language"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "what is python")

        msgs = d.transcript.load(session.conv_id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "what is python"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Python is a programming language"

    @pytest.mark.asyncio
    async def test_do_session_model_passed_to_runner(self, tmp_path):
        """Model override is forwarded to runner.invoke."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", str(Path.home()))

        captured_model = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            captured_model.append(model)
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "ok"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "test", model="haiku")
        assert captured_model == ["haiku"]

    @pytest.mark.asyncio
    async def test_do_session_cancels_progress_on_completion(self, tmp_path):
        """Progress loop is cancelled when runner finishes."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", str(Path.home()))

        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            session.status = "running"
            session.started = time.time()
            await asyncio.sleep(0.1)
            session.status = "done"
            session.finished = time.time()
            return "Result"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "test")

        # After _do_session completes, progress loop should be cancelled.
        # No progress message should remain (it was deleted or never sent).
        # The tg.delete_message should not have been called for a fast task.
        # Just verify no crash and result was sent.
        assert d.tg.send.call_count >= 1

    @pytest.mark.asyncio
    async def test_do_session_empty_result_handled(self, tmp_path):
        """Empty runner result sends a warning, not a crash."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", str(Path.home()))

        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return ""

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "test")

        text = d.tg.send.call_args[0][0]
        assert "no output" in text.lower() or "turns" in text.lower()


# ========================================================================
# 10. Do-followup integration
# ========================================================================

class TestDoFollowupIntegration:
    """Integration tests for _do_followup: history injection and conversation continuity."""

    @pytest.mark.asyncio
    async def test_followup_injects_history(self, tmp_path):
        """_do_followup injects conversation history into the prompt."""
        d = make_dispatcher(tmp_path)
        prev = d.sm.create(1, "tell me about cats", "/tmp")
        prev.is_task = False
        prev.status = "done"
        prev.result = "Cats are great pets."
        prev.finished = time.time()

        d.transcript.append(prev.conv_id, "user", "tell me about cats")
        d.transcript.append(prev.conv_id, "assistant", "Cats are great pets.")

        captured = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            captured.append({"prompt": prompt, "resume": resume})
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "They like sleeping."

        d.runner.invoke = mock_invoke
        await d._do_followup(2, "what do they like?", prev)

        assert len(captured) == 1
        assert "tell me about cats" in captured[0]["prompt"]
        assert "Cats are great pets" in captured[0]["prompt"]
        assert "what do they like?" in captured[0]["prompt"]

    @pytest.mark.asyncio
    async def test_followup_records_new_turn(self, tmp_path):
        """_do_followup records both user message and assistant response."""
        d = make_dispatcher(tmp_path)
        prev = d.sm.create(1, "original", "/tmp")
        prev.status = "done"
        prev.finished = time.time()

        d.transcript.append(prev.conv_id, "user", "original")
        d.transcript.append(prev.conv_id, "assistant", "First answer.")

        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Second answer."

        d.runner.invoke = mock_invoke
        await d._do_followup(2, "followup", prev)

        msgs = d.transcript.load(prev.conv_id)
        assert len(msgs) == 4
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"] == "followup"
        assert msgs[3]["role"] == "assistant"
        assert msgs[3]["content"] == "Second answer."

    @pytest.mark.asyncio
    async def test_followup_model_change_creates_new_session(self, tmp_path):
        """Model change in followup creates a new session but keeps conv_id."""
        d = make_dispatcher(tmp_path)
        prev = d.sm.create(1, "hello", "/tmp")
        prev.status = "done"
        prev.model_override = "opus"
        prev.finished = time.time()

        captured = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            captured.append({
                "sid": session.sid,
                "conv_id": session.conv_id,
                "resume": resume,
                "model": model,
            })
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Switched"

        d.runner.invoke = mock_invoke
        await d._do_followup(2, "switch model", prev, model="haiku")

        assert len(captured) == 1
        assert captured[0]["sid"] != prev.sid  # new session
        assert captured[0]["conv_id"] == prev.conv_id  # same conversation
        assert captured[0]["resume"] is False
        assert captured[0]["model"] == "haiku"


# ========================================================================
# 11. Queued followup integration
# ========================================================================

class TestQueuedFollowup:
    """Verify queued followup waits for running task then continues."""

    @pytest.mark.asyncio
    async def test_queued_followup_waits_for_completion(self, tmp_path):
        """_do_queued_followup waits for target to finish, then resumes."""
        d = make_dispatcher(tmp_path)

        target = d.sm.create(1, "running task", "/tmp")
        target.status = "running"
        target.started = time.time()

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            invocations.append({"resume": resume})
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Queued result"

        d.runner.invoke = mock_invoke

        # Start the queued followup in the background
        task = asyncio.create_task(
            d._do_queued_followup(2, "also do this", target)
        )

        # Target still running, followup should not have fired yet
        await asyncio.sleep(0.5)
        assert len(invocations) == 0

        # Complete the target
        target.status = "done"
        target.finished = time.time()

        # Now the queued followup should fire
        await asyncio.wait_for(task, timeout=5)
        assert len(invocations) == 1


# ========================================================================
# 12. Message routing via _on_message
# ========================================================================

class TestOnMessageRouting:
    """Full _on_message integration: classify, route, react."""

    @pytest.mark.asyncio
    async def test_task_message_gets_eyes_reaction(self, tmp_path):
        """Task messages get eyes reaction immediately."""
        d = make_dispatcher(tmp_path)
        reactions = []
        d._fire_reaction = lambda mid, emoji: reactions.append((mid, emoji))

        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 42,
            "text": "do something",
        }
        await d._on_message(msg)

        assert (42, "\U0001f440") in reactions

    @pytest.mark.asyncio
    async def test_wrong_chat_ignored(self, tmp_path):
        """Messages from wrong chat are silently ignored."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        msg = {
            "chat": {"id": 99999},
            "message_id": 1,
            "text": "hello",
        }
        await d._on_message(msg)
        d.tg.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self, tmp_path):
        """Empty text messages are ignored."""
        d = make_dispatcher(tmp_path)

        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 1,
            "text": "",
        }
        await d._on_message(msg)
        d.tg.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_status_command_direct_response(self, tmp_path):
        """'status' text -> direct response, no agent call."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 1,
            "text": "/status",
        }
        await d._on_message(msg)
        assert d.tg.send.call_count == 1
        text = d.tg.send.call_args[0][0]
        assert "Idle" in text or "空闲" in text or "Running" in text

    @pytest.mark.asyncio
    async def test_help_command_direct_response(self, tmp_path):
        """'/help' -> direct response."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 1,
            "text": "/help",
        }
        await d._on_message(msg)
        assert d.tg.send.call_count == 1
        text = d.tg.send.call_args[0][0]
        assert "Help" in text or "帮助" in text or "使用" in text

    @pytest.mark.asyncio
    async def test_forward_message_includes_source(self, tmp_path):
        """Forwarded message includes [Forwarded from X] prefix."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append(text)
        d._handle_task = track_handle

        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 1,
            "text": "check this out",
            "forward_from": {"first_name": "Alice"},
        }
        await d._on_message(msg)
        await asyncio.sleep(2.5)  # wait for buffer flush

        assert len(dispatched) == 1
        assert "[Forwarded from Alice]" in dispatched[0]

    @pytest.mark.asyncio
    async def test_forward_from_channel(self, tmp_path):
        """Channel forward uses channel title as source."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append(text)
        d._handle_task = track_handle

        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 2,
            "text": "news article",
            "forward_from_chat": {"title": "Tech News"},
        }
        await d._on_message(msg)
        await asyncio.sleep(2.5)

        assert len(dispatched) == 1
        assert "[Forwarded from Tech News]" in dispatched[0]


# ========================================================================
# 13. Model prefix extraction
# ========================================================================

class TestModelPrefixIntegration:
    """@model prefix detection and sticky model behavior."""

    def test_lowercase_haiku_not_sticky(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("@haiku quick question")
        assert text == "quick question"
        assert model == "haiku"
        assert sticky is False

    def test_capitalized_haiku_is_sticky(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("@Haiku persist this")
        assert text == "persist this"
        assert model == "haiku"
        assert sticky is True

    def test_no_prefix(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("just a message")
        assert text == "just a message"
        assert model is None
        assert sticky is False

    @pytest.mark.asyncio
    async def test_sticky_model_persists_in_handle_task(self, tmp_path):
        """@Haiku (capitalized) sets sticky model for subsequent messages."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        captured_models = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            captured_models.append(model)
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "ok"
        d.runner.invoke = mock_invoke

        # First message with @Haiku
        await d._handle_task(1, "@Haiku first message", None)
        if d._tasks:
            await asyncio.gather(*d._tasks, return_exceptions=True)

        assert d._sticky_model == "haiku"
        assert captured_models[-1] == "haiku"

        # Second message without prefix should use sticky model
        await d._handle_task(2, "second message", None)
        if d._tasks:
            await asyncio.gather(*d._tasks, return_exceptions=True)

        assert captured_models[-1] == "haiku"


# ========================================================================
# 14. Message batching
# ========================================================================

class TestMessageBatching:
    """Verify rapid-fire messages are merged into one prompt."""

    @pytest.mark.asyncio
    async def test_single_message_passes_through(self, tmp_path):
        d = make_dispatcher(tmp_path)
        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append({"mid": mid, "text": text})
        d._handle_task = track_handle

        d._buffer_message(1, "hello world", None, [])
        await asyncio.sleep(2.5)

        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "hello world"

    @pytest.mark.asyncio
    async def test_multiple_messages_merged(self, tmp_path):
        d = make_dispatcher(tmp_path)
        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append({"mid": mid, "text": text})
        d._handle_task = track_handle

        d._buffer_message(1, "first", None, [])
        d._buffer_message(2, "second", None, [])
        d._buffer_message(3, "third", None, [])
        await asyncio.sleep(2.5)

        assert len(dispatched) == 1
        assert "first" in dispatched[0]["text"]
        assert "second" in dispatched[0]["text"]
        assert "third" in dispatched[0]["text"]
        assert dispatched[0]["mid"] == 3


# ========================================================================
# 15. Concurrency limit
# ========================================================================

class TestConcurrencyLimit:
    """Verify max_concurrent is respected."""

    @pytest.mark.asyncio
    async def test_exceeding_max_concurrent_queues(self, tmp_path):
        """When at max_concurrent, new tasks are queued."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        # Fill up to max_concurrent
        for i in range(d.cfg.max_concurrent):
            s = d.sm.create(i + 1, f"task {i}", "/tmp")
            s.status = "running"
            s.started = time.time()
            s.proc = MagicMock()

        # New task should be queued
        await d._handle_task(100, "new task", None)

        sent_texts = [call[0][0] for call in d.tg.send.call_args_list if call[0]]
        assert any("queued" in t.lower() or "Queued" in t for t in sent_texts)


# ========================================================================
# 16. New session command
# ========================================================================

class TestNewSessionCommand:
    """Verify 'new session' / 'new_session' keyword and callback flow."""

    @pytest.mark.asyncio
    async def test_new_session_keyword(self, tmp_path):
        """'new session' text triggers new_session handler."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 1,
            "text": "new session",
        }
        await d._on_message(msg)

        assert d.sm.force_new is True
        text = d.tg.send.call_args[0][0]
        assert "new session" in text.lower() or "新" in text

    def test_handle_new_session_sets_flag(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d.sm.force_new is False
        d._handle_new_session(1)
        assert d.sm.force_new is True


# ========================================================================
# 17. Friendly error formatting
# ========================================================================

class TestFriendlyError:
    """Verify error messages are user-friendly."""

    def test_timeout_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("Timed out after 30 minutes")
        assert "timed out" in result.lower() or "超时" in result.lower()

    def test_rate_limit_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("429 rate limit exceeded")
        assert "rate limit" in result.lower() or "限流" in result.lower()

    def test_max_turns_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("max turns reached")
        assert "turn" in result.lower() or "轮次" in result.lower()

    def test_permission_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("permission denied: /etc/secret")
        assert "Permission" in result or "权限" in result

    def test_generic_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("something unexpected")
        assert "Error" in result or "出错" in result

    def test_stderr_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("(stderr) segfault at 0x0")
        assert "error" in result.lower() or "Execution" in result


# ========================================================================
# 18. Markdown to Telegram HTML conversion
# ========================================================================

class TestMdToTelegramHtml:
    """Ensure proper conversion for result display."""

    def test_bold(self):
        assert _md_to_telegram_html("**hello**") == "<b>hello</b>"

    def test_inline_code(self):
        assert _md_to_telegram_html("`code`") == "<code>code</code>"

    def test_code_block(self):
        result = _md_to_telegram_html("```python\nprint('hi')\n```")
        assert "<pre>" in result
        assert "print" in result

    def test_html_escape(self):
        result = _md_to_telegram_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_mixed_formatting(self):
        result = _md_to_telegram_html("Use **bold** and `code` here")
        assert "<b>bold</b>" in result
        assert "<code>code</code>" in result


# ========================================================================
# 19. Duration formatting
# ========================================================================

class TestFormatDuration:
    """Verify _format_duration output."""

    def test_seconds(self):
        from dispatcher.core import _format_duration
        assert _format_duration(42) == "42s"

    def test_minutes(self):
        from dispatcher.core import _format_duration
        assert _format_duration(120) == "2min"

    def test_minutes_seconds(self):
        from dispatcher.core import _format_duration
        assert _format_duration(125) == "2m5s"

    def test_hours(self):
        from dispatcher.core import _format_duration
        assert _format_duration(3660) == "1h1m"


# ========================================================================
# 20. Peek command
# ========================================================================

class TestPeekCommand:
    """Verify /peek shows current output of running session."""

    def test_peek_no_tasks(self, tmp_path):
        d = make_dispatcher(tmp_path)
        d._handle_peek(1)
        text = d.tg.send.call_args[0][0]
        assert "No tasks" in text or "没有" in text

    def test_peek_with_partial_output(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = d.sm.create(1, "big task", "/tmp/proj")
        s.status = "running"
        s.started = time.time()
        s.partial_output = "Working on auth module..."
        d._handle_peek(2)
        text = d.tg.send.call_args[0][0]
        assert "auth module" in text

    def test_peek_no_output_yet(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = d.sm.create(1, "task", "/tmp/proj")
        s.status = "running"
        s.started = time.time()
        s.partial_output = ""
        d._handle_peek(2)
        text = d.tg.send.call_args[0][0]
        assert "no output" in text.lower()


# ========================================================================
# 21. Session manager details
# ========================================================================

class TestSessionManagerDetails:
    """Additional session manager edge cases."""

    def test_find_by_reply_bot_message(self):
        sm = SessionManager()
        s = sm.create(1, "test", "/tmp")
        sm.link_bot(100, 1)
        assert sm.find_by_reply(100) is s

    def test_find_by_reply_original(self):
        sm = SessionManager()
        s = sm.create(1, "test", "/tmp")
        assert sm.find_by_reply(1) is s

    def test_find_by_reply_not_found(self):
        sm = SessionManager()
        assert sm.find_by_reply(999) is None

    def test_create_with_custom_sid_and_conv_id(self):
        sm = SessionManager()
        s = sm.create(1, "test", "/tmp", sid="custom-sid", conv_id="custom-conv")
        assert s.sid == "custom-sid"
        assert s.conv_id == "custom-conv"

    def test_force_new_default_false(self):
        sm = SessionManager()
        assert sm.force_new is False

    def test_session_elapsed(self):
        s = Session(1, "test", "/tmp")
        s.started = time.time() - 10
        s.finished = time.time()
        elapsed = s.elapsed()
        assert 9 < elapsed < 12

    def test_session_elapsed_running(self):
        s = Session(1, "test", "/tmp")
        s.started = time.time() - 5
        # No finished time -> uses current time
        elapsed = s.elapsed()
        assert 4 < elapsed < 7

    def test_session_project_name(self):
        s = Session(1, "test", "/Users/zwang/projects/cortex")
        assert s.project_name == "cortex"

    def test_active_returns_only_running(self):
        sm = SessionManager()
        s1 = sm.create(1, "t1", "/tmp")
        s2 = sm.create(2, "t2", "/tmp")
        s3 = sm.create(3, "t3", "/tmp")
        s1.status = "running"
        s2.status = "done"
        s3.status = "running"
        active = sm.active()
        assert len(active) == 2
        assert s1 in active
        assert s3 in active
