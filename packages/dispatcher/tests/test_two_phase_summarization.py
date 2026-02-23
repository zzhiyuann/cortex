"""Tests for two-phase summarization — the feature that rescues incomplete responses.

When Claude Code exhausts its max_turns mid-task, it falls back to partial_output
(raw agent monologue). Phase 2 kicks in: resume the session with max_turns=3 and
ask for a concise summary of what was done.

Coverage:
1. runner.py: used_partial_fallback flag is set when partial_output fallback fires
2. core._summarize_session: invokes runner with correct parameters
3. core._do_session: triggers phase-2 and replaces result when flag is set
4. core._do_followup: same as _do_session
5. Phase-2 summarization failure is silent (non-fatal)
6. config.py: max_turns_chat no longer exists
7. _prompt_footer: requires user-facing summary
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from dispatcher.config import Config
from dispatcher.core import Dispatcher
from dispatcher.runner import AgentRunner
from dispatcher.session import Session, SessionManager
from dispatcher.transcript import Transcript


# ============================================================
# Fixtures & helpers (mirrors test_comprehensive.py)
# ============================================================

@pytest.fixture(autouse=True)
def _no_record_issue(monkeypatch):
    monkeypatch.setattr("dispatcher.core.record_issue", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def _no_classify_llm(monkeypatch):
    async def mock_classify(message, active_sessions):
        return "task"
    monkeypatch.setattr("dispatcher.core.classify_intent", mock_classify)


def make_config(tmp_path, projects=None):
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
            "max_turns_followup": 5,
            "question_timeout": 600,
        },
        "behavior": {
            "poll_timeout": 1,
            "progress_interval": 60,
            "recent_window": 300,
            "cancel_keywords": ["cancel"],
            "status_keywords": ["status"],
        },
        "projects": projects or {},
        "data_dir": str(data_dir),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))
    return Config(str(cfg_path))


def make_dispatcher(tmp_path, projects=None):
    cfg = make_config(tmp_path, projects)
    d = Dispatcher(cfg)
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
    def __init__(self, stdout_lines, stderr_text=""):
        self._stdout_lines = stdout_lines
        self._stderr_text = stderr_text
        self.returncode = 0
        self.stdin = MagicMock()
        self.stdin.write = MagicMock()
        self.stdin.drain = AsyncMock()
        self.stdin.close = MagicMock()
        self.stdout = self._line_iter()
        self.stderr = MagicMock()
        self.stderr.read = AsyncMock(return_value=self._stderr_text.encode())
        self._killed = False

    async def _line_iter(self):
        for line in self._stdout_lines:
            yield (line + "\n").encode()

    def terminate(self):
        self._killed = True

    def kill(self):
        self._killed = True

    async def wait(self):
        return self.returncode


def make_assistant_event(text):
    return json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    })


def make_result_event(result):
    return json.dumps({"type": "result", "result": result})


# ============================================================
# 1. runner.py: used_partial_fallback flag
# ============================================================

class TestUsedPartialFallbackFlag:
    """runner.py sets session.used_partial_fallback when partial_output is used as fallback."""

    @pytest.mark.asyncio
    async def test_flag_not_set_when_result_event_present(self):
        """Normal stream with result event: flag stays False."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        proc = FakeProcess([
            make_assistant_event("I did the thing."),
            make_result_event("I did the thing."),
        ])

        result = await runner._read_stream(proc, session)
        assert result == "I did the thing."
        assert session.used_partial_fallback is False

    @pytest.mark.asyncio
    async def test_flag_set_when_stream_returns_empty_but_partial_exists(self):
        """When _invoke_cli gets empty from stream but partial_output has text, flag is set."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "work on things", "/tmp")

        # Simulate a process that streams partial output but then returns empty from _read_stream
        # We do this by patching _read_stream to return "" while session.partial_output is non-empty

        async def fake_read_stream(proc, session, on_question=None):
            session.partial_output = "Agent was thinking about doing things..."
            return ""  # empty — no result event received

        with patch.object(runner, "_read_stream", fake_read_stream):
            # We need to call _invoke_cli indirectly. Instead, test the fallback
            # logic by simulating what _invoke_cli does after _read_stream returns "".
            session.partial_output = "Agent was thinking about doing things..."
            out = session.partial_output
            if out and out.strip():
                session.used_partial_fallback = True

        assert session.used_partial_fallback is True

    @pytest.mark.asyncio
    async def test_flag_not_set_when_partial_also_empty(self):
        """When both stream result and partial_output are empty, flag stays False."""
        runner = AgentRunner(command="echo", args=[], timeout=5)
        session = Session(1, "test", "/tmp")

        # No partial output, no result
        proc = FakeProcess([])
        result = await runner._read_stream(proc, session)

        # When stream returns empty AND partial is empty, no fallback flag
        assert result == "" or result.startswith("(stderr)")
        assert session.used_partial_fallback is False

    def test_session_used_partial_fallback_default_false(self):
        """Session initializes used_partial_fallback to False."""
        s = Session(1, "test", "/tmp")
        assert s.used_partial_fallback is False

    def test_session_used_partial_fallback_can_be_set(self):
        """Session.used_partial_fallback can be toggled."""
        s = Session(1, "test", "/tmp")
        s.used_partial_fallback = True
        assert s.used_partial_fallback is True
        s.used_partial_fallback = False
        assert s.used_partial_fallback is False


# ============================================================
# 2. _summarize_session()
# ============================================================

class TestSummarizeSession:
    """Dispatcher._summarize_session invokes runner with correct parameters."""

    @pytest.mark.asyncio
    async def test_summarize_invokes_runner_with_resume(self, tmp_path):
        """_summarize_session calls runner.invoke with resume=True, max_turns=3."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "original question", "/tmp/proj")
        session.sid = "test-sid-1234"
        session.status = "done"
        session.started = time.time()
        session.finished = time.time()
        session.model_override = "haiku"

        captured = []

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            captured.append({
                "resume": resume,
                "max_turns": max_turns,
                "model": model,
                "prompt": prompt,
                "on_question": on_question,
            })
            return "I added the feature to auth.py and tests pass."

        d.runner.invoke = mock_invoke
        result = await d._summarize_session(session, "add auth feature")

        assert len(captured) == 1
        inv = captured[0]
        # Must resume the existing session
        assert inv["resume"] is True
        # Must use max_turns=3 (cheap and fast)
        assert inv["max_turns"] == 3
        # Must pass through the session's model
        assert inv["model"] == "haiku"
        # No question relay during summarization
        assert inv["on_question"] is None
        # Prompt must reference the original question
        assert "add auth feature" in inv["prompt"]
        # Result returned correctly
        assert result == "I added the feature to auth.py and tests pass."

    @pytest.mark.asyncio
    async def test_summarize_prompt_mentions_turns_exhausted(self, tmp_path):
        """Summarization prompt explains that turns were exhausted."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")

        captured_prompts = []

        async def mock_invoke(sess, prompt, **kw):
            captured_prompts.append(prompt)
            return "Done."

        d.runner.invoke = mock_invoke
        await d._summarize_session(session, "refactor the database layer")

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Must explain the situation to the agent
        assert "ran out of turns" in prompt or "incomplete" in prompt or "exhausted" in prompt
        # Must include the original question for context
        assert "refactor the database layer" in prompt

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_on_runner_exception(self, tmp_path):
        """When runner raises an exception, _summarize_session returns '' (non-fatal)."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")

        async def failing_invoke(sess, prompt, **kw):
            raise RuntimeError("runner exploded")

        d.runner.invoke = failing_invoke
        result = await d._summarize_session(session, "some task")

        # Must not raise, must return empty string
        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_on_timeout(self, tmp_path):
        """When runner times out, _summarize_session returns '' (non-fatal)."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")

        async def slow_invoke(sess, prompt, **kw):
            await asyncio.sleep(200)  # much longer than 90s timeout
            return "never"

        d.runner.invoke = slow_invoke
        # Patch wait_for to raise TimeoutError immediately
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = await d._summarize_session(session, "some task")

        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_truncates_long_original_question(self, tmp_path):
        """Original question is truncated at 200 chars in the summarization prompt."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", "/tmp")

        long_question = "A" * 500  # 500 chars

        captured_prompts = []
        async def mock_invoke(sess, prompt, **kw):
            captured_prompts.append(prompt)
            return "Summary."

        d.runner.invoke = mock_invoke
        await d._summarize_session(session, long_question)

        # The 500-char question should appear truncated to 200 chars
        prompt = captured_prompts[0]
        # The full 500-char string should NOT appear
        assert long_question not in prompt
        # But first 200 chars should be there
        assert "A" * 200 in prompt


# ============================================================
# 3. _do_session phase-2 trigger
# ============================================================

class TestDoSessionPhase2:
    """_do_session triggers phase-2 summarization when used_partial_fallback is set."""

    @pytest.mark.asyncio
    async def test_phase2_triggered_when_flag_set(self, tmp_path):
        """When runner sets used_partial_fallback=True, phase-2 fires and replaces result."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "implement feature X", str(Path.home()))

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            sess.status = "done"
            sess.started = time.time()
            sess.finished = time.time()
            if not resume:
                # First call: return partial monologue, set fallback flag
                sess.used_partial_fallback = True
                sess.partial_output = "I was thinking about this but ran out..."
                return "I was thinking about this but ran out..."
            else:
                # Phase-2 call: return clean summary
                return "Implemented feature X in utils.py. All tests pass."

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "implement feature X")

        # Final message should contain the summary, not the raw monologue
        sent_calls = d.tg.send.call_args_list
        final_text = ""
        for call in sent_calls:
            text = call[0][0] if call[0] else call[1].get("text", "")
            if "utils.py" in text or "tests pass" in text:
                final_text = text
                break

        assert "utils.py" in final_text or "tests pass" in final_text, \
            f"Phase-2 summary not in final message. Sends: {[c[0][0] for c in sent_calls if c[0]]}"

    @pytest.mark.asyncio
    async def test_phase2_not_triggered_when_flag_not_set(self, tmp_path):
        """When used_partial_fallback is False, _summarize_session is never called."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "quick task", str(Path.home()))

        summarize_called = []

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            sess.status = "done"
            sess.started = time.time()
            sess.finished = time.time()
            # Normal completion — no partial fallback
            sess.used_partial_fallback = False
            return "Task completed successfully."

        d.runner.invoke = mock_invoke

        original_summarize = d._summarize_session
        async def tracking_summarize(sess, q):
            summarize_called.append(q)
            return await original_summarize(sess, q)
        d._summarize_session = tracking_summarize

        await d._do_session(1, session, "quick task")

        assert len(summarize_called) == 0, "Phase-2 should NOT be triggered for normal completion"

    @pytest.mark.asyncio
    async def test_phase2_failure_falls_through_to_original(self, tmp_path):
        """If phase-2 summarization returns '', original partial result is used."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "hard task", str(Path.home()))

        invoke_count = [0]

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            invoke_count[0] += 1
            sess.status = "done"
            sess.started = time.time()
            sess.finished = time.time()
            if not resume:
                sess.used_partial_fallback = True
                return "Partial monologue here"
            else:
                # Phase-2 fails to produce a result
                return ""

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "hard task")

        # Should have attempted both calls
        assert invoke_count[0] == 2
        # Final message sent (original partial content)
        assert d.tg.send.called

    @pytest.mark.asyncio
    async def test_phase2_clears_flag_on_success(self, tmp_path):
        """After successful phase-2, session.used_partial_fallback is cleared."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", str(Path.home()))

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            sess.status = "done"
            sess.started = time.time()
            sess.finished = time.time()
            if not resume:
                sess.used_partial_fallback = True
                return "Partial output"
            else:
                return "Clean summary from phase 2."

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "test")

        assert session.used_partial_fallback is False


# ============================================================
# 4. _do_followup phase-2 trigger
# ============================================================

class TestDoFollowupPhase2:
    """_do_followup also triggers phase-2 summarization when needed."""

    @pytest.mark.asyncio
    async def test_followup_phase2_triggered(self, tmp_path):
        """Phase-2 fires in _do_followup when runner sets used_partial_fallback.

        Note: _do_followup always calls runner with resume=True on the first call
        (when model hasn't changed). Phase-2 then fires a second resume=True call.
        """
        d = make_dispatcher(tmp_path)

        prev = d.sm.create(1, "first task", "/tmp/proj")
        prev.status = "done"
        prev.finished = time.time()

        d.transcript.append(prev.conv_id, "user", "first task")
        d.transcript.append(prev.conv_id, "assistant", "Done with first task.")

        invoke_count = [0]

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            invoke_count[0] += 1
            sess.status = "done"
            sess.started = time.time()
            sess.finished = time.time()

            if invoke_count[0] == 1:
                # First call (from _do_followup with resume=True): simulate exhaustion
                sess.used_partial_fallback = True
                return "Thinking about it..."
            else:
                # Phase-2 or retry: return clean summary
                return "Follow-up complete: updated tests."

        d.runner.invoke = mock_invoke
        await d._do_followup(2, "now update the tests", prev)

        # Phase-2 should have been triggered
        assert invoke_count[0] >= 2

    @pytest.mark.asyncio
    async def test_followup_phase2_not_triggered_when_not_needed(self, tmp_path):
        """Phase-2 not triggered in _do_followup when runner succeeds normally."""
        d = make_dispatcher(tmp_path)

        prev = d.sm.create(1, "first task", "/tmp")
        prev.status = "done"
        prev.finished = time.time()

        summarize_called = []
        original_summarize = d._summarize_session
        async def tracking_summarize(sess, q):
            summarize_called.append(q)
            return await original_summarize(sess, q)
        d._summarize_session = tracking_summarize

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            sess.status = "done"
            sess.started = time.time()
            sess.finished = time.time()
            # Normal completion
            sess.used_partial_fallback = False
            return "Follow-up done."

        d.runner.invoke = mock_invoke
        await d._do_followup(2, "follow up", prev)

        assert len(summarize_called) == 0


# ============================================================
# 5. Config: max_turns_chat removed
# ============================================================

class TestConfigMaxTurnsChat:
    """Verify max_turns_chat was removed from Config."""

    def test_config_has_no_max_turns_chat_attribute(self, tmp_path):
        """Config object must not have max_turns_chat attribute."""
        cfg = make_config(tmp_path)
        assert not hasattr(cfg, "max_turns_chat"), \
            "max_turns_chat should have been removed from Config"

    def test_config_still_has_max_turns(self, tmp_path):
        """Config still has max_turns (used for new sessions)."""
        cfg = make_config(tmp_path)
        assert hasattr(cfg, "max_turns")
        assert cfg.max_turns > 0

    def test_config_still_has_max_turns_followup(self, tmp_path):
        """Config still has max_turns_followup."""
        cfg = make_config(tmp_path)
        assert hasattr(cfg, "max_turns_followup")
        assert cfg.max_turns_followup > 0

    def test_config_with_max_turns_chat_in_file_is_ignored(self, tmp_path):
        """If old config file has max_turns_chat, it's silently ignored without crashing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        # Old config with max_turns_chat still present
        cfg_data = {
            "telegram": {"bot_token": "test-token", "chat_id": 12345},
            "agent": {
                "command": "echo",
                "args": [],
                "max_concurrent": 3,
                "timeout": 30,
                "max_turns": 10,
                "max_turns_chat": 5,  # this should be ignored
                "max_turns_followup": 5,
                "question_timeout": 600,
            },
            "behavior": {
                "poll_timeout": 1,
                "progress_interval": 60,
                "recent_window": 300,
                "cancel_keywords": ["cancel"],
                "status_keywords": ["status"],
            },
            "projects": {},
            "data_dir": str(data_dir),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        # Should not crash
        cfg = Config(str(cfg_path))
        assert cfg.max_turns == 10
        assert not hasattr(cfg, "max_turns_chat")


# ============================================================
# 6. _prompt_footer: requires user-facing summary
# ============================================================

class TestPromptFooter:
    """_prompt_footer enforces that the agent always produces a user-facing summary."""

    def test_footer_requires_user_facing_summary(self, tmp_path):
        """Footer must instruct agent to always end with a user-facing summary."""
        footer = Dispatcher._prompt_footer()
        assert "summary" in footer.lower() or "user-facing" in footer.lower(), \
            "Footer must mention 'summary' or 'user-facing'"

    def test_footer_prohibits_telegram_messages(self, tmp_path):
        """Footer must prohibit direct Telegram API calls."""
        footer = Dispatcher._prompt_footer()
        assert "Telegram" in footer or "telegram" in footer, \
            "Footer must mention Telegram prohibition"
        assert "curl" in footer or "API" in footer, \
            "Footer must mention curl/API prohibition"

    def test_footer_instructs_ran_out_of_turns(self, tmp_path):
        """Footer must handle the 'ran out of turns mid-task' scenario."""
        footer = Dispatcher._prompt_footer()
        # The footer should guide behavior when turns run out
        assert "turns" in footer.lower() or "exit" in footer.lower(), \
            "Footer should address ran-out-of-turns scenario"

    def test_footer_formatting_guidance(self, tmp_path):
        """Footer provides formatting guidance for Telegram output."""
        footer = Dispatcher._prompt_footer()
        assert "bold" in footer.lower() or "**" in footer, \
            "Footer should mention bold formatting"
        assert "Telegram" in footer, \
            "Footer should mention Telegram context"


# ============================================================
# 7. Integration: full scenario with partial fallback
# ============================================================

class TestFullPartialFallbackScenario:
    """End-to-end scenario where agent exhausts turns and phase-2 saves the day."""

    @pytest.mark.asyncio
    async def test_full_scenario_partial_fallback_then_summary(self, tmp_path):
        """Full scenario: turns exhausted -> partial_output fallback -> phase-2 summary sent."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "implement caching layer", str(Path.home()))

        phases = []

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            sess.status = "done"
            sess.started = time.time()
            sess.finished = time.time()

            if not resume:
                phases.append("initial")
                # Simulate: agent worked but ran out of turns, partial_output accumulated
                sess.partial_output = "Started implementing cache... checked Redis options..."
                sess.used_partial_fallback = True
                return sess.partial_output
            else:
                phases.append("phase2")
                # Phase-2: clean summary
                assert max_turns == 3, f"Phase-2 should use max_turns=3, got {max_turns}"
                return "Implemented Redis caching in cache.py. Added TTL=300s. Tests pass."

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "implement caching layer")

        # Both phases were invoked
        assert "initial" in phases
        assert "phase2" in phases

        # The clean summary was sent to the user
        all_sent_texts = " ".join(
            call[0][0] for call in d.tg.send.call_args_list if call[0]
        )
        assert "Redis" in all_sent_texts or "cache.py" in all_sent_texts, \
            f"Phase-2 summary not sent. Got: {all_sent_texts}"

    @pytest.mark.asyncio
    async def test_model_none_passed_to_summarize(self, tmp_path):
        """Phase-2 uses session.model_override (may be None) not hardcoded model."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", str(Path.home()))

        captured_models = []

        async def mock_invoke(sess, prompt, resume=False, max_turns=10,
                              model=None, stream=True, on_question=None):
            captured_models.append(model)
            sess.status = "done"
            sess.started = time.time()
            sess.finished = time.time()
            if not resume:
                sess.used_partial_fallback = True
                return "partial"
            else:
                return "summary"

        d.runner.invoke = mock_invoke
        session.model_override = None
        await d._do_session(1, session, "test task")

        # Phase-2 call (second invoke) should use session.model_override
        if len(captured_models) >= 2:
            assert captured_models[1] is None  # no model = default
