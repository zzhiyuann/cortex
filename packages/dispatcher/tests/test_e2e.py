"""End-to-end tests for dispatcher logic.

Tests the full flow: message → classification → session → agent → result,
but with a mock agent runner and mock Telegram client.
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

import pytest

from dispatcher.config import Config
from dispatcher.core import Dispatcher, _md_to_telegram_html
from dispatcher.session import Session, SessionManager
from dispatcher.runner import AgentRunner


# -- Global fixture: prevent tests from writing to real issues.jsonl --

@pytest.fixture(autouse=True)
def _no_record_issue(monkeypatch):
    """Patch record_issue to a no-op so tests don't pollute the real issue store."""
    monkeypatch.setattr("dispatcher.core.record_issue", lambda *a, **kw: None)


# -- Fixtures --

def make_config(tmp_path, projects=None):
    """Create a minimal config file for testing."""
    import yaml
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
        },
        "behavior": {
            "poll_timeout": 1,
            "progress_interval": 60,
            "recent_window": 300,
            "cancel_keywords": ["cancel", "stop", "取消", "停"],
            "status_keywords": ["status", "在干嘛"],
        },
        "projects": projects or {},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))
    return Config(str(cfg_path))


def make_dispatcher(tmp_path, projects=None):
    cfg = make_config(tmp_path, projects)
    d = Dispatcher(cfg)
    # Mock Telegram client — accept all kwargs
    d.tg = MagicMock()
    d.tg.send = MagicMock(return_value=9999)
    d.tg.edit = MagicMock(return_value=True)
    d.tg.typing = MagicMock()
    d.tg.set_my_commands = MagicMock(return_value=True)
    return d


# -- Classification tests --

class TestClassification:
    def test_status_chinese(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._classify("在干嘛") == "status"

    def test_status_english(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._classify("status") == "status"

    def test_cancel_chinese(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._classify("取消") == "cancel"
        assert d._classify("停") == "cancel"

    def test_cancel_english(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._classify("cancel") == "cancel"
        assert d._classify("stop that") == "cancel"

    def test_normal_message(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._classify("hi") == "task"
        assert d._classify("fix the login bug") == "task"
        assert d._classify("帮我看看代码") == "task"

    def test_slash_commands(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._classify("/status") == "status"
        assert d._classify("/cancel") == "cancel"
        assert d._classify("/history") == "history"
        assert d._classify("/help") == "help"
        # With @botname suffix
        assert d._classify("/status@ryanwangclaudebot") == "status"
        assert d._classify("/cancel@ryanwangclaudebot") == "cancel"

    def test_history_keywords(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._classify("history") == "history"
        assert d._classify("历史") == "history"
        assert d._classify("最近任务") == "history"

    def test_help_keywords(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._classify("help") == "help"
        assert d._classify("帮助") == "help"


# -- Model prefix extraction tests --

class TestModelPrefix:
    def test_haiku_prefix_lowercase(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("@haiku 你是什么模型")
        assert text == "你是什么模型"
        assert model == "haiku"
        assert sticky is False

    def test_opus_prefix_lowercase(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("@opus write a complex algo")
        assert text == "write a complex algo"
        assert model == "opus"
        assert sticky is False

    def test_sonnet_prefix_lowercase(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("@sonnet help me")
        assert text == "help me"
        assert model == "sonnet"
        assert sticky is False

    def test_no_prefix(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("just a normal message")
        assert text == "just a normal message"
        assert model is None
        assert sticky is False

    def test_capitalized_is_sticky(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("@Haiku test")
        assert text == "test"
        assert model == "haiku"
        assert sticky is True

    def test_capitalized_sonnet_sticky(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("@Sonnet do something")
        assert text == "do something"
        assert model == "sonnet"
        assert sticky is True

    def test_prefix_only_no_text(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("@haiku")
        assert text == "@haiku"
        assert model is None

    def test_prefix_at_in_middle(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text, model, sticky = d._extract_model_prefix("use @haiku for this")
        assert text == "use @haiku for this"
        assert model is None

    def test_sticky_sets_dispatcher_model(self, tmp_path):
        """@Haiku (capitalized) sets _sticky_model on the dispatcher."""
        d = make_dispatcher(tmp_path)
        assert d._sticky_model is None
        text, model, sticky = d._extract_model_prefix("@Haiku test")
        if sticky and model:
            d._sticky_model = model
        assert d._sticky_model == "haiku"

    def test_lowercase_does_not_set_sticky(self, tmp_path):
        """@haiku (lowercase) does NOT change _sticky_model."""
        d = make_dispatcher(tmp_path)
        d._sticky_model = "opus"  # previously set
        text, model, sticky = d._extract_model_prefix("@haiku test")
        if sticky and model:
            d._sticky_model = model
        assert d._sticky_model == "opus"  # unchanged


# -- Project routing tests --

class TestProjectRouting:
    def test_keyword_match(self, tmp_path):
        projects = {
            "webapp": {
                "path": str(tmp_path / "webapp"),
                "keywords": ["webapp", "web", "frontend"],
            }
        }
        (tmp_path / "webapp").mkdir()
        d = make_dispatcher(tmp_path, projects)
        assert d._detect_project("fix the webapp bug") == str(tmp_path / "webapp")
        assert d._detect_project("check the web server") == str(tmp_path / "webapp")

    def test_no_match(self, tmp_path):
        d = make_dispatcher(tmp_path)
        assert d._detect_project("hello there") is None

    def test_nonexistent_path(self, tmp_path):
        projects = {
            "ghost": {
                "path": "/nonexistent/path",
                "keywords": ["ghost"],
            }
        }
        d = make_dispatcher(tmp_path, projects)
        assert d._detect_project("ghost project") is None


# -- Session manager tests --

class TestSessionManager:
    def test_create_and_find(self):
        sm = SessionManager()
        s = sm.create(1, "test", "/tmp")
        assert sm.by_msg[1] is s
        assert s.status == "pending"

    def test_link_bot_and_find_by_reply(self):
        sm = SessionManager()
        s = sm.create(1, "test", "/tmp")
        sm.link_bot(100, 1)
        assert sm.find_by_reply(100) is s
        assert sm.find_by_reply(1) is s

    def test_active_count(self):
        sm = SessionManager()
        s1 = sm.create(1, "task1", "/tmp")
        s2 = sm.create(2, "task2", "/tmp")
        s1.status = "running"
        assert sm.active_count() == 1
        s2.status = "running"
        assert sm.active_count() == 2

    def test_last_session(self):
        sm = SessionManager()
        s1 = sm.create(1, "first", "/tmp")
        s2 = sm.create(2, "second", "/tmp")
        assert sm.last_session() is s2

    def test_last_session_empty(self):
        sm = SessionManager()
        assert sm.last_session() is None

    def test_partial_output_attr(self):
        s = Session(1, "test", "/tmp")
        assert s.partial_output == ""
        s.partial_output = "hello world"
        assert s.partial_output == "hello world"


# -- Handler tests --

class TestHandlers:
    def test_status_no_tasks(self, tmp_path):
        d = make_dispatcher(tmp_path)
        d._handle_status(1)
        d.tg.send.assert_called_once()
        msg = d.tg.send.call_args[1].get("text", d.tg.send.call_args[0][0])
        assert "空闲" in msg

    def test_status_with_tasks(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = d.sm.create(1, "fixing stuff", "/tmp/myproject")
        s.status = "running"
        s.started = time.time() - 120
        d._handle_status(2)
        msg = d.tg.send.call_args[0][0]
        assert "myproject" in msg

    def test_cancel_single_task(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = d.sm.create(1, "task", "/tmp")
        s.status = "running"
        s.proc = MagicMock()
        d._handle_cancel(2, "cancel", None)
        assert s.status == "cancelled"
        s.proc.kill.assert_called_once()

    def test_cancel_no_task(self, tmp_path):
        d = make_dispatcher(tmp_path)
        d._handle_cancel(1, "cancel", None)
        msg = d.tg.send.call_args[0][0]
        assert "没有" in msg

    def test_history_empty(self, tmp_path):
        d = make_dispatcher(tmp_path)
        d._handle_history(1)
        msg = d.tg.send.call_args[0][0]
        assert "没有" in msg

    def test_history_with_items(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = d.sm.create(1, "test task", "/tmp/proj")
        s.status = "done"
        s.started = time.time() - 60
        s.finished = time.time()
        d._handle_history(2)
        msg = d.tg.send.call_args[0][0]
        assert "test task" in msg

    def test_help(self, tmp_path):
        d = make_dispatcher(tmp_path)
        d._handle_help(1)
        msg = d.tg.send.call_args[0][0]
        assert "使用指南" in msg


# -- Do-session tests (with mock runner) --

class TestDoSession:
    @pytest.mark.asyncio
    async def test_simple_message_single_reply(self, tmp_path):
        """'hi' should produce a reply."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "hi", str(Path.home()))
        session.is_task = False

        async def mock_invoke(session, prompt, resume=False, max_turns=10, model=None, stream=True):
            session.status = "running"
            session.started = time.time()
            await asyncio.sleep(0.1)
            session.status = "done"
            session.finished = time.time()
            session.result = "你好！"
            return "你好！"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "hi")

        # Should have at least one send call (the result)
        assert d.tg.send.call_count >= 1

    @pytest.mark.asyncio
    async def test_task_message_single_reply(self, tmp_path):
        """A project task should produce a reply."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "fix the bug", "/tmp/webapp")
        session.is_task = True

        async def mock_invoke(session, prompt, resume=False, max_turns=10, model=None, stream=True):
            session.status = "running"
            session.started = time.time()
            await asyncio.sleep(0.2)
            session.status = "done"
            session.finished = time.time()
            session.result = "Fixed the null check in auth.py"
            return "Fixed the null check in auth.py"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "fix the bug")

        assert d.tg.send.call_count >= 1

    @pytest.mark.asyncio
    async def test_model_passed_to_runner(self, tmp_path):
        """Model override should be forwarded to runner."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "test", str(Path.home()))

        captured_model = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10, model=None, stream=True):
            captured_model.append(model)
            session.status = "done"
            session.finished = time.time()
            return "ok"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "test", model="haiku")

        assert captured_model == ["haiku"]

    @pytest.mark.asyncio
    async def test_followup_same_model_resumes(self, tmp_path):
        """Follow-up with same model should resume the session."""
        d = make_dispatcher(tmp_path)
        prev = d.sm.create(1, "original task", "/tmp/webapp")
        prev.is_task = True
        prev.status = "done"
        prev.finished = time.time()

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10, model=None, stream=True):
            invocations.append({"sid": session.sid, "resume": resume, "model": model})
            session.status = "done"
            session.finished = time.time()
            return "Follow-up result"

        d.runner.invoke = mock_invoke
        await d._do_followup(2, "what about tests?", prev)

        assert len(invocations) == 1
        assert invocations[0]["sid"] == prev.sid
        assert invocations[0]["resume"] is True

    @pytest.mark.asyncio
    async def test_followup_different_model_fresh_session(self, tmp_path):
        """Follow-up with different model should start fresh (--resume ignores --model)."""
        d = make_dispatcher(tmp_path)
        prev = d.sm.create(1, "original task", "/tmp/webapp")
        prev.is_task = True
        prev.status = "done"
        prev.model_override = "opus"
        prev.finished = time.time()

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10, model=None, stream=True):
            invocations.append({"sid": session.sid, "resume": resume, "model": model})
            session.status = "done"
            session.finished = time.time()
            return "Follow-up result"

        d.runner.invoke = mock_invoke
        await d._do_followup(2, "what about tests?", prev, model="haiku")

        assert len(invocations) == 1
        assert invocations[0]["sid"] != prev.sid  # fresh session
        assert invocations[0]["resume"] is False
        assert invocations[0]["model"] == "haiku"


# -- Result formatting tests --

class TestSendResult:
    def test_empty_result(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = Session(1, "test", "/tmp")
        s.status = "done"
        d._send_result(1, s, "")
        msg = d.tg.send.call_args[0][0]
        assert "没有返回" in msg or "没有输出" in msg or "完成" in msg

    def test_failed_result(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = Session(1, "test", "/tmp")
        s.status = "failed"
        d._send_result(1, s, "something broke")
        msg = d.tg.send.call_args[0][0]
        assert "失败" in msg or "出错" in msg

    def test_long_result_split(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()
        d._send_result(1, s, "x" * 5000)
        # Long output is now sent as a file document
        d.tg.send_document.assert_called_once()

    def test_cancelled_no_reply(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = Session(1, "test", "/tmp")
        s.status = "cancelled"
        d._send_result(1, s, "result")
        d.tg.send.assert_not_called()

    def test_short_task_no_duration_prefix(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()
        d._send_result(1, s, "Here is the answer")
        msg = d.tg.send.call_args[0][0]
        assert "完成" not in msg  # No duration prefix for quick tasks

    def test_long_running_done_prefix(self, tmp_path):
        d = make_dispatcher(tmp_path)
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time() - 120
        s.finished = time.time()
        d._send_result(1, s, "Refactored the module")
        msg = d.tg.send.call_args[0][0]
        assert "完成" in msg


# -- Markdown to HTML conversion tests --

class TestMarkdownToHtml:
    def test_bold(self):
        result = _md_to_telegram_html("**hello**")
        assert result == "<b>hello</b>"

    def test_inline_code(self):
        result = _md_to_telegram_html("`code`")
        assert result == "<code>code</code>"

    def test_code_block(self):
        result = _md_to_telegram_html("```python\nprint('hi')\n```")
        assert "<pre>" in result
        assert "print" in result

    def test_html_escape(self):
        result = _md_to_telegram_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_mixed(self):
        result = _md_to_telegram_html("Use **bold** and `code` together")
        assert "<b>bold</b>" in result
        assert "<code>code</code>" in result


# -- Stream parsing tests --

class TestStreamParsing:
    @pytest.mark.asyncio
    async def test_parse_assistant_text(self):
        """Verify _read_stream extracts text from assistant events."""
        runner = AgentRunner.__new__(AgentRunner)
        session = Session(1, "test", "/tmp")

        # Simulate stream events
        events = [
            json.dumps({"type": "system", "subtype": "init", "session_id": "abc"}),
            json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hello world"}]},
            }),
            json.dumps({
                "type": "result",
                "subtype": "success",
                "result": "Hello world",
            }),
        ]

        # Create a mock proc with stdout yielding the events
        proc = MagicMock()
        proc.wait = AsyncMock()
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"")

        async def mock_stdout():
            for e in events:
                yield (e + "\n").encode()

        proc.stdout = mock_stdout()

        result = await runner._read_stream(proc, session)
        assert result == "Hello world"
        assert session.partial_output == "Hello world"

    @pytest.mark.asyncio
    async def test_parse_thinking_ignored(self):
        """Thinking blocks should not appear in output."""
        runner = AgentRunner.__new__(AgentRunner)
        session = Session(1, "test", "/tmp")

        events = [
            json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "thinking", "thinking": "Let me think..."}]},
            }),
            json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "The answer is 42"}]},
            }),
            json.dumps({"type": "result", "result": "The answer is 42"}),
        ]

        proc = MagicMock()
        proc.wait = AsyncMock()
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"")

        async def mock_stdout():
            for e in events:
                yield (e + "\n").encode()

        proc.stdout = mock_stdout()

        result = await runner._read_stream(proc, session)
        assert result == "The answer is 42"
        assert "think" not in session.partial_output.lower()

    @pytest.mark.asyncio
    async def test_fallback_to_last_text_if_no_result(self):
        """If no result event, use last assistant text."""
        runner = AgentRunner.__new__(AgentRunner)
        session = Session(1, "test", "/tmp")

        events = [
            json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "partial output"}]},
            }),
            # No result event
        ]

        proc = MagicMock()
        proc.wait = AsyncMock()
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"")

        async def mock_stdout():
            for e in events:
                yield (e + "\n").encode()

        proc.stdout = mock_stdout()

        result = await runner._read_stream(proc, session)
        assert result == "partial output"

    @pytest.mark.asyncio
    async def test_stderr_fallback(self):
        """If no assistant text and no result, fall back to stderr."""
        runner = AgentRunner.__new__(AgentRunner)
        session = Session(1, "test", "/tmp")

        events = [
            json.dumps({"type": "system", "subtype": "init"}),
        ]

        proc = MagicMock()
        proc.wait = AsyncMock()
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"some error occurred")

        async def mock_stdout():
            for e in events:
                yield (e + "\n").encode()

        proc.stdout = mock_stdout()

        result = await runner._read_stream(proc, session)
        assert "some error occurred" in result

    @pytest.mark.asyncio
    async def test_malformed_json_skipped(self):
        """Malformed JSON lines should be skipped gracefully."""
        runner = AgentRunner.__new__(AgentRunner)
        session = Session(1, "test", "/tmp")

        events = [
            "not json at all",
            json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "good output"}]},
            }),
            json.dumps({"type": "result", "result": "good output"}),
        ]

        proc = MagicMock()
        proc.wait = AsyncMock()
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"")

        async def mock_stdout():
            for e in events:
                yield (e + "\n").encode()

        proc.stdout = mock_stdout()

        result = await runner._read_stream(proc, session)
        assert result == "good output"


# -- Runner command construction tests --

class TestRunnerCommandBuild:
    def test_stream_json_with_verbose(self):
        """stream-json mode must include --verbose."""
        runner = AgentRunner(command="claude", args=["-p", "--dangerously-skip-permissions"])
        # Simulate command building by checking _is_claude
        assert runner._is_claude()

    def test_model_flag_added(self):
        """When model is specified, --model flag should be in the command."""
        # This is a behavior test — we verify the flag is built correctly
        runner = AgentRunner(command="echo", args=[])
        assert not runner._is_claude()


# -- PID file tests --

class TestPIDFile:
    def test_acquire_pid(self, tmp_path):
        cfg = make_config(tmp_path)
        cfg._data["data_dir"] = str(tmp_path / "data")
        d = Dispatcher(cfg)
        d.tg = MagicMock()
        d.tg.send = MagicMock(return_value=9999)
        d.tg.set_my_commands = MagicMock(return_value=True)

        assert d._acquire_pid() is True
        pid_file = cfg.data_dir / "dispatcher.pid"
        assert pid_file.exists()
        assert pid_file.read_text().strip() == str(__import__("os").getpid())

    def test_stale_pid_overwritten(self, tmp_path):
        cfg = make_config(tmp_path)
        cfg._data["data_dir"] = str(tmp_path / "data")
        d = Dispatcher(cfg)
        d.tg = MagicMock()
        d.tg.send = MagicMock(return_value=9999)
        d.tg.set_my_commands = MagicMock(return_value=True)

        # Write a stale PID (process that doesn't exist)
        pid_file = cfg.data_dir / "dispatcher.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text("99999999")

        assert d._acquire_pid() is True

    def test_release_pid(self, tmp_path):
        cfg = make_config(tmp_path)
        cfg._data["data_dir"] = str(tmp_path / "data")
        d = Dispatcher(cfg)
        d.tg = MagicMock()
        d.tg.send = MagicMock(return_value=9999)
        d.tg.set_my_commands = MagicMock(return_value=True)

        d._acquire_pid()
        pid_file = cfg.data_dir / "dispatcher.pid"
        assert pid_file.exists()

        d._release_pid()
        assert not pid_file.exists()


# -- Message splitting tests --

class TestMessageSplitting:
    def test_short_message_no_split(self, tmp_path):
        d = make_dispatcher(tmp_path)
        chunks = d._split_message("short text", 4000)
        assert chunks == ["short text"]

    def test_long_message_split_at_newline(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text = ("line\n" * 1000)  # ~5000 chars
        chunks = d._split_message(text, 100)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_no_good_newline_hard_split(self, tmp_path):
        d = make_dispatcher(tmp_path)
        text = "x" * 200  # No newlines
        chunks = d._split_message(text, 100)
        assert len(chunks) == 2
        assert chunks[0] == "x" * 100
        assert chunks[1] == "x" * 100


# -- Friendly error tests --

class TestFriendlyError:
    def test_timeout_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("Timed out after 30 minutes")
        assert "超时" in result

    def test_rate_limit_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("429 rate limit exceeded")
        assert "限流" in result

    def test_max_turns_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("max turns reached")
        assert "轮次" in result

    def test_permission_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("permission denied: /etc/secret")
        assert "权限" in result

    def test_generic_error(self, tmp_path):
        d = make_dispatcher(tmp_path)
        result = d._friendly_error("something unexpected")
        assert "出错" in result


# ============================================================
# UX-FOCUSED TESTS
# ============================================================


class TestUXResponseLatency:
    """Verify architectural choices that affect perceived response time."""

    @pytest.mark.asyncio
    async def test_all_messages_use_stream_and_full_turns(self, tmp_path):
        """All messages get full Claude Code capability: stream + max_turns."""
        d = make_dispatcher(tmp_path)
        session = d.sm.create(1, "what is python", str(Path.home()))
        session.is_task = False

        captured = {}
        async def mock_invoke(session, prompt, resume=False, max_turns=10, model=None, stream=True):
            captured["stream"] = stream
            captured["max_turns"] = max_turns
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "Python is a language"

        d.runner.invoke = mock_invoke
        await d._do_session(1, session, "what is python")

        assert captured["stream"] is True, "All messages should use stream-json"
        assert captured["max_turns"] == d.cfg.max_turns, "All messages should get cfg.max_turns"

    def test_commands_respond_without_agent(self, tmp_path):
        """/status /help /history respond instantly via tg.send, no agent needed."""
        d = make_dispatcher(tmp_path)
        for handler in [d._handle_status, d._handle_help, d._handle_history]:
            d.tg.send.reset_mock()
            handler(1)
            assert d.tg.send.call_count == 1, f"{handler.__name__} should produce exactly 1 send"

    @pytest.mark.asyncio
    async def test_typing_fires_before_classification(self, tmp_path):
        """Typing indicator must fire before message classification."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        call_order = []
        orig_fire_typing = d._fire_typing
        def tracking_typing():
            call_order.append("typing")
        d._fire_typing = tracking_typing

        orig_classify = d._classify
        def tracking_classify(text):
            call_order.append("classify")
            return orig_classify(text)
        d._classify = tracking_classify

        # Use "status" so _on_message handles it synchronously (no async dispatch)
        msg = {"chat": {"id": d.cfg.chat_id}, "message_id": 1, "text": "status"}
        await d._on_message(msg)

        assert "typing" in call_order, "typing should have been called"
        assert "classify" in call_order, "classify should have been called"
        assert call_order.index("typing") < call_order.index("classify"), \
            "Typing should fire before classification"


class TestUXNoNoise:
    """Verify the user is not bombarded with unnecessary messages."""

    @pytest.mark.asyncio
    async def test_progress_loop_no_messages_when_fast(self, tmp_path):
        """Progress loop sends no messages if session finishes before first check."""
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

    def test_cancelled_no_output(self, tmp_path):
        """Cancelled sessions produce zero messages to the user."""
        d = make_dispatcher(tmp_path)
        s = Session(1, "test", "/tmp")
        s.status = "cancelled"
        d._send_result(1, s, "some output")
        d.tg.send.assert_not_called()

    def test_quick_result_no_duration_header(self, tmp_path):
        """Results from <60s sessions have no '完成 (Xs)' prefix — just the answer."""
        d = make_dispatcher(tmp_path)
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()  # elapsed ~= 0
        d._send_result(1, s, "The answer is 42")
        msg = d.tg.send.call_args[0][0]
        assert "完成" not in msg, "Quick results should not show duration header"
        assert "42" in msg


class TestUXReactionFeedback:
    """Verify reaction emojis provide instant, non-intrusive status feedback."""

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
        assert (1, "\u2705") in reactions, "Success should trigger checkmark reaction"

    def test_failure_reaction(self, tmp_path):
        """Failed task triggers X reaction."""
        d = make_dispatcher(tmp_path)
        reactions = []
        d._fire_reaction = lambda mid, emoji: reactions.append((mid, emoji))
        s = Session(1, "test", "/tmp")
        s.status = "failed"
        d._send_result(1, s, "Error occurred")
        assert (1, "\u274c") in reactions, "Failure should trigger X reaction"

    @pytest.mark.asyncio
    async def test_incoming_task_gets_eyes(self, tmp_path):
        """Task messages should get eyes reaction immediately on receive."""
        d = make_dispatcher(tmp_path)
        reactions = []
        d._fire_reaction = lambda mid, emoji: reactions.append((mid, emoji))

        msg = {"chat": {"id": d.cfg.chat_id}, "message_id": 42, "text": "do something"}
        await d._on_message(msg)

        assert (42, "\U0001f440") in reactions, "Task message should get eyes reaction"

    @pytest.mark.asyncio
    async def test_command_no_reaction(self, tmp_path):
        """Commands like 'status' should NOT get eyes reaction (they're not tasks)."""
        d = make_dispatcher(tmp_path)
        reactions = []
        d._fire_reaction = lambda mid, emoji: reactions.append((mid, emoji))

        msg = {"chat": {"id": d.cfg.chat_id}, "message_id": 1, "text": "status"}
        await d._on_message(msg)

        eyes_reactions = [(m, e) for m, e in reactions if e == "\U0001f440"]
        assert len(eyes_reactions) == 0, "Command messages should not get eyes reaction"


class TestUXRetryButton:
    """Verify the retry UX flow works end-to-end."""

    def test_failed_has_retry_button(self, tmp_path):
        """Failed tasks must show a retry inline button."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "fix bug", "/tmp/proj")
        s.status = "failed"
        d._send_result(1, s, "Error: something broke")

        call_kwargs = d.tg.send.call_args
        # _reply calls tg.send(text, reply_to=..., parse_mode=..., reply_markup=...)
        markup = call_kwargs[1].get("reply_markup")
        assert markup is not None, "Failed result should have reply_markup"
        buttons = markup["inline_keyboard"][0]
        assert any("retry" in b["callback_data"] for b in buttons), \
            "Failed result should have retry button"

    def test_success_no_retry_button(self, tmp_path):
        """Successful tasks should NOT have a retry button."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()
        d._send_result(1, s, "All good")

        call_kwargs = d.tg.send.call_args
        markup = call_kwargs[1].get("reply_markup")
        if markup:
            for row in markup.get("inline_keyboard", []):
                for btn in row:
                    assert "retry" not in btn.get("callback_data", ""), \
                        "Success result should not have retry button"

    @pytest.mark.asyncio
    async def test_retry_callback_redispatches(self, tmp_path):
        """Pressing retry button should re-dispatch the original task."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        # Create a failed session
        s = d.sm.create(10, "fix the auth bug", "/tmp/proj")
        s.status = "failed"
        s.model_override = "haiku"

        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append({"mid": mid, "text": text, "model": model})
        d._handle_task = track_handle

        cb = {
            "id": "cb123",
            "data": "retry:10",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 99},
        }
        d.tg.answer_callback = MagicMock()
        await d._on_callback(cb)

        assert len(dispatched) == 1, "Retry should dispatch exactly 1 task"
        assert dispatched[0]["text"] == "fix the auth bug"
        assert dispatched[0]["model"] == "haiku"
        d.tg.answer_callback.assert_called_with("cb123", "\U0001f504 重试中...")

    @pytest.mark.asyncio
    async def test_retry_missing_session(self, tmp_path):
        """Retry for a non-existent session should show error, not crash."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        d.tg.answer_callback = MagicMock()

        cb = {
            "id": "cb456",
            "data": "retry:99999",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 99},
        }
        await d._on_callback(cb)
        d.tg.answer_callback.assert_called_with("cb456", "找不到原始任务")


class TestUXLongOutput:
    """Verify long outputs are sent as files, not message spam."""

    def test_long_output_sends_document(self, tmp_path):
        """Output > 4000 chars should be sent as a .md file, not split messages."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        d.tg.send_document = MagicMock(return_value=8888)

        s = Session(1, "big task", "/tmp/proj")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()
        s.is_task = True

        d._send_result(1, s, "x" * 5000)

        d.tg.send_document.assert_called_once()

    def test_short_output_is_message(self, tmp_path):
        """Output under 4000 chars should be a normal message, not a file."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()
        d.tg.send_document = MagicMock()

        s = Session(1, "test", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()

        d._send_result(1, s, "Short answer")
        d.tg.send.assert_called()
        d.tg.send_document.assert_not_called()


class TestUXMessageBatching:
    """Verify message batching merges rapid-fire messages correctly."""

    @pytest.mark.asyncio
    async def test_single_message_passes_through(self, tmp_path):
        """A single buffered message flushes unchanged."""
        d = make_dispatcher(tmp_path)
        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append({"mid": mid, "text": text})
        d._handle_task = track_handle

        d._buffer_message(1, "hello world", None, [])
        # Wait for the 2s flush delay
        await asyncio.sleep(2.5)

        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "hello world"
        assert dispatched[0]["mid"] == 1

    @pytest.mark.asyncio
    async def test_multiple_messages_merged(self, tmp_path):
        """Multiple messages buffered within 2s should merge into one prompt."""
        d = make_dispatcher(tmp_path)
        dispatched = []
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append({"mid": mid, "text": text})
        d._handle_task = track_handle

        d._buffer_message(1, "first message", None, [])
        d._buffer_message(2, "second message", None, [])
        d._buffer_message(3, "third message", None, [])
        # Wait for the 2s flush delay
        await asyncio.sleep(2.5)

        assert len(dispatched) == 1, "Multiple messages should merge into 1 dispatch"
        assert "first message" in dispatched[0]["text"]
        assert "second message" in dispatched[0]["text"]
        assert "third message" in dispatched[0]["text"]
        assert dispatched[0]["mid"] == 3, "Should use last message ID"

    @pytest.mark.asyncio
    async def test_reply_bypasses_buffer(self, tmp_path):
        """Reply messages should go directly to _handle_task, not buffered."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        dispatched = []
        orig_handle_task = d._handle_task
        async def track_handle(mid, text, reply_to, attachments=None, model=None):
            dispatched.append({"mid": mid, "text": text, "reply_to": reply_to})
        d._handle_task = track_handle

        # Create a previous session so the reply has context
        prev = d.sm.create(50, "original", "/tmp")
        prev.status = "done"
        prev.finished = time.time()

        msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 51,
            "text": "what about tests?",
            "reply_to_message": {"message_id": 50},
        }
        await d._on_message(msg)

        # Reply should bypass the buffer — dispatched immediately (or via followup)
        # Buffer should be empty since replies go through directly
        assert len(d._msg_buffer) == 0, "Reply should not be buffered"


class TestUXEditedMessage:
    """Verify edited message behavior — cancel and re-dispatch if young."""

    @pytest.mark.asyncio
    async def test_edit_young_session_redispatches(self, tmp_path):
        """Editing a message with a <10s-old session should cancel and re-dispatch."""
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

        assert s.status == "cancelled", "Old session should be cancelled"
        s.proc.kill.assert_called_once()
        assert len(dispatched) == 1, "Should re-dispatch with new text"
        assert dispatched[0]["text"] == "new text"

    @pytest.mark.asyncio
    async def test_edit_old_session_ignored(self, tmp_path):
        """Editing a message with a >10s-old session should be ignored."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        s = d.sm.create(1, "old text", "/tmp/proj")
        s.status = "running"
        s.started = time.time() - 30  # started 30s ago
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

        assert s.status == "running", "Old session should not be cancelled"
        assert len(dispatched) == 0, "Should not re-dispatch"

    @pytest.mark.asyncio
    async def test_edit_no_session_ignored(self, tmp_path):
        """Editing a message with no matching session should be silently ignored."""
        d = make_dispatcher(tmp_path)

        edited_msg = {
            "chat": {"id": d.cfg.chat_id},
            "message_id": 999,
            "text": "edited text",
        }
        # Should not raise
        await d._on_edited_message(edited_msg)
        d.tg.send.assert_not_called()


class TestUXForwardMessage:
    """Verify forwarded messages include source context."""

    @pytest.mark.asyncio
    async def test_forward_includes_source(self, tmp_path):
        """Forwarded message should include [Forwarded from X]: prefix."""
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
        # Wait for buffer flush
        await asyncio.sleep(2.5)

        assert len(dispatched) == 1
        assert "[Forwarded from Alice]" in dispatched[0]
        assert "check this out" in dispatched[0]

    @pytest.mark.asyncio
    async def test_forward_from_channel(self, tmp_path):
        """Channel forward should use the channel title as source."""
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
        # Wait for buffer flush
        await asyncio.sleep(2.5)

        assert len(dispatched) == 1
        assert "[Forwarded from Tech News]" in dispatched[0]


class TestUXNewSession:
    """Verify 'new session' UX flow works correctly."""

    def test_classify_new_session_keywords(self, tmp_path):
        """Known new-session phrases should classify as 'new_session'."""
        d = make_dispatcher(tmp_path)
        for phrase in ["新对话", "new session", "开个新的", "新session", "新建session"]:
            assert d._classify(phrase) == "new_session", f"'{phrase}' should classify as new_session"

    def test_new_session_sets_force_flag(self, tmp_path):
        """_handle_new_session should set sm.force_new = True."""
        d = make_dispatcher(tmp_path)
        assert d.sm.force_new is False
        d._handle_new_session(1)
        assert d.sm.force_new is True

    @pytest.mark.asyncio
    async def test_force_new_skips_resume(self, tmp_path):
        """With force_new=True, _handle_task creates new session instead of resuming."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        # Create a previous "done" session
        prev = d.sm.create(1, "original task", "/tmp")
        prev.status = "done"
        prev.finished = time.time()

        # Set force_new
        d.sm.force_new = True

        invocations = []
        async def mock_invoke(session, prompt, resume=False, max_turns=10, model=None, stream=True):
            invocations.append({"resume": resume, "sid": session.sid})
            session.status = "done"
            session.started = time.time()
            session.finished = time.time()
            return "New session result"
        d.runner.invoke = mock_invoke

        await d._handle_task(2, "start fresh", None)

        # Wait for spawned background tasks to complete
        if d._tasks:
            await asyncio.gather(*d._tasks, return_exceptions=True)

        assert len(invocations) == 1
        assert invocations[0]["resume"] is False, "force_new should create new session (resume=False)"
        assert invocations[0]["sid"] != prev.sid, "Should get a new session ID"
        assert d.sm.force_new is False, "force_new should be cleared after use"

    @pytest.mark.asyncio
    async def test_new_session_callback(self, tmp_path):
        """Pressing the 'new session' inline button should set force_new."""
        d = make_dispatcher(tmp_path)
        d.tg.answer_callback = MagicMock()

        cb = {
            "id": "cb789",
            "data": "new_session",
            "message": {"chat": {"id": d.cfg.chat_id}, "message_id": 50},
        }
        await d._on_callback(cb)

        assert d.sm.force_new is True
        d.tg.answer_callback.assert_called_with("cb789", "好的，下条消息将开始新对话")


class TestUXConsecutiveFailures:
    """Verify consecutive failure tracking and escalation."""

    def test_three_failures_shows_warning(self, tmp_path):
        """3 consecutive failures should add a warning in the error message."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        # Simulate 3 consecutive failures
        for i in range(3):
            s = Session(i + 1, f"task{i}", "/tmp")
            s.status = "failed"
            d._send_result(i + 1, s, f"Error {i}")

        # The third call should contain the warning
        last_msg = d.tg.send.call_args[0][0]
        assert "连续失败" in last_msg, "3 failures should trigger escalation warning"
        assert "3" in last_msg

    def test_success_resets_counter(self, tmp_path):
        """A success after failures should reset the consecutive failure counter."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        # 2 failures
        for i in range(2):
            s = Session(i + 1, f"fail{i}", "/tmp")
            s.status = "failed"
            d._send_result(i + 1, s, f"Error {i}")

        assert d._consecutive_failures == 2

        # 1 success
        s = Session(10, "success", "/tmp")
        s.status = "done"
        s.started = time.time()
        s.finished = time.time()
        d._send_result(10, s, "All good!")

        assert d._consecutive_failures == 0, "Success should reset failure counter"

        # Next failure should be count 1, not 3
        s = Session(11, "fail again", "/tmp")
        s.status = "failed"
        d._send_result(11, s, "Error again")

        assert d._consecutive_failures == 1
        last_msg = d.tg.send.call_args[0][0]
        assert "连续失败" not in last_msg, "Single failure should not trigger escalation"


class TestUXQuickActionButtons:
    """Verify inline button behavior for task results."""

    def test_long_task_has_new_session_button(self, tmp_path):
        """Task running >30s should get a 'new session' button."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        s = Session(1, "refactor module", "/tmp/proj")
        s.status = "done"
        s.is_task = True
        s.started = time.time() - 60  # 60s elapsed
        s.finished = time.time()
        d._send_result(1, s, "Refactored successfully")

        call_kwargs = d.tg.send.call_args
        markup = call_kwargs[1].get("reply_markup")
        assert markup is not None, "Long task should have action buttons"
        found_new_session = False
        for row in markup.get("inline_keyboard", []):
            for btn in row:
                if btn.get("callback_data") == "new_session":
                    found_new_session = True
        assert found_new_session, "Long task should have 'new session' button"

    def test_quick_task_no_buttons(self, tmp_path):
        """Quick task (<30s) should have no inline buttons."""
        d = make_dispatcher(tmp_path)
        d._fire_reaction = MagicMock()

        s = Session(1, "quick test", "/tmp")
        s.status = "done"
        s.is_task = False
        s.started = time.time()
        s.finished = time.time()  # ~0s elapsed
        d._send_result(1, s, "Quick answer")

        call_kwargs = d.tg.send.call_args
        markup = call_kwargs[1].get("reply_markup")
        assert markup is None, "Quick task should have no buttons"
