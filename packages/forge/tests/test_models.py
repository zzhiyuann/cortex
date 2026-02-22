"""Tests for forge.models."""

from forge.models import (
    ClarificationQuestion,
    ClarificationResult,
    GenerationResult,
    InstallResult,
    InstallTarget,
    OutputType,
    Session,
    SessionState,
    TestResult,
    ToolMetadata,
    ToolParam,
    ToolSpec,
)


class TestOutputType:
    def test_values(self):
        assert OutputType.MCP == "mcp"
        assert OutputType.CLI == "cli"
        assert OutputType.PYTHON == "python"


class TestSessionState:
    def test_all_states(self):
        states = [s.value for s in SessionState]
        assert "created" in states
        assert "succeeded" in states
        assert "failed" in states
        assert "installed" in states


class TestToolParam:
    def test_basic(self):
        p = ToolParam(name="input_path", type_hint="str", description="Path to file")
        assert p.name == "input_path"
        assert p.required is True

    def test_optional(self):
        p = ToolParam(name="verbose", type_hint="bool", required=False, default="False")
        assert p.required is False
        assert p.default == "False"


class TestToolSpec:
    def test_creation(self):
        spec = ToolSpec(
            name="my_tool",
            description="A test tool",
            params=[ToolParam(name="x", type_hint="str", description="input")],
        )
        assert spec.name == "my_tool"
        assert len(spec.params) == 1
        assert spec.return_type == "str"


class TestSession:
    def test_creation(self):
        s = Session(description="test tool")
        assert s.state == SessionState.CREATED
        assert len(s.id) == 12
        assert s.iteration == 0

    def test_update_state(self):
        s = Session(description="test")
        old_time = s.updated_at
        s.update_state(SessionState.READY)
        assert s.state == SessionState.READY
        assert s.updated_at >= old_time


class TestTestResult:
    def test_default(self):
        r = TestResult()
        assert r.passed is False
        assert r.total == 0

    def test_passed(self):
        r = TestResult(passed=True, total=5, failures=0)
        assert r.passed is True


class TestGenerationResult:
    def test_success(self):
        r = GenerationResult(success=True, tool_code="x = 1", test_code="assert True")
        assert r.success is True
        assert r.error is None


class TestInstallResult:
    def test_success(self):
        r = InstallResult(success=True, install_path="/tmp/tool.py", target=InstallTarget.LOCAL)
        assert r.success is True


class TestClarificationQuestion:
    def test_basic(self):
        q = ClarificationQuestion(
            id="q_1",
            question="What format?",
            category="input",
        )
        assert q.required is True
        assert q.options is None


class TestToolMetadata:
    def test_basic(self):
        m = ToolMetadata(
            name="test_tool",
            description="A test tool",
            output_type=OutputType.PYTHON,
        )
        assert m.name == "test_tool"
        assert m.installed is False
