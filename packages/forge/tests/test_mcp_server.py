"""Tests for forge.mcp_server — MCP tool error handling and forge_status."""

from forge.mcp_server import (
    engine,
    forge_answer,
    forge_create,
    forge_generate,
    forge_install,
    forge_iterate,
    forge_list,
    forge_status,
)
from forge.models import SessionState


class TestForgeCreate:
    """Test forge_create MCP tool."""

    def test_basic_create(self):
        result = forge_create(description="convert CSV to JSON", output_type="python")
        assert "session_id" in result
        assert result["state"] in ("clarifying", "ready")

    def test_empty_description(self):
        result = forge_create(description="", output_type="python")
        assert "error" in result

    def test_invalid_output_type(self):
        result = forge_create(description="test tool", output_type="invalid_type")
        assert "error" in result
        assert "invalid" in result["error"].lower()

    def test_module_output_type(self):
        result = forge_create(description="track expenses", output_type="module")
        assert "session_id" in result
        assert result["output_type"] == "module"


class TestForgeAnswer:
    """Test forge_answer MCP tool."""

    def test_not_found(self):
        result = forge_answer(session_id="nonexistent_id", answers={"q1": "a1"})
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_empty_answers(self):
        create = forge_create(description="test tool", output_type="python")
        result = forge_answer(session_id=create["session_id"], answers={})
        assert "error" in result

    def test_answer_questions(self):
        create = forge_create(description="filter data in a file", output_type="python")
        sid = create["session_id"]

        if "questions" in create:
            answers = {q["id"]: "test" for q in create["questions"]}
            result = forge_answer(session_id=sid, answers=answers)
            assert "error" not in result or result.get("state") == "ready"


class TestForgeGenerate:
    """Test forge_generate MCP tool."""

    def test_not_found(self):
        result = forge_generate(session_id="nonexistent_id")
        assert "error" in result

    def test_generate_from_ready(self):
        create = forge_create(description="return input string", output_type="python")
        sid = create["session_id"]
        result = forge_generate(session_id=sid)
        assert "success" in result

    def test_generate_already_succeeded(self):
        create = forge_create(description="echo tool", output_type="python")
        sid = create["session_id"]
        gen1 = forge_generate(session_id=sid)

        if gen1.get("success") and engine.get_session(sid).state == SessionState.SUCCEEDED:
            gen2 = forge_generate(session_id=sid)
            assert "error" in gen2


class TestForgeIterate:
    """Test forge_iterate MCP tool."""

    def test_not_found(self):
        result = forge_iterate(session_id="nonexistent_id")
        assert "error" in result

    def test_iterate_wrong_state(self):
        create = forge_create(description="test", output_type="python")
        sid = create["session_id"]
        # Try iterating without generating first
        result = forge_iterate(session_id=sid)
        assert "error" in result


class TestForgeInstall:
    """Test forge_install MCP tool."""

    def test_not_found(self):
        result = forge_install(session_id="nonexistent_id")
        assert "error" in result

    def test_install_wrong_state(self):
        create = forge_create(description="test", output_type="python")
        sid = create["session_id"]
        result = forge_install(session_id=sid)
        assert "error" in result
        assert "succeeded" in result["error"].lower()

    def test_invalid_target(self):
        create = forge_create(description="test tool", output_type="python")
        sid = create["session_id"]
        forge_generate(session_id=sid)
        # Force session to succeeded for install test
        session = engine.get_session(sid)
        if session:
            session.update_state(SessionState.SUCCEEDED)
        result = forge_install(session_id=sid, target="invalid_target")
        assert "error" in result


class TestForgeStatus:
    """Test forge_status MCP tool."""

    def test_overview_mode(self):
        # Create a session so there's at least one
        forge_create(description="status test tool", output_type="python")
        result = forge_status(session_id="")
        assert "active_sessions" in result
        assert "saved_tools" in result
        assert "active_session_count" in result
        assert isinstance(result["active_sessions"], list)

    def test_specific_session(self):
        create = forge_create(description="status detail test", output_type="python")
        sid = create["session_id"]
        result = forge_status(session_id=sid)
        assert result["session_id"] == sid
        assert "state" in result
        assert "description" in result

    def test_not_found(self):
        result = forge_status(session_id="nonexistent_id")
        assert "error" in result


class TestForgeList:
    """Test forge_list MCP tool."""

    def test_list(self):
        result = forge_list()
        assert "count" in result
        assert "tools" in result
        assert isinstance(result["tools"], list)
