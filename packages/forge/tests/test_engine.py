"""Tests for forge.engine."""

from forge.engine import ForgeEngine
from forge.models import OutputType, SessionState, InstallTarget


class TestForgeEngine:
    def setup_method(self):
        self.engine = ForgeEngine()

    def test_create_session(self):
        session = self.engine.create_session("test tool", OutputType.PYTHON)
        assert session.id
        assert session.state == SessionState.CREATED
        assert session.description == "test tool"

    def test_get_session(self):
        session = self.engine.create_session("test", OutputType.PYTHON)
        found = self.engine.get_session(session.id)
        assert found is not None
        assert found.id == session.id

    def test_get_missing_session(self):
        assert self.engine.get_session("nonexistent") is None

    def test_clarify(self):
        session = self.engine.create_session("do something", OutputType.PYTHON)
        result = self.engine.clarify(session)
        assert result.has_questions
        assert session.state == SessionState.CLARIFYING

    def test_answer_questions(self):
        session = self.engine.create_session("filter file data", OutputType.PYTHON)
        clarification = self.engine.clarify(session)

        # Answer all required questions
        answers = {}
        for q in clarification.questions:
            if q.required:
                answers[q.id] = q.default or "test answer"

        result = self.engine.answer(session, answers)
        # After answering all required questions, should be ready
        assert session.state == SessionState.READY

    def test_generate_simple_tool(self):
        session = self.engine.create_session(
            "count words in a text string", OutputType.PYTHON
        )
        session.update_state(SessionState.READY)

        result = self.engine.generate(session)
        assert result.success
        assert session.generated_code
        assert session.spec is not None

    def test_full_pipeline_skip_clarify(self):
        session = self.engine.run_full_pipeline(
            description="return the input string reversed",
            output_type=OutputType.PYTHON,
            skip_clarify=True,
        )
        # Should have generated something
        assert session.spec is not None
        assert session.generated_code


class TestEngineIteration:
    def setup_method(self):
        self.engine = ForgeEngine()

    def test_max_iterations_respected(self):
        session = self.engine.create_session("test", OutputType.PYTHON)
        session.update_state(SessionState.READY)
        session.max_iterations = 2

        # Simulate reaching max
        session.iteration = 2
        session.update_state(SessionState.ITERATING)

        result = self.engine.iterate(session)
        assert not result.success
        assert session.state == SessionState.FAILED
