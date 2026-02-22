"""Engine — central orchestration for the forge tool-creation pipeline.

Manages the full lifecycle: intent parsing -> clarification -> design ->
generation -> testing -> iteration -> packaging -> installation.
"""

from __future__ import annotations

from forge.clarifier import analyze_ambiguity
from forge.generator import generate_tool, regenerate_with_fixes, spec_from_description
from forge.installer import install as install_tool
from forge.models import (
    ClarificationResult,
    GenerationResult,
    InstallResult,
    InstallTarget,
    OutputType,
    Session,
    SessionState,
    TestResult,
)
from forge.storage import save_tool
from forge.tester import run_tests


class ForgeEngine:
    """Orchestrates the full tool-creation pipeline.

    Manages sessions and drives each through the state machine:
    created -> clarifying -> ready -> generating -> testing ->
    (iterating) -> succeeded -> installed

    Usage:
        engine = ForgeEngine()
        session = engine.create_session("convert CSV to JSON", OutputType.PYTHON)
        clarification = engine.clarify(session)
        engine.answer(session, {"q_1": "file path", "q_2": "stdout"})
        result = engine.generate(session)
        if not result.success:
            result = engine.iterate(session)
        install = engine.install(session, InstallTarget.LOCAL)
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create_session(
        self,
        description: str,
        output_type: OutputType = OutputType.PYTHON,
    ) -> Session:
        """Create a new forge session.

        Args:
            description: Natural language description of the desired tool.
            output_type: What kind of tool to generate (mcp, cli, python).

        Returns:
            A new Session in CREATED state.
        """
        session = Session(
            description=description,
            output_type=output_type,
        )
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    def clarify(self, session: Session) -> ClarificationResult:
        """Generate clarification questions for a session.

        Args:
            session: The session to clarify.

        Returns:
            ClarificationResult with questions (may be empty).
        """
        result = analyze_ambiguity(session.description)
        session.questions = result.questions
        if result.has_questions:
            session.update_state(SessionState.CLARIFYING)
        else:
            session.update_state(SessionState.READY)
        return result

    def answer(self, session: Session, answers: dict[str, str]) -> ClarificationResult:
        """Submit answers to clarification questions.

        After answers are provided, checks if more clarification is needed.
        If not, moves session to READY state.

        Args:
            session: The session being clarified.
            answers: Dict mapping question_id -> answer string.

        Returns:
            ClarificationResult (empty if no more questions needed).
        """
        session.answers.update(answers)

        # Check if all required questions are answered
        unanswered = [
            q for q in session.questions
            if q.required and q.id not in session.answers
        ]

        if unanswered:
            return ClarificationResult(
                questions=unanswered,
                has_questions=True,
            )

        session.update_state(SessionState.READY)
        return ClarificationResult(questions=[], has_questions=False)

    def generate(self, session: Session) -> GenerationResult:
        """Generate tool code and run tests.

        Builds a ToolSpec from the description + answers, generates code
        using templates, then runs the auto-generated tests.

        Args:
            session: A session in READY state.

        Returns:
            GenerationResult with code and test outcomes.
        """
        session.update_state(SessionState.GENERATING)

        # Build specification
        spec = spec_from_description(
            session.description,
            session.answers,
            session.output_type,
        )
        session.spec = spec

        # Generate code
        result = generate_tool(spec, session.output_type)
        if not result.success:
            session.update_state(SessionState.FAILED)
            return result

        session.generated_code = result.tool_code
        session.generated_tests = result.test_code

        # Run tests
        session.update_state(SessionState.TESTING)
        test_result = run_tests(
            tool_code=result.tool_code,
            test_code=result.test_code,
            tool_name=spec.name,
            dependencies=spec.dependencies,
        )
        session.test_results.append(test_result)

        if test_result.passed:
            session.update_state(SessionState.SUCCEEDED)
            # Auto-save on success
            save_tool(session)
        else:
            session.update_state(SessionState.ITERATING)

        return result

    def iterate(
        self,
        session: Session,
        feedback: str | None = None,
    ) -> GenerationResult:
        """Attempt to fix issues and regenerate.

        Analyzes test failures, applies fixes, regenerates, and re-tests.
        Repeats up to max_iterations times.

        Args:
            session: A session in ITERATING or FAILED state.
            feedback: Optional additional feedback from the user.

        Returns:
            GenerationResult from the latest iteration.
        """
        if session.iteration >= session.max_iterations:
            session.update_state(SessionState.FAILED)
            return GenerationResult(
                success=False,
                error=f"Max iterations ({session.max_iterations}) reached without passing tests",
            )

        session.iteration += 1
        session.update_state(SessionState.ITERATING)

        # Collect errors from last test run
        errors = []
        if session.test_results:
            errors = session.test_results[-1].errors
        if feedback:
            errors.append(f"User feedback: {feedback}")

        # Regenerate with fixes
        assert session.spec is not None
        result = regenerate_with_fixes(
            spec=session.spec,
            output_type=session.output_type,
            errors=errors,
            previous_code=session.generated_code,
        )

        if not result.success:
            session.update_state(SessionState.FAILED)
            return result

        session.generated_code = result.tool_code
        session.generated_tests = result.test_code

        # Re-test
        session.update_state(SessionState.TESTING)
        test_result = run_tests(
            tool_code=result.tool_code,
            test_code=result.test_code,
            tool_name=session.spec.name,
            dependencies=session.spec.dependencies,
        )
        session.test_results.append(test_result)

        if test_result.passed:
            session.update_state(SessionState.SUCCEEDED)
            save_tool(session)
        elif session.iteration < session.max_iterations:
            session.update_state(SessionState.ITERATING)
        else:
            session.update_state(SessionState.FAILED)

        return result

    def install(
        self,
        session: Session,
        target: InstallTarget,
    ) -> InstallResult:
        """Install the generated tool.

        Args:
            session: A session in SUCCEEDED state.
            target: Where to install (mcp, cli, local).

        Returns:
            InstallResult with installation details.
        """
        result = install_tool(session, target)
        session.install_result = result

        if result.success:
            session.update_state(SessionState.INSTALLED)
            # Update storage with install info
            save_tool(session)

        return result

    def run_full_pipeline(
        self,
        description: str,
        output_type: OutputType = OutputType.PYTHON,
        answers: dict[str, str] | None = None,
        skip_clarify: bool = False,
        install_target: InstallTarget | None = None,
    ) -> Session:
        """Run the entire pipeline end-to-end.

        Convenience method that runs all steps: create -> clarify -> generate
        -> iterate (if needed) -> install (if target specified).

        Args:
            description: Tool description.
            output_type: Desired output type.
            answers: Pre-supplied answers to clarification questions.
            skip_clarify: If True, skip clarification step.
            install_target: If set, install the tool to this target.

        Returns:
            The completed Session with all results.
        """
        session = self.create_session(description, output_type)

        # Clarification
        if not skip_clarify:
            clarification = self.clarify(session)
            if clarification.has_questions and answers:
                self.answer(session, answers)
            elif clarification.has_questions:
                # Can't proceed without answers in non-interactive mode
                return session
        else:
            session.update_state(SessionState.READY)

        # Generation
        result = self.generate(session)
        if not result.success:
            return session

        # Iteration loop
        while session.state == SessionState.ITERATING:
            self.iterate(session)

        # Installation
        if session.state == SessionState.SUCCEEDED and install_target:
            self.install(session, install_target)

        return session
