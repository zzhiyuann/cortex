"""MCP Server for Forge — exposes tool-creation capabilities to Claude Code.

Provides MCP tools that mirror the forge CLI workflow:
create -> clarify -> generate -> iterate -> install.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from forge.engine import ForgeEngine
from forge.models import InstallTarget, OutputType, SessionState
from forge.storage import list_tools

mcp = FastMCP("forge", instructions=(
    "Forge is a Self-Evolving Tool Agent. Use it to create new tools from "
    "natural language descriptions. The workflow is: forge_create -> "
    "(forge_answer if questions) -> forge_generate -> (forge_iterate if "
    "tests fail) -> forge_install."
))

engine = ForgeEngine()


@mcp.tool()
def forge_create(
    description: str,
    output_type: str = "python",
) -> dict:
    """Start creating a new tool from a natural language description.

    Args:
        description: What the tool should do (natural language).
        output_type: Output format — "mcp", "cli", or "python".

    Returns:
        Session ID and clarification questions (if any).
    """
    otype = OutputType(output_type)
    session = engine.create_session(description, otype)
    clarification = engine.clarify(session)

    result = {
        "session_id": session.id,
        "state": session.state.value,
        "description": description,
        "output_type": otype.value,
    }

    if clarification.has_questions:
        result["questions"] = [
            {
                "id": q.id,
                "question": q.question,
                "category": q.category,
                "required": q.required,
                "default": q.default,
                "options": q.options,
            }
            for q in clarification.questions
        ]
        result["message"] = (
            f"I have {len(clarification.questions)} clarification question(s). "
            "Please answer them using forge_answer."
        )
    else:
        result["message"] = "No clarification needed. Ready to generate — call forge_generate."

    return result


@mcp.tool()
def forge_answer(
    session_id: str,
    answers: dict[str, str],
) -> dict:
    """Answer clarification questions for a tool being created.

    Args:
        session_id: The session ID from forge_create.
        answers: Dict mapping question_id to answer string.

    Returns:
        Status update — either more questions or "ready to generate".
    """
    session = engine.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    result = engine.answer(session, answers)

    if result.has_questions:
        return {
            "session_id": session_id,
            "state": session.state.value,
            "remaining_questions": [
                {"id": q.id, "question": q.question}
                for q in result.questions
            ],
            "message": "Some required questions are still unanswered.",
        }

    return {
        "session_id": session_id,
        "state": session.state.value,
        "message": "All questions answered. Ready to generate — call forge_generate.",
    }


@mcp.tool()
def forge_generate(session_id: str) -> dict:
    """Generate and test the tool code.

    Args:
        session_id: The session ID.

    Returns:
        Generation result with code, test results, and status.
    """
    session = engine.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    if session.state not in (SessionState.READY, SessionState.CLARIFYING):
        # Allow re-generation from clarifying state (auto-skip remaining questions)
        if session.state == SessionState.CLARIFYING:
            session.update_state(SessionState.READY)
        elif session.state != SessionState.READY:
            return {
                "error": f"Session is in state '{session.state.value}', expected 'ready'",
                "session_id": session_id,
            }

    gen_result = engine.generate(session)

    response: dict = {
        "session_id": session_id,
        "state": session.state.value,
        "success": gen_result.success,
    }

    if gen_result.success:
        response["tool_name"] = session.spec.name if session.spec else "unknown"
        response["code_length"] = len(session.generated_code)

        if session.test_results:
            last_test = session.test_results[-1]
            response["tests"] = {
                "passed": last_test.passed,
                "total": last_test.total,
                "failures": last_test.failures,
                "errors": last_test.errors[:5],
            }

        if session.state == SessionState.SUCCEEDED:
            response["message"] = (
                "Tool generated and all tests passed! "
                "Call forge_install to install it."
            )
            response["code"] = session.generated_code
        else:
            response["message"] = (
                "Tool generated but tests failed. "
                "Call forge_iterate to attempt fixes."
            )
    else:
        response["error"] = gen_result.error

    return response


@mcp.tool()
def forge_iterate(
    session_id: str,
    feedback: str = "",
) -> dict:
    """Fix issues and regenerate the tool.

    Args:
        session_id: The session ID.
        feedback: Optional additional feedback about what's wrong.

    Returns:
        Iteration result with updated code and test status.
    """
    session = engine.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    iter_result = engine.iterate(session, feedback or None)

    response: dict = {
        "session_id": session_id,
        "state": session.state.value,
        "iteration": session.iteration,
        "max_iterations": session.max_iterations,
    }

    if session.state == SessionState.SUCCEEDED:
        response["message"] = "Tests passed after iteration! Call forge_install to install."
        response["code"] = session.generated_code
        if session.test_results:
            last_test = session.test_results[-1]
            response["tests"] = {
                "passed": True,
                "total": last_test.total,
            }
    elif session.state == SessionState.FAILED:
        response["message"] = f"Failed after {session.iteration} iterations."
        if session.test_results:
            response["last_errors"] = session.test_results[-1].errors[:5]
    else:
        response["message"] = (
            f"Iteration {session.iteration}: still failing. "
            "Call forge_iterate again or provide feedback."
        )
        if session.test_results:
            response["last_errors"] = session.test_results[-1].errors[:5]

    return response


@mcp.tool()
def forge_install(
    session_id: str,
    target: str = "local",
) -> dict:
    """Install the generated tool.

    Args:
        session_id: The session ID.
        target: Install target — "mcp", "cli", or "local".

    Returns:
        Installation result with path and instructions.
    """
    session = engine.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    if session.state != SessionState.SUCCEEDED:
        return {
            "error": f"Session is in state '{session.state.value}', must be 'succeeded' to install",
            "session_id": session_id,
        }

    install_target = InstallTarget(target)
    result = engine.install(session, install_target)

    return {
        "session_id": session_id,
        "success": result.success,
        "target": result.target.value,
        "install_path": result.install_path,
        "message": result.message,
        "error": result.error,
    }


@mcp.tool()
def forge_status(session_id: str) -> dict:
    """Check the status of a forge session.

    Args:
        session_id: The session ID.

    Returns:
        Current session state, generated code, and test results.
    """
    session = engine.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    response: dict = {
        "session_id": session_id,
        "state": session.state.value,
        "description": session.description,
        "output_type": session.output_type.value,
        "iteration": session.iteration,
        "created_at": session.created_at.isoformat(),
    }

    if session.spec:
        response["tool_name"] = session.spec.name

    if session.generated_code:
        response["code"] = session.generated_code

    if session.test_results:
        last_test = session.test_results[-1]
        response["tests"] = {
            "passed": last_test.passed,
            "total": last_test.total,
            "failures": last_test.failures,
            "errors": last_test.errors[:5],
        }

    if session.install_result:
        response["install"] = {
            "success": session.install_result.success,
            "path": session.install_result.install_path,
            "target": session.install_result.target.value,
        }

    return response


@mcp.tool()
def forge_list() -> dict:
    """List all tools created by Forge.

    Returns:
        List of tools with their metadata.
    """
    tools = list_tools()
    return {
        "count": len(tools),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "output_type": t.output_type.value,
                "test_passed": t.test_passed,
                "installed": t.installed,
                "created_at": t.created_at.isoformat(),
            }
            for t in tools
        ],
    }


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
