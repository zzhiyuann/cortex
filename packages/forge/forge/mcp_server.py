"""MCP Server for Forge — exposes tool-creation capabilities to Claude Code.

Provides MCP tools that mirror the forge CLI workflow:
create -> clarify -> generate -> iterate -> install.

All tools include robust error handling and return structured responses
with clear error messages on failure.
"""

from __future__ import annotations

import logging
import traceback

from mcp.server.fastmcp import FastMCP

from forge.engine import ForgeEngine
from forge.models import InstallTarget, OutputType, SessionState
from forge.storage import list_tools

logger = logging.getLogger(__name__)

mcp = FastMCP("forge", instructions=(
    "Forge is a Self-Evolving Tool Agent. Use it to create new tools from "
    "natural language descriptions. The workflow is: forge_create -> "
    "(forge_answer if questions) -> forge_generate -> (forge_iterate if "
    "tests fail) -> forge_install."
))

engine = ForgeEngine()


def _safe_response(func):
    """Decorator that wraps MCP tool functions with error handling.

    Catches all exceptions and returns a structured error response
    instead of letting the exception propagate.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning("Validation error in %s: %s", func.__name__, e)
            return {
                "error": str(e),
                "error_type": "validation_error",
                "tool": func.__name__,
            }
        except Exception as e:
            logger.error(
                "Unexpected error in %s: %s\n%s",
                func.__name__,
                e,
                traceback.format_exc(),
            )
            return {
                "error": f"Internal error: {e}",
                "error_type": "internal_error",
                "tool": func.__name__,
            }

    # Preserve the original function metadata for MCP
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__
    return wrapper


@mcp.tool()
@_safe_response
def forge_create(
    description: str,
    output_type: str = "python",
) -> dict:
    """Start creating a new tool from a natural language description.

    Args:
        description: What the tool should do (natural language).
        output_type: Output format — "mcp", "cli", "python", or "module".

    Returns:
        Session ID and clarification questions (if any).
    """
    if not description or not description.strip():
        return {
            "error": "Description cannot be empty. Please describe what tool you want.",
            "error_type": "validation_error",
        }

    valid_types = {t.value for t in OutputType}
    if output_type not in valid_types:
        return {
            "error": (
                f"Invalid output type '{output_type}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}"
            ),
            "error_type": "validation_error",
        }

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
@_safe_response
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
        return {
            "error": f"Session '{session_id}' not found. It may have expired or the ID is incorrect.",
            "error_type": "not_found",
        }

    if session.state not in (SessionState.CLARIFYING, SessionState.CREATED):
        return {
            "error": (
                f"Session is in state '{session.state.value}', "
                "but answers are only accepted during 'created' or 'clarifying' state."
            ),
            "error_type": "invalid_state",
            "session_id": session_id,
            "current_state": session.state.value,
        }

    if not answers:
        return {
            "error": "No answers provided. Pass a dict of question_id -> answer.",
            "error_type": "validation_error",
            "session_id": session_id,
        }

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
@_safe_response
def forge_generate(session_id: str) -> dict:
    """Generate and test the tool code.

    Args:
        session_id: The session ID.

    Returns:
        Generation result with code, test results, and status.
    """
    session = engine.get_session(session_id)
    if not session:
        return {
            "error": f"Session '{session_id}' not found. It may have expired or the ID is incorrect.",
            "error_type": "not_found",
        }

    allowed_states = (SessionState.READY, SessionState.CLARIFYING, SessionState.CREATED)
    if session.state not in allowed_states:
        if session.state in (SessionState.SUCCEEDED, SessionState.INSTALLED):
            return {
                "error": (
                    f"Session has already succeeded. "
                    "Use forge_install to install it, or create a new session."
                ),
                "error_type": "invalid_state",
                "session_id": session_id,
                "current_state": session.state.value,
            }
        return {
            "error": (
                f"Session is in state '{session.state.value}', expected one of: "
                f"{', '.join(s.value for s in allowed_states)}. "
                "If generation previously failed, create a new session."
            ),
            "error_type": "invalid_state",
            "session_id": session_id,
            "current_state": session.state.value,
        }

    # Auto-advance from clarifying/created to ready
    if session.state in (SessionState.CLARIFYING, SessionState.CREATED):
        session.update_state(SessionState.READY)

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
                "Call forge_iterate to attempt fixes, or provide feedback."
            )
    else:
        response["error"] = gen_result.error or "Unknown generation error"
        response["message"] = (
            "Generation failed. Check the error and try creating a new session "
            "with a more specific description."
        )

    return response


@mcp.tool()
@_safe_response
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
        return {
            "error": f"Session '{session_id}' not found. It may have expired or the ID is incorrect.",
            "error_type": "not_found",
        }

    if session.state not in (SessionState.ITERATING, SessionState.TESTING, SessionState.FAILED):
        return {
            "error": (
                f"Session is in state '{session.state.value}'. "
                "Iteration is only possible in 'iterating', 'testing', or 'failed' states."
            ),
            "error_type": "invalid_state",
            "session_id": session_id,
            "current_state": session.state.value,
        }

    # Allow iterating from failed state by resetting
    if session.state == SessionState.FAILED and session.iteration < session.max_iterations:
        session.update_state(SessionState.ITERATING)

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
        response["message"] = (
            f"Failed after {session.iteration} of {session.max_iterations} iterations. "
            "Consider creating a new session with a revised description."
        )
        if session.test_results:
            response["last_errors"] = session.test_results[-1].errors[:5]
    else:
        response["message"] = (
            f"Iteration {session.iteration}/{session.max_iterations}: still failing. "
            "Call forge_iterate again with feedback, or check the errors."
        )
        if session.test_results:
            response["last_errors"] = session.test_results[-1].errors[:5]

    return response


@mcp.tool()
@_safe_response
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
        return {
            "error": f"Session '{session_id}' not found. It may have expired or the ID is incorrect.",
            "error_type": "not_found",
        }

    if session.state != SessionState.SUCCEEDED:
        return {
            "error": (
                f"Session is in state '{session.state.value}'. "
                "Installation requires the session to be in 'succeeded' state. "
                "Generate and pass tests first."
            ),
            "error_type": "invalid_state",
            "session_id": session_id,
            "current_state": session.state.value,
        }

    valid_targets = {t.value for t in InstallTarget}
    if target not in valid_targets:
        return {
            "error": (
                f"Invalid install target '{target}'. "
                f"Must be one of: {', '.join(sorted(valid_targets))}"
            ),
            "error_type": "validation_error",
        }

    install_target = InstallTarget(target)
    result = engine.install(session, install_target)

    response = {
        "session_id": session_id,
        "success": result.success,
        "target": result.target.value,
        "install_path": result.install_path,
        "message": result.message,
    }

    if not result.success:
        response["error"] = result.error or "Installation failed for unknown reason"

    return response


@mcp.tool()
@_safe_response
def forge_status(session_id: str = "") -> dict:
    """Check the status of a forge session, or list all active sessions.

    If session_id is provided, returns detailed status for that session.
    If empty, returns an overview of all active sessions and saved tools.

    Args:
        session_id: The session ID (optional — omit for overview).

    Returns:
        Session state, generated code, test results, and tool inventory.
    """
    # Overview mode: list all sessions and saved tools
    if not session_id:
        active_sessions = []
        for sid, session in engine._sessions.items():
            entry = {
                "session_id": sid,
                "state": session.state.value,
                "description": session.description[:80],
                "output_type": session.output_type.value,
                "iteration": session.iteration,
                "created_at": session.created_at.isoformat(),
            }
            if session.spec:
                entry["tool_name"] = session.spec.name
            active_sessions.append(entry)

        saved_tools = []
        for tool in list_tools():
            saved_tools.append({
                "name": tool.name,
                "description": tool.description[:80],
                "output_type": tool.output_type.value,
                "test_passed": tool.test_passed,
                "installed": tool.installed,
                "created_at": tool.created_at.isoformat(),
            })

        return {
            "active_sessions": active_sessions,
            "active_session_count": len(active_sessions),
            "saved_tools": saved_tools,
            "saved_tool_count": len(saved_tools),
            "message": (
                f"{len(active_sessions)} active session(s), "
                f"{len(saved_tools)} saved tool(s)."
            ),
        }

    # Detail mode: specific session
    session = engine.get_session(session_id)
    if not session:
        return {
            "error": f"Session '{session_id}' not found. It may have expired or the ID is incorrect.",
            "error_type": "not_found",
        }

    response: dict = {
        "session_id": session_id,
        "state": session.state.value,
        "description": session.description,
        "output_type": session.output_type.value,
        "iteration": session.iteration,
        "max_iterations": session.max_iterations,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }

    if session.spec:
        response["tool_name"] = session.spec.name
        response["tool_display_name"] = session.spec.display_name
        response["dependencies"] = session.spec.dependencies

    if session.generated_code:
        response["code"] = session.generated_code
        response["code_length"] = len(session.generated_code)

    if session.generated_tests:
        response["test_code_length"] = len(session.generated_tests)

    if session.test_results:
        last_test = session.test_results[-1]
        response["tests"] = {
            "passed": last_test.passed,
            "total": last_test.total,
            "failures": last_test.failures,
            "errors": last_test.errors[:5],
        }
        response["test_history"] = [
            {"passed": t.passed, "total": t.total, "failures": t.failures}
            for t in session.test_results
        ]

    if session.install_result:
        response["install"] = {
            "success": session.install_result.success,
            "path": session.install_result.install_path,
            "target": session.install_result.target.value,
            "message": session.install_result.message,
        }

    if session.questions:
        response["questions_count"] = len(session.questions)
        response["answers_count"] = len(session.answers)

    return response


@mcp.tool()
@_safe_response
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
                "display_name": t.display_name,
                "description": t.description,
                "output_type": t.output_type.value,
                "test_passed": t.test_passed,
                "installed": t.installed,
                "install_path": t.install_path,
                "dependencies": t.dependencies,
                "created_at": t.created_at.isoformat(),
            }
            for t in tools
        ],
        "message": f"{len(tools)} tool(s) in the forge inventory." if tools else "No tools created yet.",
    }


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
