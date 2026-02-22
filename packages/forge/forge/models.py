"""Pydantic models for Forge data structures."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class OutputType(str, Enum):
    """Supported output types for generated tools."""

    MCP = "mcp"
    CLI = "cli"
    PYTHON = "python"


class SessionState(str, Enum):
    """States a forge session can be in."""

    CREATED = "created"
    CLARIFYING = "clarifying"
    READY = "ready"
    GENERATING = "generating"
    TESTING = "testing"
    ITERATING = "iterating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INSTALLED = "installed"


class ClarificationQuestion(BaseModel):
    """A single clarification question for the user."""

    id: str = Field(description="Unique identifier for the question")
    question: str = Field(description="The question text")
    category: str = Field(description="Category: input, output, edge_case, dependency, behavior")
    required: bool = Field(default=True, description="Whether an answer is required")
    default: str | None = Field(default=None, description="Suggested default answer")
    options: list[str] | None = Field(default=None, description="Multiple choice options if any")


class ClarificationResult(BaseModel):
    """Result from the clarifier."""

    questions: list[ClarificationQuestion] = Field(default_factory=list)
    has_questions: bool = Field(default=False)


class ToolParam(BaseModel):
    """A parameter for the generated tool."""

    name: str
    type_hint: str = Field(default="str")
    description: str = Field(default="")
    required: bool = Field(default=True)
    default: Any = Field(default=None)


class ToolSpec(BaseModel):
    """Full specification for a tool to be generated."""

    name: str = Field(description="Tool function/command name (snake_case)")
    display_name: str = Field(default="", description="Human-readable tool name")
    description: str = Field(description="What the tool does")
    params: list[ToolParam] = Field(default_factory=list, description="Tool parameters")
    return_type: str = Field(default="str", description="Return type hint")
    return_description: str = Field(default="", description="What the return value represents")
    dependencies: list[str] = Field(default_factory=list, description="pip packages needed")
    core_logic: str = Field(default="", description="The main logic of the tool as code")
    error_handling: str = Field(default="", description="Error handling notes")
    examples: list[dict[str, Any]] = Field(
        default_factory=list, description="Example input/output pairs"
    )


class TestResult(BaseModel):
    """Result from running tests."""

    passed: bool = Field(default=False)
    total: int = Field(default=0)
    failures: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    output: str = Field(default="")
    test_code: str = Field(default="")


class GenerationResult(BaseModel):
    """Result from code generation."""

    success: bool = Field(default=False)
    tool_code: str = Field(default="")
    test_code: str = Field(default="")
    spec: ToolSpec | None = Field(default=None)
    error: str | None = Field(default=None)


class InstallTarget(str, Enum):
    """Where to install a tool."""

    MCP = "mcp"
    CLI = "cli"
    LOCAL = "local"


class InstallResult(BaseModel):
    """Result from installing a tool."""

    success: bool = Field(default=False)
    install_path: str = Field(default="")
    target: InstallTarget = Field(default=InstallTarget.LOCAL)
    message: str = Field(default="")
    error: str | None = Field(default=None)


class Session(BaseModel):
    """A forge tool-creation session tracking all state."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = Field(description="User's original tool description")
    output_type: OutputType = Field(default=OutputType.PYTHON)
    state: SessionState = Field(default=SessionState.CREATED)

    # Clarification
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)

    # Specification & generation
    spec: ToolSpec | None = Field(default=None)
    generated_code: str = Field(default="")
    generated_tests: str = Field(default="")

    # Testing & iteration
    test_results: list[TestResult] = Field(default_factory=list)
    iteration: int = Field(default=0)
    max_iterations: int = Field(default=5)

    # Installation
    install_result: InstallResult | None = Field(default=None)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def update_state(self, state: SessionState) -> None:
        """Update session state and timestamp."""
        self.state = state
        self.updated_at = datetime.now(timezone.utc)


class ToolMetadata(BaseModel):
    """Stored metadata about a created tool."""

    name: str
    display_name: str = Field(default="")
    description: str
    output_type: OutputType
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = Field(default="")
    tool_path: str = Field(default="")
    test_path: str = Field(default="")
    install_path: str = Field(default="")
    installed: bool = Field(default=False)
    test_passed: bool = Field(default=False)
    dependencies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
