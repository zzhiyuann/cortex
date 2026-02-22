"""Pydantic models for Vibe Replay events, sessions, and analysis."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events captured during a coding session."""

    TOOL_CALL = "tool_call"
    CODE_CHANGE = "code_change"
    DECISION = "decision"
    ERROR = "error"
    USER_MESSAGE = "user_message"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    NOTIFICATION = "notification"


class SessionPhase(str, Enum):
    """Phases of a coding session identified by the analyzer."""

    EXPLORATION = "exploration"
    IMPLEMENTATION = "implementation"
    DEBUGGING = "debugging"
    TESTING = "testing"
    REFACTORING = "refactoring"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


class InsightType(str, Enum):
    """Types of insights extracted from sessions."""

    DECISION = "decision"
    LEARNING = "learning"
    MISTAKE = "mistake"
    PATTERN = "pattern"
    TURNING_POINT = "turning_point"


class Event(BaseModel):
    """A single captured event from a coding session."""

    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str
    event_type: EventType
    tool_name: str | None = None
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    code_diff: str | None = None
    files_affected: list[str] = Field(default_factory=list)

    def model_dump_jsonl(self) -> str:
        """Serialize to a single JSON line."""
        return self.model_dump_json()


class SessionMetadata(BaseModel):
    """Metadata about a captured session."""

    session_id: str
    project: str = ""
    project_path: str = ""
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None
    duration_seconds: float | None = None
    event_count: int = 0
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    tools_used: dict[str, int] = Field(default_factory=dict)


class Insight(BaseModel):
    """A structured insight extracted from session analysis."""

    insight_type: InsightType
    title: str
    description: str
    supporting_events: list[int] = Field(
        default_factory=list,
        description="Indices of events that support this insight",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for this insight",
    )


class TimelinePhase(BaseModel):
    """A phase in the session timeline."""

    phase: SessionPhase
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    event_count: int
    summary: str = ""
    key_events: list[int] = Field(
        default_factory=list,
        description="Indices of notable events in this phase",
    )


class SessionReplay(BaseModel):
    """Processed replay data for a session."""

    metadata: SessionMetadata
    timeline: list[TimelinePhase] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    key_decision_indices: list[int] = Field(default_factory=list)
    turning_point_indices: list[int] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)


class HookPayload(BaseModel):
    """Payload received from Claude Code hooks."""

    session_id: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    hook_type: str = ""
    # Additional fields that might be present
    message: str | None = None
    error: str | None = None
