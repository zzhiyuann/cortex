"""A2A Hub Protocol — Message types and validation.

Defines the JSON message format exchanged over WebSocket between agents
and the hub server. All messages are validated with Pydantic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """All supported message types in the A2A protocol."""

    REGISTER = "register"
    DEREGISTER = "deregister"
    DISCOVER = "discover"
    DELEGATE = "delegate"
    RESULT = "result"
    STATUS = "status"
    BROADCAST = "broadcast"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"


class TaskStatus(str, Enum):
    """Lifecycle states of a delegated task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AgentInfo(BaseModel):
    """Describes a registered agent."""

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Free-form capability tags (e.g. 'code-review', 'web-search')",
    )
    status: str = Field(default="online", description="Agent status")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (description, version, etc.)",
    )


class TaskRecord(BaseModel):
    """Tracks the state of a delegated task."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = Field(..., description="Agent that delegated the task")
    to_agent: str = Field(..., description="Agent that should execute the task")
    capability: str = Field(..., description="Capability being invoked")
    params: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: Any = Field(default=None)
    error: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = Field(default=None)
    ttl_seconds: int = Field(default=300, description="Time-to-live in seconds")


class A2AMessage(BaseModel):
    """Top-level protocol message exchanged over WebSocket.

    Every message flowing through the hub follows this schema.
    """

    type: MessageType
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message ID",
    )
    from_agent: str = Field(
        alias="from",
        default="unknown",
        description="Sender agent ID",
    )
    to: str = Field(
        default="hub",
        description="Recipient agent ID, 'hub', or 'broadcast'",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp",
    )
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "populate_by_name": True,
    }

    def to_json(self) -> str:
        """Serialize to JSON string using 'from' as the key name."""
        return self.model_dump_json(by_alias=True)

    @classmethod
    def from_json(cls, data: str) -> "A2AMessage":
        """Deserialize from a JSON string."""
        return cls.model_validate_json(data)


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------

def make_register(
    agent_id: str,
    name: str,
    capabilities: list[str],
    metadata: dict[str, Any] | None = None,
) -> A2AMessage:
    """Create a REGISTER message."""
    return A2AMessage(
        type=MessageType.REGISTER,
        **{"from": agent_id},
        to="hub",
        payload={
            "name": name,
            "capabilities": capabilities,
            "metadata": metadata or {},
        },
    )


def make_deregister(agent_id: str) -> A2AMessage:
    """Create a DEREGISTER message."""
    return A2AMessage(
        type=MessageType.DEREGISTER,
        **{"from": agent_id},
        to="hub",
    )


def make_discover(agent_id: str, capability: str = "") -> A2AMessage:
    """Create a DISCOVER message to find agents by capability keyword."""
    return A2AMessage(
        type=MessageType.DISCOVER,
        **{"from": agent_id},
        to="hub",
        payload={"capability": capability},
    )


def make_delegate(
    from_agent: str,
    to_agent: str,
    capability: str,
    params: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> A2AMessage:
    """Create a DELEGATE message to assign a task to an agent."""
    return A2AMessage(
        type=MessageType.DELEGATE,
        **{"from": from_agent},
        to=to_agent,
        payload={
            "task_id": task_id or str(uuid.uuid4()),
            "capability": capability,
            "params": params or {},
        },
    )


def make_result(
    from_agent: str,
    to_agent: str,
    task_id: str,
    result: Any = None,
    error: str | None = None,
) -> A2AMessage:
    """Create a RESULT message returning a task outcome."""
    return A2AMessage(
        type=MessageType.RESULT,
        **{"from": from_agent},
        to=to_agent,
        payload={
            "task_id": task_id,
            "result": result,
            "error": error,
        },
    )


def make_error(
    to_agent: str,
    error: str,
    details: dict[str, Any] | None = None,
) -> A2AMessage:
    """Create an ERROR message from the hub."""
    return A2AMessage(
        type=MessageType.ERROR,
        **{"from": "hub"},
        to=to_agent,
        payload={"error": error, "details": details or {}},
    )


def make_broadcast(from_agent: str, message: str) -> A2AMessage:
    """Create a BROADCAST message to all agents."""
    return A2AMessage(
        type=MessageType.BROADCAST,
        **{"from": from_agent},
        to="broadcast",
        payload={"message": message},
    )
