"""Tests for the A2A Hub protocol module."""

from __future__ import annotations

import json

import pytest

from a2a_hub.protocol import (
    A2AMessage,
    AgentInfo,
    MessageType,
    TaskRecord,
    TaskStatus,
    make_broadcast,
    make_delegate,
    make_deregister,
    make_discover,
    make_error,
    make_register,
    make_result,
)


class TestMessageType:
    """Test MessageType enum values."""

    def test_all_types_exist(self) -> None:
        expected = {
            "register", "deregister", "discover", "delegate",
            "result", "status", "broadcast", "error",
            "heartbeat", "heartbeat_ack",
        }
        actual = {mt.value for mt in MessageType}
        assert actual == expected

    def test_string_comparison(self) -> None:
        assert MessageType.REGISTER == "register"
        assert MessageType.DELEGATE == "delegate"


class TestA2AMessage:
    """Test A2AMessage creation and serialization."""

    def test_create_message(self) -> None:
        msg = A2AMessage(
            type=MessageType.REGISTER,
            **{"from": "agent-1"},
            to="hub",
            payload={"name": "Test Agent"},
        )
        assert msg.type == MessageType.REGISTER
        assert msg.from_agent == "agent-1"
        assert msg.to == "hub"
        assert msg.payload["name"] == "Test Agent"
        assert msg.id  # should have auto-generated UUID
        assert msg.timestamp  # should have auto-generated timestamp

    def test_serialize_deserialize(self) -> None:
        msg = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "agent-1"},
            to="agent-2",
            payload={"task_id": "t1", "capability": "echo", "params": {"msg": "hi"}},
        )
        json_str = msg.to_json()
        parsed = json.loads(json_str)

        # Should use "from" alias, not "from_agent"
        assert "from" in parsed
        assert "from_agent" not in parsed
        assert parsed["from"] == "agent-1"

        # Round-trip
        msg2 = A2AMessage.from_json(json_str)
        assert msg2.type == msg.type
        assert msg2.from_agent == msg.from_agent
        assert msg2.to == msg.to
        assert msg2.payload == msg.payload

    def test_default_values(self) -> None:
        msg = A2AMessage(type=MessageType.HEARTBEAT)
        assert msg.from_agent == "unknown"
        assert msg.to == "hub"
        assert msg.payload == {}

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(Exception):
            A2AMessage(type="not-a-type")  # type: ignore[arg-type]


class TestAgentInfo:
    """Test AgentInfo model."""

    def test_create_agent_info(self) -> None:
        info = AgentInfo(
            id="agent-1",
            name="Test Agent",
            capabilities=["echo", "code-review"],
        )
        assert info.id == "agent-1"
        assert info.name == "Test Agent"
        assert info.capabilities == ["echo", "code-review"]
        assert info.status == "online"
        assert info.metadata == {}

    def test_serialize(self) -> None:
        info = AgentInfo(id="a1", name="A1", capabilities=["x"])
        data = info.model_dump()
        assert data["id"] == "a1"
        assert data["capabilities"] == ["x"]


class TestTaskRecord:
    """Test TaskRecord model."""

    def test_create_task(self) -> None:
        task = TaskRecord(
            from_agent="requester",
            to_agent="worker",
            capability="code-review",
            params={"code": "print('hi')"},
        )
        assert task.task_id  # auto-generated
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.ttl_seconds == 300

    def test_task_status_transitions(self) -> None:
        task = TaskRecord(
            from_agent="a", to_agent="b", capability="x"
        )
        assert task.status == TaskStatus.PENDING

        task.status = TaskStatus.IN_PROGRESS
        assert task.status == TaskStatus.IN_PROGRESS

        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED


class TestHelperConstructors:
    """Test message helper constructors."""

    def test_make_register(self) -> None:
        msg = make_register("agent-1", "Test Agent", ["echo", "search"])
        assert msg.type == MessageType.REGISTER
        assert msg.from_agent == "agent-1"
        assert msg.to == "hub"
        assert msg.payload["name"] == "Test Agent"
        assert msg.payload["capabilities"] == ["echo", "search"]

    def test_make_deregister(self) -> None:
        msg = make_deregister("agent-1")
        assert msg.type == MessageType.DEREGISTER
        assert msg.from_agent == "agent-1"

    def test_make_discover(self) -> None:
        msg = make_discover("agent-1", "review")
        assert msg.type == MessageType.DISCOVER
        assert msg.payload["capability"] == "review"

    def test_make_delegate(self) -> None:
        msg = make_delegate(
            "agent-1", "agent-2", "echo", {"message": "hello"}, "task-123"
        )
        assert msg.type == MessageType.DELEGATE
        assert msg.from_agent == "agent-1"
        assert msg.to == "agent-2"
        assert msg.payload["task_id"] == "task-123"
        assert msg.payload["capability"] == "echo"
        assert msg.payload["params"] == {"message": "hello"}

    def test_make_result(self) -> None:
        msg = make_result("agent-2", "agent-1", "task-123", result={"data": 42})
        assert msg.type == MessageType.RESULT
        assert msg.payload["task_id"] == "task-123"
        assert msg.payload["result"] == {"data": 42}
        assert msg.payload["error"] is None

    def test_make_result_with_error(self) -> None:
        msg = make_result("agent-2", "agent-1", "task-123", error="Something broke")
        assert msg.payload["error"] == "Something broke"

    def test_make_error(self) -> None:
        msg = make_error("agent-1", "Not found")
        assert msg.type == MessageType.ERROR
        assert msg.from_agent == "hub"
        assert msg.payload["error"] == "Not found"

    def test_make_broadcast(self) -> None:
        msg = make_broadcast("agent-1", "Hello everyone!")
        assert msg.type == MessageType.BROADCAST
        assert msg.to == "broadcast"
        assert msg.payload["message"] == "Hello everyone!"
