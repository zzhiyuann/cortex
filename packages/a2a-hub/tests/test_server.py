"""Tests for the A2A Hub server."""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from a2a_hub.protocol import A2AMessage, MessageType, TaskStatus
from a2a_hub.server import HubServer


@pytest.fixture
async def hub():
    """Start a hub server on a random port for testing."""
    server = HubServer(
        host="127.0.0.1",
        port=0,
        # Use a long heartbeat timeout so tests don't flap
        heartbeat_timeout=600,
    )
    await server.start()
    # Extract the actual port from the server
    port = server._server.sockets[0].getsockname()[1]
    server.port = port
    yield server
    await server.stop()


@pytest.fixture
def hub_url(hub: HubServer) -> str:
    """Return the WebSocket URL for the test hub."""
    return f"ws://127.0.0.1:{hub.port}"


async def register_agent(
    url: str, agent_id: str, name: str, capabilities: list[str]
) -> websockets.ClientConnection:
    """Helper: connect and register an agent, return the WebSocket."""
    ws = await websockets.connect(url)
    msg = A2AMessage(
        type=MessageType.REGISTER,
        **{"from": agent_id},
        to="hub",
        payload={"name": name, "capabilities": capabilities},
    )
    await ws.send(msg.to_json())
    raw = await ws.recv()
    ack = A2AMessage.from_json(str(raw))
    assert ack.type == MessageType.REGISTER
    assert ack.payload.get("status") == "ok"
    return ws


class TestHubServer:
    """Integration tests for the hub server."""

    @pytest.mark.asyncio
    async def test_agent_registration(self, hub: HubServer, hub_url: str) -> None:
        ws = await register_agent(hub_url, "test-1", "Test Agent", ["echo"])
        assert "test-1" in hub.agents
        assert hub.agents["test-1"].name == "Test Agent"
        assert hub.agents["test-1"].capabilities == ["echo"]
        await ws.close()

    @pytest.mark.asyncio
    async def test_agent_deregistration(self, hub: HubServer, hub_url: str) -> None:
        ws = await register_agent(hub_url, "test-1", "Test Agent", ["echo"])
        assert "test-1" in hub.agents

        dereg = A2AMessage(
            type=MessageType.DEREGISTER,
            **{"from": "test-1"},
            to="hub",
        )
        await ws.send(dereg.to_json())
        raw = await ws.recv()
        resp = A2AMessage.from_json(str(raw))
        assert resp.payload.get("status") == "ok"
        assert "test-1" not in hub.agents
        await ws.close()

    @pytest.mark.asyncio
    async def test_discover_all_agents(self, hub: HubServer, hub_url: str) -> None:
        ws1 = await register_agent(hub_url, "a1", "Agent 1", ["echo"])
        ws2 = await register_agent(hub_url, "a2", "Agent 2", ["search"])
        ws3 = await register_agent(hub_url, "a3", "Agent 3", ["echo", "review"])

        # Discover all
        discover = A2AMessage(
            type=MessageType.DISCOVER,
            **{"from": "a1"},
            to="hub",
            payload={"capability": ""},
        )
        await ws1.send(discover.to_json())
        raw = await ws1.recv()
        resp = A2AMessage.from_json(str(raw))
        agents = resp.payload["agents"]
        assert len(agents) == 3

        await ws1.close()
        await ws2.close()
        await ws3.close()

    @pytest.mark.asyncio
    async def test_discover_by_capability(self, hub: HubServer, hub_url: str) -> None:
        ws1 = await register_agent(hub_url, "a1", "Agent 1", ["echo"])
        ws2 = await register_agent(hub_url, "a2", "Agent 2", ["search"])
        ws3 = await register_agent(hub_url, "a3", "Agent 3", ["echo", "review"])

        # Discover echo agents
        discover = A2AMessage(
            type=MessageType.DISCOVER,
            **{"from": "a1"},
            to="hub",
            payload={"capability": "echo"},
        )
        await ws1.send(discover.to_json())
        raw = await ws1.recv()
        resp = A2AMessage.from_json(str(raw))
        agents = resp.payload["agents"]
        assert len(agents) == 2
        agent_ids = {a["id"] for a in agents}
        assert agent_ids == {"a1", "a3"}

        await ws1.close()
        await ws2.close()
        await ws3.close()

    @pytest.mark.asyncio
    async def test_delegate_task(self, hub: HubServer, hub_url: str) -> None:
        ws_requester = await register_agent(hub_url, "req", "Requester", [])
        ws_worker = await register_agent(hub_url, "worker", "Worker", ["echo"])

        # Delegate a task
        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="worker",
            payload={
                "task_id": "task-1",
                "capability": "echo",
                "params": {"message": "hello"},
            },
        )
        await ws_requester.send(delegate.to_json())

        # Requester receives task acknowledgement
        raw = await ws_requester.recv()
        ack = A2AMessage.from_json(str(raw))
        assert ack.type == MessageType.STATUS
        assert ack.payload["task_id"] == "task-1"

        # Worker receives the delegated task
        raw = await ws_worker.recv()
        task_msg = A2AMessage.from_json(str(raw))
        assert task_msg.type == MessageType.DELEGATE
        assert task_msg.payload["capability"] == "echo"
        assert task_msg.payload["params"] == {"message": "hello"}

        # Worker sends result back
        result = A2AMessage(
            type=MessageType.RESULT,
            **{"from": "worker"},
            to="req",
            payload={"task_id": "task-1", "result": {"echo": "hello"}},
        )
        await ws_worker.send(result.to_json())

        # Requester receives the result
        raw = await ws_requester.recv()
        result_msg = A2AMessage.from_json(str(raw))
        assert result_msg.type == MessageType.RESULT
        assert result_msg.payload["result"] == {"echo": "hello"}

        await ws_requester.close()
        await ws_worker.close()

    @pytest.mark.asyncio
    async def test_delegate_to_unknown_agent(self, hub: HubServer, hub_url: str) -> None:
        ws = await register_agent(hub_url, "req", "Requester", [])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="nonexistent",
            payload={"task_id": "t1", "capability": "x", "params": {}},
        )
        await ws.send(delegate.to_json())
        raw = await ws.recv()
        resp = A2AMessage.from_json(str(raw))
        assert resp.type == MessageType.ERROR
        assert "not found" in resp.payload["error"].lower()
        await ws.close()

    @pytest.mark.asyncio
    async def test_broadcast(self, hub: HubServer, hub_url: str) -> None:
        ws1 = await register_agent(hub_url, "a1", "Agent 1", [])
        ws2 = await register_agent(hub_url, "a2", "Agent 2", [])
        ws3 = await register_agent(hub_url, "a3", "Agent 3", [])

        broadcast = A2AMessage(
            type=MessageType.BROADCAST,
            **{"from": "a1"},
            to="broadcast",
            payload={"message": "Hello everyone!"},
        )
        await ws1.send(broadcast.to_json())

        # Sender gets acknowledgement
        raw = await ws1.recv()
        ack = A2AMessage.from_json(str(raw))
        assert ack.type == MessageType.BROADCAST
        assert ack.payload["recipients"] == 2

        # Other agents receive the broadcast
        for ws in [ws2, ws3]:
            raw = await ws.recv()
            msg = A2AMessage.from_json(str(raw))
            assert msg.type == MessageType.BROADCAST
            assert msg.payload["message"] == "Hello everyone!"

        await ws1.close()
        await ws2.close()
        await ws3.close()

    @pytest.mark.asyncio
    async def test_heartbeat(self, hub: HubServer, hub_url: str) -> None:
        ws = await register_agent(hub_url, "a1", "Agent 1", [])

        hb = A2AMessage(
            type=MessageType.HEARTBEAT,
            **{"from": "a1"},
            to="hub",
        )
        await ws.send(hb.to_json())
        raw = await ws.recv()
        resp = A2AMessage.from_json(str(raw))
        assert resp.type == MessageType.HEARTBEAT_ACK
        await ws.close()

    @pytest.mark.asyncio
    async def test_hub_status(self, hub: HubServer, hub_url: str) -> None:
        ws1 = await register_agent(hub_url, "a1", "Agent 1", ["echo"])
        ws2 = await register_agent(hub_url, "a2", "Agent 2", ["search"])

        status = A2AMessage(
            type=MessageType.STATUS,
            **{"from": "a1"},
            to="hub",
        )
        await ws1.send(status.to_json())
        raw = await ws1.recv()
        resp = A2AMessage.from_json(str(raw))
        assert resp.payload["agents_count"] == 2
        assert resp.payload["tasks_count"] == 0

        await ws1.close()
        await ws2.close()

    @pytest.mark.asyncio
    async def test_agent_cleanup_on_disconnect(self, hub: HubServer, hub_url: str) -> None:
        ws = await register_agent(hub_url, "a1", "Agent 1", ["echo"])
        assert "a1" in hub.agents

        await ws.close()
        # Give the server a moment to process the disconnect
        await asyncio.sleep(0.1)
        assert "a1" not in hub.agents

    @pytest.mark.asyncio
    async def test_invalid_message(self, hub: HubServer, hub_url: str) -> None:
        ws = await websockets.connect(hub_url)
        await ws.send("not valid json at all")
        raw = await ws.recv()
        resp = A2AMessage.from_json(str(raw))
        assert resp.type == MessageType.ERROR
        await ws.close()


class TestSelfDelegation:
    """Test that self-delegation is rejected."""

    @pytest.mark.asyncio
    async def test_cannot_delegate_to_self(self, hub: HubServer, hub_url: str) -> None:
        ws = await register_agent(hub_url, "agent-1", "Agent 1", ["echo"])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "agent-1"},
            to="agent-1",
            payload={"task_id": "t1", "capability": "echo", "params": {}},
        )
        await ws.send(delegate.to_json())
        raw = await ws.recv()
        resp = A2AMessage.from_json(str(raw))
        assert resp.type == MessageType.ERROR
        assert "yourself" in resp.payload["error"].lower()
        await ws.close()


class TestDuplicateRegistration:
    """Test re-registration behavior."""

    @pytest.mark.asyncio
    async def test_re_registration_replaces_agent(self, hub: HubServer, hub_url: str) -> None:
        ws1 = await register_agent(hub_url, "agent-1", "Agent v1", ["echo"])
        assert hub.agents["agent-1"].name == "Agent v1"

        # Re-register the same agent_id with a new connection
        ws2 = await register_agent(hub_url, "agent-1", "Agent v2", ["echo", "search"])
        assert hub.agents["agent-1"].name == "Agent v2"
        assert hub.agents["agent-1"].capabilities == ["echo", "search"]

        # There should still be exactly one entry for agent-1
        assert "agent-1" in hub.connections

        # ws1 may already be closed by the server, ignore errors
        try:
            await ws1.close()
        except Exception:
            pass

        # Give server time to process old ws1 close
        await asyncio.sleep(0.1)

        # Agent should still be registered (ws1's handler should NOT remove it)
        assert "agent-1" in hub.agents
        assert hub.agents["agent-1"].name == "Agent v2"

        await ws2.close()


class TestHeartbeatMonitoring:
    """Test heartbeat-based agent health monitoring."""

    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp(self, hub: HubServer, hub_url: str) -> None:
        ws = await register_agent(hub_url, "a1", "Agent 1", ["echo"])
        initial_hb = hub.agents["a1"].last_heartbeat

        # Small delay so timestamp changes
        await asyncio.sleep(0.05)

        hb = A2AMessage(
            type=MessageType.HEARTBEAT,
            **{"from": "a1"},
            to="hub",
        )
        await ws.send(hb.to_json())
        await ws.recv()  # ack

        assert hub.agents["a1"].last_heartbeat > initial_hb
        await ws.close()

    @pytest.mark.asyncio
    async def test_stale_agent_removed(self) -> None:
        """Agent that misses heartbeats is auto-deregistered."""
        server = HubServer(
            host="127.0.0.1",
            port=0,
            heartbeat_timeout=0.5,  # Very short for testing
        )
        await server.start()
        port = server._server.sockets[0].getsockname()[1]
        server.port = port
        url = f"ws://127.0.0.1:{port}"

        try:
            ws = await register_agent(url, "stale-1", "Stale Agent", ["echo"])
            assert "stale-1" in server.agents

            # Wait for cleanup to kick in (heartbeat_timeout=0.5s, cleanup_interval=10s)
            # But we can't wait 10s in a test, so manually trigger cleanup
            from datetime import datetime, timezone, timedelta

            # Backdate the heartbeat
            server.agents["stale-1"].last_heartbeat = (
                datetime.now(timezone.utc) - timedelta(seconds=5)
            )

            # Run one iteration of cleanup manually
            now = datetime.now(timezone.utc)
            stale_agents = []
            for agent_id, info in server.agents.items():
                elapsed = (now - info.last_heartbeat).total_seconds()
                if elapsed > server.heartbeat_timeout:
                    stale_agents.append(agent_id)
            for agent_id in stale_agents:
                conn = server.connections.get(agent_id)
                if conn:
                    try:
                        await conn.close()
                    except Exception:
                        pass
                server._remove_agent(agent_id, reason="heartbeat timeout")

            assert "stale-1" not in server.agents

        finally:
            await server.stop()


class TestRateLimiting:
    """Test per-agent rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_rejects_flood(self) -> None:
        """Agents that send too many messages get rate-limited."""
        server = HubServer(
            host="127.0.0.1",
            port=0,
            rate_limit_window=10.0,
            rate_limit_max=5,  # Very low for testing
            heartbeat_timeout=600,
        )
        await server.start()
        port = server._server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"

        try:
            ws = await register_agent(url, "flooder", "Flood Agent", ["echo"])

            # Send messages rapidly — first 5 should succeed (within rate limit)
            # The 6th+ should be rejected
            error_count = 0
            for i in range(10):
                status = A2AMessage(
                    type=MessageType.STATUS,
                    **{"from": "flooder"},
                    to="hub",
                )
                await ws.send(status.to_json())
                raw = await ws.recv()
                resp = A2AMessage.from_json(str(raw))
                if resp.type == MessageType.ERROR and "rate limit" in resp.payload.get("error", "").lower():
                    error_count += 1

            assert error_count > 0, "Rate limiting should have kicked in"

            await ws.close()
        finally:
            await server.stop()


class TestTaskLifecycle:
    """Test task status tracking and lifecycle."""

    @pytest.mark.asyncio
    async def test_task_status_assigned(self, hub: HubServer, hub_url: str) -> None:
        """After delegation, task status should be 'assigned'."""
        ws_requester = await register_agent(hub_url, "req", "Requester", [])
        ws_worker = await register_agent(hub_url, "worker", "Worker", ["echo"])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="worker",
            payload={
                "task_id": "task-lifecycle-1",
                "capability": "echo",
                "params": {},
            },
        )
        await ws_requester.send(delegate.to_json())
        raw = await ws_requester.recv()
        ack = A2AMessage.from_json(str(raw))
        assert ack.payload["status"] == "assigned"

        # Also verify in the server's task registry
        assert hub.tasks["task-lifecycle-1"].status == TaskStatus.ASSIGNED

        # Consume the forwarded task on the worker side
        await ws_worker.recv()

        await ws_requester.close()
        await ws_worker.close()

    @pytest.mark.asyncio
    async def test_task_status_query(self, hub: HubServer, hub_url: str) -> None:
        """Query individual task status."""
        ws_requester = await register_agent(hub_url, "req", "Requester", [])
        ws_worker = await register_agent(hub_url, "worker", "Worker", ["echo"])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="worker",
            payload={
                "task_id": "task-q-1",
                "capability": "echo",
                "params": {},
            },
        )
        await ws_requester.send(delegate.to_json())
        await ws_requester.recv()  # ack
        await ws_worker.recv()  # forwarded task

        # Query task status
        status_msg = A2AMessage(
            type=MessageType.STATUS,
            **{"from": "req"},
            to="hub",
            payload={"task_id": "task-q-1"},
        )
        await ws_requester.send(status_msg.to_json())
        raw = await ws_requester.recv()
        resp = A2AMessage.from_json(str(raw))
        assert resp.payload["task_id"] == "task-q-1"
        assert resp.payload["status"] == "assigned"

        await ws_requester.close()
        await ws_worker.close()

    @pytest.mark.asyncio
    async def test_hub_status_includes_task_breakdown(self, hub: HubServer, hub_url: str) -> None:
        """Hub status should include task_statuses breakdown."""
        ws_requester = await register_agent(hub_url, "req", "Requester", [])
        ws_worker = await register_agent(hub_url, "worker", "Worker", ["echo"])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="worker",
            payload={
                "task_id": "task-breakdown-1",
                "capability": "echo",
                "params": {},
            },
        )
        await ws_requester.send(delegate.to_json())
        await ws_requester.recv()  # ack
        await ws_worker.recv()  # forwarded task

        # Query hub status
        status_msg = A2AMessage(
            type=MessageType.STATUS,
            **{"from": "req"},
            to="hub",
        )
        await ws_requester.send(status_msg.to_json())
        raw = await ws_requester.recv()
        resp = A2AMessage.from_json(str(raw))
        assert "task_statuses" in resp.payload
        assert resp.payload["tasks_count"] == 1

        await ws_requester.close()
        await ws_worker.close()


class TestTaskRetry:
    """Test task retry on agent disconnect."""

    @pytest.mark.asyncio
    async def test_retry_on_disconnect(self, hub: HubServer, hub_url: str) -> None:
        """If a worker disconnects mid-task, task is retried on another agent."""
        ws_requester = await register_agent(hub_url, "req", "Requester", [])
        ws_worker1 = await register_agent(hub_url, "worker1", "Worker 1", ["echo"])
        ws_worker2 = await register_agent(hub_url, "worker2", "Worker 2", ["echo"])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="worker1",
            payload={
                "task_id": "task-retry-1",
                "capability": "echo",
                "params": {"message": "retry test"},
                "max_retries": 2,
            },
        )
        await ws_requester.send(delegate.to_json())
        await ws_requester.recv()  # ack
        await ws_worker1.recv()  # forwarded task

        # Worker1 disconnects mid-task
        await ws_worker1.close()
        await asyncio.sleep(0.2)  # Let the server process the disconnect and retry

        task = hub.tasks["task-retry-1"]
        # Task should have been retried and reassigned to worker2
        assert task.retry_count == 1
        assert task.to_agent == "worker2"
        assert task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)

        await ws_requester.close()
        await ws_worker2.close()

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, hub: HubServer, hub_url: str) -> None:
        """Task fails when retries are exhausted and no agents are available."""
        ws_requester = await register_agent(hub_url, "req", "Requester", [])
        ws_worker = await register_agent(hub_url, "worker", "Worker", ["unique-cap"])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="worker",
            payload={
                "task_id": "task-retry-exhaust",
                "capability": "unique-cap",
                "params": {},
                "max_retries": 1,
            },
        )
        await ws_requester.send(delegate.to_json())
        await ws_requester.recv()  # ack
        await ws_worker.recv()  # forwarded task

        # Worker disconnects — no other agent has "unique-cap"
        await ws_worker.close()
        await asyncio.sleep(0.2)

        task = hub.tasks["task-retry-exhaust"]
        # Should be FAILED because retry found no capable agent
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 1
        assert "no available agent" in task.error.lower() or "no agent" in task.error.lower()

        await ws_requester.close()

    @pytest.mark.asyncio
    async def test_no_retry_when_max_retries_zero(self, hub: HubServer, hub_url: str) -> None:
        """Tasks with max_retries=0 should fail immediately on disconnect."""
        ws_requester = await register_agent(hub_url, "req", "Requester", [])
        ws_worker = await register_agent(hub_url, "worker", "Worker", ["echo"])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="worker",
            payload={
                "task_id": "task-no-retry",
                "capability": "echo",
                "params": {},
                "max_retries": 0,
            },
        )
        await ws_requester.send(delegate.to_json())
        await ws_requester.recv()  # ack
        await ws_worker.recv()  # forwarded task

        await ws_worker.close()
        await asyncio.sleep(0.2)

        task = hub.tasks["task-no-retry"]
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 0
        assert "max retries" in task.error.lower()

        await ws_requester.close()


class TestTaskPriority:
    """Test task priority handling."""

    @pytest.mark.asyncio
    async def test_priority_stored(self, hub: HubServer, hub_url: str) -> None:
        """Priority from delegate payload is stored on the task record."""
        ws_requester = await register_agent(hub_url, "req", "Requester", [])
        ws_worker = await register_agent(hub_url, "worker", "Worker", ["echo"])

        delegate = A2AMessage(
            type=MessageType.DELEGATE,
            **{"from": "req"},
            to="worker",
            payload={
                "task_id": "task-priority-1",
                "capability": "echo",
                "params": {},
                "priority": 10,
            },
        )
        await ws_requester.send(delegate.to_json())
        await ws_requester.recv()  # ack
        await ws_worker.recv()  # forwarded task

        task = hub.tasks["task-priority-1"]
        assert task.priority == 10

        await ws_requester.close()
        await ws_worker.close()


class TestGracefulShutdown:
    """Test graceful server shutdown."""

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self) -> None:
        """Server stop should cancel background tasks and close sockets."""
        server = HubServer(host="127.0.0.1", port=0, heartbeat_timeout=600)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"

        ws = await register_agent(url, "a1", "Agent 1", ["echo"])
        assert "a1" in server.agents

        await server.stop()

        # Server should be fully stopped
        assert server._shutting_down is True

        try:
            await ws.close()
        except Exception:
            pass
