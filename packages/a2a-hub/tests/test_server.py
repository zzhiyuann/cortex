"""Tests for the A2A Hub server."""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from a2a_hub.protocol import A2AMessage, MessageType
from a2a_hub.server import HubServer


@pytest.fixture
async def hub():
    """Start a hub server on a random port for testing."""
    server = HubServer(host="127.0.0.1", port=0)
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
