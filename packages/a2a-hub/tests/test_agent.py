"""Tests for the A2A Hub agent SDK."""

from __future__ import annotations

import asyncio

import pytest

from a2a_hub.agent import Agent, capability
from a2a_hub.protocol import A2AMessage, MessageType
from a2a_hub.server import HubServer


@pytest.fixture
async def hub():
    """Start a hub server on a random port for testing."""
    server = HubServer(host="127.0.0.1", port=0)
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    server.port = port
    yield server
    await server.stop()


class TestAgentCreation:
    """Test agent instantiation and configuration."""

    def test_create_agent_functional(self) -> None:
        agent = Agent("test-agent", capabilities=["echo", "search"])
        assert agent.name == "test-agent"
        assert "echo" in agent.all_capabilities
        assert "search" in agent.all_capabilities
        assert agent.agent_id.startswith("test-agent-")

    def test_create_agent_with_id(self) -> None:
        agent = Agent("test", agent_id="custom-id")
        assert agent.agent_id == "custom-id"

    def test_agent_requires_name(self) -> None:
        with pytest.raises(ValueError, match="must have a name"):
            Agent("")

    def test_on_task_decorator(self) -> None:
        agent = Agent("test")

        @agent.on_task("greet")
        async def handle(name: str) -> str:
            return f"Hello, {name}!"

        assert "greet" in agent.all_capabilities
        assert "greet" in agent._handlers

    def test_class_based_agent(self) -> None:
        class MyAgent(Agent):
            name = "my-agent"

            @capability("review", description="Review code")
            async def review(self, code: str) -> dict:
                return {"ok": True}

        agent = MyAgent()
        assert agent.name == "my-agent"
        assert "review" in agent.all_capabilities
        assert "review" in agent._handlers


class TestAgentConnection:
    """Integration tests for agent connecting to the hub."""

    @pytest.mark.asyncio
    async def test_agent_connect_and_register(self, hub: HubServer) -> None:
        agent = Agent(
            "test-agent",
            capabilities=["echo"],
            hub_host="127.0.0.1",
            hub_port=hub.port,
        )

        await agent.connect()
        assert agent.agent_id in hub.agents
        assert hub.agents[agent.agent_id].name == "test-agent"
        await agent.disconnect()

    @pytest.mark.asyncio
    async def test_agent_disconnect(self, hub: HubServer) -> None:
        agent = Agent(
            "test-agent",
            capabilities=["echo"],
            hub_host="127.0.0.1",
            hub_port=hub.port,
        )

        await agent.connect()
        agent_id = agent.agent_id
        assert agent_id in hub.agents

        await agent.disconnect()
        # Give server time to process
        await asyncio.sleep(0.1)
        assert agent_id not in hub.agents

    @pytest.mark.asyncio
    async def test_agent_handles_task(self, hub: HubServer) -> None:
        """Test that an agent receives and processes a delegated task."""
        agent = Agent(
            "worker",
            capabilities=["echo"],
            hub_host="127.0.0.1",
            hub_port=hub.port,
        )

        @agent.on_task("echo")
        async def handle_echo(message: str = "") -> dict:
            return {"echo": message}

        await agent.connect()

        # Simulate a task delegation from the hub side
        import websockets

        url = f"ws://127.0.0.1:{hub.port}"
        async with websockets.connect(url) as ws:
            # Register as requester
            reg = A2AMessage(
                type=MessageType.REGISTER,
                **{"from": "requester"},
                to="hub",
                payload={"name": "Requester", "capabilities": []},
            )
            await ws.send(reg.to_json())
            await ws.recv()  # ack

            # Delegate task to worker
            delegate = A2AMessage(
                type=MessageType.DELEGATE,
                **{"from": "requester"},
                to=agent.agent_id,
                payload={
                    "task_id": "test-task-1",
                    "capability": "echo",
                    "params": {"message": "hello world"},
                },
            )
            await ws.send(delegate.to_json())

            # Read status ack
            raw = await ws.recv()
            ack = A2AMessage.from_json(str(raw))
            assert ack.type == MessageType.STATUS

            # The agent needs to process the incoming task in its listen loop.
            # Since we called connect() directly (not _listen), we need to
            # manually receive and handle the message.
            raw = await agent._ws.recv()
            msg = A2AMessage.from_json(str(raw))
            await agent._handle_message(msg)

            # Now read the result that the agent sent back
            raw = await ws.recv()
            result = A2AMessage.from_json(str(raw))
            assert result.type == MessageType.RESULT
            assert result.payload["result"] == {"echo": "hello world"}

        await agent.disconnect()

    @pytest.mark.asyncio
    async def test_agent_hub_url(self, hub: HubServer) -> None:
        agent = Agent(
            "test",
            hub_host="127.0.0.1",
            hub_port=hub.port,
        )
        assert agent.hub_url == f"ws://127.0.0.1:{hub.port}"
