"""A2A Hub MCP Bridge — Expose the hub to Claude Code via MCP tools.

This module provides an MCP server (using FastMCP) that connects to the
A2A Hub as a special "bridge" agent. Claude Code can then discover agents,
delegate tasks, and collect results through standard MCP tool calls.

Usage:
    python -m a2a_hub.mcp_bridge          # Start MCP server (stdio)
    a2a-hub bridge                        # Via CLI
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import websockets
from fastmcp import FastMCP

from a2a_hub.protocol import A2AMessage, MessageType
from a2a_hub.utils import DEFAULT_HOST, DEFAULT_PORT, get_logger

logger = get_logger("a2a-hub.bridge")

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="A2A Hub Bridge",
    instructions=(
        "Bridge to the A2A Hub — discover agents, delegate tasks, "
        "and collect results from a network of AI agents."
    ),
    version="0.1.0",
)


class HubConnection:
    """Manages a persistent WebSocket connection to the hub."""

    def __init__(self) -> None:
        self._ws: Any = None
        self._agent_id = f"mcp-bridge-{uuid.uuid4().hex[:8]}"
        self._pending_responses: dict[str, asyncio.Future[A2AMessage]] = {}
        self._task_results: dict[str, dict[str, Any]] = {}
        self._listen_task: asyncio.Task[None] | None = None
        self.host = DEFAULT_HOST
        self.port = DEFAULT_PORT

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._ws.state.name == "OPEN"

    async def connect(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
        """Connect to the hub and register as a bridge agent."""
        self.host = host
        self.port = port
        url = f"ws://{host}:{port}"

        try:
            self._ws = await websockets.connect(url)
        except Exception as exc:
            return f"Failed to connect to hub at {url}: {exc}"

        # Register
        reg = A2AMessage(
            type=MessageType.REGISTER,
            **{"from": self._agent_id},
            to="hub",
            payload={
                "name": "MCP Bridge",
                "capabilities": ["bridge"],
                "metadata": {"type": "mcp-bridge"},
            },
        )
        await self._ws.send(reg.to_json())
        raw = await self._ws.recv()
        ack = A2AMessage.from_json(str(raw))

        # Start background listener
        self._listen_task = asyncio.create_task(self._listener())

        logger.info("Connected to hub at %s as %s", url, self._agent_id)
        return f"Connected to hub at {url}"

    async def ensure_connected(self) -> str | None:
        """Ensure we are connected; return error string if not."""
        if not self.connected:
            result = await self.connect(self.host, self.port)
            if "Failed" in result:
                return result
        return None

    async def _listener(self) -> None:
        """Background task that receives hub messages."""
        try:
            async for raw in self._ws:
                try:
                    msg = A2AMessage.from_json(str(raw))
                    await self._handle_message(msg)
                except Exception as exc:
                    logger.warning("Error processing message: %s", exc)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Hub connection closed")
        except Exception as exc:
            logger.error("Listener error: %s", exc)

    async def _handle_message(self, msg: A2AMessage) -> None:
        """Process messages from the hub."""
        # Resolve pending futures for request-response patterns
        if msg.id in self._pending_responses:
            self._pending_responses[msg.id].set_result(msg)
            return

        # Store task results
        if msg.type == MessageType.RESULT:
            task_id = msg.payload.get("task_id", "")
            self._task_results[task_id] = msg.payload

            # Also resolve any future waiting on this task
            if task_id in self._pending_responses:
                self._pending_responses[task_id].set_result(msg)

    async def send_and_wait(
        self, msg: A2AMessage, timeout: float = 10.0
    ) -> A2AMessage:
        """Send a message and wait for the response."""
        future: asyncio.Future[A2AMessage] = asyncio.get_event_loop().create_future()
        self._pending_responses[msg.id] = future

        await self._ws.send(msg.to_json())

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_responses.pop(msg.id, None)

        return result

    async def close(self) -> None:
        """Close the connection."""
        if self._listen_task:
            self._listen_task.cancel()
        if self._ws:
            await self._ws.close()


# Global connection instance
_hub = HubConnection()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool
async def hub_status(
    hub_host: str = "localhost",
    hub_port: int = 8765,
) -> str:
    """Show hub connection status and registered agents count.

    Args:
        hub_host: Hub server host (default: localhost).
        hub_port: Hub server port (default: 8765).
    """
    _hub.host = hub_host
    _hub.port = hub_port
    err = await _hub.ensure_connected()
    if err:
        return err

    msg = A2AMessage(
        type=MessageType.STATUS,
        **{"from": _hub._agent_id},
        to="hub",
    )
    resp = await _hub.send_and_wait(msg)
    payload = resp.payload

    agents_count = payload.get("agents_count", 0)
    tasks_count = payload.get("tasks_count", 0)
    agents = payload.get("agents", [])

    lines = [
        f"Hub Status: Connected",
        f"Agents: {agents_count}",
        f"Tasks: {tasks_count}",
        "",
        "Registered Agents:",
    ]
    for a in agents:
        caps = ", ".join(a.get("capabilities", []))
        lines.append(f"  - {a['name']} ({a['id']}) [{caps}]")

    return "\n".join(lines)


@mcp.tool
async def discover_agents(
    capability: str = "",
    hub_host: str = "localhost",
    hub_port: int = 8765,
) -> str:
    """Find agents by capability keyword.

    Args:
        capability: Capability keyword to search for (empty = all agents).
        hub_host: Hub server host.
        hub_port: Hub server port.
    """
    _hub.host = hub_host
    _hub.port = hub_port
    err = await _hub.ensure_connected()
    if err:
        return err

    msg = A2AMessage(
        type=MessageType.DISCOVER,
        **{"from": _hub._agent_id},
        to="hub",
        payload={"capability": capability},
    )
    resp = await _hub.send_and_wait(msg)
    agents = resp.payload.get("agents", [])

    if not agents:
        return f"No agents found matching '{capability}'"

    lines = [f"Found {len(agents)} agent(s):"]
    for a in agents:
        caps = ", ".join(a.get("capabilities", []))
        lines.append(
            f"  - {a['name']} (id: {a['id']}, status: {a.get('status', '?')}) "
            f"[{caps}]"
        )
    return "\n".join(lines)


@mcp.tool
async def delegate_task(
    agent_id: str,
    capability: str,
    params: str = "{}",
    hub_host: str = "localhost",
    hub_port: int = 8765,
) -> str:
    """Send a task to a specific agent.

    Args:
        agent_id: Target agent ID.
        capability: Which capability to invoke on the agent.
        params: JSON string of parameters for the task.
        hub_host: Hub server host.
        hub_port: Hub server port.
    """
    _hub.host = hub_host
    _hub.port = hub_port
    err = await _hub.ensure_connected()
    if err:
        return err

    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON params: {exc}"

    task_id = str(uuid.uuid4())

    msg = A2AMessage(
        type=MessageType.DELEGATE,
        **{"from": _hub._agent_id},
        to=agent_id,
        payload={
            "task_id": task_id,
            "capability": capability,
            "params": params_dict,
        },
    )
    resp = await _hub.send_and_wait(msg)

    status = resp.payload.get("status", "unknown")
    return f"Task delegated. task_id={task_id}, status={status}"


@mcp.tool
async def get_task_result(
    task_id: str,
    timeout: float = 30.0,
    hub_host: str = "localhost",
    hub_port: int = 8765,
) -> str:
    """Get the result for a delegated task.

    Waits up to `timeout` seconds for the result to arrive.

    Args:
        task_id: The task ID returned by delegate_task.
        timeout: Max seconds to wait for the result.
        hub_host: Hub server host.
        hub_port: Hub server port.
    """
    _hub.host = hub_host
    _hub.port = hub_port
    err = await _hub.ensure_connected()
    if err:
        return err

    # Check if result already arrived
    if task_id in _hub._task_results:
        payload = _hub._task_results[task_id]
        if payload.get("error"):
            return f"Task {task_id} FAILED: {payload['error']}"
        return f"Task {task_id} COMPLETED:\n{json.dumps(payload.get('result'), indent=2)}"

    # Wait for result
    future: asyncio.Future[A2AMessage] = asyncio.get_event_loop().create_future()
    _hub._pending_responses[task_id] = future

    try:
        result_msg = await asyncio.wait_for(future, timeout=timeout)
        payload = result_msg.payload
        if payload.get("error"):
            return f"Task {task_id} FAILED: {payload['error']}"
        return f"Task {task_id} COMPLETED:\n{json.dumps(payload.get('result'), indent=2)}"
    except asyncio.TimeoutError:
        return f"Task {task_id} still PENDING after {timeout}s"
    finally:
        _hub._pending_responses.pop(task_id, None)


@mcp.tool
async def broadcast_message(
    message: str,
    hub_host: str = "localhost",
    hub_port: int = 8765,
) -> str:
    """Send a broadcast message to all connected agents.

    Args:
        message: The message to broadcast.
        hub_host: Hub server host.
        hub_port: Hub server port.
    """
    _hub.host = hub_host
    _hub.port = hub_port
    err = await _hub.ensure_connected()
    if err:
        return err

    msg = A2AMessage(
        type=MessageType.BROADCAST,
        **{"from": _hub._agent_id},
        to="broadcast",
        payload={"message": message},
    )
    resp = await _hub.send_and_wait(msg)
    recipients = resp.payload.get("recipients", 0)
    return f"Broadcast sent to {recipients} agent(s)"


@mcp.tool
async def list_tasks(
    hub_host: str = "localhost",
    hub_port: int = 8765,
) -> str:
    """List all tasks tracked by this bridge session.

    Args:
        hub_host: Hub server host.
        hub_port: Hub server port.
    """
    _hub.host = hub_host
    _hub.port = hub_port

    if not _hub._task_results:
        return "No tasks recorded in this session"

    lines = [f"Tasks ({len(_hub._task_results)}):"]
    for task_id, payload in _hub._task_results.items():
        error = payload.get("error")
        status = "FAILED" if error else "COMPLETED"
        lines.append(f"  - {task_id}: {status}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_bridge() -> None:
    """Run the MCP bridge server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    run_bridge()
