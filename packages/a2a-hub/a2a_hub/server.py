"""A2A Hub Server — WebSocket-based central hub for agent communication.

The hub maintains an agent registry, routes messages between agents,
and manages delegated task lifecycle including timeouts.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

from a2a_hub.protocol import (
    A2AMessage,
    AgentInfo,
    MessageType,
    TaskRecord,
    TaskStatus,
    make_error,
)
from a2a_hub.utils import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TASK_TTL,
    HEARTBEAT_INTERVAL,
    get_logger,
)

logger = get_logger("a2a-hub.server")


class HubServer:
    """Central hub that manages agent registration, message routing, and tasks.

    Attributes:
        host: Bind address for the WebSocket server.
        port: Bind port for the WebSocket server.
        agents: Registry mapping agent_id -> AgentInfo.
        connections: Mapping agent_id -> WebSocket connection.
        tasks: Mapping task_id -> TaskRecord.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        task_ttl: int = DEFAULT_TASK_TTL,
    ) -> None:
        self.host = host
        self.port = port
        self.task_ttl = task_ttl

        self.agents: dict[str, AgentInfo] = {}
        self.connections: dict[str, ServerConnection] = {}
        self.tasks: dict[str, TaskRecord] = {}
        self._server: Any = None
        self._cleanup_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the WebSocket server and background tasks."""
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
        )
        self._cleanup_task = asyncio.create_task(self._task_cleanup_loop())
        logger.info("Hub server listening on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Hub server stopped")

    async def run_forever(self) -> None:
        """Start the server and block until interrupted."""
        await self.start()
        try:
            await asyncio.Future()  # block forever
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handler(self, ws: ServerConnection) -> None:
        """Handle a single WebSocket connection."""
        agent_id: str | None = None
        try:
            async for raw in ws:
                try:
                    msg = A2AMessage.from_json(str(raw))
                except Exception as exc:
                    logger.warning("Invalid message: %s", exc)
                    await ws.send(
                        make_error("unknown", f"Invalid message: {exc}").to_json()
                    )
                    continue

                agent_id = msg.from_agent
                response = await self._dispatch(msg, ws)
                if response:
                    await ws.send(response.to_json())

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed for agent=%s", agent_id)
        finally:
            if agent_id and agent_id in self.agents:
                self._remove_agent(agent_id)

    async def _dispatch(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage | None:
        """Route a message to the appropriate handler."""
        handlers = {
            MessageType.REGISTER: self._handle_register,
            MessageType.DEREGISTER: self._handle_deregister,
            MessageType.DISCOVER: self._handle_discover,
            MessageType.DELEGATE: self._handle_delegate,
            MessageType.RESULT: self._handle_result,
            MessageType.STATUS: self._handle_status,
            MessageType.BROADCAST: self._handle_broadcast,
            MessageType.HEARTBEAT: self._handle_heartbeat,
        }
        handler = handlers.get(msg.type)
        if handler is None:
            return make_error(msg.from_agent, f"Unknown message type: {msg.type}")
        return await handler(msg, ws)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_register(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage:
        agent_id = msg.from_agent
        payload = msg.payload

        info = AgentInfo(
            id=agent_id,
            name=payload.get("name", agent_id),
            capabilities=payload.get("capabilities", []),
            status="online",
            metadata=payload.get("metadata", {}),
        )
        self.agents[agent_id] = info
        self.connections[agent_id] = ws

        logger.info(
            "Agent registered: %s (%s) capabilities=%s",
            info.name,
            agent_id,
            info.capabilities,
        )

        return A2AMessage(
            type=MessageType.REGISTER,
            **{"from": "hub"},
            to=agent_id,
            payload={"status": "ok", "agent_id": agent_id},
        )

    async def _handle_deregister(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage:
        agent_id = msg.from_agent
        self._remove_agent(agent_id)

        return A2AMessage(
            type=MessageType.DEREGISTER,
            **{"from": "hub"},
            to=agent_id,
            payload={"status": "ok"},
        )

    async def _handle_discover(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage:
        keyword = msg.payload.get("capability", "").lower()
        matches = []
        for info in self.agents.values():
            if not keyword:
                matches.append(info.model_dump())
            elif any(keyword in cap.lower() for cap in info.capabilities):
                matches.append(info.model_dump())
            elif keyword in info.name.lower():
                matches.append(info.model_dump())

        return A2AMessage(
            type=MessageType.DISCOVER,
            **{"from": "hub"},
            to=msg.from_agent,
            payload={"agents": matches},
        )

    async def _handle_delegate(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage | None:
        to_agent = msg.to
        payload = msg.payload
        task_id = payload.get("task_id", str(uuid.uuid4()))

        if to_agent not in self.agents:
            return make_error(
                msg.from_agent, f"Agent '{to_agent}' not found"
            )

        task = TaskRecord(
            task_id=task_id,
            from_agent=msg.from_agent,
            to_agent=to_agent,
            capability=payload.get("capability", ""),
            params=payload.get("params", {}),
            status=TaskStatus.PENDING,
            ttl_seconds=payload.get("ttl", self.task_ttl),
        )
        self.tasks[task_id] = task

        # Forward the delegate message to the target agent
        target_ws = self.connections.get(to_agent)
        if target_ws:
            forward = A2AMessage(
                type=MessageType.DELEGATE,
                id=msg.id,
                **{"from": msg.from_agent},
                to=to_agent,
                payload=payload,
            )
            try:
                await target_ws.send(forward.to_json())
                task.status = TaskStatus.IN_PROGRESS
            except Exception as exc:
                logger.error("Failed to forward task to %s: %s", to_agent, exc)
                task.status = TaskStatus.FAILED
                task.error = str(exc)

        # Acknowledge task creation to the sender
        return A2AMessage(
            type=MessageType.STATUS,
            **{"from": "hub"},
            to=msg.from_agent,
            payload={
                "task_id": task_id,
                "status": task.status.value,
            },
        )

    async def _handle_result(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage | None:
        payload = msg.payload
        task_id = payload.get("task_id")
        if not task_id or task_id not in self.tasks:
            return make_error(msg.from_agent, f"Unknown task: {task_id}")

        task = self.tasks[task_id]
        task.result = payload.get("result")
        task.error = payload.get("error")
        task.status = TaskStatus.FAILED if task.error else TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)

        # Forward result to the original requester
        requester_ws = self.connections.get(task.from_agent)
        if requester_ws:
            forward = A2AMessage(
                type=MessageType.RESULT,
                **{"from": msg.from_agent},
                to=task.from_agent,
                payload=payload,
            )
            try:
                await requester_ws.send(forward.to_json())
            except Exception as exc:
                logger.error(
                    "Failed to forward result to %s: %s", task.from_agent, exc
                )

        return None  # No direct reply needed

    async def _handle_status(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage:
        task_id = msg.payload.get("task_id")
        if task_id and task_id in self.tasks:
            task = self.tasks[task_id]
            return A2AMessage(
                type=MessageType.STATUS,
                **{"from": "hub"},
                to=msg.from_agent,
                payload={
                    "task_id": task_id,
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                },
            )

        # General hub status
        return A2AMessage(
            type=MessageType.STATUS,
            **{"from": "hub"},
            to=msg.from_agent,
            payload={
                "agents_count": len(self.agents),
                "tasks_count": len(self.tasks),
                "agents": [a.model_dump() for a in self.agents.values()],
            },
        )

    async def _handle_broadcast(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage | None:
        sender = msg.from_agent
        for agent_id, conn in self.connections.items():
            if agent_id == sender:
                continue
            try:
                await conn.send(msg.to_json())
            except Exception as exc:
                logger.warning("Failed to broadcast to %s: %s", agent_id, exc)

        return A2AMessage(
            type=MessageType.BROADCAST,
            **{"from": "hub"},
            to=sender,
            payload={"status": "ok", "recipients": len(self.connections) - 1},
        )

    async def _handle_heartbeat(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage:
        return A2AMessage(
            type=MessageType.HEARTBEAT_ACK,
            **{"from": "hub"},
            to=msg.from_agent,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        self.agents.pop(agent_id, None)
        self.connections.pop(agent_id, None)
        logger.info("Agent removed: %s", agent_id)

    async def _task_cleanup_loop(self) -> None:
        """Periodically expire timed-out tasks."""
        while True:
            await asyncio.sleep(30)
            now = datetime.now(timezone.utc)
            for task in list(self.tasks.values()):
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT):
                    continue
                elapsed = (now - task.created_at).total_seconds()
                if elapsed > task.ttl_seconds:
                    task.status = TaskStatus.TIMEOUT
                    task.completed_at = now
                    logger.info("Task %s timed out after %ds", task.task_id, elapsed)


async def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Convenience function to run the hub server."""
    server = HubServer(host=host, port=port)
    await server.run_forever()


if __name__ == "__main__":
    asyncio.run(run_server())
