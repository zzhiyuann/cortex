"""A2A Hub Server — WebSocket-based central hub for agent communication.

The hub maintains an agent registry, routes messages between agents,
and manages delegated task lifecycle including timeouts, retries, and
priority ordering. Includes rate limiting, heartbeat monitoring, and
graceful shutdown.
"""

from __future__ import annotations

import asyncio
import signal
import time
import uuid
from collections import defaultdict, deque
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
    CLEANUP_LOOP_INTERVAL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TASK_TTL,
    HEARTBEAT_TIMEOUT,
    MAX_TASK_RETRIES,
    RATE_LIMIT_MAX_MESSAGES,
    RATE_LIMIT_WINDOW,
    get_logger,
)

logger = get_logger("a2a-hub.server")


class HubServer:
    """Central hub that manages agent registration, message routing, and tasks.

    Features:
        - Agent registration with heartbeat-based health monitoring
        - Task delegation with priority queue, TTL, and retry on disconnect
        - Per-agent rate limiting (sliding window)
        - Graceful shutdown with signal handling
        - Duplicate agent name / self-delegation guards

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
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
        rate_limit_window: float = RATE_LIMIT_WINDOW,
        rate_limit_max: int = RATE_LIMIT_MAX_MESSAGES,
        max_task_retries: int = MAX_TASK_RETRIES,
    ) -> None:
        self.host = host
        self.port = port
        self.task_ttl = task_ttl
        self.heartbeat_timeout = heartbeat_timeout
        self.rate_limit_window = rate_limit_window
        self.rate_limit_max = rate_limit_max
        self.max_task_retries = max_task_retries

        self.agents: dict[str, AgentInfo] = {}
        self.connections: dict[str, ServerConnection] = {}
        self.tasks: dict[str, TaskRecord] = {}
        self._server: Any = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._shutting_down = False

        # Rate-limiting: agent_id -> deque of message timestamps
        self._message_timestamps: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=rate_limit_max * 2)
        )

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
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Hub server listening on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Gracefully shut down the server.

        Notifies all connected agents, cancels background tasks, and
        closes the WebSocket server.
        """
        self._shutting_down = True

        # Notify all connected agents that the hub is shutting down
        for agent_id, conn in list(self.connections.items()):
            try:
                shutdown_msg = A2AMessage(
                    type=MessageType.ERROR,
                    **{"from": "hub"},
                    to=agent_id,
                    payload={"error": "Hub is shutting down"},
                )
                await conn.send(shutdown_msg.to_json())
            except Exception:
                pass

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
        """Start the server and block until interrupted.

        Installs signal handlers for SIGINT and SIGTERM for graceful shutdown.
        """
        await self.start()

        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logger.info("Shutdown signal received")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Signal handlers are not supported on Windows in some cases
                pass

        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, agent_id: str) -> bool:
        """Check if an agent has exceeded its message rate limit.

        Returns True if the message should be REJECTED (rate limit exceeded).
        """
        now = time.monotonic()
        timestamps = self._message_timestamps[agent_id]

        # Remove timestamps outside the window
        while timestamps and timestamps[0] < now - self.rate_limit_window:
            timestamps.popleft()

        if len(timestamps) >= self.rate_limit_max:
            return True  # rate limit exceeded

        timestamps.append(now)
        return False

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handler(self, ws: ServerConnection) -> None:
        """Handle a single WebSocket connection.

        Manages the full lifecycle: receive messages, dispatch them,
        and clean up on disconnect.
        """
        agent_id: str | None = None
        try:
            async for raw in ws:
                if self._shutting_down:
                    break

                try:
                    msg = A2AMessage.from_json(str(raw))
                except Exception as exc:
                    logger.warning("Invalid message from %s: %s", agent_id or "unknown", exc)
                    await ws.send(
                        make_error("unknown", f"Invalid message: {exc}").to_json()
                    )
                    continue

                agent_id = msg.from_agent

                # Rate limiting (skip for registration so agents can always connect)
                if msg.type != MessageType.REGISTER and self._check_rate_limit(agent_id):
                    logger.warning("Rate limit exceeded for agent %s", agent_id)
                    await ws.send(
                        make_error(
                            agent_id,
                            "Rate limit exceeded. Slow down.",
                        ).to_json()
                    )
                    continue

                response = await self._dispatch(msg, ws)
                if response:
                    await ws.send(response.to_json())

        except websockets.exceptions.ConnectionClosed as exc:
            logger.info(
                "Connection closed for agent=%s (code=%s, reason=%s)",
                agent_id,
                getattr(exc, "code", "?"),
                getattr(exc, "reason", ""),
            )
        except Exception as exc:
            logger.error("Unexpected error in handler for agent=%s: %s", agent_id, exc)
        finally:
            # Only remove the agent if this connection is still the active one.
            # If the agent re-registered on a new connection, the old handler
            # should not clean up the new registration.
            if (
                agent_id
                and agent_id in self.agents
                and self.connections.get(agent_id) is ws
            ):
                self._remove_agent(agent_id, reason="disconnected")

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
            logger.warning("Unknown message type '%s' from agent %s", msg.type, msg.from_agent)
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

        # Check for duplicate agent_id (re-registration replaces the old one).
        # We update the registry first so the old handler's finally-block
        # sees the new connection and skips cleanup.
        if agent_id in self.agents:
            logger.info(
                "Agent %s re-registering (replacing old connection)", agent_id
            )

        now = datetime.now(timezone.utc)
        info = AgentInfo(
            id=agent_id,
            name=payload.get("name", agent_id),
            capabilities=payload.get("capabilities", []),
            status="online",
            metadata=payload.get("metadata", {}),
            last_heartbeat=now,
            registered_at=now,
        )

        old_ws = self.connections.get(agent_id)
        # Update registry BEFORE closing old connection, so the old handler's
        # finally-block sees a different ws and knows not to remove the agent.
        self.agents[agent_id] = info
        self.connections[agent_id] = ws

        if old_ws and old_ws != ws:
            try:
                await old_ws.close()
            except Exception:
                pass

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
        self._remove_agent(agent_id, reason="deregistered")

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

        # Guard: self-delegation
        if msg.from_agent == to_agent:
            return make_error(
                msg.from_agent, "Cannot delegate a task to yourself"
            )

        # Guard: target agent exists
        if to_agent not in self.agents:
            return make_error(
                msg.from_agent, f"Agent '{to_agent}' not found"
            )

        priority = payload.get("priority", 0)
        ttl = payload.get("ttl", self.task_ttl)
        max_retries = payload.get("max_retries", self.max_task_retries)

        task = TaskRecord(
            task_id=task_id,
            from_agent=msg.from_agent,
            to_agent=to_agent,
            capability=payload.get("capability", ""),
            params=payload.get("params", {}),
            status=TaskStatus.PENDING,
            ttl_seconds=ttl,
            priority=priority,
            max_retries=max_retries,
        )
        self.tasks[task_id] = task

        # Forward the delegate message to the target agent
        await self._forward_task(task, msg)

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

    async def _forward_task(
        self, task: TaskRecord, original_msg: A2AMessage | None = None
    ) -> None:
        """Forward a task to its assigned agent. Updates task status accordingly."""
        target_ws = self.connections.get(task.to_agent)
        if not target_ws:
            task.status = TaskStatus.FAILED
            task.error = f"Agent '{task.to_agent}' has no active connection"
            task.completed_at = datetime.now(timezone.utc)
            logger.error("No connection for agent %s to forward task %s", task.to_agent, task.task_id)
            return

        payload = {
            "task_id": task.task_id,
            "capability": task.capability,
            "params": task.params,
            "priority": task.priority,
        }

        forward = A2AMessage(
            type=MessageType.DELEGATE,
            id=original_msg.id if original_msg else str(uuid.uuid4()),
            **{"from": task.from_agent},
            to=task.to_agent,
            payload=payload,
        )
        try:
            await target_ws.send(forward.to_json())
            task.status = TaskStatus.ASSIGNED
        except Exception as exc:
            logger.error("Failed to forward task %s to %s: %s", task.task_id, task.to_agent, exc)
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.completed_at = datetime.now(timezone.utc)

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

        logger.info(
            "Task %s %s (from=%s, to=%s)",
            task_id,
            task.status.value,
            task.from_agent,
            task.to_agent,
        )

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
                    "priority": task.priority,
                    "retry_count": task.retry_count,
                },
            )

        # General hub status — includes task breakdown
        task_statuses: dict[str, int] = {}
        for t in self.tasks.values():
            task_statuses[t.status.value] = task_statuses.get(t.status.value, 0) + 1

        return A2AMessage(
            type=MessageType.STATUS,
            **{"from": "hub"},
            to=msg.from_agent,
            payload={
                "agents_count": len(self.agents),
                "tasks_count": len(self.tasks),
                "task_statuses": task_statuses,
                "agents": [a.model_dump() for a in self.agents.values()],
            },
        )

    async def _handle_broadcast(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage | None:
        sender = msg.from_agent
        sent = 0
        for agent_id, conn in list(self.connections.items()):
            if agent_id == sender:
                continue
            try:
                await conn.send(msg.to_json())
                sent += 1
            except Exception as exc:
                logger.warning("Failed to broadcast to %s: %s", agent_id, exc)

        return A2AMessage(
            type=MessageType.BROADCAST,
            **{"from": "hub"},
            to=sender,
            payload={"status": "ok", "recipients": sent},
        )

    async def _handle_heartbeat(
        self, msg: A2AMessage, ws: ServerConnection
    ) -> A2AMessage:
        agent_id = msg.from_agent
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.now(timezone.utc)

        return A2AMessage(
            type=MessageType.HEARTBEAT_ACK,
            **{"from": "hub"},
            to=agent_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_agent(self, agent_id: str, reason: str = "unknown") -> None:
        """Remove an agent from the registry and handle orphaned tasks.

        If the agent had in-progress tasks, attempts to retry them by
        finding another capable agent.
        """
        self.agents.pop(agent_id, None)
        self.connections.pop(agent_id, None)
        self._message_timestamps.pop(agent_id, None)
        logger.info("Agent removed: %s (reason: %s)", agent_id, reason)

        # Collect tasks that need retrying
        tasks_to_retry: list[TaskRecord] = []
        for task in self.tasks.values():
            if task.to_agent == agent_id and task.status in (
                TaskStatus.PENDING,
                TaskStatus.ASSIGNED,
                TaskStatus.IN_PROGRESS,
            ):
                if task.retry_count < task.max_retries:
                    tasks_to_retry.append(task)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Agent '{agent_id}' disconnected and max retries ({task.max_retries}) exhausted"
                    task.completed_at = datetime.now(timezone.utc)
                    logger.info(
                        "Task %s failed — agent %s disconnected, no retries left",
                        task.task_id,
                        agent_id,
                    )

        # Schedule retries asynchronously (we might be called from sync context)
        if tasks_to_retry:
            for task in tasks_to_retry:
                asyncio.ensure_future(self._retry_task(task, agent_id))

    async def _retry_task(self, task: TaskRecord, failed_agent_id: str) -> None:
        """Attempt to reassign a task to another capable agent."""
        task.retry_count += 1
        logger.info(
            "Retrying task %s (attempt %d/%d) — original agent %s disconnected",
            task.task_id,
            task.retry_count,
            task.max_retries,
            failed_agent_id,
        )

        # Find another agent with the required capability
        new_agent_id = self._find_capable_agent(task.capability, exclude={failed_agent_id})

        if new_agent_id:
            task.to_agent = new_agent_id
            task.status = TaskStatus.PENDING
            task.error = None
            await self._forward_task(task)
            logger.info("Task %s reassigned to %s", task.task_id, new_agent_id)
        else:
            task.status = TaskStatus.FAILED
            task.error = f"No available agent for capability '{task.capability}' after retry"
            task.completed_at = datetime.now(timezone.utc)
            logger.warning(
                "Task %s failed — no agent available for capability '%s'",
                task.task_id,
                task.capability,
            )

    def _find_capable_agent(
        self, capability: str, exclude: set[str] | None = None
    ) -> str | None:
        """Find an online agent that has the given capability."""
        exclude = exclude or set()
        for agent_id, info in self.agents.items():
            if agent_id in exclude:
                continue
            if capability in info.capabilities:
                return agent_id
        return None

    async def _cleanup_loop(self) -> None:
        """Periodically expire timed-out tasks and deregister stale agents.

        This loop runs every CLEANUP_LOOP_INTERVAL seconds and handles:
        1. Task TTL expiry — marks overdue tasks as TIMEOUT
        2. Heartbeat monitoring — removes agents that missed heartbeats
        """
        while True:
            await asyncio.sleep(CLEANUP_LOOP_INTERVAL)

            if self._shutting_down:
                break

            now = datetime.now(timezone.utc)

            # --- Task TTL expiry ---
            for task in list(self.tasks.values()):
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT):
                    continue
                elapsed = (now - task.created_at).total_seconds()
                if elapsed > task.ttl_seconds:
                    task.status = TaskStatus.TIMEOUT
                    task.completed_at = now
                    task.error = f"Task timed out after {int(elapsed)}s (TTL={task.ttl_seconds}s)"
                    logger.info(
                        "Task %s timed out after %ds", task.task_id, int(elapsed)
                    )
                    # Notify the requester about the timeout
                    requester_ws = self.connections.get(task.from_agent)
                    if requester_ws:
                        try:
                            timeout_msg = A2AMessage(
                                type=MessageType.RESULT,
                                **{"from": "hub"},
                                to=task.from_agent,
                                payload={
                                    "task_id": task.task_id,
                                    "error": task.error,
                                },
                            )
                            await requester_ws.send(timeout_msg.to_json())
                        except Exception:
                            pass

            # --- Heartbeat monitoring ---
            stale_agents = []
            for agent_id, info in self.agents.items():
                elapsed = (now - info.last_heartbeat).total_seconds()
                if elapsed > self.heartbeat_timeout:
                    stale_agents.append(agent_id)

            for agent_id in stale_agents:
                logger.warning(
                    "Agent %s missed heartbeat (last seen %.0fs ago), removing",
                    agent_id,
                    (now - self.agents[agent_id].last_heartbeat).total_seconds(),
                )
                # Close the stale connection
                conn = self.connections.get(agent_id)
                if conn:
                    try:
                        await conn.close()
                    except Exception:
                        pass
                self._remove_agent(agent_id, reason="heartbeat timeout")


async def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Convenience function to run the hub server."""
    server = HubServer(host=host, port=port)
    await server.run_forever()


if __name__ == "__main__":
    asyncio.run(run_server())
