"""A2A Hub Agent SDK — Build agents that connect to the hub.

Provides both a class-based and a functional API for creating agents
that register with the hub, receive tasks, and return results.

Usage (class-based):

    from a2a_hub import Agent, capability

    class MyAgent(Agent):
        name = "my-agent"

        @capability("greet", description="Say hello")
        async def greet(self, name: str) -> str:
            return f"Hello, {name}!"

    agent = MyAgent()
    agent.run()

Usage (functional):

    from a2a_hub import Agent

    agent = Agent("my-agent", capabilities=["echo"])

    @agent.on_task("echo")
    async def handle_echo(message: str) -> str:
        return message

    agent.run()
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from functools import wraps
from typing import Any, Callable, Coroutine

import websockets

from a2a_hub.protocol import (
    A2AMessage,
    MessageType,
    make_deregister,
    make_register,
    make_result,
)
from a2a_hub.utils import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    HEARTBEAT_INTERVAL,
    MAX_RECONNECT_ATTEMPTS,
    RECONNECT_DELAY,
    get_logger,
)

logger = get_logger("a2a-hub.agent")

# Type alias for task handlers
TaskHandler = Callable[..., Coroutine[Any, Any, Any]]


def capability(name: str, description: str = "") -> Callable:
    """Decorator that marks a method as an agent capability.

    Args:
        name: The capability tag (e.g. "code-review").
        description: Human-readable description of the capability.
    """

    def decorator(func: Callable) -> Callable:
        func._capability_name = name  # type: ignore[attr-defined]
        func._capability_description = description  # type: ignore[attr-defined]
        return func

    return decorator


class Agent:
    """Base class / functional agent for the A2A Hub.

    Can be used directly (functional style) or subclassed (class-based style).

    Args:
        name: Human-readable agent name. Also used as agent ID if id is not given.
        agent_id: Unique agent ID. Defaults to a UUID based on name.
        capabilities: List of capability tags (for functional style).
        hub_host: Hub server host.
        hub_port: Hub server port.
        metadata: Optional metadata dict.
    """

    # Subclass can set these as class attributes
    name: str = ""
    agent_id: str = ""

    def __init__(
        self,
        name: str | None = None,
        agent_id: str | None = None,
        capabilities: list[str] | None = None,
        hub_host: str = DEFAULT_HOST,
        hub_port: int = DEFAULT_PORT,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if name:
            self.name = name
        if not self.name:
            raise ValueError("Agent must have a name")

        if agent_id:
            self.agent_id = agent_id
        if not self.agent_id:
            self.agent_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

        self.hub_host = hub_host
        self.hub_port = hub_port
        self.metadata = metadata or {}

        self._ws: Any = None
        self._running = False
        self._handlers: dict[str, TaskHandler] = {}
        self._capabilities: list[str] = list(capabilities or [])

        # Collect decorated capabilities from class methods
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_capability_name"):
                cap_name = attr._capability_name
                self._capabilities.append(cap_name)
                self._handlers[cap_name] = attr

    @property
    def hub_url(self) -> str:
        """WebSocket URL of the hub."""
        return f"ws://{self.hub_host}:{self.hub_port}"

    @property
    def all_capabilities(self) -> list[str]:
        """All registered capabilities (from decorators + explicit list)."""
        return list(set(self._capabilities))

    def on_task(self, capability_name: str) -> Callable:
        """Register a task handler for a capability (functional style).

        Usage:
            @agent.on_task("echo")
            async def handle(message: str) -> str:
                return message
        """

        def decorator(func: TaskHandler) -> TaskHandler:
            self._handlers[capability_name] = func
            if capability_name not in self._capabilities:
                self._capabilities.append(capability_name)
            return func

        return decorator

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the hub and register."""
        self._ws = await websockets.connect(self.hub_url)
        reg_msg = make_register(
            self.agent_id, self.name, self.all_capabilities, self.metadata
        )
        await self._ws.send(reg_msg.to_json())

        # Wait for registration acknowledgement
        raw = await self._ws.recv()
        ack = A2AMessage.from_json(str(raw))
        if ack.type == MessageType.REGISTER:
            logger.info(
                "Registered with hub as %s (id=%s)", self.name, self.agent_id
            )
        else:
            logger.warning("Unexpected registration response: %s", ack.type)

    async def disconnect(self) -> None:
        """Deregister and close the connection."""
        if self._ws:
            try:
                msg = make_deregister(self.agent_id)
                await self._ws.send(msg.to_json())
                await self._ws.close()
            except Exception:
                pass
        self._running = False
        logger.info("Disconnected from hub")

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the hub with exponential backoff."""
        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            delay = RECONNECT_DELAY * min(attempt, 5)
            logger.info(
                "Reconnecting in %ds (attempt %d/%d)...",
                delay,
                attempt,
                MAX_RECONNECT_ATTEMPTS,
            )
            await asyncio.sleep(delay)
            try:
                await self.connect()
                return
            except Exception as exc:
                logger.warning("Reconnect attempt %d failed: %s", attempt, exc)
        logger.error("Max reconnect attempts reached. Giving up.")
        self._running = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _listen(self) -> None:
        """Main message loop: receive and process hub messages."""
        self._running = True
        while self._running:
            try:
                await self.connect()
                heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                async for raw in self._ws:
                    try:
                        msg = A2AMessage.from_json(str(raw))
                        await self._handle_message(msg)
                    except Exception as exc:
                        logger.error("Error processing message: %s", exc)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection lost")
            except Exception as exc:
                logger.error("Connection error: %s", exc)
            finally:
                if heartbeat_task:  # type: ignore[possibly-undefined]
                    heartbeat_task.cancel()

            if self._running:
                await self._reconnect()

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the hub."""
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if self._ws:
                try:
                    msg = A2AMessage(
                        type=MessageType.HEARTBEAT,
                        **{"from": self.agent_id},
                        to="hub",
                    )
                    await self._ws.send(msg.to_json())
                except Exception:
                    break

    async def _handle_message(self, msg: A2AMessage) -> None:
        """Dispatch an incoming message to the appropriate handler."""
        if msg.type == MessageType.DELEGATE:
            await self._handle_delegate(msg)
        elif msg.type == MessageType.BROADCAST:
            await self._on_broadcast(msg)
        elif msg.type == MessageType.RESULT:
            await self._on_result(msg)
        elif msg.type == MessageType.HEARTBEAT_ACK:
            pass  # Expected
        elif msg.type == MessageType.ERROR:
            logger.warning("Error from hub: %s", msg.payload.get("error"))

    async def _handle_delegate(self, msg: A2AMessage) -> None:
        """Execute a delegated task and send the result back."""
        payload = msg.payload
        task_id = payload.get("task_id", "")
        cap = payload.get("capability", "")
        params = payload.get("params", {})

        handler = self._handlers.get(cap)
        if not handler:
            result_msg = make_result(
                self.agent_id,
                msg.from_agent,
                task_id,
                error=f"No handler for capability: {cap}",
            )
            await self._ws.send(result_msg.to_json())
            return

        try:
            # Call handler with params
            sig = inspect.signature(handler)
            if sig.parameters:
                result = await handler(**params)
            else:
                result = await handler()

            result_msg = make_result(
                self.agent_id, msg.from_agent, task_id, result=result
            )
        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc)
            result_msg = make_result(
                self.agent_id, msg.from_agent, task_id, error=str(exc)
            )

        await self._ws.send(result_msg.to_json())

    async def _on_broadcast(self, msg: A2AMessage) -> None:
        """Handle an incoming broadcast. Override in subclass if needed."""
        logger.info(
            "Broadcast from %s: %s",
            msg.from_agent,
            msg.payload.get("message", ""),
        )

    async def _on_result(self, msg: A2AMessage) -> None:
        """Handle an incoming task result. Override in subclass if needed."""
        logger.info(
            "Result for task %s: %s",
            msg.payload.get("task_id"),
            msg.payload.get("result"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        hub_host: str | None = None,
        hub_port: int | None = None,
    ) -> None:
        """Start the agent (blocking). Connects to the hub and listens.

        Args:
            hub_host: Override hub host.
            hub_port: Override hub port.
        """
        if hub_host:
            self.hub_host = hub_host
        if hub_port:
            self.hub_port = hub_port

        try:
            asyncio.run(self._listen())
        except KeyboardInterrupt:
            logger.info("Agent shutting down...")

    async def start(
        self,
        hub_host: str | None = None,
        hub_port: int | None = None,
    ) -> None:
        """Start the agent (async, non-blocking). Use in existing event loops."""
        if hub_host:
            self.hub_host = hub_host
        if hub_port:
            self.hub_port = hub_port
        await self._listen()
