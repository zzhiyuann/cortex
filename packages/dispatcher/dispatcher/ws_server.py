"""WebSocket server for native app connectivity.

Runs alongside the Telegram bot, accepting WebSocket connections
on a configurable port. Uses the same routing/session logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Awaitable

import websockets
from websockets.asyncio.server import Server, ServerConnection

log = logging.getLogger("dispatcher")

# Type for the message handler callback provided by core.py
# Args: content, project, msg_id, websocket, image_base64, audio_base64, audio_duration, language
MessageHandler = Callable[
    [str, str | None, str, "ServerConnection", str | None, str | None, float | None, str | None],
    Awaitable[str],
]

# Type for the command handler callback provided by core.py
# Args: command, data (full message dict)
# Returns: dict with structured response data
CommandHandler = Callable[
    [str, dict[str, Any]],
    Awaitable[dict[str, Any]],
]

# Type for the answer handler callback provided by core.py
# Args: session_id_prefix, answer_text, msg_id
AnswerHandler = Callable[
    [str, str, str],
    Awaitable[None],
]

# Type for the edit handler callback provided by core.py
# Args: original_msg_id, new_content, websocket
# Returns: dict with {"ok": bool, "error"?: str}
EditHandler = Callable[
    [str, str, "ServerConnection"],
    Awaitable[dict[str, Any]],
]


class WebSocketServer:
    """WebSocket server that bridges native app clients to the Dispatcher.

    Accepts connections, authenticates them (optional token), and routes
    messages through the same pipeline as Telegram.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        auth_token: str = "",
        on_message: MessageHandler | None = None,
        on_command: CommandHandler | None = None,
        on_answer: AnswerHandler | None = None,
        on_edit: EditHandler | None = None,
    ):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.on_message = on_message
        self.on_command = on_command
        self.on_answer = on_answer
        self.on_edit = on_edit
        self._server: Server | None = None
        self._clients: set[ServerConnection] = set()
        self._status_task: asyncio.Task | None = None
        self._active_sessions_count: int = 0

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        )
        log.info("WebSocket server listening on ws://%s:%d", self.host, self.port)
        # Start periodic status broadcast
        self._status_task = asyncio.create_task(self._status_loop())

    async def stop(self) -> None:
        """Stop the WebSocket server and disconnect all clients."""
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            log.info("WebSocket server stopped")

        self._clients.clear()

    def update_session_count(self, count: int) -> None:
        """Update the active session count (called by core.py)."""
        self._active_sessions_count = count

    @property
    def client_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._clients)

    # -- Connection handling --

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a new WebSocket client connection."""
        # Authenticate if token is configured
        if self.auth_token:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(raw)
                if data.get("type") != "auth" or data.get("token") != self.auth_token:
                    await self._send(websocket, {
                        "type": "error",
                        "id": data.get("id", ""),
                        "message": "Authentication failed",
                    })
                    await websocket.close(4001, "Authentication failed")
                    return
            except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as exc:
                log.warning("ws auth failed: %s", exc)
                await websocket.close(4001, "Authentication timeout or invalid")
                return

        self._clients.add(websocket)
        remote = websocket.remote_address
        log.info("ws client connected: %s (total: %d)", remote, len(self._clients))

        # Send initial status
        await self.send_status(websocket)

        try:
            async for raw_message in websocket:
                try:
                    data = json.loads(raw_message)
                    await self._handle_message(websocket, data)
                except json.JSONDecodeError:
                    await self._send(websocket, {
                        "type": "error",
                        "id": "",
                        "message": "Invalid JSON",
                    })
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            log.info("ws client disconnected: %s (remaining: %d)", remote, len(self._clients))

    async def _handle_message(self, websocket: ServerConnection, data: dict) -> None:
        """Process an incoming WebSocket message.

        Messages are dispatched concurrently so multiple user messages
        can be in-flight at the same time (first-response-first-arrive).
        """
        msg_type = data.get("type", "")
        msg_id = data.get("id", str(uuid.uuid4()))

        if msg_type == "ping":
            await self._send(websocket, {"type": "pong", "id": msg_id})
            return

        if msg_type == "command":
            cmd = data.get("command", "").strip()
            if not cmd:
                await self._send(websocket, {
                    "type": "error",
                    "id": msg_id,
                    "message": "Empty command",
                })
                return
            asyncio.create_task(self._process_command(websocket, msg_id, cmd, data))
            return

        if msg_type == "answer":
            session_id = data.get("session_id", "").strip()
            answer = data.get("answer", "").strip()
            if not session_id or not answer:
                await self._send(websocket, {
                    "type": "error",
                    "id": msg_id,
                    "message": "Answer requires session_id and answer fields",
                })
                return
            if self.on_answer:
                try:
                    await self.on_answer(session_id, answer, msg_id)
                except Exception as exc:
                    log.exception("ws answer handler error for session %s", session_id)
                    await self._send(websocket, {
                        "type": "error",
                        "id": msg_id,
                        "message": f"Answer error: {str(exc)[:200]}",
                    })
            return

        if msg_type == "edit":
            content = data.get("content", "").strip()
            if not content:
                await self._send(websocket, {
                    "type": "error",
                    "id": msg_id,
                    "message": "Edit requires non-empty content",
                })
                return
            asyncio.create_task(self._process_edit(websocket, msg_id, content))
            return

        if msg_type == "notification":
            # External services can push notifications to all connected clients
            # without going through the agent pipeline.
            content = data.get("content", "").strip()
            source = data.get("source", "system")
            if content:
                await self.broadcast({
                    "type": "notification",
                    "id": msg_id,
                    "content": content,
                    "source": source,
                })
                log.info("ws notification broadcast from %s: %s", source, content[:80])
            return

        if msg_type == "message":
            content = data.get("content", "").strip()
            image_base64 = data.get("image_base64")
            audio_base64 = data.get("audio_base64")
            audio_duration = data.get("duration")
            language = data.get("language")

            # Allow empty content if there's an image or audio
            if not content and not image_base64 and not audio_base64:
                await self._send(websocket, {
                    "type": "error",
                    "id": msg_id,
                    "message": "Empty message content",
                })
                return

            project = data.get("project")

            # Acknowledge receipt immediately so the client can show 👀
            await self._send(websocket, {
                "type": "ack",
                "id": msg_id,
            })

            # Dispatch handling concurrently (fire-and-forget) so we don't
            # block the receive loop — the next message can start processing
            # right away.
            asyncio.create_task(
                self._process_message(
                    websocket, msg_id, content, project,
                    image_base64, audio_base64, audio_duration, language,
                )
            )
            return

        # Unknown message type
        await self._send(websocket, {
            "type": "error",
            "id": msg_id,
            "message": f"Unknown message type: {msg_type}",
        })

    async def _process_message(
        self,
        websocket: ServerConnection,
        msg_id: str,
        content: str,
        project: str | None,
        image_base64: str | None,
        audio_base64: str | None,
        audio_duration: float | None,
        language: str | None,
    ) -> None:
        """Process a single message through the dispatcher pipeline.

        Runs as a concurrent task so multiple messages can be handled in parallel.
        """
        if self.on_message:
            try:
                result = await self.on_message(
                    content, project, msg_id, websocket,
                    image_base64, audio_base64, audio_duration, language,
                )
                # Send final response
                await self._send(websocket, {
                    "type": "response",
                    "id": msg_id,
                    "content": result or "",
                    "streaming": False,
                })
            except Exception as exc:
                log.exception("ws message handler error for msg %s", msg_id[:8])
                await self._send(websocket, {
                    "type": "error",
                    "id": msg_id,
                    "message": f"Handler error: {str(exc)[:200]}",
                })
        else:
            await self._send(websocket, {
                "type": "error",
                "id": msg_id,
                "message": "No message handler configured",
            })

    async def _process_command(
        self,
        websocket: ServerConnection,
        msg_id: str,
        command: str,
        data: dict,
    ) -> None:
        """Process a slash command and return a structured response.

        Runs as a concurrent task so it doesn't block the receive loop.
        """
        if self.on_command:
            try:
                result = await self.on_command(command, data)
                await self._send(websocket, {
                    "type": "command_result",
                    "id": msg_id,
                    "command": command,
                    "data": result,
                })
            except Exception as exc:
                log.exception("ws command handler error for %s", command)
                await self._send(websocket, {
                    "type": "error",
                    "id": msg_id,
                    "message": f"Command error: {str(exc)[:200]}",
                })
        else:
            await self._send(websocket, {
                "type": "error",
                "id": msg_id,
                "message": "No command handler configured",
            })

    async def _process_edit(
        self,
        websocket: ServerConnection,
        msg_id: str,
        content: str,
    ) -> None:
        """Process an edit request for a previously sent message.

        Delegates to the on_edit callback which handles session lookup,
        process killing, and re-dispatch.
        """
        if self.on_edit:
            try:
                result = await self.on_edit(msg_id, content, websocket)
                if result.get("ok"):
                    await self._send(websocket, {
                        "type": "edit_ack",
                        "id": msg_id,
                    })
                else:
                    await self._send(websocket, {
                        "type": "error",
                        "id": msg_id,
                        "message": result.get("error", "Edit failed"),
                    })
            except Exception as exc:
                log.exception("ws edit handler error for msg %s", msg_id[:8])
                await self._send(websocket, {
                    "type": "error",
                    "id": msg_id,
                    "message": f"Edit error: {str(exc)[:200]}",
                })
        else:
            await self._send(websocket, {
                "type": "error",
                "id": msg_id,
                "message": "No edit handler configured",
            })

    # -- Sending helpers --

    async def _send(self, websocket: ServerConnection, data: dict) -> None:
        """Send a JSON message to a single client. Silently handles errors."""
        try:
            await websocket.send(json.dumps(data, ensure_ascii=False))
        except websockets.exceptions.ConnectionClosed:
            self._clients.discard(websocket)
        except Exception:
            log.debug("ws send failed", exc_info=True)

    async def send_response(
        self,
        websocket: ServerConnection,
        msg_id: str,
        content: str,
        streaming: bool = True,
    ) -> None:
        """Send a response message to a specific client."""
        await self._send(websocket, {
            "type": "response",
            "id": msg_id,
            "content": content,
            "streaming": streaming,
        })

    async def send_status(self, websocket: ServerConnection) -> None:
        """Send current status to a specific client."""
        await self._send(websocket, {
            "type": "status",
            "connected": True,
            "active_sessions": self._active_sessions_count,
        })

    async def send_question(
        self,
        websocket: ServerConnection,
        msg_id: str,
        session_id: str,
        question: str,
        options: list[str],
        allow_free_text: bool = True,
    ) -> None:
        """Send an AskUserQuestion to a specific WebSocket client."""
        await self._send(websocket, {
            "type": "question",
            "id": msg_id,
            "session_id": session_id,
            "question": question,
            "options": options,
            "allow_free_text": allow_free_text,
        })

    async def broadcast(self, data: dict) -> None:
        """Send a message to all connected clients."""
        if not self._clients:
            return
        message = json.dumps(data, ensure_ascii=False)
        # Send to all clients concurrently, remove dead ones
        dead: list[ServerConnection] = []
        tasks = []
        for ws in self._clients:
            tasks.append(self._try_send(ws, message, dead))
        if tasks:
            await asyncio.gather(*tasks)
        for ws in dead:
            self._clients.discard(ws)

    async def _try_send(
        self, ws: ServerConnection, message: str, dead: list[ServerConnection]
    ) -> None:
        """Attempt to send to a client, marking it dead on failure."""
        try:
            await ws.send(message)
        except Exception:
            dead.append(ws)

    async def broadcast_response(
        self, msg_id: str, content: str, streaming: bool = True
    ) -> None:
        """Broadcast a response to all connected clients."""
        await self.broadcast({
            "type": "response",
            "id": msg_id,
            "content": content,
            "streaming": streaming,
        })

    async def broadcast_status(self) -> None:
        """Broadcast current status to all connected clients."""
        await self.broadcast({
            "type": "status",
            "connected": True,
            "active_sessions": self._active_sessions_count,
        })

    # -- Background tasks --

    async def _status_loop(self) -> None:
        """Periodically send status updates to all connected clients."""
        try:
            while True:
                await asyncio.sleep(30)
                if self._clients:
                    await self.broadcast_status()
        except asyncio.CancelledError:
            pass
