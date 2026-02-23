"""Session tracking for dispatched tasks."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional


class Session:
    """A single dispatched task or conversation turn."""

    __slots__ = (
        "msg_id", "task_text", "cwd", "sid", "conv_id", "status",
        "proc", "started", "finished", "result", "is_task",
        "bot_msgs", "partial_output", "model_override", "model_sticky",
        # F7: transient question relay fields (NOT serialized)
        "pending_question", "answer_event", "answer_data",
        "stdin_writer", "stdin_drain",
    )

    def __init__(
        self, msg_id: int, text: str, cwd: str,
        sid: str | None = None, conv_id: str | None = None,
    ):
        self.msg_id = msg_id
        self.task_text = text
        self.cwd = cwd
        self.sid = sid or str(uuid.uuid4())
        self.conv_id = conv_id or self.sid  # links sessions in same conversation
        self.status = "pending"  # pending | running | done | failed | cancelled
        self.proc = None  # asyncio.subprocess.Process
        self.started: float | None = None
        self.finished: float | None = None
        self.result = ""
        self.is_task = True
        self.bot_msgs: list[int] = []
        self.partial_output = ""  # streaming: accumulated text so far
        self.model_override: str | None = None
        self.model_sticky: bool = False  # True = model persists in follow-ups
        # F7: transient — not persisted
        self.pending_question: dict | None = None  # {tool_use_id, questions, tg_msg_id}
        self.answer_event: asyncio.Event | None = None
        self.answer_data: str | None = None
        self.stdin_writer = None   # proc.stdin.write reference
        self.stdin_drain = None    # proc.stdin.drain reference

    def elapsed(self) -> float:
        end = self.finished or time.time()
        return end - self.started if self.started else 0

    @property
    def project_name(self) -> str:
        return Path(self.cwd).name


class SessionManager:
    """Track active and recent sessions, map bot replies to originals."""

    def __init__(self, recent_window: int = 300):
        self.by_msg: dict[int, Session] = {}
        self.bot_to_orig: dict[int, int] = {}
        self.recent: list[int] = []
        self.recent_window = recent_window
        self.force_new: bool = False

    def create(
        self, msg_id: int, text: str, cwd: str,
        sid: str | None = None, conv_id: str | None = None,
    ) -> Session:
        s = Session(msg_id, text, cwd, sid, conv_id)
        self.by_msg[msg_id] = s
        self.recent.insert(0, msg_id)
        if len(self.recent) > 100:
            self.recent = self.recent[:100]
        return s

    def link_bot(self, bot_msg_id: int, orig_msg_id: int):
        self.bot_to_orig[bot_msg_id] = orig_msg_id

    def find_by_reply(self, reply_to: int) -> Session | None:
        if reply_to in self.by_msg:
            return self.by_msg[reply_to]
        orig = self.bot_to_orig.get(reply_to)
        if orig and orig in self.by_msg:
            return self.by_msg[orig]
        return None

    def last_session(self) -> Session | None:
        """Get the most recently created session (any status)."""
        for mid in self.recent:
            s = self.by_msg.get(mid)
            if s:
                return s
        return None

    def active(self) -> list[Session]:
        return [s for s in self.by_msg.values() if s.status == "running"]

    def active_count(self) -> int:
        return len(self.active())
