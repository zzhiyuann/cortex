"""Core Dispatcher — routes Telegram messages to AI agent sessions."""

from __future__ import annotations

import asyncio
import logging
import os
import html
import re
import signal
import tempfile
import time
from pathlib import Path

from .config import Config
from .memory import Memory
from .runner import AgentRunner
from .session import Session, SessionManager
from .telegram import TelegramClient

# Self-healing feedback loop — write issues to shared JSONL store.
# Works standalone (no cortex_cli dependency) by writing directly to the file.
try:
    from cortex_cli.feedback import record_issue
except ImportError:
    import json as _json
    _FEEDBACK_DIR = Path.home() / ".cortex" / "feedback"
    _ISSUES_FILE = _FEEDBACK_DIR / "issues.jsonl"

    def record_issue(source="dispatcher", category="error", description="", context=None, **_kw):
        """Standalone fallback: write issue to shared JSONL store."""
        try:
            _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
            issue = {
                "id": f"{source}-{int(time.time() * 1000)}",
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "source": source,
                "category": category,
                "description": description,
                "context": context or {},
                "status": "open",
            }
            with open(_ISSUES_FILE, "a") as f:
                f.write(_json.dumps(issue) + "\n")
        except Exception:
            pass  # never crash the dispatcher for feedback

log = logging.getLogger("dispatcher")

# Chinese aliases for common project concepts
_PROJECT_ALIASES: dict[str, list[str]] = {
    "网站": ["cortex", "website", "web", "frontend"],
    "网页": ["cortex", "website", "web", "frontend"],
    "前端": ["frontend", "web", "website"],
    "后端": ["api", "backend", "server"],
    "工具": ["forge", "tool"],
    "回放": ["vibe", "replay"],
    "录制": ["vibe", "replay"],
    "调度": ["dispatcher"],
    "机器人": ["dispatcher", "bot"],
    "签证": ["visa"],
    "书": ["book"],
}

# Progress phase descriptions (Chinese) based on elapsed time
_PROGRESS_PHASES = [
    (30, "正在分析任务..."),
    (60, "正在读取代码..."),
    (120, "正在编写修改..."),
    (180, "还在跑... 任务比较复杂"),
    (300, "跑了挺久了，再等等"),
    (600, "跑了 {m} 分钟，应该快好了"),
]


# Precompiled pattern for Markdown→HTML conversion (avoids recompiling per call)
_MD_PATTERN = re.compile(
    r'```(?:\w*\n)?(.*?)```'   # fenced code block
    r'|`([^`]+)`'              # inline code
    r'|\*\*(.+?)\*\*',        # bold
    re.DOTALL,
)


def _md_to_telegram_html(text: str) -> str:
    """Convert agent Markdown output to Telegram HTML.

    Handles **bold**, `inline code`, and ```code blocks```.
    All other text is HTML-escaped so parse_mode='HTML' is safe.
    """
    parts: list[str] = []
    pos = 0

    for m in _MD_PATTERN.finditer(text):
        # Escape plain text before this match
        if m.start() > pos:
            parts.append(html.escape(text[pos:m.start()]))

        code_block, inline_code, bold = m.groups()
        if code_block is not None:
            parts.append(f"<pre>{html.escape(code_block)}</pre>")
        elif inline_code is not None:
            parts.append(f"<code>{html.escape(inline_code)}</code>")
        else:
            parts.append(f"<b>{html.escape(bold)}</b>")

        pos = m.end()

    # Trailing plain text
    if pos < len(text):
        parts.append(html.escape(text[pos:]))

    return "".join(parts)


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable Chinese duration."""
    if seconds < 60:
        return f"{int(seconds)}秒"
    m = int(seconds) // 60
    s = int(seconds) % 60
    if m < 60:
        return f"{m}分{s}秒" if s else f"{m}分钟"
    h = m // 60
    m = m % 60
    return f"{h}小时{m}分"


class Dispatcher:
    """Main event loop: poll Telegram, classify, dispatch to agent."""

    _MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

    def __init__(self, config: Config):
        self.cfg = config
        self.tg = TelegramClient(config.bot_token, config.chat_id)
        self.runner = AgentRunner(
            command=config.agent_command,
            args=config.agent_args,
            timeout=config.timeout,
        )
        self.mem = Memory(config.memory_file)
        self.sm = SessionManager(recent_window=config.recent_window)
        self.routes = config.get_project_routes()
        self.offset = 0
        self.alive = True
        self._tasks: set[asyncio.Task] = set()
        self._start_time = time.time()
        self._consecutive_failures = 0
        self._msg_buffer: list[tuple] = []  # (mid, text, reply_to, attachments)
        self._batch_task: asyncio.Task | None = None
        self._sticky_model: str | None = None  # @Model (capitalized) sets this

    # -- Lifecycle --

    def _acquire_pid(self) -> bool:
        """Write PID file. Returns False if another instance is running."""
        self._pid_file = self.cfg.data_dir / "dispatcher.pid"
        if self._pid_file.exists():
            try:
                old_pid = int(self._pid_file.read_text().strip())
                # Check if process is alive
                os.kill(old_pid, 0)
                log.error("Another dispatcher is running (pid %d)", old_pid)
                return False
            except (ProcessLookupError, ValueError):
                pass  # Stale PID file, process is gone
            except PermissionError:
                log.error("Another dispatcher is running (pid check: permission denied)")
                return False
        self._pid_file.write_text(str(os.getpid()))
        return True

    def _release_pid(self):
        """Remove PID file on shutdown."""
        if hasattr(self, "_pid_file") and self._pid_file.exists():
            try:
                self._pid_file.unlink()
            except OSError:
                pass

    async def run(self):
        log.info("Dispatcher starting")

        if not self._acquire_pid():
            log.error("Aborting: another instance is already running")
            return

        # Register bot commands menu
        self.tg.set_my_commands([
            {"command": "status", "description": "查看当前任务状态"},
            {"command": "cancel", "description": "取消正在运行的任务"},
            {"command": "history", "description": "查看最近任务记录"},
            {"command": "help", "description": "使用帮助"},
        ])

        self.tg.send("\u2705 Dispatcher online.")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)

        while self.alive:
            try:
                updates = await asyncio.to_thread(
                    self.tg.poll, self.offset, self.cfg.poll_timeout
                )
                for u in updates:
                    self.offset = u["update_id"] + 1
                    msg = u.get("message")
                    cb = u.get("callback_query")
                    edited = u.get("edited_message")
                    if msg:
                        await self._on_message(msg)
                    elif cb:
                        await self._on_callback(cb)
                    elif edited:
                        await self._on_edited_message(edited)

            except Exception:
                log.exception("poll loop error")
                await asyncio.sleep(5)

        if self._tasks:
            log.info("Draining %d in-flight tasks", len(self._tasks))
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._release_pid()

    def _shutdown(self):
        log.info("Shutdown signal")
        self.alive = False
        for s in self.sm.active():
            if s.proc:
                s.proc.kill()
                s.status = "cancelled"
        self._release_pid()
        self.tg.send("\u26a0\ufe0f Dispatcher offline.")

    # -- Message routing --

    async def _on_message(self, msg: dict):
        if msg.get("chat", {}).get("id") != self.cfg.chat_id:
            return

        mid = msg["message_id"]
        reply_to = (msg.get("reply_to_message") or {}).get("message_id")

        text = (msg.get("text") or "").strip()
        caption = (msg.get("caption") or "").strip()
        attachments: list[dict] = []

        # Process media if present
        media = await self._extract_media(msg)
        if media:
            if media["kind"] == "text":
                # Voice/audio → whisper transcription becomes the text
                transcription = media["text"]
                if not text:
                    if caption and transcription:
                        text = f"{caption}\n\n[语音内容]: {transcription}"
                    else:
                        text = caption or transcription
            elif media["kind"] == "file":
                # Photo/video/document → attach for agent to read directly
                attachments.append(media)
                if not text:
                    text = caption or f"[User sent a {media['media_type']}]"

        if not text:
            return

        # Fire-and-forget typing — never block the event loop
        self._fire_typing()

        cat = self._classify(text)
        log.info("[%d] '%s' -> %s (attachments: %d)", mid, text[:80], cat, len(attachments))

        if cat == "status":
            self._handle_status(mid)
        elif cat == "cancel":
            self._handle_cancel(mid, text, reply_to)
        elif cat == "history":
            self._handle_history(mid)
        elif cat == "help":
            self._handle_help(mid)
        elif cat == "new_session":
            self._handle_new_session(mid)
        else:
            # F6: forward message context
            fwd_from = msg.get("forward_from", {}).get("first_name") or msg.get("forward_sender_name")
            fwd_chat = msg.get("forward_from_chat", {}).get("title")
            if fwd_from or fwd_chat:
                source = fwd_from or fwd_chat
                text = f"[Forwarded from {source}]: {text}"

            # F1: instant reaction feedback
            self._fire_reaction(mid, "\U0001f440")

            # F3: buffer non-reply tasks for batching; replies go directly
            if reply_to:
                await self._handle_task(mid, text, reply_to, attachments=attachments)
            else:
                self._buffer_message(mid, text, reply_to, attachments or [])

    async def _on_callback(self, cb: dict):
        """Handle inline keyboard button presses."""
        cb_id = cb["id"]
        data = cb.get("data", "")
        msg = cb.get("message", {})
        chat_id = msg.get("chat", {}).get("id")

        if chat_id != self.cfg.chat_id:
            self.tg.answer_callback(cb_id, "Unauthorized")
            return

        if data.startswith("cancel:"):
            session_id = data[7:]
            target = None
            for s in self.sm.active():
                if s.sid.startswith(session_id):
                    target = s
                    break
            if target and target.proc:
                target.proc.kill()
                target.status = "cancelled"
                target.finished = time.time()
                self.tg.answer_callback(cb_id, "已取消")
                self.tg.edit(
                    msg["message_id"],
                    f"\u274c 已取消: {target.task_text[:40]}",
                )
            else:
                self.tg.answer_callback(cb_id, "任务已结束")
        elif data.startswith("retry:"):
            try:
                orig_mid = int(data.split(":")[1])
            except (IndexError, ValueError):
                self.tg.answer_callback(cb_id, "无效的重试请求")
                return
            session = self.sm.by_msg.get(orig_mid)
            if session:
                self.tg.answer_callback(cb_id, "\U0001f504 重试中...")
                await self._handle_task(
                    orig_mid, session.task_text, None,
                    model=session.model_override,
                )
            else:
                self.tg.answer_callback(cb_id, "找不到原始任务")

        elif data == "new_session":
            self.sm.force_new = True
            self.tg.answer_callback(cb_id, "好的，下条消息将开始新对话")

        else:
            self.tg.answer_callback(cb_id)


    async def _on_edited_message(self, msg: dict):
        """Handle edited messages — cancel and re-dispatch if session is young."""
        if msg.get("chat", {}).get("id") != self.cfg.chat_id:
            return
        mid = msg["message_id"]
        new_text = (msg.get("text") or "").strip()
        if not new_text:
            return

        session = self.sm.by_msg.get(mid)
        if not session:
            return

        # Only re-dispatch if session started less than 10 seconds ago
        if session.status == "running" and session.started:
            age = time.time() - session.started
            if age < 10 and session.proc:
                log.info("re-dispatching edited message [%d] (age=%.1fs)", mid, age)
                session.proc.kill()
                session.status = "cancelled"
                session.finished = time.time()
                # Re-dispatch with new text
                self._fire_typing()
                await self._handle_task(mid, new_text, None)
                return

        log.debug("ignoring edit for [%d] — session too old or not running", mid)

    def _handle_new_session(self, mid: int):
        """Mark that the next message should start a fresh session."""
        self.sm.force_new = True
        self._reply(mid, "\U0001f195 好的，下条消息将开始新对话。")

    def _buffer_message(self, mid: int, text: str, reply_to: int | None, attachments: list):
        """Buffer a task message for batching. Flushes after 2s of quiet."""
        self._msg_buffer.append((mid, text, reply_to, attachments))
        # Cancel previous flush timer
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
        # Schedule new flush in 2 seconds
        self._batch_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self):
        """Wait 2s then flush buffered messages."""
        try:
            await asyncio.sleep(2)
            await self._flush_buffer()
        except asyncio.CancelledError:
            pass

    async def _flush_buffer(self):
        """Process buffered messages — merge if multiple."""
        if not self._msg_buffer:
            return
        batch = list(self._msg_buffer)
        self._msg_buffer.clear()
        self._batch_task = None

        if len(batch) == 1:
            mid, text, reply_to, atts = batch[0]
            await self._handle_task(mid, text, reply_to, attachments=atts)
        else:
            # Merge multiple messages into one prompt
            texts = [text for _, text, _, _ in batch]
            combined = "\n\n".join(texts)
            all_atts = []
            for _, _, _, atts in batch:
                if atts:
                    all_atts.extend(atts)
            last_mid = batch[-1][0]
            log.info("batched %d messages into one prompt", len(batch))
            await self._handle_task(last_mid, combined, None, attachments=all_atts)

    def _classify(self, text: str) -> str:
        low = text.strip().lower()
        # Handle /command format from bot menu
        if low.startswith("/"):
            cmd = low.split()[0].lstrip("/").split("@")[0]  # strip /cmd@botname
            if cmd in ("cancel", "stop"):
                return "cancel"
            if cmd in ("status",):
                return "status"
            if cmd in ("history",):
                return "history"
            if cmd in ("help",):
                return "help"
        if any(kw in low for kw in self.cfg.cancel_keywords):
            return "cancel"
        if any(kw in low for kw in self.cfg.status_keywords):
            return "status"
        if low in ("history", "历史", "最近任务", "跑过什么"):
            return "history"
        if low in ("help", "帮助", "命令", "怎么用"):
            return "help"
        if low in ("新session", "新建session", "new session", "开个新的", "新对话"):
            return "new_session"
        return "task"

    def _detect_project(self, text: str) -> str | None:
        low = text.lower()

        # Direct keyword matching (existing behavior)
        for name, path in self.routes.items():
            if name in low and path.exists():
                return str(path)

        # Chinese alias matching
        for alias, keywords in _PROJECT_ALIASES.items():
            if alias in low:
                for kw in keywords:
                    if kw in self.routes and self.routes[kw].exists():
                        return str(self.routes[kw])

        # Fuzzy: check if any project name appears as a substring
        for name, path in self.routes.items():
            if len(name) >= 3 and name in low and path.exists():
                return str(path)

        return None

    # -- Media processing --

    async def _extract_media(self, msg: dict) -> dict | None:
        """Extract media from a message. Returns structured result:

        For voice/audio: {"kind": "text", "text": "transcribed text"}
        For photo/video/doc: {"kind": "file", "media_type": "photo", "path": "/tmp/..."}

        Images are NOT pre-described — the agent reads them directly,
        so it can interpret them in context with the user's question.
        """
        file_id = None
        media_type = None

        if msg.get("voice"):
            file_id = msg["voice"]["file_id"]
            media_type = "voice"
        elif msg.get("audio"):
            file_id = msg["audio"]["file_id"]
            media_type = "audio"
        elif msg.get("video_note"):
            file_id = msg["video_note"]["file_id"]
            media_type = "video_note"
        elif msg.get("photo"):
            file_id = msg["photo"][-1]["file_id"]
            media_type = "photo"
        elif msg.get("document"):
            file_id = msg["document"]["file_id"]
            media_type = "document"
        elif msg.get("video"):
            file_id = msg["video"]["file_id"]
            media_type = "video"

        if not file_id:
            return None

        log.info("processing %s media", media_type)

        ext = {
            "voice": ".ogg", "audio": ".mp3", "video_note": ".mp4",
            "photo": ".jpg", "document": "", "video": ".mp4",
        }.get(media_type, "")

        fd = tempfile.NamedTemporaryFile(suffix=ext, prefix=f"dispatch_{media_type}_", delete=False)
        tmp = fd.name
        fd.close()
        if not self.tg.download_file(file_id, tmp):
            log.error("failed to download %s", media_type)
            return None

        # Check downloaded file size
        try:
            file_size = os.path.getsize(tmp)
            if file_size > self._MAX_DOWNLOAD_SIZE:
                log.warning("downloaded file too large: %d bytes", file_size)
                os.unlink(tmp)
                return None
        except OSError:
            pass

        try:
            if media_type in ("voice", "audio"):
                text = await self._transcribe_audio(tmp)
                # Voice temp file no longer needed after transcription
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                return {"kind": "text", "text": text or ""} if text else None
            else:
                # Photo, video, document → pass file path for agent to read
                return {"kind": "file", "media_type": media_type, "path": tmp}
        except Exception:
            log.exception("media processing failed for %s", media_type)
            # Still try to return file for agent to inspect
            return {"kind": "file", "media_type": media_type, "path": tmp}

    _whisper_model = None  # class-level cache to avoid reloading on every message

    async def _transcribe_audio(self, audio_path: str) -> str | None:
        """Transcribe audio using OpenAI Whisper locally — fast, no API cost."""
        def _run_whisper():
            import whisper
            if Dispatcher._whisper_model is None:
                Dispatcher._whisper_model = whisper.load_model("turbo")
            result = Dispatcher._whisper_model.transcribe(
                audio_path,
                language=None,
                initial_prompt="这是一段中文或英文的语音消息，涉及编程、代码、项目管理等话题。",
                condition_on_previous_text=False,
            )
            return result.get("text", "").strip()

        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(_run_whisper),
                timeout=120,
            )
            log.info("whisper transcribed %d chars from %s", len(text), audio_path)
            return text if text else None
        except Exception:
            log.exception("whisper transcription failed for %s", audio_path)
            return None

    # _describe_image removed — images are passed directly to the agent
    # session via attachments, so it can read them in context with the
    # user's question. No more separate Sonnet pre-description step.

    # -- Handlers --

    def _handle_status(self, mid: int):
        uptime = _format_duration(time.time() - self._start_time)
        active = self.sm.active()
        if not active:
            self._reply(mid, f"\U0001f4a4 空闲中，没有在跑的任务。\n\n\u23f1 运行时间: {uptime}")
            return

        lines = [f"\U0001f3c3 正在运行 <b>{len(active)}</b> 个任务：\n"]
        buttons = []
        for s in active:
            elapsed = _format_duration(s.elapsed())
            proj = html.escape(s.project_name)
            task = html.escape(s.task_text[:50])
            lines.append(f"\u2022 <b>{proj}</b> — {task}\n  \u23f1 {html.escape(elapsed)}")
            buttons.append({
                "text": f"\u274c 取消 {s.project_name}",
                "callback_data": f"cancel:{s.sid[:8]}",
            })

        markup = None
        if buttons:
            # One button per row
            markup = {
                "inline_keyboard": [[b] for b in buttons],
            }

        lines.append(f"\n\u23f1 运行时间: {html.escape(uptime)}")
        self._reply(mid, "\n".join(lines), parse_mode="HTML",
                    reply_markup=markup)

    def _handle_cancel(self, mid: int, text: str, reply_to: int | None):
        target = None
        if reply_to:
            target = self.sm.find_by_reply(reply_to)
        if not target:
            for s in self.sm.active():
                if s.project_name.lower() in text.lower():
                    target = s
                    break
        if not target:
            active = self.sm.active()
            if len(active) == 1:
                target = active[0]

        if target and target.status == "running" and target.proc:
            target.proc.kill()
            target.status = "cancelled"
            target.finished = time.time()
            elapsed = _format_duration(target.elapsed())
            self._reply(mid, f"\u274c 已取消 <b>{html.escape(target.project_name)}</b> 的任务\n"
                        f"运行了 {html.escape(elapsed)}",
                        parse_mode="HTML")
        elif not self.sm.active():
            self._reply(mid, "没有在跑的任务。")
        else:
            active = self.sm.active()
            lines = ["不确定要取消哪个，当前在跑的任务：\n"]
            buttons = []
            for s in active:
                lines.append(f"\u2022 {s.project_name} — {s.task_text[:40]}")
                buttons.append({
                    "text": f"\u274c 取消 {s.project_name}",
                    "callback_data": f"cancel:{s.sid[:8]}",
                })
            markup = {"inline_keyboard": [[b] for b in buttons]}
            self._reply(mid, "\n".join(lines), reply_markup=markup)

    def _handle_history(self, mid: int):
        """Show recent task history."""
        recent = []
        for rmid in self.sm.recent[:10]:
            s = self.sm.by_msg.get(rmid)
            if s and s.status in ("done", "failed", "cancelled"):
                recent.append(s)
        if not recent:
            self._reply(mid, "没有最近完成的任务。")
            return

        lines = ["\U0001f4cb <b>最近任务</b>：\n"]
        for s in recent[:8]:
            icon = {"done": "\u2705", "failed": "\u274c", "cancelled": "\u26a0\ufe0f"}.get(
                s.status, "\u2753"
            )
            elapsed = _format_duration(s.elapsed()) if s.started else "—"
            proj = html.escape(s.project_name)
            task = html.escape(s.task_text[:40])
            lines.append(f"{icon} <b>{proj}</b> — {task}  ({html.escape(elapsed)})")

        self._reply(mid, "\n".join(lines), parse_mode="HTML")

    def _handle_help(self, mid: int):
        """Show available commands."""
        help_text = (
            "\U0001f916 <b>Dispatcher 使用指南</b>\n\n"
            "\U0001f4ac <b>发任务</b>：直接发消息，自动识别项目\n"
            "\U0001f504 <b>跟进</b>：回复之前的消息继续对话\n"
            "\U0001f4ca <b>状态</b>：发「在干嘛」或「status」\n"
            "\U0001f6d1 <b>取消</b>：发「取消」或「stop」\n"
            "\U0001f4cb <b>历史</b>：发「历史」或「history」\n"
            "\u2753 <b>帮助</b>：发「帮助」或「help」\n\n"
            "\U0001f4c1 <b>已配置项目</b>：\n"
        )
        for name in self.cfg.projects:
            proj = self.cfg.projects[name]
            kws = ", ".join(proj.get("keywords", []))
            help_text += f"  \u2022 {html.escape(name)} ({html.escape(kws)})\n"

        self._reply(mid, help_text, parse_mode="HTML")

    _NEW_SESSION_KEYWORDS = {"新session", "新建session", "new session", "开个新的", "新对话"}

    # Model aliases: lowercase = current message only, capitalized = persist in follow-ups
    _MODEL_PREFIXES_TEMP = {"@haiku": "haiku", "@sonnet": "sonnet", "@opus": "opus"}
    _MODEL_PREFIXES_STICKY = {"@Haiku": "haiku", "@Sonnet": "sonnet", "@Opus": "opus"}

    def _extract_model_prefix(self, text: str) -> tuple[str, str | None, bool]:
        """Extract @model prefix from message. Returns (clean_text, model_or_none, sticky).

        Lowercase @haiku/@opus/@sonnet = current message only.
        Capitalized @Haiku/@Opus/@Sonnet = persist in follow-ups.
        """
        # Check sticky (capitalized) first — case-sensitive
        for prefix, model in self._MODEL_PREFIXES_STICKY.items():
            if text.startswith(prefix):
                rest = text[len(prefix):]
                if rest and not rest[0].isspace():
                    continue
                clean = rest.lstrip()
                if clean:
                    return clean, model, True
        # Then check temp (lowercase) — case-sensitive
        for prefix, model in self._MODEL_PREFIXES_TEMP.items():
            if text.lower().startswith(prefix):
                rest = text[len(prefix):]
                if rest and not rest[0].isspace():
                    continue
                clean = rest.lstrip()
                if clean:
                    return clean, model, False
        return text, None, False

    async def _handle_task(
        self, mid: int, text: str, reply_to: int | None,
        attachments: list[dict] | None = None, model: str | None = None,
    ):
        """Route a task message. NEVER blocks — always processes immediately.

        Architecture:
        - Chat/quick questions → lightweight parallel session (low max_turns)
        - Project tasks → full parallel session (high max_turns)
        - Explicit reply to running task → queue as follow-up (only case that waits)
        - All sessions get context about what's running in background
        - Chat window is NEVER stuck waiting
        """
        # 0. Extract model preference from @haiku/@opus/@sonnet prefix
        text, model_from_prefix, is_sticky = self._extract_model_prefix(text)
        if is_sticky and model_from_prefix:
            self._sticky_model = model_from_prefix
        # Priority: explicit arg > prefix > sticky > None
        model_override = model or model_from_prefix or self._sticky_model

        # 1. Explicit reply to a running task → only case where we queue
        if reply_to:
            prev = self.sm.find_by_reply(reply_to)
            if prev and prev.status == "running":
                self._reply(mid, "\u23f3 等这个任务跑完继续。")
                self._spawn(self._do_queued_followup(mid, text, prev, attachments, model=model_override, model_sticky=is_sticky))
                return
            if prev:
                self._spawn(self._do_followup(mid, text, prev, attachments, model=model_override, model_sticky=is_sticky))
                return

        # 2. Build background context — all sessions know what else is running
        active = self.sm.active()
        bg_context = ""
        if active:
            bg_lines = ["[Background: these tasks are currently running]"]
            for s in active:
                elapsed = _format_duration(s.elapsed())
                bg_lines.append(f"  - {s.project_name}: {s.task_text[:60]} ({elapsed})")
            bg_context = "\n".join(bg_lines)

        # 3. Quick vs task: project detected = task, otherwise = quick chat
        #    No Haiku classification overhead — project routing is zero-cost.
        cwd = self._detect_project(text) or str(Path.home())
        is_project = cwd != str(Path.home())
        is_quick = not is_project

        # 4. If nothing running, try to resume last session for continuity
        #    SKIP for quick questions — they go through the fast plain-mode path
        if not is_quick and not self.sm.force_new:
            last = self.sm.last_session()
            if not active and last and last.status in ("done", "failed"):
                self._spawn(self._do_followup(mid, text, last, attachments, model=model_override, model_sticky=is_sticky))
                return
        self.sm.force_new = False

        # 5. Always create a new session — never block
        session = self.sm.create(mid, text, cwd)
        session.is_task = is_project
        session.model_sticky = is_sticky
        self._spawn(self._do_session(
            mid, session, text, attachments, bg_context=bg_context,
            quick=is_quick, model=model_override,
        ))

    # -- Session runners --

    async def _do_session(
        self, mid: int, session: Session, text: str,
        attachments: list[dict] | None = None,
        bg_context: str = "",
        quick: bool = False,
        model: str | None = None,
    ):
        """Run a task with progress feedback."""
        prompt = self._build_prompt(text, session.cwd, attachments, bg_context)
        session.model_override = model
        if quick:
            max_turns = min(self.cfg.max_turns_chat, 5)  # Quick Q&A: 5 turns max
        else:
            max_turns = self.cfg.max_turns if session.is_task else self.cfg.max_turns_chat

        # Quick: plain mode (fast, no stream overhead). Task: stream-json (progressive updates).
        runner = asyncio.create_task(
            self.runner.invoke(
                session, prompt, resume=False, max_turns=max_turns,
                model=model, stream=not quick,
            )
        )
        # Quick questions: typing only, no progress messages (no noise)
        monitor = asyncio.create_task(self._progress_loop(mid, session, quiet=quick))
        result = await runner
        monitor.cancel()
        self._send_result(mid, session, result)

    async def _do_queued_followup(
        self, mid: int, text: str, target: Session,
        attachments: list[dict] | None = None,
        model: str | None = None,
        model_sticky: bool = False,
    ):
        """Wait for a running session to finish, then resume with the new message."""
        while target.status in ("pending", "running"):
            await asyncio.sleep(1)
        await self._do_followup(mid, text, target, attachments, model=model, model_sticky=model_sticky)

    async def _do_followup(
        self, mid: int, text: str, prev: Session,
        attachments: list[dict] | None = None,
        model: str | None = None,
        model_sticky: bool = False,
    ):
        """Resume a previous session for follow-up."""
        effective_model = model or self._sticky_model
        effective_sticky = model_sticky or (prev.model_sticky if not model else False)

        # If model differs from previous session, --resume would ignore it.
        # Start a fresh session with previous context injected instead.
        model_changed = (
            effective_model
            and prev.model_override != effective_model
        )
        if model_changed:
            session = self.sm.create(mid, text, prev.cwd)  # new session id
            resume = False
        else:
            session = self.sm.create(mid, text, prev.cwd, sid=prev.sid)
            resume = True

        session.is_task = prev.is_task
        session.model_override = effective_model
        session.model_sticky = effective_sticky

        # Build follow-up text with context
        follow_text = text

        # If previous session failed, inject error context so the agent understands
        if prev.status == "failed" and prev.result:
            error_snippet = prev.result[:500]
            follow_text = (
                f"[Previous task failed with this error]:\n{error_snippet}\n\n"
                f"[User follow-up]: {text}"
            )

        # If we can't resume (model changed), inject previous result as context
        if not resume and prev.result:
            prev_snippet = prev.result[:1000]
            follow_text = (
                f"[Previous conversation result]:\n{prev_snippet}\n\n"
                f"[User says]: {follow_text}"
            )

        if attachments:
            parts = [follow_text]
            for a in attachments:
                parts.append(
                    f"\n\n[Attached {a['media_type']}: {a['path']}]"
                    f"\nUse the Read tool to view this file."
                )
            follow_text = "".join(parts)

        prompt = follow_text if resume else self._build_prompt(
            follow_text, session.cwd,
        )

        runner = asyncio.create_task(
            self.runner.invoke(
                session, prompt, resume=resume,
                max_turns=self.cfg.max_turns_followup, model=effective_model,
            )
        )
        monitor = asyncio.create_task(self._progress_loop(mid, session))
        result = await runner
        monitor.cancel()
        self._send_result(mid, session, result)

    async def _progress_loop(self, mid: int, session: Session, quiet: bool = False):
        """Send real-time streaming updates from agent output.

        quiet=True: only send typing indicators, no progress messages.
        Used for quick Q&A so the user just gets the answer with no noise.
        """
        try:
            progress_msg_id = None
            last_partial_len = 0
            last_update_time = 0
            phase_shown = False

            while session.status in ("pending", "running"):
                await asyncio.sleep(5 if quiet else 3)
                if session.status not in ("pending", "running"):
                    break

                self._fire_typing()

                # Quick sessions: typing only, no messages
                if quiet:
                    continue

                if not session.started:
                    continue

                elapsed = session.elapsed()
                partial = session.partial_output

                # Show streaming partial output if available
                if partial and len(partial) > last_partial_len:
                    now = time.time()
                    # Throttle updates to every 5 seconds
                    if now - last_update_time < 5:
                        continue
                    last_update_time = now
                    last_partial_len = len(partial)

                    # Truncate for preview (Telegram limit + readability)
                    preview = partial
                    if len(preview) > 2000:
                        preview = "..." + preview[-1800:]
                    display = _md_to_telegram_html(preview) + "\n\n<i>\u270f\ufe0f 正在输出...</i>"

                    if progress_msg_id:
                        self.tg.edit(progress_msg_id, display, parse_mode="HTML")
                    else:
                        progress_msg_id = self._reply(mid, display, parse_mode="HTML")

                # Fall back to phase messages if no streaming output yet
                elif not partial and not phase_shown and elapsed > 60:
                    phase_shown = True
                    msg = self._get_progress_message(session)
                    if progress_msg_id:
                        self.tg.edit(progress_msg_id, msg)
                    else:
                        progress_msg_id = self._reply(mid, msg)

            # Clean up progress message
            if progress_msg_id:
                try:
                    self.tg.edit(progress_msg_id,
                                 f"\u2705 {html.escape(session.task_text[:40])} — 完成",
                                 parse_mode="HTML")
                except Exception:
                    pass

        except asyncio.CancelledError:
            pass

    def _get_progress_message(self, session: Session) -> str:
        """Generate a contextual progress message."""
        elapsed = session.elapsed()
        m = int(elapsed) // 60

        for threshold, template in _PROGRESS_PHASES:
            if elapsed < threshold:
                return template.format(m=m)

        return f"\u23f3 已经跑了 {m} 分钟... 任务还在进行中"

    # -- Helpers --

    def _fire_typing(self):
        """Send typing indicator without blocking the event loop.

        Runs in a background thread, fire-and-forget with a short timeout.
        Failures are silently ignored — typing is cosmetic, never worth blocking for.
        """
        async def _do():
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.tg.typing),
                    timeout=3,
                )
            except Exception:
                pass
        # Schedule without awaiting — truly fire-and-forget
        asyncio.ensure_future(_do())

    def _fire_reaction(self, mid: int, emoji: str):
        """Set a reaction on a message without blocking the event loop."""
        async def _do():
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.tg.react, mid, emoji),
                    timeout=3,
                )
            except Exception:
                pass
        asyncio.ensure_future(_do())

    def _build_prompt(
        self, text: str, cwd: str,
        attachments: list[dict] | None = None,
        bg_context: str = "",
    ) -> str:
        project = Path(cwd).name
        prompt = (
            f"Working directory: {cwd}  (project: {project})\n\n"
            f"User context:\n{self.mem.text}\n\n"
            f"User says: {text}\n\n"
        )

        if attachments:
            prompt += "## Attached Files\n"
            for a in attachments:
                prompt += (
                    f"- {a['media_type']}: {a['path']}\n"
                    f"  Use the Read tool to view this file. "
                    f"Interpret it in context with the user's message above.\n"
                )
            prompt += "\n"

        if bg_context:
            prompt += f"\n{bg_context}\n\n"

        prompt += (
            "Do what the user asks. Summarize the result concisely. "
            "IMPORTANT: Do NOT send any Telegram messages yourself (no curl to "
            "Telegram API). Your stdout will be relayed to the user automatically.\n"
            "FORMATTING: Your output is displayed in Telegram. "
            "You may use **bold** for emphasis and `code` for inline code. "
            "Do NOT use other Markdown (no headers, no bullet-point -, no links). "
            "Keep it simple and readable."
        )
        return prompt

    def _reply(
        self,
        reply_to: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict | None = None,
    ) -> int | None:
        bot_id = self.tg.send(text, reply_to=reply_to, parse_mode=parse_mode,
                               reply_markup=reply_markup)
        if bot_id:
            self.sm.link_bot(bot_id, reply_to)
        return bot_id

    def _reply_document(self, reply_to: int, file_path: str, caption: str = "") -> int | None:
        """Send a document as a reply."""
        bot_id = self.tg.send_document(file_path, caption=caption, reply_to=reply_to)
        if bot_id:
            self.sm.link_bot(bot_id, reply_to)
        return bot_id

    def _send_result(self, mid: int, session: Session, result: str):
        if session.status == "cancelled":
            record_issue(
                source="dispatcher",
                category="user_cancel",
                description=f"User cancelled task in {session.project_name}",
                context={
                    "user_message": session.task_text[:200],
                    "elapsed": session.elapsed(),
                    "project": session.project_name,
                },
            )
            return

        # Track consecutive failures for escalation
        if session.status == "done":
            self._consecutive_failures = 0
            self._fire_reaction(mid, "\u2705")

        if not result or not result.strip():
            self._reply(mid, "\u2705 完成了，但没有输出内容。")
            record_issue(
                source="dispatcher",
                category="empty_response",
                description=f"Agent returned empty response for {session.project_name}",
                context={"user_message": session.task_text[:200]},
            )
            return

        elapsed = _format_duration(session.elapsed()) if session.started else ""

        if session.status == "failed":
            self._consecutive_failures += 1
            self._fire_reaction(mid, "\u274c")
            friendly = self._friendly_error(result)
            if self._consecutive_failures >= 3:
                friendly += f"\n\n\u26a0\ufe0f 连续失败 {self._consecutive_failures} 次，建议检查 agent 状态。"
            # Retry button
            retry_markup = {
                "inline_keyboard": [[{
                    "text": "\U0001f504 重试",
                    "callback_data": f"retry:{mid}",
                }]]
            }
            self._reply(mid, f"\u274c <b>任务失败</b>\n\n{html.escape(friendly)}",
                        parse_mode="HTML", reply_markup=retry_markup)
            record_issue(
                source="dispatcher",
                category="error",
                description=f"Task failed in {session.project_name}: {result[:100]}",
                context={
                    "user_message": session.task_text[:200],
                    "error": result[:500],
                    "elapsed": session.elapsed(),
                    "project": session.project_name,
                    "consecutive_failures": self._consecutive_failures,
                },
            )
            return

        # Build response
        if session.elapsed() > 60 and elapsed:
            header = f"\u2705 完成 ({elapsed})\n\n"
        else:
            header = ""

        full = header + result
        formatted = _md_to_telegram_html(full)

        # Quick action button for task sessions
        result_markup = None
        if session.is_task and session.elapsed() > 30:
            result_markup = {
                "inline_keyboard": [[{
                    "text": "\U0001f195 新对话",
                    "callback_data": "new_session",
                }]]
            }

        # Long output -> send as file instead of splitting
        if len(formatted) > 4000:
            fd = tempfile.NamedTemporaryFile(
                mode='w', suffix='.md', prefix='result_', delete=False,
            )
            fd.write(result)
            tmp_path = fd.name
            fd.close()
            # Short summary as caption
            summary = result[:150].replace('\n', ' ')
            if len(result) > 150:
                summary += "..."
            caption = f"\u2705 完成" + (f" ({elapsed})" if session.elapsed() > 60 else "")
            self._reply_document(mid, tmp_path, caption=caption)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        else:
            self._reply(mid, formatted, parse_mode="HTML", reply_markup=result_markup)

    def _friendly_error(self, error: str) -> str:
        """Convert raw error text to user-friendly Chinese message."""
        low = error.lower()

        if "timed out" in low or "timeout" in low:
            return ("任务超时了。可能是任务太复杂，"
                    "或者 agent 卡住了。\n\n"
                    "可以试试拆成更小的任务重新发。")

        if "max turns" in low:
            return ("Agent 达到了最大轮次限制，任务可能没完全做完。\n\n"
                    "回复这条消息让它继续。")

        if "rate limit" in low or "429" in low:
            return ("API 限流了，等一会儿再试。")

        if "permission" in low or "denied" in low:
            return ("权限不足，agent 无法执行某些操作。\n\n"
                    f"详情：{error[:200]}")

        if "not found" in low or "no such file" in low:
            return (f"找不到文件或命令。\n\n详情：{error[:200]}")

        if "(stderr)" in error:
            # Strip the stderr prefix and give context
            clean = error.replace("(stderr) ", "").strip()
            if len(clean) > 300:
                clean = clean[:300] + "..."
            return f"执行出错：\n{clean}"

        # Generic: show a cleaned-up version
        if len(error) > 500:
            return f"出错了：\n{error[:400]}...\n\n回复「详情」看完整输出。"
        return f"出错了：\n{error}"

    def _split_message(self, text: str, max_len: int) -> list[str]:
        """Split long text into chunks, preferring line boundaries."""
        chunks = []
        while len(text) > max_len:
            # Find a good split point (newline near the limit)
            split_at = text.rfind("\n", 0, max_len)
            if split_at < max_len // 2:
                split_at = max_len  # No good newline, hard split
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        if text:
            chunks.append(text)
        return chunks

    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task):
        """Handle completed background tasks: cleanup and log exceptions."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("background task failed: %s", exc, exc_info=exc)
