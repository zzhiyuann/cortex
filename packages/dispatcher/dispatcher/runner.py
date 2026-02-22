"""Agent runner — spawns CLI-based AI coding agents."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time

from .session import Session

log = logging.getLogger("dispatcher")


class AgentRunner:
    """Spawn and manage CLI agent subprocesses.

    Default: Claude Code (`claude -p --session-id X`).
    Extensible: any CLI agent via config command + args.
    Uses stream-json output for real-time progress.
    """

    def __init__(
        self,
        command: str = "claude",
        args: list[str] | None = None,
        timeout: int = 1800,
    ):
        self.command = command
        self.args = args or ["-p", "--dangerously-skip-permissions"]
        self.timeout = timeout
        self._resolve_command()

    def _resolve_command(self):
        """Find the full path of the agent command."""
        resolved = shutil.which(self.command)
        if resolved:
            self.command = resolved

    def _is_claude(self) -> bool:
        return "claude" in self.command.lower()

    async def invoke(
        self,
        session: Session,
        prompt: str,
        resume: bool = False,
        max_turns: int = 50,
        model: str | None = None,
        stream: bool = True,
    ) -> str:
        """Spawn agent process. stream=False for fast plain mode."""
        cmd = [self.command] + list(self.args)

        # Claude-specific: session management
        use_stream = stream and self._is_claude()
        if self._is_claude():
            if resume:
                cmd += ["--resume", session.sid]
            else:
                cmd += ["--session-id", session.sid]
            cmd += ["--max-turns", str(max_turns)]
            if use_stream:
                cmd += ["--verbose", "--output-format", "stream-json"]
            if model:
                cmd += ["--model", model]

        env = os.environ.copy()
        # Prevent recursive agent invocations
        for key in ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_ENTRYPOINT"):
            env.pop(key, None)

        log.info(
            "agent  cmd=%s  resume=%s  cwd=%s  sid=%s",
            self.command, resume, session.cwd, session.sid[:8],
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=session.cwd,
                env=env,
                limit=1024 * 1024,  # 1MB line buffer (default 64KB too small for stream-json)
            )
            session.proc = proc
            session.status = "running"
            session.started = time.time()

            if use_stream:
                # Stream mode: write stdin manually, then read events line-by-line
                proc.stdin.write(prompt.encode())
                await proc.stdin.drain()
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                out = await asyncio.wait_for(
                    self._read_stream(proc, session),
                    timeout=self.timeout,
                )
            else:
                # Plain mode: use communicate() which handles stdin+stdout atomically
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode()),
                    timeout=self.timeout,
                )
                out = stdout.decode(errors="replace").strip()
                if not out and stderr:
                    err_text = stderr.decode(errors="replace").strip()
                    if err_text:
                        out = f"(stderr) {err_text[:800]}"

            session.status = "done" if proc.returncode == 0 else "failed"
            session.finished = time.time()
            session.result = out
            return out

        except asyncio.TimeoutError:
            if session.proc:
                session.proc.kill()
            session.status = "failed"
            session.finished = time.time()
            return f"Timed out after {self.timeout // 60} minutes"

        except asyncio.CancelledError:
            if session.proc:
                session.proc.kill()
            session.status = "cancelled"
            session.finished = time.time()
            return ""

        except Exception as exc:
            session.status = "failed"
            session.finished = time.time()
            log.exception("agent invoke error")
            return f"Error: {exc}"

    async def _read_stream(self, proc, session: Session) -> str:
        """Read stream-json output line by line, accumulating text.

        Claude Code stream-json format (with --verbose):
        - {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
        - {"type":"result","result":"final text"}

        Updates session.partial_output for the progress loop to display.
        Returns the final result text.
        """
        result_text = ""
        last_text = ""
        completed_turns: list[str] = []  # finished assistant turns

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            # Extract text from assistant message events
            if etype == "assistant":
                message = event.get("message", {})
                content_blocks = message.get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            # Detect new turn: current text doesn't extend previous
                            if last_text and not text.startswith(last_text):
                                completed_turns.append(last_text)
                            last_text = text
                            # Show all turns chained together
                            all_parts = completed_turns + [text]
                            session.partial_output = "\n\n".join(all_parts)

            # Final result — authoritative
            elif etype == "result":
                result_text = event.get("result", "")

        await proc.wait()

        # Always read stderr for logging, even if we have output
        stderr_data = await proc.stderr.read()
        stderr_text = stderr_data.decode(errors="replace").strip()
        if stderr_text:
            log.debug("agent stderr: %s", stderr_text[:500])

        # Prefer result event, fall back to last assistant text
        if not result_text and last_text:
            result_text = last_text

        # If still nothing, use stderr as the result
        if not result_text and stderr_text:
            result_text = f"(stderr) {stderr_text[:800]}"

        return result_text
