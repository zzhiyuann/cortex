"""Agent runner — spawns CLI-based AI coding agents.

Supports two backends:
- CLI mode: spawns `claude -p` per request (~7s cold start)
- Sidecar mode: calls a persistent Node.js SDK service (~3s, no cold start)

The sidecar is auto-started on first use if the script exists.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .session import Session

log = logging.getLogger("dispatcher")

SIDECAR_PORT = int(os.environ.get("SIDECAR_PORT", "18899"))
SIDECAR_URL = f"http://127.0.0.1:{SIDECAR_PORT}/query"


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
        self._sidecar_proc = None
        self._sidecar_healthy = False

    def _resolve_command(self):
        """Find the full path of the agent command."""
        resolved = shutil.which(self.command)
        if resolved:
            self.command = resolved

    def _is_claude(self) -> bool:
        return "claude" in self.command.lower()

    # -- Sidecar management --

    def _sidecar_script(self) -> Path | None:
        """Find the sidecar script relative to this package."""
        here = Path(__file__).resolve().parent.parent / "scripts" / "claude-sidecar.mjs"
        return here if here.exists() else None

    async def _ensure_sidecar(self) -> bool:
        """Start sidecar if not running. Returns True if available."""
        if self._sidecar_healthy:
            return True

        # Check if already running
        if self._ping_sidecar():
            self._sidecar_healthy = True
            return True

        script = self._sidecar_script()
        if not script:
            return False

        node = shutil.which("node")
        if not node:
            return False

        sdk_dir = script.parent
        sdk_pkg = sdk_dir / "node_modules" / "@anthropic-ai" / "claude-agent-sdk"
        if not sdk_pkg.exists():
            log.warning("sidecar SDK not installed at %s", sdk_dir)
            return False

        log.info("starting claude-sidecar from %s", script)
        env = os.environ.copy()
        for key in ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_ENTRYPOINT"):
            env.pop(key, None)
        env["NODE_PATH"] = str(sdk_dir / "node_modules")

        self._sidecar_proc = await asyncio.create_subprocess_exec(
            node, str(script),
            cwd=str(sdk_dir),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for it to be ready (up to 5s)
        for _ in range(50):
            await asyncio.sleep(0.1)
            if self._ping_sidecar():
                self._sidecar_healthy = True
                log.info("sidecar ready on port %d", SIDECAR_PORT)
                return True

        log.warning("sidecar failed to start within 5s")
        return False

    def _ping_sidecar(self) -> bool:
        """Check if sidecar is alive."""
        try:
            req = Request(
                SIDECAR_URL,
                data=json.dumps({"prompt": "", "maxTurns": 0}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urlopen(req, timeout=1)
            return True
        except Exception:
            return False

    async def _invoke_sidecar(
        self, session: Session, prompt: str,
        resume: bool = False, max_turns: int = 1,
        model: str | None = None,
    ) -> str:
        """Call the sidecar HTTP service."""
        payload = {
            "prompt": prompt,
            "maxTurns": max_turns,
            "cwd": session.cwd,
        }
        if resume:
            payload["sessionId"] = session.sid
            payload["resume"] = True
        else:
            payload["sessionId"] = session.sid
        if model:
            payload["model"] = model

        body = json.dumps(payload).encode()

        def _do_request():
            req = Request(
                SIDECAR_URL,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urlopen(req, timeout=self.timeout)
            return json.loads(resp.read())

        result = await asyncio.to_thread(_do_request)

        # Update session ID from sidecar response
        if result.get("sessionId"):
            session.sid = result["sessionId"]

        out = result.get("result", "")
        if result.get("error"):
            out = f"Error: {result['error']}"

        # If sidecar returned empty (e.g. maxTurns exhausted during tool use),
        # signal caller to fall back to CLI by returning empty string.
        if not out or not out.strip():
            log.warning("sidecar returned empty result, will fall back to CLI")

        return out

    # -- Main invoke --

    async def invoke(
        self,
        session: Session,
        prompt: str,
        resume: bool = False,
        max_turns: int = 50,
        model: str | None = None,
        stream: bool = True,
    ) -> str:
        """Invoke agent. Uses sidecar for fast plain-mode calls, CLI for streaming."""
        session.status = "running"
        session.started = time.time()

        # For non-streaming (plain mode), prefer sidecar if available (~3s vs ~7s)
        if not stream and self._is_claude():
            try:
                sidecar_ok = await self._ensure_sidecar()
                if sidecar_ok:
                    log.info(
                        "sidecar  resume=%s  cwd=%s  sid=%s",
                        resume, session.cwd, session.sid[:8],
                    )
                    out = await asyncio.wait_for(
                        self._invoke_sidecar(session, prompt, resume, max_turns, model),
                        timeout=self.timeout,
                    )
                    # If sidecar returned a real result, use it.
                    # Empty result → fall through to CLI for a retry.
                    if out and out.strip():
                        session.status = "done" if "Error:" not in out[:20] else "failed"
                        session.finished = time.time()
                        session.result = out
                        return out
                    log.info("sidecar returned empty, falling back to CLI")
                    # Sidecar likely exhausted turns during tool use;
                    # give CLI more turns so it can finish the response.
                    max_turns = max(max_turns * 2, 6)
            except Exception as exc:
                log.warning("sidecar failed, falling back to CLI: %s", exc)
                self._sidecar_healthy = False

        # Fall back to CLI
        return await self._invoke_cli(
            session, prompt, resume, max_turns, model, stream,
        )

    async def _invoke_cli(
        self,
        session: Session,
        prompt: str,
        resume: bool = False,
        max_turns: int = 50,
        model: str | None = None,
        stream: bool = True,
    ) -> str:
        """Spawn agent CLI process."""
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
        for key in ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_ENTRYPOINT"):
            env.pop(key, None)

        log.info(
            "cli  cmd=%s  resume=%s  cwd=%s  sid=%s",
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
                limit=1024 * 1024,
            )
            session.proc = proc

            if use_stream:
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
        """Read stream-json output line by line, accumulating text across turns."""
        result_text = ""
        last_text = ""
        completed_turns: list[str] = []

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "assistant":
                message = event.get("message", {})
                content_blocks = message.get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            if last_text and not text.startswith(last_text):
                                completed_turns.append(last_text)
                            last_text = text
                            all_parts = completed_turns + [text]
                            session.partial_output = "\n\n".join(all_parts)

            elif etype == "result":
                result_text = event.get("result", "")

        await proc.wait()

        stderr_data = await proc.stderr.read()
        stderr_text = stderr_data.decode(errors="replace").strip()
        if stderr_text:
            log.debug("agent stderr: %s", stderr_text[:500])

        if not result_text and last_text:
            result_text = last_text

        if not result_text and stderr_text:
            result_text = f"(stderr) {stderr_text[:800]}"

        return result_text
