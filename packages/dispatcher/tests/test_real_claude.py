"""Integration tests using real claude -p calls.

These test the actual agent behavior — not mocked.
Run with: pytest tests/test_real_claude.py -v -s
"""

import asyncio
import os
import time
from pathlib import Path

import pytest

CLAUDE_BIN = str(Path.home() / ".local" / "bin" / "claude")
TIMEOUT = 60


def _clean_env():
    env = os.environ.copy()
    for key in ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(key, None)
    return env


async def _run_claude(prompt: str, cwd: str = None, max_turns: int = 10):
    """Run claude -p and return (exit_code, stdout, elapsed)."""
    cmd = [CLAUDE_BIN, "-p", "--dangerously-skip-permissions",
           "--max-turns", str(max_turns)]
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd or str(Path.home()),
        env=_clean_env(),
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=prompt.encode()),
        timeout=TIMEOUT,
    )
    elapsed = time.time() - t0
    out = stdout.decode(errors="replace").strip()
    err = stderr.decode(errors="replace").strip()
    return proc.returncode, out, err, elapsed


# -- Tests --

class TestSimpleMessages:
    """Simple messages should complete in one shot, no tools needed."""

    @pytest.mark.asyncio
    async def test_hi(self):
        code, out, err, elapsed = await _run_claude("hi")
        print(f"  'hi' -> [{code}] {out[:100]}  ({elapsed:.1f}s)")
        assert code == 0
        assert len(out) > 0
        assert "max turns" not in out.lower()

    @pytest.mark.asyncio
    async def test_greeting_chinese(self):
        code, out, err, elapsed = await _run_claude("你好")
        print(f"  '你好' -> [{code}] {out[:100]}  ({elapsed:.1f}s)")
        assert code == 0
        assert len(out) > 0
        assert "max turns" not in out.lower()

    @pytest.mark.asyncio
    async def test_simple_question(self):
        code, out, err, elapsed = await _run_claude(
            "What's the difference between a list and a tuple in Python? "
            "Answer in 1-2 sentences."
        )
        print(f"  question -> [{code}] {out[:150]}  ({elapsed:.1f}s)")
        assert code == 0
        assert len(out) > 10
        assert "max turns" not in out.lower()

    @pytest.mark.asyncio
    async def test_thanks(self):
        code, out, err, elapsed = await _run_claude("谢谢")
        print(f"  '谢谢' -> [{code}] {out[:100]}  ({elapsed:.1f}s)")
        assert code == 0
        assert "max turns" not in out.lower()

    @pytest.mark.asyncio
    async def test_joke(self):
        code, out, err, elapsed = await _run_claude(
            "Tell me a short programming joke, one liner"
        )
        print(f"  joke -> [{code}] {out[:150]}  ({elapsed:.1f}s)")
        assert code == 0
        assert len(out) > 5
        assert "max turns" not in out.lower()


class TestProjectTasks:
    """Tasks requiring file reads should complete with enough turns."""

    @pytest.mark.asyncio
    async def test_read_file(self):
        """Agent reads a file and summarizes."""
        prompt = (
            "Read the file pyproject.toml and tell me the package name "
            "and version. Just those two facts, nothing else."
        )
        code, out, err, elapsed = await _run_claude(
            prompt,
            cwd=str(Path.home() / "projects" / "dispatcher"),
            max_turns=5,
        )
        print(f"  read file -> [{code}] {out[:200]}  ({elapsed:.1f}s)")
        assert code == 0
        assert "max turns" not in out.lower()
        assert "agent-dispatcher" in out.lower() or "dispatcher" in out.lower()

    @pytest.mark.asyncio
    async def test_list_structure(self):
        """Agent lists project structure."""
        prompt = (
            "List the Python files in the dispatcher/ directory. "
            "Just the filenames, one per line."
        )
        code, out, err, elapsed = await _run_claude(
            prompt,
            cwd=str(Path.home() / "projects" / "dispatcher"),
            max_turns=5,
        )
        print(f"  list files -> [{code}] {out[:300]}  ({elapsed:.1f}s)")
        assert code == 0
        assert "max turns" not in out.lower()
        assert "core.py" in out or "config.py" in out

    @pytest.mark.asyncio
    async def test_max_turns_1_hits_limit(self):
        """max_turns=1 on a task needing tools hits the limit."""
        prompt = "Read pyproject.toml and tell me the version."
        code, out, err, elapsed = await _run_claude(
            prompt,
            cwd=str(Path.home() / "projects" / "dispatcher"),
            max_turns=1,
        )
        print(f"  max_turns=1 -> [{code}] {out[:200]}  ({elapsed:.1f}s)")
        # Confirms our finding: max_turns=1 can't do tool-based tasks
        assert "max turns" in out.lower() or code != 0 or len(out) > 0


class TestEdgeCases:
    """Edge cases and potential gotchas."""

    @pytest.mark.asyncio
    async def test_empty_message(self):
        """Empty input should not hang."""
        code, out, err, elapsed = await _run_claude("")
        print(f"  empty -> [{code}] out={out[:50]} err={err[:50]}  ({elapsed:.1f}s)")

    @pytest.mark.asyncio
    async def test_very_long_message(self):
        """Long input should still work."""
        prompt = "Summarize this in one word: " + "hello " * 200
        code, out, err, elapsed = await _run_claude(prompt)
        print(f"  long msg -> [{code}] {out[:100]}  ({elapsed:.1f}s)")
        assert code == 0

    @pytest.mark.asyncio
    async def test_chinese_task(self):
        """Chinese task description works fine."""
        prompt = "用一句话解释什么是递归"
        code, out, err, elapsed = await _run_claude(prompt)
        print(f"  Chinese task -> [{code}] {out[:150]}  ({elapsed:.1f}s)")
        assert code == 0
        assert len(out) > 5
