"""Complex task integration tests -- real claude -p with project context.

These simulate what the dispatcher actually sends: wrapped prompts with
working directory, memory context, and real project tasks that need
multiple tool calls.

Run with: pytest tests/test_complex_tasks.py -v -s
"""

import asyncio
import os
import time
from pathlib import Path

import pytest

CLAUDE_BIN = str(Path.home() / ".local" / "bin" / "claude")
DISPATCHER_DIR = str(Path.home() / "projects" / "dispatcher")
TIMEOUT = 120

MEMORY = (
    "## User Preferences\n"
    "- Communicate in Chinese\n"
    "- Code/comments/commits in English\n\n"
    "## Communication Style\n"
    "- Natural, conversational\n"
)


def _clean_env():
    env = os.environ.copy()
    for key in ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(key, None)
    return env


def _wrap_prompt(text, cwd):
    """Same prompt the dispatcher builds."""
    project = Path(cwd).name
    return (
        f"Working directory: {cwd}  (project: {project})\n\n"
        f"User context:\n{MEMORY}\n\n"
        f"User says: {text}\n\n"
        "Do what the user asks. Summarize the result concisely. "
        "IMPORTANT: Do NOT send any Telegram messages yourself (no curl to "
        "Telegram API). Your stdout will be relayed to the user automatically."
    )


async def _run_task(text, cwd=DISPATCHER_DIR, max_turns=15):
    """Run a wrapped dispatcher prompt through claude -p."""
    prompt = _wrap_prompt(text, cwd)
    cmd = [CLAUDE_BIN, "-p", "--dangerously-skip-permissions",
           "--max-turns", str(max_turns)]
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
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


def _report(label, code, out, elapsed):
    """Print a human-readable test result."""
    preview = out[:300].replace("\n", " | ")
    print(f"\n  [{label}]")
    print(f"  exit={code}  time={elapsed:.1f}s  len={len(out)}")
    print(f"  output: {preview}")
    if elapsed > 60:
        print(f"  WARNING: took over 60s")


class TestComplexTasks:
    """10 complex tasks that require reading files and reasoning."""

    @pytest.mark.asyncio
    async def test_01_explain_architecture(self):
        """Explain overall project architecture from code."""
        code, out, err, elapsed = await _run_task(
            "dispatcher这个项目的整体架构是什么？模块之间怎么协作的？"
        )
        _report("explain architecture", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 50

    @pytest.mark.asyncio
    async def test_02_find_bug(self):
        """Find a real bug in core.py."""
        code, out, err, elapsed = await _run_task(
            "看看 core.py 的 _handle_status 方法，有没有什么 bug？"
        )
        _report("find bug", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 20

    @pytest.mark.asyncio
    async def test_03_compare_modules(self):
        """Compare two modules and explain differences."""
        code, out, err, elapsed = await _run_task(
            "对比 runner.py 和 session.py，各自负责什么？有耦合吗？"
        )
        _report("compare modules", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 50

    @pytest.mark.asyncio
    async def test_04_config_flow(self):
        """Trace config loading from CLI to core."""
        code, out, err, elapsed = await _run_task(
            "从用户运行 dispatcher start 到 Dispatcher 类初始化，"
            "config 是怎么一步步加载传递的？"
        )
        _report("config flow", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 50

    @pytest.mark.asyncio
    async def test_05_count_lines(self):
        """Count lines of code per module."""
        code, out, err, elapsed = await _run_task(
            "统计 dispatcher/ 目录下每个 .py 文件的行数，列个表"
        )
        _report("count lines", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert "core" in out.lower()

    @pytest.mark.asyncio
    async def test_06_test_coverage_analysis(self):
        """Analyze what the tests cover."""
        code, out, err, elapsed = await _run_task(
            "看看 tests/ 下的测试，覆盖了哪些功能？有什么重要的没测到？"
        )
        _report("test coverage", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 50

    @pytest.mark.asyncio
    async def test_07_security_review(self):
        """Review for security issues."""
        code, out, err, elapsed = await _run_task(
            "从安全角度审查一下这个项目。token 处理、输入校验、"
            "有没有什么安全风险？"
        )
        _report("security review", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 30

    @pytest.mark.asyncio
    async def test_08_error_handling(self):
        """Analyze error handling patterns."""
        code, out, err, elapsed = await _run_task(
            "项目里的错误处理做得怎么样？有没有吞异常或者该 catch 没 catch 的地方？"
        )
        _report("error handling", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 30

    @pytest.mark.asyncio
    async def test_09_suggest_improvement(self):
        """Suggest a concrete improvement."""
        code, out, err, elapsed = await _run_task(
            "如果只能改一个地方来提升这个项目的可靠性，你会改什么？为什么？"
        )
        _report("suggest improvement", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 30

    @pytest.mark.asyncio
    async def test_10_cross_project(self):
        """Task in a different project directory."""
        visa_dir = str(Path.home() / "Documents" / "spain_visa")
        code, out, err, elapsed = await _run_task(
            "这个项目是干什么的？有什么文件？简单说说",
            cwd=visa_dir,
        )
        _report("cross-project", code, out, elapsed)
        assert code == 0
        assert "max turns" not in out.lower()
        assert len(out) > 20
