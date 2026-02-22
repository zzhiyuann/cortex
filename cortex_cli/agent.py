"""Cortex Autonomous Agent — manages long-running Claude Code improvement sessions.

cortex agent start  → launch autonomous agent with task queue
cortex agent stop   → gracefully stop the agent
cortex agent status → check agent state, progress, and logs
cortex agent log    → tail the agent log
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cortex_cli.process import read_pid, write_pid, stop_process, log_size, tail_log

console = Console()

AGENT_NAME = "agent"
AGENT_DIR = Path.home() / ".cortex" / "agent"
STATE_FILE = AGENT_DIR / "state.json"
TASKS_FILE = AGENT_DIR / "tasks.json"
PROMPT_FILE = AGENT_DIR / "prompt.md"
# Agent uses its own log/pid under agent/ dir for isolation
LOG_FILE = AGENT_DIR / "agent.log"
PID_FILE = AGENT_DIR / "agent.pid"


def _ensure_dir():
    AGENT_DIR.mkdir(parents=True, exist_ok=True)


def _read_pid() -> int | None:
    """Read agent PID from its dedicated PID file."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def _save_state(state: dict):
    _ensure_dir()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _default_tasks() -> list[dict]:
    """Default task queue for Cortex improvement."""
    return [
        {
            "id": 1,
            "title": "System-level Cortex architecture improvements",
            "description": (
                "Review and improve the entire Cortex ecosystem from a systems perspective: "
                "inter-project communication, shared config management, error handling across "
                "components, unified logging, health checks, dependency management. "
                "Make the stack more robust and cohesive as a whole."
            ),
            "project": "cortex",
            "status": "pending",
        },
        {
            "id": 2,
            "title": "Improve Vibe Replay HTML template",
            "description": "Polish visual design, improve mobile responsiveness, add micro-interactions",
            "project": "vibe-replay",
            "status": "pending",
        },
        {
            "id": 3,
            "title": "Enhance Vibe Replay analysis quality",
            "description": "Better insights, smarter phase detection, more meaningful summaries",
            "project": "vibe-replay",
            "status": "pending",
        },
        {
            "id": 4,
            "title": "Fix bugs across Cortex projects",
            "description": "Run tests on all projects, fix any failures, improve test coverage",
            "project": "cortex",
            "status": "pending",
        },
        {
            "id": 5,
            "title": "Polish Cortex website and documentation",
            "description": "Improve index.html, project pages, docs page. Fix broken links, improve UX",
            "project": "cortex",
            "status": "pending",
        },
    ]


def _build_agent_prompt(tasks: list[dict], custom_prompt: str | None = None) -> str:
    """Build the autonomous agent prompt with task queue and feedback issues."""
    pending_tasks = [t for t in tasks if t["status"] == "pending"]
    task_list = "\n".join(
        f"  {t['id']}. [{t['project']}] {t['title']}: {t['description']}"
        for t in pending_tasks
    )

    # Include open feedback issues from user interactions
    feedback_section = ""
    try:
        from cortex_cli.feedback import build_improvement_prompt
        feedback_section = build_improvement_prompt()
    except Exception:
        pass

    prompt = f"""You are the Cortex Autonomous Improvement Agent. Work through the task queue systematically.

## First Steps
1. Read /Users/zwang/projects/cortex/CLAUDE.md for project instructions
2. Send a Telegram message (in Chinese) announcing you've started
3. Begin working through tasks in order

## Task Queue
{task_list}

## Rules
- After completing each task: run tests, commit changes, update progress
- Send Telegram update after every 2-3 tasks completed
- If you hit rate limits, send a summary and stop gracefully
- If stuck on a task for too long, skip it and move on
- Keep code clean, don't over-engineer
- Auto commit + push after each meaningful change

## Telegram (Chinese only)
curl -s -X POST "https://api.telegram.org/botREDACTED_BOT_TOKEN/sendMessage" \\
  -H 'Content-Type: application/json' \\
  -d "$(python3 -c "import json; print(json.dumps({{'chat_id': REDACTED_CHAT_ID, 'text': '你的消息'}}))\")"

## Testing
/Users/zwang/projects/vibe-replay/.venv/bin/python3 -m pytest /Users/zwang/projects/vibe-replay/tests/ -v
/Users/zwang/projects/forge/.venv/bin/python3 -m pytest /Users/zwang/projects/forge/tests/ -v

## State File
Write progress to {STATE_FILE} as JSON after each task:
python3 -c "import json; json.dump({{'current_task': N, 'completed': [...], 'status': 'running'}}, open('{STATE_FILE}', 'w'))"
"""

    if feedback_section:
        prompt += f"\n{feedback_section}\n"

    if custom_prompt:
        prompt += f"\n## Additional Instructions\n{custom_prompt}\n"

    prompt += "\nGo. Ship it."
    return prompt


def start_agent(
    max_turns: int = 50,
    custom_prompt: str | None = None,
    tasks: list[dict] | None = None,
):
    """Start the autonomous agent."""
    _ensure_dir()

    # Check if already running
    pid = _read_pid()
    if pid:
        console.print(f"[yellow]Agent already running (pid {pid})[/yellow]")
        console.print(f"[dim]Log: {LOG_FILE}[/dim]")
        return

    # Prepare tasks
    task_list = tasks or _default_tasks()
    TASKS_FILE.write_text(json.dumps(task_list, indent=2))

    # Build and save prompt
    prompt = _build_agent_prompt(task_list, custom_prompt)
    PROMPT_FILE.write_text(prompt)

    # Save initial state
    _save_state({
        "status": "starting",
        "started_at": datetime.now().isoformat(),
        "max_turns": max_turns,
        "tasks_total": len(task_list),
        "tasks_completed": 0,
        "current_task": None,
    })

    # Launch Claude Code in background
    env = os.environ.copy()
    # Remove vars that cause nesting detection
    for var in ("CLAUDE_CODE", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(var, None)

    with open(LOG_FILE, "w") as log:
        proc = subprocess.Popen(
            [
                "claude", "-p",
                "--dangerously-skip-permissions",
                "--max-turns", str(max_turns),
                prompt,
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
            env=env,
            cwd="/Users/zwang/projects/cortex",
        )

    PID_FILE.write_text(str(proc.pid))

    _save_state({
        "status": "running",
        "pid": proc.pid,
        "started_at": datetime.now().isoformat(),
        "max_turns": max_turns,
        "tasks_total": len(task_list),
        "tasks_completed": 0,
    })

    console.print(Panel.fit(
        f"[bold green]Agent started![/bold green]\n\n"
        f"  PID:       {proc.pid}\n"
        f"  Max turns: {max_turns}\n"
        f"  Tasks:     {len(task_list)}\n"
        f"  Log:       {LOG_FILE}\n"
        f"  State:     {STATE_FILE}\n\n"
        f"[dim]Use 'cortex agent status' to check progress\n"
        f"Use 'cortex agent log' to tail output\n"
        f"Use 'cortex agent stop' to stop[/dim]",
        title="Cortex Agent",
        border_style="green",
    ))


def stop_agent():
    """Stop the running agent."""
    pid = _read_pid()
    if not pid:
        console.print("[dim]No agent running.[/dim]")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(15):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        console.print(f"[green]Agent stopped[/green] (pid {pid})")
    except ProcessLookupError:
        console.print("[dim]Agent was already stopped.[/dim]")

    PID_FILE.unlink(missing_ok=True)

    state = _load_state()
    state["status"] = "stopped"
    state["stopped_at"] = datetime.now().isoformat()
    _save_state(state)


def agent_status():
    """Show agent status."""
    pid = _read_pid()
    state = _load_state()

    table = Table(title="Cortex Agent Status", show_header=False)
    table.add_column("Key", style="dim")
    table.add_column("Value")

    if pid:
        table.add_row("Status", "[bold green]Running[/bold green]")
        table.add_row("PID", str(pid))
    else:
        table.add_row("Status", "[dim]Stopped[/dim]")

    if state:
        if state.get("started_at"):
            table.add_row("Started", state["started_at"])
        if state.get("stopped_at"):
            table.add_row("Stopped", state["stopped_at"])
        table.add_row("Max Turns", str(state.get("max_turns", "?")))
        table.add_row("Tasks Completed",
                       f"{state.get('tasks_completed', 0)} / {state.get('tasks_total', '?')}")
        if state.get("current_task"):
            table.add_row("Current Task", str(state["current_task"]))

    table.add_row("Log", str(LOG_FILE))
    table.add_row("State", str(STATE_FILE))

    console.print()
    console.print(table)

    # Show log tail
    if LOG_FILE.exists():
        log_size = LOG_FILE.stat().st_size
        table.add_row("Log Size", f"{log_size:,} bytes")
        console.print(f"\n[dim]Last 10 lines of log:[/dim]")
        lines = LOG_FILE.read_text().strip().split("\n")
        for line in lines[-10:]:
            console.print(f"  [dim]{line[:120]}[/dim]")

    console.print()


def agent_log(lines: int = 50):
    """Show agent log."""
    if not LOG_FILE.exists():
        console.print("[dim]No log file found.[/dim]")
        return

    content = LOG_FILE.read_text().strip().split("\n")
    for line in content[-lines:]:
        console.print(line)
