"""Cortex init — one command to wire everything together.

Handles:
- Forge → MCP config
- A2A Hub → MCP config
- Vibe Replay → Claude Code hooks + hook scripts
- Dispatcher → YAML config with Telegram credentials
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cortex_cli.detect import Component, detect_all

console = Console()

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
MCP_CONFIG = Path.home() / ".mcp.json"
DISPATCHER_CONFIG = Path.home() / ".config" / "dispatcher" / "config.yaml"
VIBE_REPLAY_DIR = Path.home() / ".vibe-replay"


def run_init(
    telegram_token: str | None = None,
    telegram_chat_id: int | None = None,
    skip_dispatcher: bool = False,
    skip_forge: bool = False,
    skip_a2a: bool = False,
    skip_replay: bool = False,
):
    """Run the full init sequence."""
    console.print()
    console.print(
        Panel(
            "[bold]Cortex Init[/bold]\n"
            "[dim]Detecting components and wiring everything together...[/dim]",
            border_style="blue",
        )
    )
    console.print()

    # Step 1: Detect components
    components = detect_all()
    _print_detection(components)

    found = {k: v for k, v in components.items() if v.installed}
    if not found:
        console.print("[red]No Cortex components found.[/red]")
        console.print("Install at least one component first:")
        console.print("  pip install forge-agent")
        console.print("  pip install a2a-hub")
        console.print("  pip install vibe-replay")
        console.print("  pip install agent-dispatcher")
        return

    # Step 2: Configure MCP servers
    if not skip_forge and "forge" in found:
        _setup_mcp_forge(found["forge"])
    if not skip_a2a and "a2a-hub" in found:
        _setup_mcp_a2a(found["a2a-hub"])

    # Step 3: Install Vibe Replay hooks
    if not skip_replay and "vibe-replay" in found:
        _setup_vibe_replay(found["vibe-replay"])

    # Step 4: Configure Dispatcher
    if not skip_dispatcher and "dispatcher" in found:
        _setup_dispatcher(found["dispatcher"], telegram_token, telegram_chat_id)

    # Done
    console.print()
    console.print(
        Panel(
            "[bold green]Cortex initialized.[/bold green]\n\n"
            "[dim]Run [bold]cortex status[/bold] to verify.\n"
            "Run [bold]cortex start[/bold] to bring up services.\n"
            "Restart Claude Code to load new MCP servers.[/dim]",
            border_style="green",
        )
    )


def _print_detection(components: dict[str, Component]):
    """Print detection results."""
    table = Table(title="Component Detection", show_header=True)
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Location")

    for name, comp in components.items():
        if comp.installed:
            status = "[green]Found[/green]"
            loc = str(comp.project_dir)
        elif comp.project_dir:
            status = "[yellow]Found (no venv)[/yellow]"
            loc = str(comp.project_dir)
        else:
            status = "[dim]Not found[/dim]"
            loc = "-"
        table.add_row(name, status, loc)

    console.print(table)
    console.print()


# ---- MCP Configuration ----

def _load_mcp_config() -> dict:
    """Load existing MCP config."""
    if MCP_CONFIG.exists():
        try:
            return json.loads(MCP_CONFIG.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"mcpServers": {}}


def _save_mcp_config(config: dict):
    """Save MCP config."""
    MCP_CONFIG.write_text(json.dumps(config, indent=2) + "\n")


def _setup_mcp_forge(comp: Component):
    """Add Forge MCP server to config."""
    config = _load_mcp_config()
    servers = config.setdefault("mcpServers", {})

    if "forge" in servers:
        console.print("[dim]Forge MCP already configured, updating...[/dim]")

    mcp_path = comp.mcp_path or comp.venv_dir / "bin" / "forge-mcp"
    servers["forge"] = {"command": str(mcp_path)}
    _save_mcp_config(config)
    console.print("[green]+ Forge MCP[/green] → added to ~/.mcp.json")


def _setup_mcp_a2a(comp: Component):
    """Add A2A Hub MCP bridge to config."""
    config = _load_mcp_config()
    servers = config.setdefault("mcpServers", {})

    if "a2a-hub" in servers:
        console.print("[dim]A2A Hub MCP already configured, updating...[/dim]")

    cli_path = comp.cli_path or comp.venv_dir / "bin" / "a2a-hub"
    mcp_args = comp.extras.get("mcp_args", ["bridge"])
    servers["a2a-hub"] = {"command": str(cli_path), "args": mcp_args}
    _save_mcp_config(config)
    console.print("[green]+ A2A Hub MCP[/green] → added to ~/.mcp.json")


# ---- Vibe Replay Hooks ----

CAPTURE_HOOK_SCRIPT = '''#!/usr/bin/env python3
"""Vibe Replay capture hook — appends tool events to session log."""

import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    try:
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            return

        data = json.loads(stdin_data)
        session_id = data.get("session_id", "unknown")
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})
        tool_output = data.get("tool_result") or data.get("tool_output")

        code_tools = {"Edit", "Write", "NotebookEdit"}
        event_type = "code_change" if tool_name in code_tools else "tool_call"

        summary = _summarize(tool_name, tool_input)

        files = []
        for field in ("file_path", "path", "notebook_path"):
            val = tool_input.get(field)
            if val:
                files.append(val)

        code_diff = None
        if tool_name == "Edit":
            old = tool_input.get("old_string", "")
            new = tool_input.get("new_string", "")
            if old or new:
                code_diff = f"--- old\\n+++ new\\n-{old}\\n+{new}"
        elif tool_name == "Write":
            content = tool_input.get("content", "")
            if len(content) > 3000:
                content = content[:3000] + "..."
            code_diff = f"+++ new file\\n{content}"

        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "tool_name": tool_name,
            "summary": summary,
            "details": _safe_details(tool_input, tool_output),
            "code_diff": code_diff,
            "files_affected": files,
        }

        sessions_dir = Path.home() / ".vibe-replay" / "sessions" / session_id
        sessions_dir.mkdir(parents=True, exist_ok=True)
        with open(sessions_dir / "events.jsonl", "a") as f:
            f.write(json.dumps(event, default=str) + "\\n")

    except Exception:
        try:
            err = Path.home() / ".vibe-replay" / "capture-errors.log"
            with open(err, "a") as f:
                import traceback
                f.write(f"{datetime.now().isoformat()} {traceback.format_exc()}\\n")
        except Exception:
            pass


def _summarize(tool_name, tool_input):
    if not tool_input:
        return f"Called {tool_name}"
    if tool_name == "Edit":
        return f"Edited {Path(tool_input.get('file_path', '?')).name}"
    elif tool_name == "Write":
        return f"Wrote {Path(tool_input.get('file_path', '?')).name}"
    elif tool_name == "Read":
        return f"Read {Path(tool_input.get('file_path', '?')).name}"
    elif tool_name == "Bash":
        return f"Ran: {tool_input.get('command', '')[:100]}"
    elif tool_name == "Glob":
        return f"Searched files: {tool_input.get('pattern', '?')}"
    elif tool_name == "Grep":
        return f"Searched content: {tool_input.get('pattern', '?')}"
    elif tool_name == "WebSearch":
        return f"Web search: {tool_input.get('query', '?')}"
    return f"Called {tool_name}"


def _safe_details(tool_input, tool_output):
    details = {}
    if tool_input:
        inp = json.dumps(tool_input, default=str)
        details["input"] = {"_truncated": True, "_size": len(inp)} if len(inp) > 10000 else tool_input
    if tool_output is not None:
        out = str(tool_output)
        details["output"] = out[:5000] + "..." if len(out) > 5000 else tool_output
    return details


if __name__ == "__main__":
    main()
'''

STOP_HOOK_SCRIPT = '''#!/usr/bin/env python3
"""Vibe Replay stop hook — finalizes session metadata."""

import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    try:
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            return

        data = json.loads(stdin_data)
        session_id = data.get("session_id", "unknown")

        sessions_dir = Path.home() / ".vibe-replay" / "sessions" / session_id
        if not sessions_dir.exists():
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "event_type": "session_end",
            "tool_name": None,
            "summary": "Session ended",
            "details": {},
            "code_diff": None,
            "files_affected": [],
        }

        events_file = sessions_dir / "events.jsonl"
        with open(events_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\\n")

        event_count = 0
        first_ts = last_ts = None
        files_seen = set()
        tools_used = {}

        with open(events_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    event_count += 1
                    ts = ev.get("timestamp", "")
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
                    for fp in ev.get("files_affected", []):
                        files_seen.add(fp)
                    tn = ev.get("tool_name")
                    if tn:
                        tools_used[tn] = tools_used.get(tn, 0) + 1
                except Exception:
                    continue

        metadata = {
            "session_id": session_id,
            "project": "", "project_path": "",
            "start_time": first_ts, "end_time": last_ts,
            "event_count": event_count,
            "summary": "", "tags": [],
            "files_modified": sorted(files_seen),
            "tools_used": tools_used,
        }

        with open(sessions_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    except Exception:
        try:
            err = Path.home() / ".vibe-replay" / "capture-errors.log"
            with open(err, "a") as f:
                import traceback
                f.write(f"{datetime.now().isoformat()} STOP: {traceback.format_exc()}\\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
'''

HOOK_MARKER = "vibe-replay"


def _setup_vibe_replay(comp: Component):
    """Install Vibe Replay hooks into Claude Code."""
    # Write hook scripts
    VIBE_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    (VIBE_REPLAY_DIR / "sessions").mkdir(exist_ok=True)

    capture_path = VIBE_REPLAY_DIR / "capture-hook.py"
    stop_path = VIBE_REPLAY_DIR / "stop-hook.py"

    capture_path.write_text(CAPTURE_HOOK_SCRIPT)
    capture_path.chmod(capture_path.stat().st_mode | stat.S_IEXEC)

    stop_path.write_text(STOP_HOOK_SCRIPT)
    stop_path.chmod(stop_path.stat().st_mode | stat.S_IEXEC)

    # Update Claude Code settings
    settings = {}
    if CLAUDE_SETTINGS.exists():
        try:
            settings = json.loads(CLAUDE_SETTINGS.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Backup
    if CLAUDE_SETTINGS.exists():
        backup = CLAUDE_SETTINGS.with_suffix(".json.bak")
        shutil.copy2(CLAUDE_SETTINGS, backup)

    hooks = settings.setdefault("hooks", {})

    # Add PostToolUse hook (remove existing vibe-replay ones first)
    ptu = hooks.get("PostToolUse", [])
    ptu = [e for e in ptu if not any(HOOK_MARKER in h.get("command", "") for h in e.get("hooks", []))]
    ptu.append({
        "matcher": "",
        "hooks": [{"type": "command", "command": f"python3 {capture_path}  # {HOOK_MARKER}"}],
    })
    hooks["PostToolUse"] = ptu

    # Add Stop hook (preserve existing non-vibe-replay ones)
    stop = hooks.get("Stop", [])
    stop = [e for e in stop if not any(HOOK_MARKER in h.get("command", "") for h in e.get("hooks", []))]
    stop.append({
        "matcher": "",
        "hooks": [{"type": "command", "command": f"python3 {stop_path}  # {HOOK_MARKER}"}],
    })
    hooks["Stop"] = stop

    CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    CLAUDE_SETTINGS.write_text(json.dumps(settings, indent=2))

    console.print("[green]+ Vibe Replay hooks[/green] → installed in Claude Code settings")
    console.print(f"  [dim]capture: {capture_path}[/dim]")
    console.print(f"  [dim]stop:    {stop_path}[/dim]")


# ---- Dispatcher Configuration ----

DISPATCHER_DEFAULT_CONFIG = """\
telegram:
  bot_token: "{token}"
  chat_id: {chat_id}

agent:
  command: "claude"
  args: ["-p", "--dangerously-skip-permissions"]
  max_concurrent: 3
  timeout: 1800
  max_turns: 50
  max_turns_chat: 10
  max_turns_followup: 20

behavior:
  poll_timeout: 30
  progress_interval: 180
  recent_window: 300
  cancel_keywords: ["cancel", "stop"]
  status_keywords: ["status"]

projects: {{}}
"""


def _setup_dispatcher(
    comp: Component,
    telegram_token: str | None = None,
    telegram_chat_id: int | None = None,
):
    """Configure Dispatcher with Telegram credentials."""
    DISPATCHER_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    if DISPATCHER_CONFIG.exists():
        console.print("[dim]Dispatcher config already exists, preserving.[/dim]")
        console.print(f"[green]+ Dispatcher[/green] → {DISPATCHER_CONFIG}")
        return

    # Need credentials
    if not telegram_token:
        telegram_token = click.prompt(
            "Telegram bot token (from @BotFather)",
            type=str,
        )
    if not telegram_chat_id:
        telegram_chat_id = click.prompt(
            "Your Telegram chat ID",
            type=int,
        )

    config_text = DISPATCHER_DEFAULT_CONFIG.format(
        token=telegram_token,
        chat_id=telegram_chat_id,
    )
    DISPATCHER_CONFIG.write_text(config_text)
    console.print(f"[green]+ Dispatcher config[/green] → {DISPATCHER_CONFIG}")


# ---- Status ----

def run_status():
    """Show full Cortex status."""
    console.print()
    console.print("[bold]Cortex Status[/bold]")
    console.print()

    # Detection
    components = detect_all()
    _print_detection(components)

    # MCP config
    console.print("[bold]MCP Servers[/bold] (~/.mcp.json)")
    if MCP_CONFIG.exists():
        try:
            config = json.loads(MCP_CONFIG.read_text())
            servers = config.get("mcpServers", {})
            cortex_servers = {k: v for k, v in servers.items() if k in ("forge", "a2a-hub")}
            if cortex_servers:
                for name, cfg in cortex_servers.items():
                    cmd = cfg.get("command", "?")
                    console.print(f"  [green]{name}[/green] → {cmd}")
            else:
                console.print("  [yellow]No Cortex MCP servers configured[/yellow]")
        except Exception:
            console.print("  [red]Error reading MCP config[/red]")
    else:
        console.print("  [dim]No MCP config found[/dim]")

    # Hooks
    console.print()
    console.print("[bold]Vibe Replay Hooks[/bold] (~/.claude/settings.json)")
    if CLAUDE_SETTINGS.exists():
        try:
            settings = json.loads(CLAUDE_SETTINGS.read_text())
            hooks = settings.get("hooks", {})
            vr_hooks = []
            for hook_type, entries in hooks.items():
                for entry in entries:
                    for h in entry.get("hooks", []):
                        if HOOK_MARKER in h.get("command", ""):
                            vr_hooks.append(f"  [green]{hook_type}[/green] → {h['command'][:60]}")
            if vr_hooks:
                for line in vr_hooks:
                    console.print(line)
            else:
                console.print("  [yellow]Not installed[/yellow]")
        except Exception:
            console.print("  [red]Error reading settings[/red]")

    # Dispatcher
    console.print()
    console.print("[bold]Dispatcher[/bold]")
    if DISPATCHER_CONFIG.exists():
        console.print(f"  [green]Config[/green] → {DISPATCHER_CONFIG}")
        # Check if running
        try:
            result = subprocess.run(
                ["pgrep", "-f", "dispatcher start"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                console.print("  [green]Status[/green] → Running")
            else:
                console.print("  [yellow]Status[/yellow] → Not running")
        except Exception:
            console.print("  [dim]Status[/dim] → Unknown")
    else:
        console.print("  [dim]Not configured[/dim]")

    # Session data
    console.print()
    console.print("[bold]Vibe Replay Sessions[/bold]")
    sessions_dir = VIBE_REPLAY_DIR / "sessions"
    if sessions_dir.exists():
        sessions = [d for d in sessions_dir.iterdir() if d.is_dir()]
        console.print(f"  [green]{len(sessions)}[/green] sessions captured")
    else:
        console.print("  [dim]No sessions yet[/dim]")

    console.print()
