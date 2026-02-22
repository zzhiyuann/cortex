"""Health check — verify Cortex wiring and component status."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from cortex_cli.detect import detect_all
from cortex_cli.process import is_running, read_pid, log_file, log_size
from cortex_cli.setup import MCP_CONFIG, CLAUDE_SETTINGS, DISPATCHER_CONFIG, VIBE_REPLAY_DIR

console = Console()


def run_health():
    """Run full health check across all Cortex components."""
    console.print()
    console.print("[bold]Cortex Health Check[/bold]")
    console.print()

    checks: list[tuple[str, str, str]] = []  # (name, status, detail)

    # 1. Component detection
    components = detect_all()
    for name, comp in components.items():
        if comp.installed:
            checks.append((f"Component: {name}", "ok", str(comp.project_dir)))
        elif comp.project_dir:
            checks.append((f"Component: {name}", "warn", f"Found but no venv: {comp.project_dir}"))
        else:
            checks.append((f"Component: {name}", "skip", "Not installed"))

    # 2. MCP configuration
    if MCP_CONFIG.exists():
        try:
            config = json.loads(MCP_CONFIG.read_text())
            servers = config.get("mcpServers", {})
            for mcp_name in ("forge", "a2a-hub"):
                if mcp_name in servers:
                    cmd = servers[mcp_name].get("command", "?")
                    cmd_path = Path(cmd)
                    if cmd_path.exists():
                        checks.append((f"MCP: {mcp_name}", "ok", f"Configured → {cmd}"))
                    else:
                        checks.append((f"MCP: {mcp_name}", "fail", f"Binary missing: {cmd}"))
                else:
                    checks.append((f"MCP: {mcp_name}", "skip", "Not configured"))
        except (json.JSONDecodeError, OSError) as e:
            checks.append(("MCP config", "fail", f"Cannot read ~/.mcp.json: {e}"))
    else:
        checks.append(("MCP config", "skip", "~/.mcp.json not found"))

    # 3. Vibe Replay hooks
    if CLAUDE_SETTINGS.exists():
        try:
            settings = json.loads(CLAUDE_SETTINGS.read_text())
            hooks = settings.get("hooks", {})
            found_capture = False
            found_stop = False
            for entry in hooks.get("PostToolUse", []):
                for h in entry.get("hooks", []):
                    if "vibe-replay" in h.get("command", ""):
                        found_capture = True
            for entry in hooks.get("Stop", []):
                for h in entry.get("hooks", []):
                    if "vibe-replay" in h.get("command", ""):
                        found_stop = True
            if found_capture and found_stop:
                checks.append(("Hooks: Vibe Replay", "ok", "Capture + Stop hooks installed"))
            elif found_capture:
                checks.append(("Hooks: Vibe Replay", "warn", "Only capture hook found (missing stop)"))
            else:
                checks.append(("Hooks: Vibe Replay", "skip", "Not installed"))
        except (json.JSONDecodeError, OSError) as e:
            checks.append(("Hooks", "fail", f"Cannot read settings.json: {e}"))
    else:
        checks.append(("Hooks", "skip", "~/.claude/settings.json not found"))

    # 4. Hook scripts exist and are executable
    for script_name in ("capture-hook.py", "stop-hook.py"):
        script = VIBE_REPLAY_DIR / script_name
        if script.exists():
            import os
            if os.access(script, os.X_OK):
                checks.append((f"Script: {script_name}", "ok", str(script)))
            else:
                checks.append((f"Script: {script_name}", "warn", "Exists but not executable"))
        else:
            checks.append((f"Script: {script_name}", "skip", "Not found"))

    # 5. Dispatcher config
    if DISPATCHER_CONFIG.exists():
        checks.append(("Config: Dispatcher", "ok", str(DISPATCHER_CONFIG)))
    else:
        checks.append(("Config: Dispatcher", "skip", "Not configured"))

    # 6. Services running
    for svc_name, svc_label in [("a2a-hub", "A2A Hub"), ("dispatcher", "Dispatcher")]:
        pid = read_pid(svc_name)
        if pid:
            size = log_size(svc_name)
            checks.append((f"Service: {svc_label}", "ok", f"Running (pid {pid}, log {size:,}b)"))
        else:
            checks.append((f"Service: {svc_label}", "skip", "Not running"))

    # 7. Session data
    sessions_dir = VIBE_REPLAY_DIR / "sessions"
    if sessions_dir.exists():
        sessions = [d for d in sessions_dir.iterdir() if d.is_dir()]
        checks.append(("Data: Sessions", "ok", f"{len(sessions)} sessions captured"))
    else:
        checks.append(("Data: Sessions", "skip", "No sessions directory"))

    # Print results
    table = Table(show_header=True, title="Health Report")
    table.add_column("Check", style="bold", min_width=25)
    table.add_column("Status", min_width=6)
    table.add_column("Detail")

    status_icons = {
        "ok": "[green]OK[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
        "skip": "[dim]--[/dim]",
    }

    ok_count = sum(1 for _, s, _ in checks if s == "ok")
    warn_count = sum(1 for _, s, _ in checks if s == "warn")
    fail_count = sum(1 for _, s, _ in checks if s == "fail")

    for name, status, detail in checks:
        table.add_row(name, status_icons[status], detail)

    console.print(table)
    console.print()

    if fail_count:
        console.print(f"[red]{fail_count} issue(s) need attention.[/red]")
        console.print("[dim]Run 'cortex init' to fix configuration issues.[/dim]")
    elif warn_count:
        console.print(f"[yellow]{warn_count} warning(s), {ok_count} checks passed.[/yellow]")
    else:
        console.print(f"[green]All {ok_count} checks passed.[/green]")

    console.print()
