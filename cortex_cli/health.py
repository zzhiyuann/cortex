"""Health check — verify Cortex wiring, component status, and system health."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from rich.console import Console
from rich.table import Table

from cortex_cli.detect import detect_all
from cortex_cli.process import is_running, read_pid, log_file, log_size
from cortex_cli.setup import MCP_CONFIG, CLAUDE_SETTINGS, DISPATCHER_CONFIG, VIBE_REPLAY_DIR
from cortex_cli.config import load_config, CONFIG_PATH
from cortex_cli.errors import ERROR_LOG

console = Console()


def _get_component_version(comp) -> str:
    """Get version string for a component by reading its __init__.py or pyproject.toml."""
    if not comp.project_dir:
        return ""
    # Try __init__.py first
    pkg_names = {
        "dispatcher": "dispatcher",
        "a2a-hub": "a2a_hub",
        "forge": "forge",
        "vibe-replay": "vibe_replay",
    }
    pkg = pkg_names.get(comp.name, comp.name.replace("-", "_"))
    init_file = comp.project_dir / pkg / "__init__.py"
    if init_file.exists():
        for line in init_file.read_text().split("\n"):
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # Fallback to pyproject.toml
    pyproject = comp.project_dir / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().split("\n"):
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "?"


def _dir_size_mb(path: Path) -> float:
    """Get total size of a directory in MB."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total / (1024 * 1024)


def _check_telegram_connectivity(bot_token: str) -> tuple[str, str]:
    """Check Telegram API connectivity. Returns (status, detail)."""
    if not bot_token:
        return "skip", "No bot token configured"
    try:
        req = Request(
            f"https://api.telegram.org/bot{bot_token}/getMe",
            headers={"Content-Type": "application/json"},
        )
        resp = urlopen(req, timeout=5)
        data = json.loads(resp.read())
        if data.get("ok"):
            bot_name = data.get("result", {}).get("username", "?")
            return "ok", f"Connected (bot: @{bot_name})"
        return "fail", "API returned error"
    except URLError as e:
        return "fail", f"Cannot reach Telegram: {e.reason}"
    except OSError as e:
        return "fail", f"Network error: {e}"


def _check_a2a_hub_connectivity(host: str, port: int) -> tuple[str, str]:
    """Check if A2A Hub is reachable via WebSocket handshake probe."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return "ok", f"Listening on {host}:{port}"
        return "skip", f"Not listening on {host}:{port}"
    except OSError as e:
        return "skip", f"Cannot connect: {e}"


def run_health():
    """Run full health check across all Cortex components."""
    console.print()
    console.print("[bold]Cortex Health Check[/bold]")
    console.print()

    checks: list[tuple[str, str, str]] = []  # (name, status, detail)

    # 1. Component detection + versions
    components = detect_all()
    for name, comp in components.items():
        if comp.installed:
            version = _get_component_version(comp)
            ver_str = f" v{version}" if version and version != "?" else ""
            checks.append((f"Component: {name}", "ok", f"{comp.project_dir}{ver_str}"))
        elif comp.project_dir:
            checks.append((f"Component: {name}", "warn", f"Found but no venv: {comp.project_dir}"))
        else:
            checks.append((f"Component: {name}", "skip", "Not installed"))

    # 2. Shared config
    if CONFIG_PATH.exists():
        config = load_config()
        projects = config.get("projects", {})
        checks.append(("Config: Shared", "ok", f"~/.cortex/config.yaml ({len(projects)} projects)"))
    else:
        checks.append(("Config: Shared", "warn", "Not generated (run cortex init)"))

    # 3. MCP configuration
    if MCP_CONFIG.exists():
        try:
            mcp_conf = json.loads(MCP_CONFIG.read_text())
            servers = mcp_conf.get("mcpServers", {})
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

    # 4. Vibe Replay hooks
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

    # 5. Hook scripts exist and are executable
    for script_name in ("capture-hook.py", "stop-hook.py"):
        script = VIBE_REPLAY_DIR / script_name
        if script.exists():
            if os.access(script, os.X_OK):
                checks.append((f"Script: {script_name}", "ok", str(script)))
            else:
                checks.append((f"Script: {script_name}", "warn", "Exists but not executable"))
        else:
            checks.append((f"Script: {script_name}", "skip", "Not found"))

    # 6. Dispatcher config
    if DISPATCHER_CONFIG.exists():
        checks.append(("Config: Dispatcher", "ok", str(DISPATCHER_CONFIG)))
    else:
        checks.append(("Config: Dispatcher", "skip", "Not configured"))

    # 7. Services running
    for svc_name, svc_label in [("a2a-hub", "A2A Hub"), ("dispatcher", "Dispatcher")]:
        pid = read_pid(svc_name)
        if pid:
            size = log_size(svc_name)
            checks.append((f"Service: {svc_label}", "ok", f"Running (pid {pid}, log {size:,}b)"))
        else:
            checks.append((f"Service: {svc_label}", "skip", "Not running"))

    # 8. Disk usage for session data
    sessions_dir = VIBE_REPLAY_DIR / "sessions"
    if sessions_dir.exists():
        sessions = [d for d in sessions_dir.iterdir() if d.is_dir()]
        size_mb = _dir_size_mb(sessions_dir)
        checks.append(("Data: Sessions", "ok", f"{len(sessions)} sessions ({size_mb:.1f} MB)"))
    else:
        checks.append(("Data: Sessions", "skip", "No sessions directory"))

    # Forge tools disk usage
    forge_tools = Path.home() / ".forge" / "tools"
    if forge_tools.exists():
        tools = [d for d in forge_tools.iterdir() if d.is_dir()]
        size_mb = _dir_size_mb(forge_tools)
        checks.append(("Data: Forge Tools", "ok", f"{len(tools)} tools ({size_mb:.1f} MB)"))
    else:
        checks.append(("Data: Forge Tools", "skip", "No forge tools directory"))

    # Log files disk usage
    from cortex_cli.process import LOG_DIR
    total_log_size = 0
    for svc in ("a2a-hub", "dispatcher"):
        total_log_size += log_size(svc)
    if ERROR_LOG.exists():
        total_log_size += ERROR_LOG.stat().st_size
    if total_log_size > 0:
        checks.append(("Data: Logs", "ok", f"{total_log_size / 1024:.1f} KB total"))
    else:
        checks.append(("Data: Logs", "skip", "No log files"))

    # 9. Network: Telegram connectivity
    config = load_config()
    tg = config.get("telegram", {})
    bot_token = tg.get("bot_token", "")
    tg_status, tg_detail = _check_telegram_connectivity(bot_token)
    checks.append(("Network: Telegram", tg_status, tg_detail))

    # 10. Network: A2A Hub connectivity
    a2a_conf = config.get("services", {}).get("a2a_hub", {})
    a2a_host = a2a_conf.get("host", "localhost")
    a2a_port = a2a_conf.get("port", 8765)
    a2a_status, a2a_detail = _check_a2a_hub_connectivity(a2a_host, a2a_port)
    checks.append(("Network: A2A Hub", a2a_status, a2a_detail))

    # 11. Error log status
    if ERROR_LOG.exists():
        error_lines = ERROR_LOG.read_text().strip().split("\n")
        error_count = len([l for l in error_lines if l.strip()])
        if error_count > 0:
            # Check last error timestamp
            try:
                last = json.loads(error_lines[-1])
                last_ts = last.get("timestamp", "?")[:19]
                checks.append(("Errors: Log", "warn", f"{error_count} errors (last: {last_ts})"))
            except (json.JSONDecodeError, IndexError):
                checks.append(("Errors: Log", "warn", f"{error_count} entries"))
        else:
            checks.append(("Errors: Log", "ok", "Clean"))
    else:
        checks.append(("Errors: Log", "ok", "No errors recorded"))

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
