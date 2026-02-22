"""Cross-project information gathering for cortex status.

Queries each Cortex project for live stats without requiring imports
from those projects — uses filesystem inspection and subprocess calls.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cortex_cli.detect import detect_all
from cortex_cli.config import load_config

console = Console()


def get_vibe_replay_stats() -> dict:
    """Get vibe-replay session statistics from its SQLite index and filesystem."""
    sessions_dir = Path.home() / ".vibe-replay" / "sessions"
    index_db = Path.home() / ".vibe-replay" / "index.db"
    stats = {
        "total_sessions": 0,
        "total_events": 0,
        "projects": set(),
        "recent_sessions": [],
        "disk_mb": 0.0,
    }

    if not sessions_dir.exists():
        return stats

    # Count sessions from filesystem
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
    stats["total_sessions"] = len(session_dirs)

    # Disk usage
    total_bytes = 0
    for f in sessions_dir.rglob("*"):
        if f.is_file():
            total_bytes += f.stat().st_size
    stats["disk_mb"] = total_bytes / (1024 * 1024)

    # Try SQLite index for richer data
    if index_db.exists():
        try:
            conn = sqlite3.connect(str(index_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Total event count
            cursor.execute("SELECT SUM(event_count) as total FROM sessions")
            row = cursor.fetchone()
            if row and row["total"]:
                stats["total_events"] = int(row["total"])

            # Distinct projects
            cursor.execute("SELECT DISTINCT project FROM sessions WHERE project != ''")
            stats["projects"] = {row["project"] for row in cursor.fetchall()}

            # Recent sessions (last 5)
            cursor.execute(
                "SELECT session_id, project, start_time, event_count, summary "
                "FROM sessions ORDER BY start_time DESC LIMIT 5"
            )
            stats["recent_sessions"] = [dict(row) for row in cursor.fetchall()]

            conn.close()
        except (sqlite3.Error, OSError):
            pass

    return stats


def get_forge_stats() -> dict:
    """Get forge tool statistics from its filesystem storage."""
    tools_dir = Path.home() / ".forge" / "tools"
    stats = {
        "total_tools": 0,
        "tools": [],
    }

    if not tools_dir.exists():
        return stats

    for tool_dir in sorted(tools_dir.iterdir()):
        if not tool_dir.is_dir():
            continue
        meta_file = tool_dir / "metadata.json"
        tool_info = {"name": tool_dir.name}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                tool_info["type"] = meta.get("output_type", "?")
                tool_info["created"] = meta.get("created_at", "?")[:10]
                tool_info["test_status"] = "pass" if meta.get("test_passed") else "fail"
            except (json.JSONDecodeError, OSError):
                pass
        stats["tools"].append(tool_info)

    stats["total_tools"] = len(stats["tools"])
    return stats


def get_a2a_hub_stats() -> dict:
    """Get A2A Hub stats by querying its WebSocket endpoint."""
    import socket

    config = load_config()
    a2a_conf = config.get("services", {}).get("a2a_hub", {})
    host = a2a_conf.get("host", "localhost")
    port = a2a_conf.get("port", 8765)

    stats = {
        "running": False,
        "host": host,
        "port": port,
        "agents": [],
    }

    # Quick TCP check
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        stats["running"] = result == 0
    except OSError:
        pass

    return stats


def get_dispatcher_stats() -> dict:
    """Get Dispatcher stats from its config and log files."""
    import yaml

    config_path = Path.home() / ".config" / "dispatcher" / "config.yaml"
    log_path = Path.home() / ".config" / "dispatcher" / "data" / "logs" / "dispatcher.log"
    stats = {
        "configured": False,
        "projects": {},
        "log_lines": 0,
    }

    if config_path.exists():
        try:
            dconf = yaml.safe_load(config_path.read_text()) or {}
            stats["configured"] = True
            stats["projects"] = dconf.get("projects", {})
        except (yaml.YAMLError, OSError):
            pass

    if log_path.exists():
        try:
            stats["log_lines"] = sum(1 for _ in open(log_path))
        except OSError:
            pass

    return stats


def run_info():
    """Show comprehensive cross-project information."""
    console.print()
    console.print("[bold]Cortex Ecosystem Overview[/bold]")
    console.print()

    # Vibe Replay
    vr = get_vibe_replay_stats()
    vr_lines = []
    vr_lines.append(f"Sessions: {vr['total_sessions']}")
    if vr["total_events"]:
        vr_lines.append(f"Total events: {vr['total_events']:,}")
    if vr["projects"]:
        vr_lines.append(f"Projects: {', '.join(sorted(vr['projects']))}")
    if vr["disk_mb"] > 0:
        vr_lines.append(f"Disk: {vr['disk_mb']:.1f} MB")
    if vr["recent_sessions"]:
        vr_lines.append("")
        vr_lines.append("Recent:")
        for s in vr["recent_sessions"][:3]:
            proj = s.get("project") or "unknown"
            events = s.get("event_count", 0)
            ts = (s.get("start_time") or "")[:16]
            vr_lines.append(f"  {ts} {proj} ({events} events)")

    console.print(Panel(
        "\n".join(vr_lines) if vr_lines else "[dim]No data[/dim]",
        title="[bold]Vibe Replay[/bold]",
        border_style="cyan",
    ))

    # Forge
    fg = get_forge_stats()
    fg_lines = [f"Tools: {fg['total_tools']}"]
    if fg["tools"]:
        fg_lines.append("")
        for t in fg["tools"][:10]:
            status = "[green]pass[/green]" if t.get("test_status") == "pass" else "[red]fail[/red]"
            fg_lines.append(f"  {t['name']} ({t.get('type', '?')}) {status}")

    console.print(Panel(
        "\n".join(fg_lines),
        title="[bold]Forge[/bold]",
        border_style="yellow",
    ))

    # A2A Hub
    a2a = get_a2a_hub_stats()
    a2a_status = "[green]running[/green]" if a2a["running"] else "[dim]not running[/dim]"
    a2a_lines = [f"Status: {a2a_status}", f"Endpoint: {a2a['host']}:{a2a['port']}"]

    console.print(Panel(
        "\n".join(a2a_lines),
        title="[bold]A2A Hub[/bold]",
        border_style="blue",
    ))

    # Dispatcher
    dp = get_dispatcher_stats()
    dp_lines = []
    if dp["configured"]:
        dp_lines.append(f"Configured: yes ({len(dp['projects'])} project routes)")
        for name in sorted(dp["projects"].keys()):
            dp_lines.append(f"  {name}: {dp['projects'][name].get('path', '?')}")
    else:
        dp_lines.append("[dim]Not configured[/dim]")
    if dp["log_lines"]:
        dp_lines.append(f"Log: {dp['log_lines']:,} lines")

    console.print(Panel(
        "\n".join(dp_lines),
        title="[bold]Dispatcher[/bold]",
        border_style="magenta",
    ))

    console.print()
