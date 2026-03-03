"""Cortex Doctor — diagnose common issues across the ecosystem.

Runs a series of checks and provides actionable fix suggestions:
- Missing configuration files
- Port conflicts
- Stale processes
- Broken MCP wiring
- Missing dependencies
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cortex_cli.config import load_config, CONFIG_PATH, CORTEX_DIR
from cortex_cli.detect import detect_all, detect_system_python
from cortex_cli.process import read_pid, pid_file, log_file
from cortex_cli.setup import MCP_CONFIG, CLAUDE_SETTINGS, DISPATCHER_CONFIG, VIBE_REPLAY_DIR

console = Console()


def _check_port_available(host: str, port: int) -> bool:
    """Check if a port is available (not in use)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # 0 means connected = port in use
    except OSError:
        return True  # assume available on error


def _check_stale_pid(name: str) -> dict | None:
    """Check for stale PID file. Returns issue dict or None."""
    pf = pid_file(name)
    if not pf.exists():
        return None

    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, 0)
        return None  # Process is alive
    except (ValueError, ProcessLookupError, PermissionError):
        return {
            "issue": f"Stale PID file for '{name}' (pid file exists but process is dead)",
            "fix": f"rm {pf}",
            "auto_fix": lambda: pf.unlink(missing_ok=True),
        }


def _check_mcp_binary(server_name: str, config: dict) -> dict | None:
    """Check if an MCP server binary exists."""
    cmd = config.get("command", "")
    if not cmd:
        return None

    cmd_path = Path(cmd)
    if cmd_path.exists() or _which(cmd):
        return None

    return {
        "issue": f"MCP server '{server_name}' binary not found: {cmd}",
        "fix": f"Re-run 'cortex init' to reconfigure MCP servers, or install the missing package",
    }


def _which(name: str) -> str | None:
    """Find executable in PATH."""
    import shutil
    return shutil.which(name)


def run_doctor(auto_fix: bool = False):
    """Run all diagnostic checks."""
    console.print()
    console.print(Panel(
        "[bold]Cortex Doctor[/bold]\n[dim]Diagnosing common issues...[/dim]",
        border_style="blue",
    ))
    console.print()

    issues: list[dict] = []
    checks_passed = 0

    # 1. Check shared config exists
    if CONFIG_PATH.exists():
        checks_passed += 1
    else:
        issues.append({
            "issue": "Shared config (~/.cortex/config.yaml) not found",
            "fix": "Run 'cortex init' to generate configuration",
        })

    # 2. Check Python version
    py = detect_system_python()
    if py:
        checks_passed += 1
    else:
        issues.append({
            "issue": "Python >= 3.11 not found in PATH",
            "fix": "Install Python 3.11+ (brew install python@3.13 or pyenv install 3.13)",
        })

    # 3. Check for stale PID files
    for svc_name in ("a2a-hub", "dispatcher"):
        stale = _check_stale_pid(svc_name)
        if stale:
            issues.append(stale)
        else:
            checks_passed += 1

    # 4. Check port conflicts
    config = load_config()
    a2a_conf = config.get("services", {}).get("a2a_hub", {})
    a2a_port = a2a_conf.get("port", 8765)

    a2a_pid = read_pid("a2a-hub")
    if a2a_pid:
        # A2A Hub is running — port should be in use by it
        checks_passed += 1
    else:
        # A2A Hub is not running — check if something else uses the port
        if not _check_port_available("localhost", a2a_port):
            issues.append({
                "issue": f"Port {a2a_port} is in use but A2A Hub is not running",
                "fix": f"Find what's using port {a2a_port}: lsof -i :{a2a_port}",
            })
        else:
            checks_passed += 1

    # 5. Check MCP configuration
    if MCP_CONFIG.exists():
        try:
            mcp_conf = json.loads(MCP_CONFIG.read_text())
            servers = mcp_conf.get("mcpServers", {})

            for server_name in ("forge", "a2a-hub"):
                if server_name in servers:
                    binary_issue = _check_mcp_binary(server_name, servers[server_name])
                    if binary_issue:
                        issues.append(binary_issue)
                    else:
                        checks_passed += 1
        except (json.JSONDecodeError, OSError):
            issues.append({
                "issue": "~/.mcp.json is malformed (invalid JSON)",
                "fix": "Re-run 'cortex init' to regenerate MCP config",
            })
    else:
        checks_passed += 1  # Not having it is fine, just means MCP not set up

    # 6. Check Vibe Replay hooks
    if CLAUDE_SETTINGS.exists():
        try:
            settings = json.loads(CLAUDE_SETTINGS.read_text())
            hooks = settings.get("hooks", {})

            # Check capture hook
            capture_found = False
            stop_found = False
            for entry in hooks.get("PostToolUse", []):
                for h in entry.get("hooks", []):
                    if "vibe-replay" in h.get("command", ""):
                        capture_found = True
            for entry in hooks.get("Stop", []):
                for h in entry.get("hooks", []):
                    if "vibe-replay" in h.get("command", ""):
                        stop_found = True

            if capture_found and stop_found:
                # Verify the hook scripts actually exist
                for script_name in ("capture-hook.py", "stop-hook.py"):
                    script = VIBE_REPLAY_DIR / script_name
                    if not script.exists():
                        issues.append({
                            "issue": f"Vibe Replay hook script missing: {script}",
                            "fix": "Re-run 'cortex init' to reinstall hooks",
                        })
                    elif not os.access(script, os.X_OK):
                        issues.append({
                            "issue": f"Hook script not executable: {script}",
                            "fix": f"chmod +x {script}",
                            "auto_fix": lambda s=script: s.chmod(s.stat().st_mode | 0o111),
                        })
                    else:
                        checks_passed += 1
            elif capture_found or stop_found:
                issues.append({
                    "issue": "Vibe Replay has only partial hooks installed (missing capture or stop)",
                    "fix": "Re-run 'cortex init' to fix hook installation",
                })
            # If neither found, it's fine — hooks aren't required
            else:
                checks_passed += 1
        except (json.JSONDecodeError, OSError):
            issues.append({
                "issue": "~/.claude/settings.json is malformed",
                "fix": "Check the file manually or re-run 'cortex init'",
            })
    else:
        checks_passed += 1

    # 7. Check log file sizes (warn if > 10MB)
    for svc_name in ("a2a-hub", "dispatcher"):
        lf = log_file(svc_name)
        if lf.exists():
            size_mb = lf.stat().st_size / (1024 * 1024)
            if size_mb > 10:
                issues.append({
                    "issue": f"Log file for '{svc_name}' is large ({size_mb:.1f} MB)",
                    "fix": f"Truncate: > {lf}",
                    "auto_fix": lambda p=lf: p.write_text(""),
                })
            else:
                checks_passed += 1
        else:
            checks_passed += 1

    # 8. Check Dispatcher config
    if DISPATCHER_CONFIG.exists():
        try:
            import yaml
            dconf = yaml.safe_load(DISPATCHER_CONFIG.read_text()) or {}
            tg = dconf.get("telegram", {})
            if not tg.get("bot_token"):
                issues.append({
                    "issue": "Dispatcher config exists but has no Telegram bot token",
                    "fix": f"Edit {DISPATCHER_CONFIG} and add your bot token",
                })
            else:
                checks_passed += 1
        except Exception:
            issues.append({
                "issue": f"Dispatcher config is malformed: {DISPATCHER_CONFIG}",
                "fix": f"Delete and re-run 'cortex init': rm {DISPATCHER_CONFIG}",
            })
    else:
        checks_passed += 1  # Not required

    # 9. Check components detected
    components = detect_all()
    installed = {name for name, comp in components.items() if comp.installed}
    if not installed:
        issues.append({
            "issue": "No Cortex components detected as installed",
            "fix": "Install components: pip install cortex-cli-agent forge-agent a2a-hub vibe-replay",
        })
    else:
        checks_passed += 1

    # Print results
    if issues:
        table = Table(title=f"Found {len(issues)} issue(s)", show_header=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Issue", style="red")
        table.add_column("Fix", style="green")

        for i, issue in enumerate(issues, 1):
            table.add_row(str(i), issue["issue"], issue.get("fix", ""))

        console.print(table)

        # Auto-fix if requested
        if auto_fix:
            fixable = [iss for iss in issues if "auto_fix" in iss]
            if fixable:
                console.print(f"\n[bold]Auto-fixing {len(fixable)} issue(s)...[/bold]")
                for iss in fixable:
                    try:
                        iss["auto_fix"]()
                        console.print(f"  [green]Fixed:[/green] {iss['issue']}")
                    except Exception as exc:
                        console.print(f"  [red]Failed:[/red] {iss['issue']} ({exc})")
            else:
                console.print("\n[dim]No auto-fixable issues found.[/dim]")
    else:
        console.print(f"[bold green]All {checks_passed} checks passed. No issues found.[/bold green]")

    console.print(f"\n[dim]{checks_passed} checks passed, {len(issues)} issues found.[/dim]")
    console.print()
