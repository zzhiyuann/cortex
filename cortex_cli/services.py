"""Service management — start/stop Dispatcher and A2A Hub."""

from __future__ import annotations

import os
import signal
import subprocess
import time

from rich.console import Console

from cortex_cli.detect import detect_all

console = Console()

PID_DIR = os.path.expanduser("~/.cortex")


def _pid_file(name: str) -> str:
    return os.path.join(PID_DIR, f"{name}.pid")


def _read_pid(name: str) -> int | None:
    path = _pid_file(name)
    if os.path.exists(path):
        try:
            pid = int(open(path).read().strip())
            # Check if process is alive
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            os.unlink(path)
    return None


def _write_pid(name: str, pid: int):
    os.makedirs(PID_DIR, exist_ok=True)
    with open(_pid_file(name), "w") as f:
        f.write(str(pid))


def _remove_pid(name: str):
    path = _pid_file(name)
    if os.path.exists(path):
        os.unlink(path)


def start_all():
    """Start Dispatcher and A2A Hub."""
    components = detect_all()
    started = []

    console.print()

    # Start A2A Hub
    a2a = components.get("a2a-hub")
    if a2a and a2a.installed:
        pid = _read_pid("a2a-hub")
        if pid:
            console.print(f"[dim]A2A Hub already running (pid {pid})[/dim]")
        else:
            cli = str(a2a.cli_path)
            log = os.path.join(PID_DIR, "a2a-hub.log")
            os.makedirs(PID_DIR, exist_ok=True)
            with open(log, "a") as logf:
                proc = subprocess.Popen(
                    [cli, "start"],
                    stdout=logf, stderr=logf,
                    start_new_session=True,
                )
            _write_pid("a2a-hub", proc.pid)
            console.print(f"[green]A2A Hub[/green] started (pid {proc.pid})")
            started.append("a2a-hub")
    else:
        console.print("[dim]A2A Hub not found, skipping[/dim]")

    # Start Dispatcher
    disp = components.get("dispatcher")
    if disp and disp.installed:
        pid = _read_pid("dispatcher")
        if pid:
            console.print(f"[dim]Dispatcher already running (pid {pid})[/dim]")
        else:
            cli = str(disp.cli_path)
            log = os.path.join(PID_DIR, "dispatcher.log")
            with open(log, "a") as logf:
                proc = subprocess.Popen(
                    [cli, "start"],
                    stdout=logf, stderr=logf,
                    start_new_session=True,
                )
            _write_pid("dispatcher", proc.pid)
            console.print(f"[green]Dispatcher[/green] started (pid {proc.pid})")
            started.append("dispatcher")
    else:
        console.print("[dim]Dispatcher not found, skipping[/dim]")

    if started:
        console.print(f"\n[bold green]Started {len(started)} service(s).[/bold green]")
        console.print("[dim]Logs: ~/.cortex/*.log[/dim]")
    else:
        console.print("\n[dim]No services to start.[/dim]")

    console.print()


def stop_all():
    """Stop all running Cortex services."""
    console.print()
    stopped = []

    for name in ("dispatcher", "a2a-hub"):
        pid = _read_pid(name)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                # Wait briefly for graceful shutdown
                for _ in range(10):
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.3)
                    except ProcessLookupError:
                        break
                else:
                    # Force kill if still alive
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                console.print(f"[green]Stopped {name}[/green] (pid {pid})")
                stopped.append(name)
            except ProcessLookupError:
                console.print(f"[dim]{name} was not running[/dim]")
            _remove_pid(name)
        else:
            console.print(f"[dim]{name} not running[/dim]")

    if stopped:
        console.print(f"\n[bold]Stopped {len(stopped)} service(s).[/bold]")
    else:
        console.print("\n[dim]No services were running.[/dim]")

    console.print()
