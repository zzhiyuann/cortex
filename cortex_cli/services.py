"""Service management — start/stop Cortex background services."""

from __future__ import annotations

from rich.console import Console

from cortex_cli.detect import detect_all
from cortex_cli.process import is_running, read_pid, start_service, stop_process

console = Console()

# Services to manage, in start order
SERVICES = [
    {
        "name": "a2a-hub",
        "label": "A2A Hub",
        "component_key": "a2a-hub",
        "command_args": ["start"],
    },
    {
        "name": "dispatcher",
        "label": "Dispatcher",
        "component_key": "dispatcher",
        "command_args": ["start"],
    },
]


def start_all():
    """Start all Cortex services."""
    components = detect_all()
    started = []

    console.print()

    for svc in SERVICES:
        comp = components.get(svc["component_key"])
        if not (comp and comp.installed):
            console.print(f"[dim]{svc['label']} not found, skipping[/dim]")
            continue

        pid = read_pid(svc["name"])
        if pid:
            console.print(f"[dim]{svc['label']} already running (pid {pid})[/dim]")
            continue

        command = [str(comp.cli_path)] + svc["command_args"]
        new_pid = start_service(svc["name"], command)
        if new_pid:
            console.print(f"[green]{svc['label']}[/green] started (pid {new_pid})")
            started.append(svc["name"])

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

    for svc in SERVICES:
        name = svc["name"]
        if is_running(name):
            stop_process(name)
            console.print(f"[green]Stopped {svc['label']}[/green]")
            stopped.append(name)
        else:
            console.print(f"[dim]{svc['label']} not running[/dim]")

    if stopped:
        console.print(f"\n[bold]Stopped {len(stopped)} service(s).[/bold]")
    else:
        console.print("\n[dim]No services were running.[/dim]")

    console.print()
