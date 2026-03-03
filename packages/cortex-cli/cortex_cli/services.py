"""Service management — start/stop/restart Cortex background services."""

from __future__ import annotations

from rich.console import Console

from cortex_cli.detect import detect_all
from cortex_cli.process import is_running, read_pid, start_service, stop_process, tail_log, log_file

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

# Map of service names to their definitions
SERVICE_MAP = {svc["name"]: svc for svc in SERVICES}


def _resolve_services(service: str) -> list[dict]:
    """Resolve 'all' or a specific service name to a list of service defs."""
    if service == "all":
        return SERVICES

    svc = SERVICE_MAP.get(service)
    if svc is None:
        available = ", ".join(SERVICE_MAP.keys())
        console.print(f"[red]Unknown service: {service}[/red]")
        console.print(f"[dim]Available: {available}, all[/dim]")
        return []

    return [svc]


def start_services(service: str = "all"):
    """Start specified Cortex service(s)."""
    targets = _resolve_services(service)
    if not targets:
        return

    components = detect_all()
    started = []

    console.print()

    for svc in targets:
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


def stop_services(service: str = "all"):
    """Stop specified Cortex service(s)."""
    targets = _resolve_services(service)
    if not targets:
        return

    console.print()
    stopped = []

    for svc in targets:
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


def restart_services(service: str = "all"):
    """Restart specified Cortex service(s)."""
    targets = _resolve_services(service)
    if not targets:
        return

    console.print()
    restarted = []

    components = detect_all()

    for svc in targets:
        name = svc["name"]
        label = svc["label"]

        # Stop if running
        if is_running(name):
            stop_process(name)
            console.print(f"[dim]Stopped {label}[/dim]")

        # Start
        comp = components.get(svc["component_key"])
        if not (comp and comp.installed):
            console.print(f"[dim]{label} not found, skipping[/dim]")
            continue

        command = [str(comp.cli_path)] + svc["command_args"]
        new_pid = start_service(name, command)
        if new_pid:
            console.print(f"[green]{label}[/green] restarted (pid {new_pid})")
            restarted.append(name)

    if restarted:
        console.print(f"\n[bold green]Restarted {len(restarted)} service(s).[/bold green]")
    else:
        console.print("\n[dim]No services to restart.[/dim]")

    console.print()


def show_logs(service: str = "all", lines: int = 50):
    """Show logs for specified service(s)."""
    targets = _resolve_services(service)
    if not targets:
        return

    for svc in targets:
        path = log_file(svc["name"])
        if not path.exists():
            console.print(f"[dim]{svc['label']}: no log file[/dim]")
            continue

        console.print(f"\n[bold]--- {svc['label']} ---[/bold] ({path})")
        log_lines = tail_log(svc["name"], lines=lines)
        for line in log_lines:
            console.print(f"  {line[:200]}")

    console.print()


# Backward compatibility aliases
def start_all():
    """Start all Cortex services (backward compatible)."""
    start_services("all")


def stop_all():
    """Stop all Cortex services (backward compatible)."""
    stop_services("all")
