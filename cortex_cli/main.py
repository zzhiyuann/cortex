"""Cortex CLI — unified entry point for the agent infrastructure stack.

cortex init   → detect components, configure everything in one shot
cortex start  → bring up dispatcher + a2a-hub
cortex stop   → shut them down
cortex status → show what's running and configured
"""

from __future__ import annotations

import click
from rich.console import Console

from cortex_cli.setup import run_init, run_status
from cortex_cli.services import start_all, stop_all
from cortex_cli.agent import start_agent, stop_agent, agent_status, agent_log
from cortex_cli.health import run_health
from cortex_cli.process import tail_log, log_file

console = Console()


@click.group()
@click.version_option("0.1.0", prog_name="cortex")
def cli():
    """Cortex — Personal Agent Infrastructure Stack.

    One install. Infinite agents.
    """


@cli.command()
@click.option("--telegram-token", help="Telegram bot token (from @BotFather).")
@click.option("--telegram-chat-id", type=int, help="Your Telegram chat ID.")
@click.option("--skip-dispatcher", is_flag=True, help="Skip Dispatcher setup.")
@click.option("--skip-forge", is_flag=True, help="Skip Forge MCP setup.")
@click.option("--skip-a2a", is_flag=True, help="Skip A2A Hub MCP setup.")
@click.option("--skip-replay", is_flag=True, help="Skip Vibe Replay hooks setup.")
def init(telegram_token, telegram_chat_id, skip_dispatcher, skip_forge, skip_a2a, skip_replay):
    """Set up everything — MCP servers, hooks, Dispatcher config.

    Detects installed Cortex components and wires them together.
    One command, zero manual config editing.
    """
    run_init(
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        skip_dispatcher=skip_dispatcher,
        skip_forge=skip_forge,
        skip_a2a=skip_a2a,
        skip_replay=skip_replay,
    )


@cli.command()
def start():
    """Start all Cortex services (Dispatcher + A2A Hub)."""
    start_all()


@cli.command()
def stop():
    """Stop all Cortex services."""
    stop_all()


@cli.command()
def status():
    """Show what's installed, configured, and running."""
    run_status()


@cli.command()
def health():
    """Run health checks on all Cortex components and wiring."""
    run_health()


@cli.command()
@click.argument("service", default="all")
@click.option("--lines", "-n", default=50, help="Number of log lines to show.")
def logs(service, lines):
    """View logs for a Cortex service.

    SERVICE can be: a2a-hub, dispatcher, agent, or all (default).
    """
    services = ["a2a-hub", "dispatcher"]
    if service == "all":
        targets = services
    elif service in services or service == "agent":
        targets = [service]
    else:
        console.print(f"[red]Unknown service: {service}[/red]")
        console.print(f"[dim]Available: {', '.join(services + ['agent', 'all'])}[/dim]")
        return

    for target in targets:
        if target == "agent":
            from cortex_cli.agent import LOG_FILE
            path = LOG_FILE
        else:
            path = log_file(target)

        if not path.exists():
            console.print(f"[dim]{target}: no log file[/dim]")
            continue

        console.print(f"\n[bold]--- {target} ---[/bold] ({path})")
        log_lines = tail_log(target) if target != "agent" else []
        if target == "agent":
            content = path.read_text().strip()
            log_lines = content.split("\n")[-lines:] if content else []

        for line in log_lines[-lines:]:
            console.print(f"  {line[:200]}")

    console.print()


@cli.group()
def agent():
    """Manage the autonomous improvement agent."""


@agent.command(name="start")
@click.option("--max-turns", default=50, help="Max agent turns before stopping.")
@click.option("--prompt", default=None, help="Additional instructions for the agent.")
def agent_start(max_turns, prompt):
    """Launch the autonomous agent to improve Cortex projects."""
    start_agent(max_turns=max_turns, custom_prompt=prompt)


@agent.command(name="stop")
def agent_stop_cmd():
    """Stop the running agent."""
    stop_agent()


@agent.command(name="status")
def agent_status_cmd():
    """Check agent state and progress."""
    agent_status()


@agent.command(name="log")
@click.option("--lines", "-n", default=50, help="Number of log lines to show.")
def agent_log_cmd(lines):
    """Tail the agent log."""
    agent_log(lines=lines)
