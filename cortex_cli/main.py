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
