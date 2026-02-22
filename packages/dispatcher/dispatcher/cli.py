"""CLI entry point: `dispatcher start`, `dispatcher init`, etc."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import click
import yaml

from . import __version__
from .config import Config, DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_FILE, DEFAULTS


def _setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("dispatcher")
    log.setLevel(logging.INFO)
    fh = RotatingFileHandler(
        log_dir / "dispatcher.log", maxBytes=5_000_000, backupCount=2
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    log.addHandler(logging.StreamHandler())


@click.group()
@click.version_option(__version__, prog_name="dispatcher")
def main():
    """Dispatcher — Control your AI coding agent from your phone."""
    pass


@main.command()
def init():
    """Interactive setup: create config file and test connection."""
    click.echo("Dispatcher setup\n")

    if DEFAULT_CONFIG_FILE.exists():
        if not click.confirm(f"Config already exists at {DEFAULT_CONFIG_FILE}. Overwrite?"):
            click.echo("Aborted.")
            return

    # Gather info
    bot_token = click.prompt("Telegram Bot Token (from @BotFather)")
    chat_id = click.prompt("Your Telegram Chat ID", type=int)
    agent_cmd = click.prompt("Agent command", default="claude")

    config_data = {
        "telegram": {
            "bot_token": bot_token,
            "chat_id": chat_id,
        },
        "agent": {
            "command": agent_cmd,
            "args": ["-p", "--dangerously-skip-permissions"],
            "max_concurrent": 3,
            "timeout": 1800,
        },
        "behavior": {
            "cancel_keywords": ["cancel", "stop"],
            "status_keywords": ["status"],
        },
        "projects": {},
    }

    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_CONFIG_FILE, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    click.echo(f"\nConfig written to {DEFAULT_CONFIG_FILE}")

    # Test connection
    click.echo("\nTesting Telegram connection...")
    cfg = Config()
    from .telegram import TelegramClient
    tg = TelegramClient(cfg.bot_token, cfg.chat_id)
    result = tg.send("Dispatcher connected! Setup complete.")
    if result:
        click.echo("Success! Check your Telegram for the test message.")
    else:
        click.echo("Warning: Could not send test message. Check your token and chat ID.")

    click.echo(f"\nRun 'dispatcher start' to begin.")


@main.command()
@click.option("-c", "--config", "config_path", default=None, help="Config file path")
def start(config_path):
    """Start the dispatcher daemon (foreground)."""
    cfg = Config(config_path)
    errors = cfg.validate()
    if errors:
        click.echo("Configuration errors:", err=True)
        for e in errors:
            click.echo(f"  - {e}", err=True)
        click.echo(f"\nRun 'dispatcher init' to set up, or edit {DEFAULT_CONFIG_FILE}", err=True)
        sys.exit(1)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(cfg.log_dir)

    click.echo(f"Starting dispatcher (agent: {cfg.agent_command})...")
    from .core import Dispatcher
    d = Dispatcher(cfg)
    asyncio.run(d.run())


@main.command()
def status():
    """Show dispatcher status."""
    # Check if process is running
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.dispatcher.agent.plist"
    if plist_path.exists():
        result = subprocess.run(
            ["launchctl", "list", "com.dispatcher.agent"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            click.echo("Dispatcher is running (launchd)")
        else:
            click.echo("Dispatcher is installed but not running")
    else:
        click.echo("Dispatcher is not installed as a service")
        click.echo("Run 'dispatcher start' to run in foreground")
        click.echo("Run 'dispatcher install' to install as a service")


@main.command()
def install():
    """Install as macOS LaunchAgent (auto-start on login)."""
    if sys.platform != "darwin":
        click.echo("Auto-install currently supports macOS only.")
        click.echo("For Linux, create a systemd service manually.")
        return

    cfg = Config()
    errors = cfg.validate()
    if errors:
        click.echo("Fix config first: dispatcher init", err=True)
        sys.exit(1)

    dispatcher_bin = subprocess.run(
        ["which", "dispatcher"], capture_output=True, text=True
    ).stdout.strip()
    if not dispatcher_bin:
        # Fallback: use the venv's bin directory
        venv_bin = Path(sys.executable).parent / "dispatcher"
        if venv_bin.exists():
            dispatcher_bin = str(venv_bin)
        else:
            click.echo("Error: 'dispatcher' command not found in PATH", err=True)
            sys.exit(1)

    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "com.dispatcher.agent.plist"

    log_dir = cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_content = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.dispatcher.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>{dispatcher_bin}</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/launchd-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/launchd-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin:{Path.home()}/.npm-global/bin:{Path(dispatcher_bin).parent}</string>
    </dict>
</dict>
</plist>"""

    plist_path.write_text(plist_content)

    subprocess.run(["launchctl", "unload", str(plist_path)],
                    capture_output=True)
    subprocess.run(["launchctl", "load", str(plist_path)])

    click.echo(f"Installed and started: {plist_path}")
    click.echo("Dispatcher will auto-start on login.")


@main.command()
def uninstall():
    """Remove macOS LaunchAgent."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.dispatcher.agent.plist"
    if not plist_path.exists():
        click.echo("Not installed.")
        return

    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    plist_path.unlink()
    click.echo("Uninstalled and stopped.")


@main.command()
@click.option("-n", "--lines", default=50, help="Number of lines to show")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
def logs(lines, follow):
    """Show dispatcher logs."""
    cfg = Config()
    log_file = cfg.log_dir / "dispatcher.log"
    if not log_file.exists():
        click.echo("No logs yet.")
        return

    cmd = ["tail"]
    if follow:
        cmd.append("-f")
    cmd += ["-n", str(lines), str(log_file)]
    os.execvp("tail", cmd)
