"""A2A Hub CLI — Command-line interface for managing the hub.

Commands:
    a2a-hub start     Start the hub server
    a2a-hub bridge    Start the MCP bridge server
    a2a-hub status    Show hub status (connects and queries)
    a2a-hub agents    List connected agents
    a2a-hub tasks     List tracked tasks
"""

from __future__ import annotations

import asyncio
import json

import click
import websockets

from a2a_hub.protocol import A2AMessage, MessageType
from a2a_hub.server import HubServer
from a2a_hub.utils import DEFAULT_HOST, DEFAULT_PORT, get_logger

logger = get_logger("a2a-hub.cli")


async def _query_hub(host: str, port: int) -> dict:
    """Connect to the hub, send a status query, and return the response."""
    url = f"ws://{host}:{port}"
    async with websockets.connect(url) as ws:
        # Register temporarily
        reg = A2AMessage(
            type=MessageType.REGISTER,
            **{"from": "cli-probe"},
            to="hub",
            payload={"name": "CLI Probe", "capabilities": []},
        )
        await ws.send(reg.to_json())
        await ws.recv()  # ack

        # Query status
        status = A2AMessage(
            type=MessageType.STATUS,
            **{"from": "cli-probe"},
            to="hub",
        )
        await ws.send(status.to_json())
        raw = await ws.recv()
        resp = A2AMessage.from_json(str(raw))
        return resp.payload


@click.group()
@click.version_option(version="0.1.0", prog_name="a2a-hub")
def cli() -> None:
    """A2A Hub — Agent-to-Agent Protocol Hub."""
    pass


@cli.command()
@click.option("--host", default=DEFAULT_HOST, help="Bind address")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Bind port")
def start(host: str, port: int) -> None:
    """Start the hub server."""
    click.echo(f"Starting A2A Hub on ws://{host}:{port}")
    server = HubServer(host=host, port=port)
    try:
        asyncio.run(server.run_forever())
    except KeyboardInterrupt:
        click.echo("\nHub stopped.")


@cli.command()
@click.option("--host", default=DEFAULT_HOST, help="Hub host to connect to")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Hub port")
def bridge(host: str, port: int) -> None:
    """Start the MCP bridge server (stdio transport)."""
    from a2a_hub.mcp_bridge import _hub, run_bridge

    _hub.host = host
    _hub.port = port
    click.echo("Starting MCP Bridge...", err=True)
    run_bridge()


@cli.command()
@click.option("--host", default=DEFAULT_HOST, help="Hub host")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Hub port")
def status(host: str, port: int) -> None:
    """Show hub status."""
    try:
        payload = asyncio.run(_query_hub(host, port))
    except Exception as exc:
        click.echo(f"Cannot connect to hub at ws://{host}:{port}: {exc}")
        raise SystemExit(1)

    click.echo(f"Hub: ws://{host}:{port}")
    click.echo(f"Agents: {payload.get('agents_count', '?')}")
    click.echo(f"Tasks:  {payload.get('tasks_count', '?')}")


@cli.command()
@click.option("--host", default=DEFAULT_HOST, help="Hub host")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Hub port")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON")
def agents(host: str, port: int, as_json: bool) -> None:
    """List connected agents."""
    try:
        payload = asyncio.run(_query_hub(host, port))
    except Exception as exc:
        click.echo(f"Cannot connect to hub: {exc}")
        raise SystemExit(1)

    agents_list = payload.get("agents", [])
    if as_json:
        click.echo(json.dumps(agents_list, indent=2))
        return

    if not agents_list:
        click.echo("No agents connected.")
        return

    click.echo(f"Connected agents ({len(agents_list)}):")
    for a in agents_list:
        caps = ", ".join(a.get("capabilities", []))
        click.echo(f"  {a['name']} ({a['id']}) [{caps}] status={a.get('status', '?')}")


@cli.command()
@click.option("--host", default=DEFAULT_HOST, help="Hub host")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Hub port")
def tasks(host: str, port: int) -> None:
    """List tasks (currently shows count from hub status)."""
    try:
        payload = asyncio.run(_query_hub(host, port))
    except Exception as exc:
        click.echo(f"Cannot connect to hub: {exc}")
        raise SystemExit(1)

    click.echo(f"Tasks tracked: {payload.get('tasks_count', '?')}")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
