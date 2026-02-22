"""Echo Agent — A minimal agent that echoes back any message.

Usage:
    python examples/echo_agent.py
    python examples/echo_agent.py --host localhost --port 8765
"""

from __future__ import annotations

import argparse

from a2a_hub import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo Agent")
    parser.add_argument("--host", default="localhost", help="Hub host")
    parser.add_argument("--port", type=int, default=8765, help="Hub port")
    args = parser.parse_args()

    agent = Agent("echo-agent", capabilities=["echo"])

    @agent.on_task("echo")
    async def handle_echo(message: str = "") -> dict:
        """Echo the message back with metadata."""
        return {
            "echo": message,
            "agent": "echo-agent",
            "status": "ok",
        }

    print(f"Starting Echo Agent, connecting to ws://{args.host}:{args.port}")
    agent.run(hub_host=args.host, hub_port=args.port)


if __name__ == "__main__":
    main()
