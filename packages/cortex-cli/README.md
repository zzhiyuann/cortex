# Cortex CLI

**One command to set up your entire agent infrastructure.**

Cortex CLI is the orchestrator for the [Cortex](https://github.com/zzhiyuann/cortex) ecosystem. It configures MCP servers, installs hooks, sets up Telegram, and monitors health — so you don't have to wire things together manually.

## Quick Start

```bash
pip install cortex-cli-agent[all]

# Interactive setup — configures everything
cortex init

# Check what's running
cortex status
```

## What `cortex init` Does

1. Prompts for Telegram bot token and chat ID (for Dispatcher)
2. Installs Vibe Replay hooks into Claude Code
3. Registers all MCP servers in your Claude Code config
4. Writes `~/.cortex/config.yaml` with your settings
5. Runs health checks to verify everything works

## Commands

| Command | Description |
|---------|-------------|
| `cortex init` | Interactive first-time setup |
| `cortex status` | Health check all components |
| `cortex setup` | Re-run setup for a specific component |
| `cortex doctor` | Diagnose and fix common issues |

## What It Orchestrates

| Component | What Cortex CLI configures |
|-----------|---------------------------|
| [Dispatcher](../dispatcher) | Telegram config, daemon install |
| [Vibe Replay](../vibe-replay) | Hook installation, session directory |
| [Forge](../forge) | MCP server registration |
| [A2A Hub](../a2a-hub) | MCP bridge registration |
| [Memory](../memory) | MCP server registration |

## License

MIT
