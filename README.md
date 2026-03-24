<p align="center">
  <strong>Cortex</strong><br>
  <em>MCP plug-in layer for AI coding agents</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/vibe-replay/"><img src="https://img.shields.io/pypi/v/vibe-replay?label=vibe-replay&color=blue" alt="vibe-replay"></a>
  <a href="https://pypi.org/project/agent-dispatcher/"><img src="https://img.shields.io/pypi/v/agent-dispatcher?label=dispatcher&color=blue" alt="dispatcher"></a>
  <a href="https://pypi.org/project/forge-agent/"><img src="https://img.shields.io/pypi/v/forge-agent?label=forge&color=blue" alt="forge"></a>
  <a href="https://pypi.org/project/a2a-hub/"><img src="https://img.shields.io/pypi/v/a2a-hub?label=a2a-hub&color=blue" alt="a2a-hub"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

---

Your AI coding agent is powerful — but isolated. It can't be reached from your phone, can't talk to other agents, can't build its own tools, and forgets everything between sessions. **Cortex fixes this.**

Cortex is a **plug-in layer** that sits on top of Claude Code, Cursor, or any MCP-compatible agent — adding the capabilities they don't ship with:

```
              ┌──────────────────────────────────────────────┐
              │              CORTEX (plug-in layer)           │
              │                                               │
              │  Dispatcher   A2A Hub   Forge   Vibe Replay   │
              │  (command)    (comms)  (tools)   (memory)     │
              └──────────────────┬─────────────────────────────┘
                                 │  plugs in via MCP + hooks
              ┌──────────────────▼─────────────────────────────┐
              │     Your Agent (Claude Code / Cursor / etc.)    │
              └────────────────────────────────────────────────┘
```

## Components

| Package | What's Missing | What Cortex Adds | Install |
|---------|---------------|------------------|---------|
| **[Dispatcher](packages/dispatcher)** | Can't command your agent from your phone | Mobile JARVIS — Telegram bridge with session management, project routing, concurrent tasks | `pip install agent-dispatcher` |
| **[A2A Hub](packages/a2a-hub)** | Agents can't discover or delegate to each other | Agent-to-Agent protocol — WebSocket hub + MCP bridge for multi-agent orchestration | `pip install a2a-hub` |
| **[Forge](packages/forge)** | Agent can't build its own tools | Self-evolving tool agent — describe → generate → test → iterate → install | `pip install forge-agent` |
| **[Vibe Replay](packages/vibe-replay)** | Sessions are ephemeral — wisdom is lost | Session capture & replay — decisions, phases, patterns extracted into shareable HTML | `pip install vibe-replay` |

## Quick Start

### Option 1: Install everything

```bash
pip install cortex-cli-agent[all]

# Set up MCP servers, hooks, Telegram config
cortex init

# Check what's connected
cortex status
```

### Option 2: Install one component

```bash
# Example: capture your Claude Code sessions
pip install vibe-replay
vibe-replay install     # hooks into Claude Code
# ... code as normal ...
vibe-replay sessions    # see what was captured
vibe-replay replay <id> # open interactive HTML replay
```

### Option 3: From source (monorepo)

```bash
git clone https://github.com/zzhiyuann/cortex.git
cd cortex
uv sync --all-packages
uv run cortex init
```

## Demo: Extending Claude Code with MCP

Every Cortex component is an **MCP server** that Claude Code can call directly. Add to your MCP config:

```json
{
  "mcpServers": {
    "vibe-replay": {
      "command": "python3",
      "args": ["-m", "vibe_replay.mcp_server"]
    },
    "forge": {
      "command": "forge-mcp"
    },
    "a2a-hub": {
      "command": "a2a-hub",
      "args": ["bridge"]
    }
  }
}
```

Now Claude Code gains new abilities:

```
You: "What patterns came up in yesterday's debugging session?"
      → Claude calls vibe-replay to search session history

You: "I need a tool that validates YAML configs against a schema"
      → Claude calls forge to generate, test, and install the tool

You: "Delegate the code review to the review agent"
      → Claude calls a2a-hub to discover agents and send the task
```

**This is the plug-in model** — Cortex doesn't replace your agent, it gives it new capabilities through the standard MCP protocol.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        cortex-cli (orchestrator)                     │
│                    cortex init / cortex status                        │
├────────────┬────────────┬──────────────┬────────────────────────────┤
│            │            │              │                            │
│ Dispatcher │  A2A Hub   │    Forge     │       Vibe Replay          │
│            │            │              │                            │
│ Telegram   │ WebSocket  │ Code gen +   │ Hook-based capture →       │
│ → Agent    │ hub for    │ test loop +  │ phase detection →          │
│ bridge     │ agent-to-  │ auto-install │ HTML replay generation     │
│            │ agent      │ (MCP/CLI/    │                            │
│ Sessions,  │ comms      │ module)      │ Decision points, patterns, │
│ routing,   │            │              │ cross-session wisdom       │
│ memory     │ MCP bridge │ Clarify →    │                            │
│            │ for Claude │ Generate →   │ Exports: HTML, Markdown,   │
│            │ Code       │ Test → Fix   │ JSON                       │
├────────────┼────────────┼──────────────┼────────────────────────────┤
│    MCP     │    MCP     │     MCP      │    MCP + Hooks             │
└────────────┴────────────┴──────────────┴────────────────────────────┘
                              │
                    plugs into any MCP host
                              │
              ┌───────────────▼───────────────────┐
              │  Claude Code / Cursor / Windsurf   │
              └───────────────────────────────────┘
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project Structure

```
cortex/
├── packages/
│   ├── cortex-cli/      # Umbrella CLI — setup, status, health checks
│   ├── dispatcher/      # Telegram → agent bridge (Beta)
│   ├── a2a-hub/         # Agent-to-Agent protocol hub (Alpha)
│   ├── forge/           # Self-evolving tool generation (Alpha)
│   ├── vibe-replay/     # Session capture & replay (Alpha)
│   └── memory/          # Semantic memory module (Alpha)
├── pyproject.toml       # uv workspace definition
└── index.html           # GitHub Pages site
```

## Development

```bash
# Run all tests
uv run pytest

# Run a single package's tests
uv run pytest packages/vibe-replay/tests/ -v

# Lint
uv run ruff check packages/
```

## Website

[zzhiyuann.github.io/cortex](https://zzhiyuann.github.io/cortex/) — interactive docs and component deep-dives.

## License

MIT — see [LICENSE](LICENSE).
