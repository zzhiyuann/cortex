<p align="center">
  <strong>Cortex</strong><br>
  <em>The missing plug-in layer for AI coding agents</em>
</p>

<p align="center">
  <a href="https://github.com/zzhiyuann/cortex/actions/workflows/tests.yml"><img src="https://github.com/zzhiyuann/cortex/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/vibe-replay/"><img src="https://img.shields.io/pypi/v/vibe-replay?label=vibe-replay&color=blue" alt="vibe-replay"></a>
  <a href="https://pypi.org/project/agent-dispatcher/"><img src="https://img.shields.io/pypi/v/agent-dispatcher?label=dispatcher&color=blue" alt="dispatcher"></a>
  <a href="https://pypi.org/project/forge-agent/"><img src="https://img.shields.io/pypi/v/forge-agent?label=forge&color=blue" alt="forge"></a>
  <a href="https://pypi.org/project/a2a-hub/"><img src="https://img.shields.io/pypi/v/a2a-hub?label=a2a-hub&color=blue" alt="a2a-hub"></a>
  <a href="https://pypi.org/project/cortex-agent-memory/"><img src="https://img.shields.io/pypi/v/cortex-agent-memory?label=memory&color=blue" alt="memory"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

---

Claude Code, Cursor, Windsurf — your AI coding agent is powerful, but isolated. It can't be reached from your phone. It can't talk to other agents. It can't build its own tools. And it forgets everything between sessions.

**Cortex fixes all four.**

It's a set of independent Python packages that plug into any MCP-compatible agent via the standard [Model Context Protocol](https://modelcontextprotocol.io). No vendor lock-in, no platform to adopt — just `pip install` the capabilities you need.

```
              ┌──────────────────────────────────────────────┐
              │              CORTEX (plug-in layer)           │
              │                                               │
              │  Dispatcher  A2A Hub  Forge  Memory  Replay    │
              │  (command)   (comms) (tools) (recall) (capture)│
              └──────────────────┬─────────────────────────────┘
                                 │  MCP + hooks
              ┌──────────────────▼─────────────────────────────┐
              │     Your Agent (Claude Code / Cursor / etc.)    │
              └────────────────────────────────────────────────┘
```

## Components

| Package | Problem | Solution | Status | Install |
|---------|---------|----------|--------|---------|
| **[Dispatcher](packages/dispatcher)** | Can't reach your agent from your phone | Telegram bot that bridges to your local agent — sessions, routing, concurrent tasks | Beta | `pip install agent-dispatcher` |
| **[Forge](packages/forge)** | Agent can't build its own tools | Describe a tool in English → generates code → runs tests → iterates up to 5x → installs it | Alpha | `pip install forge-agent` |
| **[A2A Hub](packages/a2a-hub)** | Agents can't discover or delegate to each other | WebSocket hub + MCP bridge for agent-to-agent communication | Alpha | `pip install a2a-hub` |
| **[Vibe Replay](packages/vibe-replay)** | Sessions are ephemeral — decisions and learnings are lost | Auto-captures sessions via hooks, extracts phases/decisions/patterns, generates interactive HTML replays | Alpha | `pip install vibe-replay` |
| **[Memory](packages/memory)** | Agent forgets everything between sessions | Semantic memory store — LLM extraction + vector search via MCP | Alpha | `pip install cortex-agent-memory` |

Each package is independently installable and useful on its own. No coupling between components.

## Quick Start

```bash
# Install one component
pip install agent-dispatcher
dispatcher init    # interactive setup — Telegram bot token, project paths
dispatcher start   # your agent is now reachable from your phone

# Or install everything
pip install cortex-cli-agent[all]
cortex init        # sets up MCP servers, hooks, Telegram
cortex status      # health check all components
```

## How It Works: MCP Plug-ins

Every Cortex component is an **MCP server**. Add one line to your agent's config and it gains new abilities:

```json
{
  "mcpServers": {
    "forge": { "command": "forge-mcp" },
    "a2a-hub": { "command": "a2a-hub", "args": ["bridge"] },
    "memory": { "command": "python3", "args": ["-m", "memory.server"] }
  }
}
```

Now your agent can do things it couldn't before:

```
You: "I need a tool that validates YAML configs against a schema"
→ Claude calls Forge → generates code → runs tests → installs the tool

You: "Delegate the code review to the review agent"
→ Claude calls A2A Hub → discovers agents → sends the task → gets results

You: "What did we learn from yesterday's debugging session?"
→ Claude calls Memory → semantic search → returns relevant learnings
```

The agent doesn't know these are plug-ins — they appear as native capabilities through the MCP protocol.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        cortex-cli (orchestrator)                     │
│                    cortex init / cortex status                        │
├────────────┬────────────┬──────────────┬──────────────┬─────────────┤
│            │            │              │              │             │
│ Dispatcher │  A2A Hub   │    Forge     │ Vibe Replay  │   Memory    │
│            │            │              │              │             │
│ Telegram   │ WebSocket  │ Desc → Gen   │ Hook-based   │ Semantic    │
│ → Agent    │ hub for    │ → Test → Fix │ capture →    │ search +    │
│ bridge     │ agent-to-  │ → Install    │ analyze →    │ LLM         │
│            │ agent      │ (MCP/CLI/    │ HTML replay  │ extraction  │
│ Sessions,  │ comms      │ module)      │              │             │
│ routing,   │            │              │ Decisions,   │ MCP server  │
│ memory     │ MCP bridge │ 5x iterate   │ patterns,    │ for agents  │
│            │ for agents │ loop         │ wisdom       │             │
├────────────┼────────────┼──────────────┼──────────────┼─────────────┤
│    MCP     │    MCP     │     MCP      │  MCP + Hooks │    MCP      │
└────────────┴────────────┴──────────────┴──────────────┴─────────────┘
                              │
                    plugs into any MCP host
                              │
              ┌───────────────▼───────────────────┐
              │  Claude Code / Cursor / Windsurf   │
              └───────────────────────────────────┘
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Design Principles

1. **Plug-in, not platform** — extends your existing agent, doesn't replace it
2. **Each component stands alone** — install one or all, no coupling
3. **MCP-first** — every component is an MCP server, one JSON line to connect
4. **No external services** — SQLite, JSONL, local WebSocket — everything runs on your machine
5. **Fail-safe** — capture hooks are non-blocking; if Cortex fails, your agent keeps working

## Project Structure

```
cortex/
├── packages/
│   ├── cortex-cli/      # Umbrella CLI — setup, status, health checks
│   ├── dispatcher/      # Telegram → agent bridge (Beta, 310 tests)
│   ├── a2a-hub/         # Agent-to-Agent protocol hub (Alpha, 80 tests)
│   ├── forge/           # Self-evolving tool generation (Alpha, 120 tests)
│   ├── vibe-replay/     # Session capture & replay (Alpha, 50 tests)
│   └── memory/          # Semantic memory module (Alpha, 30 tests)
├── docs/                # Architecture guide
├── examples/            # Walkthroughs
├── pyproject.toml       # uv workspace
└── CONTRIBUTING.md      # How to contribute
```

## Development

```bash
git clone https://github.com/zzhiyuann/cortex.git
cd cortex
uv sync --all-packages   # install all packages in dev mode

uv run pytest                              # all 600+ tests
uv run pytest packages/dispatcher/tests/   # one package
uv run ruff check packages/                # lint
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

## Status

Cortex is **early-stage open source**. The Dispatcher is in beta (daily driver for the author); other components are alpha. APIs may change. Contributions and feedback are very welcome.

## License

MIT — see [LICENSE](LICENSE).
