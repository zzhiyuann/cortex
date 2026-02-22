# Cortex

**The infrastructure layer for AI agents.**

> Your AI agent is powerful. Cortex makes it a system.

Cortex is not another agent framework. It's a plug-in layer that sits **on top** of Claude Code, Cursor, or any MCP-compatible agent — adding the capabilities they don't have: remote command, cross-agent communication, on-demand tool generation, and session memory.

```
              ┌──────────────────────────────────────────────┐
              │              CORTEX (plug-in layer)           │
              │                                               │
              │  Dispatcher   A2A Hub   Forge   Vibe Replay   │
              │  (command)    (comms)  (tools)   (memory)     │
              └──────────────────┬─────────────────────────────┘
                                 │  plugs into via MCP & hooks
              ┌──────────────────▼─────────────────────────────┐
              │     Your Agent (Claude Code / Cursor / etc.)    │
              └────────────────────────────────────────────────┘
```

## What It Adds

| Component | What's Missing | What Cortex Adds |
|-----------|---------------|------------------|
| **[Dispatcher](https://github.com/zzhiyuann/dispatcher)** | Can't command your agent from your phone | Mobile JARVIS via Telegram. Always-on daemon. |
| **[A2A Hub](https://github.com/zzhiyuann/a2a-hub)** | Agents can't discover or delegate to each other | Agent-to-Agent protocol. WebSocket hub + MCP bridge. |
| **[Forge](https://github.com/zzhiyuann/forge)** | Agent can't build its own tools | Self-evolving tool agent. Describe → Generate → Test → Install. |
| **[Vibe Replay](https://github.com/zzhiyuann/vibe-replay)** | Sessions are ephemeral — wisdom is lost | Capture sessions, extract decisions, generate shareable replays. |

## Install

```bash
pip install cortex-cli-agent

# Or via Homebrew
brew tap zzhiyuann/tap && brew install cortex
```

## Quick Start

```bash
# Set up Cortex on top of your existing agent
cortex init

# Check what's connected
cortex status

# Or install components individually
pip install vibe-replay          # Session memory
pip install forge-agent          # Tool generation
pip install a2a-hub              # Agent communication
```

## Philosophy

Claude Code, Cursor, Windsurf — these agents are incredible on their own. But they're isolated. Each session starts from zero. They can't talk to each other, can't be reached from your phone, and can't remember what they learned last week.

Cortex doesn't replace them. It plugs in and adds the missing infrastructure — turning a powerful but isolated agent into a persistent, autonomous system that compounds over time.

## Website

[zzhiyuann.github.io/cortex](https://zzhiyuann.github.io/cortex/) — full interactive docs and architecture overview.

## License

MIT
