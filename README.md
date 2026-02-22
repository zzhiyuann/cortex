# Cortex

**Personal Agent Infrastructure Stack**

> One person. Infinite agents.

Cortex is four integrated systems that let a single person command, orchestrate, create tools, and capture wisdom — all through AI agents.

## The Stack

| Component | Role | Description |
|-----------|------|-------------|
| **[Dispatcher](https://github.com/zzhiyuann/cortex-dispatcher)** | Command | Mobile JARVIS via Telegram. Always-on daemon. Zero dependencies. |
| **[A2A Hub](https://github.com/zzhiyuann/a2a-hub)** | Communication | Agent-to-Agent protocol. WebSocket hub + MCP bridge for Claude Code. |
| **[Forge](https://github.com/zzhiyuann/forge)** | Creation | Self-evolving tool agent. Describe → Clarify → Generate → Test → Install. |
| **[Vibe Replay](https://github.com/zzhiyuann/vibe-replay)** | Memory | Capture sessions, extract learnings, generate shareable replays. |

## Architecture

```
                     ┌─────────────────────────────┐
                     │      YOU (Phone / Desktop)   │
                     └──────────────┬──────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │     DISPATCHER (Command)     │
                     │  Classify → Route → Execute  │
                     └──────┬───────────┬──────────┘
                            │           │
                  ┌─────────▼───┐  ┌────▼──────────┐
                  │   A2A HUB   │  │    FORGE       │
                  │   (Comms)   │◄►│  (Creation)    │
                  └──────┬──────┘  └───────────────┘
                         │
              ┌──────────▼─────────────────────────┐
              │     VIBE REPLAY (Memory)            │
              │  Capture → Analyze → Share Wisdom   │
              └────────────────────────────────────┘
```

## Quick Start

```bash
# Each component installs independently
pip install a2a-hub forge-agent vibe-replay

# Start the agent communication hub
a2a-hub start

# Create a tool from natural language
forge create "convert CSV to JSON with filtering"

# Capture and replay coding sessions
vibe-replay install    # Hook into Claude Code
vibe-replay replay latest
```

## Philosophy

We believe AI agents should be like utilities — always on, working for you, learning and improving. Cortex is built by one person who uses it daily to run a research lab, ship open-source projects, and automate everything in between.

The best tools don't just help you do things faster. They change what's possible for one person to do.

## Website

Open `index.html` or visit the [GitHub Pages site](https://zzhiyuann.github.io/cortex/) for the full interactive documentation.

## Stats

- **4** integrated systems
- **141** tests passing
- **0** external dependencies (Dispatcher)
- **1** person needed

## License

MIT — see individual component repos for details.

---

*Built by one person and their AI agents. Naturally.*
