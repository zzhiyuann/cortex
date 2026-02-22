# Dispatcher

**Control your AI coding agent from your phone.**

Dispatcher bridges Telegram and CLI-based AI coding agents. Send a message from your phone, get code written, review results — all without touching your laptop.

Works with [Claude Code](https://claude.com/claude-code), [Aider](https://aider.chat), or any CLI-based AI agent.

## Features

- **Session management** — Resume conversations by replying to previous messages
- **Project routing** — Auto-detect which project you're talking about from keywords
- **Smart follow-ups** — Short replies auto-link to recently finished tasks
- **Concurrent tasks** — Run multiple agent sessions in parallel
- **Agent-agnostic** — Works with any CLI agent (Claude Code, Aider, etc.)
- **Progress tracking** — Typing indicators + periodic status updates
- **Memory** — Persistent user preferences across sessions
- **Easy deployment** — `pip install` + one command

## Quick Start

```bash
# Install
pip install agent-dispatcher

# Interactive setup (creates config, tests Telegram connection)
dispatcher init

# Start
dispatcher start
```

## How It Works

```
You (Phone) → Telegram → Dispatcher → AI Agent CLI → Your Codebase
     ↑                                                      |
     └──────────────── Result ←─────────────────────────────┘
```

1. Send a message in Telegram describing what you want done
2. Dispatcher routes it to the right project directory and spawns your AI agent
3. The agent executes the task and Dispatcher sends the result back to Telegram

## Configuration

Config lives at `~/.config/dispatcher/config.yaml`. Create it with `dispatcher init` or manually:

```yaml
telegram:
  bot_token: "YOUR_BOT_TOKEN"        # Get from @BotFather
  chat_id: YOUR_CHAT_ID              # Your Telegram user ID

agent:
  command: "claude"                   # Or "aider", any CLI agent
  args: ["-p", "--dangerously-skip-permissions"]
  max_concurrent: 3
  timeout: 1800                       # 30 min

projects:
  webapp:
    path: ~/projects/webapp
    keywords: [webapp, web, frontend]
  api:
    path: ~/projects/api
    keywords: [api, backend, server]
```

Environment variables override config file values:

| Variable | Overrides |
|---|---|
| `DISPATCHER_BOT_TOKEN` | `telegram.bot_token` |
| `DISPATCHER_CHAT_ID` | `telegram.chat_id` |
| `DISPATCHER_AGENT_COMMAND` | `agent.command` |
| `DISPATCHER_MAX_CONCURRENT` | `agent.max_concurrent` |
| `DISPATCHER_TIMEOUT` | `agent.timeout` |

## CLI Commands

```bash
dispatcher init          # Interactive setup
dispatcher start         # Start daemon (foreground)
dispatcher install       # Install as macOS LaunchAgent
dispatcher uninstall     # Remove LaunchAgent
dispatcher status        # Check if running
dispatcher logs          # Show logs
dispatcher logs -f       # Follow logs
```

## Auto-Start (macOS)

```bash
# Install as LaunchAgent — auto-starts on login, restarts on crash
dispatcher install

# Or use the convenience script
./scripts/install-macos.sh

# Remove
dispatcher uninstall
```

## Setting Up Telegram

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow the prompts
3. Copy the bot token
4. Message [@userinfobot](https://t.me/userinfobot) to get your chat ID
5. Run `dispatcher init` and enter both values

## Architecture

```
dispatcher/
├── cli.py        # CLI entry point (click)
├── config.py     # YAML + env var config loading
├── core.py       # Main event loop & message routing
├── telegram.py   # Telegram Bot API client (stdlib only)
├── runner.py     # Agent subprocess management
├── session.py    # Session & follow-up tracking
└── memory.py     # Persistent user preferences
```

## Development

```bash
git clone https://github.com/mephisto0/dispatcher
cd dispatcher
pip install -e .
dispatcher start
```

## License

MIT
