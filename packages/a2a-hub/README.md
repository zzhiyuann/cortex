# A2A Hub — Agent-to-Agent Protocol Hub

A lightweight protocol and framework that enables AI agents to **discover each other, communicate, and collaborate** through a central WebSocket hub.

The killer feature: an **MCP server bridge** that lets [Claude Code](https://docs.anthropic.com/en/docs/claude-code) interact with the hub — discovering agents, delegating tasks, and collecting results through standard MCP tool calls.

## Why A2A Hub?

Building multi-agent systems is hard. You need service discovery, message routing, task lifecycle management, and a way to plug it all into your LLM workflow. A2A Hub gives you all of this in a single, lightweight package:

- **Agent Discovery** — Agents register capabilities; find the right agent for any task
- **Task Delegation** — Send work to agents and track results with built-in timeouts
- **MCP Bridge** — Claude Code can orchestrate agents directly through MCP tools
- **Simple Protocol** — JSON messages over WebSocket, validated with Pydantic
- **Python SDK** — Build agents in minutes with class-based or functional API

## Architecture

```
                          ┌─────────────────────┐
                          │                     │
                          │    A2A Hub Server    │
                          │   (WebSocket :8765)  │
                          │                     │
                          │  ┌───────────────┐  │
                          │  │ Agent Registry │  │
                          │  │ Task Manager   │  │
                          │  │ Msg Router     │  │
                          │  └───────────────┘  │
                          │                     │
                          └──┬───┬───┬───┬──────┘
                             │   │   │   │
              ┌──────────────┘   │   │   └──────────────┐
              │                  │   │                   │
    ┌─────────▼──────┐  ┌───────▼───▼────┐   ┌─────────▼──────┐
    │  Echo Agent    │  │  Code Reviewer  │   │  Research Agent │
    │  [echo]        │  │  [code-review]  │   │  [web-search,  │
    └────────────────┘  └────────────────┘   │   summarize]   │
                                              └────────────────┘
              │
    ┌─────────▼──────────┐
    │   MCP Bridge       │        ┌──────────────┐
    │   (FastMCP stdio)  │◄──────►│  Claude Code  │
    │   [bridge]         │  MCP   │              │
    └────────────────────┘        └──────────────┘
```

## Quick Start

### Install

```bash
# From source
git clone https://github.com/zzhiyuann/a2a-hub.git
cd a2a-hub
pip install -e ".[dev]"
```

### 1. Start the Hub

```bash
a2a-hub start
# Hub server listening on ws://localhost:8765
```

### 2. Run an Agent

```bash
# In a new terminal
python examples/echo_agent.py
# Starting Echo Agent, connecting to ws://localhost:8765
```

### 3. Use from Claude Code (MCP Bridge)

Add to your Claude Code MCP config (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "a2a-hub": {
      "command": "a2a-hub",
      "args": ["bridge"]
    }
  }
}
```

Now Claude Code can:

```
> Use the a2a-hub tools to discover available agents

> Delegate a task to the echo agent with message "hello world"

> Get the result for task abc-123
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `hub_status` | Show hub connection status and registered agents |
| `discover_agents` | Find agents by capability keyword |
| `delegate_task` | Send a task to a specific agent |
| `get_task_result` | Get result for a task (waits with timeout) |
| `broadcast_message` | Send message to all agents |
| `list_tasks` | List all tasks in the session |

## Building Agents

### Functional Style

```python
from a2a_hub import Agent

agent = Agent("my-agent", capabilities=["echo", "greet"])

@agent.on_task("echo")
async def handle_echo(message: str) -> str:
    return message

@agent.on_task("greet")
async def handle_greet(name: str) -> str:
    return f"Hello, {name}!"

agent.run()
```

### Class-Based Style

```python
from a2a_hub import Agent, capability

class CodeReviewAgent(Agent):
    name = "code-reviewer"

    @capability("code-review", description="Review Python code")
    async def review(self, code: str, language: str = "python") -> dict:
        issues = analyze(code)
        return {"issues": issues, "score": 8.5}

agent = CodeReviewAgent()
agent.run()
```

### Agent Features

- **Auto-reconnect** with exponential backoff
- **Heartbeat** keepalive (every 30s)
- **Graceful shutdown** on Ctrl+C
- **Task routing** by capability name

## Protocol

Messages are JSON objects over WebSocket:

```json
{
  "type": "delegate",
  "id": "msg-uuid",
  "from": "agent-a",
  "to": "agent-b",
  "timestamp": "2026-01-01T00:00:00Z",
  "payload": {
    "task_id": "task-uuid",
    "capability": "code-review",
    "params": {"code": "print('hi')"}
  }
}
```

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `register` | Agent -> Hub | Join the hub with capabilities |
| `deregister` | Agent -> Hub | Leave the hub |
| `discover` | Agent -> Hub | Find agents by capability |
| `delegate` | Agent -> Hub -> Agent | Assign a task |
| `result` | Agent -> Hub -> Agent | Return task result |
| `status` | Agent <-> Hub | Query task or hub status |
| `broadcast` | Agent -> Hub -> All | Message all agents |
| `heartbeat` | Agent -> Hub | Keepalive ping |

## CLI

```bash
a2a-hub start              # Start the hub server
a2a-hub start --port 9000  # Custom port
a2a-hub status             # Show hub status
a2a-hub agents             # List connected agents
a2a-hub agents --json      # JSON output
a2a-hub tasks              # List tasks
a2a-hub bridge             # Start MCP bridge
```

## Example Agents

| Agent | Capabilities | Description |
|-------|-------------|-------------|
| `echo_agent.py` | `echo` | Echoes messages back (testing) |
| `code_review_agent.py` | `code-review` | Reviews Python code with heuristics |
| `research_agent.py` | `web-search`, `summarize` | Simulates web research |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

## License

MIT
