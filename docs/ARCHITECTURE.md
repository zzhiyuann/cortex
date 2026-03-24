# Cortex Architecture

Cortex is a **plug-in layer for AI coding agents**. It extends MCP-compatible agents (Claude Code, Cursor, Windsurf) with capabilities they don't ship with — remote command, inter-agent communication, tool generation, and session memory.

## Design Principles

1. **Plug-in, not platform** — Cortex doesn't replace your agent. It adds capabilities via MCP servers and hooks.
2. **Each component stands alone** — Install one or all. No coupling between components.
3. **MCP-first** — Every component exposes an MCP server, making it natively accessible from any MCP host.
4. **No external services** — SQLite, JSONL, local WebSocket. Everything runs on your machine.
5. **Fail-safe** — Capture hooks are non-blocking. If Cortex fails, your agent keeps working.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interfaces                              │
│                                                                  │
│   Telegram (mobile)    Claude Code (terminal)    Browser (HTML)  │
└───────┬────────────────────┬─────────────────────────┬──────────┘
        │                    │                         │
        ▼                    ▼                         ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  Dispatcher  │  │   MCP Protocol   │  │  Vibe Replay HTML    │
│  (Telegram   │  │   (stdio/SSE)    │  │  (self-contained     │
│   Bot API)   │  │                  │  │   static files)      │
└──────┬───────┘  └────────┬─────────┘  └──────────────────────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│                    Cortex Components                          │
│                                                               │
│  ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐   │
│  │ Dispatcher │ │ A2A Hub  │ │  Forge  │ │ Vibe Replay  │   │
│  │            │ │          │ │         │ │              │   │
│  │ Telegram → │ │ Agent    │ │ Desc →  │ │ Hooks →      │   │
│  │ Agent CLI  │ │ registry │ │ Gen →   │ │ JSONL →      │   │
│  │ spawn      │ │ + task   │ │ Test →  │ │ Analyze →    │   │
│  │            │ │ routing  │ │ Fix →   │ │ Render       │   │
│  │ Sessions   │ │          │ │ Install │ │              │   │
│  │ Memory     │ │ WS hub   │ │         │ │ SQLite index │   │
│  │ Routing    │ │ MCP      │ │ Jinja2  │ │ Phase detect │   │
│  │            │ │ bridge   │ │ pytest  │ │ Decisions    │   │
│  └────────────┘ └──────────┘ └─────────┘ └──────────────┘   │
│                                                               │
│  ┌──────────────────────┐  ┌──────────────────────────────┐  │
│  │ Memory               │  │ Cortex CLI                    │  │
│  │ Semantic search +    │  │ Orchestrator — init, status,  │  │
│  │ LLM extraction       │  │ health checks, setup          │  │
│  └──────────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Storage Layer                              │
│                                                               │
│  ~/.config/dispatcher/   JSONL sessions    SQLite indexes     │
│  config.yaml             ~/.vibe-replay/   ~/.forge/tools/    │
│                          sessions/                            │
└──────────────────────────────────────────────────────────────┘
```

## Component Deep-Dives

### 1. Dispatcher — Remote Agent Command

**Problem**: You're away from your computer but need your AI agent to do something.

**Solution**: A Telegram bot that bridges messages to your local AI agent CLI.

```
Phone → Telegram API → Dispatcher daemon → Agent CLI → Codebase
  ↑                                                        │
  └─────────────── Result (Telegram message) ──────────────┘
```

**Key internals:**
- **Core loop** (`core.py`): Long-polls Telegram, routes messages, manages lifecycle
- **Runner** (`runner.py`): Spawns agent subprocesses, captures stdout/stderr, enforces timeouts
- **Session tracker** (`session.py`): Links follow-up messages to active sessions
- **Project router** (`config.py`): Maps keyword patterns → project directories
- **WebSocket server** (`ws_server.py`): Enables iOS app connectivity with ack protocol

**Data flow:**
1. Telegram message arrives via long-polling
2. Classifier determines: new task, follow-up, or command
3. Project router selects target directory from keywords
4. Runner spawns agent subprocess with message as prompt
5. Output streams back to Telegram (chunked for long responses)

**Storage**: `~/.config/dispatcher/config.yaml` (config), in-memory sessions

---

### 2. A2A Hub — Agent-to-Agent Communication

**Problem**: Agents can't discover or delegate work to each other.

**Solution**: A WebSocket hub with service discovery, task routing, and an MCP bridge.

```
Agent A ──register──→ ┌──────────┐ ←──register── Agent B
                      │  A2A Hub │
Agent A ──delegate──→ │  :8765   │ ──delegate──→ Agent B
                      │          │
Claude  ──MCP call──→ │ MCP      │ ──delegate──→ Agent B
 Code                 │ Bridge   │
                      └──────────┘
```

**Protocol** (JSON over WebSocket):
- `register` / `deregister` — join/leave the hub with capabilities list
- `discover` — find agents by capability
- `delegate` — send a task to a specific agent, get a task_id back
- `result` — agent returns completed work
- `broadcast` — message all connected agents
- `heartbeat` — keepalive (30s interval)

**MCP bridge**: A FastMCP stdio server that connects Claude Code to the hub. Claude Code calls MCP tools (`discover_agents`, `delegate_task`, `get_task_result`) which translate to WebSocket protocol messages.

**Hardening**: Rate limiting (100 msg/min per agent), heartbeat-based health checks, graceful shutdown, auto-reconnect with exponential backoff.

---

### 3. Forge — Self-Evolving Tool Agent

**Problem**: Your agent can't build its own tools on the fly.

**Solution**: A pipeline that takes a natural-language description and produces a tested, installed tool.

```
Description → Clarify → Generate → Test → Fix → Install
                ↑                          │
                └──── iterate (max 5x) ────┘
```

**Pipeline stages:**
1. **Clarifier** — Analyzes the description for ambiguities, generates targeted questions
2. **Generator** — Builds a `ToolSpec` (Pydantic model), renders code via Jinja2 templates
3. **Tester** — Auto-generates pytest tests, runs them in an isolated temp directory
4. **Iterator** — Parses test failures, applies targeted fixes, re-generates, re-tests (up to 5x)
5. **Installer** — Installs the working tool as MCP server, CLI command, or Python module

**Output types:**
| Type | Template | Installed to |
|------|----------|-------------|
| `python` | `python_function.py.jinja` | `~/.forge/tools/` |
| `cli` | `cli_tool.py.jinja` | `~/.local/bin/` |
| `mcp` | `mcp_tool.py.jinja` | MCP config |

**MCP integration**: Forge itself is an MCP server. Claude Code can call `forge_create` → `forge_answer` → `forge_generate` → `forge_install` to build and install tools without leaving the session.

---

### 4. Vibe Replay — Session Capture & Replay

**Problem**: AI coding sessions are ephemeral. Decisions, patterns, and learnings vanish when the session ends.

**Solution**: Automatic capture via Claude Code hooks, with phase detection and interactive HTML replays.

```
Claude Code session
    │
    │  PostToolUse hook (non-blocking)
    ▼
capture-hook.py → events.jsonl (append-only)
                       │
                       │  vibe-replay analyze
                       ▼
               ┌───────────────┐
               │   Analyzer    │
               │               │
               │ Phase detect  │
               │ Decision pts  │
               │ Detour detect │
               │ Hotspot files │
               │ Insights      │
               └───────┬───────┘
                       │
                       │  vibe-replay replay
                       ▼
               ┌───────────────┐
               │   Renderer    │
               │               │
               │ HTML template │
               │ Dark/light    │
               │ Timeline      │
               │ Diffs         │
               │ Search        │
               └───────────────┘
                       │
                       ▼
              session-replay.html
              (self-contained, shareable)
```

**Capture strategy**: Claude Code's hook system fires `PostToolUse` and `Stop` events. A lightweight Python script (< 50ms per event) appends structured JSON to a JSONL file. Non-blocking — if capture fails, the session continues.

**Analysis pipeline:**
1. Parse JSONL events into typed models (Pydantic)
2. Group events into temporal phases (exploration, implementation, debugging, testing)
3. Identify decision points (direction changes) and turning points (error → fix cycles)
4. Detect hotspot files (repeatedly modified)
5. Extract cross-session patterns via the `wisdom` command

**Storage**: `~/.vibe-replay/sessions/<session-id>/` with SQLite index for fast queries.

---

### 5. Memory — Semantic Search

**Problem**: Learnings from past sessions aren't accessible for future work.

**Solution**: A semantic search store backed by numpy vectors + LLM extraction.

- **Store** (`store.py`): Vector similarity search over stored learnings
- **Extractor** (`extractor.py`): Uses Claude API to extract structured learnings from raw text
- **MCP server** (`server.py`): Exposes search as MCP tools for Claude Code

---

### 6. Cortex CLI — Orchestrator

The umbrella CLI that sets up all components:

```bash
cortex init     # Configure Telegram, install hooks, register MCP servers
cortex status   # Health check all components
cortex doctor   # Diagnose issues
```

Handles: MCP config generation, hook installation, service detection, health checks.

## Integration Pattern: MCP

All Cortex components follow the same integration pattern:

```
┌──────────────┐     stdio      ┌──────────────┐
│  MCP Host    │ ◄────────────► │  MCP Server   │
│ (Claude Code)│    JSON-RPC    │ (Cortex pkg)  │
└──────────────┘                └──────────────┘
```

Each component ships an MCP server that exposes its functionality as tools. The MCP host (Claude Code, Cursor, etc.) discovers these tools at startup and can call them during sessions.

This means **Cortex components are invisible to the user** — they appear as native capabilities of the agent, not as external services.

## Data Model

| Component | Primary Storage | Index | Format |
|-----------|----------------|-------|--------|
| Dispatcher | `~/.config/dispatcher/` | — | YAML config, in-memory sessions |
| A2A Hub | In-memory | — | WebSocket connections |
| Forge | `~/.forge/tools/` | — | Python files + JSON metadata |
| Vibe Replay | `~/.vibe-replay/sessions/` | SQLite | JSONL events + HTML replays |
| Memory | `~/.cortex/memory/` | numpy vectors | JSON documents |

## Technology Stack

- **Language**: Python 3.11+
- **Package manager**: uv (workspace monorepo)
- **Build system**: Hatchling
- **Validation**: Pydantic v2
- **CLI**: Click + Rich
- **MCP**: FastMCP (A2A Hub), custom stdio (others)
- **WebSocket**: websockets library
- **Templates**: Jinja2 (Forge code generation, Vibe Replay HTML)
- **Testing**: pytest + pytest-asyncio
- **Storage**: SQLite + JSONL + numpy
