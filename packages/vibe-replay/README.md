# Vibe Replay

**Capture, reflect, and share your AI coding sessions.**

Vibe Replay automatically records your Claude Code sessions and transforms them into structured, shareable replays — not just logs, but distilled **wisdom**: what decisions were made, why, what patterns emerged, and what was learned.

```
┌─────────────────────────────────────────────────────────┐
│  🎬 Vibe Replay — "Building auth system"                │
│  Session: 2026-02-22 │ Duration: 45min │ 127 events     │
├───────────────────────┬─────────────────────────────────┤
│                       │                                 │
│  📋 REFLECTIONS       │  📊 TIMELINE                    │
│  ┌─────────────────┐  │                                 │
│  │ Key Decisions:  │  │  ● Exploration Phase            │
│  │ • Chose JWT     │  │    ├ Read auth.py               │
│  │   over sessions │  │    ├ Searched for patterns       │
│  │   for stateless │  │    └ Read 5 more files           │
│  │   auth          │  │                                 │
│  │                 │  │  ● Implementation                │
│  │ Learnings:      │  │    ├ Created middleware.py       │
│  │ • FastAPI deps  │  │    ├ ★ KEY DECISION              │
│  │   injection is  │  │    │  Chose decorator pattern    │
│  │   perfect for   │  │    │  over class-based approach  │
│  │   auth guards   │  │    ├ Edited 3 files              │
│  │                 │  │    └ Added tests                  │
│  │ Detours:        │  │                                 │
│  │ • Tried global  │  │  ● Debugging                    │
│  │   state first   │  │    ├ ⚠ Test failure              │
│  │   (didn't work) │  │    ├ Investigated imports        │
│  └─────────────────┘  │    └ ✓ Fixed & passing           │
│                       │                                 │
└───────────────────────┴─────────────────────────────────┘
```

## Why?

Every AI coding session contains valuable signal buried in noise. You make dozens of decisions, discover patterns, hit dead ends, and find solutions — but it all evaporates when the session ends.

Vibe Replay captures this signal and structures it into:

- **Timeline** — What happened, organized by phase (exploration → implementation → debugging → testing)
- **Key Decisions** — Where you chose one path over another, and why
- **Turning Points** — When things broke, and when they got fixed
- **Patterns** — Recurring approaches that worked (or didn't)
- **Aggregated Wisdom** — Learnings that compound across sessions

## Quick Start

```bash
# Install
pip install vibe-replay

# Hook into Claude Code (one-time setup)
vibe-replay install

# That's it! Sessions are now captured automatically.
# After a coding session, explore your replays:

vibe-replay sessions              # List captured sessions
vibe-replay show <session-id>     # Terminal summary
vibe-replay replay <session-id>   # Beautiful HTML replay in browser
vibe-replay wisdom                # Aggregated learnings
```

## Features

### Automatic Capture
Vibe Replay hooks into Claude Code via its native hook system. Every tool call, code change, search, and command is captured — silently, without slowing anything down.

### Smart Analysis
Raw events are processed into structured phases and insights:
- **Phase Detection** — Automatically identifies exploration, implementation, debugging, testing phases
- **Decision Points** — Spots where the direction of work changed
- **Detour Detection** — Finds when errors led to long investigation → fix cycles
- **Hotspot Files** — Files that were modified repeatedly (central pieces)

### Beautiful HTML Replays
Generate self-contained, interactive HTML files you can share with anyone:
- Dark/light theme toggle
- Expandable timeline with phase grouping
- Inline code diffs
- Decision and turning point markers
- Filterable by event type
- Insights sidebar with patterns and learnings

### Cross-Session Wisdom
```bash
vibe-replay wisdom
```
Aggregates patterns and learnings across all your sessions — building a personal knowledge base of what works.

### MCP Server (Bonus)
Let Claude Code query your past sessions:
```json
{
  "mcpServers": {
    "vibe-replay": {
      "command": "python3",
      "args": ["-m", "vibe_replay.mcp_server"]
    }
  }
}
```
Now Claude can search your history, recall what worked before, and avoid past mistakes.

## Commands

| Command | Description |
|---------|-------------|
| `vibe-replay install` | Install hooks into Claude Code |
| `vibe-replay uninstall` | Remove hooks |
| `vibe-replay status` | Check hook installation status |
| `vibe-replay sessions` | List captured sessions |
| `vibe-replay show <id>` | Show session summary in terminal |
| `vibe-replay replay <id>` | Generate HTML replay, open in browser |
| `vibe-replay export <id>` | Export as HTML, Markdown, or JSON |
| `vibe-replay analyze <id>` | Run/re-run analysis |
| `vibe-replay wisdom` | Aggregated learnings across sessions |
| `vibe-replay serve` | Local web server to browse all replays |

## How It Works

```
Claude Code Session
        │
        ▼
   ┌─────────┐     PostToolUse hook
   │  Hooks   │────────────────────────► capture-hook.py
   └─────────┘     (fast, non-blocking)       │
                                               ▼
                                        events.jsonl
                                        (append-only)
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │    Analyzer       │
                                    │  - Phase detect   │
                                    │  - Decisions      │
                                    │  - Insights       │
                                    └──────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │    Renderer       │
                                    │  - HTML replay    │
                                    │  - Markdown       │
                                    │  - JSON export    │
                                    └──────────────────┘
```

1. **Capture** — Claude Code hooks fire on every tool use. A lightweight Python script appends the event to a JSONL file.
2. **Store** — Each session gets its own directory under `~/.vibe-replay/sessions/`. A SQLite index enables fast queries.
3. **Analyze** — Groups events into phases, extracts insights, identifies decision/turning points.
4. **Render** — Generates beautiful, self-contained HTML replays (or Markdown/JSON).

## Export Formats

```bash
# Interactive HTML (default) — share with anyone
vibe-replay export <id> --format html -o replay.html

# Markdown summary — paste in README, blog, or Slack
vibe-replay export <id> --format md -o summary.md

# JSON — programmatic access to all data
vibe-replay export <id> --format json -o data.json
```

## Development

```bash
git clone https://github.com/zzhiyuann/cortex.git
cd cortex
uv sync --all-packages
uv run pytest packages/vibe-replay/tests/
```

## License

MIT
