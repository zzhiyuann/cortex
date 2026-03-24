# Cortex Memory

**Long-term semantic memory for AI coding agents.**

Memory stores learnings, patterns, and decisions from past sessions and makes them searchable via semantic similarity. It integrates with Claude Code as an MCP server — giving your agent access to everything it has ever learned.

## Quick Start

```bash
pip install cortex-memory

# Start the MCP server
python -m memory.server
```

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python3",
      "args": ["-m", "memory.server"]
    }
  }
}
```

## How It Works

1. **Extract** — LLM-powered extraction pulls structured learnings from raw session text
2. **Store** — Learnings are embedded as vectors and stored locally
3. **Search** — Semantic similarity search finds relevant past learnings for current work

## CLI

```bash
memory store "JWT refresh tokens need mutex in concurrent scenarios"
memory search "authentication race condition"
memory list
```

## Architecture

| Module | Purpose |
|--------|---------|
| `store.py` | Vector storage with numpy-based semantic search |
| `extractor.py` | LLM extraction of structured learnings from text |
| `server.py` | MCP server exposing search as tools |
| `cli.py` | Command-line interface |

## License

MIT
