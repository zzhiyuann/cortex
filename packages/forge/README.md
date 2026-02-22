# Forge — Self-Evolving Tool Agent

> Describe what you need in plain English. Forge generates the tool, tests it, iterates until it works, and installs it into your toolkit.

Most code generators stop at "here's some code, good luck." Forge doesn't stop until the tool **actually works** — it generates tests, runs them, and if they fail, analyzes the errors and regenerates. Automatically.

## The Problem

You need a quick tool — a CSV converter, an API wrapper, a file processor. You could write it yourself (15-30 min), ask an LLM to generate it (then debug it yourself), or use a template (then customize it yourself).

## The Solution

```
forge create "convert CSV to JSON with column filtering"
```

Forge will:
1. Ask clarifying questions (input format? output format? edge cases?)
2. Generate the tool code with proper types, docstrings, and error handling
3. Auto-generate and run tests
4. If tests fail: analyze errors, fix the code, re-test (up to 5 iterations)
5. Install the working tool wherever you want — MCP server, CLI command, or Python module

## Quick Start

### Install

```bash
# Clone and install
git clone https://github.com/zzhiyuann/forge.git
cd forge
pip install -e ".[dev]"
```

### Create a Tool (CLI)

```bash
# Interactive mode — Forge asks clarification questions
forge create "convert CSV to JSON with filtering"

# Quick mode — skip questions, generate immediately
forge create "count words in a file" --type cli --no-clarify

# Generate and install as MCP tool in one step
forge create "fetch URL content and extract links" --type mcp --install mcp
```

### Create a Tool (MCP / Claude Code)

Forge ships as an MCP server. Add it to your MCP config:

```json
{
  "mcpServers": {
    "forge": {
      "command": "forge-mcp"
    }
  }
}
```

Then in Claude Code:

```
You: I need a tool that converts markdown tables to CSV
Claude: [calls forge_create] → asks questions → [calls forge_generate] → tests pass → [calls forge_install]
Claude: Done! Your tool is installed. Restart to use it.
```

### Manage Tools

```bash
# List all created tools
forge list

# Show a tool's source code
forge show convert_csv_to_json

# Re-run tests
forge test convert_csv_to_json

# Remove a tool
forge uninstall convert_csv_to_json
```

## How It Works

```
User describes tool
  │
  ▼
┌─────────────────┐
│  Clarifier       │ ── Generates targeted questions about ambiguities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generator       │ ── Builds ToolSpec → renders Jinja2 templates → code
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tester          │ ── Auto-generates pytest tests, runs in isolation
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
  PASS      FAIL
    │         │
    ▼         ▼
 Install   ┌──────────┐
            │ Iterate   │ ── Analyze errors → fix code → re-test (max 5x)
            └──────────┘
```

### Key Components

| Component | What it does |
|-----------|-------------|
| **Engine** (`engine.py`) | Orchestrates the full pipeline, manages sessions |
| **Clarifier** (`clarifier.py`) | Analyzes descriptions for ambiguity, generates smart questions |
| **Generator** (`generator.py`) | Builds specs from descriptions, renders code via Jinja2 templates |
| **Tester** (`tester.py`) | Generates tests, runs pytest in temp directories, captures diagnostics |
| **Installer** (`installer.py`) | Installs to MCP config, `~/.local/bin/`, or local storage |
| **Storage** (`storage.py`) | Persists tools to `~/.forge/tools/` with metadata |
| **MCP Server** (`mcp_server.py`) | Exposes all capabilities as MCP tools for Claude Code |
| **CLI** (`cli.py`) | Interactive command-line interface with Rich output |

### Output Types

| Type | What you get | Use case |
|------|-------------|----------|
| `python` | A Python module with typed function | Import in your code |
| `cli` | A Click-based CLI script | Run from terminal |
| `mcp` | A FastMCP server tool | Use from Claude Code |

### The Iteration Loop (Key Differentiator)

When tests fail, Forge doesn't just show you the error. It:

1. Parses the pytest output for specific error types
2. Applies targeted fixes (missing imports, undefined variables, type mismatches)
3. Regenerates the code with fixes applied
4. Re-runs tests
5. Repeats up to 5 times

This means most simple tools work on the first try, and more complex ones converge within 2-3 iterations.

## Architecture

```
forge/
├── forge/
│   ├── __init__.py           # Package exports, version
│   ├── engine.py             # Core orchestration engine
│   ├── clarifier.py          # Clarification question generator
│   ├── generator.py          # Code generation from specs + templates
│   ├── tester.py             # Test generation & running
│   ├── installer.py          # Tool installation (MCP, CLI, local)
│   ├── mcp_server.py         # MCP server for Claude Code integration
│   ├── cli.py                # CLI interface (Click + Rich)
│   ├── storage.py            # Tool persistence (~/.forge/tools/)
│   ├── models.py             # Pydantic models for all data structures
│   └── templates/            # Jinja2 templates for code generation
│       ├── mcp_tool.py.jinja
│       ├── cli_tool.py.jinja
│       ├── python_function.py.jinja
│       └── test_tool.py.jinja
├── tests/                    # Project test suite
├── examples/                 # Usage examples
├── pyproject.toml
├── LICENSE                   # MIT
└── README.md
```

## MCP Tools Reference

When used as an MCP server, Forge exposes these tools:

| Tool | Description |
|------|-------------|
| `forge_create` | Start tool creation with description and output type |
| `forge_answer` | Answer clarification questions |
| `forge_generate` | Generate code and run tests |
| `forge_iterate` | Fix issues and regenerate |
| `forge_install` | Install the tool (mcp/cli/local) |
| `forge_status` | Check session state |
| `forge_list` | List all created tools |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with verbose output
pytest -v

# Lint
ruff check forge/
```

## License

MIT
