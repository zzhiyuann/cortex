# Contributing to Cortex

Thanks for your interest in contributing! Cortex is an MCP plug-in infrastructure layer for AI coding agents, organized as a Python monorepo.

## Setup

```bash
git clone https://github.com/zzhiyuann/cortex.git
cd cortex

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all packages in development mode
uv sync --all-packages
```

This installs all six packages (`cortex-cli`, `dispatcher`, `a2a-hub`, `forge`, `vibe-replay`, `memory`) into a shared virtual environment at `.venv/`.

## Project Structure

```
cortex/
├── packages/
│   ├── cortex-cli/        # Unified CLI orchestrator
│   ├── dispatcher/        # Telegram ↔ Agent bridge
│   ├── a2a-hub/           # Agent-to-Agent protocol
│   ├── forge/             # Tool generation engine
│   ├── vibe-replay/       # Session capture + replay
│   └── memory/            # Semantic memory store
├── docs/
│   └── ARCHITECTURE.md    # Deep architectural guide
├── examples/              # Walkthroughs and demos
├── pyproject.toml         # uv workspace root
└── uv.lock
```

Each package has its own `pyproject.toml`, `tests/` directory, and README. Packages are independently installable via pip but share development tooling through the workspace.

## Testing

```bash
# Run all tests (~600 across all packages)
uv run pytest

# Run tests for a single package
uv run pytest packages/forge/tests/ -v

# Run a specific test
uv run pytest packages/memory/tests/test_store.py::test_add_and_search -v

# Run with coverage
uv run pytest --cov=packages/ --cov-report=term-missing
```

Tests use `pytest` with `pytest-asyncio` for async code. All tests should pass without external services (Ollama, Telegram, etc.) — those dependencies are mocked.

## Linting

```bash
# Check
uv run ruff check packages/

# Auto-fix
uv run ruff check packages/ --fix
```

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. CI runs ruff on every PR.

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-change
   ```

2. **Make your changes** in the relevant package under `packages/`.

3. **Add or update tests** — every package has a `tests/` directory. New features need tests; bug fixes should include a regression test.

4. **Run tests and lint** before committing:
   ```bash
   uv run pytest packages/<your-package>/tests/ -v
   uv run ruff check packages/<your-package>/
   ```

5. **Commit with a clear message**:
   ```
   feat(forge): add timeout configuration for tool generation
   fix(memory): handle FTS5 special characters in search queries
   docs(cortex-cli): document init --skip flags
   ```

6. **Open a pull request** against `main`.

## Adding a Dependency

Edit the relevant package's `pyproject.toml`:

```bash
# Example: add httpx to forge
# Edit packages/forge/pyproject.toml, add to [project.dependencies]
uv sync --all-packages
```

Keep dependencies minimal. Each package should be lightweight and independently installable.

## Creating a New Package

1. Create `packages/my-package/` with:
   - `my_package/__init__.py`
   - `my_package/...` (source files)
   - `tests/test_*.py`
   - `pyproject.toml` (use existing packages as template)
   - `README.md`

2. The workspace auto-discovers packages under `packages/` via `[tool.uv.workspace]` in the root `pyproject.toml`.

3. Run `uv sync --all-packages` to register it.

## Design Principles

- **MCP-native** — agent-facing components expose MCP servers. One JSON config line to plug in.
- **Standalone packages** — each package is useful on its own. No forced coupling.
- **Graceful degradation** — if an optional service (Ollama, Telegram) is unavailable, the component still works with reduced functionality.
- **No external CDN** — HTML output (replays, docs) must be self-contained.
- **Stdlib where possible** — minimize dependencies. The Dispatcher's Telegram client uses only `urllib`.

## Questions?

Open an [issue](https://github.com/zzhiyuann/cortex/issues) or start a [discussion](https://github.com/zzhiyuann/cortex/discussions).
