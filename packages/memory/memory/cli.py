"""CLI entry point for Cortex Memory.

Usage:
    memory recall "proactive affective agent"
    memory add "Claude Code dispatcher uses stream-json mode"
    memory list
    memory list --source bot
    memory forget 42
"""

from __future__ import annotations

import sys
import time


def _fmt_row(r: dict) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"]))
    tags_str = f" [{r['tags']}]" if r.get("tags") else ""
    score_str = (
        f" [score={r['score']:.3f}]"
        if r.get("score") is not None
        else ""
    )
    return f"[{r['id']}] {ts} ({r['source']}){score_str}{tags_str}\n  {r['content']}"


def cmd_recall(args: list[str]):
    """Search memories: memory recall <query> [--n N]"""
    if not args:
        print("Usage: memory recall <query> [--n N]", file=sys.stderr)
        sys.exit(1)

    n = 5
    query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--n" and i + 1 < len(args):
            try:
                n = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)
    from .store import MemoryStore
    store = MemoryStore()
    results = store.search(query, n=n)
    if not results:
        print("No memories found.")
        return
    for r in results:
        print(_fmt_row(r))


def cmd_add(args: list[str]):
    """Add a memory: memory add <content> [--source SOURCE] [--tags TAGS]"""
    if not args:
        print("Usage: memory add <content> [--source SOURCE] [--tags TAGS]",
              file=sys.stderr)
        sys.exit(1)

    source = "manual"
    tags = ""
    content_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--source" and i + 1 < len(args):
            source = args[i + 1]
            i += 2
        elif args[i] == "--tags" and i + 1 < len(args):
            tags = args[i + 1]
            i += 2
        else:
            content_parts.append(args[i])
            i += 1

    content = " ".join(content_parts)
    from .store import MemoryStore
    store = MemoryStore()
    memory_id = store.add(content, source=source, tags=tags)
    print(f"Stored memory id={memory_id}")


def cmd_list(args: list[str]):
    """List memories: memory list [--n N] [--source SOURCE]"""
    n = 20
    source = None
    i = 0
    while i < len(args):
        if args[i] == "--n" and i + 1 < len(args):
            try:
                n = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif args[i] == "--source" and i + 1 < len(args):
            source = args[i + 1]
            i += 2
        else:
            i += 1

    from .store import MemoryStore
    store = MemoryStore()
    results = store.list(n=n, source=source)
    if not results:
        print("No memories stored yet.")
        return
    for r in results:
        print(_fmt_row(r))


def cmd_forget(args: list[str]):
    """Delete a memory: memory forget <id>"""
    if not args:
        print("Usage: memory forget <id>", file=sys.stderr)
        sys.exit(1)
    try:
        memory_id = int(args[0])
    except ValueError:
        print(f"Invalid id: {args[0]}", file=sys.stderr)
        sys.exit(1)

    from .store import MemoryStore
    store = MemoryStore()
    deleted = store.delete(memory_id)
    if deleted:
        print(f"Deleted memory id={memory_id}")
    else:
        print(f"Memory id={memory_id} not found.")


def cmd_serve(_args: list[str]):
    """Start the MCP server: memory serve"""
    from .server import main as server_main
    server_main()


_COMMANDS = {
    "recall": cmd_recall,
    "add": cmd_add,
    "list": cmd_list,
    "forget": cmd_forget,
    "serve": cmd_serve,
}

_HELP = """Cortex Memory CLI

Commands:
  recall <query> [--n N]                Search memories semantically
  add <content> [--source S] [--tags T] Store a new memory
  list [--n N] [--source S]             List recent memories
  forget <id>                           Delete a memory by ID
  serve                                 Start the MCP server
"""


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(_HELP)
        return

    cmd = args[0]
    rest = args[1:]
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(_HELP, file=sys.stderr)
        sys.exit(1)

    try:
        handler(rest)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
