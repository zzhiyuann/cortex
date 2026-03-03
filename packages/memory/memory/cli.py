"""CLI entry point for Cortex Memory.

Usage:
    memory recall "proactive affective agent"
    memory recall "proactive affective agent" --category project:ryanhub
    memory add "Claude Code dispatcher uses stream-json mode"
    memory add "temp note" --ttl 3600
    memory list
    memory list --source bot --category personal
    memory forget 42
    memory categories
    memory export --path /tmp/memories.json
    memory import /tmp/memories.json
    memory count
"""

from __future__ import annotations

import sys
import time


def _fmt_row(r: dict) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"]))
    tags_str = f" [{r['tags']}]" if r.get("tags") else ""
    cat_str = f" [{r['category']}]" if r.get("category") else ""
    score_str = (
        f" [score={r['score']:.3f}]"
        if r.get("score") is not None
        else ""
    )
    ttl_str = ""
    if r.get("expires_at"):
        remaining = r["expires_at"] - time.time()
        if remaining > 0:
            ttl_str = f" (expires in {int(remaining)}s)"
        else:
            ttl_str = " (expired)"
    return f"[{r['id']}] {ts} ({r['source']}){score_str}{cat_str}{tags_str}{ttl_str}\n  {r['content']}"


def cmd_recall(args: list[str]):
    """Search memories: memory recall <query> [--n N] [--category CAT]"""
    if not args:
        print("Usage: memory recall <query> [--n N] [--category CAT]", file=sys.stderr)
        sys.exit(1)

    n = 5
    category = None
    query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--n" and i + 1 < len(args):
            try:
                n = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)
    from .store import MemoryStore
    store = MemoryStore()
    results = store.search(query, n=n, category=category)
    if not results:
        print("No memories found.")
        return
    for r in results:
        print(_fmt_row(r))


def cmd_add(args: list[str]):
    """Add a memory: memory add <content> [--source SOURCE] [--tags TAGS] [--category CAT] [--ttl SECONDS]"""
    if not args:
        print(
            "Usage: memory add <content> [--source SOURCE] [--tags TAGS] "
            "[--category CAT] [--ttl SECONDS]",
            file=sys.stderr,
        )
        sys.exit(1)

    source = "manual"
    tags = ""
    category = None
    ttl = None
    content_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--source" and i + 1 < len(args):
            source = args[i + 1]
            i += 2
        elif args[i] == "--tags" and i + 1 < len(args):
            tags = args[i + 1]
            i += 2
        elif args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif args[i] == "--ttl" and i + 1 < len(args):
            try:
                ttl = int(args[i + 1])
            except ValueError:
                print(f"Invalid TTL value: {args[i + 1]}", file=sys.stderr)
                sys.exit(1)
            i += 2
        else:
            content_parts.append(args[i])
            i += 1

    content = " ".join(content_parts)
    from .store import MemoryStore
    store = MemoryStore()
    memory_id = store.add(content, source=source, tags=tags, category=category, ttl=ttl)
    ttl_str = f" (expires in {ttl}s)" if ttl else ""
    cat_str = f" [{category}]" if category else ""
    print(f"Stored memory id={memory_id}{cat_str}{ttl_str}")


def cmd_list(args: list[str]):
    """List memories: memory list [--n N] [--source SOURCE] [--category CAT]"""
    n = 20
    source = None
    category = None
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
        elif args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        else:
            i += 1

    from .store import MemoryStore
    store = MemoryStore()
    results = store.list(n=n, source=source, category=category)
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


def cmd_categories(_args: list[str]):
    """List memory categories: memory categories"""
    from .store import MemoryStore
    store = MemoryStore()
    cats = store.categories()
    if not cats:
        print("No categories found.")
        return
    for cat in cats:
        count = store.count(category=cat)
        print(f"  {cat}: {count} memories")


def cmd_count(args: list[str]):
    """Count memories: memory count [--category CAT]"""
    category = None
    i = 0
    while i < len(args):
        if args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        else:
            i += 1

    from .store import MemoryStore
    store = MemoryStore()
    total = store.count(category=category)
    if category:
        print(f"{total} memories in category '{category}'")
    else:
        print(f"{total} total memories")


def cmd_export(args: list[str]):
    """Export memories: memory export [--path FILE] [--category CAT]"""
    path = None
    category = None
    i = 0
    while i < len(args):
        if args[i] == "--path" and i + 1 < len(args):
            path = args[i + 1]
            i += 2
        elif args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        else:
            i += 1

    from .store import MemoryStore
    store = MemoryStore()
    out_path = store.export_memories(path=path, category=category)
    print(f"Exported to {out_path}")


def cmd_import(args: list[str]):
    """Import memories: memory import <path> [--no-merge]"""
    if not args:
        print("Usage: memory import <path> [--no-merge]", file=sys.stderr)
        sys.exit(1)

    path = args[0]
    merge = True
    if "--no-merge" in args:
        merge = False

    from .store import MemoryStore
    store = MemoryStore()
    count = store.import_memories(path=path, merge=merge)
    print(f"Imported {count} memories from {path}")


def cmd_serve(_args: list[str]):
    """Start the MCP server: memory serve"""
    from .server import main as server_main
    server_main()


_COMMANDS = {
    "recall": cmd_recall,
    "add": cmd_add,
    "list": cmd_list,
    "forget": cmd_forget,
    "categories": cmd_categories,
    "count": cmd_count,
    "export": cmd_export,
    "import": cmd_import,
    "serve": cmd_serve,
}

_HELP = """Cortex Memory CLI

Commands:
  recall <query> [--n N] [--category CAT]                      Search memories semantically
  add <content> [--source S] [--tags T] [--category C] [--ttl S]  Store a new memory
  list [--n N] [--source S] [--category C]                      List recent memories
  forget <id>                                                    Delete a memory by ID
  categories                                                     List memory categories
  count [--category C]                                           Count memories
  export [--path FILE] [--category C]                            Export memories to JSON
  import <path> [--no-merge]                                     Import memories from JSON
  serve                                                          Start the MCP server
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
