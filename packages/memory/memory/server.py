"""MCP server for Cortex Memory — exposes remember/recall/forget/list tools.

Supports categories/namespaces, TTL for temporary memories, and export/import.
"""

from __future__ import annotations

import logging

log = logging.getLogger("cortex.memory")


def _run_server():
    """Start the MCP server using the mcp library."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        try:
            from mcp import FastMCP  # type: ignore
        except ImportError:
            raise ImportError(
                "mcp library not installed. Run: pip install mcp"
            )

    from .store import MemoryStore

    store = MemoryStore()
    mcp = FastMCP("cortex-memory")

    @mcp.tool()
    def remember(
        content: str,
        source: str = "manual",
        tags: str = "",
        category: str = "",
        ttl: int = 0,
    ) -> str:
        """Store a fact directly into long-term memory.

        Args:
            content: The fact or information to remember.
            source: Origin of the memory ('manual', 'bot', 'cli', etc.).
            tags: Optional comma-separated tags for categorization.
            category: Optional category/namespace (e.g., 'project:ryanhub', 'personal', 'technical').
            ttl: Time-to-live in seconds. 0 means permanent (default).
        """
        try:
            memory_id = store.add(
                content=content,
                source=source,
                tags=tags,
                category=category if category else None,
                ttl=ttl if ttl > 0 else None,
            )
            ttl_str = f" (expires in {ttl}s)" if ttl > 0 else ""
            cat_str = f" [{category}]" if category else ""
            return f"Stored memory id={memory_id}{cat_str}{ttl_str}"
        except Exception as exc:
            return f"Error storing memory: {exc}"

    @mcp.tool()
    def recall(query: str, category: str = "", n: int = 5) -> str:
        """Search long-term memory for relevant facts.

        Args:
            query: The search query (semantic or keyword).
            category: Optional category filter (e.g., 'project:ryanhub').
            n: Maximum number of results to return (default 5).
        """
        try:
            results = store.search(
                query=query,
                n=n,
                category=category if category else None,
            )
            if not results:
                return "No memories found."
            lines = []
            for r in results:
                score_str = f" [score={r['score']:.3f}]" if r.get("score") is not None else ""
                tags_str = f" [tags: {r['tags']}]" if r.get("tags") else ""
                cat_str = f" [{r['category']}]" if r.get("category") else ""
                lines.append(f"[{r['id']}]{score_str}{cat_str}{tags_str} {r['content']}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error searching memories: {exc}"

    @mcp.tool()
    def forget(id: int) -> str:  # noqa: A002
        """Delete a memory by its ID.

        Args:
            id: The numeric ID of the memory to delete.
        """
        try:
            deleted = store.delete(id)
            if deleted:
                return f"Deleted memory id={id}"
            return f"Memory id={id} not found."
        except Exception as exc:
            return f"Error deleting memory: {exc}"

    @mcp.tool()
    def list_memories(
        category: str = "",
        n: int = 20,
        source: str = "",
    ) -> str:
        """List recent memories, optionally filtered by category and/or source.

        Args:
            category: Filter by category (e.g., 'project:ryanhub', 'personal', or '' for all).
            n: Maximum number of results (default 20).
            source: Filter by source ('bot', 'cli', 'manual', or '' for all).
        """
        try:
            results = store.list(
                n=n,
                source=source if source else None,
                category=category if category else None,
            )
            if not results:
                return "No memories stored yet."
            lines = []
            for r in results:
                import time
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"]))
                tags_str = f" [{r['tags']}]" if r.get("tags") else ""
                cat_str = f" [{r['category']}]" if r.get("category") else ""
                ttl_str = ""
                if r.get("expires_at"):
                    remaining = r["expires_at"] - time.time()
                    if remaining > 0:
                        ttl_str = f" (expires in {int(remaining)}s)"
                    else:
                        ttl_str = " (expired)"
                lines.append(
                    f"[{r['id']}] {ts} ({r['source']}){cat_str}{tags_str}{ttl_str}: {r['content']}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Error listing memories: {exc}"

    log.info("Starting cortex-memory MCP server")
    mcp.run()


def main():
    """Entry point for the MCP server process."""
    logging.basicConfig(level=logging.INFO)
    _run_server()


if __name__ == "__main__":
    main()
