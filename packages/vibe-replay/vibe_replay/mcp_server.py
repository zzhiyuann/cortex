"""Optional MCP server for Vibe Replay.

Allows Claude Code (or other MCP clients) to query past sessions,
get relevant learnings, and search session history.

Usage:
    python -m vibe_replay.mcp_server

Or add to ~/.claude/settings.json under mcpServers.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any

from .analyzer import aggregate_learnings, analyze_session
from .store import SessionStore


class MCPServer:
    """A minimal MCP server for Vibe Replay session queries.

    Implements the MCP protocol over stdin/stdout using JSON-RPC.
    """

    def __init__(self):
        self.store = SessionStore()
        self.tools = {
            "search_sessions": self._search_sessions,
            "get_learnings": self._get_learnings,
            "get_session_summary": self._get_session_summary,
            "list_recent_sessions": self._list_recent_sessions,
        }

    def run(self):
        """Run the MCP server, reading JSON-RPC messages from stdin."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "error": {"code": -32603, "message": str(e)},
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a JSON-RPC request."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "vibe-replay",
                        "version": "0.1.0",
                    },
                    "capabilities": {
                        "tools": {},
                    },
                },
            }

        elif method == "notifications/initialized":
            return None  # No response needed for notifications

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "search_sessions",
                            "description": "Search past coding sessions by keyword, date, or project.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search keyword",
                                    },
                                    "limit": {
                                        "type": "integer",
                                        "description": "Max results (default 10)",
                                        "default": 10,
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "get_learnings",
                            "description": "Get aggregated learnings and patterns from past sessions.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "limit": {
                                        "type": "integer",
                                        "description": "Max sessions to consider",
                                        "default": 30,
                                    },
                                },
                            },
                        },
                        {
                            "name": "get_session_summary",
                            "description": "Get detailed summary of a specific session.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "session_id": {
                                        "type": "string",
                                        "description": "Session ID (full or partial)",
                                    },
                                },
                                "required": ["session_id"],
                            },
                        },
                        {
                            "name": "list_recent_sessions",
                            "description": "List the most recent captured sessions.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "limit": {
                                        "type": "integer",
                                        "description": "Number of sessions",
                                        "default": 10,
                                    },
                                },
                            },
                        },
                    ]
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            handler = self.tools.get(tool_name)

            if not handler:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

            try:
                result = handler(arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2, default=str)}
                        ]
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    def _search_sessions(self, args: dict[str, Any]) -> list[dict]:
        """Search sessions by keyword."""
        query = args.get("query", "")
        limit = args.get("limit", 10)
        sessions = self.store.search_sessions(query, limit=limit)
        return [
            {
                "session_id": s.session_id,
                "project": s.project,
                "start_time": s.start_time.isoformat(),
                "event_count": s.event_count,
                "summary": s.summary,
            }
            for s in sessions
        ]

    def _get_learnings(self, args: dict[str, Any]) -> list[dict]:
        """Get aggregated learnings."""
        limit = args.get("limit", 30)
        insights = aggregate_learnings(self.store, limit=limit)
        return [
            {
                "type": i.insight_type.value,
                "title": i.title,
                "description": i.description,
                "confidence": i.confidence,
            }
            for i in insights
        ]

    def _get_session_summary(self, args: dict[str, Any]) -> dict:
        """Get session summary."""
        session_id = args.get("session_id", "")

        # Try to find session
        if not self.store.session_exists(session_id):
            # Try prefix match
            sessions_dir = self.store.sessions_dir
            matches = [
                d.name
                for d in sessions_dir.iterdir()
                if d.is_dir() and d.name.startswith(session_id)
            ]
            if matches:
                session_id = matches[0]
            else:
                return {"error": "Session not found"}

        replay = self.store.get_replay(session_id)
        if not replay:
            replay = analyze_session(self.store, session_id)

        return {
            "session_id": replay.metadata.session_id,
            "project": replay.metadata.project,
            "summary": replay.metadata.summary,
            "start_time": replay.metadata.start_time.isoformat(),
            "duration": replay.statistics.get("duration_human", "?"),
            "event_count": replay.metadata.event_count,
            "phases": [
                {
                    "phase": p.phase.value,
                    "event_count": p.event_count,
                    "summary": p.summary,
                }
                for p in replay.timeline
            ],
            "insights": [
                {
                    "type": i.insight_type.value,
                    "title": i.title,
                    "description": i.description,
                }
                for i in replay.insights
            ],
            "files_modified": replay.metadata.files_modified,
            "tools_used": replay.metadata.tools_used,
        }

    def _list_recent_sessions(self, args: dict[str, Any]) -> list[dict]:
        """List recent sessions."""
        limit = args.get("limit", 10)
        sessions = self.store.list_sessions(limit=limit)
        return [
            {
                "session_id": s.session_id,
                "project": s.project,
                "start_time": s.start_time.isoformat(),
                "event_count": s.event_count,
                "summary": s.summary,
                "duration_seconds": s.duration_seconds,
            }
            for s in sessions
        ]


def main():
    """Entry point for the MCP server."""
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()
