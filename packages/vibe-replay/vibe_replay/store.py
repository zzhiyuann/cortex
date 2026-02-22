"""Session storage for Vibe Replay.

Manages persistent storage of captured sessions, including:
- JSONL event logs (append-only)
- Session metadata
- Processed replay data
- SQLite index for cross-session queries
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import Event, SessionMetadata, SessionReplay


DEFAULT_BASE_DIR = Path.home() / ".vibe-replay"


class SessionStore:
    """Manages storage and retrieval of Vibe Replay sessions."""

    def __init__(self, base_dir: Path | str | None = None):
        """Initialize the session store.

        Args:
            base_dir: Base directory for storing sessions.
                     Defaults to ~/.vibe-replay.
        """
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.base_dir / "index.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite index database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    project TEXT DEFAULT '',
                    project_path TEXT DEFAULT '',
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    event_count INTEGER DEFAULT 0,
                    summary TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    files_modified TEXT DEFAULT '[]',
                    tools_used TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_start_time
                ON sessions(start_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_project
                ON sessions(project)
            """)
            conn.commit()

    def _session_dir(self, session_id: str) -> Path:
        """Get the directory for a specific session."""
        return self.sessions_dir / session_id

    def _events_path(self, session_id: str) -> Path:
        """Get the events JSONL file path for a session."""
        return self._session_dir(session_id) / "events.jsonl"

    def _metadata_path(self, session_id: str) -> Path:
        """Get the metadata JSON file path for a session."""
        return self._session_dir(session_id) / "metadata.json"

    def _replay_path(self, session_id: str) -> Path:
        """Get the replay JSON file path for a session."""
        return self._session_dir(session_id) / "replay.json"

    def ensure_session(self, session_id: str) -> Path:
        """Ensure a session directory exists.

        Args:
            session_id: The session identifier.

        Returns:
            Path to the session directory.
        """
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def append_event(self, event: Event) -> None:
        """Append an event to a session's event log.

        Args:
            event: The event to append.
        """
        self.ensure_session(event.session_id)
        events_path = self._events_path(event.session_id)

        with open(events_path, "a") as f:
            f.write(event.model_dump_jsonl() + "\n")

    def get_events(self, session_id: str) -> list[Event]:
        """Read all events for a session.

        Args:
            session_id: The session identifier.

        Returns:
            List of events in chronological order.
        """
        events_path = self._events_path(session_id)
        if not events_path.exists():
            return []

        events = []
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(Event.model_validate_json(line))
                    except Exception:
                        continue
        return events

    def iter_events(self, session_id: str) -> Iterator[Event]:
        """Iterate over events for a session without loading all into memory.

        Args:
            session_id: The session identifier.

        Yields:
            Events in chronological order.
        """
        events_path = self._events_path(session_id)
        if not events_path.exists():
            return

        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield Event.model_validate_json(line)
                    except Exception:
                        continue

    def event_count(self, session_id: str) -> int:
        """Get the number of events in a session without loading them all.

        Args:
            session_id: The session identifier.

        Returns:
            Number of events.
        """
        events_path = self._events_path(session_id)
        if not events_path.exists():
            return 0

        count = 0
        with open(events_path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def save_metadata(self, metadata: SessionMetadata) -> None:
        """Save session metadata.

        Args:
            metadata: Session metadata to save.
        """
        self.ensure_session(metadata.session_id)
        meta_path = self._metadata_path(metadata.session_id)

        with open(meta_path, "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        # Update SQLite index
        self._index_session(metadata)

    def get_metadata(self, session_id: str) -> SessionMetadata | None:
        """Read session metadata.

        Args:
            session_id: The session identifier.

        Returns:
            Session metadata or None if not found.
        """
        meta_path = self._metadata_path(session_id)
        if not meta_path.exists():
            return None

        try:
            return SessionMetadata.model_validate_json(meta_path.read_text())
        except Exception:
            return None

    def save_replay(self, replay: SessionReplay) -> None:
        """Save processed replay data.

        Args:
            replay: Session replay data to save.
        """
        self.ensure_session(replay.metadata.session_id)
        replay_path = self._replay_path(replay.metadata.session_id)

        with open(replay_path, "w") as f:
            f.write(replay.model_dump_json(indent=2))

    def get_replay(self, session_id: str) -> SessionReplay | None:
        """Read processed replay data.

        Args:
            session_id: The session identifier.

        Returns:
            Session replay data or None if not found.
        """
        replay_path = self._replay_path(session_id)
        if not replay_path.exists():
            return None

        try:
            return SessionReplay.model_validate_json(replay_path.read_text())
        except Exception:
            return None

    def _index_session(self, metadata: SessionMetadata) -> None:
        """Update the SQLite index with session metadata."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, project, project_path, start_time, end_time,
                 duration_seconds, event_count, summary, tags,
                 files_modified, tools_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.session_id,
                    metadata.project,
                    metadata.project_path,
                    metadata.start_time.isoformat(),
                    metadata.end_time.isoformat() if metadata.end_time else None,
                    metadata.duration_seconds,
                    metadata.event_count,
                    metadata.summary,
                    json.dumps(metadata.tags),
                    json.dumps(metadata.files_modified),
                    json.dumps(metadata.tools_used),
                ),
            )
            conn.commit()

    def list_sessions(
        self,
        project: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionMetadata]:
        """List sessions from the index.

        Args:
            project: Filter by project name.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.

        Returns:
            List of session metadata, newest first.
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row

            if project:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE project = ?
                    ORDER BY start_time DESC
                    LIMIT ? OFFSET ?
                """,
                    (project, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    ORDER BY start_time DESC
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                ).fetchall()

        sessions = []
        for row in rows:
            try:
                sessions.append(
                    SessionMetadata(
                        session_id=row["session_id"],
                        project=row["project"],
                        project_path=row["project_path"],
                        start_time=datetime.fromisoformat(row["start_time"])
                        if row["start_time"]
                        else datetime.now(),
                        end_time=datetime.fromisoformat(row["end_time"])
                        if row["end_time"]
                        else None,
                        duration_seconds=row["duration_seconds"],
                        event_count=row["event_count"],
                        summary=row["summary"],
                        tags=json.loads(row["tags"]) if row["tags"] else [],
                        files_modified=json.loads(row["files_modified"])
                        if row["files_modified"]
                        else [],
                        tools_used=json.loads(row["tools_used"])
                        if row["tools_used"]
                        else {},
                    )
                )
            except Exception:
                continue

        return sessions

    def search_sessions(self, query: str, limit: int = 20) -> list[SessionMetadata]:
        """Search sessions by keyword.

        Args:
            query: Search keyword.
            limit: Maximum results.

        Returns:
            Matching sessions.
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            pattern = f"%{query}%"
            rows = conn.execute(
                """
                SELECT * FROM sessions
                WHERE summary LIKE ? OR project LIKE ? OR tags LIKE ?
                ORDER BY start_time DESC
                LIMIT ?
            """,
                (pattern, pattern, pattern, limit),
            ).fetchall()

        sessions = []
        for row in rows:
            try:
                sessions.append(
                    SessionMetadata(
                        session_id=row["session_id"],
                        project=row["project"],
                        project_path=row["project_path"],
                        start_time=datetime.fromisoformat(row["start_time"])
                        if row["start_time"]
                        else datetime.now(),
                        end_time=datetime.fromisoformat(row["end_time"])
                        if row["end_time"]
                        else None,
                        duration_seconds=row["duration_seconds"],
                        event_count=row["event_count"],
                        summary=row["summary"],
                        tags=json.loads(row["tags"]) if row["tags"] else [],
                        files_modified=json.loads(row["files_modified"])
                        if row["files_modified"]
                        else [],
                        tools_used=json.loads(row["tools_used"])
                        if row["tools_used"]
                        else {},
                    )
                )
            except Exception:
                continue

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data.

        Args:
            session_id: The session identifier.

        Returns:
            True if session was deleted, False if not found.
        """
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return False

        import shutil

        shutil.rmtree(session_dir)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            conn.commit()

        return True

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            session_id: The session identifier.

        Returns:
            True if the session directory exists.
        """
        return self._session_dir(session_id).exists()
