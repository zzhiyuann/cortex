"""MemoryStore — SQLite-backed memory with semantic search and FTS5 fallback.

Storage: ~/.cortex/memory.db
Embeddings: mxbai-embed-large via Ollama (http://localhost:11434)
Search: cosine similarity in-memory; falls back to FTS5 keyword search
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("cortex.memory")

_DEFAULT_DB = Path.home() / ".cortex" / "memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    tags        TEXT    DEFAULT '',
    created_at  REAL    NOT NULL,
    embedding   BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    tags,
    content=memories,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
"""


def _pack_embedding(arr) -> bytes:
    """Serialize a float32 numpy array to raw bytes."""
    try:
        import numpy as np
        arr = np.asarray(arr, dtype=np.float32)
        return struct.pack(f"{len(arr)}f", *arr)
    except ImportError:
        # Fallback: use pure Python list
        floats = list(arr)
        return struct.pack(f"{len(floats)}f", *floats)


def _unpack_embedding(blob: bytes):
    """Deserialize raw bytes back to a float32 array."""
    n = len(blob) // 4
    values = struct.unpack(f"{n}f", blob)
    try:
        import numpy as np
        return np.array(values, dtype=np.float32)
    except ImportError:
        return list(values)


def _cosine_pure(a, b) -> float:
    """Pure Python cosine similarity — used when numpy is unavailable."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryStore:
    """SQLite-backed memory store with semantic search and FTS5 keyword fallback."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        log.debug("MemoryStore initialized at %s", self.db_path)

    def close(self):
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def add(self, content: str, source: str, tags: str | None = None) -> int:
        """Embed content and store as a new memory. Returns the new row ID."""
        content = content.strip()
        if not content:
            raise ValueError("content must not be empty")

        tags_str = (tags or "").strip()
        embedding = self._embed(content)
        blob = _pack_embedding(embedding) if embedding is not None else None

        cur = self._conn.execute(
            "INSERT INTO memories (content, source, tags, created_at, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (content, source, tags_str, time.time(), blob),
        )
        self._conn.commit()
        memory_id = cur.lastrowid
        log.debug("stored memory id=%d source=%s", memory_id, source)
        return memory_id

    def search(self, query: str, n: int = 5, threshold: float = 0.5) -> list[dict]:
        """Semantic search using cosine similarity.

        Falls back to FTS5 keyword search when Ollama is unavailable or
        no memories have embeddings.
        """
        query = query.strip()
        if not query:
            return []

        # Try semantic search first
        query_emb = self._embed(query)
        if query_emb is not None:
            results = self._semantic_search(query_emb, n, threshold)
            if results:
                return results
            # No results above threshold — fall through to FTS5

        # FTS5 keyword fallback
        return self._fts_search(query, n)

    def list(self, n: int = 20, source: str | None = None) -> list[dict]:
        """List recent memories, optionally filtered by source."""
        if source:
            rows = self._conn.execute(
                "SELECT id, content, source, tags, created_at FROM memories "
                "WHERE source = ? ORDER BY created_at DESC LIMIT ?",
                (source, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, content, source, tags, created_at FROM memories "
                "ORDER BY created_at DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID. Returns True if a row was deleted."""
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def _embed(self, text: str):
        """Call Ollama to get an embedding vector. Returns None on any error."""
        try:
            payload = json.dumps(
                {"model": "mxbai-embed-large", "prompt": text}
            ).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            embedding = data.get("embedding")
            if not embedding:
                return None
            try:
                import numpy as np
                return np.array(embedding, dtype=np.float32)
            except ImportError:
                return list(embedding)
        except Exception as exc:
            log.debug("Ollama embedding unavailable: %s", exc)
            return None

    def _cosine(self, a, b) -> float:
        """Cosine similarity between two vectors."""
        try:
            import numpy as np
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        except ImportError:
            return _cosine_pure(a, b)

    def _semantic_search(self, query_emb, n: int, threshold: float) -> list[dict]:
        """In-memory cosine similarity search over all stored embeddings."""
        rows = self._conn.execute(
            "SELECT id, content, source, tags, created_at, embedding FROM memories "
            "WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        scored: list[tuple[float, Any]] = []
        for row in rows:
            try:
                emb = _unpack_embedding(row["embedding"])
                score = self._cosine(query_emb, emb)
                if score >= threshold:
                    scored.append((score, row))
            except Exception as exc:
                log.debug("failed to score row id=%d: %s", row["id"], exc)

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, row in scored[:n]:
            d = self._row_to_dict(row)
            d["score"] = round(score, 4)
            results.append(d)
        return results

    def _fts_search(self, query: str, n: int) -> list[dict]:
        """FTS5 keyword search fallback."""
        try:
            # Sanitize query for FTS5 — strip special characters that break parsing
            safe_query = " ".join(
                word for word in query.split()
                if word.replace("-", "").replace("_", "").isalnum()
            ) or query
            rows = self._conn.execute(
                "SELECT m.id, m.content, m.source, m.tags, m.created_at "
                "FROM memories_fts fts "
                "JOIN memories m ON m.id = fts.rowid "
                "WHERE memories_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (safe_query, n),
            ).fetchall()
            results = [self._row_to_dict(r) for r in rows]
            for r in results:
                r["score"] = None  # no numeric score for FTS
            return results
        except Exception as exc:
            log.warning("FTS5 search failed: %s", exc)
            # Last resort: LIKE search
            like = f"%{query}%"
            rows = self._conn.execute(
                "SELECT id, content, source, tags, created_at FROM memories "
                "WHERE content LIKE ? OR tags LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (like, like, n),
            ).fetchall()
            results = [self._row_to_dict(r) for r in rows]
            for r in results:
                r["score"] = None
            return results

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "id": row["id"],
            "content": row["content"],
            "source": row["source"],
            "tags": row["tags"],
            "created_at": row["created_at"],
        }
