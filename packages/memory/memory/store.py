"""MemoryStore — SQLite-backed memory with semantic search and FTS5 fallback.

Storage: ~/.cortex/memory.db
Embeddings: mxbai-embed-large via Ollama (http://localhost:11434)
Search: cosine similarity in-memory; falls back to FTS5 keyword search

Features:
- Categories/namespaces for organizing memories (e.g., "project:ryanhub", "personal")
- TTL (time-to-live) support for temporary memories with automatic cleanup
- Export/import for portability (JSON format)
- Robust error handling: retry on transient failures, file locking, corruption recovery
- Graceful degradation: works without Ollama (keyword search only)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
import urllib.error
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
    category    TEXT    DEFAULT '',
    created_at  REAL    NOT NULL,
    expires_at  REAL    DEFAULT NULL,
    embedding   BLOB
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at) WHERE expires_at IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    tags,
    category,
    content=memories,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags, category)
    VALUES (new.id, new.content, new.tags, new.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, category)
    VALUES ('delete', old.id, old.content, old.tags, old.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, category)
    VALUES ('delete', old.id, old.content, old.tags, old.category);
    INSERT INTO memories_fts(rowid, content, tags, category)
    VALUES (new.id, new.content, new.tags, new.category);
END;
"""

# Migration: add 'category' and 'expires_at' columns if missing
_MIGRATION_SQL = [
    "ALTER TABLE memories ADD COLUMN category TEXT DEFAULT ''",
    "ALTER TABLE memories ADD COLUMN expires_at REAL DEFAULT NULL",
]


class EmbeddingError(Exception):
    """Raised when embedding generation fails due to a recoverable error."""


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
    """SQLite-backed memory store with semantic search, FTS5 fallback,
    categories, TTL, and export/import support."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "mxbai-embed-large",
        embed_timeout: float = 10.0,
        embed_retries: int = 2,
    ):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ollama_url = ollama_url.rstrip("/")
        self._embed_model = embed_model
        self._embed_timeout = embed_timeout
        self._embed_retries = embed_retries
        self._embedding_available: bool | None = None  # cached availability

        self._conn = self._connect()
        self._apply_schema()
        log.debug("MemoryStore initialized at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode for safe concurrent access."""
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            return conn
        except sqlite3.DatabaseError as exc:
            log.warning("Database may be corrupted: %s. Attempting recovery.", exc)
            return self._recover_database()

    def _recover_database(self) -> sqlite3.Connection:
        """Attempt to recover a corrupted database by rebuilding from scratch."""
        backup_path = self.db_path.with_suffix(".db.bak")
        try:
            if self.db_path.exists():
                self.db_path.rename(backup_path)
                log.info("Backed up corrupted database to %s", backup_path)
        except OSError as exc:
            log.error("Failed to back up corrupted database: %s", exc)

        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _apply_schema(self):
        """Apply schema and run migrations for new columns."""
        try:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.OperationalError:
            # Schema may already exist with different FTS columns; try migrations
            pass

        for sql in _MIGRATION_SQL:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Rebuild FTS index if needed (handles schema changes to FTS table)
        try:
            self._conn.execute(
                "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def close(self):
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    # ---- Core CRUD ----

    def add(
        self,
        content: str,
        source: str,
        tags: str | None = None,
        category: str | None = None,
        ttl: int | None = None,
    ) -> int:
        """Embed content and store as a new memory. Returns the new row ID.

        Args:
            content: The text content to remember.
            source: Origin of the memory ('manual', 'bot', 'cli', etc.).
            tags: Optional comma-separated tags.
            category: Optional category/namespace (e.g., 'project:ryanhub', 'personal').
            ttl: Optional time-to-live in seconds. Memory auto-expires after this.
        """
        content = content.strip()
        if not content:
            raise ValueError("content must not be empty")

        tags_str = (tags or "").strip()
        category_str = (category or "").strip()
        now = time.time()
        expires_at = now + ttl if ttl and ttl > 0 else None

        embedding = self._embed(content)
        blob = _pack_embedding(embedding) if embedding is not None else None

        try:
            cur = self._conn.execute(
                "INSERT INTO memories (content, source, tags, category, created_at, expires_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (content, source, tags_str, category_str, now, expires_at, blob),
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            log.error("Failed to insert memory: %s", exc)
            raise

        memory_id = cur.lastrowid
        log.debug("stored memory id=%d source=%s category=%s", memory_id, source, category_str)
        return memory_id

    def search(
        self,
        query: str,
        n: int = 5,
        threshold: float = 0.5,
        category: str | None = None,
    ) -> list[dict]:
        """Semantic search using cosine similarity.

        Falls back to FTS5 keyword search when Ollama is unavailable or
        no memories have embeddings.

        Args:
            query: Search query text.
            n: Maximum number of results.
            threshold: Minimum cosine similarity for semantic results.
            category: Filter results by category.
        """
        query = query.strip()
        if not query:
            return []

        # Clean expired memories first
        self._cleanup_expired()

        # Try semantic search first
        query_emb = self._embed(query)
        if query_emb is not None:
            results = self._semantic_search(query_emb, n, threshold, category)
            if results:
                return results
            # No results above threshold — fall through to FTS5

        # FTS5 keyword fallback
        return self._fts_search(query, n, category)

    def list(
        self,
        n: int = 20,
        source: str | None = None,
        category: str | None = None,
    ) -> list[dict]:
        """List recent memories, optionally filtered by source and/or category."""
        self._cleanup_expired()

        clauses = ["(expires_at IS NULL OR expires_at > ?)"]
        params: list[Any] = [time.time()]

        if source:
            clauses.append("source = ?")
            params.append(source)
        if category:
            clauses.append("category = ?")
            params.append(category)

        where = " AND ".join(clauses)
        params.append(n)

        rows = self._conn.execute(
            f"SELECT id, content, source, tags, category, created_at, expires_at "
            f"FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID. Returns True if a row was deleted."""
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def get(self, memory_id: int) -> dict | None:
        """Get a single memory by ID."""
        row = self._conn.execute(
            "SELECT id, content, source, tags, category, created_at, expires_at "
            "FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def count(self, category: str | None = None) -> int:
        """Count memories, optionally filtered by category."""
        if category:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE category = ? "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (category, time.time()),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM memories "
                "WHERE expires_at IS NULL OR expires_at > ?",
                (time.time(),),
            ).fetchone()
        return row["cnt"] if row else 0

    def categories(self) -> list[str]:
        """Return a list of distinct categories that have active memories."""
        rows = self._conn.execute(
            "SELECT DISTINCT category FROM memories "
            "WHERE category != '' AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY category",
            (time.time(),),
        ).fetchall()
        return [r["category"] for r in rows]

    # ---- TTL ----

    def _cleanup_expired(self):
        """Delete memories past their TTL."""
        try:
            deleted = self._conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (time.time(),),
            )
            if deleted.rowcount > 0:
                self._conn.commit()
                log.debug("cleaned up %d expired memories", deleted.rowcount)
        except sqlite3.OperationalError as exc:
            log.warning("Failed to cleanup expired memories: %s", exc)

    # ---- Export / Import ----

    def export_memories(
        self,
        path: str | Path | None = None,
        category: str | None = None,
    ) -> str:
        """Export memories to a JSON file. Returns the file path.

        Args:
            path: Output file path. Defaults to ~/.cortex/memory-export.json.
            category: If set, only export memories in this category.
        """
        if path is None:
            path = self.db_path.parent / "memory-export.json"
        path = Path(path)

        clauses = ["(expires_at IS NULL OR expires_at > ?)"]
        params: list[Any] = [time.time()]

        if category:
            clauses.append("category = ?")
            params.append(category)

        where = " AND ".join(clauses)

        rows = self._conn.execute(
            f"SELECT id, content, source, tags, category, created_at, expires_at "
            f"FROM memories WHERE {where} ORDER BY created_at ASC",
            params,
        ).fetchall()

        memories = [self._row_to_dict(r) for r in rows]
        export_data = {
            "version": 1,
            "exported_at": time.time(),
            "count": len(memories),
            "memories": memories,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(export_data, indent=2, default=str))
        log.info("exported %d memories to %s", len(memories), path)
        return str(path)

    def import_memories(
        self,
        path: str | Path,
        merge: bool = True,
    ) -> int:
        """Import memories from a JSON file. Returns count of imported memories.

        Args:
            path: Input file path.
            merge: If True, skip duplicates (same content+source). If False, import all.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Invalid import file: {exc}")

        memories = data.get("memories", [])
        if not memories:
            return 0

        imported = 0
        for mem in memories:
            content = mem.get("content", "").strip()
            if not content:
                continue

            source = mem.get("source", "import")
            tags = mem.get("tags", "")
            category = mem.get("category", "")

            if merge:
                # Check for duplicates by content + source
                existing = self._conn.execute(
                    "SELECT id FROM memories WHERE content = ? AND source = ? LIMIT 1",
                    (content, source),
                ).fetchone()
                if existing:
                    continue

            embedding = self._embed(content)
            blob = _pack_embedding(embedding) if embedding is not None else None

            self._conn.execute(
                "INSERT INTO memories (content, source, tags, category, created_at, expires_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (content, source, tags, category, time.time(), None, blob),
            )
            imported += 1

        self._conn.commit()
        log.info("imported %d memories from %s", imported, path)
        return imported

    # ---- Embedding ----

    def _embed(self, text: str):
        """Call Ollama to get an embedding vector. Returns None on any error.

        Implements retry logic for transient network errors and rate limits.
        Caches embedding availability to avoid repeated failed connection attempts.
        """
        # Fast path: if we've already determined Ollama is unavailable, skip
        if self._embedding_available is False:
            return None

        last_error = None
        for attempt in range(self._embed_retries + 1):
            try:
                payload = json.dumps(
                    {"model": self._embed_model, "prompt": text}
                ).encode()
                req = urllib.request.Request(
                    f"{self._ollama_url}/api/embeddings",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp = urllib.request.urlopen(req, timeout=self._embed_timeout)
                status = resp.getcode()

                if status == 429:
                    # Rate limited — wait and retry
                    wait_time = min(2 ** attempt, 10)
                    log.warning("Ollama rate limited, waiting %ds before retry", wait_time)
                    time.sleep(wait_time)
                    continue

                data = json.loads(resp.read())
                embedding = data.get("embedding")
                if not embedding:
                    log.debug("Ollama returned empty embedding")
                    return None

                self._embedding_available = True
                try:
                    import numpy as np
                    return np.array(embedding, dtype=np.float32)
                except ImportError:
                    return list(embedding)

            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429:
                    wait_time = min(2 ** attempt, 10)
                    log.warning("Ollama rate limited (429), waiting %ds", wait_time)
                    time.sleep(wait_time)
                    continue
                elif exc.code >= 500:
                    log.warning("Ollama server error (%d), attempt %d/%d",
                                exc.code, attempt + 1, self._embed_retries + 1)
                    if attempt < self._embed_retries:
                        time.sleep(1)
                    continue
                else:
                    log.debug("Ollama HTTP error %d: %s", exc.code, exc)
                    break

            except (urllib.error.URLError, ConnectionError, OSError) as exc:
                last_error = exc
                log.debug("Ollama connection failed (attempt %d/%d): %s",
                          attempt + 1, self._embed_retries + 1, exc)
                if attempt < self._embed_retries:
                    time.sleep(0.5)
                continue

            except Exception as exc:
                last_error = exc
                log.debug("Ollama embedding error: %s", exc)
                break

        # All retries exhausted
        if last_error:
            log.debug("Ollama embedding unavailable after %d attempts: %s",
                       self._embed_retries + 1, last_error)
            # Cache unavailability to avoid hammering the endpoint
            if isinstance(last_error, (urllib.error.URLError, ConnectionError, OSError)):
                self._embedding_available = False

        return None

    def reset_embedding_cache(self):
        """Reset the embedding availability cache. Useful after Ollama starts up."""
        self._embedding_available = None

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

    def _semantic_search(
        self, query_emb, n: int, threshold: float, category: str | None = None
    ) -> list[dict]:
        """In-memory cosine similarity search over all stored embeddings."""
        if category:
            rows = self._conn.execute(
                "SELECT id, content, source, tags, category, created_at, expires_at, embedding "
                "FROM memories WHERE embedding IS NOT NULL AND category = ? "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (category, time.time()),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, content, source, tags, category, created_at, expires_at, embedding "
                "FROM memories WHERE embedding IS NOT NULL "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (time.time(),),
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

    def _fts_search(self, query: str, n: int, category: str | None = None) -> list[dict]:
        """FTS5 keyword search fallback."""
        try:
            # Sanitize query for FTS5 — strip special characters that break parsing
            safe_query = " ".join(
                word for word in query.split()
                if word.replace("-", "").replace("_", "").isalnum()
            ) or query

            if category:
                rows = self._conn.execute(
                    "SELECT m.id, m.content, m.source, m.tags, m.category, m.created_at, m.expires_at "
                    "FROM memories_fts fts "
                    "JOIN memories m ON m.id = fts.rowid "
                    "WHERE memories_fts MATCH ? AND m.category = ? "
                    "AND (m.expires_at IS NULL OR m.expires_at > ?) "
                    "ORDER BY rank LIMIT ?",
                    (safe_query, category, time.time(), n),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT m.id, m.content, m.source, m.tags, m.category, m.created_at, m.expires_at "
                    "FROM memories_fts fts "
                    "JOIN memories m ON m.id = fts.rowid "
                    "WHERE memories_fts MATCH ? "
                    "AND (m.expires_at IS NULL OR m.expires_at > ?) "
                    "ORDER BY rank LIMIT ?",
                    (safe_query, time.time(), n),
                ).fetchall()

            results = [self._row_to_dict(r) for r in rows]
            for r in results:
                r["score"] = None  # no numeric score for FTS
            return results

        except Exception as exc:
            log.warning("FTS5 search failed: %s", exc)
            # Last resort: LIKE search
            like = f"%{query}%"
            params: list[Any] = [like, like]
            extra_clause = ""
            if category:
                extra_clause = "AND category = ? "
                params.append(category)
            params.extend([time.time(), n])

            rows = self._conn.execute(
                f"SELECT id, content, source, tags, category, created_at, expires_at "
                f"FROM memories "
                f"WHERE (content LIKE ? OR tags LIKE ?) {extra_clause}"
                f"AND (expires_at IS NULL OR expires_at > ?) "
                f"ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
            results = [self._row_to_dict(r) for r in rows]
            for r in results:
                r["score"] = None
            return results

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = {
            "id": row["id"],
            "content": row["content"],
            "source": row["source"],
            "tags": row["tags"],
            "created_at": row["created_at"],
        }
        # Include category and expires_at if available
        try:
            d["category"] = row["category"]
        except (IndexError, KeyError):
            d["category"] = ""
        try:
            d["expires_at"] = row["expires_at"]
        except (IndexError, KeyError):
            d["expires_at"] = None
        return d
