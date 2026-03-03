"""Tests for MemoryStore — add, search, list, delete, FTS5 fallback,
categories, TTL, export/import, and error handling."""

from __future__ import annotations

import json
import struct
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path, **kwargs):
    """Create a MemoryStore backed by a temp file."""
    from memory.store import MemoryStore
    return MemoryStore(db_path=tmp_path / "test.db", **kwargs)


def _fake_embedding(n: int = 8):
    """Return a deterministic fake embedding vector."""
    import numpy as np
    rng = np.random.default_rng(42)
    vec = rng.random(n).astype("float32")
    vec /= (vec @ vec) ** 0.5  # normalise
    return vec


# ---------------------------------------------------------------------------
# add / list / delete
# ---------------------------------------------------------------------------

class TestAddListDelete:
    def test_add_returns_id(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            mid = store.add("Hello world", source="cli")
        assert isinstance(mid, int)
        assert mid >= 1

    def test_list_returns_stored(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("fact one", source="manual")
            store.add("fact two", source="bot", tags="python,ai")
        results = store.list(n=10)
        assert len(results) == 2
        contents = {r["content"] for r in results}
        assert "fact one" in contents
        assert "fact two" in contents

    def test_list_filter_by_source(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("bot fact", source="bot")
            store.add("manual fact", source="manual")
        bot_results = store.list(source="bot")
        assert len(bot_results) == 1
        assert bot_results[0]["source"] == "bot"

    def test_delete_existing(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            mid = store.add("to delete", source="manual")
        assert store.delete(mid) is True
        assert store.list() == []

    def test_delete_nonexistent(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.delete(9999) is False

    def test_add_empty_content_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="content"):
            store.add("", source="manual")

    def test_tags_stored(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("tagged fact", source="manual", tags="foo,bar")
        results = store.list()
        assert results[0]["tags"] == "foo,bar"

    def test_get_by_id(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            mid = store.add("findable", source="cli")
        result = store.get(mid)
        assert result is not None
        assert result["content"] == "findable"
        assert result["id"] == mid

    def test_get_nonexistent(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get(9999) is None

    def test_count(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("one", source="cli")
            store.add("two", source="cli")
        assert store.count() == 2


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

class TestCategories:
    def test_add_with_category(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            mid = store.add("project info", source="cli", category="project:ryanhub")
        result = store.get(mid)
        assert result["category"] == "project:ryanhub"

    def test_list_filter_by_category(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("personal note", source="manual", category="personal")
            store.add("tech fact", source="manual", category="technical")
            store.add("no category", source="manual")

        personal = store.list(category="personal")
        assert len(personal) == 1
        assert personal[0]["category"] == "personal"

        technical = store.list(category="technical")
        assert len(technical) == 1
        assert technical[0]["category"] == "technical"

    def test_categories_list(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("a", source="cli", category="alpha")
            store.add("b", source="cli", category="beta")
            store.add("c", source="cli")  # no category
        cats = store.categories()
        assert "alpha" in cats
        assert "beta" in cats
        assert "" not in cats

    def test_count_by_category(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("a", source="cli", category="alpha")
            store.add("b", source="cli", category="alpha")
            store.add("c", source="cli", category="beta")
        assert store.count(category="alpha") == 2
        assert store.count(category="beta") == 1

    def test_search_with_category_filter(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("dispatcher uses stream-json mode", source="manual", category="project:cortex")
            store.add("dispatcher is cool", source="manual", category="personal")

        # FTS search with category filter
        with patch.object(store, "_embed", return_value=None):
            results = store.search("dispatcher", category="project:cortex")
        assert len(results) >= 1
        assert all(r["category"] == "project:cortex" for r in results)


# ---------------------------------------------------------------------------
# TTL (time-to-live)
# ---------------------------------------------------------------------------

class TestTTL:
    def test_add_with_ttl(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            mid = store.add("temp note", source="cli", ttl=3600)
        result = store.get(mid)
        assert result is not None
        assert result["expires_at"] is not None
        assert result["expires_at"] > time.time()

    def test_permanent_memory_has_no_expiry(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            mid = store.add("permanent", source="cli")
        result = store.get(mid)
        assert result["expires_at"] is None

    def test_expired_memories_excluded_from_list(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("permanent", source="cli")
            # Add a memory that's already expired
            now = time.time()
            store._conn.execute(
                "INSERT INTO memories (content, source, tags, category, created_at, expires_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("expired note", "cli", "", "", now - 100, now - 50, None),
            )
            store._conn.commit()

        results = store.list()
        assert len(results) == 1
        assert results[0]["content"] == "permanent"

    def test_expired_memories_excluded_from_search(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("findable keyword", source="cli")
            # Add an expired memory with the same keyword
            now = time.time()
            store._conn.execute(
                "INSERT INTO memories (content, source, tags, category, created_at, expires_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("expired keyword", "cli", "", "", now - 100, now - 50, None),
            )
            store._conn.commit()

        with patch.object(store, "_embed", return_value=None):
            results = store.search("keyword")
        # Should only find the non-expired one
        contents = [r["content"] for r in results]
        assert "findable keyword" in contents
        assert "expired keyword" not in contents

    def test_cleanup_expired(self, tmp_path):
        store = _make_store(tmp_path)
        now = time.time()
        # Insert an expired memory directly
        store._conn.execute(
            "INSERT INTO memories (content, source, tags, category, created_at, expires_at, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("to clean up", "cli", "", "", now - 100, now - 50, None),
        )
        store._conn.commit()

        store._cleanup_expired()

        row = store._conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        assert row["cnt"] == 0


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export_creates_file(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("fact one", source="cli")
            store.add("fact two", source="bot")

        out_path = store.export_memories(path=tmp_path / "export.json")
        assert Path(out_path).exists()

        data = json.loads(Path(out_path).read_text())
        assert data["count"] == 2
        assert len(data["memories"]) == 2

    def test_export_with_category_filter(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("alpha fact", source="cli", category="alpha")
            store.add("beta fact", source="cli", category="beta")

        out_path = store.export_memories(
            path=tmp_path / "alpha.json",
            category="alpha",
        )
        data = json.loads(Path(out_path).read_text())
        assert data["count"] == 1
        assert data["memories"][0]["category"] == "alpha"

    def test_import_memories(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("existing", source="cli")

        # Create import file
        import_data = {
            "version": 1,
            "memories": [
                {"content": "imported one", "source": "import", "tags": "", "category": "imported"},
                {"content": "imported two", "source": "import", "tags": "tag1", "category": ""},
            ],
        }
        import_file = tmp_path / "import.json"
        import_file.write_text(json.dumps(import_data))

        with patch.object(store, "_embed", return_value=None):
            count = store.import_memories(import_file)
        assert count == 2
        assert store.count() == 3

    def test_import_merge_skips_duplicates(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("existing fact", source="cli")

        import_data = {
            "version": 1,
            "memories": [
                {"content": "existing fact", "source": "cli"},  # duplicate
                {"content": "new fact", "source": "cli"},
            ],
        }
        import_file = tmp_path / "import.json"
        import_file.write_text(json.dumps(import_data))

        with patch.object(store, "_embed", return_value=None):
            count = store.import_memories(import_file, merge=True)
        assert count == 1  # Only the new one
        assert store.count() == 2

    def test_import_no_merge_allows_duplicates(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("existing fact", source="cli")

        import_data = {
            "version": 1,
            "memories": [
                {"content": "existing fact", "source": "cli"},  # duplicate
            ],
        }
        import_file = tmp_path / "import.json"
        import_file.write_text(json.dumps(import_data))

        with patch.object(store, "_embed", return_value=None):
            count = store.import_memories(import_file, merge=False)
        assert count == 1
        assert store.count() == 2  # Both copies

    def test_import_file_not_found(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.import_memories(tmp_path / "nonexistent.json")

    def test_import_invalid_json(self, tmp_path):
        store = _make_store(tmp_path)
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json")
        with pytest.raises(ValueError, match="Invalid import file"):
            store.import_memories(bad_file)

    def test_roundtrip_export_import(self, tmp_path):
        """Export then import should preserve all memories."""
        store1 = _make_store(tmp_path / "db1")
        with patch.object(store1, "_embed", return_value=None):
            store1.add("fact one", source="cli", category="alpha", tags="tag1")
            store1.add("fact two", source="bot", category="beta")

        export_path = tmp_path / "roundtrip.json"
        store1.export_memories(path=export_path)

        store2 = _make_store(tmp_path / "db2")
        with patch.object(store2, "_embed", return_value=None):
            count = store2.import_memories(export_path)
        assert count == 2
        results = store2.list()
        contents = {r["content"] for r in results}
        assert contents == {"fact one", "fact two"}


# ---------------------------------------------------------------------------
# Semantic search (mock Ollama)
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def test_search_returns_top_results(self, tmp_path):
        import numpy as np

        store = _make_store(tmp_path)
        rng = np.random.default_rng(0)

        # Store 5 memories, each with a distinct embedding
        embeddings = []
        for i in range(5):
            emb = rng.random(8).astype("float32")
            emb /= (emb @ emb) ** 0.5
            embeddings.append(emb)
            with patch.object(store, "_embed", return_value=emb):
                store.add(f"memory {i}", source="test")

        # Query embedding is identical to embeddings[2] -> should score 1.0
        query_emb = embeddings[2].copy()
        with patch.object(store, "_embed", return_value=query_emb):
            results = store.search("memory 2", n=3, threshold=0.0)

        assert len(results) >= 1
        top = results[0]
        assert top["content"] == "memory 2"
        assert pytest.approx(top["score"], abs=1e-4) == 1.0

    def test_threshold_filters_low_scores(self, tmp_path):
        import numpy as np

        store = _make_store(tmp_path)

        # Store a memory with a known embedding
        stored_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
        with patch.object(store, "_embed", return_value=stored_emb):
            store.add("orthogonal memory", source="test")

        # Query orthogonal to stored -> score ~0.0
        query_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
        with patch.object(store, "_embed", return_value=query_emb):
            results = store.search("something unrelated", n=5, threshold=0.5)

        assert results == []

    def test_semantic_search_with_category(self, tmp_path):
        import numpy as np

        store = _make_store(tmp_path)
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")

        with patch.object(store, "_embed", return_value=emb):
            store.add("in alpha category", source="test", category="alpha")
            store.add("in beta category", source="test", category="beta")

        with patch.object(store, "_embed", return_value=emb):
            results = store.search("category", n=5, threshold=0.0, category="alpha")

        assert len(results) == 1
        assert results[0]["category"] == "alpha"


# ---------------------------------------------------------------------------
# FTS5 fallback when Ollama unavailable
# ---------------------------------------------------------------------------

class TestFTS5Fallback:
    def test_fts_returns_keyword_match(self, tmp_path):
        store = _make_store(tmp_path)
        # Store memories with no embeddings (Ollama unavailable)
        with patch.object(store, "_embed", return_value=None):
            store.add("dispatcher uses stream-json mode", source="manual")
            store.add("vibe-replay captures session events", source="manual")

        # Search with Ollama still unavailable
        with patch.object(store, "_embed", return_value=None):
            results = store.search("dispatcher")

        assert len(results) >= 1
        assert "dispatcher" in results[0]["content"]

    def test_fts_no_embedding_score_is_none(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            store.add("test keyword search", source="manual")

        with patch.object(store, "_embed", return_value=None):
            results = store.search("keyword")

        assert results
        assert results[0]["score"] is None

    def test_search_no_results_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            results = store.search("zzznomatchwhatsoever")
        assert results == []


# ---------------------------------------------------------------------------
# Embedding error handling
# ---------------------------------------------------------------------------

class TestEmbeddingErrorHandling:
    def test_embed_returns_none_on_connection_error(self, tmp_path):
        """When Ollama is unreachable, _embed returns None."""
        store = _make_store(tmp_path, ollama_url="http://localhost:99999", embed_retries=0)
        result = store._embed("test text")
        assert result is None

    def test_embed_caches_unavailability(self, tmp_path):
        """After connection failure, subsequent calls skip the request."""
        store = _make_store(tmp_path, ollama_url="http://localhost:99999", embed_retries=0)
        store._embed("first call")
        assert store._embedding_available is False

        # Second call should short-circuit
        result = store._embed("second call")
        assert result is None

    def test_reset_embedding_cache(self, tmp_path):
        store = _make_store(tmp_path)
        store._embedding_available = False
        store.reset_embedding_cache()
        assert store._embedding_available is None


# ---------------------------------------------------------------------------
# Embedding serialization roundtrip
# ---------------------------------------------------------------------------

class TestEmbeddingSerialization:
    def test_pack_unpack_roundtrip(self):
        import numpy as np
        from memory.store import _pack_embedding, _unpack_embedding

        original = np.random.default_rng(7).random(128).astype("float32")
        blob = _pack_embedding(original)
        recovered = _unpack_embedding(blob)
        recovered_arr = np.asarray(recovered, dtype="float32")
        np.testing.assert_allclose(original, recovered_arr, rtol=1e-5)

    def test_embedding_stored_in_db(self, tmp_path):
        import numpy as np
        from memory.store import _unpack_embedding

        store = _make_store(tmp_path)
        emb = np.array([0.1, 0.2, 0.3], dtype="float32")
        with patch.object(store, "_embed", return_value=emb):
            mid = store.add("embedded fact", source="test")

        row = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        assert row is not None
        assert row["embedding"] is not None
        recovered = np.asarray(_unpack_embedding(row["embedding"]), dtype="float32")
        np.testing.assert_allclose(emb, recovered, rtol=1e-5)


# ---------------------------------------------------------------------------
# Database robustness
# ---------------------------------------------------------------------------

class TestDatabaseRobustness:
    def test_wal_mode_enabled(self, tmp_path):
        """Database should use WAL mode for concurrent access safety."""
        store = _make_store(tmp_path)
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_close_and_reopen(self, tmp_path):
        """Store should survive close and reopen."""
        from memory.store import MemoryStore

        store = _make_store(tmp_path)
        with patch.object(store, "_embed", return_value=None):
            mid = store.add("persistent", source="cli")
        store.close()

        store2 = MemoryStore(db_path=tmp_path / "test.db")
        result = store2.get(mid)
        assert result is not None
        assert result["content"] == "persistent"
        store2.close()

    def test_migration_adds_columns(self, tmp_path):
        """Migration should add category and expires_at columns to old databases."""
        import sqlite3

        # Create an old-style database without category/expires_at
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                tags TEXT DEFAULT '',
                created_at REAL NOT NULL,
                embedding BLOB
            )
        """)
        conn.execute(
            "INSERT INTO memories (content, source, tags, created_at) VALUES (?, ?, ?, ?)",
            ("old fact", "manual", "", time.time()),
        )
        conn.commit()
        conn.close()

        # Open with MemoryStore — should migrate
        from memory.store import MemoryStore
        store = MemoryStore(db_path=db_path)
        results = store.list()
        assert len(results) == 1
        assert results[0]["content"] == "old fact"
        # New columns should be accessible
        assert results[0]["category"] == "" or results[0]["category"] is None
        store.close()
