"""Tests for MemoryStore — add, search, list, delete, FTS5 fallback."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path):
    """Create a MemoryStore backed by a temp file."""
    from memory.store import MemoryStore
    return MemoryStore(db_path=tmp_path / "test.db")


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
