"""Cortex Memory — pluggable long-term memory for agent sessions.

Provides semantic search (via Ollama embeddings) and LLM-based fact extraction
(via Anthropic Haiku). Gracefully degrades to FTS5 keyword search when Ollama
is unavailable.

Features:
- Categories/namespaces for organizing memories
- TTL (time-to-live) for temporary memories
- Export/import for portability
- Robust error handling with retry and fallback
"""

from .store import MemoryStore, EmbeddingError
from .extractor import FactExtractor

__all__ = ["MemoryStore", "EmbeddingError", "FactExtractor"]
