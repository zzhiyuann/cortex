"""Cortex Memory — pluggable long-term memory for agent sessions.

Provides semantic search (via Ollama embeddings) and LLM-based fact extraction
(via Anthropic Haiku). Gracefully degrades to FTS5 keyword search when Ollama
is unavailable.
"""

from .store import MemoryStore
from .extractor import FactExtractor

__all__ = ["MemoryStore", "FactExtractor"]
