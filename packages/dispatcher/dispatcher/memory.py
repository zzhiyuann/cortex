"""Persistent memory for user preferences and context."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MEMORY = """# Dispatcher Memory

## User Preferences
- (Add your preferences here)

## Communication Style
- Natural, conversational

## Notes
"""


class Memory:
    """Persistent user preferences & style, stored as markdown."""

    _MAX_SIZE = 10_000  # Max characters to load from memory file

    def __init__(self, path: Path):
        self.path = path
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(DEFAULT_MEMORY)
        self._load()

    def _load(self):
        raw = self.path.read_text()
        # Truncate oversized memory files
        if len(raw) > self._MAX_SIZE:
            raw = raw[:self._MAX_SIZE] + "\n[truncated]"
        self.text = raw

    def reload(self):
        self._load()
