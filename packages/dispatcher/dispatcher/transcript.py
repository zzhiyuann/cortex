"""Append-only JSONL conversation transcripts.

Stores full user messages and assistant responses per conversation.
Used to build complete conversation history for follow-up turns,
ensuring the LLM has full context across the entire conversation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class Transcript:
    """Per-conversation transcript stored as JSONL.

    Each line is a JSON object:
      {"role": "user"|"assistant", "content": "...", "ts": 1234567890.0}
    """

    def __init__(self, data_dir: Path):
        self.dir = data_dir / "transcripts"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, conv_id: str) -> Path:
        return self.dir / f"{conv_id}.jsonl"

    def append(self, conv_id: str, role: str, content: str):
        """Append a message to the conversation transcript."""
        entry = {"role": role, "content": content, "ts": time.time()}
        with open(self._path(conv_id), "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self, conv_id: str) -> list[dict]:
        """Load all messages for a conversation."""
        path = self._path(conv_id)
        if not path.exists():
            return []
        msgs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        msgs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return msgs

    def build_history(self, conv_id: str, max_chars: int = 80000) -> str:
        """Build conversation history string from transcript.

        Returns formatted multi-turn history for prompt injection.
        If total exceeds max_chars, older turns are dropped with a marker.
        """
        msgs = self.load(conv_id)
        if not msgs:
            return ""

        parts = []
        for msg in msgs:
            label = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"[{label}]: {msg['content']}")

        full = "\n\n".join(parts)
        if len(full) <= max_chars:
            return full

        # Trim from beginning, keeping the most recent messages
        kept: list[str] = []
        total = 0
        for part in reversed(parts):
            if total + len(part) + 2 > max_chars - 60:
                break
            kept.append(part)
            total += len(part) + 2

        kept.reverse()
        return "[... earlier conversation omitted ...]\n\n" + "\n\n".join(kept)
