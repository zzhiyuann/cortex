"""FactExtractor — uses Claude CLI subprocess to extract key facts from conversations.

Uses `claude -p` so it works with any auth method (API key, OAuth/Claude Max).
Gracefully degrades to a truncated raw summary when claude CLI is unavailable.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess

log = logging.getLogger("cortex.memory")

_EXTRACT_PROMPT = """Extract 2-5 concise factual sentences from this conversation that would be useful to remember in future sessions. Focus on: decisions made, problems solved, technical details, user preferences, project state. Skip pleasantries and filler. Output one fact per line, no bullets.

Conversation:
{conversation}"""


class FactExtractor:
    """Extract key facts from a conversation using Claude CLI (claude -p)."""

    def __init__(self, api_key: str | None = None):
        # api_key param kept for compatibility; CLI handles auth automatically
        self._claude_path = shutil.which("claude")

    def extract(self, conversation: str, source: str = "unknown") -> list[str]:
        """Extract 2-5 key facts from a conversation string.

        Spawns `claude -p` to extract facts. Falls back to truncated raw
        summary if claude CLI is unavailable or extraction fails.
        """
        if not conversation or not conversation.strip():
            return []

        if not self._claude_path:
            log.debug("claude CLI not found; falling back to summary")
            return self._fallback_summary(conversation)

        prompt = _EXTRACT_PROMPT.format(conversation=conversation[:8000])
        try:
            result = subprocess.run(
                [
                    self._claude_path, "-p",
                    "--model", "claude-haiku-4-5-20251001",
                    "--max-turns", "1",
                    "--output-format", "json",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                log.warning("claude extraction failed: %s", result.stderr[:200])
                return self._fallback_summary(conversation)

            data = json.loads(result.stdout)
            raw = data.get("result", "").strip()
            if not raw:
                return self._fallback_summary(conversation)

            facts = [
                line.strip()
                for line in raw.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            log.debug("extracted %d facts from %s conversation", len(facts), source)
            return facts[:5]
        except subprocess.TimeoutExpired:
            log.warning("fact extraction timed out")
            return self._fallback_summary(conversation)
        except Exception as exc:
            log.warning("fact extraction failed (%s); falling back to summary", exc)
            return self._fallback_summary(conversation)

    @staticmethod
    def _fallback_summary(conversation: str) -> list[str]:
        """Return a single-item list with a truncated raw summary."""
        summary = conversation.strip()
        if len(summary) > 300:
            summary = summary[:297] + "..."
        return [summary] if summary else []
