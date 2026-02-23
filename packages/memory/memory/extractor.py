"""FactExtractor — uses Anthropic Haiku to extract key facts from conversations.

Gracefully degrades to returning a truncated raw summary when the API is
unavailable or the key is not configured.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("cortex.memory")

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

_EXTRACT_PROMPT = """Extract 2-5 concise factual sentences from this conversation that would be useful to remember in future sessions. Focus on: decisions made, problems solved, technical details, user preferences, project state. Skip pleasantries and filler. Output one fact per line, no bullets.

Conversation:
{conversation}"""


class FactExtractor:
    """Extract key facts from a conversation using Anthropic Haiku."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def extract(self, conversation: str, source: str = "unknown") -> list[str]:
        """Extract 2-5 key facts from a conversation string.

        Returns a list of fact strings. Returns an empty list if the API is
        unavailable. Falls back to a truncated raw summary on extraction error.
        """
        if not conversation or not conversation.strip():
            return []

        if not self._api_key:
            log.debug("ANTHROPIC_API_KEY not set; skipping extraction")
            return self._fallback_summary(conversation)

        try:
            import anthropic
        except ImportError:
            log.debug("anthropic package not installed; skipping extraction")
            return self._fallback_summary(conversation)

        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            prompt = _EXTRACT_PROMPT.format(conversation=conversation[:8000])
            msg = client.messages.create(
                model=_HAIKU_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            facts = [
                line.strip()
                for line in raw.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            log.debug("extracted %d facts from %s conversation", len(facts), source)
            return facts[:5]  # cap at 5 facts
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
