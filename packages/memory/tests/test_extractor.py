"""Tests for FactExtractor — uses claude CLI subprocess."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


class TestFactExtractor:
    def test_extract_returns_list_of_facts(self):
        """Happy path: claude CLI returns newline-separated facts."""
        from memory.extractor import FactExtractor

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Fact one.\nFact two.\nFact three."})

        with patch("subprocess.run", return_value=mock_result):
            extractor = FactExtractor()
            facts = extractor.extract(
                "User: Can you fix the bug?\nAssistant: Fixed in store.py.",
                source="bot",
            )

        assert facts == ["Fact one.", "Fact two.", "Fact three."]

    def test_extract_caps_at_five_facts(self):
        """Extractor should return at most 5 facts."""
        from memory.extractor import FactExtractor

        six_lines = "\n".join(f"Fact {i}." for i in range(6))
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": six_lines})

        with patch("subprocess.run", return_value=mock_result):
            extractor = FactExtractor()
            facts = extractor.extract("some long conversation", source="cli")

        assert len(facts) <= 5

    def test_extract_strips_hash_comment_lines(self):
        """Lines starting with # should be filtered out."""
        from memory.extractor import FactExtractor

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "# Ignored header\nReal fact."})

        with patch("subprocess.run", return_value=mock_result):
            extractor = FactExtractor()
            facts = extractor.extract("conversation text", source="manual")

        assert facts == ["Real fact."]

    def test_extract_no_claude_cli_returns_fallback(self):
        """Without claude CLI, should return a fallback summary."""
        from memory.extractor import FactExtractor

        with patch("shutil.which", return_value=None):
            extractor = FactExtractor()
            facts = extractor.extract("This is a test conversation.", source="cli")

        assert isinstance(facts, list)
        assert len(facts) >= 1

    def test_extract_cli_error_returns_fallback(self):
        """When claude CLI fails, fall back gracefully."""
        from memory.extractor import FactExtractor

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some error"

        with patch("subprocess.run", return_value=mock_result):
            extractor = FactExtractor()
            facts = extractor.extract("important conversation content", source="bot")

        assert isinstance(facts, list)
        assert len(facts) >= 1
        assert "important conversation content" in facts[0]

    def test_extract_empty_conversation_returns_empty(self):
        """Empty or whitespace-only conversation returns empty list."""
        from memory.extractor import FactExtractor

        extractor = FactExtractor()
        assert extractor.extract("", source="cli") == []
        assert extractor.extract("   ", source="cli") == []

    def test_fallback_truncates_long_text(self):
        """_fallback_summary should truncate at 300 chars."""
        from memory.extractor import FactExtractor

        long_text = "x" * 500
        result = FactExtractor._fallback_summary(long_text)
        assert len(result) == 1
        assert len(result[0]) <= 300
        assert result[0].endswith("...")

    def test_extract_timeout_returns_fallback(self):
        """Timeout during extraction should return fallback."""
        import subprocess
        from memory.extractor import FactExtractor

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 30)):
            extractor = FactExtractor()
            facts = extractor.extract("important content", source="cli")

        assert isinstance(facts, list)
        assert len(facts) >= 1

    def test_extract_empty_result_returns_fallback(self):
        """Empty result field in CLI output should return fallback."""
        from memory.extractor import FactExtractor

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": ""})

        with patch("subprocess.run", return_value=mock_result):
            extractor = FactExtractor()
            facts = extractor.extract("A test conversation.", source="test")

        assert isinstance(facts, list)
        assert len(facts) >= 1
