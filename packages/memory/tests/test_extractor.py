"""Tests for FactExtractor — mock Anthropic API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestFactExtractor:
    def test_extract_returns_list_of_facts(self):
        """Happy path: Haiku returns bullet-free lines."""
        from memory.extractor import FactExtractor

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Fact one.\nFact two.\nFact three.")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            extractor = FactExtractor(api_key="sk-test")
            facts = extractor.extract(
                "User: Can you fix the bug?\nAssistant: Fixed in store.py.",
                source="bot",
            )

        assert facts == ["Fact one.", "Fact two.", "Fact three."]
        mock_client.messages.create.assert_called_once()

    def test_extract_caps_at_five_facts(self):
        """Extractor should return at most 5 facts."""
        from memory.extractor import FactExtractor

        six_lines = "\n".join(f"Fact {i}." for i in range(6))
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=six_lines)]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            extractor = FactExtractor(api_key="sk-test")
            facts = extractor.extract("some long conversation", source="cli")

        assert len(facts) <= 5

    def test_extract_strips_hash_comment_lines(self):
        """Lines starting with # should be filtered out."""
        from memory.extractor import FactExtractor

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="# Ignored header\nReal fact.\n")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            extractor = FactExtractor(api_key="sk-test")
            facts = extractor.extract("conversation text", source="manual")

        assert facts == ["Real fact."]

    def test_extract_no_api_key_returns_fallback(self):
        """Without an API key, should return a fallback summary."""
        import os
        from memory.extractor import FactExtractor

        env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            extractor = FactExtractor(api_key=None)
            facts = extractor.extract("This is a test conversation.", source="cli")

        # Should return a non-empty fallback
        assert isinstance(facts, list)
        assert len(facts) >= 1

    def test_extract_anthropic_error_returns_fallback(self):
        """When Anthropic raises an exception, fall back gracefully."""
        from memory.extractor import FactExtractor

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")

        with patch("anthropic.Anthropic", return_value=mock_client):
            extractor = FactExtractor(api_key="sk-test")
            facts = extractor.extract("important conversation content", source="bot")

        # Fallback: truncated raw conversation
        assert isinstance(facts, list)
        assert len(facts) >= 1
        assert "important conversation content" in facts[0]

    def test_extract_empty_conversation_returns_empty(self):
        """Empty or whitespace-only conversation returns empty list."""
        from memory.extractor import FactExtractor

        extractor = FactExtractor(api_key="sk-test")
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

    def test_extract_without_anthropic_package(self):
        """If anthropic is not installed, fall back gracefully."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from memory.extractor import FactExtractor
            extractor = FactExtractor(api_key="sk-test")
            facts = extractor.extract("A test conversation.", source="test")

        assert isinstance(facts, list)
