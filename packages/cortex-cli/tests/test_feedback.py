"""Tests for cortex_cli.feedback — issue recording, resolution, and resolver agent."""

import json
import time

import pytest

import cortex_cli.feedback as feedback_mod


@pytest.fixture(autouse=True)
def _patch_feedback_paths(tmp_path, monkeypatch):
    """Redirect all feedback module paths to a temp directory."""
    fb_dir = tmp_path / "feedback"
    fb_dir.mkdir()
    monkeypatch.setattr(feedback_mod, "FEEDBACK_DIR", fb_dir)
    monkeypatch.setattr(feedback_mod, "ISSUES_FILE", fb_dir / "issues.jsonl")
    monkeypatch.setattr(feedback_mod, "RESOLVED_FILE", fb_dir / "resolved.jsonl")
    monkeypatch.setattr(feedback_mod, "HISTORY_FILE", fb_dir / "history.jsonl")
    monkeypatch.setattr(feedback_mod, "RESOLVER_PID", fb_dir / "resolver.pid")
    monkeypatch.setattr(feedback_mod, "RESOLVER_LOG", fb_dir / "resolver.log")


# -- record_issue ---------------------------------------------------------


class TestRecordIssue:
    """Tests for record_issue()."""

    def test_writes_jsonl_entry(self):
        """record_issue() appends a valid JSONL line to the issues file."""
        issue_id = feedback_mod.record_issue(
            source="dispatcher",
            category="error",
            description="Something broke",
            auto_resolve=False,
        )

        raw = feedback_mod.ISSUES_FILE.read_text().strip()
        entry = json.loads(raw)

        assert entry["id"] == issue_id
        assert entry["source"] == "dispatcher"
        assert entry["category"] == "error"
        assert entry["description"] == "Something broke"
        assert entry["status"] == "open"
        assert "timestamp" in entry

    def test_includes_context(self):
        """record_issue() stores optional context dict."""
        feedback_mod.record_issue(
            source="forge",
            category="timeout",
            description="timed out",
            context={"error": "Connection timeout", "project": "forge"},
            auto_resolve=False,
        )

        entry = json.loads(feedback_mod.ISSUES_FILE.read_text().strip())
        assert entry["context"]["error"] == "Connection timeout"
        assert entry["context"]["project"] == "forge"

    def test_default_context_is_empty_dict(self):
        """When no context is given, context defaults to {}."""
        feedback_mod.record_issue(
            source="cortex", category="crash", description="x", auto_resolve=False,
        )
        entry = json.loads(feedback_mod.ISSUES_FILE.read_text().strip())
        assert entry["context"] == {}

    def test_multiple_issues_appended(self):
        """Multiple calls append multiple JSONL lines."""
        for i in range(3):
            feedback_mod.record_issue(
                source="test", category="error", description=f"issue-{i}",
                auto_resolve=False,
            )
        lines = [
            l for l in feedback_mod.ISSUES_FILE.read_text().strip().split("\n") if l
        ]
        assert len(lines) == 3

    def test_returns_issue_id(self):
        """record_issue() returns a string ID starting with source prefix."""
        issue_id = feedback_mod.record_issue(
            source="vibe-replay", category="ux_friction",
            description="bad UX", auto_resolve=False,
        )
        assert issue_id.startswith("vibe-replay-")


# -- get_open_issues -------------------------------------------------------


class TestGetOpenIssues:
    """Tests for get_open_issues()."""

    def test_returns_empty_when_no_file(self):
        """Returns [] when issues file does not exist."""
        assert feedback_mod.get_open_issues() == []

    def test_returns_recorded_issues(self):
        """Recorded issues appear in the open issues list."""
        feedback_mod.record_issue(
            source="a", category="error", description="first", auto_resolve=False,
        )
        feedback_mod.record_issue(
            source="b", category="crash", description="second", auto_resolve=False,
        )

        issues = feedback_mod.get_open_issues()
        assert len(issues) == 2
        descriptions = {i["description"] for i in issues}
        assert descriptions == {"first", "second"}

    def test_excludes_resolved_issues(self):
        """Resolved issues are filtered out of the open list."""
        id1 = feedback_mod.record_issue(
            source="x", category="error", description="will resolve",
            auto_resolve=False,
        )
        feedback_mod.record_issue(
            source="y", category="error", description="stays open",
            auto_resolve=False,
        )

        feedback_mod.resolve_issue(id1, "fixed it")

        issues = feedback_mod.get_open_issues()
        assert len(issues) == 1
        assert issues[0]["description"] == "stays open"

    def test_limit_parameter(self):
        """The limit parameter caps the number of returned issues."""
        for i in range(5):
            feedback_mod.record_issue(
                source="s", category="error", description=f"i{i}",
                auto_resolve=False,
            )
        assert len(feedback_mod.get_open_issues(limit=2)) == 2

    def test_newest_first_ordering(self):
        """Issues are returned newest-first (descending timestamp)."""
        feedback_mod.record_issue(
            source="a", category="error", description="older", auto_resolve=False,
        )
        # Ensure distinct timestamps
        time.sleep(0.01)
        feedback_mod.record_issue(
            source="b", category="error", description="newer", auto_resolve=False,
        )

        issues = feedback_mod.get_open_issues()
        assert issues[0]["description"] == "newer"
        assert issues[1]["description"] == "older"


# -- resolve_issue ---------------------------------------------------------


class TestResolveIssue:
    """Tests for resolve_issue()."""

    def test_writes_to_resolved_file(self):
        """resolve_issue() appends a JSONL entry to the resolved file."""
        feedback_mod.resolve_issue("test-123", "patched the bug")

        raw = feedback_mod.RESOLVED_FILE.read_text().strip()
        entry = json.loads(raw)
        assert entry["id"] == "test-123"
        assert entry["resolution"] == "patched the bug"
        assert "resolved_at" in entry

    def test_writes_to_history_file(self):
        """resolve_issue() also writes to the history file."""
        feedback_mod.resolve_issue("test-456", "refactored")

        raw = feedback_mod.HISTORY_FILE.read_text().strip()
        entry = json.loads(raw)
        assert entry["id"] == "test-456"
        assert entry["resolution"] == "refactored"


# -- get_issue_summary -----------------------------------------------------


class TestGetIssueSummary:
    """Tests for get_issue_summary()."""

    def test_empty_summary(self):
        """Summary for no issues has zero counts."""
        summary = feedback_mod.get_issue_summary()
        assert summary == {
            "total_open": 0,
            "by_category": {},
            "by_source": {},
        }

    def test_counts_by_category_and_source(self):
        """Summary correctly aggregates by category and source."""
        feedback_mod.record_issue(
            source="dispatcher", category="error", description="a",
            auto_resolve=False,
        )
        feedback_mod.record_issue(
            source="dispatcher", category="timeout", description="b",
            auto_resolve=False,
        )
        feedback_mod.record_issue(
            source="forge", category="error", description="c",
            auto_resolve=False,
        )

        summary = feedback_mod.get_issue_summary()
        assert summary["total_open"] == 3
        assert summary["by_category"] == {"error": 2, "timeout": 1}
        assert summary["by_source"] == {"dispatcher": 2, "forge": 1}


# -- resolver_status -------------------------------------------------------


class TestResolverStatus:
    """Tests for resolver_status()."""

    def test_returns_correct_dict_shape(self):
        """resolver_status() returns a dict with all expected keys."""
        status = feedback_mod.resolver_status()
        assert "running" in status
        assert "pid" in status
        assert "open_issues" in status
        assert "total_resolved" in status
        assert "log_file" in status

    def test_not_running_by_default(self):
        """When no PID file exists, resolver is not running."""
        status = feedback_mod.resolver_status()
        assert status["running"] is False
        assert status["pid"] is None

    def test_open_and_resolved_counts(self):
        """Status counts match the actual open/resolved issues."""
        feedback_mod.record_issue(
            source="a", category="error", description="open one",
            auto_resolve=False,
        )
        id2 = feedback_mod.record_issue(
            source="b", category="error", description="will close",
            auto_resolve=False,
        )
        feedback_mod.resolve_issue(id2, "done")

        status = feedback_mod.resolver_status()
        assert status["open_issues"] == 1
        assert status["total_resolved"] == 1


# -- _resolver_running -----------------------------------------------------


class TestResolverRunning:
    """Tests for _resolver_running()."""

    def test_returns_false_when_no_pid_file(self):
        """No PID file means resolver is not running."""
        assert feedback_mod._resolver_running() is False

    def test_returns_false_for_stale_pid(self):
        """A PID file referencing a dead process returns False and cleans up."""
        feedback_mod.RESOLVER_PID.write_text("999999999")
        assert feedback_mod._resolver_running() is False
        # Stale PID file should be cleaned up
        assert not feedback_mod.RESOLVER_PID.exists()

    def test_returns_false_for_invalid_pid_content(self):
        """Non-integer PID file content is handled gracefully."""
        feedback_mod.RESOLVER_PID.write_text("not-a-number")
        assert feedback_mod._resolver_running() is False


# -- build_improvement_prompt ----------------------------------------------


class TestBuildImprovementPrompt:
    """Tests for build_improvement_prompt()."""

    def test_returns_empty_when_no_issues(self):
        """No issues → empty prompt string."""
        assert feedback_mod.build_improvement_prompt() == ""
        assert feedback_mod.build_improvement_prompt(issues=[]) == ""

    def test_includes_issue_details(self):
        """Prompt text contains issue source, category, and description."""
        issues = [
            {
                "id": "test-1",
                "timestamp": "2025-01-01T00:00:00",
                "source": "dispatcher",
                "category": "error",
                "description": "request failed spectacularly",
                "context": {"error": "timeout after 30s"},
                "status": "open",
            }
        ]

        prompt = feedback_mod.build_improvement_prompt(issues=issues)
        assert "dispatcher" in prompt
        assert "error" in prompt
        assert "request failed spectacularly" in prompt
        assert "timeout after 30s" in prompt

    def test_includes_user_message_and_suggestion(self):
        """Prompt includes user_message and suggestion from context."""
        issues = [
            {
                "id": "test-2",
                "timestamp": "2025-01-01T00:00:00",
                "source": "forge",
                "category": "ux_friction",
                "description": "confusing output",
                "context": {
                    "user_message": "I don't understand this",
                    "suggestion": "add clearer labels",
                },
                "status": "open",
            }
        ]

        prompt = feedback_mod.build_improvement_prompt(issues=issues)
        assert "I don't understand this" in prompt
        assert "add clearer labels" in prompt

    def test_contains_header_and_instructions(self):
        """Prompt includes header section and fix instructions."""
        issues = [
            {
                "id": "t-1",
                "timestamp": "2025-01-01T00:00:00",
                "source": "a",
                "category": "error",
                "description": "b",
                "context": {},
                "status": "open",
            }
        ]

        prompt = feedback_mod.build_improvement_prompt(issues=issues)
        assert "Open Issues" in prompt
        assert "Fix them" in prompt or "fix" in prompt.lower()

    def test_multiple_issues_numbered(self):
        """Each issue gets a numbered header (Issue 1, Issue 2, etc.)."""
        issues = [
            {
                "id": f"t-{i}",
                "timestamp": "2025-01-01T00:00:00",
                "source": "s",
                "category": "error",
                "description": f"desc-{i}",
                "context": {},
                "status": "open",
            }
            for i in range(3)
        ]

        prompt = feedback_mod.build_improvement_prompt(issues=issues)
        assert "Issue 1" in prompt
        assert "Issue 2" in prompt
        assert "Issue 3" in prompt
