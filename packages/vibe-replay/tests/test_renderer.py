"""Tests for the renderer."""

from datetime import datetime

import pytest

from vibe_replay.models import (
    Event,
    EventType,
    Insight,
    InsightType,
    SessionMetadata,
    SessionPhase,
    SessionReplay,
    TimelinePhase,
)
from vibe_replay.renderer import (
    render_html,
    render_json,
    render_markdown,
    _prepare_events_for_template,
)


@pytest.fixture
def sample_events():
    """Create sample events for rendering."""
    return [
        Event(
            timestamp=datetime(2026, 2, 22, 10, 0, 0),
            session_id="render-test",
            event_type=EventType.TOOL_CALL,
            tool_name="Read",
            summary="Read main.py",
            files_affected=["/project/main.py"],
        ),
        Event(
            timestamp=datetime(2026, 2, 22, 10, 5, 0),
            session_id="render-test",
            event_type=EventType.CODE_CHANGE,
            tool_name="Edit",
            summary="Edited main.py",
            code_diff="--- old\n+++ new\n-print('old')\n+print('new')",
            files_affected=["/project/main.py"],
        ),
        Event(
            timestamp=datetime(2026, 2, 22, 10, 10, 0),
            session_id="render-test",
            event_type=EventType.TOOL_CALL,
            tool_name="Bash",
            summary="Ran: pytest tests/",
        ),
    ]


@pytest.fixture
def sample_replay(sample_events):
    """Create a sample replay for rendering."""
    return SessionReplay(
        metadata=SessionMetadata(
            session_id="render-test",
            project="Test Project",
            start_time=datetime(2026, 2, 22, 10, 0, 0),
            end_time=datetime(2026, 2, 22, 10, 10, 0),
            event_count=3,
            summary="Test session with reading, editing, and testing",
            files_modified=["/project/main.py"],
            tools_used={"Read": 1, "Edit": 1, "Bash": 1},
        ),
        timeline=[
            TimelinePhase(
                phase=SessionPhase.EXPLORATION,
                start_index=0,
                end_index=0,
                start_time=datetime(2026, 2, 22, 10, 0, 0),
                end_time=datetime(2026, 2, 22, 10, 0, 0),
                event_count=1,
                summary="Explored 1 file(s) | 1 event(s)",
            ),
            TimelinePhase(
                phase=SessionPhase.IMPLEMENTATION,
                start_index=1,
                end_index=1,
                start_time=datetime(2026, 2, 22, 10, 5, 0),
                end_time=datetime(2026, 2, 22, 10, 5, 0),
                event_count=1,
                summary="Modified 1 file(s) | 1 event(s)",
                key_events=[1],
            ),
            TimelinePhase(
                phase=SessionPhase.TESTING,
                start_index=2,
                end_index=2,
                start_time=datetime(2026, 2, 22, 10, 10, 0),
                end_time=datetime(2026, 2, 22, 10, 10, 0),
                event_count=1,
                summary="Ran tests | 1 event(s)",
            ),
        ],
        insights=[
            Insight(
                insight_type=InsightType.PATTERN,
                title="Clean workflow",
                description="Read-Edit-Test cycle observed.",
                confidence=0.7,
            ),
        ],
        key_decision_indices=[1],
        turning_point_indices=[],
        statistics={
            "total_events": 3,
            "duration_seconds": 600,
            "duration_human": "10m 0s",
            "tools_used": {"Read": 1, "Edit": 1, "Bash": 1},
            "file_count": 1,
            "code_changes": 1,
            "errors_encountered": 0,
        },
    )


class TestPrepareEvents:
    """Tests for event preparation."""

    def test_prepare_events(self, sample_events):
        prepared = _prepare_events_for_template(sample_events)
        assert len(prepared) == 3
        assert prepared[0]["tool_name"] == "Read"
        assert prepared[1]["has_diff"] is True
        assert prepared[2]["has_diff"] is False


class TestRenderHTML:
    """Tests for HTML rendering."""

    def test_render_produces_html(self, sample_replay, sample_events):
        html = render_html(sample_replay, sample_events)
        assert "<!DOCTYPE html>" in html
        assert "Vibe Replay" in html
        assert "Test Project" in html

    def test_html_contains_events(self, sample_replay, sample_events):
        html = render_html(sample_replay, sample_events)
        assert "Read main.py" in html
        assert "Edited main.py" in html

    def test_html_contains_insights(self, sample_replay, sample_events):
        html = render_html(sample_replay, sample_events)
        assert "Clean workflow" in html

    def test_html_is_self_contained(self, sample_replay, sample_events):
        html = render_html(sample_replay, sample_events)
        # Should have inline CSS and JS
        assert "<style>" in html
        assert "<script>" in html
        # Should NOT have external stylesheet/script references
        # (links in footer to the project repo are okay)
        assert '<link rel="stylesheet" href="http' not in html
        assert '<script src="http' not in html


class TestRenderMarkdown:
    """Tests for Markdown rendering."""

    def test_render_markdown(self, sample_replay, sample_events):
        md = render_markdown(sample_replay, sample_events)
        assert "Session Replay" in md or "Test Project" in md
        assert "Vibe Replay" in md

    def test_markdown_has_timeline(self, sample_replay, sample_events):
        md = render_markdown(sample_replay, sample_events)
        assert "Timeline" in md or "Exploration" in md or "Implementation" in md


class TestRenderJSON:
    """Tests for JSON rendering."""

    def test_render_json_valid(self, sample_replay, sample_events):
        import json
        output = render_json(sample_replay, sample_events)
        data = json.loads(output)
        assert "replay" in data
        assert "events" in data
        assert len(data["events"]) == 3

    def test_json_contains_metadata(self, sample_replay, sample_events):
        import json
        output = render_json(sample_replay, sample_events)
        data = json.loads(output)
        assert data["replay"]["metadata"]["project"] == "Test Project"
