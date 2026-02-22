"""Tests for the session analyzer."""

from datetime import datetime, timedelta

import pytest

from vibe_replay.analyzer import (
    _detect_phase_from_event,
    _identify_phase_runs,
    _extract_insights,
    _identify_decision_points,
    _identify_turning_points,
    _compute_statistics,
    _format_duration,
    analyze_session,
)
from vibe_replay.models import (
    Event,
    EventType,
    SessionMetadata,
    SessionPhase,
)
from vibe_replay.store import SessionStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a temporary session store."""
    return SessionStore(base_dir=tmp_path / "vibe-replay")


def _make_event(
    tool_name: str,
    summary: str = "",
    event_type: EventType = EventType.TOOL_CALL,
    offset_minutes: int = 0,
    files: list[str] | None = None,
) -> Event:
    """Helper to create test events."""
    return Event(
        timestamp=datetime(2026, 2, 22, 10, 0, 0) + timedelta(minutes=offset_minutes),
        session_id="test",
        event_type=event_type,
        tool_name=tool_name,
        summary=summary or f"Called {tool_name}",
        files_affected=files or [],
    )


class TestPhaseDetection:
    """Tests for phase detection from events."""

    def test_read_is_exploration(self):
        event = _make_event("Read", "Read main.py")
        assert _detect_phase_from_event(event) == SessionPhase.EXPLORATION

    def test_edit_is_implementation(self):
        event = _make_event("Edit", "Edited main.py")
        assert _detect_phase_from_event(event) == SessionPhase.IMPLEMENTATION

    def test_error_keywords_are_debugging(self):
        event = _make_event("Bash", "Ran: pytest — error in test_main")
        assert _detect_phase_from_event(event) == SessionPhase.DEBUGGING

    def test_test_keywords_are_testing(self):
        event = _make_event("Bash", "Ran: pytest tests/")
        assert _detect_phase_from_event(event) == SessionPhase.TESTING

    def test_config_keywords(self):
        event = _make_event("Write", "Created pyproject.toml config")
        assert _detect_phase_from_event(event) == SessionPhase.CONFIGURATION


class TestPhaseRuns:
    """Tests for grouping events into phase runs."""

    def test_single_phase(self):
        events = [
            _make_event("Read", offset_minutes=0),
            _make_event("Read", offset_minutes=1),
            _make_event("Glob", offset_minutes=2),
        ]
        phases = _identify_phase_runs(events)
        assert len(phases) == 1
        assert phases[0].phase == SessionPhase.EXPLORATION

    def test_two_phases(self):
        events = [
            _make_event("Read", offset_minutes=0),
            _make_event("Read", offset_minutes=1),
            _make_event("Read", offset_minutes=2),
            _make_event("Edit", offset_minutes=3),
            _make_event("Edit", offset_minutes=4),
            _make_event("Edit", offset_minutes=5),
        ]
        phases = _identify_phase_runs(events)
        # Should have at least 2 distinct phases
        phase_types = [p.phase for p in phases]
        assert SessionPhase.EXPLORATION in phase_types
        assert SessionPhase.IMPLEMENTATION in phase_types

    def test_empty_events(self):
        assert _identify_phase_runs([]) == []


class TestInsightExtraction:
    """Tests for insight extraction."""

    def test_hotspot_detection(self):
        events = [
            _make_event("Edit", offset_minutes=i, files=["/project/hot.py"])
            for i in range(5)
        ]
        insights = _extract_insights(events)
        hotspot = [i for i in insights if "hotspot" in i.title.lower()]
        assert len(hotspot) >= 1

    def test_read_heavy_pattern(self):
        events = [
            _make_event("Read", offset_minutes=i)
            for i in range(8)
        ] + [
            _make_event("Edit", offset_minutes=9)
        ]
        insights = _extract_insights(events)
        read_heavy = [i for i in insights if "read-heavy" in i.title.lower()]
        assert len(read_heavy) >= 1

    def test_implementation_heavy_pattern(self):
        events = [
            _make_event("Edit", offset_minutes=i, files=[f"/f{i}.py"])
            for i in range(8)
        ] + [
            _make_event("Read", offset_minutes=9)
        ]
        insights = _extract_insights(events)
        impl_heavy = [i for i in insights if "implementation" in i.title.lower()]
        assert len(impl_heavy) >= 1


class TestDecisionAndTurningPoints:
    """Tests for decision and turning point identification."""

    def test_decision_at_phase_transition(self):
        events = [
            _make_event("Read", offset_minutes=0),
            _make_event("Edit", offset_minutes=1, files=["/new.py"]),
        ]
        decisions = _identify_decision_points(events)
        assert 1 in decisions

    def test_turning_point_on_error(self):
        events = [
            _make_event("Edit", offset_minutes=0),
            _make_event("Bash", "error in test", EventType.ERROR, offset_minutes=1),
        ]
        turning = _identify_turning_points(events)
        assert 1 in turning


class TestAnalyzeSession:
    """Integration tests for full session analysis."""

    def test_analyze_session(self, tmp_store):
        session_id = "integration-test"
        events = [
            _make_event("Read", "Read main.py", offset_minutes=0, files=["/main.py"]),
            _make_event("Grep", "Searched for imports", offset_minutes=1),
            _make_event("Edit", "Edited main.py", EventType.CODE_CHANGE, offset_minutes=2, files=["/main.py"]),
            _make_event("Bash", "Ran: pytest", offset_minutes=3),
        ]

        for e in events:
            e_copy = e.model_copy(update={"session_id": session_id})
            tmp_store.append_event(e_copy)

        replay = analyze_session(tmp_store, session_id)

        assert replay.metadata.session_id == session_id
        assert replay.metadata.event_count == 4
        assert len(replay.timeline) > 0
        assert replay.statistics["total_events"] == 4


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_seconds(self):
        assert _format_duration(30) == "30s"

    def test_minutes(self):
        assert _format_duration(90) == "1m 30s"

    def test_hours(self):
        assert _format_duration(3700) == "1h 1m"
