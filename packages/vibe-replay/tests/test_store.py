"""Tests for the session store."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from vibe_replay.models import Event, EventType, SessionMetadata
from vibe_replay.store import SessionStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a temporary session store."""
    return SessionStore(base_dir=tmp_path / "vibe-replay")


@pytest.fixture
def sample_event():
    """Create a sample event."""
    return Event(
        timestamp=datetime(2026, 2, 22, 10, 0, 0),
        session_id="test-session-1",
        event_type=EventType.CODE_CHANGE,
        tool_name="Edit",
        summary="Edited main.py",
        details={"input": {"file_path": "/project/main.py"}},
        files_affected=["/project/main.py"],
    )


@pytest.fixture
def sample_metadata():
    """Create sample session metadata."""
    return SessionMetadata(
        session_id="test-session-1",
        project="test-project",
        start_time=datetime(2026, 2, 22, 10, 0, 0),
        end_time=datetime(2026, 2, 22, 10, 30, 0),
        event_count=10,
        summary="Test session",
    )


class TestSessionStore:
    """Tests for SessionStore operations."""

    def test_ensure_session(self, tmp_store):
        path = tmp_store.ensure_session("s1")
        assert path.exists()
        assert path.is_dir()

    def test_append_and_get_events(self, tmp_store, sample_event):
        tmp_store.append_event(sample_event)
        events = tmp_store.get_events("test-session-1")
        assert len(events) == 1
        assert events[0].tool_name == "Edit"

    def test_append_multiple_events(self, tmp_store):
        for i in range(5):
            event = Event(
                timestamp=datetime(2026, 2, 22, 10, i, 0),
                session_id="s1",
                event_type=EventType.TOOL_CALL,
                tool_name="Read",
                summary=f"Read file {i}",
            )
            tmp_store.append_event(event)

        events = tmp_store.get_events("s1")
        assert len(events) == 5

    def test_event_count(self, tmp_store, sample_event):
        assert tmp_store.event_count("test-session-1") == 0
        tmp_store.append_event(sample_event)
        assert tmp_store.event_count("test-session-1") == 1

    def test_iter_events(self, tmp_store, sample_event):
        tmp_store.append_event(sample_event)
        events = list(tmp_store.iter_events("test-session-1"))
        assert len(events) == 1

    def test_save_and_get_metadata(self, tmp_store, sample_metadata):
        tmp_store.save_metadata(sample_metadata)
        loaded = tmp_store.get_metadata("test-session-1")
        assert loaded is not None
        assert loaded.project == "test-project"
        assert loaded.event_count == 10

    def test_get_metadata_not_found(self, tmp_store):
        assert tmp_store.get_metadata("nonexistent") is None

    def test_list_sessions(self, tmp_store, sample_metadata):
        tmp_store.save_metadata(sample_metadata)

        meta2 = SessionMetadata(
            session_id="test-session-2",
            project="other-project",
            start_time=datetime(2026, 2, 22, 11, 0, 0),
            event_count=5,
        )
        tmp_store.save_metadata(meta2)

        sessions = tmp_store.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_filter_project(self, tmp_store, sample_metadata):
        tmp_store.save_metadata(sample_metadata)

        meta2 = SessionMetadata(
            session_id="test-session-2",
            project="other-project",
            start_time=datetime(2026, 2, 22, 11, 0, 0),
        )
        tmp_store.save_metadata(meta2)

        sessions = tmp_store.list_sessions(project="test-project")
        assert len(sessions) == 1
        assert sessions[0].session_id == "test-session-1"

    def test_search_sessions(self, tmp_store, sample_metadata):
        tmp_store.save_metadata(sample_metadata)
        results = tmp_store.search_sessions("test")
        assert len(results) == 1

    def test_delete_session(self, tmp_store, sample_event, sample_metadata):
        tmp_store.append_event(sample_event)
        tmp_store.save_metadata(sample_metadata)

        assert tmp_store.session_exists("test-session-1")
        deleted = tmp_store.delete_session("test-session-1")
        assert deleted
        assert not tmp_store.session_exists("test-session-1")

    def test_delete_nonexistent(self, tmp_store):
        assert not tmp_store.delete_session("nope")

    def test_session_exists(self, tmp_store, sample_event):
        assert not tmp_store.session_exists("test-session-1")
        tmp_store.append_event(sample_event)
        assert tmp_store.session_exists("test-session-1")

    def test_get_events_empty_session(self, tmp_store):
        events = tmp_store.get_events("nonexistent")
        assert events == []
