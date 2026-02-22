"""Renderer for Vibe Replay.

Generates shareable outputs from analyzed sessions:
- HTML replay (interactive, self-contained single file)
- Markdown summary
- JSON export
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import Event, SessionReplay


TEMPLATES_DIR = Path(__file__).parent / "templates"


def _get_jinja_env() -> Environment:
    """Create a Jinja2 environment with the templates directory."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def _prepare_events_for_template(events: list[Event]) -> list[dict[str, Any]]:
    """Convert events to template-friendly dictionaries."""
    result = []
    for i, event in enumerate(events):
        d = {
            "index": i,
            "timestamp": event.timestamp.strftime("%H:%M:%S"),
            "timestamp_full": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "tool_name": event.tool_name or "",
            "summary": event.summary,
            "files_affected": event.files_affected,
            "has_diff": bool(event.code_diff),
            "code_diff": event.code_diff or "",
            "details_json": json.dumps(event.details, indent=2, default=str)[:5000],
        }
        result.append(d)
    return result


def _prepare_replay_for_template(
    replay: SessionReplay,
    events: list[Event],
    share_url: str | None = None,
) -> dict[str, Any]:
    """Prepare all replay data for the HTML template."""
    event_dicts = _prepare_events_for_template(events)

    # Build phase data with their events
    phases = []
    for phase in replay.timeline:
        phase_events = event_dicts[phase.start_index : phase.end_index + 1]
        phases.append(
            {
                "phase": phase.phase.value,
                "phase_display": phase.phase.value.replace("_", " ").title(),
                "start_time": phase.start_time.strftime("%H:%M:%S"),
                "end_time": phase.end_time.strftime("%H:%M:%S"),
                "event_count": phase.event_count,
                "summary": phase.summary,
                "key_events": phase.key_events,
                "events": phase_events,
            }
        )

    # Insights grouped by type
    insights_by_type: dict[str, list[dict[str, Any]]] = {}
    for insight in replay.insights:
        type_key = insight.insight_type.value
        if type_key not in insights_by_type:
            insights_by_type[type_key] = []
        insights_by_type[type_key].append(
            {
                "type": type_key,
                "title": insight.title,
                "description": insight.description,
                "confidence": insight.confidence,
                "supporting_events": insight.supporting_events,
            }
        )

    # Decision and turning point markers
    decision_set = set(replay.key_decision_indices)
    turning_set = set(replay.turning_point_indices)

    result = {
        "metadata": {
            "session_id": replay.metadata.session_id,
            "project": replay.metadata.project or "Unknown Project",
            "start_time": replay.metadata.start_time.strftime("%Y-%m-%d %H:%M"),
            "start_date": replay.metadata.start_time.strftime("%Y-%m-%d"),
            "duration": replay.statistics.get("duration_human", "Unknown"),
            "event_count": replay.metadata.event_count,
            "file_count": replay.statistics.get("file_count", 0),
            "summary": replay.metadata.summary,
        },
        "phases": phases,
        "events": event_dicts,
        "insights": insights_by_type,
        "statistics": replay.statistics,
        "decision_indices": list(decision_set),
        "turning_indices": list(turning_set),
        "share_url": share_url,
    }
    return result


def render_html(
    replay: SessionReplay,
    events: list[Event],
    share_url: str | None = None,
) -> str:
    """Render an interactive HTML replay page.

    Args:
        replay: Analyzed session replay data.
        events: Raw events for the session.
        share_url: Optional public URL for this replay.

    Returns:
        Self-contained HTML string.
    """
    env = _get_jinja_env()
    template = env.get_template("replay.html")
    data = _prepare_replay_for_template(replay, events, share_url=share_url)
    return template.render(**data)


def render_markdown(replay: SessionReplay, events: list[Event]) -> str:
    """Render a Markdown summary of the session.

    Args:
        replay: Analyzed session replay data.
        events: Raw events for the session.

    Returns:
        Markdown-formatted string.
    """
    env = _get_jinja_env()
    try:
        template = env.get_template("summary.md.jinja")
        data = _prepare_replay_for_template(replay, events)
        return template.render(**data)
    except Exception:
        # Fallback to programmatic markdown if template fails
        return _render_markdown_fallback(replay, events)


def _render_markdown_fallback(replay: SessionReplay, events: list[Event]) -> str:
    """Generate Markdown without a template."""
    lines: list[str] = []
    meta = replay.metadata

    lines.append(f"# Session Replay: {meta.project or meta.session_id}")
    lines.append("")
    lines.append(f"**Date:** {meta.start_time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(
        f"**Duration:** {replay.statistics.get('duration_human', 'Unknown')}"
    )
    lines.append(f"**Events:** {meta.event_count}")
    lines.append(f"**Files modified:** {len(meta.files_modified)}")
    lines.append("")

    if meta.summary:
        lines.append(f"> {meta.summary}")
        lines.append("")

    # Timeline
    lines.append("## Timeline")
    lines.append("")
    for phase in replay.timeline:
        emoji = {
            "exploration": "🔍",
            "implementation": "🔨",
            "debugging": "🐛",
            "testing": "🧪",
            "refactoring": "♻️",
            "configuration": "⚙️",
            "documentation": "📝",
        }.get(phase.phase.value, "📌")
        lines.append(
            f"### {emoji} {phase.phase.value.title()} "
            f"({phase.start_time.strftime('%H:%M')} - "
            f"{phase.end_time.strftime('%H:%M')})"
        )
        lines.append(f"_{phase.summary}_")
        lines.append("")

        # Show key events in this phase
        phase_events = events[phase.start_index : phase.end_index + 1]
        for e in phase_events[:10]:
            marker = ""
            if events.index(e) in replay.key_decision_indices:
                marker = " **[KEY DECISION]**"
            elif events.index(e) in replay.turning_point_indices:
                marker = " **[TURNING POINT]**"
            lines.append(f"- {e.summary}{marker}")
        if len(phase_events) > 10:
            lines.append(f"- _...and {len(phase_events) - 10} more events_")
        lines.append("")

    # Insights
    if replay.insights:
        lines.append("## Insights")
        lines.append("")

        for insight in replay.insights:
            icon = {
                "decision": "🎯",
                "learning": "💡",
                "mistake": "⚠️",
                "pattern": "🔄",
                "turning_point": "🔀",
            }.get(insight.insight_type.value, "•")
            lines.append(f"### {icon} {insight.title}")
            lines.append(f"{insight.description}")
            lines.append("")

    # Statistics
    lines.append("## Statistics")
    lines.append("")
    stats = replay.statistics
    if stats.get("tools_used"):
        lines.append("| Tool | Count |")
        lines.append("|------|-------|")
        for tool, count in sorted(
            stats["tools_used"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"| {tool} | {count} |")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by [Vibe Replay](https://github.com/zzhiyuann/vibe-replay)*")

    return "\n".join(lines)


def render_json(replay: SessionReplay, events: list[Event]) -> str:
    """Render the full session data as JSON.

    Args:
        replay: Analyzed session replay data.
        events: Raw events for the session.

    Returns:
        JSON-formatted string.
    """
    data = {
        "replay": json.loads(replay.model_dump_json()),
        "events": [json.loads(e.model_dump_json()) for e in events],
    }
    return json.dumps(data, indent=2, default=str)


def save_html(replay: SessionReplay, events: list[Event], output_path: Path) -> Path:
    """Render and save HTML replay to a file.

    Args:
        replay: Analyzed session replay data.
        events: Raw events for the session.
        output_path: Where to save the HTML file.

    Returns:
        Path to the saved file.
    """
    html = render_html(replay, events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def save_markdown(
    replay: SessionReplay, events: list[Event], output_path: Path
) -> Path:
    """Render and save Markdown summary to a file.

    Args:
        replay: Analyzed session replay data.
        events: Raw events for the session.
        output_path: Where to save the Markdown file.

    Returns:
        Path to the saved file.
    """
    md = render_markdown(replay, events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    return output_path
