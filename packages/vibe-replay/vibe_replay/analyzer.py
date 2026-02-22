"""Session analyzer for Vibe Replay.

Processes raw events into structured insights:
- Timeline phases (exploration, implementation, debugging, etc.)
- Key decision points and turning points
- Learnings, mistakes, and patterns
- Cross-session aggregation
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from .models import (
    Event,
    EventType,
    Insight,
    InsightType,
    SessionMetadata,
    SessionPhase,
    SessionReplay,
    TimelinePhase,
)
from .store import SessionStore


# Tool-to-phase mapping heuristics
PHASE_INDICATORS: dict[str, list[SessionPhase]] = {
    "Read": [SessionPhase.EXPLORATION],
    "Glob": [SessionPhase.EXPLORATION],
    "Grep": [SessionPhase.EXPLORATION, SessionPhase.DEBUGGING],
    "WebSearch": [SessionPhase.EXPLORATION],
    "WebFetch": [SessionPhase.EXPLORATION],
    "ToolSearch": [SessionPhase.EXPLORATION],
    "ListMcpResourcesTool": [SessionPhase.EXPLORATION],
    "ReadMcpResourceTool": [SessionPhase.EXPLORATION],
    "Edit": [SessionPhase.IMPLEMENTATION, SessionPhase.REFACTORING],
    "Write": [SessionPhase.IMPLEMENTATION, SessionPhase.CONFIGURATION],
    "Bash": [SessionPhase.TESTING, SessionPhase.DEBUGGING, SessionPhase.CONFIGURATION],
    "NotebookEdit": [SessionPhase.IMPLEMENTATION],
    "Task": [SessionPhase.IMPLEMENTATION],
    "Skill": [SessionPhase.IMPLEMENTATION],
    "AskUserQuestion": [SessionPhase.EXPLORATION],
    "EnterPlanMode": [SessionPhase.EXPLORATION],
    "ExitPlanMode": [SessionPhase.IMPLEMENTATION],
    "TaskCreate": [SessionPhase.IMPLEMENTATION],
    "TaskUpdate": [SessionPhase.IMPLEMENTATION],
    "TodoWrite": [SessionPhase.IMPLEMENTATION],
}

# Keywords that suggest debugging
DEBUG_KEYWORDS = {
    "error", "fail", "bug", "fix", "debug", "traceback", "exception",
    "broken", "crash", "issue", "wrong", "problem", "unexpected",
}

# Keywords that suggest testing
TEST_KEYWORDS = {
    "test", "pytest", "assert", "spec", "coverage", "mock", "fixture",
}

# Keywords that suggest configuration
CONFIG_KEYWORDS = {
    "config", "setup", "install", "dependency", "pyproject", "package",
    "requirements", "env", "docker", "deploy",
}


def _detect_phase_from_event(event: Event) -> SessionPhase:
    """Detect the likely phase of a single event based on heuristics."""
    summary_lower = event.summary.lower()
    tool = event.tool_name or ""

    # Check for debugging signals
    if any(kw in summary_lower for kw in DEBUG_KEYWORDS):
        return SessionPhase.DEBUGGING

    # Check for testing signals
    if any(kw in summary_lower for kw in TEST_KEYWORDS):
        return SessionPhase.TESTING

    # Check for configuration signals
    if any(kw in summary_lower for kw in CONFIG_KEYWORDS):
        return SessionPhase.CONFIGURATION

    # Check for documentation
    if any(kw in summary_lower for kw in ("readme", "doc", "comment", "docstring")):
        return SessionPhase.DOCUMENTATION

    # Use tool-based heuristic
    # Default to IMPLEMENTATION for unknown tools rather than UNKNOWN,
    # since most unrecognized tools are doing productive work
    phases = PHASE_INDICATORS.get(tool, [SessionPhase.IMPLEMENTATION])
    return phases[0]


def _identify_phase_runs(events: list[Event]) -> list[TimelinePhase]:
    """Group consecutive events into phase runs.

    Uses a sliding window approach: assigns a phase to each event,
    then merges consecutive events of the same phase.
    """
    if not events:
        return []

    # Assign phase to each event
    phases = [_detect_phase_from_event(e) for e in events]

    # Merge consecutive same-phase events (with some tolerance)
    runs: list[TimelinePhase] = []
    current_phase = phases[0]
    start_idx = 0

    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            # Check if this is just a brief interruption (1-2 events)
            # that should be merged with the surrounding phase
            look_ahead = min(i + 3, len(phases))
            if (
                look_ahead - i >= 2
                and all(p == current_phase for p in phases[i + 1 : look_ahead])
            ):
                # Skip this blip
                phases[i] = current_phase
                continue

            runs.append(
                TimelinePhase(
                    phase=current_phase,
                    start_index=start_idx,
                    end_index=i - 1,
                    start_time=events[start_idx].timestamp,
                    end_time=events[i - 1].timestamp,
                    event_count=i - start_idx,
                )
            )
            current_phase = phases[i]
            start_idx = i

    # Final run
    runs.append(
        TimelinePhase(
            phase=current_phase,
            start_index=start_idx,
            end_index=len(events) - 1,
            start_time=events[start_idx].timestamp,
            end_time=events[-1].timestamp,
            event_count=len(events) - start_idx,
        )
    )

    # Aggressive merging to reduce fragmentation
    # Step 1: Absorb tiny runs (< 3 events or < 2 min) into neighbors
    merged: list[TimelinePhase] = []
    for run in runs:
        run_duration = (run.end_time - run.start_time).total_seconds()
        if merged and (run.event_count < 3 or run_duration < 120):
            # Absorb into previous phase
            prev = merged[-1]
            merged[-1] = TimelinePhase(
                phase=prev.phase,
                start_index=prev.start_index,
                end_index=run.end_index,
                start_time=prev.start_time,
                end_time=run.end_time,
                event_count=prev.event_count + run.event_count,
                key_events=prev.key_events + run.key_events,
            )
        else:
            merged.append(run)

    # Step 2: If still too many phases, merge adjacent same-type phases
    consolidated: list[TimelinePhase] = []
    for run in merged:
        if consolidated and consolidated[-1].phase == run.phase:
            prev = consolidated[-1]
            consolidated[-1] = TimelinePhase(
                phase=prev.phase,
                start_index=prev.start_index,
                end_index=run.end_index,
                start_time=prev.start_time,
                end_time=run.end_time,
                event_count=prev.event_count + run.event_count,
                key_events=prev.key_events + run.key_events,
            )
        else:
            consolidated.append(run)

    # Step 3: Target max 4-8 phases for sessions under 60 min
    total_duration = 0
    if consolidated:
        total_duration = (consolidated[-1].end_time - consolidated[0].start_time).total_seconds()
    max_phases = 8 if total_duration > 3600 else 6
    while len(consolidated) > max_phases:
        # Find the smallest phase and merge it with its smaller neighbor
        min_idx = min(range(len(consolidated)), key=lambda i: consolidated[i].event_count)
        if min_idx == 0:
            merge_with = 1
        elif min_idx == len(consolidated) - 1:
            merge_with = min_idx - 1
        else:
            # Merge with the smaller neighbor
            left = consolidated[min_idx - 1].event_count
            right = consolidated[min_idx + 1].event_count
            merge_with = min_idx - 1 if left <= right else min_idx + 1
        a, b = sorted([min_idx, merge_with])
        pa, pb = consolidated[a], consolidated[b]
        # Keep the phase type of the larger one
        keep_phase = pa.phase if pa.event_count >= pb.event_count else pb.phase
        new = TimelinePhase(
            phase=keep_phase,
            start_index=pa.start_index,
            end_index=pb.end_index,
            start_time=pa.start_time,
            end_time=pb.end_time,
            event_count=pa.event_count + pb.event_count,
            key_events=pa.key_events + pb.key_events,
        )
        consolidated[a] = new
        del consolidated[b]

    return consolidated


def _extract_short_names(files: set[str], max_names: int = 3) -> list[str]:
    """Extract short, meaningful file names from a set of paths."""
    names = []
    for f in sorted(files):
        parts = f.rstrip("/").split("/")
        name = parts[-1] if parts else f
        # Include parent dir for context if it's a common name
        if name in ("__init__.py", "index.html", "main.py", "setup.py", "models.py"):
            if len(parts) >= 2:
                name = f"{parts[-2]}/{name}"
        names.append(name)
    # Deduplicate and limit
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
        if len(unique) >= max_names:
            break
    return unique


def _generate_phase_summaries(
    phases: list[TimelinePhase], events: list[Event]
) -> list[TimelinePhase]:
    """Generate human-readable summaries for each phase."""
    result = []
    for phase in phases:
        phase_events = events[phase.start_index : phase.end_index + 1]

        # Count tools used
        tools = Counter(e.tool_name for e in phase_events if e.tool_name)
        files = set()
        for e in phase_events:
            files.update(e.files_affected)
        short_files = _extract_short_names(files)

        # Build a descriptive summary
        summary = ""
        if phase.phase == SessionPhase.EXPLORATION:
            if short_files:
                summary = f"Read and searched through {', '.join(short_files)}"
                if len(files) > len(short_files):
                    summary += f" and {len(files) - len(short_files)} more"
            else:
                summary = f"Explored the codebase ({tools.get('Read', 0)} reads, {tools.get('Grep', 0)} searches)"
        elif phase.phase == SessionPhase.IMPLEMENTATION:
            edits = tools.get("Edit", 0)
            writes = tools.get("Write", 0)
            if short_files:
                action = "Built" if writes > edits else "Modified"
                summary = f"{action} {', '.join(short_files)}"
                if len(files) > len(short_files):
                    summary += f" (+{len(files) - len(short_files)} more)"
            else:
                summary = f"Implemented changes ({edits} edits, {writes} new files)"
        elif phase.phase == SessionPhase.DEBUGGING:
            error_count = sum(1 for e in phase_events if e.event_type == EventType.ERROR)
            if error_count:
                summary = f"Debugged {error_count} error(s)"
                if short_files:
                    summary += f" in {', '.join(short_files[:2])}"
            else:
                summary = "Investigated and fixed issues"
        elif phase.phase == SessionPhase.TESTING:
            bash_count = tools.get("Bash", 0)
            summary = f"Ran tests ({bash_count} command(s))"
            if short_files:
                summary += f" on {', '.join(short_files[:2])}"
        elif phase.phase == SessionPhase.CONFIGURATION:
            if short_files:
                summary = f"Configured {', '.join(short_files)}"
            else:
                summary = "Set up project configuration"
        elif phase.phase == SessionPhase.REFACTORING:
            if short_files:
                summary = f"Refactored {', '.join(short_files)}"
            else:
                summary = "Refactored code structure"
        elif phase.phase == SessionPhase.DOCUMENTATION:
            if short_files:
                summary = f"Updated {', '.join(short_files)}"
            else:
                summary = "Updated documentation"
        else:
            summary = f"{phase.event_count} events"

        # Identify key events (code changes, errors)
        key_indices = []
        for i, e in enumerate(phase_events):
            global_idx = phase.start_index + i
            if e.event_type in (EventType.CODE_CHANGE, EventType.ERROR):
                key_indices.append(global_idx)

        updated = phase.model_copy(
            update={
                "summary": summary,
                "key_events": key_indices[:10],
            }
        )
        result.append(updated)

    return result


def _extract_insights(events: list[Event]) -> list[Insight]:
    """Extract structured insights from the event stream."""
    insights: list[Insight] = []

    # Track patterns
    error_indices: list[int] = []
    fix_after_error: list[tuple[int, int]] = []  # (error_idx, fix_idx)
    tools_sequence: list[str] = []
    files_touched: dict[str, list[int]] = defaultdict(list)

    for i, event in enumerate(events):
        tools_sequence.append(event.tool_name or "")

        for f in event.files_affected:
            files_touched[f].append(i)

        # Track errors and subsequent fixes
        if event.event_type == EventType.ERROR:
            error_indices.append(i)
        elif error_indices and event.event_type == EventType.CODE_CHANGE:
            # A code change after an error might be a fix
            fix_after_error.append((error_indices[-1], i))

    # Insight: Detour detection (error → many exploration events → fix)
    for err_idx, fix_idx in fix_after_error:
        gap = fix_idx - err_idx
        if gap > 5:
            # Significant detour
            insights.append(
                Insight(
                    insight_type=InsightType.MISTAKE,
                    title="Debugging detour detected",
                    description=(
                        f"An error at event #{err_idx} led to {gap} events of "
                        f"investigation before a fix at event #{fix_idx}. "
                        f"Error: {events[err_idx].summary}"
                    ),
                    supporting_events=[err_idx, fix_idx],
                    confidence=0.6,
                )
            )

    # Insight: Frequently modified files (possible hotspot)
    for filepath, indices in files_touched.items():
        if len(indices) >= 4:
            insights.append(
                Insight(
                    insight_type=InsightType.PATTERN,
                    title=f"Hotspot file: {filepath.split('/')[-1]}",
                    description=(
                        f"{filepath} was modified {len(indices)} times during "
                        f"the session, suggesting it's a central piece."
                    ),
                    supporting_events=indices[:5],
                    confidence=0.7,
                )
            )

    # Insight: Exploration patterns
    exploration_runs = 0
    current_run = 0
    for tool in tools_sequence:
        if tool in ("Read", "Glob", "Grep", "WebSearch"):
            current_run += 1
        else:
            if current_run >= 5:
                exploration_runs += 1
            current_run = 0
    if current_run >= 5:
        exploration_runs += 1

    if exploration_runs >= 2:
        insights.append(
            Insight(
                insight_type=InsightType.PATTERN,
                title="Multiple exploration phases",
                description=(
                    f"The session had {exploration_runs} significant exploration "
                    f"phases (5+ consecutive read/search operations), suggesting "
                    f"an iterative discovery process."
                ),
                confidence=0.7,
            )
        )

    # Insight: Tool usage patterns
    tool_counts = Counter(t for t in tools_sequence if t)
    total = sum(tool_counts.values())
    if total > 0:
        # Heavy reading ratio
        read_tools = tool_counts.get("Read", 0) + tool_counts.get("Grep", 0) + tool_counts.get("Glob", 0)
        if read_tools / total > 0.6:
            insights.append(
                Insight(
                    insight_type=InsightType.PATTERN,
                    title="Read-heavy session",
                    description=(
                        f"{read_tools}/{total} events ({read_tools*100//total}%) "
                        f"were reading/searching operations. This session was "
                        f"primarily about understanding existing code."
                    ),
                    confidence=0.8,
                )
            )

        # Heavy writing ratio
        write_tools = tool_counts.get("Edit", 0) + tool_counts.get("Write", 0)
        if total > 5 and write_tools / total > 0.5:
            insights.append(
                Insight(
                    insight_type=InsightType.PATTERN,
                    title="Implementation-heavy session",
                    description=(
                        f"{write_tools}/{total} events ({write_tools*100//total}%) "
                        f"were code modifications. This was a focused "
                        f"implementation session."
                    ),
                    confidence=0.8,
                )
            )

    # Insight: Rapid iteration cycles (Edit → Bash → Edit within same file)
    iteration_cycles = 0
    for i in range(2, len(tools_sequence)):
        if (
            tools_sequence[i] in ("Edit", "Write")
            and tools_sequence[i - 1] == "Bash"
            and tools_sequence[i - 2] in ("Edit", "Write")
        ):
            iteration_cycles += 1
    if iteration_cycles >= 3:
        insights.append(
            Insight(
                insight_type=InsightType.PATTERN,
                title="Rapid iteration cycles",
                description=(
                    f"Detected {iteration_cycles} edit-run-edit cycles, "
                    f"indicating fast iterative development with immediate feedback."
                ),
                confidence=0.7,
            )
        )

    # Insight: Investigation-first pattern (many reads before first edit)
    first_edit_idx = None
    for i, tool in enumerate(tools_sequence):
        if tool in ("Edit", "Write"):
            first_edit_idx = i
            break
    if first_edit_idx is not None and first_edit_idx >= 8:
        insights.append(
            Insight(
                insight_type=InsightType.PATTERN,
                title="Thorough investigation before coding",
                description=(
                    f"The first code change happened at event #{first_edit_idx + 1}, "
                    f"after {first_edit_idx} events of reading and exploration. "
                    f"This suggests careful understanding before modifying code."
                ),
                confidence=0.7,
            )
        )

    # Insight: Focus score (how many distinct files vs events)
    unique_files = set()
    for e in events:
        unique_files.update(e.files_affected)
    if total > 10 and unique_files:
        focus_ratio = len(unique_files) / total
        if focus_ratio < 0.15:
            insights.append(
                Insight(
                    insight_type=InsightType.PATTERN,
                    title="Highly focused session",
                    description=(
                        f"Only {len(unique_files)} files across {total} events "
                        f"({focus_ratio:.0%} file/event ratio). Work was concentrated "
                        f"on a small, targeted set of files."
                    ),
                    confidence=0.75,
                )
            )
        elif focus_ratio > 0.6:
            insights.append(
                Insight(
                    insight_type=InsightType.PATTERN,
                    title="Broad session scope",
                    description=(
                        f"Touched {len(unique_files)} files across {total} events "
                        f"({focus_ratio:.0%} file/event ratio). Work spanned many "
                        f"different areas of the codebase."
                    ),
                    confidence=0.65,
                )
            )

    return insights


def _identify_decision_points(events: list[Event]) -> list[int]:
    """Identify events that represent key decision points.

    A decision point is when the direction of work changes significantly.
    """
    decision_indices: list[int] = []

    for i, event in enumerate(events):
        # Direction changes: switching from exploration to implementation
        if i > 0:
            prev_phase = _detect_phase_from_event(events[i - 1])
            curr_phase = _detect_phase_from_event(event)
            if prev_phase != curr_phase and curr_phase == SessionPhase.IMPLEMENTATION:
                decision_indices.append(i)

        # First code change in a new file
        if event.event_type == EventType.CODE_CHANGE and event.files_affected:
            # Check if this file was touched before
            prev_files = set()
            for prev in events[:i]:
                prev_files.update(prev.files_affected)
            new_files = [f for f in event.files_affected if f not in prev_files]
            if new_files:
                decision_indices.append(i)

    return sorted(set(decision_indices))


def _identify_turning_points(events: list[Event]) -> list[int]:
    """Identify turning points in the session.

    A turning point is when something significant changes the course:
    errors, breakthroughs, test results.
    """
    turning_points: list[int] = []

    for i, event in enumerate(events):
        # Errors are turning points
        if event.event_type == EventType.ERROR:
            turning_points.append(i)

        # Bash commands with test results
        if event.tool_name == "Bash":
            summary_lower = event.summary.lower()
            if any(kw in summary_lower for kw in ("test", "pytest", "npm test")):
                turning_points.append(i)

    return sorted(set(turning_points))


def _compute_statistics(events: list[Event], metadata: SessionMetadata) -> dict[str, Any]:
    """Compute aggregate statistics for the session."""
    tool_counts = Counter(e.tool_name for e in events if e.tool_name)
    event_type_counts = Counter(e.event_type.value for e in events)
    all_files = set()
    for e in events:
        all_files.update(e.files_affected)

    # Compute duration
    if events:
        duration = (events[-1].timestamp - events[0].timestamp).total_seconds()
    else:
        duration = 0

    return {
        "total_events": len(events),
        "duration_seconds": duration,
        "duration_human": _format_duration(duration),
        "tools_used": dict(tool_counts.most_common()),
        "event_types": dict(event_type_counts),
        "files_affected": sorted(all_files),
        "file_count": len(all_files),
        "most_used_tool": tool_counts.most_common(1)[0][0] if tool_counts else None,
        "code_changes": event_type_counts.get("code_change", 0),
        "errors_encountered": event_type_counts.get("error", 0),
    }


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def _detect_project_name(events: list[Event]) -> str:
    """Auto-detect project name from event file paths and working directories."""
    # Collect all absolute file paths from events
    all_paths: list[str] = []
    for e in events:
        for p in e.files_affected:
            if p.startswith("/"):
                all_paths.append(p)

    if not all_paths:
        return ""

    # Count project-level directories
    # Look for paths like /Users/X/projects/PROJECT/ or /home/X/PROJECT/
    skip_dirs = {
        "", "Users", "home", "root", "var", "tmp", "etc", "opt", "usr",
        "src", "lib", "bin", "projects", "repos", "code", "workspace",
    }
    project_counts: Counter = Counter()
    for p in all_paths:
        parts = p.strip("/").split("/")
        # Walk the path to find the first "project-like" directory
        # Skip: root dirs, username, and generic parent dirs
        for i, part in enumerate(parts):
            if part in skip_dirs or part.startswith("."):
                continue
            # Skip the username (usually index 1 in /Users/username)
            if i <= 1:
                continue
            # This is likely a project directory
            project_counts[part] += 1
            break

    if project_counts:
        # Return the most common project-level directory
        name, count = project_counts.most_common(1)[0]
        # Only use it if it appears in a meaningful number of paths
        if count >= 2 or len(all_paths) < 5:
            return name

    return ""


def analyze_session(
    store: SessionStore, session_id: str
) -> SessionReplay:
    """Run full analysis on a captured session.

    Args:
        store: The session store.
        session_id: The session to analyze.

    Returns:
        Complete SessionReplay with timeline, insights, and statistics.
    """
    events = store.get_events(session_id)
    metadata = store.get_metadata(session_id)

    if not metadata:
        # Generate metadata from events
        metadata = SessionMetadata(
            session_id=session_id,
            start_time=events[0].timestamp if events else datetime.now(),
            end_time=events[-1].timestamp if events else None,
            event_count=len(events),
        )

    # Build timeline phases
    phases = _identify_phase_runs(events)
    phases = _generate_phase_summaries(phases, events)

    # Extract insights
    insights = _extract_insights(events)

    # Identify decision and turning points
    decision_points = _identify_decision_points(events)
    turning_points = _identify_turning_points(events)

    # Compute statistics
    stats = _compute_statistics(events, metadata)

    # Update metadata with computed info
    all_files = set()
    tool_counts = Counter()
    for e in events:
        all_files.update(e.files_affected)
        if e.tool_name:
            tool_counts[e.tool_name] += 1

    metadata = metadata.model_copy(
        update={
            "event_count": len(events),
            "end_time": events[-1].timestamp if events else metadata.end_time,
            "duration_seconds": stats["duration_seconds"],
            "files_modified": sorted(all_files),
            "tools_used": dict(tool_counts),
        }
    )

    # Auto-detect project name if not set
    if not metadata.project and events:
        detected = _detect_project_name(events)
        if detected:
            metadata = metadata.model_copy(update={"project": detected})

    # Generate session summary
    if not metadata.summary and events:
        metadata = metadata.model_copy(
            update={"summary": _generate_session_summary(events, phases, insights)}
        )

    # Save updated metadata
    store.save_metadata(metadata)

    replay = SessionReplay(
        metadata=metadata,
        timeline=phases,
        insights=insights,
        key_decision_indices=decision_points,
        turning_point_indices=turning_points,
        statistics=stats,
    )

    # Save the replay
    store.save_replay(replay)

    return replay


def _generate_session_summary(
    events: list[Event],
    phases: list[TimelinePhase],
    insights: list[Insight],
) -> str:
    """Generate a narrative session summary."""
    if not events:
        return "Session recorded"

    # Collect all files and extract key directories
    all_files: set[str] = set()
    edited_files: set[str] = set()
    for e in events:
        all_files.update(e.files_affected)
        if e.event_type == EventType.CODE_CHANGE:
            edited_files.update(e.files_affected)

    # Identify the main project area by most common parent directory
    project_dirs: Counter = Counter()
    for f in edited_files or all_files:
        parts = f.rstrip("/").split("/")
        # Find a meaningful directory (skip generic ones)
        for i in range(len(parts) - 1, 0, -1):
            parent = parts[i - 1]
            if parent not in ("", "src", "lib", "bin", "var", "tmp", "usr"):
                project_dirs[parent] += 1
                break

    main_area = ""
    if project_dirs:
        main_area = project_dirs.most_common(1)[0][0]

    # Build narrative from phases
    activities: list[str] = []
    for phase in phases:
        phase_files = set()
        for e in events[phase.start_index : phase.end_index + 1]:
            phase_files.update(e.files_affected)
        short_files = _extract_short_names(phase_files, max_names=2)

        if phase.phase == SessionPhase.EXPLORATION:
            if short_files:
                activities.append(f"explored {', '.join(short_files)}")
            else:
                activities.append("explored the codebase")
        elif phase.phase == SessionPhase.IMPLEMENTATION:
            if short_files:
                activities.append(f"built {', '.join(short_files)}")
            else:
                activities.append("implemented new code")
        elif phase.phase == SessionPhase.DEBUGGING:
            activities.append("debugged issues")
        elif phase.phase == SessionPhase.TESTING:
            activities.append("ran tests")
        elif phase.phase == SessionPhase.CONFIGURATION:
            if short_files:
                activities.append(f"configured {', '.join(short_files)}")
            else:
                activities.append("set up configuration")
        elif phase.phase == SessionPhase.REFACTORING:
            activities.append("refactored code")
        elif phase.phase == SessionPhase.DOCUMENTATION:
            activities.append("updated documentation")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_activities: list[str] = []
    for a in activities:
        if a not in seen:
            seen.add(a)
            unique_activities.append(a)

    # Build the narrative
    if len(unique_activities) <= 1:
        narrative = unique_activities[0] if unique_activities else "worked on files"
    elif len(unique_activities) == 2:
        narrative = f"{unique_activities[0]}, then {unique_activities[1]}"
    else:
        narrative = ", ".join(unique_activities[:-1]) + f", and {unique_activities[-1]}"

    # Capitalize first letter
    narrative = narrative[0].upper() + narrative[1:]

    # Add scope context
    if main_area and len(edited_files) > 1:
        narrative += f". Focused on {main_area}/ ({len(edited_files)} files changed)"
    elif edited_files:
        short = _extract_short_names(edited_files, max_names=3)
        narrative += f". Changed {', '.join(short)}"

    # Add error context if relevant
    error_count = sum(1 for e in events if e.event_type == EventType.ERROR)
    if error_count >= 3:
        narrative += f". Hit {error_count} errors along the way"

    return narrative


def aggregate_learnings(
    store: SessionStore, limit: int = 50
) -> list[Insight]:
    """Aggregate learnings across multiple sessions.

    Args:
        store: The session store.
        limit: Maximum number of sessions to consider.

    Returns:
        Aggregated insights across sessions.
    """
    sessions = store.list_sessions(limit=limit)
    all_insights: list[Insight] = []

    for session_meta in sessions:
        replay = store.get_replay(session_meta.session_id)
        if replay:
            all_insights.extend(replay.insights)

    # Deduplicate similar insights
    unique: list[Insight] = []
    seen_titles: set[str] = set()
    for insight in all_insights:
        title_key = insight.title.lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append(insight)

    # Sort by confidence
    unique.sort(key=lambda x: x.confidence, reverse=True)
    return unique
