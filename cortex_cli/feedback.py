"""Event-driven self-healing — issues trigger a resolver agent with accumulated context.

Architecture:
- Any Cortex component calls record_issue() when something goes wrong
- record_issue() persists the issue AND wakes the resolver agent
- The resolver agent is a single persistent entity that accumulates context:
  it knows what it fixed before, what patterns it's seen, what worked
- When no issues exist, the resolver sleeps (no process running)
- When an issue arrives and no resolver is running, one spawns automatically
- The resolver fixes the issue, checks for more open issues, then exits

This is NOT a batch processor. It's an event-driven agent with memory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("cortex.feedback")

FEEDBACK_DIR = Path.home() / ".cortex" / "feedback"
ISSUES_FILE = FEEDBACK_DIR / "issues.jsonl"
RESOLVED_FILE = FEEDBACK_DIR / "resolved.jsonl"
HISTORY_FILE = FEEDBACK_DIR / "history.jsonl"  # rich resolution context
RESOLVER_PID = FEEDBACK_DIR / "resolver.pid"
RESOLVER_LOG = FEEDBACK_DIR / "resolver.log"

# Project locations for the resolver
PROJECTS = {
    "dispatcher": "/Users/zwang/projects/dispatcher",
    "vibe-replay": "/Users/zwang/projects/vibe-replay",
    "cortex": "/Users/zwang/projects/cortex",
    "forge": "/Users/zwang/projects/forge",
    "a2a-hub": "/Users/zwang/projects/a2a-hub",
}


def _ensure_dir():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def record_issue(
    source: str,
    category: str,
    description: str,
    context: dict | None = None,
    auto_resolve: bool = True,
):
    """Record an issue and wake the resolver agent.

    Args:
        source: Which component reported it (dispatcher, vibe-replay, forge, etc.)
        category: Issue category (error, timeout, user_cancel, ux_friction, crash)
        description: Human-readable description of what went wrong
        context: Additional context (error message, user message, session info, etc.)
        auto_resolve: Whether to auto-trigger the resolver agent
    """
    _ensure_dir()

    issue = {
        "id": f"{source}-{int(time.time() * 1000)}",
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "category": category,
        "description": description,
        "context": context or {},
        "status": "open",
    }

    with open(ISSUES_FILE, "a") as f:
        f.write(json.dumps(issue) + "\n")

    log.info("recorded issue %s: [%s] %s", issue["id"], category, description[:80])

    if auto_resolve:
        _wake_resolver(issue)

    return issue["id"]


def get_open_issues(limit: int = 20) -> list[dict]:
    """Get unresolved issues, newest first."""
    if not ISSUES_FILE.exists():
        return []

    resolved_ids = _get_resolved_ids()

    issues = []
    for line in ISSUES_FILE.read_text().strip().split("\n"):
        if line.strip():
            try:
                issue = json.loads(line)
                if issue["id"] not in resolved_ids:
                    issues.append(issue)
            except json.JSONDecodeError:
                pass

    issues.sort(key=lambda x: x["timestamp"], reverse=True)
    return issues[:limit]


def _get_resolved_ids() -> set[str]:
    """Get set of resolved issue IDs."""
    resolved_ids = set()
    if RESOLVED_FILE.exists():
        for line in RESOLVED_FILE.read_text().strip().split("\n"):
            if line.strip():
                try:
                    resolved_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return resolved_ids


def resolve_issue(issue_id: str, resolution: str = ""):
    """Mark an issue as resolved with context."""
    _ensure_dir()
    entry = {
        "id": issue_id,
        "resolved_at": datetime.now().isoformat(),
        "resolution": resolution,
    }
    with open(RESOLVED_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Also write to rich history for resolver context
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_issue_summary() -> dict:
    """Get a summary of issue counts by category and source."""
    issues = get_open_issues(limit=100)
    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for issue in issues:
        cat = issue.get("category", "unknown")
        src = issue.get("source", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "total_open": len(issues),
        "by_category": by_category,
        "by_source": by_source,
    }


def get_resolution_history(limit: int = 20) -> list[dict]:
    """Get past resolutions for accumulated context."""
    if not HISTORY_FILE.exists():
        return []
    entries = []
    for line in HISTORY_FILE.read_text().strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries[-limit:]


# -- Resolver Agent --

def _resolver_running() -> bool:
    """Check if the resolver agent is currently active."""
    if not RESOLVER_PID.exists():
        return False
    try:
        pid = int(RESOLVER_PID.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        RESOLVER_PID.unlink(missing_ok=True)
        return False


def _wake_resolver(trigger_issue: dict):
    """Wake the resolver agent to handle an issue.

    If already running, does nothing — the running resolver will see
    the new issue when it checks for open issues after finishing the current one.
    """
    if _resolver_running():
        log.info("resolver already running, issue queued: %s", trigger_issue["id"])
        return

    # Skip auto-resolve for cancellations (user chose to cancel, not a bug)
    if trigger_issue.get("category") == "user_cancel":
        return

    open_issues = get_open_issues(limit=10)
    if not open_issues:
        return

    history = get_resolution_history(limit=15)
    prompt = _build_resolver_prompt(trigger_issue, open_issues, history)

    # Find claude binary
    claude_bin = shutil.which("claude") or "claude"

    env = os.environ.copy()
    for var in ("CLAUDE_CODE", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(var, None)

    _ensure_dir()

    # Determine working directory from the issue source
    source = trigger_issue.get("source", "cortex")
    cwd = PROJECTS.get(source, PROJECTS.get("cortex", str(Path.home())))

    try:
        with open(RESOLVER_LOG, "a") as logf:
            logf.write(f"\n{'='*60}\n")
            logf.write(f"[{datetime.now().isoformat()}] Resolver waking for: {trigger_issue['id']}\n")
            logf.write(f"{'='*60}\n")

            proc = subprocess.Popen(
                [
                    claude_bin, "-p",
                    "--dangerously-skip-permissions",
                    "--max-turns", "15",
                    prompt,
                ],
                stdout=logf,
                stderr=logf,
                start_new_session=True,
                env=env,
                cwd=cwd,
            )

        RESOLVER_PID.write_text(str(proc.pid))
        log.info("resolver spawned (pid %d) for issue %s", proc.pid, trigger_issue["id"])

    except Exception:
        log.exception("failed to spawn resolver")


def _build_resolver_prompt(
    trigger: dict,
    open_issues: list[dict],
    history: list[dict],
) -> str:
    """Build a contextual prompt with accumulated knowledge.

    The resolver sees:
    1. The triggering issue (what just happened)
    2. All open issues (what else needs fixing)
    3. Resolution history (what was fixed before and how)
    This ensures fixes are NOT independent — each builds on prior knowledge.
    """
    parts = [
        "You are the Cortex Self-Healing Resolver. An issue just occurred in the system.",
        "Your job: diagnose the root cause, implement a fix, run tests, commit, and mark it resolved.",
        "",
    ]

    # Resolution history — accumulated context
    if history:
        parts.append("## What You've Fixed Before (accumulated context)")
        parts.append("Use this to understand patterns and avoid repeating past mistakes.\n")
        for h in history:
            ts = h.get("resolved_at", "?")[:16]
            res = h.get("resolution", "no details")
            parts.append(f"- [{ts}] {h.get('id', '?')}: {res[:150]}")
        parts.append("")

    # Current trigger
    parts.append("## Triggering Issue (fix this first)")
    parts.append(f"**ID:** {trigger['id']}")
    parts.append(f"**Source:** {trigger['source']}")
    parts.append(f"**Category:** {trigger['category']}")
    parts.append(f"**When:** {trigger['timestamp']}")
    parts.append(f"**What:** {trigger['description']}")
    ctx = trigger.get("context", {})
    if ctx.get("error"):
        parts.append(f"**Error:** {ctx['error'][:300]}")
    if ctx.get("user_message"):
        parts.append(f"**User said:** {ctx['user_message'][:150]}")
    if ctx.get("project"):
        parts.append(f"**Project:** {ctx['project']}")
    if ctx.get("suggestion"):
        parts.append(f"**Suggestion:** {ctx['suggestion']}")
    parts.append("")

    # Other open issues
    other = [i for i in open_issues if i["id"] != trigger["id"]]
    if other:
        parts.append("## Other Open Issues (fix if time permits)")
        for i in other[:5]:
            parts.append(f"- [{i['source']}] {i['category']}: {i['description'][:100]}")
        parts.append("")

    # Project locations
    parts.append("## Project Locations")
    for name, path in PROJECTS.items():
        parts.append(f"- {name}: {path}")
    parts.append("")

    # Instructions
    parts.append("## Rules")
    parts.append("1. Read the relevant source code to understand the problem")
    parts.append("2. Identify root cause (not just symptoms)")
    parts.append("3. Implement a minimal, targeted fix")
    parts.append("4. Run tests for the affected project")
    parts.append("5. Commit with a clear message")
    parts.append("6. Mark the issue as resolved by appending to " + str(RESOLVED_FILE))
    parts.append("   Format: {\"id\": \"<issue_id>\", \"resolved_at\": \"<iso_timestamp>\", \"resolution\": \"<what you did>\"}")
    parts.append("7. Check if there are more open issues in " + str(ISSUES_FILE))
    parts.append("8. If yes and you have turns left, fix the next one")
    parts.append("9. Keep fixes minimal — don't refactor unrelated code")
    parts.append("")
    parts.append("## Testing")
    for name, path in PROJECTS.items():
        venv = Path(path) / ".venv" / "bin" / "python3"
        if name != "cortex":
            parts.append(f"{venv} -m pytest {path}/tests/ -v")
    parts.append("")
    parts.append("Go. Fix it.")

    return "\n".join(parts)


def build_improvement_prompt(issues: list[dict] | None = None) -> str:
    """Build a prompt section for the autonomous agent (used by cortex agent)."""
    if issues is None:
        issues = get_open_issues(limit=10)

    if not issues:
        return ""

    prompt_parts = [
        "## Open Issues from User Interactions\n",
        "These are real problems encountered during usage. Fix them.\n",
    ]

    for i, issue in enumerate(issues, 1):
        prompt_parts.append(
            f"\n### Issue {i}: [{issue['source']}] {issue['category']}\n"
            f"**When:** {issue['timestamp']}\n"
            f"**What:** {issue['description']}\n"
        )
        ctx = issue.get("context", {})
        if ctx.get("error"):
            prompt_parts.append(f"**Error:** {ctx['error'][:200]}\n")
        if ctx.get("user_message"):
            prompt_parts.append(f"**User said:** {ctx['user_message'][:100]}\n")
        if ctx.get("suggestion"):
            prompt_parts.append(f"**Suggestion:** {ctx['suggestion']}\n")

    prompt_parts.append(
        "\nFor each issue: identify root cause, implement fix, run tests, commit.\n"
        "Mark issues as resolved by writing to the resolved file.\n"
    )

    return "\n".join(prompt_parts)


def resolver_status() -> dict:
    """Get resolver status for CLI display."""
    running = _resolver_running()
    pid = None
    if running and RESOLVER_PID.exists():
        try:
            pid = int(RESOLVER_PID.read_text().strip())
        except ValueError:
            pass

    return {
        "running": running,
        "pid": pid,
        "open_issues": len(get_open_issues(limit=100)),
        "total_resolved": len(get_resolution_history(limit=1000)),
        "log_file": str(RESOLVER_LOG),
    }
