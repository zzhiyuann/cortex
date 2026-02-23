"""Lightweight LLM intent classifier for message routing.

When tasks are actively running, the dispatcher needs to know whether
a new message is about managing a running task (status/cancel/peek)
or is new work to delegate. A fast LLM call (~200ms) handles this
instead of brittle regex patterns.

Supports: Anthropic Haiku, OpenAI GPT-4o-mini.
Falls back to "task" on any error — worst case is old behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

log = logging.getLogger("dispatcher")

CLASSIFY_TIMEOUT = 3  # seconds

_SYSTEM = (
    "You are an intent classifier for a task dispatcher. "
    "The user remotely controls an AI coding agent from their phone. "
    "Tasks run in the background and the user may send messages that are "
    "either new work or about managing running tasks.\n\n"
    "Classify the user's message into exactly ONE category:\n\n"
    "- task: New work, question, follow-up instruction, or anything to be executed by the agent\n"
    "- status: Asking about progress, state, or ETA of a running task\n"
    "- cancel: Wants to stop, kill, or abort a running task\n"
    "- peek: Wants to see current output or what the agent is doing right now\n\n"
    "Critical distinction — task vs cancel/peek:\n"
    "- If the message mentions a SPECIFIC target (a process name, container, service, "
    "file, job, cron, server, etc.) it is a TASK — the user wants the agent to go "
    "operate on that target. Examples: 'kill the zombie process', 'stop the docker "
    "container', 'cancel the cron job', 'check the output log' → all TASK.\n"
    "- cancel/peek/status are ONLY when the user is talking about the dispatcher's "
    "own running tasks shown above, with NO specific external target. Examples: "
    "'kill it', 'stop', 'show me the output', 'how's it going' → meta-commands.\n"
    "- A request to CHECK, REVIEW, or INSPECT anything inside a project is a TASK.\n"
    "- Questions about technical concepts are TASK even if they contain words like "
    "'output', 'status', 'progress'. Examples: 'how to implement a progress bar', "
    "'what does status code 500 mean', 'check the output log', "
    "'look at stdout for errors' → all TASK.\n"
    "- Short acknowledgments like 'ok', 'got it', 'sure', 'hmm' are TASK "
    "(the user is not asking about anything).\n"
    "- Health and lifestyle messages are ALWAYS TASK — never status/cancel/peek. "
    "This includes anything about food, eating, meals, weight, body, exercise, "
    "workout, sleep, mood, health symptoms, or daily routines. Examples: "
    "'今天吃了xxx', 'I had rice for lunch', '体重降了', 'went for a run', "
    "'slept badly', '感觉有点累' → all TASK.\n"
    "- When in doubt, return 'task'.\n\n"
    "Respond with ONLY the category name, nothing else."
)

_VALID_LABELS = frozenset(("task", "status", "cancel", "peek"))


def _load_key(name: str) -> str:
    """Load an API key from env or common .env files."""
    val = os.environ.get(name, "")
    if val:
        return val
    for p in (Path.home() / ".openclaw" / ".env", Path.home() / ".env"):
        if p.exists():
            try:
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if line.startswith(f"{name}="):
                        return line.split("=", 1)[1].strip().strip("\"'")
            except Exception:
                pass
    return ""


def _get_backend() -> tuple[str, str] | None:
    """Detect available API backend. Returns (backend, api_key) or None."""
    key = _load_key("ANTHROPIC_API_KEY")
    if key:
        return ("anthropic", key)
    key = _load_key("OPENAI_API_KEY")
    if key:
        return ("openai", key)
    return None


async def classify_intent(
    message: str,
    active_sessions: list[dict],
) -> str:
    """Classify user intent via a fast LLM call.

    Args:
        message: The user's raw message text.
        active_sessions: List of dicts with keys: project, task, elapsed.

    Returns:
        One of: "task", "status", "cancel", "peek".
        Defaults to "task" on any failure.
    """
    if not active_sessions:
        return "task"

    backend = _get_backend()
    if not backend:
        log.debug("no API key available, skipping classifier")
        return "task"

    backend_name, api_key = backend

    # Build context
    ctx_lines = ["Currently running tasks:"]
    for s in active_sessions:
        ctx_lines.append(f"- {s['project']}: {s['task']} (running for {s['elapsed']})")
    user_content = "\n".join(ctx_lines) + f"\n\nUser message: {message}"

    try:
        label = await asyncio.wait_for(
            asyncio.to_thread(_call, backend_name, api_key, user_content),
            timeout=CLASSIFY_TIMEOUT,
        )
        label = label.strip().lower()
        if label in _VALID_LABELS:
            log.info("classifier: '%s' -> %s", message[:60], label)
            return label
        log.warning("classifier unexpected label: '%s'", label)
        return "task"
    except asyncio.TimeoutError:
        log.warning("classifier timed out")
        return "task"
    except Exception:
        log.warning("classifier error", exc_info=True)
        return "task"


def _call(backend: str, api_key: str, user_content: str) -> str:
    """Make the API call. Runs in a thread."""
    if backend == "anthropic":
        return _call_anthropic(api_key, user_content)
    return _call_openai(api_key, user_content)


def _call_anthropic(api_key: str, user_content: str) -> str:
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 10,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user_content}],
    }).encode()
    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    resp = urlopen(req, timeout=CLASSIFY_TIMEOUT)
    data = json.loads(resp.read())
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return "task"


def _call_openai(api_key: str, user_content: str) -> str:
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "max_tokens": 10,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
    }).encode()
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    resp = urlopen(req, timeout=CLASSIFY_TIMEOUT)
    data = json.loads(resp.read())
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "task")
    return "task"
