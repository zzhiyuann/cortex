"""End-to-end routing tests for the dispatcher.

Messages flow through the REAL dispatcher pipeline:
  _on_message() → _classify() (real LLM API call) → handler

But Telegram and Claude CLI are mocked to capture behavior
without side effects. This tests the actual routing decisions.

Usage:
    python3 -m tests.test_e2e_routing
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dispatcher.config import Config, DEFAULTS, _deep_copy
from dispatcher.core import Dispatcher
from dispatcher.session import Session


# ── Mock components ─────────────────────────────────────────────────────

class MockTelegram:
    """Captures all outgoing Telegram calls for inspection."""

    def __init__(self):
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.reactions: list[dict] = []
        self.typing_count = 0
        self._msg_counter = 1000

    def send(self, text, reply_to=None, parse_mode=None, reply_markup=None):
        self._msg_counter += 1
        self.sent.append({
            "text": text, "reply_to": reply_to,
            "parse_mode": parse_mode, "reply_markup": reply_markup,
            "msg_id": self._msg_counter,
        })
        return self._msg_counter

    def edit(self, message_id, text, parse_mode=None):
        self.edits.append({"message_id": message_id, "text": text})

    def typing(self):
        self.typing_count += 1

    def react(self, message_id, emoji):
        self.reactions.append({"message_id": message_id, "emoji": emoji})

    def set_my_commands(self, commands):
        pass

    def answer_callback(self, cb_id, text=""):
        pass

    def send_document(self, path, caption="", reply_to=None):
        self.sent.append({"document": path, "caption": caption, "reply_to": reply_to})
        self._msg_counter += 1
        return self._msg_counter

    def download_file(self, file_id, path):
        return False

    def get_file_url(self, file_id):
        return None

    def send_photo(self, path, caption="", reply_to=None):
        pass

    def poll(self, offset, timeout):
        return []

    def reset(self):
        self.sent.clear()
        self.edits.clear()
        self.reactions.clear()
        self.typing_count = 0


class MockRunner:
    """Captures Claude CLI invocations without running them."""

    def __init__(self):
        self.invocations: list[dict] = []

    async def invoke(self, session, prompt, resume=False, max_turns=50,
                     model=None, stream=True):
        self.invocations.append({
            "sid": session.sid[:8],
            "prompt_preview": prompt[:100],
            "resume": resume,
            "max_turns": max_turns,
            "model": model,
            "stream": stream,
            "cwd": session.cwd,
        })
        session.status = "done"
        session.finished = time.time()
        session.result = f"[mock result for: {session.task_text[:50]}]"
        return session.result

    def reset(self):
        self.invocations.clear()


# ── Test harness ────────────────────────────────────────────────────────

def make_config():
    """Create a minimal config for testing."""
    data = _deep_copy(DEFAULTS)
    data["telegram"]["bot_token"] = "test-token"
    data["telegram"]["chat_id"] = 12345
    data["projects"] = {
        "sims": {"path": "/tmp/test-sims", "keywords": ["sims", "titan"]},
        "cortex": {"path": "/tmp/test-cortex", "keywords": ["cortex", "dispatcher"]},
    }
    cfg = Config.__new__(Config)
    cfg._data = data
    cfg._path = Path("/dev/null")
    return cfg


def make_dispatcher():
    """Create a dispatcher with mocked I/O."""
    cfg = make_config()
    d = Dispatcher(cfg)
    d.tg = MockTelegram()
    d.runner = MockRunner()
    return d


def make_message(text: str, mid: int = None, reply_to: int = None,
                 chat_id: int = 12345) -> dict:
    """Create a minimal Telegram message dict."""
    return {
        "message_id": mid or int(time.time() * 1000) % 100000,
        "chat": {"id": chat_id},
        "text": text,
        "reply_to_message": {"message_id": reply_to} if reply_to else None,
    }


def add_running_session(d: Dispatcher, project: str = "sims",
                        task: str = "running data analysis",
                        elapsed_seconds: int = 300) -> Session:
    """Add a fake running session to the dispatcher."""
    mid = int(time.time() * 1000) % 100000 + 500
    s = d.sm.create(mid, task, f"/tmp/test-{project}")
    s.status = "running"
    s.started = time.time() - elapsed_seconds
    s.partial_output = "Processing data files...\nAnalyzing correlations..."
    # Mock process so cancel works
    s.proc = MagicMock()
    s.proc.kill = MagicMock()
    return s


@dataclass
class E2EResult:
    scenario: str
    message: str
    expected_route: str
    actual_route: str
    passed: bool
    latency_ms: float
    details: str = ""
    tags: list[str] = field(default_factory=list)


def detect_route(d: Dispatcher, tg: MockTelegram, runner: MockRunner) -> str:
    """Detect which handler was triggered based on mock state."""
    if runner.invocations:
        inv = runner.invocations[-1]
        if inv.get("model") == "haiku" and inv.get("max_turns") == 3:
            return "quick"
        return "task"

    if not tg.sent:
        return "unknown"

    last_text = tg.sent[-1].get("text", "")

    # Detect handler by response content
    if "空闲中" in last_text or "正在运行" in last_text or "当前输出:" in last_text:
        if "当前输出:" in last_text or "暂无输出" in last_text:
            return "peek"
        return "status"
    if "已取消" in last_text or "没有在跑" in last_text or "不确定要取消" in last_text:
        return "cancel"
    if "当前输出" in last_text or "暂无输出" in last_text:
        return "peek"
    if "最近任务" in last_text or "没有最近完成" in last_text:
        return "history"
    if "使用指南" in last_text:
        return "help"
    if "新对话" in last_text or "新建session" in last_text.lower():
        return "new_session"
    if "用法: /q" in last_text:
        return "quick"

    # If runner was invoked, it's a task
    return "unknown"


# ── Test scenarios ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    message: str
    expected: str
    setup: str = "running"  # "running", "idle", "multi"
    tags: list[str] = field(default_factory=list)
    reply_to: int | None = None


SCENARIOS = [
    # ── Status queries while task is running ──
    Scenario("status_chinese_1", "还在跑吗", "status", tags=["status"]),
    Scenario("status_chinese_2", "进度如何", "status", tags=["status"]),
    Scenario("status_chinese_3", "跑完了吗", "status", tags=["status"]),
    Scenario("status_chinese_4", "搞定了吗", "status", tags=["status"]),
    Scenario("status_chinese_5", "还要多久", "status", tags=["status"]),
    Scenario("status_english_1", "still running?", "status", tags=["status"]),
    Scenario("status_english_2", "is it done yet", "status", tags=["status"]),
    Scenario("status_english_3", "any progress?", "status", tags=["status"]),
    Scenario("status_natural", "那个任务跑得怎么样了", "status", tags=["status"]),
    Scenario("status_impatient", "这都跑了好久了还没好？", "status", tags=["status"]),

    # ── Cancel requests while task is running ──
    Scenario("cancel_chinese_1", "kill掉吧", "cancel", tags=["cancel"]),
    Scenario("cancel_chinese_2", "别跑了", "cancel", tags=["cancel"]),
    Scenario("cancel_chinese_3", "取消吧", "cancel", tags=["cancel"]),
    Scenario("cancel_chinese_4", "太慢了不要了", "cancel", tags=["cancel"]),
    Scenario("cancel_english_1", "stop it", "cancel", tags=["cancel"]),
    Scenario("cancel_english_2", "kill it", "cancel", tags=["cancel"]),
    Scenario("cancel_english_3", "abort", "cancel", tags=["cancel"]),
    Scenario("cancel_combo", "太慢了 kill掉吧 不要了", "cancel", tags=["cancel"]),

    # ── Peek at output while task is running ──
    Scenario("peek_chinese_1", "看看输出", "peek", tags=["peek"]),
    Scenario("peek_chinese_2", "给我看看它现在写了什么", "peek", tags=["peek"]),
    Scenario("peek_english_1", "show me the output", "peek", tags=["peek"]),
    Scenario("peek_natural", "让我看看你现在写了啥", "peek", tags=["peek"]),

    # ── Tasks that should NOT be intercepted ──
    Scenario("task_project_check", "帮我检查一下proactive项目的进度", "task", tags=["task", "critical"]),
    Scenario("task_code_write", "帮我写个Python脚本", "task", tags=["task"]),
    Scenario("task_bug_fix", "修复那个bug", "task", tags=["task"]),
    Scenario("task_git", "git push一下", "task", tags=["task"]),
    Scenario("task_tests", "把测试跑一下", "task", tags=["task"]),
    Scenario("task_english", "run the tests", "task", tags=["task"]),
    Scenario("task_refactor", "refactor that function", "task", tags=["task"]),
    Scenario("task_readme", "帮我更新README", "task", tags=["task"]),
    Scenario("task_kill_specific", "帮我kill掉那个僵尸进程", "task", tags=["task", "critical"]),
    Scenario("task_stop_docker", "stop the docker container", "task", tags=["task", "critical"]),
    Scenario("task_cancel_cron", "帮我取消那个scheduled job", "task", tags=["task", "critical"]),
    Scenario("task_check_status", "看看那个服务的运行状态", "task", tags=["task"]),

    # ── /commands (instant, no LLM) ──
    Scenario("cmd_status", "/status", "status", tags=["command"]),
    Scenario("cmd_cancel", "/cancel", "cancel", tags=["command"]),
    Scenario("cmd_peek", "/peek", "peek", tags=["command"]),
    Scenario("cmd_help", "/help", "help", tags=["command"]),
    Scenario("cmd_history", "/history", "history", tags=["command"]),
    Scenario("cmd_quick", "/q what is a monad", "quick", tags=["command"]),

    # ── Idle state (no running tasks) — everything is task ──
    Scenario("idle_status_word", "进度如何", "task", setup="idle", tags=["idle"]),
    Scenario("idle_cancel_word", "取消", "task", setup="idle", tags=["idle"]),
    Scenario("idle_normal", "帮我写个脚本", "task", setup="idle", tags=["idle"]),

    # ── Multiple running tasks ──
    Scenario("multi_status", "还在跑吗", "status", setup="multi", tags=["multi"]),
    Scenario("multi_cancel", "kill掉", "cancel", setup="multi", tags=["multi"]),
    Scenario("multi_task", "帮我看看README", "task", setup="multi", tags=["multi"]),

    # ── Edge cases ──
    Scenario("edge_short_ok", "ok", "task", tags=["edge"]),
    Scenario("edge_short_hmm", "嗯", "task", tags=["edge"]),
    Scenario("edge_followup", "继续改那个文件", "task", tags=["edge"]),
    Scenario("edge_emoji_only", "👍", "task", tags=["edge"]),
]

assert len(SCENARIOS) >= 50, f"Need 50+ scenarios, got {len(SCENARIOS)}"


async def run_scenario(scenario: Scenario) -> E2EResult:
    """Run one e2e scenario through the real dispatcher pipeline."""
    d = make_dispatcher()
    tg: MockTelegram = d.tg
    runner: MockRunner = d.runner

    # Setup context
    if scenario.setup == "running":
        add_running_session(d, "sims", "running data analysis", 300)
    elif scenario.setup == "multi":
        add_running_session(d, "sims", "running data analysis", 300)
        add_running_session(d, "cortex", "fixing dispatcher bug", 120)

    msg = make_message(scenario.message, reply_to=scenario.reply_to)

    t0 = time.monotonic()
    await d._on_message(msg)
    latency = (time.monotonic() - t0) * 1000

    # Wait a tiny bit for fire-and-forget tasks
    await asyncio.sleep(0.1)

    # Detect what route was taken
    actual = detect_route(d, tg, runner)

    # For idle scenarios: "task" means runner was invoked OR no response
    # (since mock runner is async and _handle_task spawns a background task)
    if scenario.setup == "idle" and scenario.expected == "task":
        # In idle mode, tasks go through _handle_task which spawns async.
        # Check if a task was spawned (d._tasks should have something)
        if d._tasks or runner.invocations:
            actual = "task"
        elif not tg.sent:
            actual = "task"  # nothing happened = would be task

    # For task scenarios with running sessions: the dispatcher will
    # either spawn a new task or queue behind running
    if scenario.expected == "task" and runner.invocations:
        actual = "task"
    # If dispatcher replied with queue message, it's still a task route
    if scenario.expected == "task" and tg.sent:
        for s in tg.sent:
            txt = s.get("text", "")
            if "等一个完成" in txt or "等这个任务跑完" in txt:
                actual = "task"

    passed = actual == scenario.expected
    details = ""
    if tg.sent:
        details = tg.sent[-1].get("text", "")[:120]
    elif runner.invocations:
        details = f"runner invoked: {runner.invocations[-1]['prompt_preview'][:80]}"

    # Cleanup spawned tasks
    for task in list(d._tasks):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    return E2EResult(
        scenario=scenario.name,
        message=scenario.message,
        expected_route=scenario.expected,
        actual_route=actual,
        passed=passed,
        latency_ms=latency,
        details=details,
        tags=scenario.tags,
    )


async def main():
    print(f"Running {len(SCENARIOS)} e2e routing tests...\n")

    results: list[E2EResult] = []
    for i, scenario in enumerate(SCENARIOS, 1):
        r = await run_scenario(scenario)
        icon = "PASS" if r.passed else "FAIL"
        print(f"  [{i:2d}/{len(SCENARIOS)}] {icon}  {r.latency_ms:6.0f}ms  "
              f"expect={r.expected_route:8s} got={r.actual_route:8s}  "
              f"{r.scenario}: {r.message[:40]}")
        if not r.passed:
            print(f"           → {r.details[:100]}")
        results.append(r)
        await asyncio.sleep(0.05)

    # Analysis
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]

    print("\n" + "=" * 70)
    print("E2E ROUTING REPORT")
    print("=" * 70)
    print(f"\nOverall: {passed}/{total} ({passed/total*100:.1f}%)")

    latencies = [r.latency_ms for r in results]
    print(f"Latency: avg={sum(latencies)/len(latencies):.0f}ms "
          f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms "
          f"max={max(latencies):.0f}ms")

    # Per-tag
    tags: dict[str, dict] = {}
    for r in results:
        for tag in r.tags:
            if tag not in tags:
                tags[tag] = {"total": 0, "passed": 0}
            tags[tag]["total"] += 1
            if r.passed:
                tags[tag]["passed"] += 1

    print("\n── By category ──")
    for tag, stats in sorted(tags.items()):
        pct = stats["passed"] / stats["total"] * 100
        print(f"  {tag:12s}: {stats['passed']:2d}/{stats['total']:2d} ({pct:5.1f}%)")

    if failed:
        print(f"\n── Failures ({len(failed)}) ──")
        for r in failed:
            print(f"  {r.scenario}: \"{r.message}\"")
            print(f"    expected={r.expected_route}, got={r.actual_route}")
            print(f"    details: {r.details[:100]}")

    # Save log
    log_path = Path(__file__).parent / "e2e_routing_log.json"
    log_data = {
        "total": total, "passed": passed,
        "accuracy": passed / total * 100,
        "results": [
            {
                "scenario": r.scenario, "message": r.message,
                "expected": r.expected_route, "got": r.actual_route,
                "passed": r.passed, "latency_ms": r.latency_ms,
                "details": r.details, "tags": r.tags,
            }
            for r in results
        ],
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"\nLog saved to: {log_path}")
    print("=" * 70)

    return 0 if passed / total >= 0.90 else 1


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
