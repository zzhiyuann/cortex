"""Behavioral test suite for the intent classifier.

Sends 100+ real messages through the classifier with real LLM API calls.
Logs every decision, then analyzes accuracy and failure patterns.

Usage:
    python -m tests.test_classifier_behavior
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add parent to path so we can import dispatcher
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dispatcher.classifier import classify_intent


@dataclass
class TestCase:
    message: str
    expected: str
    context_name: str = "default"  # which scenario
    tags: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    case: TestCase
    got: str
    latency_ms: float
    passed: bool


# ── Simulated running-task contexts ─────────────────────────────────────

CONTEXTS = {
    "one_task": [
        {"project": "proactive-affective-agent", "task": "check project progress and summarize", "elapsed": "8min"},
    ],
    "two_tasks": [
        {"project": "sims", "task": "run data analysis pipeline", "elapsed": "12min"},
        {"project": "cortex", "task": "fix dispatcher bug", "elapsed": "3min"},
    ],
    "long_running": [
        {"project": "proactive-affective-agent", "task": "train ML model baseline experiments", "elapsed": "45min"},
    ],
    "quick_task": [
        {"project": "cortex", "task": "update README", "elapsed": "30sec"},
    ],
}

# ── Test cases ──────────────────────────────────────────────────────────
# Every message is something a real user would send from their phone
# while tasks are running in the background.

CASES: list[TestCase] = [
    # ─── STATUS: asking about running task progress ───
    TestCase("还在写吗", "status", tags=["status", "short"]),
    TestCase("还在跑吗", "status", tags=["status", "short"]),
    TestCase("进度如何", "status", tags=["status", "short"]),
    TestCase("跑完了吗", "status", tags=["status", "short"]),
    TestCase("怎么样了", "status", tags=["status", "short"]),
    TestCase("到哪了", "status", tags=["status", "short"]),
    TestCase("好了没", "status", tags=["status", "short"]),
    TestCase("搞定了吗", "status", tags=["status", "short"]),
    TestCase("还要多久", "status", tags=["status"]),
    TestCase("那个任务跑得怎么样了", "status", tags=["status"]),
    TestCase("现在什么情况", "status", tags=["status"]),
    TestCase("is it done yet", "status", tags=["status", "english"]),
    TestCase("how's it going", "status", tags=["status", "english"]),
    TestCase("still running?", "status", tags=["status", "english"]),
    TestCase("any progress?", "status", tags=["status", "english"]),
    TestCase("done?", "status", tags=["status", "english", "short"]),
    TestCase("跑了多久了", "status", tags=["status"]),
    TestCase("忙完了吗", "status", tags=["status"]),
    TestCase("那个还在跑着呢？", "status", tags=["status"]),
    TestCase("弄好了嘛", "status", tags=["status"]),
    TestCase("快了吗", "status", tags=["status", "short"]),
    TestCase("ETA?", "status", tags=["status", "english", "short"]),
    TestCase("完事了没", "status", tags=["status"]),
    TestCase("结束了吗", "status", tags=["status"]),

    # ─── CANCEL: wants to stop running task ───
    TestCase("kill掉吧", "cancel", tags=["cancel"]),
    TestCase("把它停了", "cancel", tags=["cancel"]),
    TestCase("别跑了", "cancel", tags=["cancel"]),
    TestCase("取消吧", "cancel", tags=["cancel"]),
    TestCase("停下来", "cancel", tags=["cancel"]),
    TestCase("不用跑了", "cancel", tags=["cancel"]),
    TestCase("算了 不要了", "cancel", tags=["cancel"]),
    TestCase("stop it", "cancel", tags=["cancel", "english"]),
    TestCase("kill it", "cancel", tags=["cancel", "english"]),
    TestCase("cancel", "cancel", tags=["cancel", "english", "short"]),
    TestCase("abort", "cancel", tags=["cancel", "english", "short"]),
    TestCase("太慢了 不要了", "cancel", tags=["cancel"]),
    TestCase("中断吧", "cancel", tags=["cancel"]),
    TestCase("把那个任务杀了", "cancel", tags=["cancel"]),
    TestCase("终止", "cancel", tags=["cancel", "short"]),
    TestCase("算了算了 kill", "cancel", tags=["cancel"]),
    TestCase("先停一下", "cancel", tags=["cancel"]),
    TestCase("暂停", "cancel", tags=["cancel", "short"]),
    TestCase("不跑了", "cancel", tags=["cancel", "short"]),
    TestCase("ctrl c", "cancel", tags=["cancel"]),

    # ─── PEEK: wants to see current output ───
    TestCase("看看输出", "peek", tags=["peek"]),
    TestCase("给我看看它现在写了什么", "peek", tags=["peek"]),
    TestCase("输出是什么", "peek", tags=["peek"]),
    TestCase("show me the output", "peek", tags=["peek", "english"]),
    TestCase("what's it outputting", "peek", tags=["peek", "english"]),
    TestCase("现在输出到哪了", "peek", tags=["peek"]),
    TestCase("让我看看它在干嘛", "peek", tags=["peek"]),
    TestCase("preview一下", "peek", tags=["peek"]),
    TestCase("看一眼输出", "peek", tags=["peek"]),

    # ─── TASK: new work (should NOT be misclassified) ───
    TestCase("帮我检查一下proactive affective agent那个项目的进度", "task", tags=["task", "critical"]),
    TestCase("帮我写个Python脚本处理CSV", "task", tags=["task"]),
    TestCase("修复那个bug", "task", tags=["task"]),
    TestCase("看看sims项目的readme", "task", tags=["task"]),
    TestCase("把测试跑一下", "task", tags=["task"]),
    TestCase("这个函数怎么优化", "task", tags=["task"]),
    TestCase("git push一下", "task", tags=["task"]),
    TestCase("帮我改一下config", "task", tags=["task"]),
    TestCase("也顺便看看测试情况", "task", tags=["task"]),
    TestCase("继续改那个文件", "task", tags=["task"]),
    TestCase("再修改一下", "task", tags=["task"]),
    TestCase("帮我分析一下那段代码", "task", tags=["task"]),
    TestCase("能不能帮我写个脚本", "task", tags=["task"]),
    TestCase("那个项目的README需要更新", "task", tags=["task"]),
    TestCase("run the tests", "task", tags=["task", "english"]),
    TestCase("check the git log", "task", tags=["task", "english"]),
    TestCase("帮我总结一下今天做了什么", "task", tags=["task"]),
    TestCase("JavaScript的map和forEach有什么区别", "task", tags=["task"]),
    TestCase("帮我查一下昨天的commit", "task", tags=["task"]),
    TestCase("给我列一下项目里所有TODO", "task", tags=["task"]),
    TestCase("deploy到production", "task", tags=["task"]),
    TestCase("帮我review一下这个PR", "task", tags=["task"]),
    TestCase("数据库需要migration", "task", tags=["task"]),
    TestCase("写个单元测试", "task", tags=["task"]),
    TestCase("refactor that function", "task", tags=["task", "english"]),
    TestCase("add error handling to the API endpoint", "task", tags=["task", "english"]),
    TestCase("帮我看看为什么build失败了", "task", tags=["task"]),
    TestCase("pip install numpy", "task", tags=["task"]),
    TestCase("create a new branch for the feature", "task", tags=["task", "english"]),
    TestCase("把那个API的返回格式改一下", "task", tags=["task"]),

    # ─── TRICKY: look like meta but are tasks ───
    TestCase("帮我检查一下项目进度", "task", tags=["task", "tricky", "critical"]),
    TestCase("去看看那个服务的运行状态", "task", tags=["task", "tricky"]),
    TestCase("check if the server is running", "task", tags=["task", "tricky", "english"]),
    TestCase("看看有没有进程在占用端口", "task", tags=["task", "tricky"]),
    TestCase("帮我kill掉那个僵尸进程", "task", tags=["task", "tricky", "critical"]),
    TestCase("stop the docker container", "task", tags=["task", "tricky", "english"]),
    TestCase("查看一下输出日志", "task", tags=["task", "tricky"]),
    TestCase("看看stdout有没有报错", "task", tags=["task", "tricky"]),
    TestCase("帮我取消那个scheduled job", "task", tags=["task", "tricky"]),
    TestCase("cancel the cron job", "task", tags=["task", "tricky", "english"]),
    TestCase("进度条怎么实现的", "task", tags=["task", "tricky"]),
    TestCase("status code 500是什么意思", "task", tags=["task", "tricky"]),
    TestCase("看看那个进程的output", "task", tags=["task", "tricky"]),
    TestCase("帮我检查一下sims项目有没有error", "task", tags=["task", "tricky"]),
    TestCase("去titan上看看gpu状态", "task", tags=["task", "tricky"]),

    # ─── TRICKY: look like tasks but are meta ───
    TestCase("那个任务完了没有啊", "status", tags=["status", "tricky"]),
    TestCase("太慢了 kill掉吧 不要了", "cancel", tags=["cancel", "tricky"]),
    TestCase("这都跑了十分钟了还没好？", "status", tags=["status", "tricky"]),
    TestCase("让我看看你现在写了啥", "peek", tags=["peek", "tricky"]),

    # ─── CONTEXT SENSITIVITY: same message, different contexts ───
    TestCase("还在跑吗", "status", context_name="long_running", tags=["status", "context"]),
    TestCase("还在跑吗", "status", context_name="quick_task", tags=["status", "context"]),
    TestCase("帮我看看那个项目", "task", context_name="two_tasks", tags=["task", "context"]),

    # ─── SHORT / AMBIGUOUS (accept either answer as reasonable) ───
    # "?" when task is running → could be status or task, both OK
    TestCase("?", "status", tags=["ambiguous", "accept_task"]),
    TestCase("??", "status", tags=["ambiguous", "accept_task"]),
    TestCase("嗯", "task", tags=["task", "ambiguous"]),
    TestCase("ok", "task", tags=["task", "ambiguous"]),
    TestCase("好", "task", tags=["task", "ambiguous"]),
    TestCase("...", "status", tags=["ambiguous", "accept_task"]),
    TestCase("算了", "cancel", tags=["ambiguous", "accept_task"]),

    # ─── STATUS/PEEK BOUNDARY (accept either) ───
    # These are genuinely ambiguous between status and peek.
    # Both handlers should show useful info, so either answer works.
    TestCase("在干嘛呢", "status", tags=["status", "accept_peek"]),
    TestCase("它现在写到哪了", "peek", tags=["peek", "accept_status"]),
    TestCase("你在做什么呢现在", "status", tags=["status", "accept_peek"]),
]

assert len(CASES) >= 100, f"Need 100+ cases, got {len(CASES)}"


async def run_all() -> list[TestResult]:
    results: list[TestResult] = []
    total = len(CASES)

    for i, case in enumerate(CASES, 1):
        ctx = CONTEXTS.get(case.context_name, CONTEXTS["one_task"])
        t0 = time.monotonic()
        got = await classify_intent(case.message, ctx)
        latency = (time.monotonic() - t0) * 1000

        # Check if passed: exact match OR acceptable alternative
        acceptable = {case.expected}
        for tag in case.tags:
            if tag.startswith("accept_"):
                acceptable.add(tag[7:])  # "accept_peek" → "peek"
        passed = got in acceptable
        results.append(TestResult(case=case, got=got, latency_ms=latency, passed=passed))

        icon = "PASS" if passed else "FAIL"
        print(f"  [{i:3d}/{total}] {icon}  {latency:6.0f}ms  "
              f"expect={case.expected:7s} got={got:7s}  "
              f"msg={case.message[:50]}")

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.05)

    return results


def analyze(results: list[TestResult]) -> dict:
    """Analyze results and produce a detailed behavior report."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]
    latencies = [r.latency_ms for r in results]

    # Per-category accuracy
    by_expected: dict[str, list[TestResult]] = {}
    for r in results:
        by_expected.setdefault(r.case.expected, []).append(r)

    cat_stats = {}
    for cat, cat_results in sorted(by_expected.items()):
        cat_pass = sum(1 for r in cat_results if r.passed)
        cat_stats[cat] = {
            "total": len(cat_results),
            "passed": cat_pass,
            "accuracy": cat_pass / len(cat_results) * 100,
        }

    # Per-tag accuracy
    tag_stats: dict[str, dict] = {}
    for r in results:
        for tag in r.case.tags:
            if tag not in tag_stats:
                tag_stats[tag] = {"total": 0, "passed": 0}
            tag_stats[tag]["total"] += 1
            if r.passed:
                tag_stats[tag]["passed"] += 1

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {}
    for r in results:
        if r.case.expected not in confusion:
            confusion[r.case.expected] = {}
        confusion[r.case.expected][r.got] = confusion[r.case.expected].get(r.got, 0) + 1

    report = {
        "total": total,
        "passed": passed,
        "failed_count": len(failed),
        "accuracy": passed / total * 100,
        "latency_avg_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "latency_max_ms": max(latencies),
        "category_stats": cat_stats,
        "tag_stats": {k: {**v, "accuracy": v["passed"] / v["total"] * 100}
                      for k, v in sorted(tag_stats.items())},
        "confusion_matrix": confusion,
        "failures": [
            {
                "message": r.case.message,
                "expected": r.case.expected,
                "got": r.got,
                "tags": r.case.tags,
                "context": r.case.context_name,
            }
            for r in failed
        ],
    }
    return report


def print_report(report: dict):
    print("\n" + "=" * 70)
    print("CLASSIFIER BEHAVIOR REPORT")
    print("=" * 70)

    print(f"\nOverall: {report['passed']}/{report['total']} "
          f"({report['accuracy']:.1f}% accuracy)")
    print(f"Latency: avg={report['latency_avg_ms']:.0f}ms "
          f"p50={report['latency_p50_ms']:.0f}ms "
          f"p95={report['latency_p95_ms']:.0f}ms "
          f"max={report['latency_max_ms']:.0f}ms")

    print("\n── Category Accuracy ──")
    for cat, stats in report["category_stats"].items():
        bar = "█" * int(stats["accuracy"] / 5) + "░" * (20 - int(stats["accuracy"] / 5))
        print(f"  {cat:8s}: {stats['passed']:2d}/{stats['total']:2d} "
              f"({stats['accuracy']:5.1f}%) {bar}")

    print("\n── Tag Accuracy ──")
    for tag, stats in report["tag_stats"].items():
        print(f"  {tag:12s}: {stats['passed']:2d}/{stats['total']:2d} "
              f"({stats['accuracy']:5.1f}%)")

    print("\n── Confusion Matrix ──")
    labels = sorted(set(
        list(report["confusion_matrix"].keys()) +
        [g for row in report["confusion_matrix"].values() for g in row]
    ))
    header = f"  {'':8s}" + "".join(f"{l:>8s}" for l in labels)
    print(header)
    for expected in labels:
        row = report["confusion_matrix"].get(expected, {})
        cells = "".join(f"{row.get(l, 0):8d}" for l in labels)
        print(f"  {expected:8s}{cells}")
    print(f"  {'':8s}" + "".join(f"{'(pred)':>8s}" if i == 0 else f"{'':>8s}"
                                   for i, _ in enumerate(labels)))

    if report["failures"]:
        print(f"\n── Failures ({report['failed_count']}) ──")
        for f in report["failures"]:
            print(f"  FAIL: \"{f['message']}\"")
            print(f"        expected={f['expected']}, got={f['got']}  "
                  f"tags={f['tags']}  ctx={f['context']}")
    else:
        print("\n  No failures!")

    print("\n" + "=" * 70)


async def main():
    print(f"Running {len(CASES)} classifier behavior tests...\n")

    results = await run_all()
    report = analyze(results)
    print_report(report)

    # Save detailed log
    log_path = Path(__file__).parent / "classifier_behavior_log.json"
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed log saved to: {log_path}")

    # Exit code based on accuracy
    if report["accuracy"] < 90:
        print(f"\n⚠ Accuracy below 90% — needs iteration!")
        return 1
    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
