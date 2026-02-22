"""Code Review Agent — Reviews Python code for common issues.

Uses simple heuristics (no LLM needed) to demonstrate the agent pattern.
Checks for: missing docstrings, bare excepts, print statements,
long lines, TODO comments, and mutable default arguments.

Usage:
    python examples/code_review_agent.py
"""

from __future__ import annotations

import argparse
import re

from a2a_hub import Agent, capability


class CodeReviewAgent(Agent):
    """Agent that reviews Python code for common issues."""

    name = "code-reviewer"

    @capability("code-review", description="Review Python code for bugs and style issues")
    async def review(self, code: str, language: str = "python") -> dict:
        """Analyze code and return a list of issues with a quality score."""
        issues: list[dict[str, str]] = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check for bare except
            if re.match(r"except\s*:", stripped):
                issues.append({
                    "line": str(i),
                    "severity": "warning",
                    "message": "Bare 'except:' — catch specific exceptions instead",
                })

            # Check for print statements (likely debug leftovers)
            if re.match(r"print\s*\(", stripped) and "# noqa" not in stripped:
                issues.append({
                    "line": str(i),
                    "severity": "info",
                    "message": "print() found — consider using logging instead",
                })

            # Check for long lines
            if len(line) > 100:
                issues.append({
                    "line": str(i),
                    "severity": "style",
                    "message": f"Line too long ({len(line)} chars, max 100)",
                })

            # Check for TODO/FIXME/HACK
            if re.search(r"#\s*(TODO|FIXME|HACK|XXX)", stripped, re.IGNORECASE):
                issues.append({
                    "line": str(i),
                    "severity": "info",
                    "message": "TODO/FIXME comment found",
                })

            # Check for mutable default arguments
            if re.search(r"def\s+\w+\(.*=\s*(\[\]|\{\})\s*[,)]", stripped):
                issues.append({
                    "line": str(i),
                    "severity": "warning",
                    "message": "Mutable default argument — use None instead",
                })

        # Check if module/class/function has docstring (very simple check)
        if lines and not any(
            '"""' in line or "'''" in line for line in lines[:5]
        ):
            issues.append({
                "line": "1",
                "severity": "style",
                "message": "No module-level docstring found",
            })

        # Calculate a simple quality score
        severity_weights = {"warning": 3, "info": 1, "style": 1}
        penalty = sum(severity_weights.get(i["severity"], 1) for i in issues)
        score = max(0.0, min(10.0, 10.0 - penalty * 0.5))

        return {
            "issues": issues,
            "issue_count": len(issues),
            "score": round(score, 1),
            "language": language,
            "lines_analyzed": len(lines),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Code Review Agent")
    parser.add_argument("--host", default="localhost", help="Hub host")
    parser.add_argument("--port", type=int, default=8765, help="Hub port")
    args = parser.parse_args()

    agent = CodeReviewAgent()
    print(f"Starting Code Review Agent, connecting to ws://{args.host}:{args.port}")
    agent.run(hub_host=args.host, hub_port=args.port)


if __name__ == "__main__":
    main()
