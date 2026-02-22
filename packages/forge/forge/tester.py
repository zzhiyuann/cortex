"""Test runner — generates and executes tests for forge-created tools.

Writes tool code and test code to a temporary directory, runs pytest,
and returns structured results with diagnostics for the iteration loop.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from forge.models import TestResult


def run_tests(
    tool_code: str,
    test_code: str,
    tool_name: str,
    dependencies: list[str] | None = None,
) -> TestResult:
    """Run tests for generated tool code in an isolated temporary directory.

    Creates a temp directory, writes the tool module and test file,
    installs any required dependencies, then runs pytest.

    Args:
        tool_code: The generated tool source code.
        test_code: The generated test source code.
        tool_name: The module name for the tool (used as filename).
        dependencies: Optional list of pip packages to install first.

    Returns:
        TestResult with pass/fail status, output, and error details.
    """
    with tempfile.TemporaryDirectory(prefix="forge_test_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Write tool module
        tool_file = tmp_path / f"{tool_name}.py"
        tool_file.write_text(tool_code, encoding="utf-8")

        # Write test file
        test_file = tmp_path / f"test_{tool_name}.py"
        test_file.write_text(test_code, encoding="utf-8")

        # Install dependencies if any
        if dependencies:
            _install_deps(dependencies)

        # Run pytest
        return _run_pytest(tmp_path, test_file)


def _install_deps(dependencies: list[str]) -> None:
    """Install pip dependencies (best-effort, non-fatal)."""
    for dep in dependencies:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", dep, "--quiet"],
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Non-fatal: tests will reveal the missing import


def _run_pytest(work_dir: Path, test_file: Path) -> TestResult:
    """Run pytest on the test file and parse results.

    Args:
        work_dir: Directory containing the tool module.
        test_file: Path to the test file.

    Returns:
        Structured TestResult.
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--no-header",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(work_dir),
            env=_test_env(work_dir),
        )

        output = result.stdout + result.stderr
        passed = result.returncode == 0
        errors = _extract_errors(output) if not passed else []
        total, failures = _parse_counts(output)

        return TestResult(
            passed=passed,
            total=total,
            failures=failures,
            errors=errors,
            output=output,
            test_code=test_file.read_text(encoding="utf-8"),
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            errors=["Test execution timed out after 120 seconds"],
            output="TIMEOUT",
        )
    except Exception as e:
        return TestResult(
            passed=False,
            errors=[f"Failed to run tests: {e}"],
            output=str(e),
        )


def _test_env(work_dir: Path) -> dict[str, str]:
    """Build environment variables for the test subprocess."""
    import os

    env = os.environ.copy()
    # Ensure the work directory is on PYTHONPATH so imports work
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{work_dir}:{existing}" if existing else str(work_dir)
    return env


def _extract_errors(output: str) -> list[str]:
    """Extract error messages from pytest output."""
    errors: list[str] = []
    lines = output.splitlines()

    for i, line in enumerate(lines):
        # Capture FAILED lines
        if "FAILED" in line:
            errors.append(line.strip())
        # Capture exception lines
        if "Error:" in line or "Exception:" in line:
            errors.append(line.strip())
        # Capture assertion errors with context
        if "AssertionError" in line or "assert " in line and "FAILED" not in line:
            # Get surrounding context
            context = lines[max(0, i - 2) : i + 1]
            errors.append(" | ".join(l.strip() for l in context if l.strip()))

    return errors if errors else ["Tests failed (see output for details)"]


def _parse_counts(output: str) -> tuple[int, int]:
    """Parse total and failure counts from pytest output.

    Returns:
        Tuple of (total_tests, failed_tests).
    """
    import re

    # pytest summary line: "X passed, Y failed" or "X passed" or "X failed"
    total = 0
    failures = 0

    m = re.search(r"(\d+)\s+passed", output)
    if m:
        total += int(m.group(1))

    m = re.search(r"(\d+)\s+failed", output)
    if m:
        failures = int(m.group(1))
        total += failures

    m = re.search(r"(\d+)\s+error", output)
    if m:
        err_count = int(m.group(1))
        failures += err_count
        total += err_count

    return total, failures
