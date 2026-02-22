"""Clarifier — generates intelligent clarification questions for ambiguous requests.

Analyzes a user's tool description and identifies what additional information
is needed to generate a complete, working tool.
"""

from __future__ import annotations

import re
import uuid

from forge.models import ClarificationQuestion, ClarificationResult


# Keywords that suggest specific categories of ambiguity
_INPUT_INDICATORS = {
    "file", "files", "csv", "json", "xml", "yaml", "text", "data", "input",
    "read", "load", "parse", "import", "from",
}
_OUTPUT_INDICATORS = {
    "output", "save", "write", "export", "to", "convert", "generate", "create",
    "format", "print",
}
_NETWORK_INDICATORS = {
    "api", "http", "url", "fetch", "download", "request", "endpoint", "webhook",
    "scrape", "crawl",
}
_FILTER_INDICATORS = {
    "filter", "search", "find", "match", "select", "where", "query", "grep",
    "exclude", "include",
}
_TRANSFORM_INDICATORS = {
    "convert", "transform", "map", "translate", "encode", "decode", "compress",
    "decompress", "encrypt", "decrypt", "hash",
}


def _words(text: str) -> set[str]:
    """Extract lowercase word set from text."""
    return set(re.findall(r"[a-z_]+", text.lower()))


def analyze_ambiguity(description: str) -> ClarificationResult:
    """Analyze a tool description and generate clarification questions.

    Examines the description for missing or ambiguous elements across
    five categories: input, output, edge cases, dependencies, and behavior.

    Args:
        description: The user's natural language tool description.

    Returns:
        ClarificationResult with generated questions.
    """
    words = _words(description)
    questions: list[ClarificationQuestion] = []

    # --- Input questions ---
    if words & _INPUT_INDICATORS:
        if not _has_input_format(description):
            questions.append(
                ClarificationQuestion(
                    id=_qid(),
                    question="What is the input format? (e.g., file path, string, stdin, JSON object)",
                    category="input",
                    options=["file_path", "string", "stdin", "json_object", "other"],
                )
            )
    else:
        questions.append(
            ClarificationQuestion(
                id=_qid(),
                question="What input does this tool accept? Please describe the input type and format.",
                category="input",
            )
        )

    # --- Output questions ---
    if words & _OUTPUT_INDICATORS:
        if not _has_output_format(description):
            questions.append(
                ClarificationQuestion(
                    id=_qid(),
                    question="What should the output format be? (e.g., print to stdout, save to file, return value)",
                    category="output",
                    options=["stdout", "file", "return_value", "json", "other"],
                )
            )
    else:
        questions.append(
            ClarificationQuestion(
                id=_qid(),
                question="What output should this tool produce? Describe the expected result.",
                category="output",
            )
        )

    # --- Filter/behavior questions ---
    if words & _FILTER_INDICATORS:
        questions.append(
            ClarificationQuestion(
                id=_qid(),
                question="What criteria should be used for filtering/searching? Please be specific.",
                category="behavior",
            )
        )

    # --- Network dependency questions ---
    if words & _NETWORK_INDICATORS:
        questions.append(
            ClarificationQuestion(
                id=_qid(),
                question="Does this tool need to authenticate with any API? If so, how? (API key, OAuth, etc.)",
                category="dependency",
                required=False,
                default="no authentication",
            )
        )

    # --- Edge case questions ---
    if words & (_INPUT_INDICATORS | {"file", "files"}):
        questions.append(
            ClarificationQuestion(
                id=_qid(),
                question="How should the tool handle errors? (e.g., missing file, invalid format, empty input)",
                category="edge_case",
                required=False,
                default="Print error message and exit with non-zero code",
            )
        )

    # --- Transformation questions ---
    if words & _TRANSFORM_INDICATORS and len(words & _TRANSFORM_INDICATORS) > 0:
        if "convert" in words or "transform" in words:
            questions.append(
                ClarificationQuestion(
                    id=_qid(),
                    question="Can you provide an example of the input and the expected output after transformation?",
                    category="behavior",
                )
            )

    # Always ask about dependencies if not obvious
    if not (words & _NETWORK_INDICATORS):
        questions.append(
            ClarificationQuestion(
                id=_qid(),
                question="Does this tool need any external dependencies beyond the Python standard library?",
                category="dependency",
                required=False,
                default="standard library only",
            )
        )

    return ClarificationResult(
        questions=questions,
        has_questions=len(questions) > 0,
    )


def _has_input_format(description: str) -> bool:
    """Check if description already specifies the input format."""
    patterns = [
        r"takes?\s+(a\s+)?file",
        r"from\s+(a\s+)?(csv|json|xml|yaml|text|string)",
        r"input\s*:\s*\w+",
        r"reads?\s+(a\s+)?\w+\s+file",
        r"accepts?\s+(a\s+)?\w+",
    ]
    return any(re.search(p, description, re.IGNORECASE) for p in patterns)


def _has_output_format(description: str) -> bool:
    """Check if description already specifies the output format."""
    patterns = [
        r"(outputs?|returns?|produces?|saves?|writes?)\s+(to\s+)?(a\s+)?(csv|json|xml|yaml|stdout|file|string)",
        r"output\s*:\s*\w+",
        r"to\s+(a\s+)?(csv|json|xml|yaml|file)",
    ]
    return any(re.search(p, description, re.IGNORECASE) for p in patterns)


def _qid() -> str:
    """Generate a short question ID."""
    return f"q_{uuid.uuid4().hex[:8]}"
