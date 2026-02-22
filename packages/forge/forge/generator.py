"""Code generator — produces tool code from specifications using Jinja2 templates.

Transforms a ToolSpec into working Python code for the chosen output type
(MCP tool, CLI tool, or Python function). Also generates accompanying tests.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from forge.models import (
    GenerationResult,
    OutputType,
    ToolParam,
    ToolSpec,
)

# Template directory
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_jinja_env() -> Environment:
    """Create a Jinja2 environment with the templates directory."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def spec_from_description(
    description: str,
    answers: dict[str, str] | None = None,
    output_type: OutputType = OutputType.PYTHON,
) -> ToolSpec:
    """Build a ToolSpec from a user description and optional clarification answers.

    This is the template-based "intelligence" — it parses the description to
    extract tool name, parameters, core logic pattern, and dependencies.

    Args:
        description: The user's natural language tool description.
        answers: Optional dict of clarification question_id -> answer.
        output_type: The desired output type.

    Returns:
        A populated ToolSpec ready for code generation.
    """
    answers = answers or {}
    name = _extract_name(description)
    params = _extract_params(description, answers)
    core_logic = _build_core_logic(description, params, answers)
    deps = _extract_dependencies(description, answers)
    return_type = _infer_return_type(description, answers)

    return ToolSpec(
        name=name,
        display_name=name.replace("_", " ").title(),
        description=_clean_description(description),
        params=params,
        return_type=return_type,
        return_description=f"Result of {name}",
        dependencies=deps,
        core_logic=core_logic,
        error_handling="Raises ValueError on invalid input",
        examples=_build_examples(name, params),
    )


def generate_tool(
    spec: ToolSpec,
    output_type: OutputType = OutputType.PYTHON,
) -> GenerationResult:
    """Generate tool code from a ToolSpec.

    Args:
        spec: The tool specification.
        output_type: What kind of tool to generate.

    Returns:
        GenerationResult with the generated source code.
    """
    env = _get_jinja_env()

    template_map = {
        OutputType.MCP: "mcp_tool.py.jinja",
        OutputType.CLI: "cli_tool.py.jinja",
        OutputType.PYTHON: "python_function.py.jinja",
    }

    try:
        template = env.get_template(template_map[output_type])
        tool_code = template.render(spec=spec)

        # Generate tests
        test_template = env.get_template("test_tool.py.jinja")
        module_name = spec.name
        test_code = test_template.render(
            spec=spec,
            output_type=output_type.value,
            module_name=module_name,
        )

        return GenerationResult(
            success=True,
            tool_code=tool_code,
            test_code=test_code,
            spec=spec,
        )
    except Exception as e:
        return GenerationResult(
            success=False,
            error=f"Template rendering failed: {e}",
        )


def regenerate_with_fixes(
    spec: ToolSpec,
    output_type: OutputType,
    errors: list[str],
    previous_code: str,
) -> GenerationResult:
    """Attempt to fix generation issues based on test errors.

    Analyzes errors from a failed test run and adjusts the spec or code
    to fix common issues, then regenerates.

    Args:
        spec: The current tool specification.
        output_type: The output type.
        errors: List of error messages from test failures.
        previous_code: The code that failed tests.

    Returns:
        A new GenerationResult with fixes applied.
    """
    fixed_spec = _apply_fixes(spec, errors)
    return generate_tool(fixed_spec, output_type)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_name(description: str) -> str:
    """Extract a reasonable function name from the description."""
    # Common patterns: "a tool that <verb>s ...", "convert X to Y", etc.
    desc_lower = description.lower().strip()

    # "converts X to Y" -> "convert_x_to_y"
    m = re.match(r"(?:a\s+tool\s+(?:that|to)\s+)?(\w+)\w*\s+(.+)", desc_lower)
    if m:
        verb = m.group(1)
        rest = m.group(2)
        # Take first few meaningful words
        words = re.findall(r"[a-z]+", rest)[:4]
        candidate = "_".join([verb] + words)
        candidate = re.sub(r"_+", "_", candidate).strip("_")
        if len(candidate) > 40:
            candidate = candidate[:40].rstrip("_")
        return candidate

    # Fallback: first few words
    words = re.findall(r"[a-z]+", desc_lower)[:3]
    return "_".join(words) if words else "my_tool"


def _extract_params(
    description: str,
    answers: dict[str, str],
) -> list[ToolParam]:
    """Extract tool parameters from the description and answers."""
    params: list[ToolParam] = []
    desc_lower = description.lower()

    # File-based tools
    if any(w in desc_lower for w in ("file", "csv", "json", "xml", "yaml", "path")):
        params.append(
            ToolParam(
                name="input_path",
                type_hint="str",
                description="Path to the input file",
                required=True,
            )
        )

    # URL-based tools
    if any(w in desc_lower for w in ("url", "http", "api", "fetch", "download")):
        params.append(
            ToolParam(
                name="url",
                type_hint="str",
                description="URL to fetch or interact with",
                required=True,
            )
        )

    # String input tools
    if any(w in desc_lower for w in ("text", "string", "content", "message")):
        if not params:  # Don't add if we already have file/url
            params.append(
                ToolParam(
                    name="text",
                    type_hint="str",
                    description="Input text to process",
                    required=True,
                )
            )

    # Filter/query tools get a query param
    if any(w in desc_lower for w in ("filter", "search", "find", "query", "grep")):
        params.append(
            ToolParam(
                name="query",
                type_hint="str",
                description="Filter/search query",
                required=True,
            )
        )

    # Output file for converters
    if any(w in desc_lower for w in ("convert", "export", "save", "write")):
        if any(w in desc_lower for w in ("file", "csv", "json", "xml", "yaml")):
            params.append(
                ToolParam(
                    name="output_path",
                    type_hint="str",
                    description="Path to write output file",
                    required=False,
                    default="None",
                )
            )

    # If no params detected, add a generic input
    if not params:
        params.append(
            ToolParam(
                name="input_data",
                type_hint="str",
                description="Input data to process",
                required=True,
            )
        )

    return params


def _build_core_logic(
    description: str,
    params: list[ToolParam],
    answers: dict[str, str],
) -> str:
    """Build the core logic code for the tool.

    Uses pattern matching to select appropriate logic templates
    based on what the tool needs to do.
    """
    desc_lower = description.lower()
    lines: list[str] = []

    # CSV to JSON conversion
    if "csv" in desc_lower and "json" in desc_lower:
        return _logic_csv_to_json(params, desc_lower)

    # JSON to CSV
    if "json" in desc_lower and "csv" in desc_lower:
        return _logic_json_to_csv(params, desc_lower)

    # File reading tools
    if any(w in desc_lower for w in ("read", "load", "parse")) and "file" in desc_lower:
        return _logic_read_file(params)

    # URL fetching
    if any(w in desc_lower for w in ("fetch", "download", "request")):
        return _logic_fetch_url(params)

    # Text transformation
    if any(w in desc_lower for w in ("convert", "transform", "encode", "decode")):
        return _logic_text_transform(params, desc_lower)

    # Search/filter
    if any(w in desc_lower for w in ("filter", "search", "find", "grep")):
        return _logic_filter(params)

    # Generic fallback
    return _logic_generic(params, description)


def _logic_csv_to_json(params: list[ToolParam], desc: str) -> str:
    """Generate CSV-to-JSON conversion logic."""
    lines = [
        "import csv",
        "import json",
        "from pathlib import Path",
        "",
        "file_path = Path(input_path)",
        "if not file_path.exists():",
        '    raise ValueError(f"File not found: {input_path}")',
        "",
        "rows = []",
        'with open(file_path, newline="", encoding="utf-8") as f:',
        "    reader = csv.DictReader(f)",
        "    for row in reader:",
        "        rows.append(dict(row))",
        "",
    ]

    # Check if filtering is mentioned
    if "filter" in desc:
        lines.extend([
            "# Apply filter if query is provided",
            'if query:',
            '    rows = [r for r in rows if any(query.lower() in str(v).lower() for v in r.values())]',
            "",
        ])

    lines.extend([
        "result = json.dumps(rows, indent=2, ensure_ascii=False)",
        "",
        "if output_path:",
        '    Path(output_path).write_text(result, encoding="utf-8")',
        '    return f"Wrote {len(rows)} records to {output_path}"',
        "",
        "return result",
    ])
    return "\n".join(lines)


def _logic_json_to_csv(params: list[ToolParam], desc: str) -> str:
    """Generate JSON-to-CSV conversion logic."""
    return "\n".join([
        "import csv",
        "import json",
        "import io",
        "from pathlib import Path",
        "",
        "file_path = Path(input_path)",
        "if not file_path.exists():",
        '    raise ValueError(f"File not found: {input_path}")',
        "",
        'data = json.loads(file_path.read_text(encoding="utf-8"))',
        "if not isinstance(data, list):",
        '    raise ValueError("JSON must contain an array of objects")',
        "",
        "if not data:",
        '    return "Empty dataset"',
        "",
        "fieldnames = list(data[0].keys())",
        "output = io.StringIO()",
        "writer = csv.DictWriter(output, fieldnames=fieldnames)",
        "writer.writeheader()",
        "writer.writerows(data)",
        "result = output.getvalue()",
        "",
        "if output_path:",
        '    Path(output_path).write_text(result, encoding="utf-8")',
        '    return f"Wrote {len(data)} records to {output_path}"',
        "",
        "return result",
    ])


def _logic_read_file(params: list[ToolParam]) -> str:
    """Generate file reading logic."""
    return "\n".join([
        "from pathlib import Path",
        "",
        "file_path = Path(input_path)",
        "if not file_path.exists():",
        '    raise ValueError(f"File not found: {input_path}")',
        "",
        'content = file_path.read_text(encoding="utf-8")',
        "return content",
    ])


def _logic_fetch_url(params: list[ToolParam]) -> str:
    """Generate URL fetching logic."""
    return "\n".join([
        "import urllib.request",
        "import urllib.error",
        "",
        "try:",
        "    with urllib.request.urlopen(url) as response:",
        '        content = response.read().decode("utf-8")',
        "    return content",
        "except urllib.error.URLError as e:",
        '    raise ValueError(f"Failed to fetch URL: {e}") from e',
    ])


def _logic_text_transform(params: list[ToolParam], desc: str) -> str:
    """Generate text transformation logic."""
    if "base64" in desc:
        return "\n".join([
            "import base64",
            "",
            'if "encode" in __import__("sys").argv[0].lower():',
            '    return base64.b64encode(text.encode("utf-8")).decode("ascii")',
            "else:",
            '    return base64.b64decode(text.encode("ascii")).decode("utf-8")',
        ])

    # Generic transform
    return "\n".join([
        "# Transform the input",
        "result = text  # Replace with actual transformation logic",
        "return result",
    ])


def _logic_filter(params: list[ToolParam]) -> str:
    """Generate filter/search logic."""
    return "\n".join([
        "from pathlib import Path",
        "",
        "file_path = Path(input_path)",
        "if not file_path.exists():",
        '    raise ValueError(f"File not found: {input_path}")',
        "",
        'lines = file_path.read_text(encoding="utf-8").splitlines()',
        "matches = [line for line in lines if query.lower() in line.lower()]",
        "",
        'return "\\n".join(matches) if matches else "No matches found."',
    ])


def _logic_generic(params: list[ToolParam], description: str) -> str:
    """Generate generic tool logic as a starting point."""
    param_name = params[0].name if params else "input_data"
    return "\n".join([
        f"# Process the input",
        f"result = str({param_name})",
        f"return result",
    ])


def _extract_dependencies(description: str, answers: dict[str, str]) -> list[str]:
    """Determine required pip packages from the description."""
    desc_lower = description.lower()
    deps: list[str] = []

    dep_map = {
        "requests": ["api", "http", "rest", "webhook"],
        "beautifulsoup4": ["scrape", "html", "crawl", "parse html"],
        "pandas": ["dataframe", "pandas"],
        "pyyaml": ["yaml"],
        "toml": ["toml"],
        "pillow": ["image", "photo", "picture", "resize"],
        "httpx": ["async http", "httpx"],
    }

    for pkg, keywords in dep_map.items():
        if any(kw in desc_lower for kw in keywords):
            deps.append(pkg)

    # Check answers for dependency info
    for answer in answers.values():
        answer_lower = answer.lower()
        for pkg, keywords in dep_map.items():
            if any(kw in answer_lower for kw in keywords):
                if pkg not in deps:
                    deps.append(pkg)

    return deps


def _infer_return_type(description: str, answers: dict[str, str]) -> str:
    """Infer the return type from the description."""
    desc_lower = description.lower()

    if any(w in desc_lower for w in ("count", "number", "total")):
        return "int"
    if any(w in desc_lower for w in ("true", "false", "check", "validate", "exists")):
        return "bool"
    if any(w in desc_lower for w in ("list", "array", "items")):
        return "list"
    if any(w in desc_lower for w in ("dict", "mapping", "object")):
        return "dict"

    return "str"


def _clean_description(description: str) -> str:
    """Clean up the description for use in docstrings."""
    desc = description.strip()
    # Remove common prefixes
    for prefix in ("a tool that ", "a tool to ", "tool to ", "i need ", "i want "):
        if desc.lower().startswith(prefix):
            desc = desc[len(prefix):]
    # Capitalize first letter
    if desc:
        desc = desc[0].upper() + desc[1:]
    # Ensure it ends with a period
    if not desc.endswith("."):
        desc += "."
    return desc


def _build_examples(name: str, params: list[ToolParam]) -> list[dict]:
    """Build example input/output pairs for test generation."""
    examples = []
    # Build a simple example call
    arg_parts = []
    for p in params:
        if p.type_hint == "str":
            if "path" in p.name:
                arg_parts.append(f'{p.name}="test.txt"')
            elif "url" in p.name:
                arg_parts.append(f'{p.name}="https://example.com"')
            else:
                arg_parts.append(f'{p.name}="test"')
        elif p.type_hint == "int":
            arg_parts.append(f"{p.name}=1")
        elif p.type_hint == "bool":
            arg_parts.append(f"{p.name}=True")
        else:
            arg_parts.append(f'{p.name}="test"')

    examples.append({
        "description": "basic invocation",
        "args": ", ".join(arg_parts),
    })
    return examples


def _apply_fixes(spec: ToolSpec, errors: list[str]) -> ToolSpec:
    """Apply fixes to a ToolSpec based on test error messages.

    Common fixes:
    - ImportError -> add missing import to core_logic
    - NameError -> fix variable references
    - TypeError -> fix parameter types
    - FileNotFoundError -> add file existence checks
    """
    logic = spec.core_logic

    for error in errors:
        error_lower = error.lower()

        # Missing import
        if "importerror" in error_lower or "no module named" in error_lower:
            m = re.search(r"no module named '(\w+)'", error_lower)
            if m:
                module = m.group(1)
                if module not in spec.dependencies:
                    spec.dependencies.append(module)

        # Name not defined
        if "nameerror" in error_lower:
            m = re.search(r"name '(\w+)' is not defined", error_lower)
            if m:
                var_name = m.group(1)
                logic = f"{var_name} = None  # Auto-fixed: was undefined\n" + logic

        # File not found in tests -> make file operations conditional
        if "filenotfounderror" in error_lower:
            if "Path(" in logic and "exists()" not in logic:
                logic = logic.replace(
                    "Path(input_path)",
                    'Path(input_path)\nif not file_path.exists():\n    raise ValueError(f"File not found: {input_path}")',
                )

    spec.core_logic = logic
    return spec
