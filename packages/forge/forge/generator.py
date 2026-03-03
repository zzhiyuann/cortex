"""Code generator — produces tool code from specifications using Jinja2 templates.

Transforms a ToolSpec into working Python code for the chosen output type
(MCP tool, CLI tool, Python function, or app module). Also generates
accompanying tests. Includes retry logic and robust error handling for
resilient generation.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError, select_autoescape

from forge.models import (
    GenerationResult,
    ModuleSpec,
    OutputType,
    ToolParam,
    ToolSpec,
)

logger = logging.getLogger(__name__)

# Template directory
_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Retry configuration for LLM / template operations
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 0.5


def _get_jinja_env() -> Environment:
    """Create a Jinja2 environment with the templates directory."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _retry_with_backoff(
    func: Any,
    *args: Any,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    **kwargs: Any,
) -> Any:
    """Execute a function with exponential backoff retry logic.

    Args:
        func: The callable to execute.
        *args: Positional arguments for the callable.
        max_retries: Maximum number of retry attempts.
        base_backoff: Base delay in seconds (doubles each retry).
        **kwargs: Keyword arguments for the callable.

    Returns:
        The return value of the callable.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_backoff * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d attempts failed. Last error: %s", max_retries, e
                )
    raise last_error  # type: ignore[misc]


def _validate_spec(spec: ToolSpec) -> list[str]:
    """Validate a ToolSpec for common issues before generation.

    Returns a list of warning messages (empty if all is well).
    """
    warnings: list[str] = []

    if not spec.name:
        warnings.append("Tool name is empty")
    elif not re.match(r"^[a-z][a-z0-9_]*$", spec.name):
        warnings.append(
            f"Tool name '{spec.name}' is not a valid Python identifier "
            "(must be lowercase, start with a letter, use underscores)"
        )

    if not spec.core_logic or not spec.core_logic.strip():
        warnings.append("Core logic is empty — generated tool will be a no-op")

    if len(spec.name) > 60:
        warnings.append(
            f"Tool name is very long ({len(spec.name)} chars), consider shortening"
        )

    return warnings


def _sanitize_generated_code(code: str) -> str:
    """Clean up generated code for common issues.

    Handles trailing whitespace, missing newlines, and encoding artifacts.
    """
    if not code:
        return code

    # Ensure the code ends with a single newline
    code = code.rstrip() + "\n"

    # Remove any null bytes (can appear from malformed LLM output)
    code = code.replace("\x00", "")

    # Fix common encoding issues
    code = code.replace("\r\n", "\n")

    return code


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

    Raises:
        ValueError: If the description is empty or unparseable.
    """
    if not description or not description.strip():
        raise ValueError("Tool description cannot be empty")

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


def module_spec_from_description(
    description: str,
    answers: dict[str, str] | None = None,
) -> ModuleSpec:
    """Build a ModuleSpec from a user description for generating app modules.

    Used by the /new-app feature to create complete personal app modules
    with data models, services, UI schema, and context providers.

    Args:
        description: Natural language description of the desired module.
        answers: Optional clarification answers.

    Returns:
        A populated ModuleSpec ready for module generation.

    Raises:
        ValueError: If the description is empty.
    """
    if not description or not description.strip():
        raise ValueError("Module description cannot be empty")

    answers = answers or {}
    name = _extract_name(description)
    deps = _extract_dependencies(description, answers)
    params = _extract_params(description, answers)

    # Build module-specific components
    data_model = _build_data_model(name, description, params)
    service = _build_service(name, description, params)
    ui_schema = _build_ui_schema(name, description, params)
    context_provider = _build_context_provider(name, description)

    return ModuleSpec(
        name=name,
        display_name=name.replace("_", " ").title(),
        description=_clean_description(description),
        data_model=data_model,
        service=service,
        ui_schema=ui_schema,
        context_provider=context_provider,
        dependencies=deps,
        params=params,
        examples=_build_examples(name, params),
    )


def generate_tool(
    spec: ToolSpec,
    output_type: OutputType = OutputType.PYTHON,
) -> GenerationResult:
    """Generate tool code from a ToolSpec.

    Includes retry logic for template rendering failures and validates
    the spec before generation.

    Args:
        spec: The tool specification.
        output_type: What kind of tool to generate.

    Returns:
        GenerationResult with the generated source code.
    """
    # Validate the spec first
    warnings = _validate_spec(spec)
    for w in warnings:
        logger.warning("Spec validation: %s", w)

    if not spec.name:
        return GenerationResult(
            success=False,
            error="Cannot generate tool: tool name is empty",
        )

    env = _get_jinja_env()

    template_map = {
        OutputType.MCP: "mcp_tool.py.jinja",
        OutputType.CLI: "cli_tool.py.jinja",
        OutputType.PYTHON: "python_function.py.jinja",
        OutputType.MODULE: "module.py.jinja",
    }

    template_name = template_map.get(output_type)
    if template_name is None:
        return GenerationResult(
            success=False,
            error=f"Unsupported output type: {output_type}",
        )

    def _render() -> GenerationResult:
        try:
            template = env.get_template(template_name)
        except TemplateSyntaxError as e:
            return GenerationResult(
                success=False,
                error=f"Template '{template_name}' has syntax errors: {e}",
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                error=f"Failed to load template '{template_name}': {e}",
            )

        try:
            tool_code = template.render(spec=spec)
            tool_code = _sanitize_generated_code(tool_code)
        except Exception as e:
            return GenerationResult(
                success=False,
                error=(
                    f"Template rendering failed for '{spec.name}' "
                    f"(template: {template_name}): {e}"
                ),
            )

        # Generate tests (module type uses its own test pattern)
        test_code = ""
        try:
            if output_type == OutputType.MODULE:
                test_template = env.get_template("test_module.py.jinja")
                test_code = test_template.render(spec=spec)
            else:
                test_template = env.get_template("test_tool.py.jinja")
                test_code = test_template.render(
                    spec=spec,
                    output_type=output_type.value,
                    module_name=spec.name,
                )
            test_code = _sanitize_generated_code(test_code)
        except Exception as e:
            # Test generation failure is non-fatal — return tool code anyway
            logger.warning("Test generation failed: %s", e)
            test_code = (
                f"# Test generation failed: {e}\n"
                f"# Please write tests manually for {spec.name}\n"
            )

        return GenerationResult(
            success=True,
            tool_code=tool_code,
            test_code=test_code,
            spec=spec,
        )

    # Use retry logic for rendering
    try:
        return _retry_with_backoff(_render, max_retries=_MAX_RETRIES)
    except Exception as e:
        return GenerationResult(
            success=False,
            error=f"Generation failed after {_MAX_RETRIES} attempts: {e}",
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
    if not errors:
        logger.info("No errors to fix, regenerating as-is")
        return generate_tool(spec, output_type)

    fixed_spec = _apply_fixes(spec, errors)
    return generate_tool(fixed_spec, output_type)


# ---------------------------------------------------------------------------
# Module generation helpers
# ---------------------------------------------------------------------------

def _build_data_model(
    name: str,
    description: str,
    params: list[ToolParam],
) -> str:
    """Build a Pydantic data model for the module."""
    class_name = name.replace("_", " ").title().replace(" ", "")
    fields = []
    for p in params:
        default = f' = Field(default={p.default})' if not p.required else ""
        fields.append(
            f'    {p.name}: {p.type_hint}{default}  # {p.description}'
        )

    if not fields:
        fields.append('    data: str = Field(default="")  # Primary data field')

    return "\n".join([
        "from __future__ import annotations",
        "",
        "from datetime import datetime, timezone",
        "from typing import Any",
        "",
        "from pydantic import BaseModel, Field",
        "",
        "",
        f"class {class_name}Item(BaseModel):",
        f'    """Data model for {name} module."""',
        "",
        '    id: str = Field(default="", description="Unique identifier")',
        *fields,
        "    created_at: datetime = Field(",
        "        default_factory=lambda: datetime.now(timezone.utc)",
        "    )",
        "    metadata: dict[str, Any] = Field(default_factory=dict)",
        "",
        "",
        f"class {class_name}Config(BaseModel):",
        f'    """Configuration for {name} module."""',
        "",
        '    enabled: bool = Field(default=True, description="Whether the module is active")',
        '    refresh_interval_seconds: int = Field(',
        '        default=300, description="How often to refresh data"',
        "    )",
    ])


def _build_service(
    name: str,
    description: str,
    params: list[ToolParam],
) -> str:
    """Build a service/API layer for the module."""
    class_name = name.replace("_", " ").title().replace(" ", "")
    return "\n".join([
        "from __future__ import annotations",
        "",
        "import logging",
        "from typing import Any",
        "",
        f"from .models import {class_name}Item, {class_name}Config",
        "",
        f"logger = logging.getLogger(__name__)",
        "",
        "",
        f"class {class_name}Service:",
        f'    """Service layer for {name} module.',
        "",
        f"    {_clean_description(description)}",
        '    """',
        "",
        f"    def __init__(self, config: {class_name}Config | None = None) -> None:",
        f"        self.config = config or {class_name}Config()",
        "        self._items: list[" + class_name + "Item] = []",
        "",
        f"    async def create(self, **kwargs: Any) -> {class_name}Item:",
        '        """Create a new item."""',
        f"        item = {class_name}Item(**kwargs)",
        "        self._items.append(item)",
        "        return item",
        "",
        f"    async def get(self, item_id: str) -> {class_name}Item | None:",
        '        """Get an item by ID."""',
        "        for item in self._items:",
        "            if item.id == item_id:",
        "                return item",
        "        return None",
        "",
        f"    async def list_all(self) -> list[{class_name}Item]:",
        '        """List all items."""',
        "        return list(self._items)",
        "",
        "    async def delete(self, item_id: str) -> bool:",
        '        """Delete an item by ID."""',
        "        for i, item in enumerate(self._items):",
        "            if item.id == item_id:",
        "                self._items.pop(i)",
        "                return True",
        "        return False",
        "",
        f"    async def update(self, item_id: str, **kwargs: Any) -> {class_name}Item | None:",
        '        """Update an item by ID."""',
        "        for item in self._items:",
        "            if item.id == item_id:",
        "                for key, value in kwargs.items():",
        "                    if hasattr(item, key):",
        "                        setattr(item, key, value)",
        "                return item",
        "        return None",
    ])


def _build_ui_schema(
    name: str,
    description: str,
    params: list[ToolParam],
) -> dict[str, Any]:
    """Build a JSON UI schema for the module's frontend."""
    class_name = name.replace("_", " ").title().replace(" ", "")

    # Build field definitions for the form
    form_fields = []
    for p in params:
        field_type = "text"
        if p.type_hint == "int":
            field_type = "number"
        elif p.type_hint == "bool":
            field_type = "toggle"
        elif "path" in p.name:
            field_type = "file"
        elif "url" in p.name:
            field_type = "url"

        form_fields.append({
            "name": p.name,
            "label": p.name.replace("_", " ").title(),
            "type": field_type,
            "required": p.required,
            "description": p.description,
        })

    if not form_fields:
        form_fields.append({
            "name": "data",
            "label": "Data",
            "type": "text",
            "required": False,
            "description": "Primary data input",
        })

    return {
        "module": name,
        "displayName": name.replace("_", " ").title(),
        "description": _clean_description(description),
        "version": "1.0.0",
        "views": {
            "list": {
                "type": "table",
                "columns": ["id", "created_at"] + [p.name for p in params[:3]],
                "actions": ["create", "edit", "delete"],
            },
            "detail": {
                "type": "form",
                "fields": form_fields,
            },
            "create": {
                "type": "form",
                "fields": form_fields,
                "submitLabel": f"Create {class_name}",
            },
        },
        "contextBus": {
            "publishes": [f"{name}.updated", f"{name}.created"],
            "subscribes": ["personal_context.refresh"],
        },
    }


def _build_context_provider(name: str, description: str) -> str:
    """Build a PersonalContext bus integration interface."""
    class_name = name.replace("_", " ").title().replace(" ", "")
    return "\n".join([
        "from __future__ import annotations",
        "",
        "import logging",
        "from typing import Any, Protocol",
        "",
        f"from .models import {class_name}Config",
        f"from .service import {class_name}Service",
        "",
        f"logger = logging.getLogger(__name__)",
        "",
        "",
        "class ContextBus(Protocol):",
        '    """Protocol for the PersonalContext bus."""',
        "",
        "    async def publish(self, event: str, data: Any) -> None: ...",
        "    async def subscribe(self, event: str, handler: Any) -> None: ...",
        "",
        "",
        f"class {class_name}ContextProvider:",
        f'    """Context provider for {name} module.',
        "",
        "    Integrates with the PersonalContext bus to publish and consume",
        "    events for cross-module communication.",
        '    """',
        "",
        "    def __init__(",
        "        self,",
        f"        service: {class_name}Service,",
        "        bus: ContextBus,",
        "    ) -> None:",
        "        self.service = service",
        "        self.bus = bus",
        "",
        "    async def initialize(self) -> None:",
        '        """Register event handlers with the context bus."""',
        '        await self.bus.subscribe("personal_context.refresh", self._on_refresh)',
        f'        logger.info("{class_name} context provider initialized")',
        "",
        "    async def _on_refresh(self, data: Any) -> None:",
        '        """Handle context refresh events."""',
        "        items = await self.service.list_all()",
        f'        await self.bus.publish("{name}.updated", ' + "{",
        f'            "count": len(items),',
        f'            "items": [item.model_dump() for item in items],',
        "        })",
        "",
        "    async def notify_created(self, item_data: dict) -> None:",
        '        """Publish a creation event."""',
        f'        await self.bus.publish("{name}.created", item_data)',
    ])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_name(description: str) -> str:
    """Extract a reasonable function name from the description.

    Handles edge cases like empty strings, special characters, and
    overly long descriptions.
    """
    if not description or not description.strip():
        return "my_tool"

    desc_lower = description.lower().strip()

    # Remove special characters that could break names
    desc_lower = re.sub(r"[^a-z0-9\s_]", " ", desc_lower)
    desc_lower = re.sub(r"\s+", " ", desc_lower).strip()

    if not desc_lower:
        return "my_tool"

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
        return candidate if candidate else "my_tool"

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
    - SyntaxError -> attempt to clean up malformed code
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
                # Only add fix if the variable isn't a parameter
                param_names = {p.name for p in spec.params}
                if var_name not in param_names:
                    logic = f"{var_name} = None  # Auto-fixed: was undefined\n" + logic

        # File not found in tests -> make file operations conditional
        if "filenotfounderror" in error_lower:
            if "Path(" in logic and "exists()" not in logic:
                logic = logic.replace(
                    "Path(input_path)",
                    'Path(input_path)\nif not file_path.exists():\n    raise ValueError(f"File not found: {input_path}")',
                )

        # Type errors — attempt to add type conversion
        if "typeerror" in error_lower:
            m = re.search(r"expected (\w+), got (\w+)", error_lower)
            if m:
                expected = m.group(1)
                logger.info("Type mismatch detected: expected %s", expected)

        # Syntax errors in generated code
        if "syntaxerror" in error_lower:
            # Try to clean up common syntax issues
            logic = logic.replace("  \n", "\n")  # Trailing whitespace
            logic = re.sub(r"\n{3,}", "\n\n", logic)  # Excessive blank lines

    spec.core_logic = logic
    return spec
