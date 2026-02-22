"""Tests for forge.generator."""

from forge.generator import generate_tool, spec_from_description
from forge.models import OutputType, ToolSpec, ToolParam


class TestSpecFromDescription:
    def test_csv_to_json(self):
        spec = spec_from_description("convert CSV to JSON", output_type=OutputType.PYTHON)
        assert spec.name
        assert "csv" in spec.name.lower() or "convert" in spec.name.lower()
        assert len(spec.params) > 0

    def test_url_fetcher(self):
        spec = spec_from_description("fetch content from a URL")
        assert any(p.name == "url" for p in spec.params)

    def test_text_tool(self):
        spec = spec_from_description("count words in a text string")
        assert len(spec.params) > 0

    def test_filter_tool(self):
        spec = spec_from_description("filter lines in a file by query")
        param_names = {p.name for p in spec.params}
        assert "query" in param_names

    def test_description_cleaned(self):
        spec = spec_from_description("a tool that does something cool")
        assert not spec.description.lower().startswith("a tool that")

    def test_name_is_snake_case(self):
        spec = spec_from_description("convert CSV files to JSON format")
        assert "_" in spec.name or spec.name.isalpha()
        assert spec.name == spec.name.lower()


class TestGenerateTool:
    def test_python_generation(self):
        spec = ToolSpec(
            name="add_numbers",
            description="Add two numbers together.",
            params=[
                ToolParam(name="a", type_hint="int", description="First number"),
                ToolParam(name="b", type_hint="int", description="Second number"),
            ],
            return_type="int",
            core_logic="return a + b",
        )
        result = generate_tool(spec, OutputType.PYTHON)
        assert result.success
        assert "def add_numbers" in result.tool_code
        assert "return a + b" in result.tool_code
        assert result.test_code  # Tests should be generated

    def test_cli_generation(self):
        spec = ToolSpec(
            name="greet",
            description="Greet a person by name.",
            params=[
                ToolParam(name="name", type_hint="str", description="Person's name"),
            ],
            return_type="str",
            core_logic='return f"Hello, {name}!"',
        )
        result = generate_tool(spec, OutputType.CLI)
        assert result.success
        assert "click" in result.tool_code.lower()
        assert "def main" in result.tool_code

    def test_mcp_generation(self):
        spec = ToolSpec(
            name="echo",
            description="Echo the input back.",
            params=[
                ToolParam(name="message", type_hint="str", description="Message to echo"),
            ],
            return_type="str",
            core_logic="return message",
        )
        result = generate_tool(spec, OutputType.MCP)
        assert result.success
        assert "FastMCP" in result.tool_code
        assert "@mcp.tool()" in result.tool_code

    def test_generation_includes_docstring(self):
        spec = ToolSpec(
            name="test_func",
            description="A test function.",
            params=[ToolParam(name="x", type_hint="str", description="Input")],
            core_logic="return x",
        )
        result = generate_tool(spec, OutputType.PYTHON)
        assert result.success
        assert "A test function." in result.tool_code

    def test_generation_handles_empty_core_logic(self):
        spec = ToolSpec(
            name="noop",
            description="Do nothing.",
            params=[],
            core_logic="return None",
        )
        result = generate_tool(spec, OutputType.PYTHON)
        assert result.success
