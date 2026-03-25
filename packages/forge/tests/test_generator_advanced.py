"""Tests for advanced generator features: module generation, retry logic, validation."""

import pytest

from forge.generator import (
    _sanitize_generated_code,
    _validate_spec,
    generate_tool,
    module_spec_from_description,
    spec_from_description,
)
from forge.models import (
    ModuleSpec,
    OutputType,
    ToolParam,
    ToolSpec,
)


class TestSpecValidation:
    """Test the spec validation logic."""

    def test_valid_spec(self):
        spec = ToolSpec(
            name="my_tool",
            description="A test tool.",
            core_logic="return 'hello'",
        )
        warnings = _validate_spec(spec)
        assert len(warnings) == 0

    def test_empty_name_warning(self):
        spec = ToolSpec(name="", description="test", core_logic="pass")
        warnings = _validate_spec(spec)
        assert any("empty" in w.lower() for w in warnings)

    def test_invalid_name_warning(self):
        spec = ToolSpec(name="123invalid", description="test", core_logic="pass")
        warnings = _validate_spec(spec)
        assert any("not a valid" in w.lower() for w in warnings)

    def test_empty_core_logic_warning(self):
        spec = ToolSpec(name="my_tool", description="test", core_logic="")
        warnings = _validate_spec(spec)
        assert any("empty" in w.lower() for w in warnings)

    def test_long_name_warning(self):
        spec = ToolSpec(name="a" * 61, description="test", core_logic="pass")
        warnings = _validate_spec(spec)
        assert any("long" in w.lower() for w in warnings)


class TestSanitizeCode:
    """Test the code sanitization logic."""

    def test_trailing_whitespace(self):
        code = "def foo():\n    pass\n\n\n"
        result = _sanitize_generated_code(code)
        assert result.endswith("\n")
        assert not result.endswith("\n\n")

    def test_null_bytes(self):
        code = "def foo():\x00\n    pass"
        result = _sanitize_generated_code(code)
        assert "\x00" not in result

    def test_crlf_normalization(self):
        code = "def foo():\r\n    pass\r\n"
        result = _sanitize_generated_code(code)
        assert "\r" not in result

    def test_empty_string(self):
        assert _sanitize_generated_code("") == ""


class TestSpecFromDescriptionEdgeCases:
    """Test edge cases in spec_from_description."""

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="empty"):
            spec_from_description("")

    def test_whitespace_description_raises(self):
        with pytest.raises(ValueError, match="empty"):
            spec_from_description("   ")

    def test_special_characters_in_description(self):
        spec = spec_from_description("convert @#$% data!!! to JSON???")
        assert spec.name  # Should still produce a valid name
        assert "_" in spec.name or spec.name.isalpha()

    def test_very_long_description(self):
        long_desc = "convert data " * 100
        spec = spec_from_description(long_desc)
        assert len(spec.name) <= 40


class TestModuleSpecFromDescription:
    """Test module spec generation."""

    def test_basic_module(self):
        spec = module_spec_from_description("a personal expense tracker")
        assert isinstance(spec, ModuleSpec)
        assert spec.name
        assert spec.description

    def test_module_has_all_components(self):
        spec = module_spec_from_description("track daily health metrics")
        assert spec.data_model  # Should have generated data model code
        assert spec.service  # Should have generated service code
        assert spec.ui_schema  # Should have generated UI schema
        assert spec.context_provider  # Should have context provider

    def test_module_ui_schema_structure(self):
        spec = module_spec_from_description("manage personal bookmarks")
        schema = spec.ui_schema
        assert "module" in schema
        assert "views" in schema
        assert "contextBus" in schema
        assert "publishes" in schema["contextBus"]
        assert "subscribes" in schema["contextBus"]

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="empty"):
            module_spec_from_description("")


class TestGenerateModule:
    """Test module code generation via templates."""

    def test_module_generation(self):
        spec = ToolSpec(
            name="expense_tracker",
            display_name="Expense Tracker",
            description="Track personal expenses.",
            params=[
                ToolParam(name="amount", type_hint="float", description="Expense amount"),
                ToolParam(name="category", type_hint="str", description="Expense category"),
            ],
            return_type="dict",
            core_logic="return {'amount': amount, 'category': category}",
        )
        result = generate_tool(spec, OutputType.MODULE)
        assert result.success
        assert "ExpenseTracker" in result.tool_code  # Pascal case class name
        assert "Service" in result.tool_code
        assert "ContextProvider" in result.tool_code
        assert "UI_SCHEMA" in result.tool_code

    def test_module_generates_tests(self):
        spec = ToolSpec(
            name="habit_log",
            display_name="Habit Log",
            description="Log daily habits.",
            params=[
                ToolParam(name="habit", type_hint="str", description="Habit name"),
            ],
            core_logic="return habit",
        )
        result = generate_tool(spec, OutputType.MODULE)
        assert result.success
        assert result.test_code
        assert "TestHabitLog" in result.test_code


class TestGenerateToolWithEmptyName:
    """Test that generation fails gracefully with empty tool name."""

    def test_empty_name_fails(self):
        spec = ToolSpec(name="", description="test", core_logic="pass")
        result = generate_tool(spec, OutputType.PYTHON)
        assert not result.success
        assert "empty" in result.error.lower()
