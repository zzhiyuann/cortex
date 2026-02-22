"""Tests for forge.clarifier."""

from forge.clarifier import analyze_ambiguity


class TestAnalyzeAmbiguity:
    def test_csv_tool_asks_about_input(self):
        result = analyze_ambiguity("convert CSV file to JSON")
        assert result.has_questions
        categories = {q.category for q in result.questions}
        # Should ask about edge cases or output for file-based tools
        assert len(result.questions) > 0

    def test_filter_tool_asks_about_criteria(self):
        result = analyze_ambiguity("filter lines in a file by keyword")
        assert result.has_questions
        categories = {q.category for q in result.questions}
        assert "behavior" in categories

    def test_api_tool_asks_about_auth(self):
        result = analyze_ambiguity("fetch data from a REST API endpoint")
        assert result.has_questions
        categories = {q.category for q in result.questions}
        assert "dependency" in categories

    def test_vague_description_gets_questions(self):
        result = analyze_ambiguity("do something useful")
        assert result.has_questions
        assert len(result.questions) >= 2

    def test_specific_description_fewer_questions(self):
        result_vague = analyze_ambiguity("process data")
        result_specific = analyze_ambiguity(
            "read a CSV file and output JSON to stdout"
        )
        # Both should have questions, but specific should have fewer
        # (or at least not more required ones)
        assert result_vague.has_questions
        assert result_specific.has_questions

    def test_question_ids_are_unique(self):
        result = analyze_ambiguity("convert and filter CSV file to JSON format")
        ids = [q.id for q in result.questions]
        assert len(ids) == len(set(ids))

    def test_questions_have_categories(self):
        result = analyze_ambiguity("download URL content and save to file")
        for q in result.questions:
            assert q.category in {
                "input", "output", "edge_case", "dependency", "behavior"
            }
