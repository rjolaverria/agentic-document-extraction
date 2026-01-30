"""Tests for the ExtractionAgent with lightweight verification loop."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentic_document_extraction.agents.extraction_agent import (
    ExtractionAgent,
    ExtractionAgentState,
)
from agentic_document_extraction.agents.refiner import AgenticLoopResult
from agentic_document_extraction.agents.verifier import (
    IssueSeverity,
    IssueType,
    QualityMetrics,
    VerificationIssue,
    VerificationReport,
    VerificationStatus,
)
from agentic_document_extraction.models import (
    FormatFamily,
    FormatInfo,
    ProcessingCategory,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionImage,
    RegionType,
)
from agentic_document_extraction.services.schema_validator import (
    FieldInfo,
    SchemaInfo,
)
from agentic_document_extraction.utils.exceptions import DocumentProcessingError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema_info() -> SchemaInfo:
    """Create a minimal SchemaInfo for testing."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person name"},
            "age": {"type": "integer", "description": "Person age"},
        },
        "required": ["name"],
    }
    return SchemaInfo(
        schema=schema,
        required_fields=[FieldInfo("name", "string", required=True, path="name")],
        optional_fields=[FieldInfo("age", "integer", required=False, path="age")],
        schema_type="object",
    )


def _make_format_info(
    category: ProcessingCategory = ProcessingCategory.TEXT_BASED,
) -> FormatInfo:
    return FormatInfo(
        mime_type="text/plain",
        format_family=FormatFamily.PLAIN_TEXT,
        processing_category=category,
        extension=".txt",
    )


def _make_visual_format_info() -> FormatInfo:
    return FormatInfo(
        mime_type="image/png",
        format_family=FormatFamily.IMAGE,
        processing_category=ProcessingCategory.VISUAL,
        extension=".png",
    )


def _make_regions(
    *,
    chart: bool = False,
    table: bool = False,
) -> list[LayoutRegion]:
    regions: list[LayoutRegion] = []
    bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
    if chart:
        regions.append(
            LayoutRegion(
                region_type=RegionType.PICTURE,
                bbox=bbox,
                confidence=0.95,
                page_number=1,
                region_id="chart-1",
                region_image=RegionImage(image=None, base64="fake_b64"),
            )
        )
    if table:
        regions.append(
            LayoutRegion(
                region_type=RegionType.TABLE,
                bbox=bbox,
                confidence=0.92,
                page_number=1,
                region_id="table-1",
                region_image=RegionImage(image=None, base64="fake_b64"),
            )
        )
    return regions


def _mock_agent_result(data: dict[str, Any]) -> dict[str, Any]:
    """Build a fake agent.invoke() return value."""
    msg = MagicMock()
    msg.content = json.dumps(data)
    msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    return {"messages": [msg], "structured_response": data}


def _make_passing_verification() -> VerificationReport:
    """Create a passing verification report."""
    return VerificationReport(
        status=VerificationStatus.PASSED,
        metrics=QualityMetrics(
            overall_confidence=0.9,
            schema_coverage=1.0,
            required_field_coverage=1.0,
            optional_field_coverage=1.0,
            completeness_score=1.0,
            consistency_score=1.0,
            min_field_confidence=0.85,
            fields_with_low_confidence=0,
            total_fields=2,
            extracted_fields=2,
        ),
        issues=[],
        total_tokens=0,
    )


def _make_failing_verification(missing_field: str = "name") -> VerificationReport:
    """Create a failing verification report with missing required field."""
    return VerificationReport(
        status=VerificationStatus.FAILED,
        metrics=QualityMetrics(
            overall_confidence=0.5,
            schema_coverage=0.5,
            required_field_coverage=0.0,
            optional_field_coverage=1.0,
            completeness_score=0.3,
            consistency_score=1.0,
            min_field_confidence=0.85,
            fields_with_low_confidence=0,
            total_fields=2,
            extracted_fields=1,
        ),
        issues=[
            VerificationIssue(
                issue_type=IssueType.MISSING_REQUIRED_FIELD,
                field_path=missing_field,
                message=f"Required field '{missing_field}' is missing",
                severity=IssueSeverity.CRITICAL,
                suggestion=f"Re-extract with focus on finding {missing_field}",
            )
        ],
        total_tokens=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractionAgentInit:
    def test_defaults_from_settings(self) -> None:
        with patch(
            "agentic_document_extraction.agents.extraction_agent.settings"
        ) as mock_settings:
            mock_settings.get_openai_api_key.return_value = "sk-test"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.0
            mock_settings.openai_max_tokens = 4096
            mock_settings.max_refinement_iterations = 3
            agent = ExtractionAgent()
            assert agent.api_key == "sk-test"
            assert agent.model == "gpt-4o"
            assert agent.temperature == 0.0
            assert agent.max_tokens == 4096
            assert agent.max_iterations == 3

    def test_custom_params(self) -> None:
        agent = ExtractionAgent(
            api_key="sk-custom",
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=2048,
            max_iterations=5,
        )
        assert agent.api_key == "sk-custom"
        assert agent.model == "gpt-4o-mini"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 2048
        assert agent.max_iterations == 5

    def test_use_llm_verification_default(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        assert agent.use_llm_verification is False


class TestExtractionAgentState:
    def test_state_has_regions_field(self) -> None:
        assert "regions" in ExtractionAgentState.__annotations__


class TestSystemPrompt:
    """Tests for system prompt building.

    Per Task 0054, tool instructions are always included in the system prompt.
    The agent decides when to use tools based on document content and regions.
    """

    def test_text_only_prompt_includes_tool_instructions(self) -> None:
        """Tool instructions are always included - agent decides usage."""
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        prompt = agent._build_system_prompt("Hello world text", schema_info, regions=[])
        assert "Hello world text" in prompt
        assert "Target JSON Schema" in prompt
        # Tool instructions always included (Task 0054)
        assert "Visual Analysis Tools" in prompt
        assert "When to use tools" in prompt
        assert "When NOT to use tools" in prompt
        # No regions section/table when no regions (section starts with ## Document Regions)
        assert "## Document Regions" not in prompt
        assert "| region_id |" not in prompt  # No table header

    def test_visual_prompt_with_regions(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        regions = _make_regions(chart=True, table=True)
        prompt = agent._build_system_prompt(
            "OCR text here", schema_info, regions=regions
        )
        assert "OCR text here" in prompt
        assert "chart-1" in prompt
        assert "table-1" in prompt
        assert "analyze_chart" in prompt
        assert "analyze_table" in prompt
        assert "Document Regions" in prompt
        # Tool instructions always included
        assert "Visual Analysis Tools" in prompt

    def test_long_text_truncated(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        long_text = "x" * 20000
        prompt = agent._build_system_prompt(long_text, schema_info, regions=[])
        assert "[text truncated]" in prompt

    def test_schema_included(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        prompt = agent._build_system_prompt("text", schema_info, regions=[])
        assert '"name"' in prompt
        assert '"age"' in prompt

    def test_tool_instructions_include_all_tools(self) -> None:
        """All 9 tools are documented in the instructions."""
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        prompt = agent._build_system_prompt("text", schema_info, regions=[])
        # All 9 tools should be mentioned
        assert "analyze_chart" in prompt
        assert "analyze_table" in prompt
        assert "analyze_diagram" in prompt
        assert "analyze_form" in prompt
        assert "analyze_handwriting" in prompt
        assert "analyze_image" in prompt
        assert "analyze_logo" in prompt
        assert "analyze_math" in prompt
        assert "analyze_signature" in prompt

    def test_tool_instructions_include_skip_guidance(self) -> None:
        """Instructions guide agent to skip tools when not needed."""
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        prompt = agent._build_system_prompt("text", schema_info, regions=[])
        # Guidance on when NOT to use tools
        assert "When NOT to use tools" in prompt
        assert "OCR text" in prompt
        assert "skip tool calls" in prompt


class TestRefinementPrompt:
    """Tests for refinement prompt building.

    Per Task 0054, tool instructions are always included in refinement prompts.
    """

    def test_refinement_prompt_includes_issues(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        prev_extraction = {"age": 30}
        issues = [
            "[CRITICAL] name: Required field 'name' is missing",
        ]
        prompt = agent._build_refinement_prompt(
            "text", schema_info, [], prev_extraction, issues
        )
        assert "Issues to Fix" in prompt
        assert "name" in prompt
        assert "Previous Extraction" in prompt
        assert '"age": 30' in prompt
        # Tool instructions always included (Task 0054)
        assert "Visual Analysis Tools" in prompt

    def test_refinement_prompt_no_issues(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        prompt = agent._build_refinement_prompt(
            "text", schema_info, [], {"name": "Test"}, []
        )
        assert "None" in prompt  # No issues
        # Tool instructions always included
        assert "Visual Analysis Tools" in prompt


class TestToolRegistration:
    """Tests for tool registration - all 9 tools always provided.

    Per Task 0051, the agent should always have all tools available and
    let the LLM decide which tools to use based on context.
    """

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_all_tools_registered_for_text_documents(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Text documents get all 9 tools - agent decides whether to use them."""
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Alice"}
        )
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        agent.extract(
            text="Name: Alice",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(ProcessingCategory.TEXT_BASED),
        )
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        # All 9 tools always provided
        assert len(tools) == 9

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_all_tools_registered_for_visual_with_regions(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Visual documents with regions get all 9 tools."""
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Bob"}
        )
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        regions = _make_regions(chart=True, table=True)
        agent.extract(
            text="OCR text",
            schema_info=_make_schema_info(),
            format_info=_make_visual_format_info(),
            layout_regions=regions,
        )
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        # All 9 visual analysis tools registered
        assert len(tools) == 9

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_all_tools_registered_visual_without_visual_regions(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Visual doc with only text regions still gets all 9 tools."""
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Carol"}
        )
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        text_region = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=RegionBoundingBox(x0=0, y0=0, x1=100, y1=100),
            confidence=0.9,
            page_number=1,
            region_id="text-1",
        )
        agent.extract(
            text="OCR text",
            schema_info=_make_schema_info(),
            format_info=_make_visual_format_info(),
            layout_regions=[text_region],
        )
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        # All 9 tools always provided
        assert len(tools) == 9


class TestExtract:
    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_text_extraction_returns_result(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        extracted = {"name": "Alice", "age": 30}
        mock_create.return_value.invoke.return_value = _mock_agent_result(extracted)
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        result = agent.extract(
            text="Name: Alice, Age: 30",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        assert isinstance(result, AgenticLoopResult)
        assert result.final_result.extracted_data == extracted
        assert result.converged is True
        assert result.iterations_completed == 1

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_field_extractions_built(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        extracted = {"name": "Bob", "age": 25}
        mock_create.return_value.invoke.return_value = _mock_agent_result(extracted)
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        result = agent.extract(
            text="Name: Bob, Age: 25",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        field_paths = [f.field_path for f in result.final_result.field_extractions]
        assert "name" in field_paths
        assert "age" in field_paths

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_regions_passed_to_agent_invoke(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Chart Person"}
        )
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        regions = _make_regions(chart=True)
        agent.extract(
            text="OCR text",
            schema_info=_make_schema_info(),
            format_info=_make_visual_format_info(),
            layout_regions=regions,
        )
        invoke_args = mock_create.return_value.invoke.call_args[0][0]
        assert "regions" in invoke_args
        assert invoke_args["regions"] is regions


class TestVerificationLoop:
    """Tests for the lightweight verification and refinement loop."""

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_single_pass_when_quality_passes(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """First extraction passes quality - no refinement needed."""
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Alice", "age": 30}
        )
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=3)
        result = agent.extract(
            text="Name: Alice, Age: 30",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        assert result.iterations_completed == 1
        assert result.converged is True
        assert mock_create.return_value.invoke.call_count == 1

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_multiple_iterations_until_quality_passes(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """First extraction fails, second passes."""
        # First call returns incomplete data, second returns complete
        mock_create.return_value.invoke.side_effect = [
            _mock_agent_result({"age": 30}),  # Missing name
            _mock_agent_result({"name": "Alice", "age": 30}),  # Complete
        ]
        # First verification fails, second passes
        mock_verifier.return_value.verify.side_effect = [
            _make_failing_verification("name"),
            _make_passing_verification(),
        ]
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=3)
        result = agent.extract(
            text="Name: Alice, Age: 30",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        assert result.iterations_completed == 2
        assert result.converged is True
        assert result.best_iteration == 2
        assert mock_create.return_value.invoke.call_count == 2

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_max_iterations_reached(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """All iterations fail - returns best result."""
        mock_create.return_value.invoke.return_value = _mock_agent_result({"age": 30})
        mock_verifier.return_value.verify.return_value = _make_failing_verification(
            "name"
        )
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=2)
        result = agent.extract(
            text="Age: 30",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        assert result.iterations_completed == 2
        assert result.converged is False
        assert mock_create.return_value.invoke.call_count == 2

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_best_result_tracked_across_iterations(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Best result is returned even if later iterations are worse."""
        # First result is better (has name), second is worse (regression)
        mock_create.return_value.invoke.side_effect = [
            _mock_agent_result({"name": "Alice"}),  # Has name
            _mock_agent_result({"age": 30}),  # Lost name
        ]

        # Create verification reports with different scores
        good_verification = VerificationReport(
            status=VerificationStatus.NEEDS_IMPROVEMENT,
            metrics=QualityMetrics(
                overall_confidence=0.7,
                schema_coverage=0.5,
                required_field_coverage=1.0,  # Has required field
                optional_field_coverage=0.0,
                completeness_score=0.7,
                consistency_score=1.0,
                min_field_confidence=0.85,
                fields_with_low_confidence=0,
                total_fields=2,
                extracted_fields=1,
            ),
            total_tokens=0,
        )
        bad_verification = VerificationReport(
            status=VerificationStatus.FAILED,
            metrics=QualityMetrics(
                overall_confidence=0.5,
                schema_coverage=0.5,
                required_field_coverage=0.0,  # Missing required field
                optional_field_coverage=1.0,
                completeness_score=0.3,
                consistency_score=1.0,
                min_field_confidence=0.85,
                fields_with_low_confidence=0,
                total_fields=2,
                extracted_fields=1,
            ),
            issues=[
                VerificationIssue(
                    issue_type=IssueType.MISSING_REQUIRED_FIELD,
                    field_path="name",
                    message="Required field 'name' is missing",
                    severity=IssueSeverity.CRITICAL,
                )
            ],
            total_tokens=0,
        )

        mock_verifier.return_value.verify.side_effect = [
            good_verification,
            bad_verification,
        ]
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=2)
        result = agent.extract(
            text="Name: Alice",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        # Should return the better first result
        assert result.best_iteration == 1
        assert result.final_result.extracted_data == {"name": "Alice"}

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_iteration_history_tracked(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Iteration history captures metrics from each iteration."""
        mock_create.return_value.invoke.side_effect = [
            _mock_agent_result({"age": 30}),
            _mock_agent_result({"name": "Alice", "age": 30}),
        ]
        mock_verifier.return_value.verify.side_effect = [
            _make_failing_verification("name"),
            _make_passing_verification(),
        ]
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=3)
        result = agent.extract(
            text="Name: Alice, Age: 30",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        assert len(result.iteration_history) == 2
        # First iteration failed
        assert result.iteration_history[0].iteration_number == 1
        assert (
            result.iteration_history[0].verification_status == VerificationStatus.FAILED
        )
        # Second iteration passed
        assert result.iteration_history[1].iteration_number == 2
        assert (
            result.iteration_history[1].verification_status == VerificationStatus.PASSED
        )


class TestParseAgentResult:
    def test_structured_response_preferred(self) -> None:
        result = {"messages": [], "structured_response": {"name": "Alice"}}
        parsed = ExtractionAgent._parse_agent_result(result, _make_schema_info())
        assert parsed == {"name": "Alice"}

    def test_fallback_to_message_content(self) -> None:
        msg = MagicMock()
        msg.content = '{"name": "Bob"}'
        result = {"messages": [msg]}
        parsed = ExtractionAgent._parse_agent_result(result, _make_schema_info())
        assert parsed == {"name": "Bob"}

    def test_fallback_embedded_json(self) -> None:
        msg = MagicMock()
        msg.content = 'Here is the result: {"name": "Carol"}\nDone.'
        result = {"messages": [msg]}
        parsed = ExtractionAgent._parse_agent_result(result, _make_schema_info())
        assert parsed == {"name": "Carol"}

    def test_no_parseable_output(self) -> None:
        msg = MagicMock()
        msg.content = "I cannot extract anything."
        result = {"messages": [msg]}
        parsed = ExtractionAgent._parse_agent_result(result, _make_schema_info())
        assert parsed == {}


class TestTokenUsage:
    def test_sums_across_messages(self) -> None:
        msg1 = MagicMock()
        msg1.usage_metadata = {"input_tokens": 50, "output_tokens": 20}
        msg2 = MagicMock()
        msg2.usage_metadata = {"input_tokens": 30, "output_tokens": 10}
        result = {"messages": [msg1, msg2]}
        total, prompt, completion = ExtractionAgent._extract_token_usage(result)
        assert total == 110
        assert prompt == 80
        assert completion == 30

    def test_handles_missing_usage(self) -> None:
        msg = MagicMock(spec=[])  # no usage_metadata attribute
        result = {"messages": [msg]}
        total, prompt, completion = ExtractionAgent._extract_token_usage(result)
        assert total == 0


class TestFormatIssuesForRefinement:
    def test_formats_issues_with_severity(self) -> None:
        verification = VerificationReport(
            status=VerificationStatus.FAILED,
            metrics=QualityMetrics(
                overall_confidence=0.5,
                schema_coverage=0.5,
                required_field_coverage=0.0,
                optional_field_coverage=1.0,
                completeness_score=0.3,
                consistency_score=1.0,
                min_field_confidence=0.85,
                fields_with_low_confidence=0,
                total_fields=2,
                extracted_fields=1,
            ),
            issues=[
                VerificationIssue(
                    issue_type=IssueType.MISSING_REQUIRED_FIELD,
                    field_path="name",
                    message="Required field 'name' is missing",
                    severity=IssueSeverity.CRITICAL,
                    suggestion="Re-extract with focus on finding name",
                )
            ],
            total_tokens=0,
        )

        issues = ExtractionAgent._format_issues_for_refinement(verification)

        assert len(issues) == 1
        assert "[CRITICAL]" in issues[0]
        assert "name" in issues[0]
        assert "Re-extract" in issues[0]

    def test_sorts_by_severity(self) -> None:
        verification = VerificationReport(
            status=VerificationStatus.FAILED,
            metrics=QualityMetrics(
                overall_confidence=0.5,
                schema_coverage=0.5,
                required_field_coverage=0.5,
                optional_field_coverage=0.5,
                completeness_score=0.5,
                consistency_score=1.0,
                min_field_confidence=0.85,
                fields_with_low_confidence=0,
                total_fields=2,
                extracted_fields=1,
            ),
            issues=[
                VerificationIssue(
                    issue_type=IssueType.LOW_CONFIDENCE,
                    field_path="age",
                    message="Low confidence",
                    severity=IssueSeverity.MEDIUM,
                ),
                VerificationIssue(
                    issue_type=IssueType.MISSING_REQUIRED_FIELD,
                    field_path="name",
                    message="Missing required",
                    severity=IssueSeverity.CRITICAL,
                ),
            ],
            total_tokens=0,
        )

        issues = ExtractionAgent._format_issues_for_refinement(verification)

        assert len(issues) == 2
        # Critical should come first
        assert "[CRITICAL]" in issues[0]
        assert "[MEDIUM]" in issues[1]

    def test_limits_to_10_issues(self) -> None:
        verification = VerificationReport(
            status=VerificationStatus.FAILED,
            metrics=QualityMetrics(
                overall_confidence=0.5,
                schema_coverage=0.5,
                required_field_coverage=0.5,
                optional_field_coverage=0.5,
                completeness_score=0.5,
                consistency_score=1.0,
                min_field_confidence=0.85,
                fields_with_low_confidence=0,
                total_fields=15,
                extracted_fields=5,
            ),
            issues=[
                VerificationIssue(
                    issue_type=IssueType.LOW_CONFIDENCE,
                    field_path=f"field_{i}",
                    message=f"Issue {i}",
                    severity=IssueSeverity.MEDIUM,
                )
                for i in range(15)
            ],
            total_tokens=0,
        )

        issues = ExtractionAgent._format_issues_for_refinement(verification)
        assert len(issues) == 10


class TestCalculateResultScore:
    def test_passing_verification_gets_bonus(self) -> None:
        passing = _make_passing_verification()
        failing = _make_failing_verification()

        passing_score = ExtractionAgent._calculate_result_score(passing)
        failing_score = ExtractionAgent._calculate_result_score(failing)

        assert passing_score > failing_score

    def test_critical_issues_penalized(self) -> None:
        no_issues = _make_passing_verification()

        with_issues = VerificationReport(
            status=VerificationStatus.NEEDS_IMPROVEMENT,
            metrics=no_issues.metrics,
            issues=[
                VerificationIssue(
                    issue_type=IssueType.MISSING_REQUIRED_FIELD,
                    field_path="name",
                    message="Missing",
                    severity=IssueSeverity.CRITICAL,
                )
            ],
            total_tokens=0,
        )

        score_no_issues = ExtractionAgent._calculate_result_score(no_issues)
        score_with_issues = ExtractionAgent._calculate_result_score(with_issues)

        assert score_no_issues > score_with_issues


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


class TestExtractErrorHandling:
    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_invoke_failure_raises_document_processing_error(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
        mock_verifier: MagicMock,  # noqa: ARG002
    ) -> None:
        mock_create.return_value.invoke.side_effect = RuntimeError("LLM timeout")
        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        with pytest.raises(DocumentProcessingError, match="Extraction agent failed"):
            agent.extract(
                text="some text",
                schema_info=_make_schema_info(),
                format_info=_make_format_info(),
            )

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_invoke_failure_returns_best_if_available(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """If we have a result and a later iteration fails, return the result."""
        # First call succeeds, second fails
        mock_create.return_value.invoke.side_effect = [
            _mock_agent_result({"name": "Alice"}),
            RuntimeError("LLM timeout"),
        ]
        mock_verifier.return_value.verify.return_value = _make_failing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=2)
        result = agent.extract(
            text="Name: Alice",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        # Should return the first result
        assert result.final_result.extracted_data == {"name": "Alice"}
        assert result.iterations_completed == 1

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_invoke_failure_logs_error(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
        mock_verifier: MagicMock,  # noqa: ARG002
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_create.return_value.invoke.side_effect = ValueError("bad input")
        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        with (
            caplog.at_level(
                logging.ERROR,
                logger="agentic_document_extraction.agents.extraction_agent",
            ),
            pytest.raises(DocumentProcessingError),
        ):
            agent.extract(
                text="some text",
                schema_info=_make_schema_info(),
                format_info=_make_format_info(),
            )
        assert any("invoke failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration-style tests (mocked LLM, realistic inputs)
# ---------------------------------------------------------------------------


def _make_rich_schema_info() -> SchemaInfo:
    """Schema with multiple field types for integration tests."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "amount": {"type": "number"},
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["title"],
    }
    return SchemaInfo(
        schema=schema,
        required_fields=[
            FieldInfo("title", "string", required=True, path="title"),
        ],
        optional_fields=[
            FieldInfo("amount", "number", required=False, path="amount"),
            FieldInfo("items", "array", required=False, path="items"),
        ],
        schema_type="object",
    )


class TestExtractionAgentIntegration:
    """Integration-level tests exercising full extract() with mocked LLM."""

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_text_only_document_all_tools_registered(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Text format, no regions → all 9 tools registered, agent decides usage."""
        data = {"title": "Invoice #42", "amount": 99.50, "items": ["widget"]}
        mock_create.return_value.invoke.return_value = _mock_agent_result(data)
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        result = agent.extract(
            text="Invoice #42\nAmount: $99.50\nItems: widget",
            schema_info=_make_rich_schema_info(),
            format_info=_make_format_info(ProcessingCategory.TEXT_BASED),
        )

        assert isinstance(result, AgenticLoopResult)
        assert result.final_result.extracted_data == data
        assert result.loop_metadata["agent_type"] == "tool_agent_with_verification"
        # All 9 tools always registered
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        assert len(tools) == 9

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_visual_document_with_chart_region(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Visual format + PICTURE region → tools registered, regions passed."""
        data = {"title": "Sales Report", "amount": 1500.0}
        mock_create.return_value.invoke.return_value = _mock_agent_result(data)
        mock_verifier.return_value.verify.return_value = _make_passing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        regions = _make_regions(chart=True)
        result = agent.extract(
            text="Sales Report\nQ1: $1500",
            schema_info=_make_rich_schema_info(),
            format_info=_make_visual_format_info(),
            layout_regions=regions,
        )

        assert result.final_result.extracted_data == data
        # All 9 tools should be registered
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        assert len(tools) == 9
        # Regions passed to invoke
        invoke_args = mock_create.return_value.invoke.call_args[0][0]
        assert invoke_args["regions"] is regions

    @patch(
        "agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent"
    )
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_empty_extraction_result(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
        mock_verifier: MagicMock,
    ) -> None:
        """Agent returns no parseable output → graceful empty result."""
        msg = MagicMock()
        msg.content = "I cannot extract anything meaningful."
        msg.usage_metadata = {"input_tokens": 80, "output_tokens": 20}
        mock_create.return_value.invoke.return_value = {"messages": [msg]}
        mock_verifier.return_value.verify.return_value = _make_failing_verification()
        mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

        agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
        result = agent.extract(
            text="Garbled text with no useful data",
            schema_info=_make_rich_schema_info(),
            format_info=_make_format_info(),
        )

        assert isinstance(result, AgenticLoopResult)
        assert result.final_result.extracted_data == {}
        # Fields still present, but values are None
        for fe in result.final_result.field_extractions:
            assert fe.value is None
