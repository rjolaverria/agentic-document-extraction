"""Tests for the ExtractionAgent."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentic_document_extraction.agents.extraction_agent import (
    _TOOL_INSTRUCTIONS_VISUAL,
    ExtractionAgent,
    ExtractionAgentState,
)
from agentic_document_extraction.agents.refiner import AgenticLoopResult
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
            agent = ExtractionAgent()
            assert agent.api_key == "sk-test"
            assert agent.model == "gpt-4o"
            assert agent.temperature == 0.0
            assert agent.max_tokens == 4096

    def test_custom_params(self) -> None:
        agent = ExtractionAgent(
            api_key="sk-custom",
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=2048,
        )
        assert agent.api_key == "sk-custom"
        assert agent.model == "gpt-4o-mini"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 2048


class TestExtractionAgentState:
    def test_state_has_regions_field(self) -> None:
        assert "regions" in ExtractionAgentState.__annotations__


class TestSystemPrompt:
    def test_text_only_prompt_no_tools(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        prompt = agent._build_system_prompt(
            "Hello world text", schema_info, regions=[], has_tools=False
        )
        assert "Hello world text" in prompt
        assert "Target JSON Schema" in prompt
        assert _TOOL_INSTRUCTIONS_VISUAL not in prompt
        assert "Document Regions" not in prompt

    def test_visual_prompt_with_regions_and_tools(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        regions = _make_regions(chart=True, table=True)
        prompt = agent._build_system_prompt(
            "OCR text here", schema_info, regions=regions, has_tools=True
        )
        assert "OCR text here" in prompt
        assert "chart-1" in prompt
        assert "table-1" in prompt
        assert "analyze_chart" in prompt or "analyze_table" in prompt
        assert "Document Regions" in prompt

    def test_long_text_truncated(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        long_text = "x" * 20000
        prompt = agent._build_system_prompt(
            long_text, schema_info, regions=[], has_tools=False
        )
        assert "[text truncated]" in prompt

    def test_schema_included(self) -> None:
        agent = ExtractionAgent(api_key="sk-test")
        schema_info = _make_schema_info()
        prompt = agent._build_system_prompt(
            "text", schema_info, regions=[], has_tools=False
        )
        assert '"name"' in prompt
        assert '"age"' in prompt


class TestToolRegistration:
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_no_tools_for_text_documents(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
    ) -> None:
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Alice"}
        )
        agent = ExtractionAgent(api_key="sk-test")
        agent.extract(
            text="Name: Alice",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(ProcessingCategory.TEXT_BASED),
        )
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        assert tools == []

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_tools_registered_for_visual_with_regions(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
    ) -> None:
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Bob"}
        )
        agent = ExtractionAgent(api_key="sk-test")
        regions = _make_regions(chart=True, table=True)
        agent.extract(
            text="OCR text",
            schema_info=_make_schema_info(),
            format_info=_make_visual_format_info(),
            layout_regions=regions,
        )
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        assert len(tools) == 2  # analyze_chart + analyze_table

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_no_tools_visual_without_visual_regions(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
    ) -> None:
        """Visual doc but only text regions - no tools needed."""
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Carol"}
        )
        agent = ExtractionAgent(api_key="sk-test")
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
        assert len(tools) == 0


class TestExtract:
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_text_extraction_returns_result(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
    ) -> None:
        extracted = {"name": "Alice", "age": 30}
        mock_create.return_value.invoke.return_value = _mock_agent_result(extracted)

        agent = ExtractionAgent(api_key="sk-test")
        result = agent.extract(
            text="Name: Alice, Age: 30",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        assert isinstance(result, AgenticLoopResult)
        assert result.final_result.extracted_data == extracted
        assert result.converged is True
        assert result.iterations_completed == 1
        assert result.total_tokens == 150  # 100 + 50

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_field_extractions_built(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
    ) -> None:
        extracted = {"name": "Bob", "age": 25}
        mock_create.return_value.invoke.return_value = _mock_agent_result(extracted)

        agent = ExtractionAgent(api_key="sk-test")
        result = agent.extract(
            text="Name: Bob, Age: 25",
            schema_info=_make_schema_info(),
            format_info=_make_format_info(),
        )

        field_paths = [f.field_path for f in result.final_result.field_extractions]
        assert "name" in field_paths
        assert "age" in field_paths

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_regions_passed_to_agent_invoke(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
    ) -> None:
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Chart Person"}
        )
        agent = ExtractionAgent(api_key="sk-test")
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

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_response_format_is_schema(
        self,
        _mock_llm_cls: MagicMock,  # noqa: ARG002
        mock_create: MagicMock,
    ) -> None:
        mock_create.return_value.invoke.return_value = _mock_agent_result(
            {"name": "Test"}
        )
        schema_info = _make_schema_info()
        agent = ExtractionAgent(api_key="sk-test")
        agent.extract(
            text="text",
            schema_info=schema_info,
            format_info=_make_format_info(),
        )
        call_kwargs = mock_create.call_args
        rf = call_kwargs.kwargs.get("response_format") or call_kwargs[1].get(
            "response_format"
        )
        assert rf == schema_info.schema


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


class TestWrapResult:
    def test_wrap_produces_valid_agentic_loop_result(self) -> None:
        from agentic_document_extraction.services.extraction.text_extraction import (
            ExtractionResult,
            FieldExtraction,
        )

        extraction = ExtractionResult(
            extracted_data={"name": "Test"},
            field_extractions=[
                FieldExtraction(field_path="name", value="Test", confidence=0.9)
            ],
            model_used="gpt-4o",
            total_tokens=100,
        )
        format_info = _make_format_info()
        wrapped = ExtractionAgent._wrap_result(extraction, format_info, 1.5, 100)

        assert isinstance(wrapped, AgenticLoopResult)
        assert wrapped.converged is True
        assert wrapped.iterations_completed == 1
        assert wrapped.total_tokens == 100
        assert wrapped.plan.extraction_strategy == "tool_agent"
        assert wrapped.final_verification.status.value == "passed"
        assert wrapped.loop_metadata["agent_type"] == "tool_agent"


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


class TestExtractErrorHandling:
    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_invoke_failure_raises_document_processing_error(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        mock_create.return_value.invoke.side_effect = RuntimeError("LLM timeout")
        agent = ExtractionAgent(api_key="sk-test")
        with pytest.raises(DocumentProcessingError, match="Extraction agent failed"):
            agent.extract(
                text="some text",
                schema_info=_make_schema_info(),
                format_info=_make_format_info(),
            )

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_invoke_failure_logs_error(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_create.return_value.invoke.side_effect = ValueError("bad input")
        agent = ExtractionAgent(api_key="sk-test")
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

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_text_only_document_no_tools(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        """Text format, no regions → no tools, result shape valid."""
        data = {"title": "Invoice #42", "amount": 99.50, "items": ["widget"]}
        mock_create.return_value.invoke.return_value = _mock_agent_result(data)

        agent = ExtractionAgent(api_key="sk-test")
        result = agent.extract(
            text="Invoice #42\nAmount: $99.50\nItems: widget",
            schema_info=_make_rich_schema_info(),
            format_info=_make_format_info(ProcessingCategory.TEXT_BASED),
        )

        assert isinstance(result, AgenticLoopResult)
        assert result.final_result.extracted_data == data
        assert result.loop_metadata["agent_type"] == "tool_agent"
        # No tools registered
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        assert tools == []

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_visual_document_with_chart_region(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        """Visual format + PICTURE region → tools registered, regions passed."""
        data = {"title": "Sales Report", "amount": 1500.0}
        mock_create.return_value.invoke.return_value = _mock_agent_result(data)

        agent = ExtractionAgent(api_key="sk-test")
        regions = _make_regions(chart=True)
        result = agent.extract(
            text="Sales Report\nQ1: $1500",
            schema_info=_make_rich_schema_info(),
            format_info=_make_visual_format_info(),
            layout_regions=regions,
        )

        assert result.final_result.extracted_data == data
        # Tools should be registered
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        assert len(tools) >= 1
        # Regions passed to invoke
        invoke_args = mock_create.return_value.invoke.call_args[0][0]
        assert invoke_args["regions"] is regions

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_visual_document_with_table_region(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        """Visual format + TABLE region → tools registered."""
        data = {"title": "Quarterly Data", "items": ["row1", "row2"]}
        mock_create.return_value.invoke.return_value = _mock_agent_result(data)

        agent = ExtractionAgent(api_key="sk-test")
        regions = _make_regions(table=True)
        result = agent.extract(
            text="Quarterly Data\nrow1 row2",
            schema_info=_make_rich_schema_info(),
            format_info=_make_visual_format_info(),
            layout_regions=regions,
        )

        assert result.final_result.extracted_data == data
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        assert len(tools) >= 1

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_visual_document_mixed_regions(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        """Visual format with TEXT + PICTURE + TABLE regions."""
        data = {"title": "Annual Report", "amount": 5000.0, "items": ["a", "b"]}
        mock_create.return_value.invoke.return_value = _mock_agent_result(data)

        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.TEXT,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="text-1",
            ),
            LayoutRegion(
                region_type=RegionType.PICTURE,
                bbox=bbox,
                confidence=0.95,
                page_number=1,
                region_id="chart-1",
                region_image=RegionImage(image=None, base64="b64"),
            ),
            LayoutRegion(
                region_type=RegionType.TABLE,
                bbox=bbox,
                confidence=0.92,
                page_number=2,
                region_id="table-1",
                region_image=RegionImage(image=None, base64="b64"),
            ),
        ]

        agent = ExtractionAgent(api_key="sk-test")
        result = agent.extract(
            text="Annual Report\nSection A\nSection B",
            schema_info=_make_rich_schema_info(),
            format_info=_make_visual_format_info(),
            layout_regions=regions,
        )

        assert result.final_result.extracted_data == data
        # Both chart and table tools registered
        call_kwargs = mock_create.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools", [])
        assert len(tools) == 2
        # All regions included in invoke
        invoke_args = mock_create.return_value.invoke.call_args[0][0]
        assert len(invoke_args["regions"]) == 3

    @patch("agentic_document_extraction.agents.extraction_agent.create_agent")
    @patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
    def test_empty_extraction_result(
        self,
        _mock_llm_cls: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        """Agent returns no parseable output → graceful empty result."""
        msg = MagicMock()
        msg.content = "I cannot extract anything meaningful."
        msg.usage_metadata = {"input_tokens": 80, "output_tokens": 20}
        mock_create.return_value.invoke.return_value = {"messages": [msg]}

        agent = ExtractionAgent(api_key="sk-test")
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
