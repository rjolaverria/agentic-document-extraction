from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from agentic_document_extraction.agents.extraction_agent import (
    ExtractionAgent,
    ExtractionAgentState,
)
from agentic_document_extraction.agents.verifier import (
    QualityMetrics,
    VerificationReport,
    VerificationStatus,
)
from agentic_document_extraction.utils.exceptions import DocumentProcessingError


def _agent_result(data: dict[str, Any]) -> dict[str, Any]:
    """Build a fake agent.invoke() return value."""
    msg = MagicMock()
    msg.content = json.dumps(data)
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    return {"messages": [msg], "structured_response": data}


def _passing_verification() -> VerificationReport:
    return VerificationReport(
        status=VerificationStatus.PASSED,
        metrics=QualityMetrics(
            overall_confidence=0.95,
            schema_coverage=1.0,
            required_field_coverage=1.0,
            optional_field_coverage=1.0,
            completeness_score=1.0,
            consistency_score=1.0,
            min_field_confidence=0.9,
            fields_with_low_confidence=0,
            total_fields=2,
            extracted_fields=2,
        ),
        issues=[],
        total_tokens=0,
    )


@patch("agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent")
@patch("agentic_document_extraction.agents.extraction_agent.create_agent")
@patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
def test_text_only_documents_skip_tool_usage_guidance(
    _mock_llm_cls: MagicMock,
    mock_create: MagicMock,
    mock_verifier: MagicMock,
    schema_info_basic,
    format_info_text,
) -> None:
    """Text-only docs: no regions passed, prompt tells agent to skip tools."""
    mock_create.return_value.invoke.return_value = _agent_result({"name": "Alice"})
    mock_verifier.return_value.verify.return_value = _passing_verification()
    mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

    agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
    agent.extract(
        text="Name: Alice",
        schema_info=schema_info_basic,
        format_info=format_info_text,
        layout_regions=[],
    )

    payload = mock_create.return_value.invoke.call_args[0][0]
    assert payload["regions"] == []

    create_kwargs = mock_create.call_args.kwargs
    prompt = create_kwargs["system_prompt"]
    assert "When NOT to use tools" in prompt
    assert "| region_id | type | page | confidence |" not in prompt


@patch("agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent")
@patch("agentic_document_extraction.agents.extraction_agent.create_agent")
@patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
def test_table_region_exposed_for_tool_selection(
    _mock_llm_cls: MagicMock,
    mock_create: MagicMock,
    mock_verifier: MagicMock,
    schema_info_basic,
    format_info_visual,
    visual_regions_with_table,
) -> None:
    """Visual docs surface TABLE regions so the agent can pick table tools."""
    mock_create.return_value.invoke.return_value = _agent_result({"name": "Bob"})
    mock_verifier.return_value.verify.return_value = _passing_verification()
    mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

    agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
    agent.extract(
        text="Name: Bob",
        schema_info=schema_info_basic,
        format_info=format_info_visual,
        layout_regions=visual_regions_with_table,
    )

    payload = mock_create.return_value.invoke.call_args[0][0]
    assert payload["regions"] == visual_regions_with_table

    create_kwargs = mock_create.call_args.kwargs
    prompt = create_kwargs["system_prompt"]
    assert "table-1" in prompt
    assert "| table |" in prompt

    tools = create_kwargs["tools"]
    tool_names = {tool.name for tool in tools}
    assert "analyze_table_agent" in tool_names


@patch("agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent")
@patch("agentic_document_extraction.agents.extraction_agent.create_agent")
@patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
def test_mixed_regions_all_passed_to_agent_state(
    _mock_llm_cls: MagicMock,
    mock_create: MagicMock,
    mock_verifier: MagicMock,
    schema_info_basic,
    format_info_visual,
    mixed_regions,
) -> None:
    """Agent receives full mixed region list so it can pick the right tool per type."""
    mock_create.return_value.invoke.return_value = _agent_result({"name": "Eve"})
    mock_verifier.return_value.verify.return_value = _passing_verification()
    mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

    agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
    agent.extract(
        text="Report with multiple visual elements",
        schema_info=schema_info_basic,
        format_info=format_info_visual,
        layout_regions=mixed_regions,
    )

    payload = mock_create.return_value.invoke.call_args[0][0]
    assert payload["regions"] == mixed_regions

    prompt = mock_create.call_args.kwargs["system_prompt"]
    for region in mixed_regions:
        assert region.region_id in prompt
        assert region.region_type.value in prompt

    assert mock_create.call_args.kwargs["state_schema"] is ExtractionAgentState


@patch("agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent")
@patch("agentic_document_extraction.agents.extraction_agent.create_agent")
@patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
def test_all_tool_names_registered_and_non_null(
    _mock_llm_cls: MagicMock,
    mock_create: MagicMock,
    mock_verifier: MagicMock,
    schema_info_basic,
    format_info_text,
) -> None:
    """All 9 tools should be present with explicit names for autonomous selection."""
    mock_create.return_value.invoke.return_value = _agent_result({"name": "Alice"})
    mock_verifier.return_value.verify.return_value = _passing_verification()
    mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

    agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
    agent.extract(
        text="Name: Alice",
        schema_info=schema_info_basic,
        format_info=format_info_text,
    )

    tools = mock_create.call_args.kwargs["tools"]
    tool_names = {tool.name for tool in tools}
    expected = {
        "analyze_chart_agent",
        "analyze_diagram_agent",
        "analyze_form_agent",
        "analyze_handwriting_agent",
        "analyze_image_agent",
        "analyze_logo_agent",
        "analyze_math_agent",
        "analyze_signature_agent",
        "analyze_table_agent",
    }
    assert tool_names == expected


@patch("agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent")
@patch("agentic_document_extraction.agents.extraction_agent.create_agent")
@patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
def test_response_format_title_sanitized(
    _mock_llm_cls: MagicMock,
    mock_create: MagicMock,
    mock_verifier: MagicMock,
    format_info_text,
) -> None:
    """Schema titles with spaces/punctuation are sanitized for response_format."""
    schema = {
        "title": "Invoice Data v1.0",
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    schema_info = MagicMock()
    schema_info.schema = schema
    schema_info.all_fields = []

    mock_create.return_value.invoke.return_value = _agent_result({"name": "Alice"})
    mock_verifier.return_value.verify.return_value = _passing_verification()
    mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

    agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
    agent.extract(
        text="Name: Alice",
        schema_info=schema_info,
        format_info=format_info_text,
    )

    response_format = mock_create.call_args.kwargs["response_format"]
    assert response_format["title"] == "Invoice_Data_v1_0"


@patch("agentic_document_extraction.agents.extraction_agent.QualityVerificationAgent")
@patch("agentic_document_extraction.agents.extraction_agent.create_agent")
@patch("agentic_document_extraction.agents.extraction_agent.ChatOpenAI")
def test_tool_failure_raises_processing_error(
    _mock_llm_cls: MagicMock,
    mock_create: MagicMock,
    mock_verifier: MagicMock,
    schema_info_basic,
    format_info_visual,
    visual_regions_with_table,
) -> None:
    """Tool/agent failure surfaces as DocumentProcessingError when no fallback exists."""
    mock_create.return_value.invoke.side_effect = ToolException("tool crashed")
    mock_verifier.return_value.get_default_thresholds.return_value = MagicMock()

    agent = ExtractionAgent(api_key="sk-test", max_iterations=1)
    with pytest.raises(DocumentProcessingError):
        agent.extract(
            text="Table data present",
            schema_info=schema_info_basic,
            format_info=format_info_visual,
            layout_regions=visual_regions_with_table,
        )

    create_kwargs = mock_create.call_args.kwargs
    assert create_kwargs["state_schema"] is ExtractionAgentState
    assert any(tool.name == "analyze_table_agent" for tool in create_kwargs["tools"])
