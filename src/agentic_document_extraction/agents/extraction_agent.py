"""Tool-based extraction agent for document processing."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI

from agentic_document_extraction.agents.tools import AnalyzeChart, AnalyzeTable
from agentic_document_extraction.config import settings
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.layout_detector import LayoutRegion
from agentic_document_extraction.services.reading_order_detector import OrderedRegion
from agentic_document_extraction.services.schema_validator import SchemaInfo
from agentic_document_extraction.utils.agent_helpers import (
    build_agent,
    get_message_content,
    get_usage_metadata,
    invoke_agent,
)

logger = logging.getLogger(__name__)


class ExtractionAgentError(Exception):
    """Raised when tool-based extraction fails."""

    def __init__(
        self,
        message: str,
        error_type: str = "extraction_agent_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class ExtractionAgent:
    """Tool-based extraction agent that combines OCR and layout metadata."""

    SYSTEM_PROMPT_TEMPLATE = """You are an expert document extraction agent. Your task is to extract information
from a document according to the provided JSON schema.

DOCUMENT OCR TEXT (in reading order):
{ordered_text}

LAYOUT REGIONS DETECTED:
{regions}

TARGET SCHEMA:
{json_schema}

AVAILABLE TOOLS:
- AnalyzeChart: Use for chart/graph regions when you need structured data extraction
- AnalyzeTable: Use for complex table regions when OCR text is insufficient

INSTRUCTIONS:
1. Review the OCR text and layout regions
2. Identify which fields in the schema can be filled from OCR text alone
3. For chart/graph regions, use AnalyzeChart tool to extract structured data
4. For complex tables, use AnalyzeTable tool when needed
5. Synthesize all information into final JSON matching the schema
6. Return valid JSON only

Extract the information now:"""

    MAX_OCR_CHARS = 12000

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm: Any | None = None,
    ) -> None:
        """Initialize the extraction agent.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Model name to use. Defaults to settings.openai_model.
            temperature: Sampling temperature. Defaults to settings.openai_temperature.
            max_tokens: Maximum tokens for response. Defaults to settings.openai_max_tokens.
            llm: Optional LLM override (useful for testing).
        """
        self.api_key = api_key if api_key is not None else settings.get_openai_api_key()
        self.model = model or settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self.max_tokens = max_tokens or settings.openai_max_tokens
        self._llm_override = llm

    @property
    def llm(self) -> Any:
        """Get or create the ChatOpenAI instance.

        Raises:
            ExtractionAgentError: If API key is not configured.
        """
        if self._llm_override is not None:
            return self._llm_override

        if not self.api_key:
            raise ExtractionAgentError(
                "OpenAI API key not configured",
                error_type="configuration_error",
                details={"missing": "openai_api_key"},
            )

        return ChatOpenAI(
            api_key=self.api_key,  # type: ignore[arg-type]
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def extract(
        self,
        ordered_text: str,
        schema_info: SchemaInfo,
        regions: Sequence[LayoutRegion] | None = None,
        ordered_regions: Sequence[OrderedRegion] | None = None,
    ) -> ExtractionResult:
        """Extract structured data using OCR text, layout metadata, and tools.

        Args:
            ordered_text: OCR text in reading order.
            schema_info: Validated schema information.
            regions: Detected layout regions (for tool calls).
            ordered_regions: Ordered regions for metadata presentation.

        Returns:
            ExtractionResult containing extracted data and metadata.
        """
        start_time = time.time()
        region_list = list(regions or [])
        region_metadata = self._serialize_regions(region_list, ordered_regions)
        system_prompt = self._build_system_prompt(
            ordered_text=ordered_text,
            regions=region_metadata,
            schema_info=schema_info,
        )
        tools = self._build_tools(region_list)

        try:
            agent = build_agent(
                model=self.llm,
                name="tool-based-extraction-agent",
                system_prompt=system_prompt,
                tools=tools,
            )
            response = invoke_agent(
                agent,
                [HumanMessage(content="Extract the information now.")],
                metadata={
                    "component": "tool_based_extraction",
                    "agent_name": "tool-based-extraction-agent",
                    "model": self.model,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise ExtractionAgentError(
                f"Extraction agent invocation failed: {exc}",
                error_type="llm_error",
                details={"original_error": str(exc)},
            ) from exc

        content = get_message_content(response)
        extracted_data = self._parse_json_response(content)
        field_extractions = self._build_field_extractions(extracted_data, schema_info)

        usage_metadata = get_usage_metadata(response)
        prompt_tokens = usage_metadata.get("input_tokens", 0)
        completion_tokens = usage_metadata.get("output_tokens", 0)
        total_tokens = usage_metadata.get(
            "total_tokens", prompt_tokens + completion_tokens
        )

        return ExtractionResult(
            extracted_data=extracted_data,
            field_extractions=field_extractions,
            model_used=self.model,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            processing_time_seconds=time.time() - start_time,
            raw_response=content,
        )

    def _build_tools(self, regions: Sequence[LayoutRegion]) -> list[BaseTool]:
        def analyze_chart(region_id: str) -> dict[str, Any]:
            result = AnalyzeChart.invoke(
                {"region_id": region_id, "regions": list(regions)}
            )
            return cast(dict[str, Any], result)

        def analyze_table(region_id: str) -> dict[str, Any]:
            result = AnalyzeTable.invoke(
                {"region_id": region_id, "regions": list(regions)}
            )
            return cast(dict[str, Any], result)

        return [
            StructuredTool.from_function(
                func=analyze_chart,
                name="AnalyzeChart",
                description=("Use for chart/graph regions to extract structured data."),
            ),
            StructuredTool.from_function(
                func=analyze_table,
                name="AnalyzeTable",
                description=(
                    "Use for complex table regions when OCR text is insufficient."
                ),
            ),
        ]

    def _build_system_prompt(
        self,
        ordered_text: str,
        regions: list[dict[str, Any]],
        schema_info: SchemaInfo,
    ) -> str:
        trimmed_text = self._truncate_text(ordered_text)
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            ordered_text=trimmed_text,
            regions=json.dumps(regions, indent=2),
            json_schema=json.dumps(schema_info.schema, indent=2),
        )

    def _truncate_text(self, ordered_text: str) -> str:
        if len(ordered_text) <= self.MAX_OCR_CHARS:
            return ordered_text
        logger.info(
            "OCR text truncated for prompt", extra={"max_chars": self.MAX_OCR_CHARS}
        )
        return (
            f"{ordered_text[: self.MAX_OCR_CHARS]}\n[...truncated for prompt length...]"
        )

    def _serialize_regions(
        self,
        regions: Sequence[LayoutRegion],
        ordered_regions: Sequence[OrderedRegion] | None,
    ) -> list[dict[str, Any]]:
        if ordered_regions:
            return [region.to_dict() for region in ordered_regions]

        return [
            {
                "region_id": region.region_id,
                "region_type": region.region_type.value,
                "page_number": region.page_number,
                "confidence": region.confidence,
                "bbox": region.bbox.to_dict(),
            }
            for region in regions
        ]

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return data
            raise ExtractionAgentError(
                "Extraction response is not a JSON object",
                error_type="parse_error",
                details={"response_preview": response[:200]},
            )
        except json.JSONDecodeError as exc:
            raise ExtractionAgentError(
                f"Failed to parse JSON response: {exc}",
                error_type="parse_error",
                details={"response_preview": response[:200]},
            ) from exc

    def _build_field_extractions(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
    ) -> list[FieldExtraction]:
        extractions: list[FieldExtraction] = []

        for field_info in schema_info.all_fields:
            value = self._get_nested_value(data, field_info.path)
            extractions.append(
                FieldExtraction(
                    field_path=field_info.path,
                    value=value,
                    confidence=None,
                    source_text=None,
                    reasoning=None,
                )
            )

            if field_info.nested_fields and isinstance(value, dict):
                for nested_field in field_info.nested_fields:
                    nested_value = self._get_nested_value(data, nested_field.path)
                    extractions.append(
                        FieldExtraction(
                            field_path=nested_field.path,
                            value=nested_value,
                            confidence=None,
                            source_text=None,
                            reasoning=None,
                        )
                    )

        return extractions

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        keys = path.split(".")
        current: Any = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
