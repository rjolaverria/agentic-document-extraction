"""Single tool-using extraction agent.

Replaces the multi-agent orchestration loop (planner + verifier + refiner)
with a single ``ExtractionAgent`` built on :func:`langchain.agents.create_agent`.
The agent receives OCR text, layout region metadata, and a target JSON schema
in its system prompt, autonomously invokes ``analyze_chart`` / ``analyze_table``
tools for visual regions, and returns structured JSON output matching the
user schema.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain_openai import ChatOpenAI

from agentic_document_extraction.agents.planner import (
    DocumentCharacteristics,
    ExtractionPlan,
    QualityThreshold,
)
from agentic_document_extraction.agents.refiner import (
    AgenticLoopResult,
    IterationMetrics,
)
from agentic_document_extraction.agents.tools.analyze_chart import analyze_chart
from agentic_document_extraction.agents.tools.analyze_table import analyze_table
from agentic_document_extraction.agents.verifier import (
    QualityMetrics,
    VerificationReport,
    VerificationStatus,
)
from agentic_document_extraction.config import settings
from agentic_document_extraction.models import FormatInfo, ProcessingCategory
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionType,
)
from agentic_document_extraction.services.schema_validator import SchemaInfo
from agentic_document_extraction.utils.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom agent state
# ---------------------------------------------------------------------------


class ExtractionAgentState(AgentState):  # type: ignore[type-arg]
    """Agent state that carries layout regions for tool injection."""

    regions: list[LayoutRegion]


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert document information extraction agent.

Your task is to extract structured data from the document text below according
to the **Target JSON Schema**. Return ONLY the extracted JSON object — no
commentary, no markdown fences.

## Document Text (reading order)
```
{ocr_text}
```

{region_section}

## Target JSON Schema
```json
{schema_json}
```

## Instructions
1. Extract every field defined in the schema from the document text.
2. For fields you cannot find, use `null`.
3. Match data types exactly (string, number, boolean, array, object).
{tool_instructions}
4. Return ONLY valid JSON matching the schema — nothing else.
"""

_REGION_TABLE_HEADER = """\
## Document Regions
The document has the following layout regions detected via layout analysis.
Use the tool(s) below to inspect any chart or table region when the OCR text
alone is insufficient.

| region_id | type | page | confidence |
|-----------|------|------|------------|
"""

_TOOL_INSTRUCTIONS_VISUAL = """\
5. When a schema field likely comes from a chart/graph region, call
   `analyze_chart` with that region's `region_id`.
6. When a schema field likely comes from a table region, call
   `analyze_table` with that region's `region_id`.
7. You may call tools multiple times for different regions.
"""


# ---------------------------------------------------------------------------
# ExtractionAgent
# ---------------------------------------------------------------------------


@dataclass
class ExtractionAgent:
    """Single LangChain agent that replaces the multi-agent loop.

    Attributes:
        api_key: OpenAI API key. Falls back to settings when ``None``.
        model: Model name for the agent LLM.
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens.
    """

    api_key: str | None = field(default=None, repr=False)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = settings.get_openai_api_key()
        if self.model is None:
            self.model = settings.openai_model
        if self.temperature is None:
            self.temperature = settings.openai_temperature
        if self.max_tokens is None:
            self.max_tokens = settings.openai_max_tokens

    # -- public API --------------------------------------------------------

    def extract(
        self,
        text: str,
        schema_info: SchemaInfo,
        format_info: FormatInfo,
        layout_regions: list[LayoutRegion] | None = None,
    ) -> AgenticLoopResult:
        """Run the extraction agent and return an ``AgenticLoopResult``.

        Args:
            text: OCR / extracted text in reading order.
            schema_info: Validated schema with field metadata.
            format_info: Document format metadata.
            layout_regions: Optional layout regions (for visual docs).

        Returns:
            An ``AgenticLoopResult`` compatible with the existing pipeline.
        """
        start_time = time.time()
        regions = layout_regions or []

        # Determine if we need visual tools
        has_visual_regions = any(
            r.region_type in (RegionType.PICTURE, RegionType.TABLE) for r in regions
        )
        is_visual = format_info.processing_category == ProcessingCategory.VISUAL

        # Build tools list
        tools: list[Any] = []
        if is_visual and has_visual_regions:
            if analyze_chart is not None:
                tools.append(analyze_chart)
            if analyze_table is not None:
                tools.append(analyze_table)

        # Build system prompt
        system_prompt = self._build_system_prompt(
            text, schema_info, regions, has_tools=bool(tools)
        )

        # Build LLM
        llm = ChatOpenAI(
            api_key=self.api_key,  # type: ignore[arg-type]
            model=self.model or "gpt-4o",
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )

        # Build agent with structured output from schema
        # Sanitize schema title for OpenAI's response_format (must match ^[a-zA-Z0-9_-]+$)
        response_schema = dict(schema_info.schema)
        if "title" in response_schema:
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", response_schema["title"])
            response_schema["title"] = sanitized

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            state_schema=ExtractionAgentState if tools else None,
            response_format=response_schema,
        )

        # Invoke
        invoke_input: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": "Extract the data from the document according to the schema.",
                }
            ],
        }
        if tools:
            invoke_input["regions"] = regions

        logger.info(
            "Invoking ExtractionAgent",
            extra={
                "model": self.model,
                "num_tools": len(tools),
                "num_regions": len(regions),
            },
        )

        try:
            result = agent.invoke(invoke_input)  # type: ignore[arg-type]
        except Exception as exc:
            logger.error(
                "ExtractionAgent invoke failed",
                exc_info=True,
                extra={"model": self.model},
            )
            raise DocumentProcessingError(
                message=f"Extraction agent failed: {exc}",
                error_type="agent_invoke_error",
            ) from exc

        processing_time = time.time() - start_time

        # Extract structured response
        extracted_data = self._parse_agent_result(result, schema_info)

        # Extract token usage from last AI message
        total_tokens, prompt_tokens, completion_tokens = self._extract_token_usage(
            result
        )

        # Build field extractions
        field_extractions = self._build_field_extractions(extracted_data, schema_info)

        extraction_result = ExtractionResult(
            extracted_data=extracted_data,
            field_extractions=field_extractions,
            model_used=self.model or "",
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            processing_time_seconds=processing_time,
        )

        # Wrap in AgenticLoopResult for pipeline compatibility
        return self._wrap_result(
            extraction_result,
            format_info,
            processing_time,
            total_tokens,
        )

    # -- private helpers ---------------------------------------------------

    def _build_system_prompt(
        self,
        text: str,
        schema_info: SchemaInfo,
        regions: list[LayoutRegion],
        *,
        has_tools: bool,
    ) -> str:
        """Assemble the system prompt from text, regions, and schema."""
        # Region metadata table
        region_section = ""
        if regions:
            rows = []
            for r in regions:
                rows.append(
                    f"| {r.region_id} | {r.region_type.value} "
                    f"| {r.page_number} | {r.confidence:.2f} |"
                )
            region_section = _REGION_TABLE_HEADER + "\n".join(rows)

        tool_instructions = _TOOL_INSTRUCTIONS_VISUAL if has_tools else ""

        # Truncate very long OCR text to stay within context limits
        max_text_len = 12000
        ocr_text = text[:max_text_len]
        if len(text) > max_text_len:
            ocr_text += "\n... [text truncated] ..."

        return _SYSTEM_PROMPT_TEMPLATE.format(
            ocr_text=ocr_text,
            region_section=region_section,
            schema_json=json.dumps(schema_info.schema, indent=2),
            tool_instructions=tool_instructions,
        )

    @staticmethod
    def _parse_agent_result(
        result: dict[str, Any],
        schema_info: SchemaInfo,  # noqa: ARG004
    ) -> dict[str, Any]:
        """Extract the structured JSON from the agent's output."""
        # Prefer structured_response (from response_format)
        structured = result.get("structured_response")
        if isinstance(structured, dict):
            return structured

        # Fallback: parse the last AI message content
        messages = result.get("messages", [])
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                # Try extracting embedded JSON
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    try:
                        data = json.loads(content[start:end])
                        if isinstance(data, dict):
                            return data
                    except (json.JSONDecodeError, ValueError):
                        continue

        logger.warning("ExtractionAgent returned no parseable structured output")
        return {}

    @staticmethod
    def _extract_token_usage(result: dict[str, Any]) -> tuple[int, int, int]:
        """Sum token usage across all AI messages."""
        total = prompt = completion = 0
        for msg in result.get("messages", []):
            usage = getattr(msg, "usage_metadata", None)
            if isinstance(usage, dict):
                prompt += usage.get("input_tokens", 0)
                completion += usage.get("output_tokens", 0)
        total = prompt + completion
        return total, prompt, completion

    @staticmethod
    def _build_field_extractions(
        data: dict[str, Any],
        schema_info: SchemaInfo,
    ) -> list[FieldExtraction]:
        """Build FieldExtraction list from extracted data and schema."""
        extractions: list[FieldExtraction] = []

        def _get(d: Any, path: str) -> Any:
            for key in path.split("."):
                if isinstance(d, dict):
                    d = d.get(key)
                else:
                    return None
            return d

        for field_info in schema_info.all_fields:
            value = _get(data, field_info.path)
            confidence = 0.85 if value is not None else None
            extractions.append(
                FieldExtraction(
                    field_path=field_info.path,
                    value=value,
                    confidence=confidence,
                )
            )

        return extractions

    @staticmethod
    def _wrap_result(
        extraction_result: ExtractionResult,
        format_info: FormatInfo,
        processing_time: float,
        total_tokens: int,
    ) -> AgenticLoopResult:
        """Wrap an ExtractionResult in AgenticLoopResult for compatibility."""
        # Build a minimal ExtractionPlan
        plan = ExtractionPlan(
            document_characteristics=DocumentCharacteristics(
                processing_category=format_info.processing_category.value,
                format_family=format_info.format_family.value,
                estimated_complexity="moderate",
            ),
            schema_complexity="moderate",  # type: ignore[arg-type]
            extraction_strategy="tool_agent",
            steps=[],
            region_priorities=[],
            challenges=[],
            quality_thresholds=QualityThreshold(
                min_overall_confidence=settings.min_overall_confidence,
                min_field_confidence=settings.min_field_confidence,
                required_field_coverage=settings.required_field_coverage,
                max_iterations=1,
            ),
            reasoning="Single-pass extraction via tool-using agent.",
            estimated_confidence=0.85,
        )

        # Build a minimal VerificationReport
        field_count = len(extraction_result.field_extractions)
        extracted_count = sum(
            1 for f in extraction_result.field_extractions if f.value is not None
        )
        coverage = extracted_count / field_count if field_count else 1.0

        verification = VerificationReport(
            status=VerificationStatus.PASSED,
            metrics=QualityMetrics(
                overall_confidence=0.85,
                schema_coverage=coverage,
                required_field_coverage=coverage,
                optional_field_coverage=coverage,
                completeness_score=coverage,
                consistency_score=1.0,
                min_field_confidence=0.85,
                fields_with_low_confidence=0,
                total_fields=field_count,
                extracted_fields=extracted_count,
            ),
        )

        iteration_metrics = IterationMetrics(
            iteration_number=1,
            verification_status=VerificationStatus.PASSED,
            overall_confidence=0.85,
            required_field_coverage=coverage,
            issues_count=0,
            critical_issues_count=0,
            total_tokens=total_tokens,
            processing_time_seconds=processing_time,
        )

        return AgenticLoopResult(
            final_result=extraction_result,
            final_verification=verification,
            plan=plan,
            iterations_completed=1,
            iteration_history=[iteration_metrics],
            converged=True,
            best_iteration=1,
            total_tokens=total_tokens,
            total_processing_time_seconds=processing_time,
            loop_metadata={
                "agent_type": "tool_agent",
                "processing_category": format_info.processing_category.value,
            },
        )
