"""Single tool-using extraction agent with lightweight verification loop.

Replaces the multi-agent orchestration loop (planner + verifier + refiner)
with a single ``ExtractionAgent`` built on :func:`langchain.agents.create_agent`.
The agent receives OCR text, layout region metadata, and a target JSON schema
in its system prompt, autonomously invokes ``analyze_chart``, ``analyze_table``,
``analyze_form``, and ``analyze_image`` tools for visual regions, and returns
structured JSON output matching the user schema.

The agent includes a lightweight verification and refinement loop that:
1. Performs extraction using the tool-based agent
2. Verifies quality using rule-based checks (fast, no extra LLM call)
3. If issues found, provides feedback and re-extracts (limited iterations)
4. Returns the best result with proper quality metrics
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
from agentic_document_extraction.agents.tools.analyze_diagram import analyze_diagram
from agentic_document_extraction.agents.tools.analyze_form import analyze_form
from agentic_document_extraction.agents.tools.analyze_handwriting import (
    analyze_handwriting,
)
from agentic_document_extraction.agents.tools.analyze_image import analyze_image
from agentic_document_extraction.agents.tools.analyze_logo import analyze_logo
from agentic_document_extraction.agents.tools.analyze_math import analyze_math
from agentic_document_extraction.agents.tools.analyze_signature import analyze_signature
from agentic_document_extraction.agents.tools.analyze_spreadsheet import (
    analyze_spreadsheet,
)
from agentic_document_extraction.agents.tools.analyze_table import analyze_table
from agentic_document_extraction.agents.verifier import (
    IssueSeverity,
    QualityVerificationAgent,
    VerificationReport,
    VerificationStatus,
)
from agentic_document_extraction.config import settings
from agentic_document_extraction.excel_document import ExcelDocument
from agentic_document_extraction.models import FormatInfo
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.layout_detector import LayoutRegion
from agentic_document_extraction.services.schema_validator import SchemaInfo
from agentic_document_extraction.utils.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom agent state
# ---------------------------------------------------------------------------


class ExtractionAgentState(AgentState):  # type: ignore[type-arg]
    """Agent state that carries layout regions for tool injection."""

    regions: list[LayoutRegion]
    spreadsheet: ExcelDocument | None


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
{spreadsheet_section}

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

_REFINEMENT_PROMPT_TEMPLATE = """\
You are an expert document information extraction agent.

Your previous extraction had the following quality issues that need to be addressed:

## Issues to Fix
{issues}

## Previous Extraction
```json
{previous_extraction}
```

## Document Text (reading order)
```
{ocr_text}
```

{region_section}
{spreadsheet_section}

## Target JSON Schema
```json
{schema_json}
```

## Instructions
1. Fix the specific issues listed above.
2. Preserve correct values from the previous extraction.
3. For fields you cannot find in the document, use `null`.
4. Match data types exactly (string, number, boolean, array, object).
{tool_instructions}
5. Return ONLY valid JSON matching the schema — nothing else.
"""

_REGION_TABLE_HEADER = """\
## Document Regions
The document has the following layout regions detected via layout analysis.
Use the tool(s) below to inspect any chart or table region when the OCR text
alone is insufficient.

| region_id | type | page | confidence |
|-----------|------|------|------------|
"""

_TOOL_INSTRUCTIONS = """\

## Visual Analysis Tools

You have access to visual analysis tools for extracting data from document regions.
Use these tools ONLY when the OCR text above is insufficient for a schema field.

**When to use tools:**
- The Document Regions table shows PICTURE or TABLE regions that may contain data
- A schema field requires data that appears to be in an image (chart, table, form)
- OCR text is missing, garbled, or incomplete for visual content

**When NOT to use tools:**
- All required data is already in the OCR text above
- The document has no PICTURE or TABLE regions (only TEXT regions)
- Simple text extraction is sufficient to fill the schema

**Available tools:**
- `analyze_chart`: Extract data from charts and graphs (bar, line, pie, etc.)
- `analyze_table`: Extract structured data from tables (headers, rows, values)
- `analyze_diagram`: Extract flowcharts, org charts, process diagrams
- `analyze_form`: Extract form fields, checkboxes, radio buttons, values
- `analyze_handwriting`: Transcribe handwritten text
- `analyze_image`: Describe photos, illustrations, general images
- `analyze_logo`: Identify logos, certifications (ISO, FDA, CE), brand marks
- `analyze_math`: Extract equations, formulas, scientific notation (LaTeX)
- `analyze_signature`: Extract signature blocks, stamps, seals
- `analyze_spreadsheet`: Access native Excel cell values, formulas, and ranges

**How to use:**
5. Call tools with the `region_id` from the Document Regions table above.
6. For spreadsheets, call `analyze_spreadsheet` with optional sheet name, row range, or columns.
7. You may call multiple tools as needed.
8. If the OCR text or spreadsheet preview already contains the data, skip tool calls.
"""


# ---------------------------------------------------------------------------
# ExtractionAgent
# ---------------------------------------------------------------------------


@dataclass
class ExtractionAgent:
    """Single LangChain agent with lightweight verification and refinement loop.

    Performs extraction using a tool-based agent, then verifies quality and
    optionally refines the extraction if issues are found.

    Attributes:
        api_key: OpenAI API key. Falls back to settings when ``None``.
        model: Model name for the agent LLM.
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens.
        max_iterations: Maximum extraction iterations. Defaults to settings.
        use_llm_verification: Whether to use LLM for deep verification analysis.
    """

    api_key: str | None = field(default=None, repr=False)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_iterations: int | None = None
    use_llm_verification: bool = False  # Fast verification by default

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = settings.get_openai_api_key()
        if self.model is None:
            self.model = settings.openai_model
        if self.temperature is None:
            self.temperature = settings.openai_temperature
        if self.max_tokens is None:
            self.max_tokens = settings.openai_max_tokens
        if self.max_iterations is None:
            self.max_iterations = settings.max_refinement_iterations

    # -- public API --------------------------------------------------------

    def extract(
        self,
        text: str,
        schema_info: SchemaInfo,
        format_info: FormatInfo,
        layout_regions: list[LayoutRegion] | None = None,
        spreadsheet: ExcelDocument | None = None,
    ) -> AgenticLoopResult:
        """Run the extraction agent with lightweight verification loop.

        Args:
            text: OCR / extracted text in reading order.
            schema_info: Validated schema with field metadata.
            format_info: Document format metadata.
            layout_regions: Optional layout regions (for visual docs).

        Returns:
            An ``AgenticLoopResult`` with the best extraction and quality report.
        """
        start_time = time.time()
        regions = layout_regions or []

        # Build tools list - always provide all tools, let agent decide
        tools: list[Any] = [
            analyze_chart,
            analyze_diagram,
            analyze_form,
            analyze_handwriting,
            analyze_image,
            analyze_logo,
            analyze_math,
            analyze_signature,
            analyze_table,
            analyze_spreadsheet,
        ]

        # Build LLM
        llm = ChatOpenAI(
            api_key=self.api_key,  # type: ignore[arg-type]
            model=self.model or "gpt-4o",
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )

        # Build agent with structured output from schema
        # Sanitize schema title for OpenAI's response_format
        response_schema = dict(schema_info.schema)
        if "title" in response_schema:
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", response_schema["title"])
            response_schema["title"] = sanitized

        # Initialize verifier for quality checking
        verifier = QualityVerificationAgent(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        thresholds = verifier.get_default_thresholds()
        processing_category = format_info.processing_category

        # Track best result across iterations
        best_result: ExtractionResult | None = None
        best_verification: VerificationReport | None = None
        best_score: float = -1.0
        best_iteration: int = 0

        # Track iteration history
        iteration_history: list[IterationMetrics] = []
        total_tokens = 0
        previous_extraction: dict[str, Any] | None = None
        previous_issues: list[str] | None = None

        max_iter = self.max_iterations or settings.max_refinement_iterations

        for iteration in range(1, max_iter + 1):
            iteration_start = time.time()
            iteration_tokens = 0

            logger.info(
                f"ExtractionAgent iteration {iteration}/{max_iter}",
                extra={
                    "model": self.model,
                    "num_tools": len(tools),
                    "num_regions": len(regions),
                },
            )

            # Build system prompt (initial or refinement)
            if iteration == 1 or previous_extraction is None:
                system_prompt = self._build_system_prompt(
                    text, schema_info, regions, spreadsheet
                )
            else:
                system_prompt = self._build_refinement_prompt(
                    text,
                    schema_info,
                    regions,
                    previous_extraction,
                    previous_issues or [],
                    spreadsheet,
                )

            # Create agent for this iteration
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
                state_schema=ExtractionAgentState if tools else None,
                response_format=response_schema,
            )

            # Build invoke input
            invoke_input: dict[str, Any] = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Extract the data from the document according to the "
                            "schema."
                            if iteration == 1
                            else "Fix the issues and re-extract the data."
                        ),
                    }
                ],
            }
            if tools:
                invoke_input["regions"] = regions
                invoke_input["spreadsheet"] = spreadsheet

            try:
                result = agent.invoke(invoke_input)  # type: ignore[arg-type]
            except Exception as exc:
                logger.error(
                    f"ExtractionAgent invoke failed at iteration {iteration}",
                    exc_info=True,
                    extra={"model": self.model},
                )
                # If we have a previous result, return it; otherwise raise
                if best_result is not None and best_verification is not None:
                    break
                raise DocumentProcessingError(
                    message=f"Extraction agent failed: {exc}",
                    error_type="agent_invoke_error",
                ) from exc

            iteration_time = time.time() - iteration_start

            # Extract structured response
            extracted_data = self._parse_agent_result(result, schema_info)
            previous_extraction = extracted_data

            # Extract token usage
            tokens, prompt_tokens, completion_tokens = self._extract_token_usage(result)
            iteration_tokens += tokens
            total_tokens += tokens

            # Build field extractions
            field_extractions = self._build_field_extractions(
                extracted_data, schema_info
            )

            extraction_result = ExtractionResult(
                extracted_data=extracted_data,
                field_extractions=field_extractions,
                model_used=self.model or "",
                total_tokens=tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                processing_time_seconds=iteration_time,
            )

            # Verify extraction quality
            verification = verifier.verify(
                extraction_result=extraction_result,
                schema_info=schema_info,
                thresholds=thresholds,
                processing_category=processing_category,
                use_llm_analysis=self.use_llm_verification,
            )
            iteration_tokens += verification.total_tokens
            total_tokens += verification.total_tokens

            # Calculate score for tracking best result
            score = self._calculate_result_score(verification)
            if score > best_score:
                best_score = score
                best_result = extraction_result
                best_verification = verification
                best_iteration = iteration

            # Record iteration metrics
            iteration_metrics = IterationMetrics(
                iteration_number=iteration,
                verification_status=verification.status,
                overall_confidence=verification.metrics.overall_confidence,
                required_field_coverage=verification.metrics.required_field_coverage,
                issues_count=len(verification.issues),
                critical_issues_count=len(verification.critical_issues),
                total_tokens=iteration_tokens,
                processing_time_seconds=iteration_time,
            )
            iteration_history.append(iteration_metrics)

            logger.info(
                f"Iteration {iteration}: status={verification.status.value}, "
                f"confidence={verification.metrics.overall_confidence:.2f}, "
                f"coverage={verification.metrics.required_field_coverage:.2f}, "
                f"issues={len(verification.issues)}"
            )

            # Check if quality threshold met
            if verification.status == VerificationStatus.PASSED:
                logger.info(f"Quality thresholds met at iteration {iteration}")
                break

            # Check if max iterations reached
            if iteration >= max_iter:
                logger.info(f"Max iterations ({max_iter}) reached without convergence")
                break

            # Prepare feedback for next iteration
            previous_issues = self._format_issues_for_refinement(verification)

        processing_time = time.time() - start_time

        # Ensure we have a result
        if best_result is None or best_verification is None:
            raise DocumentProcessingError(
                message="No extraction result produced",
                error_type="extraction_error",
            )

        # Build final result
        return self._wrap_result(
            best_result,
            best_verification,
            format_info,
            iteration_history,
            processing_time,
            total_tokens,
            best_iteration,
            thresholds,
        )

    # -- private helpers ---------------------------------------------------

    def _build_system_prompt(
        self,
        text: str,
        schema_info: SchemaInfo,
        regions: list[LayoutRegion],
        spreadsheet: ExcelDocument | None = None,
    ) -> str:
        """Assemble the initial extraction system prompt.

        Tool instructions are always included since all 9 tools are always
        provided (per Task 0051). The agent decides when to use them based
        on document content. Instructions guide the agent to skip tools when
        OCR text is sufficient.
        """
        region_section = self._build_region_section(regions)
        spreadsheet_section = self._build_spreadsheet_section(spreadsheet)
        # Always include tool instructions - agent decides when to use
        tool_instructions = _TOOL_INSTRUCTIONS
        ocr_text = self._truncate_text(text)

        return _SYSTEM_PROMPT_TEMPLATE.format(
            ocr_text=ocr_text,
            region_section=region_section,
            spreadsheet_section=spreadsheet_section,
            schema_json=json.dumps(schema_info.schema, indent=2),
            tool_instructions=tool_instructions,
        )

    def _build_refinement_prompt(
        self,
        text: str,
        schema_info: SchemaInfo,
        regions: list[LayoutRegion],
        previous_extraction: dict[str, Any],
        issues: list[str],
        spreadsheet: ExcelDocument | None = None,
    ) -> str:
        """Assemble the refinement system prompt with feedback.

        Tool instructions are always included since all 9 tools are always
        provided (per Task 0051).
        """
        region_section = self._build_region_section(regions)
        spreadsheet_section = self._build_spreadsheet_section(spreadsheet)
        # Always include tool instructions - agent decides when to use
        tool_instructions = _TOOL_INSTRUCTIONS
        ocr_text = self._truncate_text(text)
        issues_text = "\n".join(f"- {issue}" for issue in issues) if issues else "None"

        return _REFINEMENT_PROMPT_TEMPLATE.format(
            issues=issues_text,
            previous_extraction=json.dumps(previous_extraction, indent=2),
            ocr_text=ocr_text,
            region_section=region_section,
            spreadsheet_section=spreadsheet_section,
            schema_json=json.dumps(schema_info.schema, indent=2),
            tool_instructions=tool_instructions,
        )

    @staticmethod
    def _build_region_section(regions: list[LayoutRegion]) -> str:
        """Build the region metadata table section."""
        if not regions:
            return ""

        rows = []
        for r in regions:
            rows.append(
                f"| {r.region_id} | {r.region_type.value} "
                f"| {r.page_number} | {r.confidence:.2f} |"
            )
        return _REGION_TABLE_HEADER + "\n".join(rows)

    @staticmethod
    def _build_spreadsheet_section(spreadsheet: ExcelDocument | None) -> str:
        """Build a brief spreadsheet overview for the prompt."""
        if spreadsheet is None:
            return ""

        lines = ["## Spreadsheet Overview", "Sheets available:"]
        for sheet in spreadsheet.sheets:
            lines.append(
                f"- {sheet.name} ({sheet.row_count} rows x {sheet.column_count} cols)"
            )
        lines.append(f"Active sheet: {spreadsheet.active_sheet}")
        lines.append(
            "Use `analyze_spreadsheet` to inspect specific sheets, ranges, or columns."
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _truncate_text(text: str, max_len: int = 12000) -> str:
        """Truncate text to stay within context limits."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n... [text truncated] ..."

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
    def _format_issues_for_refinement(
        verification: VerificationReport,
    ) -> list[str]:
        """Format verification issues as feedback for the next iteration."""
        issues: list[str] = []

        # Sort by severity (critical first)
        sorted_issues = sorted(
            verification.issues,
            key=lambda i: (
                0
                if i.severity == IssueSeverity.CRITICAL
                else 1
                if i.severity == IssueSeverity.HIGH
                else 2
                if i.severity == IssueSeverity.MEDIUM
                else 3
            ),
        )

        for issue in sorted_issues[:10]:  # Limit to top 10 issues
            severity = issue.severity.value.upper()
            msg = f"[{severity}] {issue.field_path}: {issue.message}"
            if issue.suggestion:
                msg += f" ({issue.suggestion})"
            issues.append(msg)

        return issues

    @staticmethod
    def _calculate_result_score(verification: VerificationReport) -> float:
        """Calculate a score for ranking extraction results."""
        metrics = verification.metrics

        # Weight different metrics
        score = (
            metrics.overall_confidence * 0.3
            + metrics.required_field_coverage * 0.4
            + metrics.completeness_score * 0.2
            + metrics.consistency_score * 0.1
        )

        # Penalize for critical issues
        score -= len(verification.critical_issues) * 0.2

        # Bonus for passing
        if verification.status == VerificationStatus.PASSED:
            score += 0.1

        return max(0.0, min(1.0, score))

    @staticmethod
    def _wrap_result(
        extraction_result: ExtractionResult,
        verification: VerificationReport,
        format_info: FormatInfo,
        iteration_history: list[IterationMetrics],
        processing_time: float,
        total_tokens: int,
        best_iteration: int,
        thresholds: QualityThreshold,
    ) -> AgenticLoopResult:
        """Wrap extraction result in AgenticLoopResult for compatibility."""
        iterations_completed = len(iteration_history)
        converged = verification.status == VerificationStatus.PASSED

        # Build ExtractionPlan
        plan = ExtractionPlan(
            document_characteristics=DocumentCharacteristics(
                processing_category=format_info.processing_category.value,
                format_family=format_info.format_family.value,
                estimated_complexity="moderate",
            ),
            schema_complexity="moderate",  # type: ignore[arg-type]
            extraction_strategy="tool_agent_with_verification",
            steps=[],
            region_priorities=[],
            challenges=[],
            quality_thresholds=thresholds,
            reasoning=(
                f"Tool-based extraction with lightweight verification loop. "
                f"Completed {iterations_completed} iteration(s), "
                f"{'converged' if converged else 'did not converge'}."
            ),
            estimated_confidence=verification.metrics.overall_confidence,
        )

        return AgenticLoopResult(
            final_result=extraction_result,
            final_verification=verification,
            plan=plan,
            iterations_completed=iterations_completed,
            iteration_history=iteration_history,
            converged=converged,
            best_iteration=best_iteration,
            total_tokens=total_tokens,
            total_processing_time_seconds=processing_time,
            loop_metadata={
                "agent_type": "tool_agent_with_verification",
                "processing_category": format_info.processing_category.value,
                "max_iterations": len(iteration_history),
            },
        )
