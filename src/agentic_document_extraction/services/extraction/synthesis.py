"""Visual document synthesis service.

This module provides functionality to combine region-level extractions
from visual documents into a coherent document-level extraction using
LangChain with OpenAI GPT models.

Key features:
- Combines information from all regions respecting reading order
- Resolves conflicts or redundancies between regions using LLM reasoning
- Maps combined information to user's JSON schema
- Validates against schema using jsonschema
- Generates JSON + Markdown output similar to text-based extraction
- Includes source references (page numbers, region types, coordinates)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agentic_document_extraction.config import settings
from agentic_document_extraction.services.extraction.region_visual_extraction import (
    DocumentRegionExtractionResult,
    RegionExtractionResult,
)
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.layout_detector import RegionType
from agentic_document_extraction.services.reading_order_detector import (
    DocumentReadingOrder,
    PageReadingOrder,
)
from agentic_document_extraction.services.schema_validator import SchemaInfo
from agentic_document_extraction.utils.agent_helpers import (
    build_agent,
    get_message_content,
    get_usage_metadata,
    invoke_agent,
)

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    """Raised when synthesis fails."""

    def __init__(
        self,
        message: str,
        error_type: str = "synthesis_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with message and error details.

        Args:
            message: Error message.
            error_type: Type of error for categorization.
            details: Optional additional error details.
        """
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


@dataclass
class RegionSourceReference:
    """Source reference for a region that contributed to extraction."""

    region_id: str
    """Unique identifier of the region."""

    region_type: str
    """Type of the region (text, table, picture, etc.)."""

    page_number: int
    """Page number where the region appears."""

    reading_order_position: int
    """Position in the reading order sequence."""

    bbox: dict[str, float] | None = None
    """Bounding box coordinates (x0, y0, x1, y1)."""

    confidence: float = 0.0
    """Confidence score for this region's extraction."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with source reference information.
        """
        return {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "page_number": self.page_number,
            "reading_order_position": self.reading_order_position,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }


@dataclass
class SynthesizedContent:
    """Content synthesized from a region with source tracking."""

    content: Any
    """The synthesized content."""

    content_type: str
    """Type of content (text, table_data, figure_description, etc.)."""

    source_ref: RegionSourceReference
    """Source reference for this content."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with synthesized content information.
        """
        return {
            "content": self.content,
            "content_type": self.content_type,
            "source_ref": self.source_ref.to_dict(),
        }


@dataclass
class SynthesisResult:
    """Result of visual document synthesis."""

    extracted_data: dict[str, Any]
    """The extracted data matching the user's JSON schema."""

    field_extractions: list[FieldExtraction] = field(default_factory=list)
    """Detailed extraction info for each field."""

    source_references: list[RegionSourceReference] = field(default_factory=list)
    """References to source regions that contributed to extraction."""

    synthesized_contents: list[SynthesizedContent] = field(default_factory=list)
    """Individual synthesized content blocks before schema mapping."""

    model_used: str = ""
    """Name of the model used for synthesis."""

    total_tokens: int = 0
    """Total tokens used in the synthesis process."""

    prompt_tokens: int = 0
    """Tokens used in prompts."""

    completion_tokens: int = 0
    """Tokens used in completions."""

    processing_time_seconds: float = 0.0
    """Time taken for synthesis in seconds."""

    total_regions_processed: int = 0
    """Total number of regions processed."""

    total_pages: int = 0
    """Total number of pages in the document."""

    synthesis_confidence: float = 0.0
    """Overall confidence in the synthesis result."""

    raw_response: str | None = None
    """Raw response from the synthesis model."""

    pipeline_metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata about the processing pipeline."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with synthesis result information.
        """
        return {
            "extracted_data": self.extracted_data,
            "field_extractions": [
                {
                    "field_path": fe.field_path,
                    "value": fe.value,
                    "confidence": fe.confidence,
                    "source_text": fe.source_text,
                    "reasoning": fe.reasoning,
                }
                for fe in self.field_extractions
            ],
            "source_references": [sr.to_dict() for sr in self.source_references],
            "metadata": {
                "model_used": self.model_used,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "processing_time_seconds": self.processing_time_seconds,
                "total_regions_processed": self.total_regions_processed,
                "total_pages": self.total_pages,
                "synthesis_confidence": self.synthesis_confidence,
                "pipeline": self.pipeline_metadata,
            },
        }

    def to_extraction_result(self) -> ExtractionResult:
        """Convert to ExtractionResult for compatibility with OutputService.

        Returns:
            ExtractionResult compatible with the output generation pipeline.
        """
        return ExtractionResult(
            extracted_data=self.extracted_data,
            field_extractions=self.field_extractions,
            model_used=self.model_used,
            total_tokens=self.total_tokens,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            processing_time_seconds=self.processing_time_seconds,
            chunks_processed=self.total_pages,
            is_chunked=self.total_pages > 1,
            raw_response=self.raw_response,
        )


class SynthesisService:
    """Service for synthesizing visual document extractions into structured output.

    Combines region-level extractions into a coherent document-level
    extraction using LangChain with OpenAI GPT models. Handles conflict
    resolution, reading order, and schema mapping.
    """

    SYNTHESIS_SYSTEM_PROMPT = """You are an expert document synthesis assistant.
Your task is to combine extracted information from multiple document regions into a single coherent structured output.

IMPORTANT RULES:
1. Combine information from all provided regions respecting the reading order
2. Resolve conflicts by preferring higher confidence extractions and more complete information
3. For redundant information, merge and deduplicate appropriately
4. Map the combined information to match the requested JSON schema exactly
5. Preserve all relevant details from each region
6. For tables, preserve structure and all data
7. For figures, preserve descriptions and any extracted data
8. If information is missing or unclear, use null values
9. Provide confidence scores for the final extraction

ARRAY HANDLING RULES:
When extracting into an array of strings (e.g., skills, tags, keywords):
10. Split comma-separated, semicolon-separated, or line-separated lists into individual array items
11. Remove category prefixes (e.g., "Languages: Python, JavaScript" becomes ["Python", "JavaScript"])
12. Remove bullet points, dashes, or other list markers from individual items
13. Trim whitespace from each item
14. Each element should be a single, atomic value - not a grouped or prefixed string

You must respond with ONLY valid JSON matching this structure:
{{
  "extracted_data": {{ ... }},
  "field_confidence": {{
    "field_path": confidence_score (0.0-1.0),
    ...
  }},
  "overall_confidence": 0.0-1.0,
  "reasoning": "Brief explanation of synthesis decisions"
}}"""

    SYNTHESIS_USER_PROMPT = """Synthesize the following extracted regions into structured data matching the provided schema.

## Target JSON Schema:
```json
{schema}
```

## Required Fields:
{required_fields}

## Optional Fields:
{optional_fields}

## Extracted Regions (in reading order):
{regions_content}

## Task:
1. Combine information from all regions
2. Resolve any conflicts or redundancies
3. Map to the target schema
4. Provide confidence scores

Respond with ONLY the JSON structure specified."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the synthesis service.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Model name to use. Defaults to settings.openai_model.
            temperature: Sampling temperature. Defaults to settings.openai_temperature.
            max_tokens: Maximum tokens for response. Defaults to settings.openai_max_tokens.
        """
        self.api_key = api_key if api_key is not None else settings.get_openai_api_key()
        self.model = model or settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self.max_tokens = max_tokens or settings.openai_max_tokens

        self._llm: ChatOpenAI | None = None
        self._agent: Any | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LangChain ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            SynthesisError: If API key is not configured.
        """
        if self._llm is None:
            if not self.api_key:
                raise SynthesisError(
                    "OpenAI API key not configured",
                    error_type="configuration_error",
                    details={"missing": "openai_api_key"},
                )

            self._llm = ChatOpenAI(
                api_key=self.api_key,  # type: ignore[arg-type]
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

        return self._llm

    @property
    def agent(self) -> Any:
        """Get or create the LangChain agent for synthesis."""
        if self._agent is None:
            self._agent = build_agent(
                model=self.llm,
                name="synthesis-agent",
            )
        return self._agent

    def synthesize(
        self,
        region_extraction: DocumentRegionExtractionResult,
        reading_order: DocumentReadingOrder,
        schema_info: SchemaInfo,
    ) -> SynthesisResult:
        """Synthesize region extractions into structured output.

        Args:
            region_extraction: Extraction results from all regions.
            reading_order: Reading order for the document.
            schema_info: Schema to map extracted data to.

        Returns:
            SynthesisResult with combined extraction.

        Raises:
            SynthesisError: If synthesis fails.
        """
        start_time = time.time()

        # Build ordered content from regions
        synthesized_contents, source_refs = self._build_ordered_content(
            region_extraction, reading_order
        )

        if not synthesized_contents:
            # Handle empty documents
            return self._create_empty_result(schema_info, region_extraction)

        # Build prompt content from synthesized regions
        regions_content = self._format_regions_for_prompt(synthesized_contents)

        # Build and execute synthesis prompt
        prompt = self._build_synthesis_prompt(schema_info)

        try:
            # Use partial to properly escape JSON content with curly braces
            formatted_prompt = prompt.partial(
                schema=json.dumps(schema_info.schema, indent=2),
                required_fields=self._format_fields(schema_info.required_fields),
                optional_fields=self._format_fields(schema_info.optional_fields),
                regions_content=regions_content,
            )
            messages = formatted_prompt.format_messages()

            response = invoke_agent(
                self.agent,
                messages,
                metadata={
                    "component": "visual_synthesis",
                    "agent_name": "synthesis-agent",
                    "model": self.model,
                },
            )

            # Parse response
            content = get_message_content(response)

            extracted_data, field_confidence, overall_confidence, reasoning = (
                self._parse_synthesis_response(content, schema_info)
            )

            # Extract token usage
            usage_metadata = get_usage_metadata(response)
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            # Add tokens from region extraction
            total_tokens += region_extraction.total_tokens
            prompt_tokens += region_extraction.prompt_tokens
            completion_tokens += region_extraction.completion_tokens

            # Build field extractions with confidence
            field_extractions = self._build_field_extractions(
                extracted_data, schema_info, field_confidence, source_refs
            )

            processing_time = time.time() - start_time

            # Build pipeline metadata
            pipeline_metadata = self._build_pipeline_metadata(
                region_extraction, reading_order
            )

            logger.info(
                f"Synthesis completed: "
                f"regions={len(synthesized_contents)}, "
                f"pages={reading_order.total_pages}, "
                f"confidence={overall_confidence:.2f}, "
                f"tokens={total_tokens}, "
                f"time={processing_time:.2f}s"
            )

            return SynthesisResult(
                extracted_data=extracted_data,
                field_extractions=field_extractions,
                source_references=source_refs,
                synthesized_contents=synthesized_contents,
                model_used=self.model,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                processing_time_seconds=processing_time,
                total_regions_processed=len(synthesized_contents),
                total_pages=reading_order.total_pages,
                synthesis_confidence=overall_confidence,
                raw_response=content,
                pipeline_metadata=pipeline_metadata,
            )

        except Exception as e:
            if isinstance(e, SynthesisError):
                raise
            raise SynthesisError(
                f"Synthesis failed: {e}",
                error_type="llm_error",
                details={"original_error": str(e)},
            ) from e

    def _build_ordered_content(
        self,
        region_extraction: DocumentRegionExtractionResult,
        reading_order: DocumentReadingOrder,
    ) -> tuple[list[SynthesizedContent], list[RegionSourceReference]]:
        """Build ordered content from region extractions following reading order.

        Args:
            region_extraction: Extraction results from regions.
            reading_order: Document reading order.

        Returns:
            Tuple of (synthesized_contents, source_references).
        """
        synthesized_contents: list[SynthesizedContent] = []
        source_refs: list[RegionSourceReference] = []

        # Create lookup for region extractions
        extraction_lookup: dict[str, RegionExtractionResult] = {
            r.region_id: r for r in region_extraction.region_results
        }

        # Process pages in order
        global_order_index = 0
        for page_order in sorted(reading_order.pages, key=lambda p: p.page_number):
            for ordered_region in sorted(
                page_order.ordered_regions, key=lambda r: r.order_index
            ):
                # Skip regions marked for skipping
                if ordered_region.skip_in_reading:
                    continue

                region = ordered_region.region
                extraction = extraction_lookup.get(region.region_id)

                if extraction is None or not extraction.extracted_content:
                    continue

                # Build source reference
                source_ref = RegionSourceReference(
                    region_id=region.region_id,
                    region_type=region.region_type.value,
                    page_number=page_order.page_number,
                    reading_order_position=global_order_index,
                    bbox={
                        "x0": region.bbox.x0,
                        "y0": region.bbox.y0,
                        "x1": region.bbox.x1,
                        "y1": region.bbox.y1,
                    },
                    confidence=extraction.confidence,
                )
                source_refs.append(source_ref)

                # Determine content type
                content_type = self._get_content_type(region.region_type)

                # Build synthesized content
                synthesized = SynthesizedContent(
                    content=extraction.extracted_content,
                    content_type=content_type,
                    source_ref=source_ref,
                )
                synthesized_contents.append(synthesized)

                global_order_index += 1

        return synthesized_contents, source_refs

    def _get_content_type(self, region_type: RegionType) -> str:
        """Map region type to content type.

        Args:
            region_type: Type of the region.

        Returns:
            Content type string.
        """
        content_type_map = {
            RegionType.TEXT: "text",
            RegionType.TITLE: "title",
            RegionType.SECTION_HEADER: "section_header",
            RegionType.LIST_ITEM: "list_item",
            RegionType.TABLE: "table_data",
            RegionType.PICTURE: "figure_description",
            RegionType.CAPTION: "caption",
            RegionType.FORMULA: "formula",
            RegionType.FOOTNOTE: "footnote",
            RegionType.PAGE_HEADER: "page_header",
            RegionType.PAGE_FOOTER: "page_footer",
        }
        return content_type_map.get(region_type, "other")

    def _format_regions_for_prompt(
        self, synthesized_contents: list[SynthesizedContent]
    ) -> str:
        """Format synthesized contents for the LLM prompt.

        Args:
            synthesized_contents: List of synthesized content blocks.

        Returns:
            Formatted string for the prompt.
        """
        parts: list[str] = []

        for content in synthesized_contents:
            source = content.source_ref
            content_str = json.dumps(content.content, indent=2, default=str)

            part = f"""
### Region {source.reading_order_position + 1}: {content.content_type.upper()}
- Page: {source.page_number}
- Region Type: {source.region_type}
- Confidence: {source.confidence:.2f}

Content:
```json
{content_str}
```
"""
            parts.append(part)

        return "\n".join(parts)

    def _format_fields(self, fields: list[Any]) -> str:
        """Format schema fields for the prompt.

        Args:
            fields: List of FieldInfo objects.

        Returns:
            Formatted string listing fields.
        """
        if not fields:
            return "None"

        return "\n".join(
            f"- {f.path}: {f.field_type}"
            + (f" - {f.description}" if f.description else "")
            for f in fields
        )

    def _build_synthesis_prompt(
        self,
        schema_info: SchemaInfo,  # noqa: ARG002
    ) -> ChatPromptTemplate:
        """Build the synthesis prompt template.

        Args:
            schema_info: Schema information.

        Returns:
            Configured ChatPromptTemplate.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.SYNTHESIS_SYSTEM_PROMPT),
                ("human", self.SYNTHESIS_USER_PROMPT),
            ]
        )

    def _parse_synthesis_response(
        self,
        response: str,
        schema_info: SchemaInfo,  # noqa: ARG002
    ) -> tuple[dict[str, Any], dict[str, float], float, str | None]:
        """Parse the synthesis response.

        Args:
            response: Raw response from the model.
            schema_info: Schema information.

        Returns:
            Tuple of (extracted_data, field_confidence, overall_confidence, reasoning).

        Raises:
            SynthesisError: If parsing fails.
        """
        try:
            data = json.loads(response)

            if not isinstance(data, dict):
                raise SynthesisError(
                    "Expected JSON object response",
                    error_type="parse_error",
                    details={"received_type": type(data).__name__},
                )

            extracted_data = data.get("extracted_data", {})
            field_confidence = data.get("field_confidence", {})
            overall_confidence = float(data.get("overall_confidence", 0.5))
            reasoning = data.get("reasoning")

            # Clamp confidence
            overall_confidence = max(0.0, min(1.0, overall_confidence))

            return extracted_data, field_confidence, overall_confidence, reasoning

        except json.JSONDecodeError:
            # Try to extract JSON from the response
            try:
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        extracted_data = data.get("extracted_data", data)
                        field_confidence = data.get("field_confidence", {})
                        overall_confidence = float(data.get("overall_confidence", 0.5))
                        reasoning = data.get("reasoning")
                        return (
                            extracted_data,
                            field_confidence,
                            overall_confidence,
                            reasoning,
                        )
            except json.JSONDecodeError:
                pass

            raise SynthesisError(
                "Failed to parse synthesis response as JSON",
                error_type="parse_error",
                details={"response_preview": response[:500]},
            ) from None

    def _build_field_extractions(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
        field_confidence: dict[str, float],
        source_refs: list[RegionSourceReference],
    ) -> list[FieldExtraction]:
        """Build detailed field extraction objects.

        Args:
            data: Extracted data.
            schema_info: Schema information.
            field_confidence: Confidence scores for each field.
            source_refs: Source references for content.

        Returns:
            List of FieldExtraction objects.
        """
        extractions: list[FieldExtraction] = []

        # Format source refs summary
        source_text = (
            f"Synthesized from {len(source_refs)} regions "
            f"across {len({s.page_number for s in source_refs})} page(s)"
        )

        for field_info in schema_info.all_fields:
            field_path = field_info.path
            value = self._get_nested_value(data, field_path)

            confidence = field_confidence.get(field_path, 0.5)

            extractions.append(
                FieldExtraction(
                    field_path=field_path,
                    value=value,
                    confidence=confidence,
                    source_text=source_text,
                    reasoning="Synthesized from visual document regions",
                )
            )

            # Handle nested fields
            if field_info.nested_fields and isinstance(value, dict):
                for nested_field in field_info.nested_fields:
                    nested_path = nested_field.path
                    nested_value = self._get_nested_value(data, nested_path)
                    nested_confidence = field_confidence.get(nested_path, 0.5)

                    extractions.append(
                        FieldExtraction(
                            field_path=nested_path,
                            value=nested_value,
                            confidence=nested_confidence,
                            source_text=source_text,
                            reasoning="Synthesized from visual document regions",
                        )
                    )

        return extractions

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get a value from nested dictionary using dot notation.

        Args:
            data: Dictionary to get value from.
            path: Dot-separated path (e.g., 'address.city').

        Returns:
            Value at path or None if not found.
        """
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _create_empty_result(
        self,
        schema_info: SchemaInfo,
        region_extraction: DocumentRegionExtractionResult,
    ) -> SynthesisResult:
        """Create an empty synthesis result for documents with no content.

        Args:
            schema_info: Schema information.
            region_extraction: Original extraction result.

        Returns:
            SynthesisResult with empty data.
        """
        # Create empty data structure matching schema
        empty_data: dict[str, Any] = {}
        for f in schema_info.required_fields:
            self._set_nested_value(empty_data, f.path, None)
        for f in schema_info.optional_fields:
            self._set_nested_value(empty_data, f.path, None)

        return SynthesisResult(
            extracted_data=empty_data,
            field_extractions=[],
            source_references=[],
            synthesized_contents=[],
            model_used=self.model,
            total_tokens=region_extraction.total_tokens,
            prompt_tokens=region_extraction.prompt_tokens,
            completion_tokens=region_extraction.completion_tokens,
            processing_time_seconds=0.0,
            total_regions_processed=0,
            total_pages=0,
            synthesis_confidence=0.0,
            pipeline_metadata={"status": "empty_document"},
        )

    def _set_nested_value(
        self,
        data: dict[str, Any],
        path: str,
        value: Any,
    ) -> None:
        """Set a value in nested dictionary using dot notation.

        Args:
            data: Dictionary to set value in.
            path: Dot-separated path.
            value: Value to set.
        """
        keys = path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _build_pipeline_metadata(
        self,
        region_extraction: DocumentRegionExtractionResult,
        reading_order: DocumentReadingOrder,
    ) -> dict[str, Any]:
        """Build metadata about the processing pipeline.

        Args:
            region_extraction: Region extraction result.
            reading_order: Document reading order.

        Returns:
            Dictionary with pipeline metadata.
        """
        # Count regions by type
        region_type_counts: dict[str, int] = {}
        for result in region_extraction.region_results:
            type_name = result.region_type.value
            region_type_counts[type_name] = region_type_counts.get(type_name, 0) + 1

        # Count layout types
        layout_types: list[str] = [p.layout_type for p in reading_order.pages]
        layout_type_counts: dict[str, int] = {}
        for lt in layout_types:
            layout_type_counts[lt] = layout_type_counts.get(lt, 0) + 1

        return {
            "region_extraction": {
                "total_regions": region_extraction.total_regions,
                "successful_extractions": region_extraction.successful_extractions,
                "failed_extractions": region_extraction.failed_extractions,
                "region_types": region_type_counts,
                "model": region_extraction.model_used,
                "tokens": region_extraction.total_tokens,
            },
            "reading_order": {
                "total_pages": reading_order.total_pages,
                "total_regions": reading_order.total_regions,
                "layout_types": layout_type_counts,
                "model": reading_order.model_used,
            },
        }

    def synthesize_from_page_results(
        self,
        page_extractions: list[DocumentRegionExtractionResult],
        page_orders: list[PageReadingOrder],
        schema_info: SchemaInfo,
    ) -> SynthesisResult:
        """Synthesize from separate per-page results.

        This is a convenience method for when extraction is done per-page.

        Args:
            page_extractions: List of extraction results per page.
            page_orders: List of reading orders per page.
            schema_info: Schema to map to.

        Returns:
            SynthesisResult with combined extraction.
        """
        # Combine into document-level structures
        all_region_results: list[RegionExtractionResult] = []
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_time = 0.0

        for extraction in page_extractions:
            all_region_results.extend(extraction.region_results)
            total_tokens += extraction.total_tokens
            prompt_tokens += extraction.prompt_tokens
            completion_tokens += extraction.completion_tokens
            total_time += extraction.processing_time_seconds

        combined_extraction = DocumentRegionExtractionResult(
            region_results=all_region_results,
            total_regions=len(all_region_results),
            successful_extractions=sum(
                e.successful_extractions for e in page_extractions
            ),
            failed_extractions=sum(e.failed_extractions for e in page_extractions),
            model_used=page_extractions[0].model_used if page_extractions else "",
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            processing_time_seconds=total_time,
        )

        combined_order = DocumentReadingOrder(
            pages=page_orders,
            total_pages=len(page_orders),
            total_regions=sum(len(p.ordered_regions) for p in page_orders),
            model_used=page_orders[0].layout_type if page_orders else "",
        )

        return self.synthesize(combined_extraction, combined_order, schema_info)
