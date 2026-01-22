"""Extraction planning agent using LangChain.

This module provides a LangChain agent that creates extraction plans
before processing documents. The agent analyzes schema complexity and
document characteristics to determine optimal extraction strategies.

Key features:
- Analyzes schema complexity and document characteristics
- Determines extraction strategy based on document type (text vs. visual)
- For visual documents: plans region processing order and prioritization
- Identifies potential challenges (complex tables, multi-column, charts)
- Generates step-by-step extraction plan using LLM reasoning
- Estimates confidence and quality thresholds
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agentic_document_extraction.config import settings
from agentic_document_extraction.models import FormatInfo, ProcessingCategory
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionType,
)
from agentic_document_extraction.services.schema_validator import FieldInfo, SchemaInfo

logger = logging.getLogger(__name__)


class PlanningError(Exception):
    """Raised when extraction planning fails."""

    def __init__(
        self,
        message: str,
        error_type: str = "planning_error",
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


class SchemaComplexity(str, Enum):
    """Schema complexity levels."""

    SIMPLE = "simple"  # Few fields, flat structure
    MODERATE = "moderate"  # Multiple fields, some nesting
    COMPLEX = "complex"  # Many fields, deep nesting, arrays of objects


class ExtractionChallenge(str, Enum):
    """Types of extraction challenges."""

    COMPLEX_TABLE = "complex_table"
    MULTI_COLUMN = "multi_column"
    NESTED_LAYOUT = "nested_layout"
    HANDWRITTEN_TEXT = "handwritten_text"
    CHART_GRAPH = "chart_graph"
    MIXED_CONTENT = "mixed_content"
    LOW_QUALITY_IMAGE = "low_quality_image"
    DENSE_TEXT = "dense_text"
    SPARSE_CONTENT = "sparse_content"
    MULTILINGUAL = "multilingual"
    LARGE_DOCUMENT = "large_document"


@dataclass
class RegionPriority:
    """Priority assignment for a document region."""

    region_type: str
    """Type of the region (e.g., 'table', 'text', 'title')."""

    priority: int
    """Priority level (1 = highest, larger numbers = lower priority)."""

    reason: str
    """Reason for this priority assignment."""

    schema_fields: list[str] = field(default_factory=list)
    """Schema fields that this region is likely to contain."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with region priority information.
        """
        return {
            "region_type": self.region_type,
            "priority": self.priority,
            "reason": self.reason,
            "schema_fields": self.schema_fields,
        }


@dataclass
class ExtractionStep:
    """A single step in the extraction plan."""

    step_number: int
    """Sequential step number."""

    action: str
    """Description of the action to take."""

    target: str
    """Target of the action (e.g., field name, region type)."""

    strategy: str
    """Strategy to use for this step."""

    expected_output: str
    """Expected output from this step."""

    fallback: str | None = None
    """Fallback strategy if primary approach fails."""

    depends_on: list[int] | None = None
    """Step numbers this step depends on."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with extraction step information.
        """
        return {
            "step_number": self.step_number,
            "action": self.action,
            "target": self.target,
            "strategy": self.strategy,
            "expected_output": self.expected_output,
            "fallback": self.fallback,
            "depends_on": self.depends_on,
        }


@dataclass
class QualityThreshold:
    """Quality thresholds for extraction verification."""

    min_overall_confidence: float
    """Minimum overall confidence score (0.0-1.0)."""

    min_field_confidence: float
    """Minimum confidence for individual fields (0.0-1.0)."""

    required_field_coverage: float
    """Percentage of required fields that must be extracted (0.0-1.0)."""

    max_iterations: int
    """Maximum refinement iterations allowed."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with quality threshold information.
        """
        return {
            "min_overall_confidence": self.min_overall_confidence,
            "min_field_confidence": self.min_field_confidence,
            "required_field_coverage": self.required_field_coverage,
            "max_iterations": self.max_iterations,
        }


@dataclass
class DocumentCharacteristics:
    """Analyzed characteristics of a document."""

    processing_category: str
    """Processing category (text_based or visual)."""

    format_family: str
    """Document format family (pdf, image, etc.)."""

    estimated_complexity: str
    """Estimated document complexity (simple, moderate, complex)."""

    page_count: int | None = None
    """Number of pages (if applicable)."""

    has_tables: bool = False
    """Whether document likely contains tables."""

    has_images: bool = False
    """Whether document contains images/figures."""

    has_charts: bool = False
    """Whether document contains charts/graphs."""

    is_multi_column: bool = False
    """Whether document has multi-column layout."""

    text_density: str = "normal"
    """Text density (sparse, normal, dense)."""

    region_count: int = 0
    """Number of detected layout regions."""

    region_types: list[str] = field(default_factory=list)
    """Types of regions detected."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with document characteristics.
        """
        return {
            "processing_category": self.processing_category,
            "format_family": self.format_family,
            "estimated_complexity": self.estimated_complexity,
            "page_count": self.page_count,
            "has_tables": self.has_tables,
            "has_images": self.has_images,
            "has_charts": self.has_charts,
            "is_multi_column": self.is_multi_column,
            "text_density": self.text_density,
            "region_count": self.region_count,
            "region_types": self.region_types,
        }


@dataclass
class ExtractionPlan:
    """Complete extraction plan for a document."""

    document_characteristics: DocumentCharacteristics
    """Analyzed document characteristics."""

    schema_complexity: SchemaComplexity
    """Complexity level of the target schema."""

    extraction_strategy: str
    """Overall extraction strategy (text_direct, visual_ocr, visual_vlm, etc.)."""

    steps: list[ExtractionStep]
    """Ordered list of extraction steps."""

    region_priorities: list[RegionPriority]
    """Priority assignments for document regions (visual docs only)."""

    challenges: list[ExtractionChallenge]
    """Identified extraction challenges."""

    quality_thresholds: QualityThreshold
    """Quality thresholds for verification."""

    reasoning: str
    """LLM reasoning for this plan."""

    estimated_confidence: float
    """Estimated confidence for successful extraction (0.0-1.0)."""

    model_used: str = ""
    """Model used for planning."""

    total_tokens: int = 0
    """Total tokens used in planning."""

    prompt_tokens: int = 0
    """Prompt tokens used."""

    completion_tokens: int = 0
    """Completion tokens used."""

    processing_time_seconds: float = 0.0
    """Time taken for planning."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with complete plan information.
        """
        return {
            "document_characteristics": self.document_characteristics.to_dict(),
            "schema_complexity": self.schema_complexity.value,
            "extraction_strategy": self.extraction_strategy,
            "steps": [s.to_dict() for s in self.steps],
            "region_priorities": [r.to_dict() for r in self.region_priorities],
            "challenges": [c.value for c in self.challenges],
            "quality_thresholds": self.quality_thresholds.to_dict(),
            "reasoning": self.reasoning,
            "estimated_confidence": self.estimated_confidence,
            "metadata": {
                "model_used": self.model_used,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "processing_time_seconds": self.processing_time_seconds,
            },
        }


class ExtractionPlanningAgent:
    """Agent for creating extraction plans using LangChain.

    Uses GPT-4 with reasoning capabilities to analyze document characteristics
    and schema complexity, then generates an optimal extraction plan with
    step-by-step instructions and quality thresholds.
    """

    PLANNING_SYSTEM_PROMPT = """You are an expert document extraction planning assistant.
Your task is to analyze document characteristics and JSON schemas to create optimal extraction plans.

IMPORTANT RESPONSIBILITIES:
1. Analyze schema complexity (number of fields, nesting depth, required vs optional)
2. Evaluate document characteristics (format, layout complexity, content types)
3. Identify potential extraction challenges
4. Determine the best extraction strategy
5. Create step-by-step extraction plan
6. Set appropriate quality thresholds based on complexity
7. For visual documents: prioritize regions based on their likely contribution to schema fields

EXTRACTION STRATEGIES:
- text_direct: Direct text extraction for simple text documents
- text_chunked: Chunked extraction for large text documents
- visual_ocr_simple: OCR-based extraction for simple visual documents
- visual_vlm: Full VLM pipeline for complex visual documents with tables/charts
- visual_vlm_enhanced: Enhanced VLM with multiple passes for challenging documents

You must respond with ONLY valid JSON matching this structure:
{{
    "schema_complexity": "simple" | "moderate" | "complex",
    "extraction_strategy": "strategy_name",
    "steps": [
        {{
            "step_number": 1,
            "action": "description of action",
            "target": "what is being targeted",
            "strategy": "specific strategy for this step",
            "expected_output": "what to expect",
            "fallback": "fallback approach if needed",
            "depends_on": [step_numbers] or null
        }}
    ],
    "region_priorities": [
        {{
            "region_type": "type",
            "priority": 1-10,
            "reason": "why this priority",
            "schema_fields": ["field1", "field2"]
        }}
    ],
    "challenges": ["challenge_type1", "challenge_type2"],
    "quality_thresholds": {{
        "min_overall_confidence": 0.0-1.0,
        "min_field_confidence": 0.0-1.0,
        "required_field_coverage": 0.0-1.0,
        "max_iterations": 1-5
    }},
    "reasoning": "Detailed explanation of your planning decisions",
    "estimated_confidence": 0.0-1.0
}}"""

    PLANNING_USER_PROMPT = """Create an extraction plan for the following document and schema.

## Document Characteristics:
- Processing Category: {processing_category}
- Format Family: {format_family}
- Page Count: {page_count}
- Region Count: {region_count}
- Detected Region Types: {region_types}

## Document Content Summary:
{content_summary}

## Target JSON Schema:
```json
{schema}
```

## Required Fields:
{required_fields}

## Optional Fields:
{optional_fields}

## Additional Context:
- Total fields: {total_fields}
- Nesting depth: {nesting_depth}
- Contains arrays: {has_arrays}

Create a detailed extraction plan that will maximize extraction quality and efficiency.
Consider all potential challenges and provide appropriate fallback strategies."""

    # Region types that indicate content complexity
    COMPLEX_REGION_TYPES = {RegionType.TABLE, RegionType.PICTURE, RegionType.FORMULA}
    STRUCTURAL_REGION_TYPES = {
        RegionType.TITLE,
        RegionType.SECTION_HEADER,
        RegionType.PAGE_HEADER,
        RegionType.PAGE_FOOTER,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the planning agent.

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

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LangChain ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            PlanningError: If API key is not configured.
        """
        if self._llm is None:
            if not self.api_key:
                raise PlanningError(
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

    def create_plan(
        self,
        schema_info: SchemaInfo,
        format_info: FormatInfo,
        content_summary: str | None = None,
        layout_regions: list[LayoutRegion] | None = None,
        page_count: int | None = None,
    ) -> ExtractionPlan:
        """Create an extraction plan for a document.

        Args:
            schema_info: Validated schema information.
            format_info: Detected document format information.
            content_summary: Optional summary of document content.
            layout_regions: Optional list of detected layout regions (for visual docs).
            page_count: Optional number of pages.

        Returns:
            ExtractionPlan with complete extraction strategy.

        Raises:
            PlanningError: If planning fails.
        """
        start_time = time.time()

        # Analyze document characteristics
        doc_characteristics = self._analyze_document(
            format_info, layout_regions, page_count
        )

        # Analyze schema complexity
        schema_complexity = self._analyze_schema_complexity(schema_info)

        # For simple text documents with simple schemas, use fast path
        if self._can_use_fast_path(doc_characteristics, schema_complexity):
            plan = self._create_fast_plan(
                doc_characteristics, schema_complexity, schema_info
            )
            plan.processing_time_seconds = time.time() - start_time
            logger.info(
                f"Fast planning completed: "
                f"strategy={plan.extraction_strategy}, "
                f"time={plan.processing_time_seconds:.2f}s"
            )
            return plan

        # Use LLM for complex planning
        try:
            plan = self._create_llm_plan(
                doc_characteristics,
                schema_complexity,
                schema_info,
                content_summary,
                layout_regions,
            )
            plan.processing_time_seconds = time.time() - start_time

            logger.info(
                f"Planning completed: "
                f"strategy={plan.extraction_strategy}, "
                f"steps={len(plan.steps)}, "
                f"challenges={len(plan.challenges)}, "
                f"tokens={plan.total_tokens}, "
                f"time={plan.processing_time_seconds:.2f}s"
            )

            return plan

        except Exception as e:
            if isinstance(e, PlanningError):
                raise
            raise PlanningError(
                f"Planning failed: {e}",
                error_type="llm_error",
                details={"original_error": str(e)},
            ) from e

    def _analyze_document(
        self,
        format_info: FormatInfo,
        layout_regions: list[LayoutRegion] | None,
        page_count: int | None,
    ) -> DocumentCharacteristics:
        """Analyze document characteristics.

        Args:
            format_info: Document format information.
            layout_regions: Detected layout regions.
            page_count: Number of pages.

        Returns:
            DocumentCharacteristics with analysis results.
        """
        region_types: list[str] = []
        has_tables = False
        has_images = False
        has_charts = False
        is_multi_column = False
        text_density = "normal"

        if layout_regions:
            region_types = list({r.region_type.value for r in layout_regions})

            # Check for specific content types
            type_set = {r.region_type for r in layout_regions}
            has_tables = RegionType.TABLE in type_set
            has_images = RegionType.PICTURE in type_set
            has_charts = has_images  # Charts often detected as pictures

            # Estimate layout complexity from region positions
            is_multi_column = self._detect_multi_column(layout_regions)

            # Estimate text density
            text_density = self._estimate_text_density(layout_regions)

        # Determine complexity
        complexity = "simple"
        if format_info.processing_category == ProcessingCategory.VISUAL:
            if has_tables or has_charts or is_multi_column:
                complexity = "complex"
            elif len(region_types) > 3:
                complexity = "moderate"
        elif page_count and page_count > 10:
            complexity = "moderate"

        return DocumentCharacteristics(
            processing_category=format_info.processing_category.value,
            format_family=format_info.format_family.value,
            estimated_complexity=complexity,
            page_count=page_count,
            has_tables=has_tables,
            has_images=has_images,
            has_charts=has_charts,
            is_multi_column=is_multi_column,
            text_density=text_density,
            region_count=len(layout_regions) if layout_regions else 0,
            region_types=region_types,
        )

    def _detect_multi_column(self, regions: list[LayoutRegion]) -> bool:
        """Detect if document has multi-column layout.

        Uses a simple heuristic: check if there are multiple text regions
        at similar vertical positions with non-overlapping horizontal positions.

        Args:
            regions: List of layout regions.

        Returns:
            True if multi-column layout detected.
        """
        text_regions = [
            r for r in regions if r.region_type in (RegionType.TEXT, RegionType.TITLE)
        ]

        if len(text_regions) < 2:
            return False

        # Group regions by approximate vertical position
        vertical_groups: dict[int, list[LayoutRegion]] = {}
        tolerance = 50  # Pixel tolerance for grouping

        for region in text_regions:
            y_center = (region.bbox.y0 + region.bbox.y1) / 2
            group_key = int(y_center / tolerance)

            if group_key not in vertical_groups:
                vertical_groups[group_key] = []
            vertical_groups[group_key].append(region)

        # Check if any group has multiple non-overlapping horizontal regions
        for group in vertical_groups.values():
            if len(group) < 2:
                continue

            # Sort by x position
            sorted_group = sorted(group, key=lambda r: r.bbox.x0)

            # Check for non-overlapping
            for i in range(len(sorted_group) - 1):
                if sorted_group[i].bbox.x1 < sorted_group[i + 1].bbox.x0:
                    return True

        return False

    def _estimate_text_density(self, regions: list[LayoutRegion]) -> str:
        """Estimate text density from layout regions.

        Args:
            regions: List of layout regions.

        Returns:
            Text density: "sparse", "normal", or "dense".
        """
        text_regions = [
            r
            for r in regions
            if r.region_type
            in (RegionType.TEXT, RegionType.LIST_ITEM, RegionType.SECTION_HEADER)
        ]

        if not text_regions:
            return "sparse"

        # Calculate text region coverage
        total_text_area = sum(
            (r.bbox.x1 - r.bbox.x0) * (r.bbox.y1 - r.bbox.y0) for r in text_regions
        )

        # Estimate page area (use bounding box of all regions)
        if regions:
            page_width = max(r.bbox.x1 for r in regions) - min(
                r.bbox.x0 for r in regions
            )
            page_height = max(r.bbox.y1 for r in regions) - min(
                r.bbox.y0 for r in regions
            )
            page_area = page_width * page_height
        else:
            return "normal"

        coverage = total_text_area / page_area if page_area > 0 else 0

        if coverage < 0.3:
            return "sparse"
        elif coverage > 0.7:
            return "dense"
        return "normal"

    def _analyze_schema_complexity(self, schema_info: SchemaInfo) -> SchemaComplexity:
        """Analyze schema complexity.

        Args:
            schema_info: Schema information.

        Returns:
            SchemaComplexity level.
        """
        total_fields = len(schema_info.all_fields)
        max_depth = self._calculate_nesting_depth(schema_info.schema)
        has_arrays = self._schema_has_arrays(schema_info.schema)

        # Simple: <= 5 fields, no nesting, no arrays
        if total_fields <= 5 and max_depth <= 1 and not has_arrays:
            return SchemaComplexity.SIMPLE

        # Complex: > 15 fields OR deep nesting OR arrays of objects
        if (
            total_fields > 15
            or max_depth > 3
            or self._has_array_of_objects(schema_info)
        ):
            return SchemaComplexity.COMPLEX

        return SchemaComplexity.MODERATE

    def _calculate_nesting_depth(
        self, schema: dict[str, Any], current_depth: int = 0
    ) -> int:
        """Calculate maximum nesting depth of a schema.

        Args:
            schema: Schema or sub-schema to analyze.
            current_depth: Current nesting depth.

        Returns:
            Maximum nesting depth.
        """
        if not isinstance(schema, dict):
            return current_depth

        max_depth = current_depth

        # Check properties
        if "properties" in schema:
            for prop_schema in schema["properties"].values():
                if isinstance(prop_schema, dict):
                    depth = self._calculate_nesting_depth(
                        prop_schema, current_depth + 1
                    )
                    max_depth = max(max_depth, depth)

        # Check array items
        if "items" in schema and isinstance(schema["items"], dict):
            depth = self._calculate_nesting_depth(schema["items"], current_depth + 1)
            max_depth = max(max_depth, depth)

        return max_depth

    def _schema_has_arrays(self, schema: dict[str, Any]) -> bool:
        """Check if schema contains array types.

        Args:
            schema: Schema to check.

        Returns:
            True if schema contains arrays.
        """
        if not isinstance(schema, dict):
            return False

        if schema.get("type") == "array":
            return True

        if "properties" in schema:
            for prop_schema in schema["properties"].values():
                if self._schema_has_arrays(prop_schema):
                    return True

        return "items" in schema and self._schema_has_arrays(schema["items"])

    def _has_array_of_objects(self, schema_info: SchemaInfo) -> bool:
        """Check if schema has arrays of objects.

        Args:
            schema_info: Schema information.

        Returns:
            True if schema contains arrays of objects.
        """
        for field_info in schema_info.all_fields:
            if field_info.field_type == "array" and field_info.nested_fields:
                return True
        return False

    def _can_use_fast_path(
        self, doc_chars: DocumentCharacteristics, schema_complexity: SchemaComplexity
    ) -> bool:
        """Check if fast planning path can be used.

        Args:
            doc_chars: Document characteristics.
            schema_complexity: Schema complexity.

        Returns:
            True if fast path can be used.
        """
        # Fast path for simple text documents with simple/moderate schemas
        return (
            doc_chars.processing_category == ProcessingCategory.TEXT_BASED.value
            and schema_complexity
            in (SchemaComplexity.SIMPLE, SchemaComplexity.MODERATE)
        )

    def _create_fast_plan(
        self,
        doc_chars: DocumentCharacteristics,
        schema_complexity: SchemaComplexity,
        schema_info: SchemaInfo,  # noqa: ARG002
    ) -> ExtractionPlan:
        """Create a plan without LLM for simple cases.

        Args:
            doc_chars: Document characteristics.
            schema_complexity: Schema complexity.
            schema_info: Schema information (reserved for future use).

        Returns:
            ExtractionPlan for simple extraction.
        """
        # Determine strategy
        strategy = "text_direct"
        if doc_chars.page_count and doc_chars.page_count > 5:
            strategy = "text_chunked"

        # Create simple extraction steps
        steps = [
            ExtractionStep(
                step_number=1,
                action="Extract text content",
                target="full_document",
                strategy=strategy,
                expected_output="Raw text content",
                fallback=None,
                depends_on=None,
            ),
            ExtractionStep(
                step_number=2,
                action="Apply schema extraction",
                target="extracted_text",
                strategy="llm_structured_output",
                expected_output="Structured data matching schema",
                fallback="retry_with_simplified_prompt",
                depends_on=[1],
            ),
            ExtractionStep(
                step_number=3,
                action="Validate against schema",
                target="extracted_data",
                strategy="jsonschema_validation",
                expected_output="Validated extraction result",
                fallback="return_partial_result",
                depends_on=[2],
            ),
        ]

        # Set quality thresholds based on complexity
        if schema_complexity == SchemaComplexity.SIMPLE:
            thresholds = QualityThreshold(
                min_overall_confidence=0.8,
                min_field_confidence=0.7,
                required_field_coverage=0.95,
                max_iterations=2,
            )
        else:
            thresholds = QualityThreshold(
                min_overall_confidence=0.7,
                min_field_confidence=0.6,
                required_field_coverage=0.9,
                max_iterations=3,
            )

        return ExtractionPlan(
            document_characteristics=doc_chars,
            schema_complexity=schema_complexity,
            extraction_strategy=strategy,
            steps=steps,
            region_priorities=[],
            challenges=[],
            quality_thresholds=thresholds,
            reasoning=f"Fast path planning for {doc_chars.processing_category} document "
            f"with {schema_complexity.value} schema complexity.",
            estimated_confidence=0.85
            if schema_complexity == SchemaComplexity.SIMPLE
            else 0.75,
            model_used="fast_path",
        )

    def _create_llm_plan(
        self,
        doc_chars: DocumentCharacteristics,
        schema_complexity: SchemaComplexity,
        schema_info: SchemaInfo,
        content_summary: str | None,
        layout_regions: list[LayoutRegion] | None,
    ) -> ExtractionPlan:
        """Create extraction plan using LLM reasoning.

        Args:
            doc_chars: Document characteristics.
            schema_complexity: Schema complexity (pre-analyzed).
            schema_info: Schema information.
            content_summary: Optional content summary.
            layout_regions: Optional layout regions.

        Returns:
            ExtractionPlan with LLM-generated strategy.

        Raises:
            PlanningError: If LLM planning fails.
        """
        # Build prompt
        prompt = self._build_planning_prompt(
            doc_chars, schema_info, content_summary, layout_regions
        )

        # Invoke LLM
        messages = prompt.format_messages()
        response = self.llm.invoke(messages)

        # Parse response
        content = response.content
        if not isinstance(content, str):
            content = str(content)

        plan_data = self._parse_plan_response(content)

        # Extract token usage
        usage_metadata = getattr(response, "usage_metadata", None) or {}
        prompt_tokens = usage_metadata.get("input_tokens", 0)
        completion_tokens = usage_metadata.get("output_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Build extraction steps
        steps = self._build_steps_from_response(plan_data.get("steps", []))

        # Build region priorities
        region_priorities = self._build_region_priorities(
            plan_data.get("region_priorities", [])
        )

        # Parse challenges
        challenges = self._parse_challenges(plan_data.get("challenges", []))

        # Build quality thresholds
        thresholds = self._build_quality_thresholds(
            plan_data.get("quality_thresholds", {})
        )

        # Get strategy and confidence
        strategy = plan_data.get("extraction_strategy", "visual_vlm")
        reasoning = plan_data.get("reasoning", "LLM planning completed.")
        estimated_confidence = float(plan_data.get("estimated_confidence", 0.7))

        # Override schema complexity if LLM suggests differently
        llm_complexity_str = plan_data.get("schema_complexity", "")
        if llm_complexity_str:
            try:
                llm_complexity = SchemaComplexity(llm_complexity_str)
            except ValueError:
                llm_complexity = schema_complexity
        else:
            llm_complexity = schema_complexity

        return ExtractionPlan(
            document_characteristics=doc_chars,
            schema_complexity=llm_complexity,
            extraction_strategy=strategy,
            steps=steps,
            region_priorities=region_priorities,
            challenges=challenges,
            quality_thresholds=thresholds,
            reasoning=reasoning,
            estimated_confidence=max(0.0, min(1.0, estimated_confidence)),
            model_used=self.model,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _build_planning_prompt(
        self,
        doc_chars: DocumentCharacteristics,
        schema_info: SchemaInfo,
        content_summary: str | None,
        layout_regions: list[LayoutRegion] | None,  # noqa: ARG002
    ) -> ChatPromptTemplate:
        """Build the planning prompt.

        Args:
            doc_chars: Document characteristics.
            schema_info: Schema information.
            content_summary: Optional content summary.
            layout_regions: Optional layout regions (reserved for future use).

        Returns:
            Configured ChatPromptTemplate.
        """
        # Format fields
        required_fields = self._format_fields(schema_info.required_fields)
        optional_fields = self._format_fields(schema_info.optional_fields)

        # Calculate schema metrics
        nesting_depth = self._calculate_nesting_depth(schema_info.schema)
        has_arrays = self._schema_has_arrays(schema_info.schema)

        # Format region types
        region_types_str = (
            ", ".join(doc_chars.region_types) if doc_chars.region_types else "none"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.PLANNING_SYSTEM_PROMPT),
                ("human", self.PLANNING_USER_PROMPT),
            ]
        )

        return prompt.partial(
            processing_category=doc_chars.processing_category,
            format_family=doc_chars.format_family,
            page_count=str(doc_chars.page_count or "unknown"),
            region_count=str(doc_chars.region_count),
            region_types=region_types_str,
            content_summary=content_summary or "No content summary available.",
            schema=json.dumps(schema_info.schema, indent=2),
            required_fields=required_fields,
            optional_fields=optional_fields,
            total_fields=str(len(schema_info.all_fields)),
            nesting_depth=str(nesting_depth),
            has_arrays=str(has_arrays),
        )

    def _format_fields(self, fields: list[FieldInfo]) -> str:
        """Format field information for the prompt.

        Args:
            fields: List of field info objects.

        Returns:
            Formatted string of field information.
        """
        if not fields:
            return "None"

        return "\n".join(
            f"- {f.path}: {f.field_type}"
            + (f" - {f.description}" if f.description else "")
            for f in fields
        )

    def _parse_plan_response(self, response: str) -> dict[str, Any]:
        """Parse the planning response from LLM.

        Args:
            response: Raw response string.

        Returns:
            Parsed plan data.

        Raises:
            PlanningError: If parsing fails.
        """
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                raise PlanningError(
                    "Expected JSON object response",
                    error_type="parse_error",
                    details={"received_type": type(data).__name__},
                )
            return data
        except json.JSONDecodeError:
            # Try to extract JSON from response
            try:
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        return data
            except json.JSONDecodeError:
                pass

            raise PlanningError(
                "Failed to parse planning response as JSON",
                error_type="parse_error",
                details={"response_preview": response[:500]},
            ) from None

    def _build_steps_from_response(
        self, steps_data: list[dict[str, Any]]
    ) -> list[ExtractionStep]:
        """Build ExtractionStep objects from response data.

        Args:
            steps_data: List of step dictionaries from LLM.

        Returns:
            List of ExtractionStep objects.
        """
        steps: list[ExtractionStep] = []

        for step_dict in steps_data:
            depends_on = step_dict.get("depends_on")
            if depends_on is not None and not isinstance(depends_on, list):
                depends_on = [depends_on] if depends_on else None

            step = ExtractionStep(
                step_number=int(step_dict.get("step_number", len(steps) + 1)),
                action=str(step_dict.get("action", "Unknown action")),
                target=str(step_dict.get("target", "Unknown target")),
                strategy=str(step_dict.get("strategy", "default")),
                expected_output=str(step_dict.get("expected_output", "")),
                fallback=step_dict.get("fallback"),
                depends_on=depends_on,
            )
            steps.append(step)

        return steps

    def _build_region_priorities(
        self, priorities_data: list[dict[str, Any]]
    ) -> list[RegionPriority]:
        """Build RegionPriority objects from response data.

        Args:
            priorities_data: List of priority dictionaries from LLM.

        Returns:
            List of RegionPriority objects.
        """
        priorities: list[RegionPriority] = []

        for priority_dict in priorities_data:
            schema_fields = priority_dict.get("schema_fields", [])
            if not isinstance(schema_fields, list):
                schema_fields = []

            priority = RegionPriority(
                region_type=str(priority_dict.get("region_type", "unknown")),
                priority=int(priority_dict.get("priority", 5)),
                reason=str(priority_dict.get("reason", "")),
                schema_fields=schema_fields,
            )
            priorities.append(priority)

        return priorities

    def _parse_challenges(
        self, challenges_data: list[str]
    ) -> list[ExtractionChallenge]:
        """Parse challenge strings into ExtractionChallenge enums.

        Args:
            challenges_data: List of challenge strings from LLM.

        Returns:
            List of ExtractionChallenge enums.
        """
        challenges: list[ExtractionChallenge] = []

        for challenge_str in challenges_data:
            try:
                challenge = ExtractionChallenge(challenge_str)
                challenges.append(challenge)
            except ValueError:
                # Try to map common variations
                mapping = {
                    "table": ExtractionChallenge.COMPLEX_TABLE,
                    "tables": ExtractionChallenge.COMPLEX_TABLE,
                    "multi-column": ExtractionChallenge.MULTI_COLUMN,
                    "multicolumn": ExtractionChallenge.MULTI_COLUMN,
                    "chart": ExtractionChallenge.CHART_GRAPH,
                    "charts": ExtractionChallenge.CHART_GRAPH,
                    "graph": ExtractionChallenge.CHART_GRAPH,
                    "image": ExtractionChallenge.LOW_QUALITY_IMAGE,
                    "nested": ExtractionChallenge.NESTED_LAYOUT,
                    "large": ExtractionChallenge.LARGE_DOCUMENT,
                    "dense": ExtractionChallenge.DENSE_TEXT,
                }
                normalized = challenge_str.lower().replace("_", "-")
                if normalized in mapping:
                    challenges.append(mapping[normalized])
                else:
                    logger.warning(f"Unknown challenge type: {challenge_str}")

        return challenges

    def _build_quality_thresholds(
        self, thresholds_data: dict[str, Any]
    ) -> QualityThreshold:
        """Build QualityThreshold from response data.

        Args:
            thresholds_data: Threshold dictionary from LLM.

        Returns:
            QualityThreshold object with reasonable defaults.
        """
        return QualityThreshold(
            min_overall_confidence=float(
                thresholds_data.get("min_overall_confidence", 0.7)
            ),
            min_field_confidence=float(
                thresholds_data.get("min_field_confidence", 0.6)
            ),
            required_field_coverage=float(
                thresholds_data.get("required_field_coverage", 0.9)
            ),
            max_iterations=int(thresholds_data.get("max_iterations", 3)),
        )
