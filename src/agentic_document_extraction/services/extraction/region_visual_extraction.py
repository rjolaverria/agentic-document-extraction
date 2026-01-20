"""Region-based visual extraction using Vision Language Models (VLM).

This module provides functionality to extract structured information from
individual layout regions of visual documents using LangChain with OpenAI
GPT-4V (Vision) model.

Key features:
- Sends segmented region images to VLM for analysis
- Provides context for each region (type, position, surrounding regions)
- Extracts structured information from tables, charts, figures
- Handles text-heavy regions with combined OCR + VLM analysis
- Returns extraction results per region with confidence scores
- Supports batch processing for efficiency
"""

import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image

from agentic_document_extraction.config import settings
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionType,
)
from agentic_document_extraction.services.reading_order_detector import OrderedRegion

logger = logging.getLogger(__name__)


class RegionVisualExtractionError(Exception):
    """Raised when region visual extraction fails."""

    def __init__(
        self,
        message: str,
        region_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with message and optional context.

        Args:
            message: Error message.
            region_id: Optional ID of the region that failed.
            details: Optional additional error details.
        """
        super().__init__(message)
        self.region_id = region_id
        self.details = details or {}


class ExtractionStrategy(str, Enum):
    """Strategy for extracting information from a region."""

    VLM_ONLY = "vlm_only"
    """Use only the Vision Language Model."""

    OCR_ENHANCED = "ocr_enhanced"
    """Combine OCR text with VLM analysis for text-heavy regions."""

    TABLE_SPECIALIZED = "table_specialized"
    """Use specialized table extraction prompts."""

    FIGURE_ANALYSIS = "figure_analysis"
    """Analyze figures, charts, and diagrams."""


# Map region types to appropriate extraction strategies
REGION_TYPE_STRATEGIES: dict[RegionType, ExtractionStrategy] = {
    RegionType.TEXT: ExtractionStrategy.OCR_ENHANCED,
    RegionType.TITLE: ExtractionStrategy.OCR_ENHANCED,
    RegionType.SECTION_HEADER: ExtractionStrategy.OCR_ENHANCED,
    RegionType.LIST_ITEM: ExtractionStrategy.OCR_ENHANCED,
    RegionType.CAPTION: ExtractionStrategy.OCR_ENHANCED,
    RegionType.FOOTNOTE: ExtractionStrategy.OCR_ENHANCED,
    RegionType.PAGE_HEADER: ExtractionStrategy.OCR_ENHANCED,
    RegionType.PAGE_FOOTER: ExtractionStrategy.OCR_ENHANCED,
    RegionType.TABLE: ExtractionStrategy.TABLE_SPECIALIZED,
    RegionType.PICTURE: ExtractionStrategy.FIGURE_ANALYSIS,
    RegionType.FORMULA: ExtractionStrategy.FIGURE_ANALYSIS,
    RegionType.UNKNOWN: ExtractionStrategy.VLM_ONLY,
}


@dataclass
class RegionContext:
    """Context information about a region and its surroundings.

    Provides spatial and semantic context to help the VLM understand
    the region within the document structure.
    """

    region_type: RegionType
    """Type of the region being analyzed."""

    page_number: int
    """Page number where the region appears."""

    page_dimensions: tuple[float, float]
    """Page dimensions (width, height) in pixels."""

    bbox_normalized: tuple[float, float, float, float]
    """Normalized bounding box coordinates (x0, y0, x1, y1) as percentages."""

    reading_order_position: int | None = None
    """Position in the reading order sequence (0-indexed)."""

    total_regions_on_page: int = 0
    """Total number of regions on this page."""

    preceding_region_types: list[str] = field(default_factory=list)
    """Types of regions that come before this one in reading order."""

    following_region_types: list[str] = field(default_factory=list)
    """Types of regions that come after this one in reading order."""

    parent_region_type: str | None = None
    """Type of parent region if this is a nested region."""

    ocr_text: str | None = None
    """OCR-extracted text from this region, if available."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with context information.
        """
        return {
            "region_type": self.region_type.value,
            "page_number": self.page_number,
            "page_dimensions": {
                "width": self.page_dimensions[0],
                "height": self.page_dimensions[1],
            },
            "bbox_normalized": {
                "x0": self.bbox_normalized[0],
                "y0": self.bbox_normalized[1],
                "x1": self.bbox_normalized[2],
                "y1": self.bbox_normalized[3],
            },
            "reading_order_position": self.reading_order_position,
            "total_regions_on_page": self.total_regions_on_page,
            "preceding_region_types": self.preceding_region_types,
            "following_region_types": self.following_region_types,
            "parent_region_type": self.parent_region_type,
            "has_ocr_text": self.ocr_text is not None,
        }


@dataclass
class RegionExtractionResult:
    """Extraction result for a single region."""

    region_id: str
    """Unique identifier of the region."""

    region_type: RegionType
    """Type of the region."""

    extracted_content: dict[str, Any]
    """Extracted content from the region."""

    confidence: float
    """Overall confidence score (0.0-1.0) for this extraction."""

    strategy_used: ExtractionStrategy
    """Extraction strategy that was used."""

    reasoning: str | None = None
    """Model's reasoning about the extraction."""

    raw_response: str | None = None
    """Raw response from the VLM."""

    processing_time_seconds: float = 0.0
    """Time taken to process this region."""

    prompt_tokens: int = 0
    """Tokens used in the prompt."""

    completion_tokens: int = 0
    """Tokens used in the completion."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with extraction result information.
        """
        return {
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "extracted_content": self.extracted_content,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used.value,
            "reasoning": self.reasoning,
            "processing_time_seconds": self.processing_time_seconds,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            },
        }


@dataclass
class DocumentRegionExtractionResult:
    """Combined extraction results for all regions in a document."""

    region_results: list[RegionExtractionResult]
    """Extraction results for each region."""

    total_regions: int
    """Total number of regions processed."""

    successful_extractions: int
    """Number of successful extractions."""

    failed_extractions: int
    """Number of failed extractions."""

    model_used: str
    """Name of the VLM model used."""

    total_tokens: int = 0
    """Total tokens used across all extractions."""

    prompt_tokens: int = 0
    """Total prompt tokens used."""

    completion_tokens: int = 0
    """Total completion tokens used."""

    processing_time_seconds: float = 0.0
    """Total time taken for all extractions."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the extraction process."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with document extraction result information.
        """
        return {
            "region_results": [r.to_dict() for r in self.region_results],
            "summary": {
                "total_regions": self.total_regions,
                "successful_extractions": self.successful_extractions,
                "failed_extractions": self.failed_extractions,
            },
            "metadata": {
                "model_used": self.model_used,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "processing_time_seconds": self.processing_time_seconds,
                **self.metadata,
            },
        }

    def get_results_by_type(
        self, region_type: RegionType
    ) -> list[RegionExtractionResult]:
        """Get extraction results for a specific region type.

        Args:
            region_type: The type of regions to retrieve results for.

        Returns:
            List of extraction results matching the region type.
        """
        return [r for r in self.region_results if r.region_type == region_type]

    def get_result_by_id(self, region_id: str) -> RegionExtractionResult | None:
        """Get extraction result for a specific region.

        Args:
            region_id: The ID of the region.

        Returns:
            Extraction result or None if not found.
        """
        for result in self.region_results:
            if result.region_id == region_id:
                return result
        return None


class RegionVisualExtractor:
    """Extracts structured information from document regions using VLM.

    Uses LangChain with OpenAI GPT-4V (Vision) to analyze cropped region
    images and extract structured information. Provides context about
    each region's position and type to improve extraction quality.
    """

    # System prompts for different extraction strategies
    SYSTEM_PROMPT_BASE = """You are an expert document analyst with vision capabilities.
Your task is to extract structured information from document regions.

IMPORTANT RULES:
1. Analyze the image carefully and extract all relevant information
2. Maintain the exact structure and formatting of the content
3. Provide confidence scores for your extractions
4. If information is unclear or partially visible, indicate uncertainty
5. Return your response as valid JSON

Your response MUST be valid JSON with this structure:
{
  "extracted_content": { ... },
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of your analysis"
}"""

    SYSTEM_PROMPT_TABLE = """You are an expert at analyzing and extracting data from tables.
Your task is to extract structured tabular data from the provided image.

IMPORTANT RULES:
1. Identify all rows and columns in the table
2. Preserve the table structure in your extraction
3. Handle merged cells appropriately
4. Extract headers separately from data rows
5. Note any special formatting (bold, italics, etc.)
6. If cell content is unclear, indicate with [unclear] marker

Your response MUST be valid JSON with this structure:
{
  "extracted_content": {
    "headers": ["col1", "col2", ...],
    "rows": [["cell1", "cell2", ...], ...],
    "has_header_row": true/false,
    "total_rows": number,
    "total_columns": number,
    "notes": "any observations about the table"
  },
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of your analysis"
}"""

    SYSTEM_PROMPT_FIGURE = """You are an expert at analyzing visual content including charts, graphs, diagrams, and images.
Your task is to describe and extract information from the provided visual element.

IMPORTANT RULES:
1. Identify the type of visual (chart, graph, diagram, photograph, etc.)
2. Describe the main content and any labels/legends
3. Extract data values if visible (for charts/graphs)
4. Note colors, patterns, and visual hierarchy
5. Identify any text annotations within the image

Your response MUST be valid JSON with this structure:
{
  "extracted_content": {
    "visual_type": "chart|graph|diagram|photograph|illustration|other",
    "description": "detailed description of the visual",
    "labels": ["label1", "label2", ...],
    "data_points": [{"label": "x", "value": "y"}, ...] (if applicable),
    "colors_used": ["color1", "color2", ...],
    "annotations": ["text1", "text2", ...]
  },
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of your analysis"
}"""

    SYSTEM_PROMPT_TEXT = """You are an expert at reading and extracting text from document images.
Your task is to extract and structure the text content from the provided region.

IMPORTANT RULES:
1. Extract all visible text accurately
2. Preserve formatting (paragraphs, bullet points, etc.)
3. Note any emphasized text (bold, italic, underlined)
4. Identify the semantic role of the text (heading, body, list, etc.)
5. If text is partially obscured, indicate with [...] marker

Your response MUST be valid JSON with this structure:
{
  "extracted_content": {
    "text": "the extracted text content",
    "text_type": "heading|paragraph|list|caption|other",
    "formatting": {
      "has_bold": true/false,
      "has_italic": true/false,
      "has_bullet_points": true/false
    },
    "line_count": number
  },
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of your analysis"
}"""

    USER_PROMPT_TEMPLATE = """Analyze the following document region and extract its content.

REGION CONTEXT:
- Region Type: {region_type}
- Page Number: {page_number}
- Position on Page: {position_description}
- Reading Order Position: {reading_order}
- Surrounding Context: {surrounding_context}
{ocr_text_section}

Extract the content from this region according to the instructions provided.
Respond with ONLY valid JSON."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the region visual extractor.

        Args:
            api_key: OpenAI API key. Defaults to settings.openai_api_key.
            model: Model name to use. Defaults to gpt-4o for vision support.
            temperature: Sampling temperature. Defaults to settings.openai_temperature.
            max_tokens: Maximum tokens for response. Defaults to settings.openai_max_tokens.
        """
        self.api_key = api_key or settings.openai_api_key
        # Use gpt-4o by default as it has excellent vision capabilities
        self.model = model or settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self.max_tokens = max_tokens or settings.openai_max_tokens

        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LangChain ChatOpenAI instance with vision support.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            RegionVisualExtractionError: If API key is not configured.
        """
        if self._llm is None:
            if not self.api_key:
                raise RegionVisualExtractionError(
                    "OpenAI API key not configured",
                    details={"missing": "openai_api_key"},
                )

            self._llm = ChatOpenAI(
                api_key=self.api_key,  # type: ignore[arg-type]
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )

        return self._llm

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode a PIL Image to base64 string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64 encoded string of the image.
        """
        # Convert to RGB if necessary (e.g., for RGBA images)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _get_system_prompt(self, strategy: ExtractionStrategy) -> str:
        """Get the appropriate system prompt for the extraction strategy.

        Args:
            strategy: The extraction strategy to use.

        Returns:
            System prompt string.
        """
        if strategy == ExtractionStrategy.TABLE_SPECIALIZED:
            return self.SYSTEM_PROMPT_TABLE
        elif strategy == ExtractionStrategy.FIGURE_ANALYSIS:
            return self.SYSTEM_PROMPT_FIGURE
        elif strategy == ExtractionStrategy.OCR_ENHANCED:
            return self.SYSTEM_PROMPT_TEXT
        else:
            return self.SYSTEM_PROMPT_BASE

    def _get_extraction_strategy(self, region: LayoutRegion) -> ExtractionStrategy:
        """Determine the best extraction strategy for a region.

        Args:
            region: The layout region to analyze.

        Returns:
            Appropriate extraction strategy.
        """
        return REGION_TYPE_STRATEGIES.get(
            region.region_type, ExtractionStrategy.VLM_ONLY
        )

    def _build_context(
        self,
        region: LayoutRegion,
        page_width: float,
        page_height: float,
        ordered_regions: list[OrderedRegion] | None = None,
        ocr_text: str | None = None,
        parent_region: LayoutRegion | None = None,
    ) -> RegionContext:
        """Build context information for a region.

        Args:
            region: The layout region.
            page_width: Width of the page in pixels.
            page_height: Height of the page in pixels.
            ordered_regions: Optional list of ordered regions for context.
            ocr_text: Optional OCR-extracted text from this region.
            parent_region: Optional parent region if nested.

        Returns:
            RegionContext with context information.
        """
        # Normalize bounding box to percentages
        bbox_normalized = (
            region.bbox.x0 / page_width,
            region.bbox.y0 / page_height,
            region.bbox.x1 / page_width,
            region.bbox.y1 / page_height,
        )

        # Find reading order position and surrounding regions
        reading_order_position = None
        preceding_types: list[str] = []
        following_types: list[str] = []

        if ordered_regions:
            for i, ordered_region in enumerate(ordered_regions):
                if ordered_region.region.region_id == region.region_id:
                    reading_order_position = i
                    # Get preceding region types (up to 3)
                    preceding_types = [
                        ordered_regions[j].region.region_type.value
                        for j in range(max(0, i - 3), i)
                    ]
                    # Get following region types (up to 3)
                    following_types = [
                        ordered_regions[j].region.region_type.value
                        for j in range(i + 1, min(len(ordered_regions), i + 4))
                    ]
                    break

        return RegionContext(
            region_type=region.region_type,
            page_number=region.page_number,
            page_dimensions=(page_width, page_height),
            bbox_normalized=bbox_normalized,
            reading_order_position=reading_order_position,
            total_regions_on_page=len(ordered_regions) if ordered_regions else 0,
            preceding_region_types=preceding_types,
            following_region_types=following_types,
            parent_region_type=parent_region.region_type.value
            if parent_region
            else None,
            ocr_text=ocr_text,
        )

    def _build_user_prompt(self, context: RegionContext) -> str:
        """Build the user prompt with context information.

        Args:
            context: Region context information.

        Returns:
            Formatted user prompt string.
        """
        # Describe position on page
        x_center = (context.bbox_normalized[0] + context.bbox_normalized[2]) / 2
        y_center = (context.bbox_normalized[1] + context.bbox_normalized[3]) / 2

        if y_center < 0.33:
            v_pos = "top"
        elif y_center < 0.67:
            v_pos = "middle"
        else:
            v_pos = "bottom"

        if x_center < 0.33:
            h_pos = "left"
        elif x_center < 0.67:
            h_pos = "center"
        else:
            h_pos = "right"

        position_description = f"{v_pos}-{h_pos} of the page"

        # Build reading order description
        if context.reading_order_position is not None:
            reading_order = f"{context.reading_order_position + 1} of {context.total_regions_on_page}"
        else:
            reading_order = "Unknown"

        # Build surrounding context description
        surrounding_parts = []
        if context.preceding_region_types:
            surrounding_parts.append(
                f"Preceded by: {', '.join(context.preceding_region_types)}"
            )
        if context.following_region_types:
            surrounding_parts.append(
                f"Followed by: {', '.join(context.following_region_types)}"
            )
        if context.parent_region_type:
            surrounding_parts.append(f"Nested within: {context.parent_region_type}")

        surrounding_context = (
            "; ".join(surrounding_parts) if surrounding_parts else "None"
        )

        # Add OCR text section if available
        ocr_text_section = ""
        if context.ocr_text:
            ocr_text_section = (
                f"\nOCR TEXT (for reference):\n```\n{context.ocr_text}\n```"
            )

        return self.USER_PROMPT_TEMPLATE.format(
            region_type=context.region_type.value,
            page_number=context.page_number,
            position_description=position_description,
            reading_order=reading_order,
            surrounding_context=surrounding_context,
            ocr_text_section=ocr_text_section,
        )

    def _parse_response(
        self, response_text: str, region_id: str
    ) -> tuple[dict[str, Any], float, str | None]:
        """Parse the VLM response.

        Args:
            response_text: Raw response text from the VLM.
            region_id: ID of the region being processed.

        Returns:
            Tuple of (extracted_content, confidence, reasoning).

        Raises:
            RegionVisualExtractionError: If parsing fails.
        """
        try:
            # Try to parse as JSON
            data = json.loads(response_text)

            extracted_content = data.get("extracted_content", {})
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning")

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            return extracted_content, confidence, reasoning

        except json.JSONDecodeError:
            # Try to extract JSON from the response if it contains extra text
            try:
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    data = json.loads(json_str)

                    extracted_content = data.get("extracted_content", {})
                    confidence = float(data.get("confidence", 0.5))
                    reasoning = data.get("reasoning")

                    return extracted_content, confidence, reasoning
            except json.JSONDecodeError:
                pass

            raise RegionVisualExtractionError(
                "Failed to parse VLM response as JSON",
                region_id=region_id,
                details={"response_preview": response_text[:500]},
            ) from None

    def extract_from_region(
        self,
        region: LayoutRegion,
        region_image: Image.Image,
        page_width: float,
        page_height: float,
        ordered_regions: list[OrderedRegion] | None = None,
        ocr_text: str | None = None,
        parent_region: LayoutRegion | None = None,
        schema: dict[str, Any] | None = None,
    ) -> RegionExtractionResult:
        """Extract information from a single region.

        Args:
            region: The layout region to extract from.
            region_image: Cropped image of the region.
            page_width: Width of the source page in pixels.
            page_height: Height of the source page in pixels.
            ordered_regions: Optional list of ordered regions for context.
            ocr_text: Optional OCR-extracted text from this region.
            parent_region: Optional parent region if nested.
            schema: Optional JSON schema to guide extraction.

        Returns:
            RegionExtractionResult with extracted information.

        Raises:
            RegionVisualExtractionError: If extraction fails.
        """
        start_time = time.time()

        # Determine extraction strategy
        strategy = self._get_extraction_strategy(region)

        # Build context
        context = self._build_context(
            region=region,
            page_width=page_width,
            page_height=page_height,
            ordered_regions=ordered_regions,
            ocr_text=ocr_text,
            parent_region=parent_region,
        )

        # Encode image
        image_base64 = self._encode_image_to_base64(region_image)

        # Build prompts
        system_prompt = self._get_system_prompt(strategy)
        user_prompt = self._build_user_prompt(context)

        # If schema is provided, add it to the system prompt
        if schema:
            system_prompt += (
                f"\n\nEXTRACT ACCORDING TO THIS SCHEMA:\n{json.dumps(schema, indent=2)}"
            )

        # Create messages with image
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ]
            ),
        ]

        try:
            # Call the VLM
            response = self.llm.invoke(messages)

            # Extract response content
            response_text = response.content
            if not isinstance(response_text, str):
                response_text = str(response_text)

            # Parse response
            extracted_content, confidence, reasoning = self._parse_response(
                response_text, region.region_id
            )

            # Extract token usage
            usage_metadata = getattr(response, "usage_metadata", None) or {}
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)

            processing_time = time.time() - start_time

            logger.debug(
                f"Extracted region {region.region_id}: "
                f"type={region.region_type.value}, "
                f"strategy={strategy.value}, "
                f"confidence={confidence:.2f}, "
                f"time={processing_time:.2f}s"
            )

            return RegionExtractionResult(
                region_id=region.region_id,
                region_type=region.region_type,
                extracted_content=extracted_content,
                confidence=confidence,
                strategy_used=strategy,
                reasoning=reasoning,
                raw_response=response_text,
                processing_time_seconds=processing_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except Exception as e:
            if isinstance(e, RegionVisualExtractionError):
                raise

            processing_time = time.time() - start_time
            raise RegionVisualExtractionError(
                f"VLM extraction failed for region {region.region_id}: {e}",
                region_id=region.region_id,
                details={
                    "error_type": type(e).__name__,
                    "processing_time": processing_time,
                },
            ) from e

    def extract_from_regions(
        self,
        regions: list[LayoutRegion],
        region_images: dict[str, Image.Image],
        page_width: float,
        page_height: float,
        ordered_regions: list[OrderedRegion] | None = None,
        ocr_texts: dict[str, str] | None = None,
        parent_regions: dict[str, LayoutRegion] | None = None,
        schema: dict[str, Any] | None = None,
        continue_on_error: bool = True,
    ) -> DocumentRegionExtractionResult:
        """Extract information from multiple regions.

        Args:
            regions: List of layout regions to extract from.
            region_images: Dictionary mapping region_id to cropped image.
            page_width: Width of the source page in pixels.
            page_height: Height of the source page in pixels.
            ordered_regions: Optional list of ordered regions for context.
            ocr_texts: Optional dictionary mapping region_id to OCR text.
            parent_regions: Optional dictionary mapping region_id to parent region.
            schema: Optional JSON schema to guide extraction.
            continue_on_error: Whether to continue if a region fails.

        Returns:
            DocumentRegionExtractionResult with all extraction results.

        Raises:
            RegionVisualExtractionError: If extraction fails and continue_on_error is False.
        """
        start_time = time.time()

        results: list[RegionExtractionResult] = []
        successful = 0
        failed = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        ocr_texts = ocr_texts or {}
        parent_regions = parent_regions or {}

        for region in regions:
            region_image = region_images.get(region.region_id)
            if region_image is None:
                logger.warning(f"No image found for region {region.region_id}")
                failed += 1
                continue

            try:
                result = self.extract_from_region(
                    region=region,
                    region_image=region_image,
                    page_width=page_width,
                    page_height=page_height,
                    ordered_regions=ordered_regions,
                    ocr_text=ocr_texts.get(region.region_id),
                    parent_region=parent_regions.get(region.region_id),
                    schema=schema,
                )
                results.append(result)
                successful += 1
                total_prompt_tokens += result.prompt_tokens
                total_completion_tokens += result.completion_tokens

            except RegionVisualExtractionError as e:
                failed += 1
                logger.error(f"Failed to extract region {region.region_id}: {e}")

                if not continue_on_error:
                    raise

                # Add a failed result
                results.append(
                    RegionExtractionResult(
                        region_id=region.region_id,
                        region_type=region.region_type,
                        extracted_content={},
                        confidence=0.0,
                        strategy_used=self._get_extraction_strategy(region),
                        reasoning=f"Extraction failed: {e}",
                    )
                )

        processing_time = time.time() - start_time
        total_tokens = total_prompt_tokens + total_completion_tokens

        logger.info(
            f"Region extraction complete: "
            f"{successful}/{len(regions)} successful, "
            f"{failed} failed, "
            f"tokens={total_tokens}, "
            f"time={processing_time:.2f}s"
        )

        return DocumentRegionExtractionResult(
            region_results=results,
            total_regions=len(regions),
            successful_extractions=successful,
            failed_extractions=failed,
            model_used=self.model,
            total_tokens=total_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            processing_time_seconds=processing_time,
        )

    def extract_from_page(
        self,
        page_image: Image.Image,
        regions: list[LayoutRegion],
        page_width: float,
        page_height: float,
        ordered_regions: list[OrderedRegion] | None = None,
        ocr_texts: dict[str, str] | None = None,
        schema: dict[str, Any] | None = None,
        padding: int = 5,
        continue_on_error: bool = True,
    ) -> DocumentRegionExtractionResult:
        """Extract information from all regions on a page.

        Automatically crops regions from the page image and extracts.

        Args:
            page_image: The full page image.
            regions: List of layout regions on this page.
            page_width: Width of the page in pixels.
            page_height: Height of the page in pixels.
            ordered_regions: Optional list of ordered regions for context.
            ocr_texts: Optional dictionary mapping region_id to OCR text.
            schema: Optional JSON schema to guide extraction.
            padding: Padding around cropped regions in pixels.
            continue_on_error: Whether to continue if a region fails.

        Returns:
            DocumentRegionExtractionResult with all extraction results.
        """
        # Crop regions from the page image
        region_images: dict[str, Image.Image] = {}
        parent_regions: dict[str, LayoutRegion] = {}

        # Build parent region lookup
        region_lookup = {r.region_id: r for r in regions}

        for region in regions:
            # Crop the region with padding
            bbox = region.bbox
            x0 = max(0, int(bbox.x0) - padding)
            y0 = max(0, int(bbox.y0) - padding)
            x1 = min(page_image.width, int(bbox.x1) + padding)
            y1 = min(page_image.height, int(bbox.y1) + padding)

            region_images[region.region_id] = page_image.crop((x0, y0, x1, y1))

            # Track parent regions
            if region.parent_region_id and region.parent_region_id in region_lookup:
                parent_regions[region.region_id] = region_lookup[
                    region.parent_region_id
                ]

        return self.extract_from_regions(
            regions=regions,
            region_images=region_images,
            page_width=page_width,
            page_height=page_height,
            ordered_regions=ordered_regions,
            ocr_texts=ocr_texts,
            parent_regions=parent_regions,
            schema=schema,
            continue_on_error=continue_on_error,
        )

    @staticmethod
    def get_supported_strategies() -> list[str]:
        """Get list of supported extraction strategies.

        Returns:
            List of strategy values.
        """
        return [s.value for s in ExtractionStrategy]

    @staticmethod
    def get_strategy_for_region_type(region_type: RegionType) -> ExtractionStrategy:
        """Get the recommended extraction strategy for a region type.

        Args:
            region_type: The type of region.

        Returns:
            Recommended extraction strategy.
        """
        return REGION_TYPE_STRATEGIES.get(region_type, ExtractionStrategy.VLM_ONLY)
