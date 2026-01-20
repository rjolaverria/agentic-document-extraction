"""Reading order detection service using LLM.

This module provides functionality to detect the reading order of text elements
and layout regions in visual documents using LangChain with OpenAI GPT models.
It analyzes spatial relationships to determine the logical flow of the document.

Key features:
- Takes text elements/regions with bounding box coordinates as input
- Uses LangChain with OpenAI GPT-4 to analyze spatial relationships
- Handles complex layouts (multi-column, sidebar, headers/footers)
- Provides confidence scores for reading order decisions
- Works across multi-page documents
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agentic_document_extraction.config import settings
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    PageLayoutResult,
    RegionType,
)

logger = logging.getLogger(__name__)


class ReadingOrderError(Exception):
    """Raised when reading order detection fails."""

    def __init__(
        self,
        message: str,
        page_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with message and optional context.

        Args:
            message: Error message.
            page_number: Optional page number where detection failed.
            details: Optional additional error details.
        """
        super().__init__(message)
        self.page_number = page_number
        self.details = details or {}


@dataclass
class OrderedRegion:
    """A layout region with its determined reading order position.

    Represents a region that has been assigned a position in the
    logical reading sequence of the document.
    """

    region: LayoutRegion
    """The original layout region."""

    order_index: int
    """Position in the reading order (0-indexed within page)."""

    confidence: float
    """Confidence score (0.0-1.0) for this ordering decision."""

    reasoning: str | None = None
    """Explanation for why this region was placed in this position."""

    skip_in_reading: bool = False
    """Whether this region should be skipped during reading (e.g., page headers)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with ordered region information.
        """
        return {
            "region_id": self.region.region_id,
            "region_type": self.region.region_type.value,
            "order_index": self.order_index,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "skip_in_reading": self.skip_in_reading,
            "bbox": self.region.bbox.to_dict(),
        }


@dataclass
class PageReadingOrder:
    """Reading order result for a single page."""

    page_number: int
    """Page number (1-indexed)."""

    ordered_regions: list[OrderedRegion]
    """Regions ordered by reading sequence."""

    overall_confidence: float
    """Overall confidence for the page ordering."""

    layout_type: str
    """Detected layout type (single-column, multi-column, mixed, etc.)."""

    processing_time_seconds: float = 0.0
    """Time taken to determine reading order."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with page reading order information.
        """
        return {
            "page_number": self.page_number,
            "ordered_regions": [r.to_dict() for r in self.ordered_regions],
            "overall_confidence": self.overall_confidence,
            "layout_type": self.layout_type,
            "processing_time_seconds": self.processing_time_seconds,
            "region_count": len(self.ordered_regions),
        }

    def get_reading_sequence(self, include_skipped: bool = False) -> list[LayoutRegion]:
        """Get regions in reading order.

        Args:
            include_skipped: Whether to include regions marked as skip_in_reading.

        Returns:
            List of regions in reading order.
        """
        regions = self.ordered_regions
        if not include_skipped:
            regions = [r for r in regions if not r.skip_in_reading]
        return [r.region for r in sorted(regions, key=lambda r: r.order_index)]


@dataclass
class DocumentReadingOrder:
    """Reading order result for an entire document."""

    pages: list[PageReadingOrder]
    """Reading order for each page."""

    total_pages: int
    """Total number of pages processed."""

    total_regions: int
    """Total number of regions ordered."""

    model_used: str
    """Name of the model used for analysis."""

    total_tokens: int = 0
    """Total tokens used in the analysis."""

    prompt_tokens: int = 0
    """Tokens used in prompts."""

    completion_tokens: int = 0
    """Tokens used in completions."""

    processing_time_seconds: float = 0.0
    """Total time taken for analysis."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with document reading order information.
        """
        return {
            "pages": [p.to_dict() for p in self.pages],
            "total_pages": self.total_pages,
            "total_regions": self.total_regions,
            "metadata": {
                "model_used": self.model_used,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "processing_time_seconds": self.processing_time_seconds,
            },
        }

    def get_all_regions_ordered(
        self, include_skipped: bool = False
    ) -> list[tuple[int, LayoutRegion]]:
        """Get all regions across all pages in reading order.

        Args:
            include_skipped: Whether to include regions marked as skip_in_reading.

        Returns:
            List of (page_number, region) tuples in reading order.
        """
        result: list[tuple[int, LayoutRegion]] = []
        for page in sorted(self.pages, key=lambda p: p.page_number):
            regions = page.get_reading_sequence(include_skipped=include_skipped)
            result.extend((page.page_number, r) for r in regions)
        return result


# Region types that are typically skipped in main reading flow
SKIP_REGION_TYPES: set[RegionType] = {
    RegionType.PAGE_HEADER,
    RegionType.PAGE_FOOTER,
    RegionType.FOOTNOTE,
}


class ReadingOrderDetector:
    """Detects reading order of layout regions using LLM analysis.

    Uses LangChain with OpenAI GPT-4 to analyze spatial relationships
    between regions and determine the logical reading order. Handles
    complex layouts including multi-column, sidebars, and headers/footers.
    """

    # System prompt for reading order analysis
    SYSTEM_PROMPT = """You are an expert document layout analyst. Your task is to determine the correct reading order of document regions based on their positions and types.

RULES FOR READING ORDER:
1. For single-column layouts: Read top to bottom, left to right
2. For multi-column layouts: Read each column top to bottom before moving to the next column
3. Titles and section headers come before their content
4. Captions should be read adjacent to their figures/tables
5. Sidebars and callout boxes may interrupt main flow but should be grouped
6. Page headers and footers are typically read separately (mark as skip_in_reading: true)
7. Footnotes are typically read at the end of main content (mark as skip_in_reading: true)
8. Tables should be read as single units
9. List items should be read in their visual order

LAYOUT TYPE DETECTION:
- "single_column": One main column of content
- "multi_column": Multiple columns (2 or more)
- "mixed": Combination of single and multi-column sections
- "complex": Non-standard layout requiring special handling

You must respond with ONLY valid JSON matching this structure:
{
  "layout_type": "single_column|multi_column|mixed|complex",
  "ordered_regions": [
    {
      "region_id": "string",
      "order_index": 0,
      "confidence": 0.95,
      "reasoning": "brief explanation",
      "skip_in_reading": false
    }
  ],
  "overall_confidence": 0.9
}"""

    USER_PROMPT_TEMPLATE = """Analyze the following document regions and determine their reading order.

Page dimensions: {page_width}x{page_height} pixels

Regions to order:
{regions_json}

Determine the correct reading order based on spatial positions and region types.
Respond with ONLY the JSON structure specified."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        """Initialize the reading order detector.

        Args:
            api_key: OpenAI API key. Defaults to settings.openai_api_key.
            model: Model name to use. Defaults to settings.openai_model.
            temperature: Sampling temperature. Defaults to 0.0 for determinism.
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.temperature = temperature if temperature is not None else 0.0

        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LangChain ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            ReadingOrderError: If API key is not configured.
        """
        if self._llm is None:
            if not self.api_key:
                raise ReadingOrderError(
                    "OpenAI API key not configured",
                    details={"missing": "openai_api_key"},
                )

            self._llm = ChatOpenAI(
                api_key=self.api_key,  # type: ignore[arg-type]
                model=self.model,
                temperature=self.temperature,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

        return self._llm

    def _format_regions_for_prompt(self, regions: list[LayoutRegion]) -> str:
        """Format regions as JSON for the LLM prompt.

        Args:
            regions: List of layout regions to format.

        Returns:
            JSON string with region information.
        """
        regions_data = []
        for region in regions:
            region_info = {
                "region_id": region.region_id,
                "region_type": region.region_type.value,
                "bbox": {
                    "x0": round(region.bbox.x0, 1),
                    "y0": round(region.bbox.y0, 1),
                    "x1": round(region.bbox.x1, 1),
                    "y1": round(region.bbox.y1, 1),
                },
                "center": {
                    "x": round(region.bbox.center[0], 1),
                    "y": round(region.bbox.center[1], 1),
                },
                "confidence": round(region.confidence, 2),
            }
            if region.parent_region_id:
                region_info["parent_region_id"] = region.parent_region_id
            regions_data.append(region_info)

        return json.dumps(regions_data, indent=2)

    def _parse_llm_response(
        self,
        response_text: str,
        regions: list[LayoutRegion],
    ) -> tuple[list[OrderedRegion], str, float]:
        """Parse the LLM response into ordered regions.

        Args:
            response_text: Raw JSON response from LLM.
            regions: Original list of regions.

        Returns:
            Tuple of (ordered_regions, layout_type, overall_confidence).

        Raises:
            ReadingOrderError: If response parsing fails.
        """
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ReadingOrderError(
                f"Failed to parse LLM response as JSON: {e}",
                details={"raw_response": response_text[:500]},
            ) from e

        # Create a lookup map for regions by ID
        region_map = {r.region_id: r for r in regions}

        # Parse ordered regions
        ordered_regions: list[OrderedRegion] = []
        seen_ids: set[str] = set()

        for item in response_data.get("ordered_regions", []):
            region_id = item.get("region_id")
            if not region_id or region_id not in region_map:
                logger.warning(f"Unknown region_id in LLM response: {region_id}")
                continue

            if region_id in seen_ids:
                logger.warning(f"Duplicate region_id in LLM response: {region_id}")
                continue

            seen_ids.add(region_id)

            ordered_region = OrderedRegion(
                region=region_map[region_id],
                order_index=item.get("order_index", len(ordered_regions)),
                confidence=float(item.get("confidence", 0.5)),
                reasoning=item.get("reasoning"),
                skip_in_reading=bool(item.get("skip_in_reading", False)),
            )
            ordered_regions.append(ordered_region)

        # Handle any regions not in the response (shouldn't happen, but be safe)
        for region_id, region in region_map.items():
            if region_id not in seen_ids:
                logger.warning(f"Region {region_id} not in LLM response, adding at end")
                ordered_region = OrderedRegion(
                    region=region,
                    order_index=len(ordered_regions),
                    confidence=0.3,  # Low confidence for missing regions
                    reasoning="Region not analyzed by LLM",
                    skip_in_reading=region.region_type in SKIP_REGION_TYPES,
                )
                ordered_regions.append(ordered_region)

        # Sort by order_index to ensure correct order
        ordered_regions.sort(key=lambda r: r.order_index)

        layout_type = response_data.get("layout_type", "unknown")
        overall_confidence = float(response_data.get("overall_confidence", 0.5))

        return ordered_regions, layout_type, overall_confidence

    def detect_reading_order_for_page(
        self,
        page_result: PageLayoutResult,
    ) -> PageReadingOrder:
        """Detect reading order for a single page.

        Args:
            page_result: Layout detection result for the page.

        Returns:
            Reading order result for the page.

        Raises:
            ReadingOrderError: If reading order detection fails.
        """
        start_time = time.time()

        # Handle empty pages
        if not page_result.regions:
            return PageReadingOrder(
                page_number=page_result.page_number,
                ordered_regions=[],
                overall_confidence=1.0,
                layout_type="empty",
                processing_time_seconds=time.time() - start_time,
            )

        # Format regions for the prompt
        regions_json = self._format_regions_for_prompt(page_result.regions)

        # Create the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", self.USER_PROMPT_TEMPLATE),
            ]
        )

        # Call the LLM
        try:
            chain = prompt | self.llm
            response = chain.invoke(
                {
                    "page_width": page_result.page_width,
                    "page_height": page_result.page_height,
                    "regions_json": regions_json,
                }
            )
        except Exception as e:
            raise ReadingOrderError(
                f"LLM call failed: {e}",
                page_number=page_result.page_number,
                details={"error_type": type(e).__name__},
            ) from e

        # Parse the response
        response_text = response.content
        if not isinstance(response_text, str):
            response_text = str(response_text)

        ordered_regions, layout_type, overall_confidence = self._parse_llm_response(
            response_text, page_result.regions
        )

        processing_time = time.time() - start_time

        logger.info(
            f"Page {page_result.page_number}: "
            f"ordered {len(ordered_regions)} regions, "
            f"layout={layout_type}, "
            f"confidence={overall_confidence:.2f}, "
            f"time={processing_time:.2f}s"
        )

        return PageReadingOrder(
            page_number=page_result.page_number,
            ordered_regions=ordered_regions,
            overall_confidence=overall_confidence,
            layout_type=layout_type,
            processing_time_seconds=processing_time,
        )

    def detect_reading_order(
        self,
        page_results: list[PageLayoutResult],
    ) -> DocumentReadingOrder:
        """Detect reading order for an entire document.

        Args:
            page_results: Layout detection results for all pages.

        Returns:
            Reading order result for the document.

        Raises:
            ReadingOrderError: If reading order detection fails.
        """
        start_time = time.time()
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        page_orders: list[PageReadingOrder] = []
        total_regions = 0

        for page_result in page_results:
            page_order = self.detect_reading_order_for_page(page_result)
            page_orders.append(page_order)
            total_regions += len(page_order.ordered_regions)

            # Try to extract token usage from LLM (if available)
            # Note: Token tracking may need enhancement based on LangChain version

        processing_time = time.time() - start_time

        logger.info(
            f"Document reading order complete: "
            f"{len(page_orders)} pages, "
            f"{total_regions} regions, "
            f"time={processing_time:.2f}s"
        )

        return DocumentReadingOrder(
            pages=page_orders,
            total_pages=len(page_orders),
            total_regions=total_regions,
            model_used=self.model,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            processing_time_seconds=processing_time,
        )

    def detect_reading_order_simple(
        self,
        regions: list[LayoutRegion],
        page_width: float = 1000.0,
        page_height: float = 1000.0,  # noqa: ARG002
    ) -> PageReadingOrder:
        """Detect reading order using simple heuristics (no LLM).

        This method provides a fallback when LLM is not available or for
        simple single-column layouts. Uses top-to-bottom, left-to-right
        ordering with special handling for common region types.

        Args:
            regions: List of layout regions to order.
            page_width: Page width in pixels (for layout analysis).
            page_height: Page height in pixels (for layout analysis).

        Returns:
            Reading order result for the regions.
        """
        start_time = time.time()

        if not regions:
            return PageReadingOrder(
                page_number=1,
                ordered_regions=[],
                overall_confidence=1.0,
                layout_type="empty",
                processing_time_seconds=time.time() - start_time,
            )

        # Detect layout type based on region positions
        layout_type = self._detect_layout_type_heuristic(regions, page_width)

        # Sort regions based on detected layout
        if layout_type == "multi_column":
            sorted_regions = self._sort_multi_column(regions, page_width)
        else:
            sorted_regions = self._sort_single_column(regions)

        # Create ordered regions with assigned indices
        ordered_regions: list[OrderedRegion] = []
        for idx, region in enumerate(sorted_regions):
            skip = region.region_type in SKIP_REGION_TYPES
            ordered_region = OrderedRegion(
                region=region,
                order_index=idx,
                confidence=0.7 if layout_type == "single_column" else 0.5,
                reasoning=f"Heuristic ordering: {layout_type}",
                skip_in_reading=skip,
            )
            ordered_regions.append(ordered_region)

        processing_time = time.time() - start_time

        # Lower confidence for heuristic-based ordering
        overall_confidence = 0.6 if layout_type == "single_column" else 0.4

        return PageReadingOrder(
            page_number=regions[0].page_number if regions else 1,
            ordered_regions=ordered_regions,
            overall_confidence=overall_confidence,
            layout_type=layout_type,
            processing_time_seconds=processing_time,
        )

    def _detect_layout_type_heuristic(
        self,
        regions: list[LayoutRegion],
        page_width: float,
    ) -> str:
        """Detect layout type using heuristics.

        Args:
            regions: List of layout regions.
            page_width: Page width in pixels.

        Returns:
            Detected layout type string.
        """
        if len(regions) < 2:
            return "single_column"

        # Filter out headers/footers for column analysis
        content_regions = [r for r in regions if r.region_type not in SKIP_REGION_TYPES]

        if len(content_regions) < 2:
            return "single_column"

        # Check if regions are horizontally distributed
        # indicating potential multi-column layout
        x_centers = [r.bbox.center[0] for r in content_regions]
        x_spread = max(x_centers) - min(x_centers)

        # If horizontal spread is > 40% of page width with similar y values,
        # likely multi-column
        if x_spread > page_width * 0.4:
            # Check if multiple regions are at similar y positions
            y_centers = sorted(r.bbox.center[1] for r in content_regions)
            y_clusters = self._count_y_clusters(y_centers, page_width * 0.1)

            if y_clusters > 1 and len(content_regions) > 3:
                return "multi_column"

        return "single_column"

    def _count_y_clusters(self, y_values: list[float], threshold: float) -> int:
        """Count clusters of Y values.

        Args:
            y_values: Sorted list of Y coordinates.
            threshold: Distance threshold for clustering.

        Returns:
            Number of clusters.
        """
        if not y_values:
            return 0

        clusters = 1
        prev_y = y_values[0]

        for y in y_values[1:]:
            if y - prev_y > threshold:
                clusters += 1
            prev_y = y

        return clusters

    def _sort_single_column(
        self,
        regions: list[LayoutRegion],
    ) -> list[LayoutRegion]:
        """Sort regions for single-column layout.

        Orders by Y position (top to bottom), with X as tiebreaker.

        Args:
            regions: Regions to sort.

        Returns:
            Sorted regions.
        """
        return sorted(
            regions,
            key=lambda r: (r.bbox.y0, r.bbox.x0),
        )

    def _sort_multi_column(
        self,
        regions: list[LayoutRegion],
        page_width: float,
    ) -> list[LayoutRegion]:
        """Sort regions for multi-column layout.

        Detects columns and orders by column (left to right),
        then by Y position within each column.

        Args:
            regions: Regions to sort.
            page_width: Page width in pixels.

        Returns:
            Sorted regions.
        """
        # Determine column boundaries by clustering X centers
        x_centers = [(r, r.bbox.center[0]) for r in regions]

        # Simple two-column detection: split at middle
        mid_x = page_width / 2

        left_column = [r for r, x in x_centers if x < mid_x]
        right_column = [r for r, x in x_centers if x >= mid_x]

        # Sort each column by Y position
        left_sorted = sorted(left_column, key=lambda r: (r.bbox.y0, r.bbox.x0))
        right_sorted = sorted(right_column, key=lambda r: (r.bbox.y0, r.bbox.x0))

        # Combine: left column first, then right
        return left_sorted + right_sorted

    @staticmethod
    def get_default_model() -> str:
        """Get the default model name from settings.

        Returns:
            Default model name.
        """
        return settings.openai_model
