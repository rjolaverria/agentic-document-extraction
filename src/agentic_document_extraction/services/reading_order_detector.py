"""Reading order detection service using LayoutReader.

This module provides functionality to detect the reading order of layout
regions in visual documents using the LayoutReader model. It analyzes
spatial relationships locally to determine the logical flow of the document.

Key features:
- Takes layout regions with bounding boxes as input
- Uses LayoutReader (LayoutLMv3) to infer reading order
- Handles complex layouts (multi-column, mixed content)
- Provides confidence scores for reading order decisions
- Works across multi-page documents
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch
from transformers import LayoutLMv3ForTokenClassification

from agentic_document_extraction.config import settings
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    PageLayoutResult,
    RegionType,
)

logger = logging.getLogger(__name__)

MAX_LEN = 510
CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2


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


def _boxes_to_inputs(boxes: list[list[int]]) -> dict[str, torch.Tensor]:
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * len(boxes) + [EOS_TOKEN_ID]
    attention_mask = [1] + [1] * len(boxes) + [1]
    return {
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids": torch.tensor([input_ids]),
    }


def _prepare_inputs(
    inputs: dict[str, torch.Tensor],
    model: LayoutLMv3ForTokenClassification,
) -> dict[str, torch.Tensor]:
    if not list(model.parameters()):
        raise ReadingOrderError("LayoutReader model has no parameters")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prepared: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        value = value.to(device)
        if torch.is_floating_point(value):
            value = value.to(dtype)
        prepared[key] = value
    return prepared


def _parse_logits(logits: torch.Tensor, length: int) -> list[int]:
    """Parse LayoutReader logits into reading order indices."""
    if length <= 0:
        return []

    trimmed = logits[1 : length + 1, :length]
    orders = trimmed.argsort(descending=False).tolist()
    ret = [order.pop() for order in orders]

    while True:
        order_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_indices[order].append(idx)
        duplicates = {k: v for k, v in order_to_indices.items() if len(v) > 1}
        if not duplicates:
            break
        for order, indices in duplicates.items():
            indices_to_logits = {idx: trimmed[idx, order] for idx in indices}
            sorted_indices = sorted(
                indices_to_logits.items(), key=lambda item: item[1], reverse=True
            )
            for idx, _ in sorted_indices[1:]:
                ret[idx] = orders[idx].pop()

    return ret


@lru_cache(maxsize=1)
def _load_layoutreader_model(
    model_name: str,
    device: str,
    use_bfloat16: bool,
) -> LayoutLMv3ForTokenClassification:
    model = cast(
        LayoutLMv3ForTokenClassification,
        LayoutLMv3ForTokenClassification.from_pretrained(model_name),
    )
    model_device = torch.device(device)
    if use_bfloat16 and model_device.type == "cuda":
        model = model.bfloat16()
    model = cast(LayoutLMv3ForTokenClassification, model.to(model_device))  # type: ignore[arg-type]
    model.eval()  # type: ignore[no-untyped-call]
    return model


class ReadingOrderDetector:
    """Detects reading order of layout regions using LayoutReader."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_bfloat16: bool = True,
    ) -> None:
        """Initialize the reading order detector.

        Args:
            model_name: LayoutReader model name. Defaults to settings.
            device: Torch device string (cpu/cuda). Defaults to auto-detect.
            use_bfloat16: Whether to prefer bfloat16 on CUDA.
        """
        self.model_name = model_name or settings.layoutreader_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bfloat16 = use_bfloat16

    @property
    def model(self) -> LayoutLMv3ForTokenClassification:
        """Load or retrieve the cached LayoutReader model."""
        return _load_layoutreader_model(self.model_name, self.device, self.use_bfloat16)

    def _scale_boxes(
        self, regions: list[LayoutRegion], page_width: float, page_height: float
    ) -> list[list[int]]:
        x_scale = 1000.0 / page_width
        y_scale = 1000.0 / page_height

        boxes: list[list[int]] = []
        for region in regions:
            left = int(round(region.bbox.x0 * x_scale))
            top = int(round(region.bbox.y0 * y_scale))
            right = int(round(region.bbox.x1 * x_scale))
            bottom = int(round(region.bbox.y1 * y_scale))

            left = max(0, min(1000, left))
            top = max(0, min(1000, top))
            right = max(left, min(1000, right))
            bottom = max(top, min(1000, bottom))

            boxes.append([left, top, right, bottom])

        return boxes

    def _predict_orders(self, boxes: list[list[int]]) -> tuple[list[int], list[float]]:
        inputs = _boxes_to_inputs(boxes)
        prepared = _prepare_inputs(inputs, self.model)
        logits = self.model(**prepared).logits.detach().cpu().squeeze(0)

        orders = _parse_logits(logits, len(boxes))
        if not orders:
            return orders, []

        region_logits = logits[1 : len(boxes) + 1, : len(boxes)].float()
        probs = torch.softmax(region_logits, dim=-1)
        confidences: list[float] = []
        for idx, order in enumerate(orders):
            if 0 <= order < len(boxes):
                confidences.append(float(probs[idx, order].item()))
            else:
                confidences.append(0.0)

        return orders, confidences

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

        if not page_result.regions:
            return PageReadingOrder(
                page_number=page_result.page_number,
                ordered_regions=[],
                overall_confidence=1.0,
                layout_type="empty",
                processing_time_seconds=time.time() - start_time,
            )

        if page_result.page_width <= 0 or page_result.page_height <= 0:
            logger.warning(
                "Invalid page dimensions for LayoutReader; falling back to heuristic.",
                extra={
                    "component": "reading_order_detector",
                    "page_number": page_result.page_number,
                    "page_width": page_result.page_width,
                    "page_height": page_result.page_height,
                },
            )
            fallback = self.detect_reading_order_simple(
                page_result.regions,
                page_width=page_result.page_width,
                page_height=page_result.page_height,
            )
            fallback.processing_time_seconds = time.time() - start_time
            return fallback

        if len(page_result.regions) > MAX_LEN:
            logger.warning(
                "LayoutReader max region count exceeded; falling back to heuristic.",
                extra={
                    "component": "reading_order_detector",
                    "page_number": page_result.page_number,
                    "region_count": len(page_result.regions),
                    "max_len": MAX_LEN,
                },
            )
            fallback = self.detect_reading_order_simple(
                page_result.regions,
                page_width=page_result.page_width,
                page_height=page_result.page_height,
            )
            fallback.processing_time_seconds = time.time() - start_time
            return fallback

        boxes = self._scale_boxes(
            page_result.regions, page_result.page_width, page_result.page_height
        )

        try:
            orders, confidences = self._predict_orders(boxes)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ReadingOrderError(
                f"LayoutReader inference failed: {exc}",
                page_number=page_result.page_number,
                details={"error_type": type(exc).__name__},
            ) from exc

        if len(orders) != len(page_result.regions):
            raise ReadingOrderError(
                "LayoutReader returned an unexpected number of orders",
                page_number=page_result.page_number,
                details={
                    "expected": len(page_result.regions),
                    "actual": len(orders),
                },
            )

        ordered_regions: list[OrderedRegion] = []
        for idx, region in enumerate(page_result.regions):
            skip = region.region_type in SKIP_REGION_TYPES
            confidence = confidences[idx] if idx < len(confidences) else 0.0
            ordered_regions.append(
                OrderedRegion(
                    region=region,
                    order_index=orders[idx],
                    confidence=confidence,
                    reasoning="LayoutReader model",
                    skip_in_reading=skip,
                )
            )

        ordered_regions.sort(key=lambda r: r.order_index)

        layout_type = self._detect_layout_type_heuristic(
            page_result.regions, page_result.page_width
        )
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        processing_time = time.time() - start_time

        logger.info(
            "Page reading order complete",
            extra={
                "component": "reading_order_detector",
                "page_number": page_result.page_number,
                "region_count": len(ordered_regions),
                "layout_type": layout_type,
                "overall_confidence": round(overall_confidence, 4),
                "processing_time_seconds": round(processing_time, 4),
                "model": self.model_name,
            },
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

        page_orders: list[PageReadingOrder] = []
        total_regions = 0

        for page_result in page_results:
            page_order = self.detect_reading_order_for_page(page_result)
            page_orders.append(page_order)
            total_regions += len(page_order.ordered_regions)

        processing_time = time.time() - start_time

        logger.info(
            "Document reading order complete",
            extra={
                "component": "reading_order_detector",
                "page_count": len(page_orders),
                "region_count": total_regions,
                "processing_time_seconds": round(processing_time, 4),
                "model": self.model_name,
            },
        )

        return DocumentReadingOrder(
            pages=page_orders,
            total_pages=len(page_orders),
            total_regions=total_regions,
            model_used=self.model_name,
            processing_time_seconds=processing_time,
        )

    def detect_reading_order_simple(
        self,
        regions: list[LayoutRegion],
        page_width: float = 1000.0,
        page_height: float = 1000.0,  # noqa: ARG002
    ) -> PageReadingOrder:
        """Detect reading order using simple heuristics (no model).

        This method provides a fallback when LayoutReader is unavailable or for
        large pages. Uses top-to-bottom, left-to-right ordering with special
        handling for common region types.

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

        layout_type = self._detect_layout_type_heuristic(regions, page_width)

        if layout_type == "multi_column":
            sorted_regions = self._sort_multi_column(regions, page_width)
        else:
            sorted_regions = self._sort_single_column(regions)

        ordered_regions: list[OrderedRegion] = []
        for idx, region in enumerate(sorted_regions):
            skip = region.region_type in SKIP_REGION_TYPES
            ordered_regions.append(
                OrderedRegion(
                    region=region,
                    order_index=idx,
                    confidence=0.7 if layout_type == "single_column" else 0.5,
                    reasoning=f"Heuristic ordering: {layout_type}",
                    skip_in_reading=skip,
                )
            )

        processing_time = time.time() - start_time

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

        content_regions = [r for r in regions if r.region_type not in SKIP_REGION_TYPES]

        if len(content_regions) < 2:
            return "single_column"

        x_centers = [r.bbox.center[0] for r in content_regions]
        x_spread = max(x_centers) - min(x_centers)

        if x_spread > page_width * 0.4:
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
        x_centers = [(r, r.bbox.center[0]) for r in regions]
        mid_x = page_width / 2

        left_column = [r for r, x in x_centers if x < mid_x]
        right_column = [r for r, x in x_centers if x >= mid_x]

        left_sorted = sorted(left_column, key=lambda r: (r.bbox.y0, r.bbox.x0))
        right_sorted = sorted(right_column, key=lambda r: (r.bbox.y0, r.bbox.x0))

        return left_sorted + right_sorted

    @staticmethod
    def get_default_model() -> str:
        """Get the default LayoutReader model name from settings."""
        return settings.layoutreader_model
