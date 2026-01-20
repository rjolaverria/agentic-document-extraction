"""Layout detection service using Hugging Face Transformers.

This module provides functionality to detect and segment different layout regions
in visual documents using pre-trained models from Hugging Face. It identifies
regions such as headers, footers, body text, tables, figures, captions, etc.

Uses:
- Deformable DETR model fine-tuned on DocLayNet for document layout analysis
- Hugging Face transformers library for model inference
"""

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


class LayoutDetectionError(Exception):
    """Raised when layout detection fails."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        page_number: int | None = None,
    ) -> None:
        """Initialize with message and optional context.

        Args:
            message: Error message.
            file_path: Optional path to the file that failed.
            page_number: Optional page number where detection failed.
        """
        super().__init__(message)
        self.file_path = file_path
        self.page_number = page_number


class RegionType(str, Enum):
    """Types of document layout regions.

    Based on DocLayNet labels which include common document elements.
    """

    CAPTION = "caption"
    """Caption text for figures or tables."""

    FOOTNOTE = "footnote"
    """Footnote text at bottom of page."""

    FORMULA = "formula"
    """Mathematical formula or equation."""

    LIST_ITEM = "list_item"
    """Item in a list (bulleted or numbered)."""

    PAGE_FOOTER = "page_footer"
    """Footer area of the page."""

    PAGE_HEADER = "page_header"
    """Header area of the page."""

    PICTURE = "picture"
    """Image or photograph."""

    SECTION_HEADER = "section_header"
    """Section or subsection heading."""

    TABLE = "table"
    """Tabular data."""

    TEXT = "text"
    """Body text or paragraph."""

    TITLE = "title"
    """Document or section title."""

    UNKNOWN = "unknown"
    """Unknown or unclassified region."""


# Mapping from model label IDs to RegionType
# Based on DocLayNet dataset labels
DOCLAYNET_LABEL_MAP: dict[int, RegionType] = {
    0: RegionType.CAPTION,
    1: RegionType.FOOTNOTE,
    2: RegionType.FORMULA,
    3: RegionType.LIST_ITEM,
    4: RegionType.PAGE_FOOTER,
    5: RegionType.PAGE_HEADER,
    6: RegionType.PICTURE,
    7: RegionType.SECTION_HEADER,
    8: RegionType.TABLE,
    9: RegionType.TEXT,
    10: RegionType.TITLE,
}


@dataclass
class RegionBoundingBox:
    """Bounding box coordinates for a detected region.

    Coordinates are in pixels from the top-left corner of the image.
    """

    x0: float
    """Left edge x-coordinate."""

    y0: float
    """Top edge y-coordinate."""

    x1: float
    """Right edge x-coordinate."""

    y1: float
    """Bottom edge y-coordinate."""

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        """Center point (x, y) of the bounding box."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with bounding box coordinates.
        """
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "height": self.height,
        }

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the bounding box.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if point is inside the box.
        """
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def intersects(self, other: "RegionBoundingBox") -> bool:
        """Check if this bounding box intersects with another.

        Args:
            other: Another bounding box.

        Returns:
            True if the boxes intersect.
        """
        return not (
            self.x1 < other.x0
            or self.x0 > other.x1
            or self.y1 < other.y0
            or self.y0 > other.y1
        )

    def intersection_area(self, other: "RegionBoundingBox") -> float:
        """Calculate the intersection area with another bounding box.

        Args:
            other: Another bounding box.

        Returns:
            Intersection area in square pixels.
        """
        x_left = max(self.x0, other.x0)
        y_top = max(self.y0, other.y0)
        x_right = min(self.x1, other.x1)
        y_bottom = min(self.y1, other.y1)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        return (x_right - x_left) * (y_bottom - y_top)


@dataclass
class LayoutRegion:
    """A detected layout region in a document.

    Represents a classified region with its location and confidence.
    """

    region_type: RegionType
    """Type of the region (e.g., TABLE, TEXT, PICTURE)."""

    bbox: RegionBoundingBox
    """Bounding box coordinates."""

    confidence: float
    """Confidence score (0.0-1.0) for the detection."""

    page_number: int
    """Page number (1-indexed) where this region was found."""

    region_id: str
    """Unique identifier for this region."""

    parent_region_id: str | None = None
    """ID of parent region if nested (e.g., caption within figure)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the region."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with region information.
        """
        return {
            "region_type": self.region_type.value,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "page_number": self.page_number,
            "region_id": self.region_id,
            "parent_region_id": self.parent_region_id,
            "metadata": self.metadata,
        }


@dataclass
class PageLayoutResult:
    """Result of layout detection for a single page."""

    page_number: int
    """Page number (1-indexed)."""

    regions: list[LayoutRegion]
    """List of detected regions on this page."""

    page_width: float
    """Width of the page in pixels."""

    page_height: float
    """Height of the page in pixels."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with page layout result.
        """
        return {
            "page_number": self.page_number,
            "regions": [r.to_dict() for r in self.regions],
            "page_width": self.page_width,
            "page_height": self.page_height,
            "region_count": len(self.regions),
        }

    def get_regions_by_type(self, region_type: RegionType) -> list[LayoutRegion]:
        """Get all regions of a specific type.

        Args:
            region_type: The type of regions to retrieve.

        Returns:
            List of regions matching the type.
        """
        return [r for r in self.regions if r.region_type == region_type]


@dataclass
class LayoutDetectionResult:
    """Result of layout detection for a document."""

    pages: list[PageLayoutResult]
    """Layout detection results for each page."""

    total_pages: int
    """Total number of pages processed."""

    total_regions: int
    """Total number of regions detected across all pages."""

    model_name: str
    """Name of the model used for detection."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the detection."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with layout detection result.
        """
        return {
            "pages": [p.to_dict() for p in self.pages],
            "total_pages": self.total_pages,
            "total_regions": self.total_regions,
            "model_name": self.model_name,
            "metadata": self.metadata,
        }

    def get_all_regions(self) -> list[LayoutRegion]:
        """Get all regions from all pages.

        Returns:
            List of all regions.
        """
        regions: list[LayoutRegion] = []
        for page in self.pages:
            regions.extend(page.regions)
        return regions

    def get_regions_by_type(self, region_type: RegionType) -> list[LayoutRegion]:
        """Get all regions of a specific type across all pages.

        Args:
            region_type: The type of regions to retrieve.

        Returns:
            List of regions matching the type.
        """
        return [r for r in self.get_all_regions() if r.region_type == region_type]


@lru_cache(maxsize=1)
def _load_model(
    model_name: str,
) -> tuple[Any, Any]:
    """Load the layout detection model and processor.

    Uses LRU cache to avoid reloading the model for each request.

    Args:
        model_name: Name of the HuggingFace model.

    Returns:
        Tuple of (model, processor).
    """
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    logger.info(f"Loading layout detection model: {model_name}")

    processor = AutoImageProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    model = AutoModelForObjectDetection.from_pretrained(model_name)

    logger.info(f"Model loaded successfully: {model_name}")
    return model, processor


class LayoutDetector:
    """Detects and segments layout regions in document images.

    Uses Deformable DETR model fine-tuned on DocLayNet for document
    layout analysis. Supports detection of headers, footers, text blocks,
    tables, figures, and other document elements.
    """

    # Default model: Deformable DETR fine-tuned on DocLayNet
    DEFAULT_MODEL = "pascalrai/Deformable-DETR-Document-Layout-Analysis"

    # Default confidence threshold for detections
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: str | None = None,
    ) -> None:
        """Initialize the layout detector.

        Args:
            model_name: HuggingFace model name for layout detection.
            confidence_threshold: Minimum confidence for including detections.
            device: Device to run inference on ('cpu', 'cuda', etc.).
                   If None, automatically selects best available device.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._region_counter = 0

    @property
    def device(self) -> str:
        """Get the device for inference."""
        if self._device is not None:
            return self._device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self._model is None or self._processor is None:
            self._model, self._processor = _load_model(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def _generate_region_id(self, page_number: int) -> str:
        """Generate a unique region ID.

        Args:
            page_number: Page number for the region.

        Returns:
            Unique region ID string.
        """
        self._region_counter += 1
        return f"region_p{page_number}_{self._region_counter}"

    def detect_from_image(
        self,
        image: Image.Image,
        page_number: int = 1,
    ) -> PageLayoutResult:
        """Detect layout regions in a single image.

        Args:
            image: PIL Image to analyze.
            page_number: Page number for this image (default 1).

        Returns:
            PageLayoutResult with detected regions.

        Raises:
            LayoutDetectionError: If detection fails.
        """
        import torch

        try:
            self._ensure_model_loaded()

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_width = float(image.width)
            image_height = float(image.height)

            # Preprocess image
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process results
            target_sizes = torch.tensor([[image.height, image.width]])
            results = self._processor.post_process_object_detection(
                outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes,
            )[0]

            # Convert to LayoutRegion objects
            regions: list[LayoutRegion] = []

            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            boxes = results["boxes"].cpu().numpy()

            for score, label, box in zip(scores, labels, boxes, strict=True):
                # Map label to RegionType
                region_type = DOCLAYNET_LABEL_MAP.get(int(label), RegionType.UNKNOWN)

                # Create bounding box (convert from [x_min, y_min, x_max, y_max])
                bbox = RegionBoundingBox(
                    x0=float(box[0]),
                    y0=float(box[1]),
                    x1=float(box[2]),
                    y1=float(box[3]),
                )

                region = LayoutRegion(
                    region_type=region_type,
                    bbox=bbox,
                    confidence=float(score),
                    page_number=page_number,
                    region_id=self._generate_region_id(page_number),
                )
                regions.append(region)

            # Sort regions by vertical position (top to bottom)
            regions.sort(key=lambda r: (r.bbox.y0, r.bbox.x0))

            # Detect nested regions
            regions = self._detect_nested_regions(regions)

            logger.debug(f"Page {page_number}: detected {len(regions)} regions")

            return PageLayoutResult(
                page_number=page_number,
                regions=regions,
                page_width=image_width,
                page_height=image_height,
            )

        except Exception as e:
            if isinstance(e, LayoutDetectionError):
                raise
            raise LayoutDetectionError(
                f"Layout detection failed: {e}",
                page_number=page_number,
            ) from e

    def _detect_nested_regions(
        self,
        regions: list[LayoutRegion],
    ) -> list[LayoutRegion]:
        """Detect and link nested regions (e.g., captions within figures).

        Args:
            regions: List of detected regions.

        Returns:
            Updated list with parent_region_id set for nested regions.
        """
        # Define containment threshold (percentage of smaller region
        # that must be inside larger region to be considered nested)
        containment_threshold = 0.8

        # Region types that can contain other regions
        container_types = {RegionType.PICTURE, RegionType.TABLE}

        # Region types that can be nested
        nested_types = {RegionType.CAPTION, RegionType.TEXT}

        for region in regions:
            if region.region_type not in nested_types:
                continue

            # Find potential parent containers
            for potential_parent in regions:
                if potential_parent.region_type not in container_types:
                    continue

                if region.region_id == potential_parent.region_id:
                    continue

                # Check if region is mostly inside potential parent
                intersection = region.bbox.intersection_area(potential_parent.bbox)
                if intersection > 0:
                    containment_ratio = intersection / region.bbox.area
                    if containment_ratio >= containment_threshold:
                        region.parent_region_id = potential_parent.region_id
                        break

        return regions

    def detect_from_path(
        self,
        file_path: str | Path,
    ) -> LayoutDetectionResult:
        """Detect layout regions from an image file.

        Args:
            file_path: Path to the image file.

        Returns:
            LayoutDetectionResult with detected regions.

        Raises:
            LayoutDetectionError: If detection fails.
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        image = Image.open(path)
        page_result = self.detect_from_image(image, page_number=1)

        return LayoutDetectionResult(
            pages=[page_result],
            total_pages=1,
            total_regions=len(page_result.regions),
            model_name=self.model_name,
            metadata={"source": str(file_path)},
        )

    def detect_from_content(
        self,
        content: bytes,
        filename: str | None = None,
    ) -> LayoutDetectionResult:
        """Detect layout regions from image content bytes.

        Args:
            content: Image file content as bytes.
            filename: Optional filename for metadata.

        Returns:
            LayoutDetectionResult with detected regions.

        Raises:
            LayoutDetectionError: If detection fails.
        """
        try:
            image = Image.open(io.BytesIO(content))
            page_result = self.detect_from_image(image, page_number=1)

            return LayoutDetectionResult(
                pages=[page_result],
                total_pages=1,
                total_regions=len(page_result.regions),
                model_name=self.model_name,
                metadata={"source": filename},
            )
        except Exception as e:
            if isinstance(e, LayoutDetectionError):
                raise
            raise LayoutDetectionError(
                f"Failed to detect layout from content: {e}",
                file_path=filename,
            ) from e

    def detect_from_images(
        self,
        images: list[Image.Image],
    ) -> LayoutDetectionResult:
        """Detect layout regions from multiple images (multi-page document).

        Args:
            images: List of PIL Images representing document pages.

        Returns:
            LayoutDetectionResult with detected regions for all pages.

        Raises:
            LayoutDetectionError: If detection fails.
        """
        pages: list[PageLayoutResult] = []
        total_regions = 0

        for page_num, image in enumerate(images, start=1):
            page_result = self.detect_from_image(image, page_number=page_num)
            pages.append(page_result)
            total_regions += len(page_result.regions)

        logger.info(
            f"Layout detection complete: {len(pages)} pages, {total_regions} regions"
        )

        return LayoutDetectionResult(
            pages=pages,
            total_pages=len(pages),
            total_regions=total_regions,
            model_name=self.model_name,
        )

    def crop_region(
        self,
        image: Image.Image,
        region: LayoutRegion,
        padding: int = 0,
    ) -> Image.Image:
        """Crop a region from an image.

        Args:
            image: Source image.
            region: Region to crop.
            padding: Optional padding around the region in pixels.

        Returns:
            Cropped image of the region.
        """
        bbox = region.bbox

        # Apply padding while respecting image boundaries
        x0 = max(0, int(bbox.x0) - padding)
        y0 = max(0, int(bbox.y0) - padding)
        x1 = min(image.width, int(bbox.x1) + padding)
        y1 = min(image.height, int(bbox.y1) + padding)

        return image.crop((x0, y0, x1, y1))

    def segment_document(
        self,
        images: list[Image.Image],
    ) -> dict[str, list[tuple[LayoutRegion, Image.Image]]]:
        """Segment a document into regions with cropped images.

        Args:
            images: List of PIL Images representing document pages.

        Returns:
            Dictionary mapping region type to list of (region, cropped_image) tuples.
        """
        result = self.detect_from_images(images)

        segmented: dict[str, list[tuple[LayoutRegion, Image.Image]]] = {}

        for page in result.pages:
            page_image = images[page.page_number - 1]

            for region in page.regions:
                type_key = region.region_type.value
                if type_key not in segmented:
                    segmented[type_key] = []

                cropped = self.crop_region(page_image, region)
                segmented[type_key].append((region, cropped))

        return segmented

    @staticmethod
    def get_supported_region_types() -> list[str]:
        """Get list of supported region types.

        Returns:
            List of region type values.
        """
        return [rt.value for rt in RegionType if rt != RegionType.UNKNOWN]

    @staticmethod
    def get_default_model() -> str:
        """Get the default model name.

        Returns:
            Default model name.
        """
        return LayoutDetector.DEFAULT_MODEL
