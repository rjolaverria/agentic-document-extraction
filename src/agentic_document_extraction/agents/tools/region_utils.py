"""Shared utilities for region-based tool execution."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from agentic_document_extraction.services.layout_detector import (
    LayoutDetectionResult,
    LayoutRegion,
)


def crop_region_from_image(
    image: Image.Image,
    region: LayoutRegion,
    *,
    padding: int = 0,
) -> Image.Image:
    """Crop a layout region from an image with optional padding."""
    bbox = region.bbox

    x0 = max(0, int(bbox.x0) - padding)
    y0 = max(0, int(bbox.y0) - padding)
    x1 = min(image.width, int(bbox.x1) + padding)
    y1 = min(image.height, int(bbox.y1) + padding)

    return image.crop((x0, y0, x1, y1))


class RegionImageStore:
    """Lookup and crop images for layout regions."""

    def __init__(
        self,
        *,
        layout_result: LayoutDetectionResult,
        page_images: Sequence[Image.Image],
        padding: int = 0,
    ) -> None:
        self._layout_result = layout_result
        self._page_images = list(page_images)
        self._padding = padding
        self._region_lookup = {
            region.region_id: region for region in layout_result.get_all_regions()
        }

    def get_region_image(self, region_id: str) -> tuple[LayoutRegion, Image.Image]:
        """Return the region metadata and cropped image for a region id."""
        region = self._region_lookup.get(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")

        page_index = region.page_number - 1
        if page_index < 0 or page_index >= len(self._page_images):
            raise ValueError(
                f"Region {region_id} references page {region.page_number}, "
                "but no image is available."
            )

        page_image = self._page_images[page_index]
        cropped = crop_region_from_image(page_image, region, padding=self._padding)
        return region, cropped
