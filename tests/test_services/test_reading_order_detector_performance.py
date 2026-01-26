"""Performance comparisons for reading order detection."""

import time

import pytest

from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    PageLayoutResult,
    RegionBoundingBox,
    RegionType,
)
from agentic_document_extraction.services.reading_order_detector import (
    ReadingOrderDetector,
)


def create_region(
    region_id: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> LayoutRegion:
    return LayoutRegion(
        region_type=RegionType.TEXT,
        bbox=RegionBoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        confidence=0.9,
        page_number=1,
        region_id=region_id,
    )


def create_page_layout(regions: list[LayoutRegion]) -> PageLayoutResult:
    return PageLayoutResult(
        page_number=1,
        regions=regions,
        page_width=1000.0,
        page_height=1200.0,
    )


@pytest.mark.performance
def test_layoutreader_outperforms_legacy_llm_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LayoutReader should be faster than a simulated LLM call."""
    detector = ReadingOrderDetector(model_name="test-layoutreader", device="cpu")
    regions = [
        create_region(f"r{i}", 100.0, 50.0 + i * 10, 900.0, 80.0 + i * 10)
        for i in range(40)
    ]
    page_layout = create_page_layout(regions)

    def fake_predict_orders(boxes: list[list[int]]) -> tuple[list[int], list[float]]:
        return list(range(len(boxes))), [0.9] * len(boxes)

    monkeypatch.setattr(detector, "_predict_orders", fake_predict_orders)

    start = time.perf_counter()
    detector.detect_reading_order_for_page(page_layout)
    layoutreader_time = time.perf_counter() - start

    start = time.perf_counter()
    time.sleep(0.02)
    detector.detect_reading_order_simple(
        page_layout.regions,
        page_width=page_layout.page_width,
        page_height=page_layout.page_height,
    )
    legacy_time = time.perf_counter() - start

    assert layoutreader_time < legacy_time
