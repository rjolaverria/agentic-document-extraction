from __future__ import annotations

import pytest

from agentic_document_extraction.models import (
    FormatFamily,
    FormatInfo,
    ProcessingCategory,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionImage,
    RegionType,
)
from agentic_document_extraction.services.schema_validator import FieldInfo, SchemaInfo


@pytest.fixture
def schema_info_basic() -> SchemaInfo:
    """Minimal schema used across agent tests."""
    schema = {
        "type": "object",
        "title": "Basic Schema",
        "properties": {
            "name": {"type": "string"},
            "amount": {"type": "number"},
        },
        "required": ["name"],
    }
    return SchemaInfo(
        schema=schema,
        required_fields=[FieldInfo("name", "string", required=True, path="name")],
        optional_fields=[FieldInfo("amount", "number", required=False, path="amount")],
        schema_type="object",
    )


@pytest.fixture
def format_info_text() -> FormatInfo:
    return FormatInfo(
        mime_type="text/plain",
        format_family=FormatFamily.PLAIN_TEXT,
        processing_category=ProcessingCategory.TEXT_BASED,
        extension=".txt",
    )


@pytest.fixture
def format_info_visual() -> FormatInfo:
    return FormatInfo(
        mime_type="image/png",
        format_family=FormatFamily.IMAGE,
        processing_category=ProcessingCategory.VISUAL,
        extension=".png",
    )


@pytest.fixture
def region_bbox() -> RegionBoundingBox:
    return RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)


@pytest.fixture
def text_only_regions(region_bbox: RegionBoundingBox) -> list[LayoutRegion]:
    return [
        LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=region_bbox,
            confidence=0.9,
            page_number=1,
            region_id="text-1",
        ),
        LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=region_bbox,
            confidence=0.88,
            page_number=1,
            region_id="text-2",
        ),
    ]


@pytest.fixture
def visual_regions_with_chart(region_bbox: RegionBoundingBox) -> list[LayoutRegion]:
    return [
        LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=region_bbox,
            confidence=0.9,
            page_number=1,
            region_id="text-1",
        ),
        LayoutRegion(
            region_type=RegionType.PICTURE,
            bbox=region_bbox,
            confidence=0.95,
            page_number=1,
            region_id="chart-1",
            region_image=RegionImage(image=None, base64="chart_base64"),
        ),
    ]


@pytest.fixture
def visual_regions_with_table(region_bbox: RegionBoundingBox) -> list[LayoutRegion]:
    return [
        LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=region_bbox,
            confidence=0.9,
            page_number=1,
            region_id="text-1",
        ),
        LayoutRegion(
            region_type=RegionType.TABLE,
            bbox=region_bbox,
            confidence=0.92,
            page_number=1,
            region_id="table-1",
            region_image=RegionImage(image=None, base64="table_base64"),
        ),
    ]


@pytest.fixture
def mixed_regions(region_bbox: RegionBoundingBox) -> list[LayoutRegion]:
    return [
        LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=region_bbox,
            confidence=0.9,
            page_number=1,
            region_id="text-1",
        ),
        LayoutRegion(
            region_type=RegionType.PICTURE,
            bbox=region_bbox,
            confidence=0.95,
            page_number=1,
            region_id="chart-1",
            region_image=RegionImage(image=None, base64="chart_base64"),
        ),
        LayoutRegion(
            region_type=RegionType.TABLE,
            bbox=region_bbox,
            confidence=0.92,
            page_number=1,
            region_id="table-1",
            region_image=RegionImage(image=None, base64="table_base64"),
        ),
        LayoutRegion(
            region_type=RegionType.FORMULA,
            bbox=region_bbox,
            confidence=0.85,
            page_number=1,
            region_id="formula-1",
        ),
    ]


@pytest.fixture
def simple_invoice_text() -> str:
    return """\
INVOICE #12345
Date: 2024-01-15
Total: $500.00
"""


@pytest.fixture
def invoice_with_chart_text() -> str:
    return """\
INVOICE #12345
[Chart showing monthly sales - data not in OCR text]
"""
