"""Tests for the AnalyzeTable tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_table import AnalyzeTable
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionImage,
    RegionType,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
    "ADE_OPENAI_API_KEY", ""
)
SKIP_INTEGRATION = not OPENAI_API_KEY or os.environ.get("ADE_SKIP_REAL_API_TESTS")
INTEGRATION_SKIP_REASON = (
    "OpenAI API key not set (set OPENAI_API_KEY or ADE_OPENAI_API_KEY) "
    "or ADE_SKIP_REAL_API_TESTS is enabled"
)


def create_simple_table_image() -> Image.Image:
    image = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(image)
    for x in (50, 250, 450, 550):
        draw.line((x, 50, x, 250), fill="black", width=2)
    for y in (50, 120, 190, 250):
        draw.line((50, y, 550, y), fill="black", width=2)
    draw.text((70, 70), "Item", fill="black")
    draw.text((270, 70), "Qty", fill="black")
    draw.text((470, 70), "Price", fill="black")
    draw.text((70, 140), "Widget", fill="black")
    draw.text((270, 140), "2", fill="black")
    draw.text((470, 140), "$10", fill="black")
    draw.text((70, 210), "Gadget", fill="black")
    draw.text((270, 210), "5", fill="black")
    draw.text((470, 210), "$7", fill="black")
    return image


def create_complex_table_image() -> Image.Image:
    image = Image.new("RGB", (700, 320), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 660, 280), outline="black", width=2)
    draw.line((40, 90, 660, 90), fill="black", width=2)
    draw.line((40, 160, 660, 160), fill="black", width=2)
    draw.line((40, 230, 660, 230), fill="black", width=2)
    draw.line((240, 40, 240, 280), fill="black", width=2)
    draw.line((460, 40, 460, 280), fill="black", width=2)
    draw.text((60, 60), "Region", fill="black")
    draw.text((280, 60), "Q1-Q2", fill="black")
    draw.text((500, 60), "Q3-Q4", fill="black")
    draw.text((60, 120), "North", fill="black")
    draw.text((280, 120), "120", fill="black")
    draw.text((500, 120), "140", fill="black")
    draw.text((60, 190), "South", fill="black")
    draw.text((280, 190), "90", fill="black")
    draw.text((500, 190), "110", fill="black")
    return image


def create_regions(
    image: Image.Image, *, region_id: str = "region-1"
) -> list[LayoutRegion]:
    bbox = RegionBoundingBox(x0=0, y0=0, x1=image.width, y1=image.height)
    return [
        LayoutRegion(
            region_type=RegionType.TABLE,
            bbox=bbox,
            confidence=0.99,
            page_number=1,
            region_id=region_id,
            region_image=RegionImage(image=image),
        )
    ]


class TestAnalyzeTableTool:
    def test_analyze_table_parses_response(self) -> None:
        response_payload: dict[str, Any] = {
            "headers": ["Item", "Qty", "Price"],
            "rows": [
                {"Item": "Widget", "Qty": "2", "Price": "$10"},
                {"Item": "Gadget", "Qty": "5", "Price": "$7"},
            ],
            "notes": None,
        }

        regions = create_regions(create_simple_table_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_table.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeTable.invoke({"region_id": "region-1", "regions": regions})

        assert result["headers"] == ["Item", "Qty", "Price"]
        assert isinstance(result["rows"], list)

    def test_analyze_table_handles_wrapped_json(self) -> None:
        wrapped = (
            "Table summary:\n"
            + json.dumps({"headers": ["A"], "rows": [{"A": "1"}], "notes": None})
            + "\nEnd."
        )
        regions = create_regions(create_simple_table_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_table.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeTable.invoke({"region_id": "region-1", "regions": regions})

        assert result["headers"] == ["A"]
        assert result["rows"] == [{"A": "1"}]

    def test_analyze_table_invalid_region_id(self) -> None:
        regions = create_regions(create_simple_table_image())
        with pytest.raises(ToolException):
            AnalyzeTable.invoke({"region_id": "missing-region", "regions": regions})

    def test_analyze_table_vlm_failure(self) -> None:
        regions = create_regions(create_simple_table_image())
        with (
            patch(
                "agentic_document_extraction.agents.tools.analyze_table.call_vlm_with_image",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(ToolException),
        ):
            AnalyzeTable.invoke({"region_id": "region-1", "regions": regions})

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_table_integration_simple(self) -> None:
        regions = create_regions(create_simple_table_image())
        result = AnalyzeTable.invoke({"region_id": "region-1", "regions": regions})
        assert result["headers"]
        assert result["rows"]

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_table_integration_complex(self) -> None:
        regions = create_regions(create_complex_table_image())
        result = AnalyzeTable.invoke({"region_id": "region-1", "regions": regions})
        assert result["headers"]
        assert result["rows"]
