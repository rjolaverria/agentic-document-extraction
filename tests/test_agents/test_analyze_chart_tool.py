"""Tests for the AnalyzeChart tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_chart import AnalyzeChart
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


def create_chart_image() -> Image.Image:
    image = Image.new("RGB", (600, 400), color="white")
    draw = ImageDraw.Draw(image)
    draw.line((60, 320, 560, 320), fill="black", width=2)
    draw.line((60, 320, 60, 60), fill="black", width=2)
    bars = [120, 180, 150, 230]
    x_positions = [120, 220, 320, 420]
    for x, height in zip(x_positions, bars, strict=True):
        draw.rectangle((x, 320 - height, x + 40, 320), fill="skyblue", outline="black")
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for x, label in zip(x_positions, labels, strict=True):
        draw.text((x, 330), label, fill="black")
    draw.text((250, 10), "Quarterly Revenue", fill="black")
    draw.text((250, 360), "Quarter", fill="black")
    draw.text((10, 60), "Revenue", fill="black")
    return image


def create_regions(image: Image.Image) -> list[LayoutRegion]:
    bbox = RegionBoundingBox(x0=0, y0=0, x1=image.width, y1=image.height)
    return [
        LayoutRegion(
            region_type=RegionType.PICTURE,
            bbox=bbox,
            confidence=0.99,
            page_number=1,
            region_id="region-1",
            region_image=RegionImage(image=image),
        )
    ]


class TestAnalyzeChartTool:
    def test_analyze_chart_parses_response(self) -> None:
        response_payload: dict[str, Any] = {
            "chart_type": "bar",
            "title": "Quarterly Revenue",
            "x_axis": {"label": "Quarter", "ticks": ["Q1", "Q2", "Q3", "Q4"]},
            "y_axis": {"label": "Revenue", "ticks": [0, 100, 200]},
            "key_data_points": [
                {"label": "Q1", "value": 120},
                {"label": "Q2", "value": 180},
            ],
            "trends": "Revenue increases in Q2 then stabilizes.",
            "legend": ["Revenue"],
        }

        regions = create_regions(create_chart_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_chart.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeChart.invoke({"region_id": "region-1", "regions": regions})

        assert result["chart_type"] == "bar"
        assert result["x_axis"]["label"] == "Quarter"
        assert result["y_axis"]["label"] == "Revenue"
        assert isinstance(result["key_data_points"], list)
        assert result["trends"]

    def test_analyze_chart_parses_wrapped_json(self) -> None:
        wrapped = (
            "Here is the analysis:\n"
            + json.dumps({"chart_type": "line", "trends": "Upward"})
            + "\nThanks!"
        )
        regions = create_regions(create_chart_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_chart.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeChart.invoke({"region_id": "region-1", "regions": regions})

        assert result["chart_type"] == "line"
        assert result["trends"] == "Upward"

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_chart_integration(self) -> None:
        regions = create_regions(create_chart_image())
        result = AnalyzeChart.invoke({"region_id": "region-1", "regions": regions})

        assert result["chart_type"]
        assert result["trends"]
