"""LangChain tool for analyzing charts using a VLM."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import InjectedState

from agentic_document_extraction.agents.tools.vlm_utils import (
    call_vlm_with_image,
    encode_image_to_base64,
    parse_json_response,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionType,
)

logger = logging.getLogger(__name__)


CHART_ANALYSIS_PROMPT = """You are a Chart Analysis specialist.
Analyze this chart/figure image and extract:

1. **Chart Type**: (line, bar, scatter, pie, etc.)
2. **Title**: (if visible)
3. **Axes**: X-axis label, Y-axis label, and tick values
4. **Data Points**: Key values (peaks, troughs, endpoints)
5. **Trends**: Overall pattern description
6. **Legend**: (if present)

Return a JSON object with this structure:
```json
{
  "chart_type": "...",
  "title": "...",
  "x_axis": {"label": "...", "ticks": [...]},
  "y_axis": {"label": "...", "ticks": [...]},
  "key_data_points": [...],
  "trends": "...",
  "legend": [...]
}
```
"""

CHART_DEFAULT_RESPONSE: dict[str, Any] = {
    "chart_type": "unknown",
    "title": None,
    "x_axis": {"label": None, "ticks": []},
    "y_axis": {"label": None, "ticks": []},
    "key_data_points": [],
    "trends": "Unable to parse structured response.",
    "legend": [],
}


def analyze_chart_impl(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Core chart analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.

    Returns:
        Parsed chart analysis result dict.

    Raises:
        ToolException: When the region or its image cannot be found.
    """
    region = next((r for r in regions if r.region_id == region_id), None)
    if region is None:
        raise ToolException(f"Unknown region_id: {region_id}")

    region_image = region.region_image
    if region_image is None:
        raise ToolException(f"Region image not provided for region_id: {region_id}")

    if region_image.base64:
        image_base64 = region_image.base64
    elif region_image.image is not None:
        image_base64 = encode_image_to_base64(region_image.image)
    else:
        raise ToolException(f"Region image missing image/base64 for {region_id}")

    try:
        response_text = call_vlm_with_image(image_base64, CHART_ANALYSIS_PROMPT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeChart VLM call failed: %s", exc)
        raise ToolException("AnalyzeChart VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(CHART_DEFAULT_RESPONSE),
        tool_name="AnalyzeChart",
    )

    if region.region_type != RegionType.PICTURE:
        notes = result.get("notes")
        note = f"Region type is {region.region_type.value}; expected picture/chart."
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


@tool("analyze_chart")
def AnalyzeChart(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze chart/graph regions when OCR text is insufficient."""
    return analyze_chart_impl(region_id, regions)


@tool("analyze_chart_agent")
def analyze_chart(
    region_id: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> dict[str, Any]:
    """Analyze a chart/graph region by its ID to extract structured data
    (chart type, axes, data points, trends). Use when OCR text alone
    cannot capture chart information."""
    regions: list[LayoutRegion] = state.get("regions", [])
    return analyze_chart_impl(region_id, regions)
