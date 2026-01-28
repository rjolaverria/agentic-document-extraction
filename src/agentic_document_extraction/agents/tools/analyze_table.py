"""LangChain tool for analyzing tables using a VLM."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.tools import ToolException, tool

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


TABLE_ANALYSIS_PROMPT = """You are a Table Extraction specialist.
Analyze the table image and extract structured data.

Return a JSON object with this structure:
```json
{
  "headers": ["Column A", "Column B", "..."],
  "rows": [
    {"Column A": "value", "Column B": "value"},
    {"Column A": "value", "Column B": "value"}
  ],
  "notes": "Optional notes about merged cells, multi-line headers, or formatting."
}
```
"""

TABLE_DEFAULT_RESPONSE: dict[str, Any] = {
    "headers": [],
    "rows": [],
    "notes": None,
}


def analyze_table_impl(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Core table analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.

    Returns:
        Parsed table analysis result dict.

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
        response_text = call_vlm_with_image(image_base64, TABLE_ANALYSIS_PROMPT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeTable VLM call failed: %s", exc)
        raise ToolException("AnalyzeTable VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(TABLE_DEFAULT_RESPONSE),
        tool_name="AnalyzeTable",
    )

    if region.region_type != RegionType.TABLE:
        notes = result.get("notes")
        note = f"Region type is {region.region_type.value}; expected table."
        result["notes"] = f"{notes} {note}".strip() if notes else note

    if region.metadata.get("spans_pages"):
        notes = result.get("notes")
        note = "Table spans multiple pages; extraction may be partial."
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


@tool("analyze_table")
def AnalyzeTable(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze table regions when OCR text is unreliable or formatting is complex."""
    return analyze_table_impl(region_id, regions)


try:
    from langchain.tools import InjectedState

    @tool("analyze_table_agent")
    def analyze_table(
        region_id: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> dict[str, Any]:
        """Analyze a table region by its ID to extract structured data
        (headers and rows). Use when OCR text alone cannot reliably
        capture table structure."""
        regions: list[LayoutRegion] = state.get("regions", [])
        return analyze_table_impl(region_id, regions)

except ImportError:  # pragma: no cover
    analyze_table = None  # type: ignore[assignment]
