"""LangChain tool for analyzing embedded images using a VLM."""

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


IMAGE_ANALYSIS_PROMPT = """You are an Image Analysis specialist.
Analyze this embedded image from a document and provide detailed information about its contents.

TASK:
1. Describe what the image shows overall
2. Identify and count distinct objects/items visible
3. For each significant object, extract:
   - Type/category (product, item, equipment, damage, etc.)
   - Description
   - Quantity (if countable)
   - Visible attributes (color, brand, size, condition, model, etc.)
4. Assess overall condition if relevant (excellent, good, fair, poor, damaged)
5. Extract any visible text (product names, labels, signs, serial numbers)

Be specific and factual. If counting items, be precise.
For products, note brand names, model numbers, or identifying features.
For damage assessment, describe the nature and extent of any damage observed.

Return a JSON object with this structure:
```json
{
  "description": "Overall description of what the image shows",
  "objects": [
    {
      "object_type": "product",
      "description": "Red iPhone 14 Pro smartphone",
      "count": 1,
      "attributes": {"color": "red", "brand": "Apple", "model": "iPhone 14 Pro", "condition": "new"},
      "confidence": "high"
    }
  ],
  "total_items": 1,
  "condition_assessment": "excellent",
  "extracted_text": ["iPhone", "Apple", "SN: ABC123"],
  "notes": "Product appears unused in original packaging."
}
```

Field descriptions:
- `description`: Overall image description in 1-2 sentences
- `objects`: List of detected objects with:
  - `object_type`: Category like "product", "item", "equipment", "damage", "vehicle", "property", "document", "person", "other"
  - `description`: Detailed description of the object
  - `count`: Number of this type of object (null if not countable)
  - `attributes`: Dict of observable attributes (color, brand, size, condition, model, material, etc.)
  - `confidence`: Detection confidence - "high", "medium", or "low"
- `total_items`: Total count of all countable items (null if not applicable)
- `condition_assessment`: Overall condition - "excellent", "good", "fair", "poor", "damaged", or null
- `extracted_text`: List of any visible text in the image (null if none)
- `notes`: Additional observations or context (null if none)
"""

IMAGE_ANALYSIS_WITH_FOCUS_PROMPT = """You are an Image Analysis specialist.
Analyze this embedded image from a document and provide detailed information about its contents.

FOCUS: {focus}

Pay special attention to the focus area specified above.

TASK:
1. Describe what the image shows overall
2. Identify and count distinct objects/items visible
3. For each significant object, extract:
   - Type/category (product, item, equipment, damage, etc.)
   - Description
   - Quantity (if countable)
   - Visible attributes (color, brand, size, condition, model, etc.)
4. Assess overall condition if relevant (excellent, good, fair, poor, damaged)
5. Extract any visible text (product names, labels, signs, serial numbers)

Be specific and factual. If counting items, be precise.
For products, note brand names, model numbers, or identifying features.
For damage assessment, describe the nature and extent of any damage observed.

Return a JSON object with this structure:
```json
{{
  "description": "Overall description of what the image shows",
  "objects": [
    {{
      "object_type": "product",
      "description": "Red iPhone 14 Pro smartphone",
      "count": 1,
      "attributes": {{"color": "red", "brand": "Apple", "model": "iPhone 14 Pro", "condition": "new"}},
      "confidence": "high"
    }}
  ],
  "total_items": 1,
  "condition_assessment": "excellent",
  "extracted_text": ["iPhone", "Apple", "SN: ABC123"],
  "notes": "Product appears unused in original packaging."
}}
```

Field descriptions:
- `description`: Overall image description in 1-2 sentences
- `objects`: List of detected objects with:
  - `object_type`: Category like "product", "item", "equipment", "damage", "vehicle", "property", "document", "person", "other"
  - `description`: Detailed description of the object
  - `count`: Number of this type of object (null if not countable)
  - `attributes`: Dict of observable attributes (color, brand, size, condition, model, material, etc.)
  - `confidence`: Detection confidence - "high", "medium", or "low"
- `total_items`: Total count of all countable items (null if not applicable)
- `condition_assessment`: Overall condition - "excellent", "good", "fair", "poor", "damaged", or null
- `extracted_text`: List of any visible text in the image (null if none)
- `notes`: Additional observations or context (null if none)
"""

IMAGE_DEFAULT_RESPONSE: dict[str, Any] = {
    "description": "",
    "objects": [],
    "total_items": None,
    "condition_assessment": None,
    "extracted_text": None,
    "notes": None,
}

VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}
VALID_OBJECT_TYPES = {
    "product",
    "item",
    "equipment",
    "damage",
    "vehicle",
    "property",
    "document",
    "person",
    "other",
}
VALID_CONDITIONS = {"excellent", "good", "fair", "poor", "damaged"}


def analyze_image_impl(
    region_id: str,
    regions: list[LayoutRegion],
    focus: str | None = None,
) -> dict[str, Any]:
    """Core image analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.
        focus: Optional focus area to guide analysis (e.g., "count items", "assess damage").

    Returns:
        Parsed image analysis result dict with objects and metadata.

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

    # Select prompt based on whether focus is provided
    if focus and focus.strip():
        prompt = IMAGE_ANALYSIS_WITH_FOCUS_PROMPT.format(focus=focus.strip())
    else:
        prompt = IMAGE_ANALYSIS_PROMPT

    try:
        response_text = call_vlm_with_image(image_base64, prompt)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeImage VLM call failed: %s", exc)
        raise ToolException("AnalyzeImage VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(IMAGE_DEFAULT_RESPONSE),
        tool_name="AnalyzeImage",
    )

    # Normalize and validate result
    result = _normalize_image_result(result)

    # Add note if region type doesn't match expected image-related types
    image_types = {RegionType.PICTURE}
    if region.region_type not in image_types:
        notes = result.get("notes")
        note = (
            f"Region type is {region.region_type.value}; may not be a picture region."
        )
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


def _normalize_image_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate image analysis result.

    Args:
        result: Raw result dict from VLM response.

    Returns:
        Normalized result dictionary.
    """
    normalized: dict[str, Any] = {}

    # Description (required, default to empty string)
    description = result.get("description")
    if description:
        normalized["description"] = str(description).strip()
    else:
        normalized["description"] = ""

    # Objects list
    objects = result.get("objects")
    if isinstance(objects, list):
        normalized["objects"] = [
            _normalize_object(obj) for obj in objects if isinstance(obj, dict)
        ]
    else:
        normalized["objects"] = []

    # Total items (integer or None)
    total_items = result.get("total_items")
    if total_items is not None:
        try:
            normalized["total_items"] = int(total_items)
        except (ValueError, TypeError):
            normalized["total_items"] = None
    else:
        normalized["total_items"] = None

    # Condition assessment validation
    condition = result.get("condition_assessment")
    if condition and str(condition).lower() in VALID_CONDITIONS:
        normalized["condition_assessment"] = str(condition).lower()
    else:
        normalized["condition_assessment"] = None

    # Extracted text (list of strings)
    extracted_text = result.get("extracted_text")
    if isinstance(extracted_text, list):
        normalized["extracted_text"] = [
            str(t).strip() for t in extracted_text if t and str(t).strip()
        ]
        if not normalized["extracted_text"]:
            normalized["extracted_text"] = None
    else:
        normalized["extracted_text"] = None

    # Notes
    notes = result.get("notes")
    notes_stripped = str(notes).strip() if notes else ""
    normalized["notes"] = notes_stripped or None

    return normalized


def _normalize_object(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single object entry.

    Args:
        obj: Raw object dict from VLM response.

    Returns:
        Normalized object dictionary.
    """
    normalized: dict[str, Any] = {}

    # Object type validation
    object_type = obj.get("object_type")
    if object_type and str(object_type).lower() in VALID_OBJECT_TYPES:
        normalized["object_type"] = str(object_type).lower()
    else:
        normalized["object_type"] = "other"

    # Description
    description = obj.get("description")
    if description:
        normalized["description"] = str(description).strip()
    else:
        normalized["description"] = ""

    # Count (integer or None)
    count = obj.get("count")
    if count is not None:
        try:
            normalized["count"] = int(count)
        except (ValueError, TypeError):
            normalized["count"] = None
    else:
        normalized["count"] = None

    # Attributes (dict or None)
    attributes = obj.get("attributes")
    if isinstance(attributes, dict):
        normalized["attributes"] = {
            str(k): v for k, v in attributes.items() if k and v is not None
        }
        if not normalized["attributes"]:
            normalized["attributes"] = None
    else:
        normalized["attributes"] = None

    # Confidence validation
    confidence = obj.get("confidence")
    if confidence and str(confidence).lower() in VALID_CONFIDENCE_LEVELS:
        normalized["confidence"] = str(confidence).lower()
    else:
        normalized["confidence"] = "low"

    return normalized


@tool("analyze_image")
def AnalyzeImage(
    region_id: str,
    regions: list[LayoutRegion],
    focus: str | None = None,
) -> dict[str, Any]:
    """Analyze embedded image regions to identify objects, count items, and assess condition."""
    return analyze_image_impl(region_id, regions, focus)


@tool("analyze_image_agent")
def analyze_image(
    region_id: str,
    focus: Annotated[str | None, "Optional focus for analysis"] = None,
    state: Annotated[dict[str, Any], InjectedState] = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Analyze photos, illustrations, or general images for content description.

    Use this tool when:
    - The region_type is PICTURE and contains a photo, illustration, or artwork
    - You need to identify objects, people, scenes, or assess image content
    - You need to count items or describe visual attributes

    Do NOT use when:
    - The image is a chart or graph (use analyze_chart_agent)
    - The image is a diagram or flowchart (use analyze_diagram_agent)
    - The image is a logo or seal (use analyze_logo_agent)

    Args:
        region_id: The ID from the Document Regions table (e.g., "region_6")
        focus: Optional specific aspect to analyze (e.g., "count items", "assess damage")

    Returns:
        JSON with description, objects (list), total_items, condition_assessment
    """
    if state is None:
        state = {}
    regions: list[LayoutRegion] = state.get("regions", [])
    return analyze_image_impl(region_id, regions, focus)
