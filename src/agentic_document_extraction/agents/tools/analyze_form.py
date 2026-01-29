"""LangChain tool for analyzing form regions using a VLM."""

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


FORM_ANALYSIS_PROMPT = """You are a Form Analysis specialist.
Analyze this form image and extract all form fields.

For each field found, identify:
1. **Label**: The field label or question text
2. **Value**: The filled value (text, selected option, or checkbox state)
3. **Field Type**: One of: "text", "checkbox", "radio", "dropdown", "signature", "date"
4. **Handwritten**: Whether the value appears to be handwritten (true/false)
5. **Required**: Whether the field appears required (marked with * or "required")
6. **Position**: Approximate position (e.g., "top-left", "middle", "bottom-right")

Pay special attention to:
- Checkbox states: Look for check marks, X marks, or filled boxes vs empty boxes
- Radio buttons: Identify which option is selected (filled circle vs empty)
- Handwritten entries: Text that appears written by hand vs typed
- Signature fields: Areas with signatures or "Sign here" markers

Return a JSON object with this structure:
```json
{
  "fields": [
    {
      "label": "Full Name",
      "value": "John Doe",
      "field_type": "text",
      "is_handwritten": true,
      "is_required": true,
      "position": "top-left"
    },
    {
      "label": "I agree to terms",
      "value": true,
      "field_type": "checkbox",
      "is_handwritten": false,
      "is_required": true,
      "position": "bottom-left"
    }
  ],
  "form_title": "Application Form",
  "notes": "Form appears to be a job application with handwritten entries."
}
```

For checkbox/radio values:
- Use `true` for checked/selected
- Use `false` for unchecked/unselected
- Use the selected option text for radio button groups if you can identify the label

Return all fields found in the form, in reading order (top-to-bottom, left-to-right).
"""

FORM_DEFAULT_RESPONSE: dict[str, Any] = {
    "fields": [],
    "form_title": None,
    "notes": None,
}


def analyze_form_impl(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Core form analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.

    Returns:
        Parsed form analysis result dict with fields, form_title, and notes.

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
        response_text = call_vlm_with_image(image_base64, FORM_ANALYSIS_PROMPT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeForm VLM call failed: %s", exc)
        raise ToolException("AnalyzeForm VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(FORM_DEFAULT_RESPONSE),
        tool_name="AnalyzeForm",
    )

    # Validate and normalize fields
    result["fields"] = _normalize_fields(result.get("fields", []))

    # Add note if region type doesn't match expected form-related types
    # Forms can appear as TEXT, PICTURE, or other region types
    form_types = {RegionType.TEXT, RegionType.PICTURE, RegionType.TABLE}
    if region.region_type not in form_types:
        notes = result.get("notes")
        note = f"Region type is {region.region_type.value}; may not be a form region."
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


def _normalize_fields(fields: Any) -> list[dict[str, Any]]:
    """Normalize and validate form fields from VLM response.

    Args:
        fields: Raw fields list from VLM response.

    Returns:
        Normalized list of field dictionaries.
    """
    if not isinstance(fields, list):
        return []

    normalized = []
    for field in fields:
        if not isinstance(field, dict):
            continue

        # Ensure required keys with defaults
        normalized_field: dict[str, Any] = {
            "label": str(field.get("label", "Unknown")),
            "value": field.get("value"),
            "field_type": str(field.get("field_type", "text")).lower(),
            "is_handwritten": bool(field.get("is_handwritten", False)),
            "is_required": field.get("is_required"),  # Can be None
            "position": field.get("position"),  # Can be None
        }

        # Validate field_type
        valid_types = {"text", "checkbox", "radio", "dropdown", "signature", "date"}
        if normalized_field["field_type"] not in valid_types:
            normalized_field["field_type"] = "text"

        # Normalize checkbox/radio values to boolean
        if normalized_field["field_type"] in ("checkbox", "radio"):
            value = normalized_field["value"]
            if isinstance(value, str):
                value_lower = value.lower()
                if value_lower in ("true", "checked", "yes", "selected", "x", "✓", "☑"):
                    normalized_field["value"] = True
                elif value_lower in ("false", "unchecked", "no", "unselected", "", "☐"):
                    normalized_field["value"] = False
                # Keep string value for radio with option text

        normalized.append(normalized_field)

    return normalized


@tool("analyze_form")
def AnalyzeForm(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze form regions to extract fields including checkboxes, radio buttons, and handwritten entries."""
    return analyze_form_impl(region_id, regions)


try:
    from langchain.tools import InjectedState

    @tool("analyze_form_agent")
    def analyze_form(
        region_id: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> dict[str, Any]:
        """Analyze a form region by its ID to extract structured form data
        including text fields, checkboxes, radio buttons, dropdowns, and
        signature fields. Use when you need to extract form field values,
        especially for checkbox states or handwritten entries."""
        regions: list[LayoutRegion] = state.get("regions", [])
        return analyze_form_impl(region_id, regions)

except ImportError:  # pragma: no cover
    analyze_form = None  # type: ignore[assignment]
