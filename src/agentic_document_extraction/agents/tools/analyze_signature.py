"""LangChain tool for analyzing signature blocks using a VLM."""

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


SIGNATURE_ANALYSIS_PROMPT = """You are a Signature Block Analysis specialist.
Analyze this signature region image and identify all signature-related elements.

For each element found, identify:

1. SIGNATURE:
   - Is a handwritten signature present?
   - Is there a printed name below/near the signature?
   - Is there a title or role indicated?
   - Is there a date signed?
   - Is there a location/city indicated?

2. STAMPS/SEALS:
   - Are there any official stamps or seals?
   - What type? (company_seal, notary, certification, watermark)
   - What text appears on the stamp?

3. CERTIFICATION MARKS:
   - Are there any certification marks or logos? (e.g., ISO 9001, Certified Copy)

4. COMPLETENESS:
   - Are all expected signature block elements present?
   - What is missing (if anything)?

Return a JSON object with this structure:
```json
{
  "signature_present": true,
  "signer_name": "John Smith",
  "signer_title": "CEO",
  "date_signed": "January 15, 2024",
  "location": "New York, NY",

  "stamp_present": true,
  "stamp_type": "company_seal",
  "stamp_text": "ACME Corporation Official Seal",

  "certification_marks": ["ISO 9001", "Certified Copy"],

  "is_complete": true,
  "missing_elements": null,
  "notes": "Signature block appears complete with all standard elements."
}
```

Important guidelines:
- `signature_present`: true if any handwritten or digital signature mark is visible
- `signer_name`: The printed name if visible (not the handwritten signature itself)
- `signer_title`: Job title or role (e.g., "CEO", "Witness", "Notary Public")
- `date_signed`: The date in the format shown in the document
- `location`: City/state/country if indicated
- `stamp_type`: One of "company_seal", "notary", "certification", "watermark", or null
- `is_complete`: true if all typical signature block elements are present
- `missing_elements`: List of missing elements (e.g., ["date", "printed name", "title"])

Return structured information about all signature-related elements found.
"""

SIGNATURE_DEFAULT_RESPONSE: dict[str, Any] = {
    "signature_present": False,
    "signer_name": None,
    "signer_title": None,
    "date_signed": None,
    "location": None,
    "stamp_present": False,
    "stamp_type": None,
    "stamp_text": None,
    "certification_marks": None,
    "is_complete": False,
    "missing_elements": None,
    "notes": None,
}

VALID_STAMP_TYPES = {"company_seal", "notary", "certification", "watermark"}


def analyze_signature_impl(
    region_id: str, regions: list[LayoutRegion]
) -> dict[str, Any]:
    """Core signature analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.

    Returns:
        Parsed signature analysis result dict with signature and stamp information.

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
        response_text = call_vlm_with_image(image_base64, SIGNATURE_ANALYSIS_PROMPT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeSignature VLM call failed: %s", exc)
        raise ToolException("AnalyzeSignature VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(SIGNATURE_DEFAULT_RESPONSE),
        tool_name="AnalyzeSignature",
    )

    # Normalize and validate result
    result = _normalize_signature_result(result)

    # Add note if region type doesn't match expected signature-related types
    # Signatures typically appear in TEXT or PICTURE regions
    signature_types = {RegionType.TEXT, RegionType.PICTURE}
    if region.region_type not in signature_types:
        notes = result.get("notes")
        note = (
            f"Region type is {region.region_type.value}; may not be a signature region."
        )
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


def _normalize_signature_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate signature analysis result.

    Args:
        result: Raw result dict from VLM response.

    Returns:
        Normalized result dictionary.
    """
    normalized: dict[str, Any] = {}

    # Boolean fields
    normalized["signature_present"] = bool(result.get("signature_present", False))
    normalized["stamp_present"] = bool(result.get("stamp_present", False))
    normalized["is_complete"] = bool(result.get("is_complete", False))

    # String fields (can be None)
    for field in ("signer_name", "signer_title", "date_signed", "location"):
        value = result.get(field)
        stripped = str(value).strip() if value else ""
        normalized[field] = stripped or None

    # Stamp type validation
    stamp_type = result.get("stamp_type")
    if stamp_type and str(stamp_type).lower() in VALID_STAMP_TYPES:
        normalized["stamp_type"] = str(stamp_type).lower()
    else:
        normalized["stamp_type"] = None

    # Stamp text
    stamp_text = result.get("stamp_text")
    stamp_text_stripped = str(stamp_text).strip() if stamp_text else ""
    normalized["stamp_text"] = stamp_text_stripped or None

    # Certification marks (list of strings)
    cert_marks = result.get("certification_marks")
    if isinstance(cert_marks, list):
        normalized["certification_marks"] = [
            str(m).strip() for m in cert_marks if m and str(m).strip()
        ]
        if not normalized["certification_marks"]:
            normalized["certification_marks"] = None
    else:
        normalized["certification_marks"] = None

    # Missing elements (list of strings)
    missing = result.get("missing_elements")
    if isinstance(missing, list):
        normalized["missing_elements"] = [
            str(m).strip() for m in missing if m and str(m).strip()
        ]
        if not normalized["missing_elements"]:
            normalized["missing_elements"] = None
    else:
        normalized["missing_elements"] = None

    # Notes
    notes = result.get("notes")
    notes_stripped = str(notes).strip() if notes else ""
    normalized["notes"] = notes_stripped or None

    return normalized


@tool("analyze_signature")
def AnalyzeSignature(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze signature regions to extract signature and stamp information."""
    return analyze_signature_impl(region_id, regions)


@tool("analyze_signature_agent")
def analyze_signature(
    region_id: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> dict[str, Any]:
    """Analyze a signature block region by its ID to extract signature information
    including signer name, title, date, stamps, seals, and certification marks.
    Use when you need to verify signature presence, extract signer details,
    or identify official stamps and certifications."""
    regions: list[LayoutRegion] = state.get("regions", [])
    return analyze_signature_impl(region_id, regions)
