"""LangChain tool for analyzing logos and brand marks using a VLM."""

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


LOGO_ANALYSIS_PROMPT = """You are a Logo and Brand Mark Analysis specialist.
Analyze this logo or brand mark image and identify its characteristics.

TASK:
1. Identify the logo type:
   - company_logo: Company or brand logo
   - certification_badge: Quality or regulatory certification (ISO, FDA, CE, etc.)
   - official_seal: Government, notary, academic, or institutional seal
   - brand_mark: Brand symbol without text
   - trade_mark: Registered trademark symbol or mark

2. For company logos:
   - What company/organization does this represent?
   - What is the brand name visible?

3. For certification badges:
   - What certification does it represent?
   - What standard or authority? (e.g., "ISO 9001", "FDA Approved", "USDA Organic", "CE")

4. For official seals:
   - What institution or authority does it represent?
   - Is it a government, academic, or professional seal?

5. Extract any text:
   - Text within the logo
   - Text immediately adjacent to the logo
   - Taglines or slogans

6. Assess confidence:
   - high: Clear, well-known logo that is easily recognizable
   - medium: Recognizable but less common or partially obscured
   - low: Unclear, unfamiliar, or significantly degraded

Return a JSON object with this structure:
```json
{
  "logo_type": "company_logo",
  "organization_name": "Apple Inc.",
  "description": "Apple logo - stylized apple with a bite taken out",
  "certification_type": null,
  "associated_text": ["Think Different", "Apple"],
  "confidence": "high",
  "notes": "Well-known technology company logo."
}
```

Field descriptions:
- `logo_type`: One of "company_logo", "certification_badge", "official_seal", "brand_mark", "trade_mark"
- `organization_name`: The company, organization, or certifying body name (null if unknown)
- `description`: A brief description of what the logo represents or depicts
- `certification_type`: For certification badges, the specific certification (e.g., "ISO 9001", "FDA Approved")
- `associated_text`: List of text found in or near the logo (null if none)
- `confidence`: Recognition confidence - "high", "medium", or "low"
- `notes`: Additional observations or context (null if none)
"""

LOGO_DEFAULT_RESPONSE: dict[str, Any] = {
    "logo_type": "brand_mark",
    "organization_name": None,
    "description": "",
    "certification_type": None,
    "associated_text": None,
    "confidence": "low",
    "notes": None,
}

VALID_LOGO_TYPES = {
    "company_logo",
    "certification_badge",
    "official_seal",
    "brand_mark",
    "trade_mark",
}
VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}


def analyze_logo_impl(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Core logo analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.

    Returns:
        Parsed logo analysis result dict with logo identification information.

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
        response_text = call_vlm_with_image(image_base64, LOGO_ANALYSIS_PROMPT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeLogo VLM call failed: %s", exc)
        raise ToolException("AnalyzeLogo VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(LOGO_DEFAULT_RESPONSE),
        tool_name="AnalyzeLogo",
    )

    # Normalize and validate result
    result = _normalize_logo_result(result)

    # Add note if region type doesn't match expected logo-related types
    # Logos typically appear in PICTURE regions
    logo_types = {RegionType.PICTURE}
    if region.region_type not in logo_types:
        notes = result.get("notes")
        note = f"Region type is {region.region_type.value}; may not be a logo region."
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


def _normalize_logo_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate logo analysis result.

    Args:
        result: Raw result dict from VLM response.

    Returns:
        Normalized result dictionary.
    """
    normalized: dict[str, Any] = {}

    # Logo type validation
    logo_type = result.get("logo_type")
    if logo_type and str(logo_type).lower() in VALID_LOGO_TYPES:
        normalized["logo_type"] = str(logo_type).lower()
    else:
        normalized["logo_type"] = "brand_mark"

    # Organization name (string or None)
    org_name = result.get("organization_name")
    org_name_stripped = str(org_name).strip() if org_name else ""
    normalized["organization_name"] = org_name_stripped or None

    # Description (required, default to empty string)
    description = result.get("description")
    if description:
        normalized["description"] = str(description).strip()
    else:
        normalized["description"] = ""

    # Certification type (string or None)
    cert_type = result.get("certification_type")
    cert_type_stripped = str(cert_type).strip() if cert_type else ""
    normalized["certification_type"] = cert_type_stripped or None

    # Associated text (list of strings or None)
    associated_text = result.get("associated_text")
    if isinstance(associated_text, list):
        normalized["associated_text"] = [
            str(t).strip() for t in associated_text if t and str(t).strip()
        ]
        if not normalized["associated_text"]:
            normalized["associated_text"] = None
    else:
        normalized["associated_text"] = None

    # Confidence validation
    confidence = result.get("confidence")
    if confidence and str(confidence).lower() in VALID_CONFIDENCE_LEVELS:
        normalized["confidence"] = str(confidence).lower()
    else:
        normalized["confidence"] = "low"

    # Notes
    notes = result.get("notes")
    notes_stripped = str(notes).strip() if notes else ""
    normalized["notes"] = notes_stripped or None

    return normalized


@tool("analyze_logo")
def AnalyzeLogo(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze logo regions to identify company logos, certification badges, and official seals."""
    return analyze_logo_impl(region_id, regions)


@tool("analyze_logo_agent")
def analyze_logo(
    region_id: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> dict[str, Any]:
    """Analyze a logo or brand mark region by its ID to identify company logos,
    certification badges, official seals, and brand marks. Use when you need to
    identify a company or organization from a logo, verify certification badges
    (ISO, FDA, CE, USDA), detect official seals, or extract text associated with
    brand marks."""
    regions: list[LayoutRegion] = state.get("regions", [])
    return analyze_logo_impl(region_id, regions)
