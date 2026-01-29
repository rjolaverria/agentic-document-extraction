"""LangChain tool for analyzing handwritten text using a VLM."""

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


HANDWRITING_ANALYSIS_PROMPT = """You are a Handwriting Transcription specialist.
Analyze this image containing handwritten text and provide an accurate transcription.

TASK:
1. Transcribe the handwritten text as accurately as possible
2. If uncertain about specific words or letters, provide alternative readings
3. Assess the overall legibility of the handwriting
4. Identify the type of annotation or content
5. Note the handwriting style (cursive, print, mixed)

Pay special attention to:
- Ambiguous letters (e.g., 'a' vs 'o', 'u' vs 'v', 'n' vs 'u')
- Numbers vs letters (e.g., '1' vs 'l', '0' vs 'O')
- Common abbreviations
- Context clues that help disambiguate unclear text

If the handwriting is illegible, indicate this clearly rather than guessing.

Return a JSON object with this structure:
```json
{
  "transcribed_text": "The full transcribed text here",
  "confidence": "high",
  "annotation_type": "margin_note",
  "position": "right_margin",
  "is_legible": true,
  "alternative_readings": ["possible alternate word 1", "possible alternate word 2"],
  "style": "cursive",
  "notes": "Neat cursive handwriting in blue ink."
}
```

Field descriptions:
- `transcribed_text`: The complete transcription of all handwritten text
- `confidence`: Overall confidence level - "high", "medium", or "low"
- `annotation_type`: Type of handwritten content - one of:
  - "margin_note": Notes written in margins
  - "correction": Edits or corrections to existing text
  - "answer": Handwritten answer to a question or form field
  - "signature_text": Text accompanying a signature
  - "comment": General comments or remarks
  - "label": Labels or captions
  - "other": Any other type
- `position`: Where the handwriting appears (e.g., "top_margin", "bottom", "right_margin", "inline", "full_page")
- `is_legible`: true if the handwriting is readable, false if too unclear
- `alternative_readings`: List of possible alternative transcriptions for unclear portions (null if confident)
- `style`: Handwriting style - "cursive", "print", "mixed", or "unknown"
- `notes`: Additional observations about the handwriting (ink color, neatness, language, etc.)

If the text is not legible, still provide your best attempt in `transcribed_text` but set `is_legible` to false and `confidence` to "low".
"""

HANDWRITING_WITH_CONTEXT_PROMPT = """You are a Handwriting Transcription specialist.
Analyze this image containing handwritten text and provide an accurate transcription.

SURROUNDING CONTEXT: {context}

Use the surrounding context to help disambiguate unclear words or abbreviations.

TASK:
1. Transcribe the handwritten text as accurately as possible
2. If uncertain about specific words or letters, provide alternative readings
3. Assess the overall legibility of the handwriting
4. Identify the type of annotation or content
5. Note the handwriting style (cursive, print, mixed)

Pay special attention to:
- Ambiguous letters (e.g., 'a' vs 'o', 'u' vs 'v', 'n' vs 'u')
- Numbers vs letters (e.g., '1' vs 'l', '0' vs 'O')
- Common abbreviations
- Context clues that help disambiguate unclear text

If the handwriting is illegible, indicate this clearly rather than guessing.

Return a JSON object with this structure:
```json
{{
  "transcribed_text": "The full transcribed text here",
  "confidence": "high",
  "annotation_type": "margin_note",
  "position": "right_margin",
  "is_legible": true,
  "alternative_readings": ["possible alternate word 1", "possible alternate word 2"],
  "style": "cursive",
  "notes": "Neat cursive handwriting in blue ink."
}}
```

Field descriptions:
- `transcribed_text`: The complete transcription of all handwritten text
- `confidence`: Overall confidence level - "high", "medium", or "low"
- `annotation_type`: Type of handwritten content - one of:
  - "margin_note": Notes written in margins
  - "correction": Edits or corrections to existing text
  - "answer": Handwritten answer to a question or form field
  - "signature_text": Text accompanying a signature
  - "comment": General comments or remarks
  - "label": Labels or captions
  - "other": Any other type
- `position`: Where the handwriting appears (e.g., "top_margin", "bottom", "right_margin", "inline", "full_page")
- `is_legible`: true if the handwriting is readable, false if too unclear
- `alternative_readings`: List of possible alternative transcriptions for unclear portions (null if confident)
- `style`: Handwriting style - "cursive", "print", "mixed", or "unknown"
- `notes`: Additional observations about the handwriting (ink color, neatness, language, etc.)

If the text is not legible, still provide your best attempt in `transcribed_text` but set `is_legible` to false and `confidence` to "low".
"""

HANDWRITING_DEFAULT_RESPONSE: dict[str, Any] = {
    "transcribed_text": "",
    "confidence": "low",
    "annotation_type": None,
    "position": None,
    "is_legible": False,
    "alternative_readings": None,
    "style": None,
    "notes": None,
}

VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}
VALID_ANNOTATION_TYPES = {
    "margin_note",
    "correction",
    "answer",
    "signature_text",
    "comment",
    "label",
    "other",
}
VALID_STYLES = {"cursive", "print", "mixed", "unknown"}


def analyze_handwriting_impl(
    region_id: str,
    regions: list[LayoutRegion],
    context: str | None = None,
) -> dict[str, Any]:
    """Core handwriting analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.
        context: Optional surrounding context to help with transcription.

    Returns:
        Parsed handwriting analysis result dict with transcription and metadata.

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

    # Select prompt based on whether context is provided
    if context and context.strip():
        prompt = HANDWRITING_WITH_CONTEXT_PROMPT.format(context=context.strip())
    else:
        prompt = HANDWRITING_ANALYSIS_PROMPT

    try:
        response_text = call_vlm_with_image(image_base64, prompt)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeHandwriting VLM call failed: %s", exc)
        raise ToolException("AnalyzeHandwriting VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(HANDWRITING_DEFAULT_RESPONSE),
        tool_name="AnalyzeHandwriting",
    )

    # Normalize and validate result
    result = _normalize_handwriting_result(result)

    # Add note if region type doesn't match expected handwriting-related types
    # Handwriting can appear in TEXT or PICTURE regions
    handwriting_types = {RegionType.TEXT, RegionType.PICTURE}
    if region.region_type not in handwriting_types:
        notes = result.get("notes")
        note = (
            f"Region type is {region.region_type.value}; "
            "may not be a handwriting region."
        )
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


def _normalize_handwriting_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate handwriting analysis result.

    Args:
        result: Raw result dict from VLM response.

    Returns:
        Normalized result dictionary.
    """
    normalized: dict[str, Any] = {}

    # Transcribed text (required, default to empty string)
    transcribed = result.get("transcribed_text")
    if transcribed:
        normalized["transcribed_text"] = str(transcribed).strip()
    else:
        normalized["transcribed_text"] = ""

    # Confidence validation
    confidence = result.get("confidence")
    if confidence and str(confidence).lower() in VALID_CONFIDENCE_LEVELS:
        normalized["confidence"] = str(confidence).lower()
    else:
        normalized["confidence"] = "low"

    # Annotation type validation
    annotation_type = result.get("annotation_type")
    if annotation_type and str(annotation_type).lower() in VALID_ANNOTATION_TYPES:
        normalized["annotation_type"] = str(annotation_type).lower()
    else:
        normalized["annotation_type"] = None

    # Position (free-form string, can be None)
    position = result.get("position")
    position_stripped = str(position).strip() if position else ""
    normalized["position"] = position_stripped or None

    # Is legible (boolean)
    normalized["is_legible"] = bool(result.get("is_legible", False))

    # Alternative readings (list of strings)
    alternatives = result.get("alternative_readings")
    if isinstance(alternatives, list):
        normalized["alternative_readings"] = [
            str(a).strip() for a in alternatives if a and str(a).strip()
        ]
        if not normalized["alternative_readings"]:
            normalized["alternative_readings"] = None
    else:
        normalized["alternative_readings"] = None

    # Style validation
    style = result.get("style")
    if style and str(style).lower() in VALID_STYLES:
        normalized["style"] = str(style).lower()
    else:
        normalized["style"] = None

    # Notes
    notes = result.get("notes")
    notes_stripped = str(notes).strip() if notes else ""
    normalized["notes"] = notes_stripped or None

    return normalized


@tool("analyze_handwriting")
def AnalyzeHandwriting(
    region_id: str,
    regions: list[LayoutRegion],
    context: str | None = None,
) -> dict[str, Any]:
    """Analyze handwritten text regions to transcribe handwriting and identify annotation types."""
    return analyze_handwriting_impl(region_id, regions, context)


try:
    from langchain.tools import InjectedState

    @tool("analyze_handwriting_agent")
    def analyze_handwriting(
        region_id: str,
        context: Annotated[str | None, "Optional surrounding context"] = None,
        state: Annotated[dict[str, Any], InjectedState] = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Analyze a handwritten text region by its ID to transcribe the handwriting.
        Use when you need to read handwritten notes, margin annotations, corrections,
        form answers, or any other handwritten content that standard OCR may struggle with.
        Optionally provide surrounding context to help disambiguate unclear text."""
        if state is None:
            state = {}
        regions: list[LayoutRegion] = state.get("regions", [])
        return analyze_handwriting_impl(region_id, regions, context)

except ImportError:  # pragma: no cover
    analyze_handwriting = None  # type: ignore[assignment]
