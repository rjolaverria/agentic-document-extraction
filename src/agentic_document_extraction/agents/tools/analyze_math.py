"""LangChain tool for analyzing mathematical content regions using a VLM."""

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


MATH_ANALYSIS_PROMPT = """You are a Mathematical Content Transcription specialist.
Analyze this image and extract any mathematical equations, chemical formulas,
or scientific notation present.

**ANALYSIS STEPS:**

1. **CONTENT TYPE** - Identify the type:
   - equation: Mathematical equation or formula (e.g., quadratic formula, calculus)
   - chemical_formula: Chemical formula or reaction (e.g., H2O, NaCl + HCl -> ...)
   - matrix: Matrix or vector notation
   - notation: Scientific notation or physical formulas with units
   - mixed: Contains multiple types of mathematical content

2. **LATEX REPRESENTATION** - Convert to LaTeX:
   - Use standard LaTeX math syntax
   - Preserve structure: fractions (\\frac{}{}), exponents (^{}), subscripts (_{})
   - Use proper symbols: \\sum, \\int, \\sqrt, \\alpha, \\beta, etc.
   - For chemical formulas, use subscripts: H_2O, CO_2
   - For matrices, use \\begin{bmatrix}...\\end{bmatrix}
   - For equation systems, use \\begin{aligned}...\\end{aligned}

3. **PLAIN TEXT DESCRIPTION** - Describe in human-readable terms:
   - What the equation represents
   - The relationship it describes
   - Any physical meaning (if clear from context)

4. **VARIABLES** - Identify key variables and their likely meanings:
   - List variable symbols and their probable meaning
   - Only include variables where meaning is reasonably clear
   - For chemical formulas, list element symbols and compounds

**OUTPUT FORMAT** - Return a JSON object:
```json
{
  "content_type": "equation",
  "latex": "\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
  "plain_text": "The quadratic formula for finding roots of ax^2 + bx + c = 0",
  "variables": {
    "a": "coefficient of x squared",
    "b": "coefficient of x",
    "c": "constant term",
    "x": "unknown variable"
  },
  "notes": "Standard form of the quadratic formula."
}
```

**EXAMPLES:**

1. Chemical formula:
```json
{
  "content_type": "chemical_formula",
  "latex": "2H_2 + O_2 \\rightarrow 2H_2O",
  "plain_text": "Two molecules of hydrogen gas react with one molecule of oxygen gas to produce two molecules of water",
  "variables": {
    "H_2": "hydrogen gas",
    "O_2": "oxygen gas",
    "H_2O": "water"
  },
  "notes": "Combustion reaction of hydrogen."
}
```

2. Matrix:
```json
{
  "content_type": "matrix",
  "latex": "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}",
  "plain_text": "A 2x2 matrix with elements 1, 2, 3, 4",
  "variables": null,
  "notes": null
}
```

3. Scientific notation:
```json
{
  "content_type": "notation",
  "latex": "E = mc^2",
  "plain_text": "Einstein's mass-energy equivalence formula",
  "variables": {
    "E": "energy",
    "m": "mass",
    "c": "speed of light"
  },
  "notes": "Famous equation from special relativity."
}
```

Be precise with LaTeX syntax. If the content is unclear or ambiguous,
note this in the response. Use null for fields that cannot be determined.

Analyze the mathematical content and return accurate transcription.
"""

MATH_DEFAULT_RESPONSE: dict[str, Any] = {
    "content_type": "equation",
    "latex": "",
    "plain_text": "Unable to transcribe mathematical content.",
    "variables": None,
    "notes": None,
}


def analyze_math_impl(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Core mathematical content analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.

    Returns:
        Parsed math analysis result dict with LaTeX, plain text, and variables.

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
        response_text = call_vlm_with_image(image_base64, MATH_ANALYSIS_PROMPT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeMath VLM call failed: %s", exc)
        raise ToolException("AnalyzeMath VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(MATH_DEFAULT_RESPONSE),
        tool_name="AnalyzeMath",
    )

    # Normalize and validate the result
    result = _normalize_math_result(result)

    # Add note if region type doesn't match expected math types
    # Mathematical content typically appears as FORMULA regions
    math_region_types = {RegionType.FORMULA, RegionType.TEXT, RegionType.PICTURE}
    if region.region_type not in math_region_types:
        notes = result.get("notes")
        note = (
            f"Region type is {region.region_type.value}; "
            "may not be a mathematical content region."
        )
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


def _normalize_math_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate math analysis result from VLM response.

    Args:
        result: Raw result dictionary from VLM response.

    Returns:
        Normalized result dictionary.
    """
    # Valid content types
    valid_content_types = {
        "equation",
        "chemical_formula",
        "matrix",
        "notation",
        "mixed",
    }

    # Normalize content_type
    content_type = str(result.get("content_type", "equation")).lower().strip()
    if content_type not in valid_content_types:
        content_type = "equation"
    result["content_type"] = content_type

    # Normalize latex
    latex = result.get("latex")
    if isinstance(latex, str):
        result["latex"] = latex.strip()
    else:
        result["latex"] = ""

    # Normalize plain_text
    plain_text = result.get("plain_text")
    if isinstance(plain_text, str):
        result["plain_text"] = plain_text.strip()
    else:
        result["plain_text"] = ""

    # Normalize variables
    variables = result.get("variables")
    if isinstance(variables, dict):
        normalized_vars: dict[str, str] = {}
        for key, value in variables.items():
            if key is not None and value is not None:
                key_str = str(key).strip()
                value_str = str(value).strip()
                if key_str and value_str:
                    normalized_vars[key_str] = value_str
        result["variables"] = normalized_vars if normalized_vars else None
    else:
        result["variables"] = None

    # Normalize notes
    notes = result.get("notes")
    if isinstance(notes, str):
        notes = notes.strip()
        result["notes"] = notes if notes else None
    else:
        result["notes"] = None

    return result


@tool("analyze_math")
def AnalyzeMath(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze mathematical content regions to extract equations and formulas."""
    return analyze_math_impl(region_id, regions)


@tool("analyze_math_agent")
def analyze_math(
    region_id: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> dict[str, Any]:
    """Analyze a mathematical content region by its ID to extract equations,
    chemical formulas, and scientific notation. Use when you need to
    accurately transcribe mathematical equations, chemical formulas,
    matrices, or scientific notation that standard OCR cannot handle.
    Returns LaTeX representation and plain text description."""
    regions: list[LayoutRegion] = state.get("regions", [])
    return analyze_math_impl(region_id, regions)
