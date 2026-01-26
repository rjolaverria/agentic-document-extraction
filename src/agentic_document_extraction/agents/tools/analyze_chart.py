"""LangChain tool for analyzing charts using a VLM."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agentic_document_extraction.config import settings

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


@lru_cache(maxsize=1)
def _get_vlm() -> ChatOpenAI:
    api_key = settings.get_openai_api_key()
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    return ChatOpenAI(
        api_key=api_key,  # type: ignore[arg-type]
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_completion_tokens=settings.openai_max_tokens,
    )


def _call_vlm_with_image(image_base64: str, prompt: str) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
        ]
    )
    response = _get_vlm().invoke([message])
    return str(response.content)


def _parse_json_response(response_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}") + 1
    if start_idx != -1 and end_idx > start_idx:
        try:
            parsed = json.loads(response_text[start_idx:end_idx])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning("AnalyzeChart returned non-JSON response")
    return {
        "chart_type": "unknown",
        "title": None,
        "x_axis": {"label": None, "ticks": []},
        "y_axis": {"label": None, "ticks": []},
        "key_data_points": [],
        "trends": "Unable to parse structured response.",
        "legend": [],
        "raw_response": response_text[:200],
    }


@tool
def AnalyzeChart(image_base64: str) -> dict[str, Any]:
    """Analyze a chart or figure image using a vision-language model.

    Args:
        image_base64: Base64-encoded image content (PNG/JPEG).

    Returns:
        Parsed JSON object with chart type, axes, key data points, trends, and legend.
    """
    response_text = _call_vlm_with_image(image_base64, CHART_ANALYSIS_PROMPT)
    return _parse_json_response(response_text)
