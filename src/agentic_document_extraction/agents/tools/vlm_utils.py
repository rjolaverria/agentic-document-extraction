"""Shared utilities for VLM tool interactions."""

from __future__ import annotations

import base64
import io
import json
import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image

from agentic_document_extraction.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vlm() -> ChatOpenAI:
    api_key = settings.get_openai_api_key()
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    return ChatOpenAI(
        api_key=api_key,  # type: ignore[arg-type]
        model=settings.openai_vlm_model,
        temperature=settings.openai_temperature,
        max_completion_tokens=settings.openai_max_tokens,
    )


def call_vlm_with_image(image_base64: str, prompt: str) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
        ]
    )
    response = get_vlm().invoke([message])
    return str(response.content)


def encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_json_response(
    response_text: str,
    *,
    default: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
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

    logger.warning("%s returned non-JSON response", tool_name)
    fallback = dict(default)
    fallback.setdefault("raw_response", response_text[:200])
    return fallback
