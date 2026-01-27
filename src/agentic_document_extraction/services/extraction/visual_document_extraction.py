"""Tool-based extraction service for visual documents.

This module provides functionality to extract structured information from visual
documents (images, PDFs rendered as images) using a tool-based LangChain agent.
The agent receives OCR text in reading order alongside layout metadata and can
invoke VLM tools for charts and tables when OCR is insufficient.

Key features:
- Tool-based extraction using OCR + layout metadata
- Schema-guided extraction for structured output
- Selective VLM tool usage for charts and tables
- Handles complex layouts with reading order context
"""

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from PIL import Image

from agentic_document_extraction.agents.extraction_agent import ExtractionAgent
from agentic_document_extraction.config import settings
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.layout_detector import LayoutDetector
from agentic_document_extraction.services.reading_order_detector import (
    OrderedRegion,
    ReadingOrderDetector,
)
from agentic_document_extraction.services.schema_validator import SchemaInfo
from agentic_document_extraction.utils.agent_helpers import build_agent

logger = logging.getLogger(__name__)


class VisualExtractionError(Exception):
    """Raised when visual extraction fails."""

    def __init__(
        self,
        message: str,
        error_type: str = "visual_extraction_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with message and error details.

        Args:
            message: Error message.
            error_type: Type of error for categorization.
            details: Optional additional error details.
        """
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class VisualDocumentExtractionService:
    """Service for extracting structured data from visual documents.

    Uses layout detection + reading order to provide a tool-based extraction
    agent with OCR text and region metadata. The agent selectively invokes
    chart/table VLM tools when necessary.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the visual document extraction service.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Model name to use. Defaults to settings.openai_model.
            temperature: Sampling temperature. Defaults to settings.openai_temperature.
            max_tokens: Maximum tokens for response. Defaults to settings.openai_max_tokens.
        """
        self.api_key = api_key if api_key is not None else settings.get_openai_api_key()
        self.model = model or settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self.max_tokens = max_tokens or settings.openai_max_tokens

        self._llm: ChatOpenAI | None = None
        self._agent: Any | None = None
        self._tool_agent: ExtractionAgent | None = None
        self._layout_detector: LayoutDetector | None = None
        self._reading_order_detector: ReadingOrderDetector | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LangChain ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            VisualExtractionError: If API key is not configured.
        """
        if self._llm is None:
            if not self.api_key:
                raise VisualExtractionError(
                    "OpenAI API key not configured",
                    error_type="configuration_error",
                    details={"missing": "openai_api_key"},
                )

            self._llm = ChatOpenAI(
                api_key=self.api_key,  # type: ignore[arg-type]
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )

        return self._llm

    @property
    def agent(self) -> Any:
        """Get or create the LangChain agent for visual extraction."""
        if self._agent is None:
            self._agent = build_agent(
                model=self.llm,
                name="visual-extraction-agent",
            )
        return self._agent

    @property
    def tool_agent(self) -> ExtractionAgent:
        """Get or create the tool-based extraction agent."""
        if self._tool_agent is None:
            self._tool_agent = ExtractionAgent(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._tool_agent

    @property
    def layout_detector(self) -> LayoutDetector:
        """Get or create the layout detector."""
        if self._layout_detector is None:
            self._layout_detector = LayoutDetector()
        return self._layout_detector

    @property
    def reading_order_detector(self) -> ReadingOrderDetector:
        """Get or create the reading order detector."""
        if self._reading_order_detector is None:
            self._reading_order_detector = ReadingOrderDetector()
        return self._reading_order_detector

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode a PIL Image to base64 string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64 encoded string of the image.
        """
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _load_image(self, image_source: str | Path | Image.Image) -> Image.Image:
        """Load an image from various sources.

        Args:
            image_source: Path to image file or PIL Image.

        Returns:
            PIL Image object.

        Raises:
            VisualExtractionError: If image cannot be loaded.
        """
        if isinstance(image_source, Image.Image):
            return image_source

        path = Path(image_source)
        if not path.exists():
            raise VisualExtractionError(
                f"Image file not found: {path}",
                error_type="file_not_found",
                details={"path": str(path)},
            )

        try:
            return Image.open(path)
        except Exception as e:
            raise VisualExtractionError(
                f"Failed to load image: {e}",
                error_type="image_load_error",
                details={"path": str(path), "error": str(e)},
            ) from e

    def extract(
        self,
        image_source: str | Path | Image.Image,
        schema_info: SchemaInfo,
        ocr_text: str | None = None,
    ) -> ExtractionResult:
        """Extract structured information from a document image.

        Args:
            image_source: Path to image file or PIL Image.
            schema_info: Validated schema information.
            ocr_text: Optional OCR-extracted text for reference.

        Returns:
            ExtractionResult with extracted data (compatible with text extraction).

        Raises:
            VisualExtractionError: If extraction fails.
        """
        start_time = time.time()

        image = self._load_image(image_source)

        layout_result = self.layout_detector.detect_from_images([image])
        all_regions = layout_result.get_all_regions()
        if all_regions:
            reading_order = self.reading_order_detector.detect_reading_order(
                layout_result.pages
            )
            ordered_regions = [
                ordered
                for page in reading_order.pages
                for ordered in page.ordered_regions
                if not ordered.skip_in_reading
            ]
        else:
            ordered_regions = []

        ordered_text = ocr_text or self._build_text_from_regions(ordered_regions)

        try:
            result = self.tool_agent.extract(
                ordered_text=ordered_text,
                schema_info=schema_info,
                regions=all_regions,
                ordered_regions=ordered_regions,
            )
        except Exception as exc:
            raise VisualExtractionError(
                f"Visual extraction failed: {exc}",
                error_type="tool_agent_error",
                details={"original_error": str(exc)},
            ) from exc

        result.processing_time_seconds = time.time() - start_time
        return result

    def _build_text_from_regions(self, ordered_regions: list[OrderedRegion]) -> str:
        if not ordered_regions:
            return ""
        parts: list[str] = []
        for ordered in ordered_regions:
            text = ordered.region.metadata.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)

    def _parse_json_response(
        self,
        response: str,
        schema_info: SchemaInfo,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Parse JSON response from the model.

        Args:
            response: Raw response string.
            schema_info: Schema information for validation.

        Returns:
            Parsed JSON data.

        Raises:
            VisualExtractionError: If parsing fails.
        """
        try:
            data = json.loads(response)

            if not isinstance(data, dict):
                raise VisualExtractionError(
                    "Expected JSON object response",
                    error_type="parse_error",
                    details={"received_type": type(data).__name__},
                )

            return data

        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            try:
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    extracted = json.loads(json_str)
                    if isinstance(extracted, dict):
                        return extracted
            except json.JSONDecodeError:
                pass

            raise VisualExtractionError(
                f"Failed to parse JSON response: {e}",
                error_type="parse_error",
                details={"response_preview": response[:200]},
            ) from e

    def _build_field_extractions(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
    ) -> list[FieldExtraction]:
        """Build detailed field extraction objects.

        Args:
            data: Extracted data.
            schema_info: Schema information.

        Returns:
            List of FieldExtraction objects.
        """
        extractions: list[FieldExtraction] = []

        for field_info in schema_info.all_fields:
            field_path = field_info.path
            value = self._get_nested_value(data, field_path)

            extractions.append(
                FieldExtraction(
                    field_path=field_path,
                    value=value,
                    confidence=None,
                    source_text=None,
                    reasoning=None,
                )
            )

        return extractions

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get a value from nested dictionary using dot notation.

        Args:
            data: Dictionary to get value from.
            path: Dot-separated path (e.g., 'address.city').

        Returns:
            Value at path or None if not found.
        """
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current
