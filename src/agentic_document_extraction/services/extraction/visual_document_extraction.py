"""VLM-based extraction service for visual documents.

This module provides functionality to extract structured information from visual
documents (images, PDFs rendered as images) using LangChain with OpenAI GPT-4V
vision model. This bypasses OCR limitations by allowing the VLM to "see" the
document directly.

Key features:
- Direct image-to-JSON extraction using GPT-4V
- Schema-guided extraction for structured output
- Combines visual understanding with OCR text for enhanced accuracy
- Handles forms, handwritten text, and complex layouts
"""

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image

from agentic_document_extraction.config import settings
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.schema_validator import SchemaInfo

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
    """Service for extracting structured data from document images using VLM.

    Uses GPT-4V (Vision) to directly analyze document images and extract
    structured information according to a JSON schema. This bypasses OCR
    limitations for forms, handwritten text, and complex layouts.
    """

    EXTRACTION_SYSTEM_PROMPT = """You are an expert document analysis assistant with vision capabilities.
Your task is to extract structured information from the provided document image according to a JSON schema.

CRITICAL RULES:
1. CAREFULLY examine the entire document image to find ALL requested fields
2. For form documents: Labels are usually on the LEFT, values are on the RIGHT or BELOW
3. Look for underlined text, handwritten entries, filled-in boxes, and typed values
4. Extract the ACTUAL VALUES, not the field labels/headers
5. For required fields, make your best effort to extract a value - examine the image carefully
6. If a field is truly not present or illegible, use null for optional fields only
7. Preserve exact text as it appears (including case, punctuation, abbreviations)
8. For dates, preserve the original format
9. For arrays, extract each item separately

FORM DOCUMENT TIPS:
- Field labels like "BRAND(S) APPLICABLE:" should have their VALUE extracted (not the label)
- Look for text that appears AFTER labels, often underlined or in a different column
- Handwritten text and typed text both count as valid values
- Tables often have labels in the left column and values in the right column

You must respond with ONLY valid JSON that matches the schema. Do not include any explanation or text outside the JSON."""

    EXTRACTION_USER_PROMPT = """Extract information from this document image according to this JSON schema:

## JSON Schema:
```json
{schema}
```

## Required Fields (MUST extract these):
{required_fields}

## Optional Fields:
{optional_fields}

{ocr_text_section}

IMPORTANT: Examine the image carefully. For forms, values are typically to the RIGHT of or BELOW their labels.
Look for ALL text in the image including underlined entries, handwritten text, and filled-in fields.

Respond with ONLY the extracted JSON data. Ensure all required fields have values if they appear in the document."""

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

        # Load and encode image
        image = self._load_image(image_source)
        image_base64 = self._encode_image_to_base64(image)

        # Build prompts
        required_fields_str = (
            "\n".join(
                f"- {f.path}: {f.field_type}"
                + (f" - {f.description}" if f.description else "")
                for f in schema_info.required_fields
            )
            or "None"
        )

        optional_fields_str = (
            "\n".join(
                f"- {f.path}: {f.field_type}"
                + (f" - {f.description}" if f.description else "")
                for f in schema_info.optional_fields
            )
            or "None"
        )

        # Add OCR text section if available
        ocr_text_section = ""
        if ocr_text:
            ocr_text_section = f"""
## OCR Text (for reference, may be incomplete):
```
{ocr_text[:4000]}
```
Note: The OCR text above may have missed some values. Always verify against the actual image."""

        user_prompt = self.EXTRACTION_USER_PROMPT.format(
            schema=json.dumps(schema_info.schema, indent=2),
            required_fields=required_fields_str,
            optional_fields=optional_fields_str,
            ocr_text_section=ocr_text_section,
        )

        # Create messages with image
        messages = [
            SystemMessage(content=self.EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ]
            ),
        ]

        try:
            # Call the VLM
            response = self.llm.invoke(messages)

            # Extract response content
            content = response.content
            if not isinstance(content, str):
                content = str(content)

            # Parse JSON response
            extracted_data = self._parse_json_response(content, schema_info)

            # Extract token usage
            usage_metadata = getattr(response, "usage_metadata", None) or {}
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get(
                "total_tokens", prompt_tokens + completion_tokens
            )

            # Build field extractions
            field_extractions = self._build_field_extractions(
                extracted_data, schema_info
            )

            processing_time = time.time() - start_time

            logger.info(
                f"Visual extraction completed: model={self.model}, "
                f"tokens={total_tokens}, time={processing_time:.2f}s"
            )

            return ExtractionResult(
                extracted_data=extracted_data,
                field_extractions=field_extractions,
                model_used=self.model,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                processing_time_seconds=processing_time,
                chunks_processed=1,
                is_chunked=False,
                raw_response=content,
            )

        except Exception as e:
            if isinstance(e, VisualExtractionError):
                raise
            raise VisualExtractionError(
                f"Visual extraction failed: {e}",
                error_type="vlm_error",
                details={"original_error": str(e)},
            ) from e

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
