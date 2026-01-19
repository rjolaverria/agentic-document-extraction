"""Output service for generating combined JSON and Markdown outputs.

This module provides the main service for generating complete extraction
outputs including JSON data, Markdown summary, and metadata.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agentic_document_extraction.output.json_generator import (
    JsonGenerator,
    JsonOutputResult,
)
from agentic_document_extraction.output.markdown_generator import (
    MarkdownGenerator,
    MarkdownOutputResult,
)
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    TextExtractionService,
)
from agentic_document_extraction.services.schema_validator import SchemaInfo

logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetadata:
    """Metadata about the extraction process."""

    processing_time_seconds: float = 0.0
    """Total processing time in seconds."""

    model_used: str = ""
    """Name of the model used for extraction."""

    prompt_tokens: int = 0
    """Total prompt tokens used."""

    completion_tokens: int = 0
    """Total completion tokens used."""

    total_tokens: int = 0
    """Total tokens used."""

    chunks_processed: int = 1
    """Number of chunks processed."""

    is_chunked: bool = False
    """Whether the document was chunked."""

    retry_count: int = 0
    """Number of extraction retries performed."""

    validation_passed: bool = True
    """Whether the final output passed validation."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with metadata information.
        """
        return {
            "processing_time_seconds": self.processing_time_seconds,
            "model_used": self.model_used,
            "token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "chunks_processed": self.chunks_processed,
            "is_chunked": self.is_chunked,
            "retry_count": self.retry_count,
            "validation_passed": self.validation_passed,
        }


@dataclass
class ExtractionOutput:
    """Complete extraction output including JSON, Markdown, and metadata."""

    json_result: JsonOutputResult
    """Validated JSON output result."""

    markdown_result: MarkdownOutputResult
    """Generated Markdown output result."""

    metadata: ExtractionMetadata
    """Extraction process metadata."""

    extraction_result: ExtractionResult | None = None
    """Original extraction result (for debugging)."""

    validation_errors: list[str] = field(default_factory=list)
    """Any validation errors encountered."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with complete output information.
        """
        return {
            "data": self.json_result.data,
            "markdown": self.markdown_result.markdown,
            "validation": self.json_result.validation_result.to_dict(),
            "metadata": self.metadata.to_dict(),
            "source_references": self.markdown_result.source_references,
        }

    def to_api_response(self) -> dict[str, Any]:
        """Convert to API response format.

        Returns:
            Dictionary formatted for API response.
        """
        return {
            "extracted_data": self.json_result.data,
            "markdown_summary": self.markdown_result.markdown,
            "is_valid": self.json_result.validation_result.is_valid,
            "validation_errors": self.json_result.validation_result.errors,
            "missing_required_fields": self.json_result.validation_result.missing_required_fields,
            "metadata": self.metadata.to_dict(),
        }


class OutputService:
    """Service for generating complete extraction outputs.

    Coordinates JSON generation, validation, retry logic, and Markdown
    generation to produce complete extraction outputs.
    """

    def __init__(
        self,
        extraction_service: TextExtractionService | None = None,
        json_generator: JsonGenerator | None = None,
        markdown_generator: MarkdownGenerator | None = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize the output service.

        Args:
            extraction_service: Service for text extraction with retry.
            json_generator: Generator for JSON output.
            markdown_generator: Generator for Markdown output.
            max_retries: Maximum number of retries for validation failures.
        """
        self.extraction_service = extraction_service
        self.json_generator = json_generator or JsonGenerator(max_retries=max_retries)
        self.markdown_generator = markdown_generator or MarkdownGenerator()
        self.max_retries = max_retries

    def generate_output(
        self,
        extraction_result: ExtractionResult,
        schema_info: SchemaInfo,
        include_markdown: bool = True,
        include_source_refs: bool = True,
        include_confidence: bool = True,
    ) -> ExtractionOutput:
        """Generate complete output from extraction result.

        Args:
            extraction_result: The extraction result to process.
            schema_info: Schema information for validation.
            include_markdown: Whether to generate Markdown summary.
            include_source_refs: Whether to include source references.
            include_confidence: Whether to include confidence indicators.

        Returns:
            ExtractionOutput with JSON, Markdown, and metadata.
        """
        start_time = time.time()

        # Generate JSON output with validation
        json_result = self.json_generator.generate(
            extraction_result.extracted_data,
            schema_info,
            handle_nulls=True,
        )

        # Generate Markdown output
        if include_markdown:
            markdown_result = self.markdown_generator.generate(
                extraction_result,
                schema_info,
                include_source_refs=include_source_refs,
                include_confidence=include_confidence,
            )
        else:
            markdown_result = MarkdownOutputResult(
                markdown="",
                generated_by_llm=False,
            )

        # Calculate total tokens including markdown generation
        total_prompt_tokens = extraction_result.prompt_tokens
        total_completion_tokens = extraction_result.completion_tokens

        if markdown_result.token_usage:
            total_prompt_tokens += markdown_result.token_usage.get("prompt_tokens", 0)
            total_completion_tokens += markdown_result.token_usage.get(
                "completion_tokens", 0
            )

        total_time = (
            time.time() - start_time + extraction_result.processing_time_seconds
        )

        # Build metadata
        metadata = ExtractionMetadata(
            processing_time_seconds=total_time,
            model_used=extraction_result.model_used,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            chunks_processed=extraction_result.chunks_processed,
            is_chunked=extraction_result.is_chunked,
            retry_count=json_result.retry_count,
            validation_passed=json_result.validation_result.is_valid,
        )

        return ExtractionOutput(
            json_result=json_result,
            markdown_result=markdown_result,
            metadata=metadata,
            extraction_result=extraction_result,
            validation_errors=json_result.validation_result.errors,
        )

    def extract_and_generate(
        self,
        text: str,
        schema_info: SchemaInfo,
        retry_on_validation_failure: bool = True,
        include_markdown: bool = True,
    ) -> ExtractionOutput:
        """Extract from text and generate complete output with retry logic.

        Args:
            text: Text to extract from.
            schema_info: Schema information.
            retry_on_validation_failure: Whether to retry on validation failure.
            include_markdown: Whether to generate Markdown.

        Returns:
            ExtractionOutput with complete results.

        Raises:
            ValidationError: If extraction fails validation after retries.
        """
        if self.extraction_service is None:
            raise ValueError("Extraction service not provided")

        # First extraction attempt
        extraction_result = self.extraction_service.extract(text, schema_info)

        # Validate and potentially retry
        output = self.generate_output(
            extraction_result,
            schema_info,
            include_markdown=include_markdown,
        )

        if (
            output.json_result.validation_result.is_valid
            or not retry_on_validation_failure
        ):
            return output

        # Retry logic
        for retry in range(self.max_retries):
            logger.info(
                f"Validation failed, retry attempt {retry + 1}/{self.max_retries}"
            )

            # Generate feedback for retry
            feedback = self.json_generator.get_validation_feedback(
                output.json_result.validation_result,
                schema_info,
            )

            # Create enhanced text with feedback
            enhanced_text = f"{text}\n\n[IMPORTANT: {feedback}]"

            # Re-extract
            extraction_result = self.extraction_service.extract(
                enhanced_text, schema_info
            )

            # Generate new output
            output = self.generate_output(
                extraction_result,
                schema_info,
                include_markdown=include_markdown,
            )

            output.metadata.retry_count = retry + 1

            if output.json_result.validation_result.is_valid:
                logger.info(f"Validation passed on retry {retry + 1}")
                return output

        logger.warning(
            f"Validation failed after {self.max_retries} retries, "
            "returning best effort result"
        )

        return output

    def format_for_api(
        self,
        output: ExtractionOutput,
    ) -> dict[str, Any]:
        """Format output for API response.

        Args:
            output: Extraction output to format.

        Returns:
            Dictionary formatted for API response.
        """
        return output.to_api_response()
