"""Extraction processor for background job processing.

This module provides the main extraction orchestration logic that:
1. Detects document format
2. Extracts text using appropriate method (text-based or visual)
3. Runs the agentic extraction loop
4. Returns results for Docket storage

This is the entry point for processing extraction jobs in background tasks.
"""

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from docket import Progress

from agentic_document_extraction.agents.extraction_agent import ExtractionAgent
from agentic_document_extraction.agents.refiner import AgenticLoop
from agentic_document_extraction.config import settings as app_settings
from agentic_document_extraction.models import ProcessingCategory
from agentic_document_extraction.output.json_generator import JsonGenerator
from agentic_document_extraction.output.markdown_generator import MarkdownGenerator
from agentic_document_extraction.services.format_detector import FormatDetector
from agentic_document_extraction.services.schema_validator import SchemaValidator
from agentic_document_extraction.services.text_extractor import TextExtractor
from agentic_document_extraction.services.visual_text_extractor import (
    VisualTextExtractor,
)
from agentic_document_extraction.utils.exceptions import (
    ADEFileNotFoundError,
    DocumentProcessingError,
)
from agentic_document_extraction.utils.logging import (
    LogContext,
    PerformanceMetrics,
    get_logger,
    set_job_id,
)

logger = get_logger(__name__)
_DEFAULT_PROGRESS = Progress()


def _get_nested_value(data: dict[str, Any], path: str) -> Any:
    """Get a value from nested dictionary using dot notation.

    Args:
        data: Dictionary to get value from.
        path: Dot-separated path (e.g., 'address.city').

    Returns:
        Value at path or None if not found.
    """
    keys = path.split(".")
    current: Any = data

    for key in keys:
        is_array = key.endswith("[]")
        base_key = key[:-2] if is_array else key

        if not isinstance(current, dict) or base_key not in current:
            return None

        current = current[base_key]

        if is_array:
            if not isinstance(current, list):
                return None
            # For array paths, return the first element's value
            if current:
                current = current[0]
            else:
                return None

    return current


async def _set_progress(progress: Progress | None, message: str) -> None:
    """Update Docket progress messages when available."""
    if progress is None:
        return
    # Progress dependency is only valid inside a Docket worker context.
    with suppress(AssertionError):
        await progress.set_message(message)


# Re-export DocumentProcessingError as ExtractionProcessorError for backward compatibility
ExtractionProcessorError = DocumentProcessingError


async def process_extraction_job(
    job_id: str,
    filename: str,
    file_path: str,
    schema_path: str,
    progress: Progress | None = _DEFAULT_PROGRESS,
) -> dict[str, Any]:
    """Process an extraction job.

    This is the main entry point for background job processing. It:
    1. Loads schema and detects document format
    2. Detects document format
    3. Extracts text using appropriate method
    4. Runs the agentic extraction loop
    5. Returns results for Docket result storage

    Args:
        job_id: The job ID to process.
        filename: Original uploaded filename.
        file_path: Path to the uploaded file.
        schema_path: Path to the schema file.
        progress: Optional Docket progress reporter.
    """
    # Set job ID in context for logging correlation
    set_job_id(job_id)
    metrics = PerformanceMetrics(operation="extraction")

    try:
        with LogContext(job_id=job_id, filename=filename):
            logger.info(
                "Starting extraction",
                filename=filename,
            )

            await _set_progress(progress, "Processing started")

            # Load schema from file
            schema_file_path = Path(schema_path)
            if not schema_file_path.exists():
                raise ADEFileNotFoundError(
                    file_path=str(schema_file_path),
                    message=f"Schema file not found: {schema_file_path}",
                )

            with open(schema_file_path) as f:
                schema_content = json.load(f)

            # Validate schema
            schema_validator = SchemaValidator()
            schema_info = schema_validator.validate(schema_content)
            logger.info("Schema validated")

            await _set_progress(progress, "Schema validated, detecting document format")

            # Detect document format
            document_path = Path(file_path)
            if not document_path.exists():
                raise ADEFileNotFoundError(
                    file_path=str(document_path),
                    message=f"Document file not found: {document_path}",
                )

            format_detector = FormatDetector()
            format_info = format_detector.detect_from_path(document_path)
            logger.info(
                "Format detected",
                mime_type=format_info.mime_type,
                category=format_info.processing_category.value,
            )

            await _set_progress(
                progress,
                (
                    "Document format: "
                    f"{format_info.format_family.value}, extracting text"
                ),
            )

            # Extract text based on document type
            if format_info.processing_category == ProcessingCategory.TEXT_BASED:
                # Text-based extraction
                text_extractor = TextExtractor()
                text_extraction_result = text_extractor.extract_from_path(document_path)
                text = text_extraction_result.text
                logger.info("Text extracted", length=len(text))
            else:
                # Visual document - use OCR and layout detection
                visual_text_extractor = VisualTextExtractor()
                visual_extraction_result = visual_text_extractor.extract_from_path(
                    document_path
                )
                text = visual_extraction_result.full_text
                logger.info(
                    "Text extracted from visual document",
                    length=len(text),
                    method=visual_extraction_result.extraction_method.value,
                    pages=visual_extraction_result.total_pages,
                    confidence=visual_extraction_result.average_confidence,
                )

            await _set_progress(progress, "Running agentic extraction loop")

            if app_settings.use_tool_agent:
                # New: single tool-using ExtractionAgent
                extraction_agent = ExtractionAgent()

                # For visual documents, run layout detection to get regions
                layout_regions = None
                if format_info.processing_category == ProcessingCategory.VISUAL:
                    try:
                        from agentic_document_extraction.services.layout_detector import (
                            LayoutDetector,
                            LayoutRegion,
                            RegionBoundingBox,
                            RegionImage,
                            RegionType,
                        )

                        layout_detector = LayoutDetector()
                        layout_result = layout_detector.detect_from_path(document_path)
                        layout_regions = layout_result.get_all_regions()

                        # Attach cropped images to visual regions (TABLE, PICTURE)
                        visual_types = {RegionType.TABLE, RegionType.PICTURE}

                        import base64
                        import io

                        from PIL import Image

                        source_image: Image.Image = Image.open(document_path)
                        if source_image.mode != "RGB":
                            source_image = source_image.convert("RGB")

                        # Fallback: if no regions detected, treat whole image as PICTURE
                        # This handles standalone charts/images where layout model
                        # can't find distinct regions
                        if not layout_regions:
                            buffer = io.BytesIO()
                            source_image.save(buffer, format="PNG")
                            b64_str = base64.b64encode(buffer.getvalue()).decode(
                                "utf-8"
                            )
                            layout_regions = [
                                LayoutRegion(
                                    region_type=RegionType.PICTURE,
                                    bbox=RegionBoundingBox(
                                        x0=0,
                                        y0=0,
                                        x1=float(source_image.width),
                                        y1=float(source_image.height),
                                    ),
                                    confidence=1.0,
                                    page_number=1,
                                    region_id="fallback_full_image",
                                    region_image=RegionImage(
                                        image=source_image, base64=b64_str
                                    ),
                                )
                            ]
                            logger.info(
                                "No regions detected; using full image as PICTURE"
                            )
                        elif any(r.region_type in visual_types for r in layout_regions):
                            for region in layout_regions:
                                if region.region_type in visual_types:
                                    cropped = layout_detector.crop_region(
                                        source_image, region, padding=5
                                    )
                                    # Encode as base64
                                    buffer = io.BytesIO()
                                    cropped.save(buffer, format="PNG")
                                    b64_str = base64.b64encode(
                                        buffer.getvalue()
                                    ).decode("utf-8")
                                    region.region_image = RegionImage(
                                        image=cropped, base64=b64_str
                                    )

                        logger.info(
                            "Layout detected for tool agent",
                            regions=len(layout_regions),
                        )
                    except Exception:
                        logger.warning(
                            "Layout detection failed; proceeding without regions",
                            exc_info=True,
                        )

                loop_result = extraction_agent.extract(
                    text=text,
                    schema_info=schema_info,
                    format_info=format_info,
                    layout_regions=layout_regions,
                )
            else:
                # Legacy: multi-agent orchestration loop
                agentic_loop = AgenticLoop()

                # Import extraction services here to avoid circular imports
                from agentic_document_extraction.services.extraction.text_extraction import (
                    ExtractionResult,
                    TextExtractionService,
                )
                from agentic_document_extraction.services.extraction.visual_document_extraction import (
                    VisualDocumentExtractionService,
                )
                from agentic_document_extraction.services.schema_validator import (
                    SchemaInfo,
                )

                # Use VLM-based extraction for visual documents
                if format_info.processing_category == ProcessingCategory.VISUAL:
                    visual_extraction_service = VisualDocumentExtractionService()

                    captured_file_path = document_path
                    captured_ocr_text = text

                    def extraction_func(
                        t: str,  # noqa: ARG001
                        s: SchemaInfo,
                    ) -> ExtractionResult:
                        return visual_extraction_service.extract(
                            image_source=captured_file_path,
                            schema_info=s,
                            ocr_text=captured_ocr_text,
                        )
                else:
                    text_extraction_service = TextExtractionService()

                    def extraction_func(t: str, s: SchemaInfo) -> ExtractionResult:
                        return text_extraction_service.extract(t, s)

                loop_result = agentic_loop.run(
                    text=text,
                    schema_info=schema_info,
                    format_info=format_info,
                    extraction_func=extraction_func,
                    use_llm_verification=True,
                )

            logger.info(
                "Extraction completed",
                iterations=loop_result.iterations_completed,
                converged=loop_result.converged,
            )

            # Normalize extracted data through JsonGenerator
            # This handles format-specific normalization (e.g., dates to ISO format)
            json_generator = JsonGenerator()
            json_result = json_generator.generate(
                loop_result.final_result.extracted_data,
                schema_info,
                handle_nulls=True,
            )
            normalized_data = json_result.data

            # Generate markdown summary
            markdown_generator = MarkdownGenerator()
            markdown_output = markdown_generator.generate(
                extraction_result=loop_result.final_result,
                schema_info=schema_info,
            )
            markdown_summary = markdown_output.markdown

            # Finish metrics
            metrics.finish()
            metrics.tokens_used = loop_result.total_tokens
            metrics.iterations = loop_result.iterations_completed

            # Build metadata
            metadata = {
                "processing_time_seconds": metrics.duration_seconds,
                "model_used": loop_result.final_result.model_used,
                "total_tokens": loop_result.total_tokens,
                "iterations_completed": loop_result.iterations_completed,
                "converged": loop_result.converged,
                "document_type": format_info.processing_category.value,
            }

            # Build quality report
            quality_report = loop_result.final_verification.to_dict()

            # Update quality report issues with normalized values
            # The verification happened before normalization, so current_value
            # may reflect un-normalized data (e.g., "January 15, 2024" instead
            # of "2024-01-15" for dates). Update to show actual final values.
            if quality_report.get("issues"):
                for issue in quality_report["issues"]:
                    field_path = issue.get("field_path", "")
                    if field_path:
                        # Get normalized value using dot notation path
                        normalized_value = _get_nested_value(
                            normalized_data, field_path
                        )
                        if normalized_value is not None:
                            issue["current_value"] = normalized_value

            await _set_progress(progress, "Extraction completed")

            logger.log_extraction_result(
                job_id=job_id,
                success=True,
                duration_seconds=metrics.duration_seconds,
                iterations=loop_result.iterations_completed,
                tokens_used=loop_result.total_tokens,
                converged=loop_result.converged,
                confidence=loop_result.final_verification.metrics.overall_confidence,
            )

            return {
                "extracted_data": normalized_data,
                "markdown_summary": markdown_summary,
                "metadata": metadata,
                "quality_report": quality_report,
            }

    except Exception as e:
        logger.error(
            "Job failed",
            job_id=job_id,
            error=f"{type(e).__name__}: {e}",
            error_type=type(e).__name__,
            exc_info=True,
        )
        if progress is not None:
            with suppress(AssertionError):
                await progress.set_message("Job failed")
        raise
