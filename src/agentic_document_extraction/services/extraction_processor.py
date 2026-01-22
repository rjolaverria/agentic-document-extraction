"""Extraction processor for background job processing.

This module provides the main extraction orchestration logic that:
1. Detects document format
2. Extracts text using appropriate method (text-based or visual)
3. Runs the agentic extraction loop
4. Saves results to job manager

This is the entry point for processing extraction jobs in background tasks.
"""

import json
from pathlib import Path

from agentic_document_extraction.agents.refiner import AgenticLoop
from agentic_document_extraction.models import ProcessingCategory
from agentic_document_extraction.output.markdown_generator import MarkdownGenerator
from agentic_document_extraction.services.format_detector import FormatDetector
from agentic_document_extraction.services.job_manager import (
    JobManager,
    JobNotFoundError,
    get_job_manager,
)
from agentic_document_extraction.services.schema_validator import SchemaValidator
from agentic_document_extraction.services.text_extractor import TextExtractor
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


# Re-export DocumentProcessingError as ExtractionProcessorError for backward compatibility
ExtractionProcessorError = DocumentProcessingError


def process_extraction_job(
    job_id: str,
    job_manager: JobManager | None = None,
) -> None:
    """Process an extraction job.

    This is the main entry point for background job processing. It:
    1. Retrieves job data from the job manager
    2. Detects document format
    3. Extracts text using appropriate method
    4. Runs the agentic extraction loop
    5. Saves results or error to job manager

    Args:
        job_id: The job ID to process.
        job_manager: Optional job manager instance. Uses global if not provided.
    """
    if job_manager is None:
        job_manager = get_job_manager()

    # Set job ID in context for logging correlation
    set_job_id(job_id)
    metrics = PerformanceMetrics(operation="extraction")

    try:
        # Get job data
        job = job_manager.get_job(job_id)

        with LogContext(job_id=job_id, filename=job.filename):
            logger.info(
                "Starting extraction",
                filename=job.filename,
            )

            # Update status to processing
            job_manager.update_status(
                job_id,
                status=job.status.PROCESSING,
                progress="Processing started",
            )

            # Load schema from file
            schema_path = Path(job.schema_path)
            if not schema_path.exists():
                raise ADEFileNotFoundError(
                    file_path=str(schema_path),
                    message=f"Schema file not found: {schema_path}",
                )

            with open(schema_path) as f:
                schema_content = json.load(f)

            # Validate schema
            schema_validator = SchemaValidator()
            schema_info = schema_validator.validate(schema_content)
            logger.info("Schema validated")

            job_manager.update_status(
                job_id,
                status=job.status.PROCESSING,
                progress="Schema validated, detecting document format",
            )

            # Detect document format
            file_path = Path(job.file_path)
            if not file_path.exists():
                raise ADEFileNotFoundError(
                    file_path=str(file_path),
                    message=f"Document file not found: {file_path}",
                )

            format_detector = FormatDetector()
            format_info = format_detector.detect_from_path(file_path)
            logger.info(
                "Format detected",
                mime_type=format_info.mime_type,
                category=format_info.processing_category.value,
            )

            job_manager.update_status(
                job_id,
                status=job.status.PROCESSING,
                progress=f"Document format: {format_info.format_family.value}, extracting text",
            )

            # Extract text based on document type
            text_extractor = TextExtractor()

            if format_info.processing_category == ProcessingCategory.TEXT_BASED:
                # Text-based extraction
                text_extraction_result = text_extractor.extract_from_path(file_path)
                text = text_extraction_result.text
                logger.info("Text extracted", length=len(text))
            else:
                # Visual document - use OCR and layout detection
                # For now, extract basic text; visual pipeline can be enhanced later
                text_extraction_result = text_extractor.extract_from_path(file_path)
                text = text_extraction_result.text
                logger.info(
                    "Text extracted from visual document",
                    length=len(text),
                )

            job_manager.update_status(
                job_id,
                status=job.status.PROCESSING,
                progress="Running agentic extraction loop",
            )

            # Run agentic extraction loop
            agentic_loop = AgenticLoop()

            # Import TextExtractionService here to avoid circular imports
            from agentic_document_extraction.services.extraction.text_extraction import (
                ExtractionResult,
                TextExtractionService,
            )
            from agentic_document_extraction.services.schema_validator import SchemaInfo

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
                "Agentic loop completed",
                iterations=loop_result.iterations_completed,
                converged=loop_result.converged,
            )

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

            # Set result on job
            job_manager.set_result(
                job_id=job_id,
                extracted_data=loop_result.final_result.extracted_data,
                markdown_summary=markdown_summary,
                metadata=metadata,
                quality_report=quality_report,
            )

            logger.log_extraction_result(
                job_id=job_id,
                success=True,
                duration_seconds=metrics.duration_seconds,
                iterations=loop_result.iterations_completed,
                tokens_used=loop_result.total_tokens,
                converged=loop_result.converged,
                confidence=loop_result.final_verification.metrics.overall_confidence,
            )

    except JobNotFoundError:
        logger.error("Job not found, cannot process", job_id=job_id)
        raise

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        logger.error(
            "Job failed",
            job_id=job_id,
            error=error_message,
            error_type=type(e).__name__,
            exc_info=True,
        )

        try:
            job_manager.set_failed(job_id, error_message)
        except JobNotFoundError:
            logger.error(
                "Could not mark job as failed - job not found",
                job_id=job_id,
            )


async def process_extraction_job_async(
    job_id: str,
    job_manager: JobManager | None = None,
) -> None:
    """Async wrapper for process_extraction_job.

    Used with FastAPI BackgroundTasks for async execution.

    Args:
        job_id: The job ID to process.
        job_manager: Optional job manager instance.
    """
    # Run the synchronous processing in a way that doesn't block
    # For CPU-bound work, this runs in the same thread
    # For I/O-bound work (API calls), the underlying libraries handle async
    process_extraction_job(job_id, job_manager)
