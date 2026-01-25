"""Docket task registry for worker CLI usage."""

from agentic_document_extraction.services.extraction_processor import (
    process_extraction_job,
)

tasks = [process_extraction_job]
