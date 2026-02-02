"""Docket task registry for worker CLI usage."""

from agentic_document_extraction.services.extraction_processor import (
    process_extraction_job,
)
from agentic_document_extraction.services.retention import purge_expired_artifacts


async def retention_purge_task() -> dict[str, int]:
    """Docket task to purge expired uploads and artifacts."""
    result = purge_expired_artifacts()
    return result.to_dict()


tasks = [process_extraction_job, retention_purge_task]
