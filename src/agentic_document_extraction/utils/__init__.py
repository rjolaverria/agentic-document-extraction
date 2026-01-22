"""Utilities package for agentic document extraction.

This package provides:
- Centralized exception classes (exceptions.py)
- Structured logging utilities (logging.py)
"""

from agentic_document_extraction.utils.exceptions import (
    ADEError,
    DocumentProcessingError,
    ErrorCode,
    ExtractionError,
    FileError,
    HTTPStatusMixin,
    JobError,
    SchemaError,
    UnsupportedFormatError,
    ValidationError,
)
from agentic_document_extraction.utils.logging import (
    LogContext,
    StructuredLogger,
    get_logger,
    get_request_id,
    set_request_id,
)

__all__ = [
    # Exceptions
    "ADEError",
    "DocumentProcessingError",
    "ErrorCode",
    "ExtractionError",
    "FileError",
    "HTTPStatusMixin",
    "JobError",
    "SchemaError",
    "UnsupportedFormatError",
    "ValidationError",
    # Logging
    "LogContext",
    "StructuredLogger",
    "get_logger",
    "get_request_id",
    "set_request_id",
]
