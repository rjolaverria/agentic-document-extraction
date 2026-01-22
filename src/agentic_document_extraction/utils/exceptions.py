"""Centralized exception classes for agentic document extraction.

This module provides a hierarchy of custom exceptions with error codes,
HTTP status code mapping, and structured error details for consistent
error handling throughout the application.

Exception Hierarchy:
    ADEError (base)
    ├── FileError
    │   ├── FileNotFoundError (shadows built-in, use ADEFileNotFoundError)
    │   ├── FileTooLargeError
    │   └── UnsupportedFormatError
    ├── SchemaError
    │   └── SchemaValidationError
    ├── JobError
    │   ├── JobNotFoundError
    │   └── JobExpiredError
    ├── ExtractionError
    │   ├── TextExtractionError
    │   ├── OCRError
    │   └── LayoutDetectionError
    ├── DocumentProcessingError
    └── ValidationError

Error Codes:
    All errors have a unique error code (e.g., "E1001") that can be used
    for programmatic error handling and documentation.
"""

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Enumeration of all error codes used in the application.

    Error codes are grouped by category:
    - E1xxx: File/document errors
    - E2xxx: Schema errors
    - E3xxx: Job management errors
    - E4xxx: Extraction/processing errors
    - E5xxx: External service errors
    - E9xxx: Internal/unexpected errors
    """

    # File errors (E1xxx)
    FILE_NOT_FOUND = "E1001"
    FILE_TOO_LARGE = "E1002"
    UNSUPPORTED_FORMAT = "E1003"
    FILE_READ_ERROR = "E1004"
    FILE_WRITE_ERROR = "E1005"
    ENCODING_ERROR = "E1006"

    # Schema errors (E2xxx)
    INVALID_SCHEMA = "E2001"
    SCHEMA_VALIDATION_FAILED = "E2002"
    UNSUPPORTED_SCHEMA_TYPE = "E2003"
    SCHEMA_PARSE_ERROR = "E2004"

    # Job errors (E3xxx)
    JOB_NOT_FOUND = "E3001"
    JOB_EXPIRED = "E3002"
    JOB_ALREADY_EXISTS = "E3003"
    JOB_PROCESSING_FAILED = "E3004"
    JOB_NOT_COMPLETE = "E3005"

    # Extraction errors (E4xxx)
    EXTRACTION_FAILED = "E4001"
    TEXT_EXTRACTION_FAILED = "E4002"
    OCR_FAILED = "E4003"
    LAYOUT_DETECTION_FAILED = "E4004"
    READING_ORDER_FAILED = "E4005"
    VISUAL_EXTRACTION_FAILED = "E4006"
    SYNTHESIS_FAILED = "E4007"
    QUALITY_THRESHOLD_NOT_MET = "E4008"

    # External service errors (E5xxx)
    LLM_API_ERROR = "E5001"
    LLM_RATE_LIMIT = "E5002"
    LLM_TOKEN_LIMIT = "E5003"
    EXTERNAL_SERVICE_UNAVAILABLE = "E5004"

    # Internal errors (E9xxx)
    INTERNAL_ERROR = "E9001"
    CONFIGURATION_ERROR = "E9002"
    UNEXPECTED_ERROR = "E9999"


class HTTPStatusMixin:
    """Mixin that provides HTTP status code for exceptions.

    This mixin allows exceptions to declare their appropriate HTTP status code
    for API responses. Subclasses should set the `http_status` class attribute.
    """

    http_status: int = 500

    def get_http_status(self) -> int:
        """Get the HTTP status code for this exception.

        Returns:
            HTTP status code appropriate for this error.
        """
        return self.http_status


class ADEError(Exception, HTTPStatusMixin):
    """Base exception for all Agentic Document Extraction errors.

    All custom exceptions in the application should inherit from this class.
    It provides:
    - Unique error codes for programmatic handling
    - HTTP status code mapping for API responses
    - Structured error details for logging and debugging

    Attributes:
        message: Human-readable error message.
        error_code: Unique error code from ErrorCode enum.
        details: Optional dictionary with additional error details.
        http_status: HTTP status code for API responses (default 500).
    """

    http_status: int = 500

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            error_code: Error code from ErrorCode enum.
            details: Optional additional details about the error.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert the exception to a dictionary for API responses.

        Returns:
            Dictionary with error information.
        """
        result: dict[str, Any] = {
            "error_code": self.error_code.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        """Return string representation with error code."""
        return f"[{self.error_code.value}] {self.message}"


# =============================================================================
# File Errors (E1xxx)
# =============================================================================


class FileError(ADEError):
    """Base class for file-related errors."""

    http_status: int = 400

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.FILE_READ_ERROR,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with file path information.

        Args:
            message: Error message.
            error_code: Error code.
            file_path: Path to the problematic file.
            details: Additional details.
        """
        details = details or {}
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, error_code, details)
        self.file_path = file_path


class ADEFileNotFoundError(FileError):
    """Raised when a required file is not found.

    Note: Named ADEFileNotFoundError to avoid shadowing built-in FileNotFoundError.
    """

    http_status: int = 404

    def __init__(
        self,
        file_path: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with file path.

        Args:
            file_path: Path to the file that was not found.
            message: Optional custom message.
            details: Additional details.
        """
        message = message or f"File not found: {file_path}"
        super().__init__(
            message=message,
            error_code=ErrorCode.FILE_NOT_FOUND,
            file_path=file_path,
            details=details,
        )


class FileTooLargeError(FileError):
    """Raised when a file exceeds the maximum allowed size."""

    http_status: int = 413

    def __init__(
        self,
        file_size: int,
        max_size: int,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with size information.

        Args:
            file_size: Actual file size in bytes.
            max_size: Maximum allowed size in bytes.
            file_path: Optional file path.
            details: Additional details.
        """
        details = details or {}
        details["file_size_bytes"] = file_size
        details["max_size_bytes"] = max_size
        message = (
            f"File size ({file_size} bytes) exceeds maximum "
            f"allowed size ({max_size} bytes)"
        )
        super().__init__(
            message=message,
            error_code=ErrorCode.FILE_TOO_LARGE,
            file_path=file_path,
            details=details,
        )
        self.file_size = file_size
        self.max_size = max_size


class UnsupportedFormatError(FileError):
    """Raised when a document format is not supported."""

    http_status: int = 400

    def __init__(
        self,
        message: str,
        detected_mime: str | None = None,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with format information.

        Args:
            message: Error message.
            detected_mime: MIME type that was detected.
            file_path: Optional file path.
            details: Additional details.
        """
        details = details or {}
        if detected_mime:
            details["detected_mime_type"] = detected_mime
        super().__init__(
            message=message,
            error_code=ErrorCode.UNSUPPORTED_FORMAT,
            file_path=file_path,
            details=details,
        )
        self.detected_mime = detected_mime


class EncodingError(FileError):
    """Raised when there's an encoding/decoding error with file content."""

    http_status: int = 400

    def __init__(
        self,
        message: str,
        encoding: str | None = None,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with encoding information.

        Args:
            message: Error message.
            encoding: The encoding that caused the error.
            file_path: Optional file path.
            details: Additional details.
        """
        details = details or {}
        if encoding:
            details["encoding"] = encoding
        super().__init__(
            message=message,
            error_code=ErrorCode.ENCODING_ERROR,
            file_path=file_path,
            details=details,
        )
        self.encoding = encoding


# =============================================================================
# Schema Errors (E2xxx)
# =============================================================================


class SchemaError(ADEError):
    """Base class for schema-related errors."""

    http_status: int = 400

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INVALID_SCHEMA,
        errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with validation errors.

        Args:
            message: Main error message.
            error_code: Error code.
            errors: List of specific validation error messages.
            details: Additional details.
        """
        details = details or {}
        if errors:
            details["validation_errors"] = errors
        super().__init__(message, error_code, details)
        self.errors = errors or []


class SchemaValidationError(SchemaError):
    """Raised when a JSON schema is invalid or doesn't validate."""

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with validation errors.

        Args:
            message: Main error message.
            errors: List of specific validation errors.
            details: Additional details.
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.SCHEMA_VALIDATION_FAILED,
            errors=errors,
            details=details,
        )


class SchemaParseError(SchemaError):
    """Raised when a schema cannot be parsed (invalid JSON, etc.)."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with parse error.

        Args:
            message: Error message describing the parse failure.
            details: Additional details.
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.SCHEMA_PARSE_ERROR,
            details=details,
        )


# =============================================================================
# Job Errors (E3xxx)
# =============================================================================


class JobError(ADEError):
    """Base class for job management errors."""

    http_status: int = 400

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.JOB_PROCESSING_FAILED,
        job_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with job ID.

        Args:
            message: Error message.
            error_code: Error code.
            job_id: ID of the affected job.
            details: Additional details.
        """
        details = details or {}
        if job_id:
            details["job_id"] = job_id
        super().__init__(message, error_code, details)
        self.job_id = job_id


class JobNotFoundError(JobError):
    """Raised when a job is not found."""

    http_status: int = 404

    def __init__(
        self,
        job_id: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with job ID.

        Args:
            job_id: The job ID that was not found.
            message: Optional custom message.
            details: Additional details.
        """
        message = message or f"Job not found: {job_id}"
        super().__init__(
            message=message,
            error_code=ErrorCode.JOB_NOT_FOUND,
            job_id=job_id,
            details=details,
        )


class JobExpiredError(JobError):
    """Raised when a job has expired (TTL exceeded)."""

    http_status: int = 410

    def __init__(
        self,
        job_id: str,
        ttl_hours: int | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with job ID and TTL.

        Args:
            job_id: The job ID that has expired.
            ttl_hours: The TTL in hours (for the error message).
            message: Optional custom message.
            details: Additional details.
        """
        details = details or {}
        if ttl_hours:
            details["ttl_hours"] = ttl_hours
        message = message or f"Job has expired: {job_id}"
        super().__init__(
            message=message,
            error_code=ErrorCode.JOB_EXPIRED,
            job_id=job_id,
            details=details,
        )


class JobNotCompleteError(JobError):
    """Raised when trying to get results from an incomplete job."""

    http_status: int = 425  # Too Early

    def __init__(
        self,
        job_id: str,
        status: str,
        progress: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with job status.

        Args:
            job_id: The job ID.
            status: Current job status.
            progress: Current progress description.
            details: Additional details.
        """
        details = details or {}
        details["status"] = status
        if progress:
            details["progress"] = progress
        message = f"Job {job_id} is not complete (status: {status})"
        super().__init__(
            message=message,
            error_code=ErrorCode.JOB_NOT_COMPLETE,
            job_id=job_id,
            details=details,
        )


# =============================================================================
# Extraction Errors (E4xxx)
# =============================================================================


class ExtractionError(ADEError):
    """Base class for extraction-related errors."""

    http_status: int = 500

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.EXTRACTION_FAILED,
        stage: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with extraction stage.

        Args:
            message: Error message.
            error_code: Error code.
            stage: The extraction stage where the error occurred.
            details: Additional details.
        """
        details = details or {}
        if stage:
            details["extraction_stage"] = stage
        super().__init__(message, error_code, details)
        self.stage = stage


class TextExtractionError(ExtractionError):
    """Raised when text extraction fails."""

    def __init__(
        self,
        message: str,
        file_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with file type.

        Args:
            message: Error message.
            file_type: Type of file being processed.
            details: Additional details.
        """
        details = details or {}
        if file_type:
            details["file_type"] = file_type
        super().__init__(
            message=message,
            error_code=ErrorCode.TEXT_EXTRACTION_FAILED,
            stage="text_extraction",
            details=details,
        )


class OCRError(ExtractionError):
    """Raised when OCR processing fails."""

    def __init__(
        self,
        message: str,
        page_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with page information.

        Args:
            message: Error message.
            page_number: Page number where OCR failed.
            details: Additional details.
        """
        details = details or {}
        if page_number is not None:
            details["page_number"] = page_number
        super().__init__(
            message=message,
            error_code=ErrorCode.OCR_FAILED,
            stage="ocr",
            details=details,
        )


class LayoutDetectionError(ExtractionError):
    """Raised when layout detection fails."""

    def __init__(
        self,
        message: str,
        page_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with page information.

        Args:
            message: Error message.
            page_number: Page number where layout detection failed.
            details: Additional details.
        """
        details = details or {}
        if page_number is not None:
            details["page_number"] = page_number
        super().__init__(
            message=message,
            error_code=ErrorCode.LAYOUT_DETECTION_FAILED,
            stage="layout_detection",
            details=details,
        )


class QualityThresholdError(ExtractionError):
    """Raised when extraction quality doesn't meet thresholds."""

    def __init__(
        self,
        message: str,
        achieved_confidence: float | None = None,
        required_confidence: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with confidence scores.

        Args:
            message: Error message.
            achieved_confidence: The confidence score that was achieved.
            required_confidence: The minimum required confidence.
            details: Additional details.
        """
        details = details or {}
        if achieved_confidence is not None:
            details["achieved_confidence"] = achieved_confidence
        if required_confidence is not None:
            details["required_confidence"] = required_confidence
        super().__init__(
            message=message,
            error_code=ErrorCode.QUALITY_THRESHOLD_NOT_MET,
            stage="quality_verification",
            details=details,
        )


# =============================================================================
# Document Processing Errors
# =============================================================================


class DocumentProcessingError(ADEError):
    """General document processing error."""

    http_status: int = 500

    def __init__(
        self,
        message: str,
        error_type: str = "processing_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with error type.

        Args:
            message: Error message.
            error_type: Type categorization for the error.
            details: Additional details.
        """
        details = details or {}
        details["error_type"] = error_type
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            details=details,
        )


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(ADEError):
    """General validation error for input data."""

    http_status: int = 400

    def __init__(
        self,
        message: str,
        field: str | None = None,
        errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with validation details.

        Args:
            message: Main error message.
            field: Field that failed validation.
            errors: List of validation errors.
            details: Additional details.
        """
        details = details or {}
        if field:
            details["field"] = field
        if errors:
            details["validation_errors"] = errors
        super().__init__(
            message=message,
            error_code=ErrorCode.SCHEMA_VALIDATION_FAILED,
            details=details,
        )


# =============================================================================
# External Service Errors (E5xxx)
# =============================================================================


class LLMError(ADEError):
    """Base class for LLM/AI service errors."""

    http_status: int = 502

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.LLM_API_ERROR,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with model information.

        Args:
            message: Error message.
            error_code: Error code.
            model: The LLM model that caused the error.
            details: Additional details.
        """
        details = details or {}
        if model:
            details["model"] = model
        super().__init__(message, error_code, details)
        self.model = model


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded."""

    http_status: int = 429

    def __init__(
        self,
        message: str = "LLM API rate limit exceeded",
        retry_after: int | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with retry information.

        Args:
            message: Error message.
            retry_after: Seconds to wait before retry.
            model: The LLM model.
            details: Additional details.
        """
        details = details or {}
        if retry_after is not None:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_RATE_LIMIT,
            model=model,
            details=details,
        )


class LLMTokenLimitError(LLMError):
    """Raised when LLM token limit is exceeded."""

    def __init__(
        self,
        message: str = "LLM token limit exceeded",
        token_count: int | None = None,
        token_limit: int | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with token information.

        Args:
            message: Error message.
            token_count: Number of tokens attempted.
            token_limit: Maximum allowed tokens.
            model: The LLM model.
            details: Additional details.
        """
        details = details or {}
        if token_count is not None:
            details["token_count"] = token_count
        if token_limit is not None:
            details["token_limit"] = token_limit
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_TOKEN_LIMIT,
            model=model,
            details=details,
        )
