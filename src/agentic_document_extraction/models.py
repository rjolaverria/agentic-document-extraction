"""Pydantic models for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_document_extraction.utils.exceptions import ErrorCode


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    timestamp: str
    version: str


class JobStatus(str, Enum):
    """Status of an extraction job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""

    job_id: str = Field(..., description="Unique identifier for the extraction job")
    filename: str = Field(..., description="Original filename of uploaded document")
    file_size: int = Field(..., description="Size of uploaded file in bytes")
    status: JobStatus = Field(
        default=JobStatus.PENDING, description="Initial job status"
    )
    message: str = Field(..., description="Status message")


class JobStatusResponse(BaseModel):
    """Response model for job status endpoint."""

    job_id: str = Field(..., description="Unique identifier for the extraction job")
    status: JobStatus = Field(..., description="Current job status")
    filename: str = Field(..., description="Original filename of uploaded document")
    created_at: datetime = Field(..., description="When the job was created")
    updated_at: datetime = Field(..., description="When the job was last updated")
    progress: str | None = Field(
        default=None, description="Current progress description"
    )
    error_message: str | None = Field(
        default=None, description="Error message if job failed"
    )


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""

    processing_time_seconds: float = Field(
        ..., description="Total processing time in seconds"
    )
    model_used: str = Field(..., description="LLM model used for extraction")
    total_tokens: int = Field(..., description="Total tokens used")
    iterations_completed: int = Field(
        ..., description="Number of agentic loop iterations"
    )
    converged: bool = Field(..., description="Whether quality thresholds were met")
    document_type: str = Field(
        ..., description="Processing type (text_based or visual)"
    )


class JobResultResponse(BaseModel):
    """Response model for job result endpoint."""

    job_id: str = Field(..., description="Unique identifier for the extraction job")
    status: JobStatus = Field(..., description="Job status")
    extracted_data: dict[str, Any] | None = Field(
        default=None, description="Extracted data matching the schema"
    )
    markdown_summary: str | None = Field(
        default=None, description="Markdown summary of extracted data"
    )
    metadata: ExtractionMetadata | None = Field(
        default=None, description="Extraction process metadata"
    )
    quality_report: dict[str, Any] | None = Field(
        default=None, description="Quality verification report"
    )
    error_message: str | None = Field(
        default=None, description="Error message if job failed"
    )


class ErrorDetail(BaseModel):
    """Error detail model for API error responses.

    This model provides structured error responses with:
    - Human-readable error message
    - Machine-readable error code
    - Optional additional details for debugging
    - Optional request ID for correlation
    """

    detail: str = Field(..., description="Human-readable error message")
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error code (e.g., 'E1001')",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details for debugging",
    )
    request_id: str | None = Field(
        default=None,
        description="Request ID for error correlation",
    )

    @classmethod
    def from_error_code(
        cls,
        error_code: ErrorCode,
        detail: str,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> "ErrorDetail":
        """Create an ErrorDetail from an ErrorCode enum value.

        Args:
            error_code: The error code enum.
            detail: Human-readable error message.
            details: Optional additional details.
            request_id: Optional request ID.

        Returns:
            ErrorDetail instance.
        """
        return cls(
            detail=detail,
            error_code=error_code.value,
            details=details,
            request_id=request_id,
        )


class ProcessingCategory(str, Enum):
    """Processing category for documents."""

    TEXT_BASED = "text_based"
    VISUAL = "visual"


class FormatFamily(str, Enum):
    """Document format family classification."""

    PLAIN_TEXT = "plain_text"
    SPREADSHEET = "spreadsheet"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    IMAGE = "image"
    PDF = "pdf"
    UNKNOWN = "unknown"


class FormatInfo(BaseModel):
    """Document format detection result."""

    mime_type: str = Field(
        ..., description="MIME type of the document (e.g., 'application/pdf')"
    )
    extension: str = Field(
        ..., description="File extension including the dot (e.g., '.pdf')"
    )
    format_family: FormatFamily = Field(
        ..., description="Document format family classification"
    )
    processing_category: ProcessingCategory = Field(
        ..., description="Processing category (text_based or visual)"
    )
    detected_from_content: bool = Field(
        default=False,
        description="Whether format was detected from file content (magic bytes)",
    )
    original_extension: str | None = Field(
        default=None,
        description="Original file extension if different from detected format",
    )
