"""Pydantic models for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
    """Error detail model for API error responses."""

    detail: str
    error_code: str | None = None


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
