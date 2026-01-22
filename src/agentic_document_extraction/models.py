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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2024-01-15T10:30:00.000000+00:00",
                    "version": "0.1.0",
                }
            ]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "filename": "invoice.pdf",
                    "file_size": 102400,
                    "status": "pending",
                    "message": "Document uploaded successfully. Processing will begin shortly.",
                }
            ]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "processing",
                    "filename": "invoice.pdf",
                    "created_at": "2024-01-15T10:30:00.000000+00:00",
                    "updated_at": "2024-01-15T10:30:05.000000+00:00",
                    "progress": "Analyzing document layout (page 2 of 3)",
                    "error_message": None,
                }
            ]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "processing_time_seconds": 12.5,
                    "model_used": "gpt-4o",
                    "total_tokens": 2500,
                    "iterations_completed": 2,
                    "converged": True,
                    "document_type": "visual",
                }
            ]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "completed",
                    "extracted_data": {
                        "invoice_number": "INV-2024-001",
                        "date": "2024-01-15",
                        "total_amount": 1250.00,
                        "line_items": [
                            {
                                "description": "Widget A",
                                "quantity": 5,
                                "price": 100.00,
                            },
                            {
                                "description": "Widget B",
                                "quantity": 10,
                                "price": 75.00,
                            },
                        ],
                    },
                    "markdown_summary": "# Invoice Extraction Summary\n\n**Invoice Number:** INV-2024-001\n**Date:** 2024-01-15\n**Total:** $1,250.00",
                    "metadata": {
                        "processing_time_seconds": 12.5,
                        "model_used": "gpt-4o",
                        "total_tokens": 2500,
                        "iterations_completed": 2,
                        "converged": True,
                        "document_type": "visual",
                    },
                    "quality_report": {
                        "overall_confidence": 0.92,
                        "field_scores": {
                            "invoice_number": 0.98,
                            "date": 0.95,
                            "total_amount": 0.90,
                        },
                        "issues": [],
                    },
                    "error_message": None,
                }
            ]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "Either 'schema' (JSON string) or 'schema_file' must be provided",
                    "error_code": "E1001",
                    "details": {"field": "schema"},
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            ]
        }
    }

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
