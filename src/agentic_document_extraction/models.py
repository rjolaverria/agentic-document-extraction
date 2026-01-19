"""Pydantic models for API requests and responses."""

from enum import Enum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    timestamp: str
    version: str


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""

    job_id: str = Field(..., description="Unique identifier for the extraction job")
    filename: str = Field(..., description="Original filename of uploaded document")
    file_size: int = Field(..., description="Size of uploaded file in bytes")
    message: str = Field(..., description="Status message")


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
