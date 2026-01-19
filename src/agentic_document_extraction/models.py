"""Pydantic models for API requests and responses."""

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
