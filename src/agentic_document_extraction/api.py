"""FastAPI application for agentic document extraction."""

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_document_extraction.config import settings
from agentic_document_extraction.models import (
    ErrorDetail,
    HealthResponse,
    UploadResponse,
)
from agentic_document_extraction.services.schema_validator import (
    SchemaValidationError,
    SchemaValidator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agentic Document Extraction API",
        description=(
            "Vision-first, agentic document extraction system that intelligently "
            "processes documents and extracts structured information using AI."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Custom exception handler for HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorDetail(detail=str(exc.detail)).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> dict[str, Any]:
        """Check the health status of the service.

        Returns:
            HealthResponse: Service status information including status,
                timestamp, and version.
        """
        logger.info("Health check requested")
        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "0.1.0",
        }

    @app.post(
        "/extract",
        response_model=UploadResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Extraction"],
        responses={
            400: {"model": ErrorDetail, "description": "Missing required fields"},
            413: {"model": ErrorDetail, "description": "File too large"},
        },
    )
    async def extract_document(
        file: Annotated[UploadFile, File(description="Document file to extract from")],
        schema: Annotated[str | None, Form(description="JSON schema as string")] = None,
        schema_file: Annotated[
            UploadFile | None, File(description="JSON schema as file upload")
        ] = None,
    ) -> dict[str, Any]:
        """Upload a document and JSON schema to trigger extraction.

        Accepts a document file and a JSON schema (either as a string or file).
        The document will be processed asynchronously and structured information
        will be extracted according to the provided schema.

        Args:
            file: The document file to process
            schema: JSON schema as a string (alternative to schema_file)
            schema_file: JSON schema as a file upload (alternative to schema)

        Returns:
            UploadResponse: Job information including unique job ID

        Raises:
            HTTPException: 400 if schema is missing or invalid
            HTTPException: 413 if file exceeds size limit
        """
        # Validate that at least one schema input is provided
        if schema is None and (schema_file is None or schema_file.filename == ""):
            logger.warning("Extract request missing schema")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'schema' (JSON string) or 'schema_file' must be provided",
            )

        # Validate file is provided with content
        if file.filename is None or file.filename == "":
            logger.warning("Extract request missing file")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A document file must be provided",
            )

        # Read file content to check size
        file_content = await file.read()
        file_size = len(file_content)

        # Check file size
        if file_size > settings.max_file_size_bytes:
            logger.warning(
                f"File too large: {file_size} bytes (max: {settings.max_file_size_bytes})"
            )
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File size ({file_size} bytes) exceeds maximum allowed size "
                f"({settings.max_file_size_bytes} bytes / {settings.max_file_size_mb} MB)",
            )

        # Parse JSON schema
        schema_content: dict[str, Any]
        if schema is not None:
            try:
                schema_content = json.loads(schema)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON schema string: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON in schema: {e}",
                ) from e
        else:
            # schema_file must be provided (validated above)
            assert schema_file is not None
            schema_bytes = await schema_file.read()
            try:
                schema_content = json.loads(schema_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Invalid JSON schema file: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON in schema file: {e}",
                ) from e

        # Validate JSON Schema syntax and structure
        schema_validator = SchemaValidator()
        try:
            schema_info = schema_validator.validate(schema_content)
            logger.info(
                f"Schema validated: {schema_info.schema_type} with "
                f"{len(schema_info.required_fields)} required and "
                f"{len(schema_info.optional_fields)} optional fields"
            )
        except SchemaValidationError as e:
            error_details = "; ".join(e.errors) if e.errors else str(e)
            logger.warning(f"Schema validation failed: {error_details}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON Schema: {error_details}",
            ) from e

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Create temp directory if it doesn't exist
        upload_dir = Path(settings.temp_upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file with unique identifier
        file_extension = Path(file.filename).suffix if file.filename else ""
        temp_filename = f"{job_id}{file_extension}"
        temp_filepath = upload_dir / temp_filename

        with open(temp_filepath, "wb") as f:
            f.write(file_content)

        # Save schema alongside the document
        schema_filepath = upload_dir / f"{job_id}_schema.json"
        with open(schema_filepath, "w") as f:
            json.dump(schema_content, f)

        logger.info(
            f"Document uploaded: job_id={job_id}, "
            f"filename={file.filename}, size={file_size}"
        )

        return {
            "job_id": job_id,
            "filename": file.filename or "unknown",
            "file_size": file_size,
            "message": "Document uploaded successfully. Processing will begin shortly.",
        }

    logger.info("FastAPI application created successfully")
    return app


# Create the application instance
app = create_app()
