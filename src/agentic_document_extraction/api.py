"""FastAPI application for agentic document extraction."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_document_extraction.config import settings, validate_settings_on_startup
from agentic_document_extraction.models import (
    ErrorDetail,
    ExtractionMetadata,
    HealthResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    UploadResponse,
)
from agentic_document_extraction.services.extraction_processor import (
    process_extraction_job,
)
from agentic_document_extraction.services.job_manager import (
    JobExpiredError,
    JobNotFoundError,
    get_job_manager,
)
from agentic_document_extraction.services.schema_validator import (
    SchemaValidationError,
    SchemaValidator,
)
from agentic_document_extraction.utils.exceptions import (
    ADEError,
    ErrorCode,
    FileTooLargeError,
    SchemaParseError,
    ValidationError,
)
from agentic_document_extraction.utils.logging import (
    clear_context,
    configure_logging,
    get_logger,
    get_request_id,
    set_job_id,
    set_request_id,
)

# Configure structured logging using settings
configure_logging(
    level=settings.log_level_int,
    use_structured_formatter=True,
)
logger = get_logger(__name__)


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

    # Configure CORS using settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Validate settings on startup
    validate_settings_on_startup(settings)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Any:
        """Middleware to assign and track request IDs.

        This middleware:
        1. Generates a unique request ID for each request
        2. Sets it in context for logging correlation
        3. Adds it to the response headers
        4. Clears context after request completes
        """
        # Generate or use existing request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_request_id(request_id)

        # Store request ID in request state for access in handlers
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # Clean up context after request
            clear_context()

    @app.exception_handler(ADEError)
    async def ade_exception_handler(request: Request, exc: ADEError) -> JSONResponse:
        """Custom exception handler for ADE exceptions.

        Handles all custom exceptions from the utils.exceptions module
        and returns structured error responses with error codes.
        """
        request_id = getattr(request.state, "request_id", get_request_id())
        logger.error(
            f"ADE Error: {exc.message}",
            error_code=exc.error_code.value,
            http_status=exc.http_status,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorDetail(
                detail=exc.message,
                error_code=exc.error_code.value,
                details=exc.details if exc.details else None,
                request_id=request_id,
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Custom exception handler for HTTP exceptions."""
        request_id = getattr(request.state, "request_id", get_request_id())
        logger.warning(
            f"HTTP Error: {exc.detail}",
            status_code=exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorDetail(
                detail=str(exc.detail),
                request_id=request_id,
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all exception handler for unexpected errors.

        Logs the full exception and returns a generic error response
        to avoid leaking internal details.
        """
        request_id = getattr(request.state, "request_id", get_request_id())
        logger.exception(
            f"Unexpected error: {type(exc).__name__}",
            error_type=type(exc).__name__,
        )
        # In debug mode, include more details
        if settings.debug:
            detail = f"Internal server error: {type(exc).__name__}: {exc}"
        else:
            detail = "Internal server error. Please try again later."

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorDetail(
                detail=detail,
                error_code=ErrorCode.INTERNAL_ERROR.value,
                request_id=request_id,
            ).model_dump(exclude_none=True),
        )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(request: Request) -> dict[str, Any]:
        """Check the health status of the service.

        Returns:
            HealthResponse: Service status information including status,
                timestamp, and version.
        """
        request_id = getattr(request.state, "request_id", None)
        logger.debug("Health check requested", request_id=request_id)
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
        request: Request,
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File(description="Document file to extract from")],
        extraction_schema: Annotated[
            str | None, Form(alias="schema", description="JSON schema as string")
        ] = None,
        schema_file: Annotated[
            UploadFile | None, File(description="JSON schema as file upload")
        ] = None,
    ) -> dict[str, Any]:
        """Upload a document and JSON schema to trigger extraction.

        Accepts a document file and a JSON schema (either as a string or file).
        The document will be processed asynchronously and structured information
        will be extracted according to the provided schema.

        For large documents, this endpoint returns immediately with a job ID.
        Use GET /jobs/{job_id} to check status and GET /jobs/{job_id}/result
        to retrieve results when processing is complete.

        Args:
            request: FastAPI request object
            background_tasks: FastAPI background tasks handler
            file: The document file to process
            extraction_schema: JSON schema as a string (alternative to schema_file)
            schema_file: JSON schema as a file upload (alternative to schema)

        Returns:
            UploadResponse: Job information including unique job ID and status

        Raises:
            HTTPException: 400 if schema is missing or invalid
            HTTPException: 413 if file exceeds size limit
        """
        request_id = getattr(request.state, "request_id", None)

        # Validate that at least one schema input is provided
        if extraction_schema is None and (
            schema_file is None or schema_file.filename == ""
        ):
            logger.warning(
                "Extract request missing schema",
                request_id=request_id,
            )
            raise ValidationError(
                message="Either 'schema' (JSON string) or 'schema_file' must be provided",
                field="schema",
            )

        # Validate file is provided with content
        if file.filename is None or file.filename == "":
            logger.warning(
                "Extract request missing file",
                request_id=request_id,
            )
            raise ValidationError(
                message="A document file must be provided",
                field="file",
            )

        # Read file content to check size
        file_content = await file.read()
        file_size = len(file_content)

        # Check file size
        if file_size > settings.max_file_size_bytes:
            logger.warning(
                "File too large",
                file_size=file_size,
                max_size=settings.max_file_size_bytes,
                request_id=request_id,
            )
            raise FileTooLargeError(
                file_size=file_size,
                max_size=settings.max_file_size_bytes,
            )

        # Parse JSON schema
        schema_content: dict[str, Any]
        if extraction_schema is not None:
            try:
                schema_content = json.loads(extraction_schema)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Invalid JSON schema string",
                    error=str(e),
                    request_id=request_id,
                )
                raise SchemaParseError(
                    message=f"Invalid JSON in schema: {e}",
                    details={"parse_error": str(e)},
                ) from e
        else:
            # schema_file must be provided (validated above)
            assert schema_file is not None
            schema_bytes = await schema_file.read()
            try:
                schema_content = json.loads(schema_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(
                    "Invalid JSON schema file",
                    error=str(e),
                    request_id=request_id,
                )
                raise SchemaParseError(
                    message=f"Invalid JSON in schema file: {e}",
                    details={"parse_error": str(e)},
                ) from e

        # Validate JSON Schema syntax and structure
        schema_validator = SchemaValidator()
        try:
            schema_info = schema_validator.validate(schema_content)
            logger.info(
                "Schema validated",
                schema_type=schema_info.schema_type,
                required_fields=len(schema_info.required_fields),
                optional_fields=len(schema_info.optional_fields),
                request_id=request_id,
            )
        except SchemaValidationError as e:
            error_details = "; ".join(e.errors) if e.errors else str(e)
            logger.warning(
                "Schema validation failed",
                errors=e.errors,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON Schema: {error_details}",
            ) from e

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Set job ID in context for logging
        set_job_id(job_id)

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

        # Create job in job manager
        job_manager = get_job_manager()
        job_manager.create_job(
            job_id=job_id,
            filename=file.filename or "unknown",
            file_path=str(temp_filepath),
            schema_path=str(schema_filepath),
        )

        logger.info(
            "Document uploaded successfully",
            job_id=job_id,
            filename=file.filename,
            file_size=file_size,
            request_id=request_id,
        )

        # Start background processing
        background_tasks.add_task(process_extraction_job, job_id)

        return {
            "job_id": job_id,
            "filename": file.filename or "unknown",
            "file_size": file_size,
            "status": JobStatus.PENDING,
            "message": "Document uploaded successfully. Processing will begin shortly.",
        }

    @app.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["Jobs"],
        responses={
            404: {"model": ErrorDetail, "description": "Job not found"},
            410: {"model": ErrorDetail, "description": "Job expired"},
        },
    )
    async def get_job_status(request: Request, job_id: str) -> dict[str, Any]:
        """Get the status of an extraction job.

        Args:
            request: FastAPI request object
            job_id: The unique job identifier returned from POST /extract

        Returns:
            JobStatusResponse: Current job status and metadata

        Raises:
            HTTPException: 404 if job not found
            HTTPException: 410 if job has expired
        """
        request_id = getattr(request.state, "request_id", None)
        job_manager = get_job_manager()

        try:
            job = job_manager.get_job(job_id)
        except JobNotFoundError:
            logger.warning(
                "Job not found",
                job_id=job_id,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            ) from None
        except JobExpiredError:
            logger.info(
                "Job expired",
                job_id=job_id,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"Job has expired: {job_id}. Results are only retained for "
                f"{settings.job_ttl_hours} hours.",
            ) from None

        logger.debug(
            "Job status retrieved",
            job_id=job_id,
            status=job.status.value,
            request_id=request_id,
        )

        return {
            "job_id": job.job_id,
            "status": job.status,
            "filename": job.filename,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "progress": job.progress,
            "error_message": job.error_message,
        }

    @app.get(
        "/jobs/{job_id}/result",
        response_model=JobResultResponse,
        tags=["Jobs"],
        responses={
            404: {"model": ErrorDetail, "description": "Job not found"},
            410: {"model": ErrorDetail, "description": "Job expired"},
            425: {"model": ErrorDetail, "description": "Job not yet complete"},
        },
    )
    async def get_job_result(request: Request, job_id: str) -> dict[str, Any]:
        """Get the result of a completed extraction job.

        Args:
            request: FastAPI request object
            job_id: The unique job identifier returned from POST /extract

        Returns:
            JobResultResponse: Extraction results including data, markdown,
                metadata, and quality report

        Raises:
            HTTPException: 404 if job not found
            HTTPException: 410 if job has expired
            HTTPException: 425 if job is not yet complete
        """
        request_id = getattr(request.state, "request_id", None)
        job_manager = get_job_manager()

        try:
            job = job_manager.get_job(job_id)
        except JobNotFoundError:
            logger.warning(
                "Job not found for result retrieval",
                job_id=job_id,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            ) from None
        except JobExpiredError:
            logger.info(
                "Job expired for result retrieval",
                job_id=job_id,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"Job has expired: {job_id}. Results are only retained for "
                f"{settings.job_ttl_hours} hours.",
            ) from None

        # Check if job is complete
        if job.status == JobStatus.PENDING:
            logger.debug(
                "Result requested for pending job",
                job_id=job_id,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail=f"Job {job_id} is pending. Check status at GET /jobs/{job_id}",
            )

        if job.status == JobStatus.PROCESSING:
            logger.debug(
                "Result requested for processing job",
                job_id=job_id,
                progress=job.progress,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_425_TOO_EARLY,
                detail=f"Job {job_id} is still processing. "
                f"Progress: {job.progress or 'Unknown'}. "
                f"Check status at GET /jobs/{job_id}",
            )

        # Build response based on job status
        metadata = None
        if job.metadata:
            metadata = ExtractionMetadata(
                processing_time_seconds=job.metadata.get("processing_time_seconds", 0),
                model_used=job.metadata.get("model_used", "unknown"),
                total_tokens=job.metadata.get("total_tokens", 0),
                iterations_completed=job.metadata.get("iterations_completed", 0),
                converged=job.metadata.get("converged", False),
                document_type=job.metadata.get("document_type", "unknown"),
            )

        logger.info(
            "Job result retrieved",
            job_id=job_id,
            status=job.status.value,
            has_data=job.extracted_data is not None,
            request_id=request_id,
        )

        return {
            "job_id": job.job_id,
            "status": job.status,
            "extracted_data": job.extracted_data,
            "markdown_summary": job.markdown_summary,
            "metadata": metadata,
            "quality_report": job.quality_report,
            "error_message": job.error_message,
        }

    logger.info("FastAPI application created successfully")
    return app


# Create the application instance
app = create_app()
