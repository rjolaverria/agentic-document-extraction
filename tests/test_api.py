"""Tests for the FastAPI application."""

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from docket import Progress, Worker
from fastapi import status

from agentic_document_extraction.api import create_app


@asynccontextmanager
async def create_test_client(
    patches: dict[str, Any] | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    """Create an async test client with proper lifespan handling.

    Args:
        patches: Optional dictionary of patch targets and values.
    """
    patch_targets = {
        "agentic_document_extraction.api.settings.docket_url": "memory://",
        "agentic_document_extraction.api.settings.docket_name": "test-docket",
        **(patches or {}),
    }

    with patch.dict("os.environ", {}, clear=False):
        for target, value in patch_targets.items():
            patch(target, value).start()
        try:
            app = create_app()
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                ) as client,
            ):
                client.app = app  # type: ignore[attr-defined]
                yield client
        finally:
            patch.stopall()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Create an async test client for the FastAPI application."""
    async with create_test_client() as ac:
        yield ac


@pytest.fixture
def sample_schema() -> dict[str, object]:
    """Return a sample JSON schema for testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "amount": {"type": "number"},
        },
        "required": ["name"],
    }


@pytest.fixture
def sample_file_content() -> bytes:
    """Return sample file content for testing."""
    return b"This is a test document content."


_DEFAULT_PROGRESS = Progress()


async def _stub_extraction_job(
    _job_id: str,
    _filename: str,
    _file_path: str,
    _schema_path: str,
    progress: Progress | None = _DEFAULT_PROGRESS,
) -> dict[str, object]:
    if progress is not None:
        await progress.set_message("Extraction completed")
    return {
        "extracted_data": {"name": "John Doe", "amount": 100},
        "markdown_summary": "# Results\n\n- Name: John Doe\n- Amount: 100",
        "metadata": {
            "processing_time_seconds": 1.5,
            "model_used": "gpt-4",
            "total_tokens": 500,
            "iterations_completed": 1,
            "converged": True,
            "document_type": "text_based",
        },
        "quality_report": {"status": "passed", "confidence": 0.95},
    }


async def _failing_extraction_job(
    _job_id: str,
    _filename: str,
    _file_path: str,
    _schema_path: str,
    progress: Progress | None = _DEFAULT_PROGRESS,
) -> dict[str, object]:
    if progress is not None:
        await progress.set_message("Job failed")
    raise RuntimeError("Extraction failed due to API error")


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    async def test_health_check_returns_200(self, client: httpx.AsyncClient) -> None:
        """Test that health check returns 200 status code."""
        response = await client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    async def test_health_check_response_structure(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test that health check returns expected response structure."""
        response = await client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data

    async def test_health_check_status_is_healthy(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test that health check status is 'healthy'."""
        response = await client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    async def test_health_check_version_matches(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test that health check returns correct version."""
        response = await client.get("/health")
        data = response.json()

        assert data["version"] == "0.1.0"

    async def test_health_check_timestamp_is_valid_iso_format(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test that health check timestamp is a valid ISO format datetime."""
        response = await client.get("/health")
        data = response.json()

        # Should not raise an exception if valid ISO format
        parsed_timestamp = datetime.fromisoformat(data["timestamp"])
        assert isinstance(parsed_timestamp, datetime)


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation availability."""

    async def test_docs_endpoint_available(self, client: httpx.AsyncClient) -> None:
        """Test that Swagger UI docs are available at /docs."""
        response = await client.get("/docs")
        assert response.status_code == status.HTTP_200_OK

    async def test_redoc_endpoint_available(self, client: httpx.AsyncClient) -> None:
        """Test that ReDoc documentation is available at /redoc."""
        response = await client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK

    async def test_openapi_json_available(self, client: httpx.AsyncClient) -> None:
        """Test that OpenAPI JSON schema is available."""
        response = await client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Agentic Document Extraction API"


class TestExtractEndpoint:
    """Tests for the document upload/extract endpoint."""

    async def test_extract_accepts_file_with_schema_string(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test successful upload with file and schema as string."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "job_id" in data
        assert data["filename"] == "test.txt"
        assert data["file_size"] == len(sample_file_content)
        assert "message" in data

    async def test_extract_accepts_file_with_schema_file(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test successful upload with file and schema as file upload."""
        schema_bytes = json.dumps(sample_schema).encode("utf-8")

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={
                    "file": ("test.pdf", sample_file_content, "application/pdf"),
                    "schema_file": (
                        "schema.json",
                        schema_bytes,
                        "application/json",
                    ),
                },
            )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "job_id" in data
        assert data["filename"] == "test.pdf"

    async def test_extract_returns_400_when_schema_missing(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that 400 is returned when schema is missing."""
        response = await client.post(
            "/extract",
            files={"file": ("test.txt", sample_file_content, "text/plain")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "schema" in data["detail"].lower()

    async def test_extract_returns_400_for_invalid_json_schema(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that 400 is returned for invalid JSON in schema string."""
        response = await client.post(
            "/extract",
            files={"file": ("test.txt", sample_file_content, "text/plain")},
            data={"schema": "{ invalid json }"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "json" in data["detail"].lower()

    async def test_extract_returns_400_for_invalid_schema_file(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that 400 is returned for invalid JSON in schema file."""
        response = await client.post(
            "/extract",
            files={
                "file": ("test.txt", sample_file_content, "text/plain"),
                "schema_file": ("schema.json", b"not valid json", "application/json"),
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data

    async def test_extract_returns_413_for_large_file(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
    ) -> None:
        """Test that 413 is returned when file exceeds size limit."""
        # Create content larger than the limit (set to small value for test)
        large_content = b"x" * 1000  # 1KB

        # Create a mock settings object with a small file size limit
        mock_settings = type(
            "MockSettings",
            (),
            {
                "max_file_size_bytes": 500,
                "max_file_size_mb": 0,
                "temp_upload_dir": "/tmp",
            },
        )()

        with patch("agentic_document_extraction.api.settings", mock_settings):
            response = await client.post(
                "/extract",
                files={"file": ("large.txt", large_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )

        assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
        data = response.json()
        assert "detail" in data
        assert "size" in data["detail"].lower()

    async def test_extract_saves_file_with_unique_id(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that uploaded file is saved with unique identifier."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )

            data = response.json()
            job_id = data["job_id"]

            # Check that files were created
            upload_path = Path(temp_dir)
            saved_files = list(upload_path.glob(f"{job_id}*"))
            assert len(saved_files) == 2  # document + schema

            # Check document file exists and has correct content
            doc_file = upload_path / f"{job_id}.txt"
            assert doc_file.exists()
            assert doc_file.read_bytes() == sample_file_content

            # Check schema file exists
            schema_file = upload_path / f"{job_id}_schema.json"
            assert schema_file.exists()

    async def test_extract_job_id_is_valid_uuid(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that returned job_id is a valid UUID."""
        import uuid

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )

        data = response.json()
        # Should not raise an exception if valid UUID
        parsed_uuid = uuid.UUID(data["job_id"])
        assert isinstance(parsed_uuid, uuid.UUID)

    async def test_extract_preserves_file_extension(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that file extension is preserved in saved file."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={
                    "file": ("document.pdf", sample_file_content, "application/pdf")
                },
                data={"schema": json.dumps(sample_schema)},
            )

            data = response.json()
            job_id = data["job_id"]

            # Check that file has .pdf extension
            doc_file = Path(temp_dir) / f"{job_id}.pdf"
            assert doc_file.exists()

    async def test_extract_supports_various_file_formats(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that various file formats are accepted."""
        formats = [
            ("test.txt", "text/plain"),
            ("test.pdf", "application/pdf"),
            (
                "test.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("test.png", "image/png"),
            ("test.jpg", "image/jpeg"),
            ("test.csv", "text/csv"),
        ]

        for filename, mime_type in formats:
            with (
                tempfile.TemporaryDirectory() as temp_dir,
                patch(
                    "agentic_document_extraction.api.settings.temp_upload_dir", temp_dir
                ),
            ):
                response = await client.post(
                    "/extract",
                    files={"file": (filename, sample_file_content, mime_type)},
                    data={"schema": json.dumps(sample_schema)},
                )

            assert response.status_code == status.HTTP_202_ACCEPTED, (
                f"Failed for format: {filename}"
            )


class TestSchemaValidation:
    """Tests for JSON Schema validation in the extract endpoint."""

    async def test_valid_schema_accepted(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that valid JSON schemas are accepted."""
        valid_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["name"],
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(valid_schema)},
            )

        assert response.status_code == status.HTTP_202_ACCEPTED

    async def test_nested_schema_accepted(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that nested JSON schemas are accepted."""
        nested_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(nested_schema)},
            )

        assert response.status_code == status.HTTP_202_ACCEPTED

    async def test_array_schema_accepted(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that array schemas are accepted."""
        array_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(array_schema)},
            )

        assert response.status_code == status.HTTP_202_ACCEPTED

    async def test_invalid_type_returns_400(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that schemas with invalid types return 400."""
        invalid_schema = {
            "type": "object",
            "properties": {
                "field": {"type": "invalid_type"},
            },
        }

        response = await client.post(
            "/extract",
            files={"file": ("test.txt", sample_file_content, "text/plain")},
            data={"schema": json.dumps(invalid_schema)},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert (
            "invalid" in data["detail"].lower()
            or "unsupported" in data["detail"].lower()
        )

    async def test_malformed_required_returns_400(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that schemas with malformed 'required' field return 400."""
        invalid_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": "name",  # Should be an array
        }

        response = await client.post(
            "/extract",
            files={"file": ("test.txt", sample_file_content, "text/plain")},
            data={"schema": json.dumps(invalid_schema)},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data

    async def test_non_object_schema_returns_400(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that non-object schemas return 400."""
        response = await client.post(
            "/extract",
            files={"file": ("test.txt", sample_file_content, "text/plain")},
            data={"schema": json.dumps("not a schema object")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "object" in data["detail"].lower()

    async def test_draft7_schema_accepted(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that explicit Draft 7 schemas are accepted."""
        draft7_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(draft7_schema)},
            )

        assert response.status_code == status.HTTP_202_ACCEPTED

    async def test_schema_with_all_common_types(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that all common JSON Schema types are supported."""
        comprehensive_schema = {
            "type": "object",
            "properties": {
                "string_field": {"type": "string"},
                "number_field": {"type": "number"},
                "integer_field": {"type": "integer"},
                "boolean_field": {"type": "boolean"},
                "object_field": {"type": "object"},
                "array_field": {"type": "array"},
                "null_field": {"type": "null"},
                "nullable_string": {"type": ["string", "null"]},
            },
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(comprehensive_schema)},
            )

        assert response.status_code == status.HTTP_202_ACCEPTED

    async def test_empty_schema_accepted(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that empty schemas are accepted (matches any JSON)."""
        empty_schema: dict[str, object] = {}

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(empty_schema)},
            )

        assert response.status_code == status.HTTP_202_ACCEPTED

    async def test_schema_error_message_is_descriptive(
        self,
        client: httpx.AsyncClient,
        sample_file_content: bytes,
    ) -> None:
        """Test that schema validation errors include descriptive details."""
        invalid_schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "not_a_valid_type"},
                    },
                },
            },
        }

        response = await client.post(
            "/extract",
            files={"file": ("test.txt", sample_file_content, "text/plain")},
            data={"schema": json.dumps(invalid_schema)},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        # Error should mention the problematic field path
        assert "nested" in data["detail"] or "not_a_valid_type" in data["detail"]


class TestJobStatusEndpoint:
    """Tests for the job status endpoint."""

    async def test_get_job_status_returns_job_info(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that GET /jobs/{job_id} returns job status info."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            # Create a job via the extract endpoint
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )

            job_id = response.json()["job_id"]

            # Get job status
            status_response = await client.get(f"/jobs/{job_id}")

            assert status_response.status_code == status.HTTP_200_OK
            data = status_response.json()
            assert data["job_id"] == job_id
            assert data["status"] == "pending"
            assert data["filename"] == "test.txt"
            assert "created_at" in data
            assert "updated_at" in data

    async def test_get_job_status_returns_404_for_unknown_job(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Test that GET /jobs/{job_id} returns 404 for unknown job."""
        response = await client.get("/jobs/nonexistent-job-id")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    async def test_get_job_status_tracks_progress(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that job status includes progress information."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            # Create a job
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )
            job_id = response.json()["job_id"]

            # Access docket from the client's app reference
            app = client.app  # type: ignore[attr-defined]
            docket = app.state.docket
            execution = await docket.get_execution(job_id)
            assert execution is not None
            await execution.progress.set_message("Processing document")

            # Get job status
            status_response = await client.get(f"/jobs/{job_id}")

            assert status_response.status_code == status.HTTP_200_OK
            data = status_response.json()
            assert data["status"] == "pending"
            assert data["progress"] == "Processing document"


class TestJobResultEndpoint:
    """Tests for the job result endpoint."""

    async def test_get_job_result_returns_completed_result(
        self,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that GET /jobs/{job_id}/result returns completed job result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            async with create_test_client(
                {
                    "agentic_document_extraction.api.settings.temp_upload_dir": temp_dir,
                    "agentic_document_extraction.api.process_extraction_job": (
                        _stub_extraction_job
                    ),
                }
            ) as local_client:
                # Create a job
                response = await local_client.post(
                    "/extract",
                    files={"file": ("test.txt", sample_file_content, "text/plain")},
                    data={"schema": json.dumps(sample_schema)},
                )
                job_id = response.json()["job_id"]

                app = local_client.app  # type: ignore[attr-defined]
                docket = app.state.docket
                async with Worker(docket) as worker:
                    await worker.run_until_finished()

                # Get job result
                result_response = await local_client.get(f"/jobs/{job_id}/result")

                assert result_response.status_code == status.HTTP_200_OK
                data = result_response.json()
                assert data["job_id"] == job_id
                assert data["status"] == "completed"
                assert data["extracted_data"] == {"name": "John Doe", "amount": 100}
                assert "markdown_summary" in data
                assert data["metadata"]["model_used"] == "gpt-4"
                assert data["quality_report"]["status"] == "passed"

    async def test_get_job_result_returns_425_for_pending_job(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that GET /jobs/{job_id}/result returns 425 for pending job."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            # Create a job (stays pending)
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )
            job_id = response.json()["job_id"]

            # Try to get result of pending job
            result_response = await client.get(f"/jobs/{job_id}/result")

            assert result_response.status_code == status.HTTP_425_TOO_EARLY
            data = result_response.json()
            assert "detail" in data
            assert "pending" in data["detail"].lower()

    async def test_get_job_result_returns_425_for_processing_job(
        self,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that GET /jobs/{job_id}/result returns 425 for processing job."""
        task_ready = asyncio.Event()
        task_release = asyncio.Event()

        async def slow_extraction_job(
            _job_id: str,
            _filename: str,
            _file_path: str,
            _schema_path: str,
            progress: Progress | None = _DEFAULT_PROGRESS,
        ) -> dict[str, object]:
            if progress is not None:
                await progress.set_message("Processing document")
            task_ready.set()
            await task_release.wait()
            return {"extracted_data": {"status": "ok"}}

        with tempfile.TemporaryDirectory() as temp_dir:
            async with create_test_client(
                {
                    "agentic_document_extraction.api.settings.temp_upload_dir": temp_dir,
                    "agentic_document_extraction.api.process_extraction_job": (
                        slow_extraction_job
                    ),
                }
            ) as local_client:
                response = await local_client.post(
                    "/extract",
                    files={"file": ("test.txt", sample_file_content, "text/plain")},
                    data={"schema": json.dumps(sample_schema)},
                )
                job_id = response.json()["job_id"]

                app = local_client.app  # type: ignore[attr-defined]
                docket = app.state.docket
                async with Worker(docket) as worker:
                    worker_task = asyncio.create_task(worker.run_until_finished())

                    await asyncio.wait_for(task_ready.wait(), timeout=2)

                    result_response = await local_client.get(f"/jobs/{job_id}/result")

                    assert result_response.status_code == status.HTTP_425_TOO_EARLY
                    data = result_response.json()
                    assert "detail" in data
                    assert "processing" in data["detail"].lower()

                    task_release.set()
                    await asyncio.wait_for(worker_task, timeout=2)

    async def test_get_job_result_returns_failed_result(
        self,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that GET /jobs/{job_id}/result returns failed job info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            async with create_test_client(
                {
                    "agentic_document_extraction.api.settings.temp_upload_dir": temp_dir,
                    "agentic_document_extraction.api.process_extraction_job": (
                        _failing_extraction_job
                    ),
                }
            ) as local_client:
                response = await local_client.post(
                    "/extract",
                    files={"file": ("test.txt", sample_file_content, "text/plain")},
                    data={"schema": json.dumps(sample_schema)},
                )
                job_id = response.json()["job_id"]

                app = local_client.app  # type: ignore[attr-defined]
                docket = app.state.docket
                async with Worker(docket) as worker:
                    await worker.run_until_finished()

                result_response = await local_client.get(f"/jobs/{job_id}/result")

                assert result_response.status_code == status.HTTP_200_OK
                data = result_response.json()
                assert data["job_id"] == job_id
                assert data["status"] == "failed"
                assert (
                    data["error_message"]
                    == "RuntimeError: Extraction failed due to API error"
                )
                assert data["extracted_data"] is None

    async def test_get_job_result_returns_404_for_unknown_job(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Test that GET /jobs/{job_id}/result returns 404 for unknown job."""
        response = await client.get("/jobs/nonexistent-job-id/result")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()


class TestExtractWithDocket:
    """Tests for Docket integration in extract endpoint."""

    async def test_extract_returns_pending_status(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that extract returns pending status initially."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )

            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert data["status"] == "pending"

    async def test_extract_creates_job_in_docket_store(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that extract creates a job in Docket metadata store."""
        from agentic_document_extraction.services.docket_jobs import DocketJobStore

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )

            job_id = response.json()["job_id"]

            # Access docket from the client's app reference
            app = client.app  # type: ignore[attr-defined]
            docket = app.state.docket
            job_store = DocketJobStore(docket)
            metadata = await job_store.get(job_id)
            assert metadata.filename == "test.txt"

    async def test_extract_creates_docket_execution(
        self,
        client: httpx.AsyncClient,
        sample_schema: dict[str, object],
        sample_file_content: bytes,
    ) -> None:
        """Test that extract schedules a Docket execution."""

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("agentic_document_extraction.api.settings.temp_upload_dir", temp_dir),
        ):
            response = await client.post(
                "/extract",
                files={"file": ("test.txt", sample_file_content, "text/plain")},
                data={"schema": json.dumps(sample_schema)},
            )

            _job_id = response.json()["job_id"]
            # Access docket from the client's app reference
            app = client.app  # type: ignore[attr-defined]
            docket = app.state.docket
            execution = await docket.get_execution(_job_id)
            assert execution is not None
