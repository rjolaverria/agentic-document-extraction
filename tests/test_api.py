"""Tests for the FastAPI application."""

import json
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi import status

from agentic_document_extraction.api import app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Create an async test client for the FastAPI application."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
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
