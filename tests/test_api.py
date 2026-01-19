"""Tests for the FastAPI application."""

from collections.abc import AsyncIterator
from datetime import datetime

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
