"""FastAPI application for agentic document extraction."""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    timestamp: str
    version: str


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

    logger.info("FastAPI application created successfully")
    return app


# Create the application instance
app = create_app()
