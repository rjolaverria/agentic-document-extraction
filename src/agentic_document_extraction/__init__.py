"""Agentic Document Extraction - Vision-first document extraction system."""

from agentic_document_extraction.api import app, create_app

__all__ = ["app", "create_app"]
__version__ = "0.1.0"


def main() -> None:
    """Run the FastAPI server using uvicorn."""
    import uvicorn

    uvicorn.run(
        "agentic_document_extraction.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
