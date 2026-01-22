"""Test fixtures and helpers for loading sample documents and schemas.

This module provides utility functions for loading sample documents and
JSON schemas used in tests and examples.

Example usage:
    from tests.fixtures import load_sample_document, load_sample_schema

    # Load a sample document
    content = load_sample_document("sample_invoice.txt")

    # Load a sample schema
    schema = load_sample_schema("invoice_schema.json")
"""

import json
from pathlib import Path
from typing import Any

# Base path to fixtures directory
FIXTURES_DIR = Path(__file__).parent
SAMPLE_DOCUMENTS_DIR = FIXTURES_DIR / "sample_documents"
SAMPLE_SCHEMAS_DIR = FIXTURES_DIR / "sample_schemas"


def load_sample_document(filename: str) -> str:
    """Load a sample document file as a string.

    Args:
        filename: Name of the file in sample_documents directory.

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        >>> content = load_sample_document("sample_invoice.txt")
        >>> "ACME Corporation" in content
        True
    """
    filepath = SAMPLE_DOCUMENTS_DIR / filename
    return filepath.read_text(encoding="utf-8")


def load_sample_document_bytes(filename: str) -> bytes:
    """Load a sample document file as bytes.

    Useful for binary files like PDFs or images.

    Args:
        filename: Name of the file in sample_documents directory.

    Returns:
        The file contents as bytes.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    filepath = SAMPLE_DOCUMENTS_DIR / filename
    return filepath.read_bytes()


def load_sample_schema(filename: str) -> dict[str, Any]:
    """Load a sample JSON schema as a dictionary.

    Args:
        filename: Name of the schema file in sample_schemas directory.

    Returns:
        The parsed JSON schema as a dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.

    Example:
        >>> schema = load_sample_schema("invoice_schema.json")
        >>> schema["title"]
        'Invoice'
    """
    filepath = SAMPLE_SCHEMAS_DIR / filename
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def get_sample_document_path(filename: str) -> Path:
    """Get the full path to a sample document.

    Args:
        filename: Name of the file in sample_documents directory.

    Returns:
        Path object pointing to the file.
    """
    return SAMPLE_DOCUMENTS_DIR / filename


def get_sample_schema_path(filename: str) -> Path:
    """Get the full path to a sample schema file.

    Args:
        filename: Name of the file in sample_schemas directory.

    Returns:
        Path object pointing to the file.
    """
    return SAMPLE_SCHEMAS_DIR / filename


def list_sample_documents() -> list[str]:
    """List all available sample documents.

    Returns:
        List of filenames in the sample_documents directory.
    """
    if not SAMPLE_DOCUMENTS_DIR.exists():
        return []
    return [f.name for f in SAMPLE_DOCUMENTS_DIR.iterdir() if f.is_file()]


def list_sample_schemas() -> list[str]:
    """List all available sample schemas.

    Returns:
        List of filenames in the sample_schemas directory.
    """
    if not SAMPLE_SCHEMAS_DIR.exists():
        return []
    return [f.name for f in SAMPLE_SCHEMAS_DIR.iterdir() if f.is_file()]
