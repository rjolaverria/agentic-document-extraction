"""Docket client factory for API lifecycle management."""

from __future__ import annotations

from docket import Docket, _result_store

from agentic_document_extraction.config import settings


def build_docket() -> Docket:
    """Build a configured Docket instance."""
    result_storage = None
    if settings.docket_result_storage_url:
        # Docket uses Redis-compatible stores for result persistence.
        result_storage = _result_store.RedisStore(  # type: ignore[attr-defined]
            url=settings.docket_result_storage_url
        )
    return Docket(
        name=settings.docket_name,
        url=settings.docket_url,
        execution_ttl=settings.docket_execution_ttl,
        result_storage=result_storage,
        enable_internal_instrumentation=settings.docket_enable_internal_instrumentation,
    )
