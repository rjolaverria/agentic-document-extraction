"""Tests for Docket-backed job helpers."""

import pytest
from docket import Docket

from agentic_document_extraction.services.docket_jobs import (
    DocketJobStore,
    load_execution,
)
from agentic_document_extraction.utils.exceptions import JobNotFoundError


@pytest.mark.asyncio
async def test_job_store_round_trip() -> None:
    """Job metadata should persist in Docket's backend."""
    docket = Docket(name="test-docket", url="memory://")
    await docket.__aenter__()
    try:
        store = DocketJobStore(docket)
        created = await store.create(
            job_id="job-123",
            filename="example.txt",
            file_path="/tmp/example.txt",
            schema_path="/tmp/schema.json",
        )
        loaded = await store.get("job-123")
        assert loaded == created
    finally:
        await docket.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_load_execution_raises_for_missing_job() -> None:
    """Missing executions should raise JobNotFoundError."""
    docket = Docket(name="test-docket", url="memory://")
    await docket.__aenter__()
    try:
        with pytest.raises(JobNotFoundError):
            await load_execution(docket, "missing-job")
    finally:
        await docket.__aexit__(None, None, None)
