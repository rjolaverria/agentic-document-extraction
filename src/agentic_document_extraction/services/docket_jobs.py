"""Docket-backed job metadata and status helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from docket import Docket
from docket.execution import Execution, ExecutionState

from agentic_document_extraction.config import settings
from agentic_document_extraction.models import JobStatus
from agentic_document_extraction.utils.exceptions import (
    JobExpiredError,
    JobNotFoundError,
)


@dataclass(frozen=True)
class JobMetadata:
    """Metadata tracked for each extraction job."""

    job_id: str
    filename: str
    file_path: str
    schema_path: str
    created_at: datetime
    expires_at: datetime

    @classmethod
    def from_redis(cls, data: dict[str, str]) -> JobMetadata:
        """Create metadata from Redis hash data."""
        return cls(
            job_id=data["job_id"],
            filename=data["filename"],
            file_path=data["file_path"],
            schema_path=data["schema_path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
        )

    def to_redis(self) -> dict[str, str]:
        """Serialize metadata for Redis storage."""
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "schema_path": self.schema_path,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass(frozen=True)
class JobStatusSnapshot:
    """Status snapshot built from Docket execution state."""

    job_id: str
    status: JobStatus
    filename: str
    created_at: datetime
    updated_at: datetime
    progress: str | None
    error_message: str | None


class DocketJobStore:
    """Store and retrieve job metadata in Docket's Redis backend."""

    def __init__(self, docket: Docket) -> None:
        self._docket = docket

    def _metadata_key(self, job_id: str) -> str:
        return self._docket.key(f"jobs:{job_id}")

    async def create(
        self,
        job_id: str,
        filename: str,
        file_path: str,
        schema_path: str,
    ) -> JobMetadata:
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=settings.job_ttl_seconds)
        metadata = JobMetadata(
            job_id=job_id,
            filename=filename,
            file_path=file_path,
            schema_path=schema_path,
            created_at=now,
            expires_at=expires_at,
        )
        async with self._docket.redis() as redis:
            await redis.hset(  # type: ignore[misc]
                self._metadata_key(job_id), mapping=metadata.to_redis()
            )
        return metadata

    async def get(self, job_id: str) -> JobMetadata:
        async with self._docket.redis() as redis:
            data = await redis.hgetall(  # type: ignore[misc]
                self._metadata_key(job_id)
            )

        if not data:
            raise JobNotFoundError(job_id)

        decoded = {
            (key.decode() if isinstance(key, bytes) else str(key)): (
                value.decode() if isinstance(value, bytes) else str(value)
            )
            for key, value in data.items()
        }
        metadata = JobMetadata.from_redis(decoded)
        if datetime.now(UTC) > metadata.expires_at:
            await self.delete(job_id)
            raise JobExpiredError(job_id, ttl_hours=settings.job_ttl_hours)
        return metadata

    async def delete(self, job_id: str) -> None:
        async with self._docket.redis() as redis:
            await redis.delete(self._metadata_key(job_id))


def map_execution_state(state: ExecutionState) -> JobStatus:
    """Map Docket execution state to API job status."""
    if state in (ExecutionState.SCHEDULED, ExecutionState.QUEUED):
        return JobStatus.PENDING
    if state == ExecutionState.RUNNING:
        return JobStatus.PROCESSING
    if state == ExecutionState.COMPLETED:
        return JobStatus.COMPLETED
    return JobStatus.FAILED


async def load_execution(docket: Docket, job_id: str) -> Execution:
    """Load execution state and progress for a job."""
    execution = await docket.get_execution(job_id)
    if execution is None:
        raise JobNotFoundError(job_id)
    await execution.sync()
    await execution.progress.sync()
    return execution


def build_status_snapshot(
    metadata: JobMetadata,
    execution: Execution,
) -> JobStatusSnapshot:
    """Build a status snapshot from metadata and execution state."""
    progress_message = execution.progress.message
    updated_candidates: list[datetime] = [metadata.created_at]
    if execution.started_at:
        updated_candidates.append(execution.started_at)
    if execution.completed_at:
        updated_candidates.append(execution.completed_at)
    if execution.progress.updated_at:
        updated_candidates.append(execution.progress.updated_at)
    updated_at = max(updated_candidates)
    return JobStatusSnapshot(
        job_id=metadata.job_id,
        status=map_execution_state(execution.state),
        filename=metadata.filename,
        created_at=metadata.created_at,
        updated_at=updated_at,
        progress=progress_message,
        error_message=execution.error,
    )
