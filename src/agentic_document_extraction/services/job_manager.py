"""Job management service for async extraction processing.

This module provides in-memory job storage with TTL support for tracking
extraction jobs and their results. Jobs progress through states:
pending -> processing -> completed/failed.

Key features:
- Thread-safe job storage and updates
- Automatic TTL-based cleanup of old jobs
- Job state tracking with timestamps
- Result storage with quality reports
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agentic_document_extraction.config import settings
from agentic_document_extraction.models import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class JobData:
    """Internal data structure for tracking a job."""

    job_id: str
    status: JobStatus
    filename: str
    file_path: str
    schema_path: str
    created_at: datetime
    updated_at: datetime
    progress: str | None = None
    error_message: str | None = None
    extracted_data: dict[str, Any] | None = None
    markdown_summary: str | None = None
    metadata: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert job data to dictionary.

        Returns:
            Dictionary representation of job data.
        """
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "filename": self.filename,
            "file_path": self.file_path,
            "schema_path": self.schema_path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "error_message": self.error_message,
            "extracted_data": self.extracted_data,
            "markdown_summary": self.markdown_summary,
            "metadata": self.metadata,
            "quality_report": self.quality_report,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class JobNotFoundError(Exception):
    """Raised when a job is not found."""

    def __init__(self, job_id: str) -> None:
        """Initialize with job ID.

        Args:
            job_id: The job ID that was not found.
        """
        super().__init__(f"Job not found: {job_id}")
        self.job_id = job_id


class JobExpiredError(Exception):
    """Raised when a job has expired."""

    def __init__(self, job_id: str) -> None:
        """Initialize with job ID.

        Args:
            job_id: The job ID that has expired.
        """
        super().__init__(f"Job has expired: {job_id}")
        self.job_id = job_id


@dataclass
class JobManagerConfig:
    """Configuration for the job manager."""

    ttl_seconds: int = field(default_factory=lambda: settings.job_ttl_seconds)
    cleanup_interval_seconds: int = 300  # 5 minutes
    enable_auto_cleanup: bool = True


class JobManager:
    """Thread-safe in-memory job manager with TTL support.

    Manages extraction jobs through their lifecycle, providing:
    - Job creation and tracking
    - Status updates with progress
    - Result storage
    - Automatic cleanup of expired jobs
    """

    def __init__(self, config: JobManagerConfig | None = None) -> None:
        """Initialize the job manager.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or JobManagerConfig()
        self._jobs: dict[str, JobData] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: threading.Thread | None = None
        self._stop_cleanup = threading.Event()

        if self.config.enable_auto_cleanup:
            self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="JobManagerCleanup",
        )
        self._cleanup_thread.start()
        logger.info("Job manager cleanup thread started")

    def _cleanup_loop(self) -> None:
        """Background loop for cleaning up expired jobs."""
        while not self._stop_cleanup.wait(self.config.cleanup_interval_seconds):
            try:
                self.cleanup_expired_jobs()
            except Exception as e:
                logger.error(f"Error in job cleanup: {e}")

    def stop_cleanup(self) -> None:
        """Stop the background cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
            logger.info("Job manager cleanup thread stopped")

    def create_job(
        self,
        job_id: str,
        filename: str,
        file_path: str,
        schema_path: str,
    ) -> JobData:
        """Create a new job entry.

        Args:
            job_id: Unique job identifier.
            filename: Original filename.
            file_path: Path to the uploaded file.
            schema_path: Path to the schema file.

        Returns:
            The created JobData instance.
        """
        now = datetime.now(UTC)
        job = JobData(
            job_id=job_id,
            status=JobStatus.PENDING,
            filename=filename,
            file_path=file_path,
            schema_path=schema_path,
            created_at=now,
            updated_at=now,
            progress="Job created, waiting to start",
        )

        with self._lock:
            self._jobs[job_id] = job

        logger.info(f"Job created: {job_id}, filename={filename}")
        return job

    def get_job(self, job_id: str) -> JobData:
        """Get a job by ID.

        Args:
            job_id: The job ID to look up.

        Returns:
            The JobData for the job.

        Raises:
            JobNotFoundError: If the job doesn't exist.
            JobExpiredError: If the job has expired.
        """
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                raise JobNotFoundError(job_id)

            if job.expires_at and datetime.now(UTC) > job.expires_at:
                # Clean up expired job
                del self._jobs[job_id]
                raise JobExpiredError(job_id)

            return job

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists and is not expired.

        Args:
            job_id: The job ID to check.

        Returns:
            True if job exists and is not expired.
        """
        try:
            self.get_job(job_id)
            return True
        except (JobNotFoundError, JobExpiredError):
            return False

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: str | None = None,
        error_message: str | None = None,
    ) -> JobData:
        """Update a job's status.

        Args:
            job_id: The job ID to update.
            status: New status.
            progress: Optional progress description.
            error_message: Optional error message (for failed status).

        Returns:
            The updated JobData.

        Raises:
            JobNotFoundError: If the job doesn't exist.
        """
        with self._lock:
            job = self.get_job(job_id)
            job.status = status
            job.updated_at = datetime.now(UTC)

            if progress is not None:
                job.progress = progress

            if error_message is not None:
                job.error_message = error_message

            # Set expiration when job is complete
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.expires_at = datetime.fromtimestamp(
                    time.time() + self.config.ttl_seconds, tz=UTC
                )

            logger.info(f"Job {job_id} status updated to {status.value}")
            return job

    def set_result(
        self,
        job_id: str,
        extracted_data: dict[str, Any],
        markdown_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        quality_report: dict[str, Any] | None = None,
    ) -> JobData:
        """Set the extraction result for a completed job.

        Args:
            job_id: The job ID.
            extracted_data: The extracted data.
            markdown_summary: Optional markdown summary.
            metadata: Optional extraction metadata.
            quality_report: Optional quality verification report.

        Returns:
            The updated JobData.

        Raises:
            JobNotFoundError: If the job doesn't exist.
        """
        with self._lock:
            job = self.get_job(job_id)
            job.extracted_data = extracted_data
            job.markdown_summary = markdown_summary
            job.metadata = metadata
            job.quality_report = quality_report
            job.updated_at = datetime.now(UTC)
            job.status = JobStatus.COMPLETED
            job.progress = "Extraction completed"
            job.expires_at = datetime.fromtimestamp(
                time.time() + self.config.ttl_seconds, tz=UTC
            )

            logger.info(f"Job {job_id} completed with results")
            return job

    def set_failed(
        self,
        job_id: str,
        error_message: str,
    ) -> JobData:
        """Mark a job as failed.

        Args:
            job_id: The job ID.
            error_message: Description of the error.

        Returns:
            The updated JobData.

        Raises:
            JobNotFoundError: If the job doesn't exist.
        """
        with self._lock:
            job = self.get_job(job_id)
            job.status = JobStatus.FAILED
            job.error_message = error_message
            job.progress = "Job failed"
            job.updated_at = datetime.now(UTC)
            job.expires_at = datetime.fromtimestamp(
                time.time() + self.config.ttl_seconds, tz=UTC
            )

            logger.error(f"Job {job_id} failed: {error_message}")
            return job

    def cleanup_expired_jobs(self) -> int:
        """Remove expired jobs from storage.

        Returns:
            Number of jobs cleaned up.
        """
        now = datetime.now(UTC)
        removed_count = 0

        with self._lock:
            expired_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.expires_at and now > job.expires_at
            ]

            for job_id in expired_ids:
                del self._jobs[job_id]
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired jobs")

        return removed_count

    def get_all_jobs(self) -> list[JobData]:
        """Get all non-expired jobs.

        Returns:
            List of all current jobs.
        """
        now = datetime.now(UTC)
        with self._lock:
            return [
                job
                for job in self._jobs.values()
                if not job.expires_at or now <= job.expires_at
            ]

    def get_job_count(self) -> int:
        """Get the current number of jobs.

        Returns:
            Number of jobs in storage.
        """
        with self._lock:
            return len(self._jobs)

    def clear_all(self) -> None:
        """Clear all jobs from storage. Used primarily for testing."""
        with self._lock:
            self._jobs.clear()
        logger.info("All jobs cleared")


# Global job manager instance
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        The global JobManager instance.
    """
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


def reset_job_manager() -> None:
    """Reset the global job manager. Used primarily for testing."""
    global _job_manager
    if _job_manager is not None:
        _job_manager.stop_cleanup()
        _job_manager = None
