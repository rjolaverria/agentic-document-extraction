"""Tests for the job manager service."""

import threading
import time
from datetime import UTC, datetime

import pytest

from agentic_document_extraction.models import JobStatus
from agentic_document_extraction.services.job_manager import (
    JobData,
    JobExpiredError,
    JobManager,
    JobManagerConfig,
    JobNotFoundError,
    get_job_manager,
    reset_job_manager,
)


@pytest.fixture
def job_manager() -> JobManager:
    """Create a job manager for testing with auto-cleanup disabled."""
    config = JobManagerConfig(
        ttl_seconds=3600,
        cleanup_interval_seconds=300,
        enable_auto_cleanup=False,
    )
    return JobManager(config=config)


@pytest.fixture
def short_ttl_job_manager() -> JobManager:
    """Create a job manager with a very short TTL for expiration testing."""
    config = JobManagerConfig(
        ttl_seconds=1,  # 1 second TTL
        cleanup_interval_seconds=300,
        enable_auto_cleanup=False,
    )
    return JobManager(config=config)


@pytest.fixture(autouse=True)
def cleanup_global_manager() -> None:
    """Reset the global job manager after each test."""
    yield
    reset_job_manager()


class TestJobManagerCreation:
    """Tests for job creation functionality."""

    def test_create_job_returns_job_data(self, job_manager: JobManager) -> None:
        """Test that create_job returns a JobData instance."""
        job = job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        assert isinstance(job, JobData)
        assert job.job_id == "test-123"
        assert job.filename == "test.txt"
        assert job.file_path == "/tmp/test.txt"
        assert job.schema_path == "/tmp/schema.json"

    def test_create_job_sets_initial_status_to_pending(
        self, job_manager: JobManager
    ) -> None:
        """Test that newly created jobs have PENDING status."""
        job = job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        assert job.status == JobStatus.PENDING

    def test_create_job_sets_timestamps(self, job_manager: JobManager) -> None:
        """Test that created_at and updated_at are set on creation."""
        before = datetime.now(UTC)

        job = job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        after = datetime.now(UTC)

        assert before <= job.created_at <= after
        assert before <= job.updated_at <= after
        assert job.created_at == job.updated_at

    def test_create_job_sets_initial_progress(self, job_manager: JobManager) -> None:
        """Test that initial progress message is set."""
        job = job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        assert job.progress is not None
        assert len(job.progress) > 0


class TestJobRetrieval:
    """Tests for job retrieval functionality."""

    def test_get_job_returns_correct_job(self, job_manager: JobManager) -> None:
        """Test that get_job returns the correct job by ID."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        retrieved = job_manager.get_job("test-123")

        assert retrieved.job_id == "test-123"
        assert retrieved.filename == "test.txt"

    def test_get_job_raises_not_found_for_missing_job(
        self, job_manager: JobManager
    ) -> None:
        """Test that get_job raises JobNotFoundError for missing jobs."""
        with pytest.raises(JobNotFoundError) as exc_info:
            job_manager.get_job("nonexistent")

        assert exc_info.value.job_id == "nonexistent"

    def test_get_job_raises_expired_for_old_job(
        self, short_ttl_job_manager: JobManager
    ) -> None:
        """Test that get_job raises JobExpiredError for expired jobs."""
        short_ttl_job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        # Mark job as completed to set expiration
        short_ttl_job_manager.update_status(
            "test-123", JobStatus.COMPLETED, progress="Done"
        )

        # Wait for TTL to expire
        time.sleep(1.5)

        with pytest.raises(JobExpiredError) as exc_info:
            short_ttl_job_manager.get_job("test-123")

        assert exc_info.value.job_id == "test-123"

    def test_job_exists_returns_true_for_existing_job(
        self, job_manager: JobManager
    ) -> None:
        """Test that job_exists returns True for existing jobs."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        assert job_manager.job_exists("test-123") is True

    def test_job_exists_returns_false_for_missing_job(
        self, job_manager: JobManager
    ) -> None:
        """Test that job_exists returns False for missing jobs."""
        assert job_manager.job_exists("nonexistent") is False


class TestJobStatusUpdates:
    """Tests for job status update functionality."""

    def test_update_status_changes_status(self, job_manager: JobManager) -> None:
        """Test that update_status changes the job status."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        job_manager.update_status("test-123", JobStatus.PROCESSING)

        job = job_manager.get_job("test-123")
        assert job.status == JobStatus.PROCESSING

    def test_update_status_changes_progress(self, job_manager: JobManager) -> None:
        """Test that update_status can change progress message."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        job_manager.update_status(
            "test-123", JobStatus.PROCESSING, progress="Processing started"
        )

        job = job_manager.get_job("test-123")
        assert job.progress == "Processing started"

    def test_update_status_changes_updated_at(self, job_manager: JobManager) -> None:
        """Test that update_status changes the updated_at timestamp."""
        job = job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )
        original_updated_at = job.updated_at

        time.sleep(0.01)  # Small delay to ensure timestamp differs

        job_manager.update_status("test-123", JobStatus.PROCESSING)

        updated_job = job_manager.get_job("test-123")
        assert updated_job.updated_at > original_updated_at

    def test_update_status_sets_expiration_on_completion(
        self, job_manager: JobManager
    ) -> None:
        """Test that completing a job sets the expiration time."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        job_manager.update_status("test-123", JobStatus.COMPLETED)

        job = job_manager.get_job("test-123")
        assert job.expires_at is not None

    def test_update_status_sets_expiration_on_failure(
        self, job_manager: JobManager
    ) -> None:
        """Test that failing a job sets the expiration time."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        job_manager.update_status("test-123", JobStatus.FAILED, error_message="Error")

        job = job_manager.get_job("test-123")
        assert job.expires_at is not None

    def test_update_status_raises_not_found(self, job_manager: JobManager) -> None:
        """Test that update_status raises JobNotFoundError for missing jobs."""
        with pytest.raises(JobNotFoundError):
            job_manager.update_status("nonexistent", JobStatus.PROCESSING)


class TestJobResults:
    """Tests for job result functionality."""

    def test_set_result_stores_extracted_data(self, job_manager: JobManager) -> None:
        """Test that set_result stores extracted data."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        extracted_data = {"name": "John", "age": 30}
        job_manager.set_result("test-123", extracted_data)

        job = job_manager.get_job("test-123")
        assert job.extracted_data == extracted_data

    def test_set_result_stores_markdown_summary(self, job_manager: JobManager) -> None:
        """Test that set_result stores markdown summary."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        job_manager.set_result(
            "test-123",
            extracted_data={"name": "John"},
            markdown_summary="# Results\n\n- Name: John",
        )

        job = job_manager.get_job("test-123")
        assert job.markdown_summary == "# Results\n\n- Name: John"

    def test_set_result_stores_metadata(self, job_manager: JobManager) -> None:
        """Test that set_result stores metadata."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        metadata = {"processing_time_seconds": 1.5, "model_used": "gpt-4"}
        job_manager.set_result(
            "test-123",
            extracted_data={"name": "John"},
            metadata=metadata,
        )

        job = job_manager.get_job("test-123")
        assert job.metadata == metadata

    def test_set_result_stores_quality_report(self, job_manager: JobManager) -> None:
        """Test that set_result stores quality report."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        quality_report = {"status": "passed", "confidence": 0.95}
        job_manager.set_result(
            "test-123",
            extracted_data={"name": "John"},
            quality_report=quality_report,
        )

        job = job_manager.get_job("test-123")
        assert job.quality_report == quality_report

    def test_set_result_marks_job_completed(self, job_manager: JobManager) -> None:
        """Test that set_result changes status to COMPLETED."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        job_manager.set_result("test-123", extracted_data={"name": "John"})

        job = job_manager.get_job("test-123")
        assert job.status == JobStatus.COMPLETED

    def test_set_result_sets_expiration(self, job_manager: JobManager) -> None:
        """Test that set_result sets expiration time."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        job_manager.set_result("test-123", extracted_data={"name": "John"})

        job = job_manager.get_job("test-123")
        assert job.expires_at is not None

    def test_set_failed_marks_job_failed(self, job_manager: JobManager) -> None:
        """Test that set_failed changes status to FAILED."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        job_manager.set_failed("test-123", "Something went wrong")

        job = job_manager.get_job("test-123")
        assert job.status == JobStatus.FAILED
        assert job.error_message == "Something went wrong"


class TestJobCleanup:
    """Tests for job cleanup functionality."""

    def test_cleanup_expired_jobs_removes_old_jobs(
        self, short_ttl_job_manager: JobManager
    ) -> None:
        """Test that cleanup_expired_jobs removes expired jobs."""
        # Create a job and complete it (triggers expiration)
        short_ttl_job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )
        short_ttl_job_manager.set_result("test-123", extracted_data={"name": "John"})

        # Wait for TTL to expire
        time.sleep(1.5)

        # Run cleanup
        removed = short_ttl_job_manager.cleanup_expired_jobs()

        assert removed == 1
        assert short_ttl_job_manager.get_job_count() == 0

    def test_cleanup_expired_jobs_keeps_non_expired_jobs(
        self, job_manager: JobManager
    ) -> None:
        """Test that cleanup keeps non-expired jobs."""
        # Create a job (not completed, so no expiration)
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        removed = job_manager.cleanup_expired_jobs()

        assert removed == 0
        assert job_manager.get_job_count() == 1

    def test_clear_all_removes_all_jobs(self, job_manager: JobManager) -> None:
        """Test that clear_all removes all jobs."""
        job_manager.create_job(
            job_id="test-1",
            filename="test1.txt",
            file_path="/tmp/test1.txt",
            schema_path="/tmp/schema1.json",
        )
        job_manager.create_job(
            job_id="test-2",
            filename="test2.txt",
            file_path="/tmp/test2.txt",
            schema_path="/tmp/schema2.json",
        )

        job_manager.clear_all()

        assert job_manager.get_job_count() == 0


class TestJobManagerUtilities:
    """Tests for job manager utility methods."""

    def test_get_all_jobs_returns_all_jobs(self, job_manager: JobManager) -> None:
        """Test that get_all_jobs returns all non-expired jobs."""
        job_manager.create_job(
            job_id="test-1",
            filename="test1.txt",
            file_path="/tmp/test1.txt",
            schema_path="/tmp/schema1.json",
        )
        job_manager.create_job(
            job_id="test-2",
            filename="test2.txt",
            file_path="/tmp/test2.txt",
            schema_path="/tmp/schema2.json",
        )

        all_jobs = job_manager.get_all_jobs()

        assert len(all_jobs) == 2
        job_ids = {job.job_id for job in all_jobs}
        assert job_ids == {"test-1", "test-2"}

    def test_get_job_count_returns_correct_count(self, job_manager: JobManager) -> None:
        """Test that get_job_count returns the correct count."""
        assert job_manager.get_job_count() == 0

        job_manager.create_job(
            job_id="test-1",
            filename="test1.txt",
            file_path="/tmp/test1.txt",
            schema_path="/tmp/schema1.json",
        )
        assert job_manager.get_job_count() == 1

        job_manager.create_job(
            job_id="test-2",
            filename="test2.txt",
            file_path="/tmp/test2.txt",
            schema_path="/tmp/schema2.json",
        )
        assert job_manager.get_job_count() == 2

    def test_job_data_to_dict(self, job_manager: JobManager) -> None:
        """Test that JobData.to_dict returns correct dictionary."""
        job = job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        data = job.to_dict()

        assert data["job_id"] == "test-123"
        assert data["filename"] == "test.txt"
        assert data["status"] == "pending"
        assert "created_at" in data
        assert "updated_at" in data


class TestThreadSafety:
    """Tests for thread safety of job manager."""

    def test_concurrent_job_creation(self, job_manager: JobManager) -> None:
        """Test that concurrent job creation is thread-safe."""
        num_threads = 10
        jobs_per_thread = 10
        threads = []
        errors = []

        def create_jobs(thread_id: int) -> None:
            try:
                for i in range(jobs_per_thread):
                    job_manager.create_job(
                        job_id=f"thread-{thread_id}-job-{i}",
                        filename=f"test-{thread_id}-{i}.txt",
                        file_path=f"/tmp/test-{thread_id}-{i}.txt",
                        schema_path=f"/tmp/schema-{thread_id}-{i}.json",
                    )
            except Exception as e:
                errors.append(e)

        for i in range(num_threads):
            t = threading.Thread(target=create_jobs, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert job_manager.get_job_count() == num_threads * jobs_per_thread

    def test_concurrent_status_updates(self, job_manager: JobManager) -> None:
        """Test that concurrent status updates are thread-safe."""
        job_manager.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        num_threads = 10
        threads = []
        errors = []

        def update_status(thread_id: int) -> None:
            try:
                for _ in range(10):
                    job_manager.update_status(
                        "test-123",
                        JobStatus.PROCESSING,
                        progress=f"Thread {thread_id} update",
                    )
            except Exception as e:
                errors.append(e)

        for i in range(num_threads):
            t = threading.Thread(target=update_status, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        # Job should still exist and be valid
        job = job_manager.get_job("test-123")
        assert job.status == JobStatus.PROCESSING


class TestGlobalJobManager:
    """Tests for global job manager access."""

    def test_get_job_manager_returns_same_instance(self) -> None:
        """Test that get_job_manager returns the same instance."""
        manager1 = get_job_manager()
        manager2 = get_job_manager()

        assert manager1 is manager2

    def test_reset_job_manager_clears_instance(self) -> None:
        """Test that reset_job_manager clears the global instance."""
        manager1 = get_job_manager()
        manager1.create_job(
            job_id="test-123",
            filename="test.txt",
            file_path="/tmp/test.txt",
            schema_path="/tmp/schema.json",
        )

        reset_job_manager()

        manager2 = get_job_manager()
        assert manager1 is not manager2
        assert manager2.get_job_count() == 0
