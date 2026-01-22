"""Tests for the structured logging utilities."""

import logging
import time
from unittest.mock import MagicMock, patch

from agentic_document_extraction.utils.logging import (
    LogContext,
    PerformanceMetrics,
    ProgressTracker,
    StructuredLogFormatter,
    StructuredLogger,
    clear_context,
    configure_logging,
    get_extra_context,
    get_job_id,
    get_logger,
    get_request_id,
    set_extra_context,
    set_job_id,
    set_request_id,
    timed_operation,
)


class TestContextVariables:
    """Tests for context variable management."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_context()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_context()

    def test_request_id_default_none(self) -> None:
        """Request ID should default to None."""
        assert get_request_id() is None

    def test_set_and_get_request_id(self) -> None:
        """Should be able to set and get request ID."""
        set_request_id("req-123")
        assert get_request_id() == "req-123"

    def test_clear_request_id(self) -> None:
        """Should be able to clear request ID."""
        set_request_id("req-123")
        set_request_id(None)
        assert get_request_id() is None

    def test_job_id_default_none(self) -> None:
        """Job ID should default to None."""
        assert get_job_id() is None

    def test_set_and_get_job_id(self) -> None:
        """Should be able to set and get job ID."""
        set_job_id("job-456")
        assert get_job_id() == "job-456"

    def test_extra_context_default_empty(self) -> None:
        """Extra context should default to empty dict."""
        assert get_extra_context() == {}

    def test_set_and_get_extra_context(self) -> None:
        """Should be able to set and get extra context."""
        ctx = {"operation": "extraction", "page": 5}
        set_extra_context(ctx)
        assert get_extra_context() == ctx

    def test_clear_context(self) -> None:
        """Clear context should reset all context variables."""
        set_request_id("req-123")
        set_job_id("job-456")
        set_extra_context({"key": "value"})

        clear_context()

        assert get_request_id() is None
        assert get_job_id() is None
        assert get_extra_context() == {}


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        metrics = PerformanceMetrics(operation="test_op")
        assert metrics.operation == "test_op"
        assert metrics.duration_seconds == 0.0
        assert metrics.tokens_used == 0
        assert metrics.pages_processed == 0
        assert metrics.regions_analyzed == 0
        assert metrics.iterations == 0
        assert metrics.api_calls == 0
        assert metrics.custom_metrics == {}

    def test_finish_calculates_duration(self) -> None:
        """Finish should calculate duration."""
        metrics = PerformanceMetrics(operation="test_op")
        time.sleep(0.01)  # Small delay
        metrics.finish()
        assert metrics.duration_seconds > 0
        assert metrics.end_time is not None

    def test_to_dict_basic(self) -> None:
        """Test to_dict with basic data."""
        metrics = PerformanceMetrics(operation="test_op")
        metrics.duration_seconds = 1.5
        result = metrics.to_dict()
        assert result["operation"] == "test_op"
        assert result["duration_seconds"] == 1.5

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields populated."""
        metrics = PerformanceMetrics(operation="test_op")
        metrics.duration_seconds = 2.0
        metrics.tokens_used = 1000
        metrics.pages_processed = 5
        metrics.regions_analyzed = 20
        metrics.iterations = 3
        metrics.api_calls = 10
        metrics.custom_metrics = {"accuracy": 0.95}

        result = metrics.to_dict()
        assert result["tokens_used"] == 1000
        assert result["pages_processed"] == 5
        assert result["regions_analyzed"] == 20
        assert result["iterations"] == 3
        assert result["api_calls"] == 10
        assert result["custom_metrics"]["accuracy"] == 0.95

    def test_to_dict_excludes_zero_values(self) -> None:
        """to_dict should exclude zero values."""
        metrics = PerformanceMetrics(operation="test_op")
        metrics.duration_seconds = 1.0
        result = metrics.to_dict()
        assert "tokens_used" not in result
        assert "pages_processed" not in result


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.logger = get_logger("test_logger")

    def test_get_logger_returns_structured_logger(self) -> None:
        """get_logger should return StructuredLogger."""
        logger = get_logger(__name__)
        assert isinstance(logger, StructuredLogger)

    def test_logger_property(self) -> None:
        """Logger property should return underlying Python logger."""
        assert isinstance(self.logger.logger, logging.Logger)

    def test_build_message_without_kwargs(self) -> None:
        """_build_message without kwargs should return original message."""
        msg = self.logger._build_message("Test message")
        assert msg == "Test message"

    def test_build_message_with_kwargs(self) -> None:
        """_build_message with kwargs should include key-value pairs."""
        msg = self.logger._build_message("Test message", key="value", count=42)
        assert "Test message" in msg
        assert "key=value" in msg
        assert "count=42" in msg

    @patch.object(logging.Logger, "info")
    def test_info_logging(self, mock_info: MagicMock) -> None:
        """Info method should log at INFO level."""
        self.logger.info("Test info", status="ok")
        mock_info.assert_called_once()
        call_args = mock_info.call_args[0][0]
        assert "Test info" in call_args
        assert "status=ok" in call_args

    @patch.object(logging.Logger, "debug")
    def test_debug_logging(self, mock_debug: MagicMock) -> None:
        """Debug method should log at DEBUG level."""
        self.logger.debug("Test debug")
        mock_debug.assert_called_once()

    @patch.object(logging.Logger, "warning")
    def test_warning_logging(self, mock_warning: MagicMock) -> None:
        """Warning method should log at WARNING level."""
        self.logger.warning("Test warning")
        mock_warning.assert_called_once()

    @patch.object(logging.Logger, "error")
    def test_error_logging(self, mock_error: MagicMock) -> None:
        """Error method should log at ERROR level."""
        self.logger.error("Test error", exc_info=False)
        mock_error.assert_called_once()

    @patch.object(logging.Logger, "critical")
    def test_critical_logging(self, mock_critical: MagicMock) -> None:
        """Critical method should log at CRITICAL level."""
        self.logger.critical("Test critical")
        mock_critical.assert_called_once()

    @patch.object(logging.Logger, "exception")
    def test_exception_logging(self, mock_exception: MagicMock) -> None:
        """Exception method should log with traceback."""
        self.logger.exception("Test exception")
        mock_exception.assert_called_once()

    @patch.object(logging.Logger, "info")
    def test_log_performance(self, mock_info: MagicMock) -> None:
        """log_performance should log metrics."""
        metrics = PerformanceMetrics(operation="test")
        metrics.duration_seconds = 1.5
        metrics.tokens_used = 1000
        self.logger.log_performance(metrics)
        mock_info.assert_called_once()
        call_args = mock_info.call_args[0][0]
        assert "Performance: test" in call_args

    @patch.object(logging.Logger, "info")
    def test_log_progress(self, mock_info: MagicMock) -> None:
        """log_progress should log progress information."""
        self.logger.log_progress("Processing pages", current=5, total=10)
        mock_info.assert_called_once()
        call_args = mock_info.call_args[0][0]
        assert "Progress: Processing pages" in call_args
        assert "current=5" in call_args
        assert "total=10" in call_args
        assert "50.0%" in call_args

    @patch.object(logging.Logger, "log")
    def test_log_api_call_success(self, mock_log: MagicMock) -> None:
        """log_api_call should log successful API calls."""
        self.logger.log_api_call(
            service="openai",
            operation="completion",
            duration_seconds=0.5,
            tokens_used=100,
            success=True,
        )
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[0][0] == logging.INFO
        assert "service=openai" in call_args[0][1]

    @patch.object(logging.Logger, "log")
    def test_log_api_call_failure(self, mock_log: MagicMock) -> None:
        """log_api_call should log failed API calls at ERROR level."""
        self.logger.log_api_call(
            service="openai",
            operation="completion",
            duration_seconds=0.5,
            success=False,
            error_message="Rate limit exceeded",
        )
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[0][0] == logging.ERROR

    @patch.object(logging.Logger, "log")
    def test_log_extraction_result(self, mock_log: MagicMock) -> None:
        """log_extraction_result should log extraction completion."""
        self.logger.log_extraction_result(
            job_id="job-123",
            success=True,
            duration_seconds=5.0,
            iterations=3,
            tokens_used=5000,
            converged=True,
            confidence=0.95,
        )
        mock_log.assert_called_once()
        call_args = mock_log.call_args[0][1]
        assert "Extraction completed" in call_args
        assert "job_id=job-123" in call_args


class TestLogContext:
    """Tests for LogContext context manager."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_context()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_context()

    def test_context_sets_values(self) -> None:
        """LogContext should set context values within block."""
        with LogContext(job_id="job-123", operation="test"):
            assert get_job_id() == "job-123"
            extra = get_extra_context()
            assert extra.get("operation") == "test"

    def test_context_restores_values(self) -> None:
        """LogContext should restore original values after block."""
        set_job_id("original-job")
        set_extra_context({"original": "value"})

        with LogContext(job_id="new-job", operation="test"):
            assert get_job_id() == "new-job"

        assert get_job_id() == "original-job"
        assert get_extra_context() == {"original": "value"}

    def test_context_with_request_id(self) -> None:
        """LogContext should handle request_id."""
        with LogContext(request_id="req-456", job_id="job-789"):
            assert get_request_id() == "req-456"
            assert get_job_id() == "job-789"

        assert get_request_id() is None
        assert get_job_id() is None

    def test_nested_contexts(self) -> None:
        """Nested LogContext should work correctly."""
        with LogContext(job_id="outer-job"):
            assert get_job_id() == "outer-job"
            with LogContext(job_id="inner-job"):
                assert get_job_id() == "inner-job"
            assert get_job_id() == "outer-job"


class TestTimedOperation:
    """Tests for timed_operation context manager."""

    @patch.object(StructuredLogger, "log_performance")
    def test_timed_operation_logs_metrics(self, mock_log: MagicMock) -> None:
        """timed_operation should log metrics on exit."""
        logger = get_logger("test")
        with timed_operation(logger, "test_op") as metrics:
            metrics.tokens_used = 100

        mock_log.assert_called_once()
        logged_metrics = mock_log.call_args[0][0]
        assert logged_metrics.operation == "test_op"
        assert logged_metrics.tokens_used == 100
        assert logged_metrics.duration_seconds > 0

    @patch.object(StructuredLogger, "log_performance")
    def test_timed_operation_yields_metrics(self, _mock_log: MagicMock) -> None:
        """timed_operation should yield PerformanceMetrics."""
        logger = get_logger("test")
        with timed_operation(logger, "test_op") as metrics:
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.operation == "test_op"


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    @patch.object(StructuredLogger, "log_progress")
    def test_update_logs_progress(self, mock_log: MagicMock) -> None:
        """update should log progress."""
        logger = get_logger("test")
        tracker = ProgressTracker(logger, "Processing", total=10)
        tracker.update()
        mock_log.assert_called_once()

    @patch.object(StructuredLogger, "log_progress")
    def test_update_with_interval(self, mock_log: MagicMock) -> None:
        """update should respect log_interval."""
        logger = get_logger("test")
        tracker = ProgressTracker(logger, "Processing", total=10, log_interval=5)
        for _ in range(4):
            tracker.update()
        assert mock_log.call_count == 0
        tracker.update()  # 5th update
        assert mock_log.call_count == 1

    @patch.object(StructuredLogger, "info")
    @patch.object(StructuredLogger, "log_progress")
    def test_complete_returns_duration(
        self, _mock_progress: MagicMock, mock_info: MagicMock
    ) -> None:
        """complete should return total duration."""
        logger = get_logger("test")
        tracker = ProgressTracker(logger, "Processing", total=10)
        duration = tracker.complete()
        assert duration > 0
        mock_info.assert_called_once()


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_string_level(self) -> None:
        """configure_logging should accept string level."""
        configure_logging(level="DEBUG")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_with_int_level(self) -> None:
        """configure_logging should accept int level."""
        configure_logging(level=logging.WARNING)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_configure_with_structured_formatter(self) -> None:
        """configure_logging should use StructuredLogFormatter by default."""
        configure_logging(use_structured_formatter=True)
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, StructuredLogFormatter)

    def test_configure_without_structured_formatter(self) -> None:
        """configure_logging can use standard formatter."""
        configure_logging(use_structured_formatter=False)
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
        handler = root_logger.handlers[0]
        assert not isinstance(handler.formatter, StructuredLogFormatter)


class TestStructuredLogFormatter:
    """Tests for StructuredLogFormatter class."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_context()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_context()

    def test_format_without_context(self) -> None:
        """Formatter should work without context."""
        formatter = StructuredLogFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "Test message" in result

    def test_format_with_request_id(self) -> None:
        """Formatter should include request_id when set."""
        set_request_id("req-123")
        formatter = StructuredLogFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "request_id=req-123" in result
        assert "Test message" in result

    def test_format_with_job_id(self) -> None:
        """Formatter should include job_id when set."""
        set_job_id("job-456")
        formatter = StructuredLogFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "job_id=job-456" in result

    def test_format_with_extra_context(self) -> None:
        """Formatter should include extra context."""
        set_extra_context({"operation": "extraction"})
        formatter = StructuredLogFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "operation=extraction" in result
