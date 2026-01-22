"""Structured logging utilities for agentic document extraction.

This module provides:
- Request ID tracking using contextvars for correlation across the pipeline
- Structured logging with consistent format and metadata
- Performance metrics logging helpers
- Progress tracking for long-running operations

Usage:
    from agentic_document_extraction.utils.logging import (
        get_logger,
        set_request_id,
        LogContext,
    )

    logger = get_logger(__name__)

    # Set request ID for correlation
    set_request_id("abc-123")

    # Log with context
    with LogContext(job_id="job-456", operation="extraction"):
        logger.info("Processing document")

    # Log performance metrics
    logger.log_performance(
        operation="text_extraction",
        duration_seconds=1.5,
        tokens_used=1000,
    )
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Context variables for request tracking
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
_extra_context_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "extra_context", default=None
)


def get_request_id() -> str | None:
    """Get the current request ID from context.

    Returns:
        The current request ID or None if not set.
    """
    return _request_id_var.get()


def set_request_id(request_id: str | None) -> None:
    """Set the request ID in context.

    Args:
        request_id: The request ID to set, or None to clear.
    """
    _request_id_var.set(request_id)


def get_job_id() -> str | None:
    """Get the current job ID from context.

    Returns:
        The current job ID or None if not set.
    """
    return _job_id_var.get()


def set_job_id(job_id: str | None) -> None:
    """Set the job ID in context.

    Args:
        job_id: The job ID to set, or None to clear.
    """
    _job_id_var.set(job_id)


def get_extra_context() -> dict[str, Any]:
    """Get additional context from context vars.

    Returns:
        Dictionary of extra context values.
    """
    ctx = _extra_context_var.get()
    return ctx if ctx is not None else {}


def set_extra_context(context: dict[str, Any]) -> None:
    """Set additional context in context vars.

    Args:
        context: Dictionary of extra context values.
    """
    _extra_context_var.set(context)


def clear_context() -> None:
    """Clear all context variables."""
    _request_id_var.set(None)
    _job_id_var.set(None)
    _extra_context_var.set(None)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics during processing.

    Attributes:
        operation: Name of the operation being measured.
        start_time: When the operation started.
        end_time: When the operation ended.
        duration_seconds: Duration in seconds.
        tokens_used: Number of tokens used (if applicable).
        pages_processed: Number of pages processed (if applicable).
        regions_analyzed: Number of regions analyzed (if applicable).
        iterations: Number of iterations completed.
        api_calls: Number of API calls made.
        custom_metrics: Additional custom metrics.
    """

    operation: str
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    tokens_used: int = 0
    pages_processed: int = 0
    regions_analyzed: int = 0
    iterations: int = 0
    api_calls: int = 0
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        """Mark the operation as complete and calculate duration."""
        self.end_time = datetime.now(UTC)
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging.

        Returns:
            Dictionary with all metrics.
        """
        result = {
            "operation": self.operation,
            "duration_seconds": self.duration_seconds,
        }
        if self.tokens_used > 0:
            result["tokens_used"] = self.tokens_used
        if self.pages_processed > 0:
            result["pages_processed"] = self.pages_processed
        if self.regions_analyzed > 0:
            result["regions_analyzed"] = self.regions_analyzed
        if self.iterations > 0:
            result["iterations"] = self.iterations
        if self.api_calls > 0:
            result["api_calls"] = self.api_calls
        if self.custom_metrics:
            result["custom_metrics"] = self.custom_metrics
        return result


class StructuredLogFormatter(logging.Formatter):
    """Custom log formatter that includes context variables.

    This formatter adds request_id and job_id to log records when available,
    creating a consistent structured format for all log messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with context information.

        Args:
            record: The log record to format.

        Returns:
            Formatted log message string.
        """
        # Add context information to the record
        request_id = get_request_id()
        job_id = get_job_id()

        # Build prefix with context info
        prefix_parts = []
        if request_id:
            prefix_parts.append(f"request_id={request_id}")
        if job_id:
            prefix_parts.append(f"job_id={job_id}")

        # Add extra context if any
        extra = get_extra_context()
        for key, value in extra.items():
            prefix_parts.append(f"{key}={value}")

        prefix = f"[{' '.join(prefix_parts)}] " if prefix_parts else ""

        # Store original message
        original_msg = record.msg

        # Prepend context to message
        record.msg = f"{prefix}{original_msg}"

        # Format with parent formatter
        result = super().format(record)

        # Restore original message
        record.msg = original_msg

        return result


class StructuredLogger:
    """Enhanced logger with structured logging capabilities.

    Wraps a standard Python logger with additional methods for:
    - Logging with automatic context inclusion
    - Performance metrics logging
    - Progress tracking
    - Structured error logging with exception details
    """

    def __init__(self, name: str) -> None:
        """Initialize the structured logger.

        Args:
            name: Logger name (typically __name__ of the module).
        """
        self._logger = logging.getLogger(name)
        self._name = name

    @property
    def logger(self) -> logging.Logger:
        """Access the underlying Python logger.

        Returns:
            The standard logging.Logger instance.
        """
        return self._logger

    def _build_message(
        self,
        message: str,
        **kwargs: Any,
    ) -> str:
        """Build a message with structured key-value pairs.

        Args:
            message: Base message.
            **kwargs: Additional key-value pairs to include.

        Returns:
            Formatted message string.
        """
        if not kwargs:
            return message

        parts = [f"{k}={v}" for k, v in kwargs.items()]
        return f"{message} | {', '.join(parts)}"

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message.

        Args:
            message: Log message.
            **kwargs: Additional structured data.
        """
        self._logger.debug(self._build_message(message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message.

        Args:
            message: Log message.
            **kwargs: Additional structured data.
        """
        self._logger.info(self._build_message(message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message.

        Args:
            message: Log message.
            **kwargs: Additional structured data.
        """
        self._logger.warning(self._build_message(message, **kwargs))

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log an error message.

        Args:
            message: Log message.
            exc_info: Whether to include exception info.
            **kwargs: Additional structured data.
        """
        self._logger.error(self._build_message(message, **kwargs), exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log a critical message.

        Args:
            message: Log message.
            exc_info: Whether to include exception info.
            **kwargs: Additional structured data.
        """
        self._logger.critical(self._build_message(message, **kwargs), exc_info=exc_info)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with traceback.

        Args:
            message: Log message.
            **kwargs: Additional structured data.
        """
        self._logger.exception(self._build_message(message, **kwargs))

    def log_performance(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics.

        Args:
            metrics: Performance metrics to log.
        """
        self.info(
            f"Performance: {metrics.operation}",
            **metrics.to_dict(),
        )

    def log_progress(
        self,
        stage: str,
        current: int,
        total: int,
        details: str | None = None,
    ) -> None:
        """Log progress for long-running operations.

        Args:
            stage: Current processing stage.
            current: Current progress count.
            total: Total items to process.
            details: Optional additional details.
        """
        percentage = (current / total * 100) if total > 0 else 0
        message = f"Progress: {stage}"
        kwargs: dict[str, Any] = {
            "current": current,
            "total": total,
            "percentage": f"{percentage:.1f}%",
        }
        if details:
            kwargs["details"] = details
        self.info(message, **kwargs)

    def log_api_call(
        self,
        service: str,
        operation: str,
        duration_seconds: float,
        tokens_used: int | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Log an API call with relevant metrics.

        Args:
            service: Service name (e.g., "openai").
            operation: Operation performed.
            duration_seconds: Time taken for the call.
            tokens_used: Tokens consumed (if applicable).
            success: Whether the call succeeded.
            error_message: Error message if call failed.
        """
        kwargs: dict[str, Any] = {
            "service": service,
            "operation": operation,
            "duration_seconds": f"{duration_seconds:.3f}",
            "success": success,
        }
        if tokens_used is not None:
            kwargs["tokens_used"] = tokens_used
        if error_message:
            kwargs["error"] = error_message

        level = logging.INFO if success else logging.ERROR
        self._logger.log(level, self._build_message("API call", **kwargs))

    def log_extraction_result(
        self,
        job_id: str,
        success: bool,
        duration_seconds: float,
        iterations: int,
        tokens_used: int,
        converged: bool,
        confidence: float | None = None,
    ) -> None:
        """Log extraction job completion.

        Args:
            job_id: Job identifier.
            success: Whether extraction succeeded.
            duration_seconds: Total processing time.
            iterations: Number of refinement iterations.
            tokens_used: Total tokens consumed.
            converged: Whether quality thresholds were met.
            confidence: Final confidence score.
        """
        kwargs: dict[str, Any] = {
            "job_id": job_id,
            "success": success,
            "duration_seconds": f"{duration_seconds:.2f}",
            "iterations": iterations,
            "tokens_used": tokens_used,
            "converged": converged,
        }
        if confidence is not None:
            kwargs["confidence"] = f"{confidence:.3f}"

        level = logging.INFO if success else logging.ERROR
        self._logger.log(level, self._build_message("Extraction completed", **kwargs))


class LogContext:
    """Context manager for adding temporary context to logs.

    Usage:
        with LogContext(job_id="123", operation="extraction"):
            logger.info("Processing...")  # Will include job_id and operation
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with context values.

        Args:
            **kwargs: Key-value pairs to add to log context.
        """
        self._new_context = kwargs
        self._old_context: dict[str, Any] = {}
        self._old_job_id: str | None = None
        self._old_request_id: str | None = None

    def __enter__(self) -> "LogContext":
        """Enter the context, saving old values and setting new ones."""
        # Save current context
        self._old_context = get_extra_context().copy()
        self._old_job_id = get_job_id()
        self._old_request_id = get_request_id()

        # Extract special keys
        job_id = self._new_context.pop("job_id", None)
        request_id = self._new_context.pop("request_id", None)

        # Set job_id if provided
        if job_id is not None:
            set_job_id(job_id)

        # Set request_id if provided
        if request_id is not None:
            set_request_id(request_id)

        # Merge remaining context
        merged = self._old_context.copy()
        merged.update(self._new_context)
        set_extra_context(merged)

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context, restoring old values."""
        set_extra_context(self._old_context)
        set_job_id(self._old_job_id)
        set_request_id(self._old_request_id)


@contextmanager
def timed_operation(
    logger: StructuredLogger,
    operation: str,
) -> Generator[PerformanceMetrics, None, None]:
    """Context manager for timing operations.

    Usage:
        with timed_operation(logger, "extraction") as metrics:
            # Do work
            metrics.tokens_used = 1000

        # Automatically logs: "Performance: extraction | duration_seconds=..."

    Args:
        logger: Logger to use for output.
        operation: Name of the operation.

    Yields:
        PerformanceMetrics instance for tracking.
    """
    metrics = PerformanceMetrics(operation=operation)
    try:
        yield metrics
    finally:
        metrics.finish()
        logger.log_performance(metrics)


def configure_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
    use_structured_formatter: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (int or string like "INFO").
        format_string: Custom format string (uses default if None).
        use_structured_formatter: Whether to use the structured formatter.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Set formatter
    formatter: logging.Formatter
    if use_structured_formatter:
        formatter = StructuredLogFormatter(format_string)
    else:
        formatter = logging.Formatter(format_string)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a module.

    This is the primary way to get a logger in the application.
    It returns a StructuredLogger that wraps the standard Python logger
    with additional capabilities.

    Args:
        name: Logger name (typically __name__).

    Returns:
        StructuredLogger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing document", doc_type="pdf", pages=10)
    """
    return StructuredLogger(name)


class ProgressTracker:
    """Helper for tracking and logging progress of multi-step operations.

    Usage:
        tracker = ProgressTracker(logger, "Processing pages", total=10)
        for page in pages:
            process(page)
            tracker.update(details=f"Page {page.num}")
        tracker.complete()
    """

    def __init__(
        self,
        logger: StructuredLogger,
        stage: str,
        total: int,
        log_interval: int = 1,
    ) -> None:
        """Initialize the progress tracker.

        Args:
            logger: Logger to use.
            stage: Description of the stage being tracked.
            total: Total number of items.
            log_interval: Log every N updates (1 = every update).
        """
        self._logger = logger
        self._stage = stage
        self._total = total
        self._current = 0
        self._log_interval = log_interval
        self._start_time = time.time()

    def update(
        self,
        increment: int = 1,
        details: str | None = None,
    ) -> None:
        """Update progress.

        Args:
            increment: Number of items completed.
            details: Optional details about current item.
        """
        self._current += increment
        if self._current % self._log_interval == 0 or self._current == self._total:
            self._logger.log_progress(
                self._stage,
                self._current,
                self._total,
                details,
            )

    def complete(self) -> float:
        """Mark progress as complete.

        Returns:
            Total duration in seconds.
        """
        duration = time.time() - self._start_time
        self._logger.info(
            f"Completed: {self._stage}",
            total_items=self._total,
            duration_seconds=f"{duration:.2f}",
        )
        return duration
