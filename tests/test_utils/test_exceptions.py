"""Tests for the centralized exception classes."""

from agentic_document_extraction.utils.exceptions import (
    ADEError,
    ADEFileNotFoundError,
    DocumentProcessingError,
    EncodingError,
    ErrorCode,
    ExtractionError,
    FileError,
    FileTooLargeError,
    JobError,
    JobExpiredError,
    JobNotCompleteError,
    JobNotFoundError,
    LayoutDetectionError,
    LLMError,
    LLMRateLimitError,
    LLMTokenLimitError,
    OCRError,
    QualityThresholdError,
    SchemaError,
    SchemaParseError,
    SchemaValidationError,
    TextExtractionError,
    UnsupportedFormatError,
    ValidationError,
)


class TestErrorCode:
    """Tests for ErrorCode enumeration."""

    def test_error_codes_are_unique(self) -> None:
        """All error codes should have unique values."""
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values))

    def test_error_code_format(self) -> None:
        """Error codes should follow Exxxx format."""
        for code in ErrorCode:
            assert code.value.startswith("E")
            assert len(code.value) == 5
            assert code.value[1:].isdigit()

    def test_file_errors_start_with_e1(self) -> None:
        """File error codes should start with E1."""
        file_codes = [
            ErrorCode.FILE_NOT_FOUND,
            ErrorCode.FILE_TOO_LARGE,
            ErrorCode.UNSUPPORTED_FORMAT,
            ErrorCode.FILE_READ_ERROR,
            ErrorCode.FILE_WRITE_ERROR,
            ErrorCode.ENCODING_ERROR,
        ]
        for code in file_codes:
            assert code.value.startswith("E1")

    def test_schema_errors_start_with_e2(self) -> None:
        """Schema error codes should start with E2."""
        schema_codes = [
            ErrorCode.INVALID_SCHEMA,
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            ErrorCode.UNSUPPORTED_SCHEMA_TYPE,
            ErrorCode.SCHEMA_PARSE_ERROR,
        ]
        for code in schema_codes:
            assert code.value.startswith("E2")

    def test_job_errors_start_with_e3(self) -> None:
        """Job error codes should start with E3."""
        job_codes = [
            ErrorCode.JOB_NOT_FOUND,
            ErrorCode.JOB_EXPIRED,
            ErrorCode.JOB_ALREADY_EXISTS,
            ErrorCode.JOB_PROCESSING_FAILED,
            ErrorCode.JOB_NOT_COMPLETE,
        ]
        for code in job_codes:
            assert code.value.startswith("E3")

    def test_extraction_errors_start_with_e4(self) -> None:
        """Extraction error codes should start with E4."""
        extraction_codes = [
            ErrorCode.EXTRACTION_FAILED,
            ErrorCode.TEXT_EXTRACTION_FAILED,
            ErrorCode.OCR_FAILED,
            ErrorCode.LAYOUT_DETECTION_FAILED,
            ErrorCode.READING_ORDER_FAILED,
            ErrorCode.VISUAL_EXTRACTION_FAILED,
            ErrorCode.SYNTHESIS_FAILED,
            ErrorCode.QUALITY_THRESHOLD_NOT_MET,
        ]
        for code in extraction_codes:
            assert code.value.startswith("E4")


class TestADEError:
    """Tests for base ADEError class."""

    def test_basic_initialization(self) -> None:
        """Test basic error initialization."""
        error = ADEError("Test error message")
        assert str(error) == "[E9001] Test error message"
        assert error.message == "Test error message"
        assert error.error_code == ErrorCode.INTERNAL_ERROR
        assert error.details == {}
        assert error.http_status == 500

    def test_with_custom_error_code(self) -> None:
        """Test initialization with custom error code."""
        error = ADEError("Custom error", error_code=ErrorCode.FILE_NOT_FOUND)
        assert error.error_code == ErrorCode.FILE_NOT_FOUND
        assert str(error) == "[E1001] Custom error"

    def test_with_details(self) -> None:
        """Test initialization with details."""
        details = {"field": "value", "count": 42}
        error = ADEError("Error with details", details=details)
        assert error.details == details

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        error = ADEError(
            "Test error",
            error_code=ErrorCode.FILE_NOT_FOUND,
            details={"path": "/test/file"},
        )
        result = error.to_dict()
        assert result["error_code"] == "E1001"
        assert result["message"] == "Test error"
        assert result["details"]["path"] == "/test/file"

    def test_to_dict_without_details(self) -> None:
        """Test to_dict without details."""
        error = ADEError("Test error")
        result = error.to_dict()
        assert "details" not in result

    def test_get_http_status(self) -> None:
        """Test get_http_status method."""
        error = ADEError("Test")
        assert error.get_http_status() == 500


class TestFileErrors:
    """Tests for file-related exceptions."""

    def test_ade_file_not_found_error(self) -> None:
        """Test ADEFileNotFoundError."""
        error = ADEFileNotFoundError("/path/to/file.txt")
        assert error.file_path == "/path/to/file.txt"
        assert error.error_code == ErrorCode.FILE_NOT_FOUND
        assert error.http_status == 404
        assert "file.txt" in str(error)
        assert error.details["file_path"] == "/path/to/file.txt"

    def test_ade_file_not_found_error_custom_message(self) -> None:
        """Test ADEFileNotFoundError with custom message."""
        error = ADEFileNotFoundError("/path/to/file.txt", message="Custom not found")
        assert "Custom not found" in str(error)

    def test_file_too_large_error(self) -> None:
        """Test FileTooLargeError."""
        error = FileTooLargeError(file_size=20_000_000, max_size=10_000_000)
        assert error.file_size == 20_000_000
        assert error.max_size == 10_000_000
        assert error.error_code == ErrorCode.FILE_TOO_LARGE
        assert error.http_status == 413
        assert error.details["file_size_bytes"] == 20_000_000
        assert error.details["max_size_bytes"] == 10_000_000

    def test_file_too_large_error_with_path(self) -> None:
        """Test FileTooLargeError with file path."""
        error = FileTooLargeError(
            file_size=20_000_000,
            max_size=10_000_000,
            file_path="/path/to/large.pdf",
        )
        assert error.details["file_path"] == "/path/to/large.pdf"

    def test_unsupported_format_error(self) -> None:
        """Test UnsupportedFormatError."""
        error = UnsupportedFormatError(
            "Unsupported format",
            detected_mime="application/x-unknown",
        )
        assert error.detected_mime == "application/x-unknown"
        assert error.error_code == ErrorCode.UNSUPPORTED_FORMAT
        assert error.http_status == 400
        assert error.details["detected_mime_type"] == "application/x-unknown"

    def test_encoding_error(self) -> None:
        """Test EncodingError."""
        error = EncodingError(
            "Cannot decode file",
            encoding="utf-16",
            file_path="/path/to/file.txt",
        )
        assert error.encoding == "utf-16"
        assert error.error_code == ErrorCode.ENCODING_ERROR
        assert error.details["encoding"] == "utf-16"


class TestSchemaErrors:
    """Tests for schema-related exceptions."""

    def test_schema_validation_error(self) -> None:
        """Test SchemaValidationError."""
        errors = ["Missing type", "Invalid property"]
        error = SchemaValidationError("Schema invalid", errors=errors)
        assert error.errors == errors
        assert error.error_code == ErrorCode.SCHEMA_VALIDATION_FAILED
        assert error.http_status == 400
        assert error.details["validation_errors"] == errors

    def test_schema_validation_error_without_errors(self) -> None:
        """Test SchemaValidationError without error list."""
        error = SchemaValidationError("Schema invalid")
        assert error.errors == []

    def test_schema_parse_error(self) -> None:
        """Test SchemaParseError."""
        error = SchemaParseError("Invalid JSON syntax")
        assert error.error_code == ErrorCode.SCHEMA_PARSE_ERROR
        assert error.http_status == 400


class TestJobErrors:
    """Tests for job-related exceptions."""

    def test_job_not_found_error(self) -> None:
        """Test JobNotFoundError."""
        error = JobNotFoundError("job-123")
        assert error.job_id == "job-123"
        assert error.error_code == ErrorCode.JOB_NOT_FOUND
        assert error.http_status == 404
        assert error.details["job_id"] == "job-123"
        assert "job-123" in str(error)

    def test_job_expired_error(self) -> None:
        """Test JobExpiredError."""
        error = JobExpiredError("job-456", ttl_hours=24)
        assert error.job_id == "job-456"
        assert error.error_code == ErrorCode.JOB_EXPIRED
        assert error.http_status == 410
        assert error.details["ttl_hours"] == 24

    def test_job_not_complete_error(self) -> None:
        """Test JobNotCompleteError."""
        error = JobNotCompleteError(
            "job-789",
            status="processing",
            progress="Extracting text",
        )
        assert error.error_code == ErrorCode.JOB_NOT_COMPLETE
        assert error.http_status == 425
        assert error.details["status"] == "processing"
        assert error.details["progress"] == "Extracting text"


class TestExtractionErrors:
    """Tests for extraction-related exceptions."""

    def test_extraction_error(self) -> None:
        """Test base ExtractionError."""
        error = ExtractionError("Extraction failed", stage="parsing")
        assert error.stage == "parsing"
        assert error.error_code == ErrorCode.EXTRACTION_FAILED
        assert error.http_status == 500
        assert error.details["extraction_stage"] == "parsing"

    def test_text_extraction_error(self) -> None:
        """Test TextExtractionError."""
        error = TextExtractionError("Cannot extract text", file_type="pdf")
        assert error.error_code == ErrorCode.TEXT_EXTRACTION_FAILED
        assert error.stage == "text_extraction"
        assert error.details["file_type"] == "pdf"

    def test_ocr_error(self) -> None:
        """Test OCRError."""
        error = OCRError("OCR failed", page_number=5)
        assert error.error_code == ErrorCode.OCR_FAILED
        assert error.stage == "ocr"
        assert error.details["page_number"] == 5

    def test_layout_detection_error(self) -> None:
        """Test LayoutDetectionError."""
        error = LayoutDetectionError("Layout detection failed", page_number=2)
        assert error.error_code == ErrorCode.LAYOUT_DETECTION_FAILED
        assert error.stage == "layout_detection"
        assert error.details["page_number"] == 2

    def test_quality_threshold_error(self) -> None:
        """Test QualityThresholdError."""
        error = QualityThresholdError(
            "Quality threshold not met",
            achieved_confidence=0.65,
            required_confidence=0.80,
        )
        assert error.error_code == ErrorCode.QUALITY_THRESHOLD_NOT_MET
        assert error.stage == "quality_verification"
        assert error.details["achieved_confidence"] == 0.65
        assert error.details["required_confidence"] == 0.80


class TestLLMErrors:
    """Tests for LLM-related exceptions."""

    def test_llm_error(self) -> None:
        """Test base LLMError."""
        error = LLMError("API error", model="gpt-4o")
        assert error.model == "gpt-4o"
        assert error.error_code == ErrorCode.LLM_API_ERROR
        assert error.http_status == 502
        assert error.details["model"] == "gpt-4o"

    def test_llm_rate_limit_error(self) -> None:
        """Test LLMRateLimitError."""
        error = LLMRateLimitError(retry_after=60, model="gpt-4o")
        assert error.error_code == ErrorCode.LLM_RATE_LIMIT
        assert error.http_status == 429
        assert error.details["retry_after_seconds"] == 60

    def test_llm_token_limit_error(self) -> None:
        """Test LLMTokenLimitError."""
        error = LLMTokenLimitError(
            token_count=150000,
            token_limit=128000,
            model="gpt-4o",
        )
        assert error.error_code == ErrorCode.LLM_TOKEN_LIMIT
        assert error.details["token_count"] == 150000
        assert error.details["token_limit"] == 128000


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error(self) -> None:
        """Test ValidationError."""
        error = ValidationError(
            "Validation failed",
            field="schema",
            errors=["Missing required field"],
        )
        assert error.error_code == ErrorCode.SCHEMA_VALIDATION_FAILED
        assert error.http_status == 400
        assert error.details["field"] == "schema"
        assert error.details["validation_errors"] == ["Missing required field"]


class TestDocumentProcessingError:
    """Tests for DocumentProcessingError."""

    def test_document_processing_error(self) -> None:
        """Test DocumentProcessingError."""
        error = DocumentProcessingError(
            "Processing failed",
            error_type="conversion_error",
            details={"page": 5},
        )
        assert error.error_code == ErrorCode.INTERNAL_ERROR
        assert error.http_status == 500
        assert error.details["error_type"] == "conversion_error"
        assert error.details["page"] == 5


class TestHTTPStatusMapping:
    """Tests for HTTP status code mapping."""

    def test_http_status_codes(self) -> None:
        """Test that all error types have appropriate HTTP status codes."""
        # 400 Bad Request
        assert ADEError("test").http_status == 500  # Base is internal error
        assert FileError("test").http_status == 400
        assert SchemaError("test").http_status == 400
        assert ValidationError("test").http_status == 400

        # 404 Not Found
        assert ADEFileNotFoundError("/test").http_status == 404
        assert JobNotFoundError("job").http_status == 404

        # 410 Gone
        assert JobExpiredError("job").http_status == 410

        # 413 Payload Too Large
        assert FileTooLargeError(10, 5).http_status == 413

        # 425 Too Early
        assert JobNotCompleteError("job", "pending").http_status == 425

        # 429 Too Many Requests
        assert LLMRateLimitError().http_status == 429

        # 500 Internal Server Error
        assert ExtractionError("test").http_status == 500
        assert DocumentProcessingError("test").http_status == 500

        # 502 Bad Gateway
        assert LLMError("test").http_status == 502


class TestExceptionInheritance:
    """Tests for exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_ade_error(self) -> None:
        """All custom exceptions should inherit from ADEError."""
        exceptions = [
            FileError,
            ADEFileNotFoundError,
            FileTooLargeError,
            UnsupportedFormatError,
            EncodingError,
            SchemaError,
            SchemaValidationError,
            SchemaParseError,
            JobError,
            JobNotFoundError,
            JobExpiredError,
            JobNotCompleteError,
            ExtractionError,
            TextExtractionError,
            OCRError,
            LayoutDetectionError,
            QualityThresholdError,
            DocumentProcessingError,
            ValidationError,
            LLMError,
            LLMRateLimitError,
            LLMTokenLimitError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, ADEError)

    def test_file_errors_inherit_from_file_error(self) -> None:
        """File-related errors should inherit from FileError."""
        file_exceptions = [
            ADEFileNotFoundError,
            FileTooLargeError,
            UnsupportedFormatError,
            EncodingError,
        ]
        for exc_class in file_exceptions:
            assert issubclass(exc_class, FileError)

    def test_job_errors_inherit_from_job_error(self) -> None:
        """Job-related errors should inherit from JobError."""
        job_exceptions = [
            JobNotFoundError,
            JobExpiredError,
            JobNotCompleteError,
        ]
        for exc_class in job_exceptions:
            assert issubclass(exc_class, JobError)

    def test_extraction_errors_inherit_from_extraction_error(self) -> None:
        """Extraction-related errors should inherit from ExtractionError."""
        extraction_exceptions = [
            TextExtractionError,
            OCRError,
            LayoutDetectionError,
            QualityThresholdError,
        ]
        for exc_class in extraction_exceptions:
            assert issubclass(exc_class, ExtractionError)

    def test_llm_errors_inherit_from_llm_error(self) -> None:
        """LLM-related errors should inherit from LLMError."""
        llm_exceptions = [
            LLMRateLimitError,
            LLMTokenLimitError,
        ]
        for exc_class in llm_exceptions:
            assert issubclass(exc_class, LLMError)
