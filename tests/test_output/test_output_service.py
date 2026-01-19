"""Tests for the output service."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_document_extraction.output.json_generator import (
    JsonGenerator,
    JsonOutputResult,
    ValidationResult,
)
from agentic_document_extraction.output.markdown_generator import (
    MarkdownGenerator,
    MarkdownOutputResult,
)
from agentic_document_extraction.output.output_service import (
    ExtractionMetadata,
    ExtractionOutput,
    OutputService,
)
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.schema_validator import SchemaValidator


@pytest.fixture
def simple_schema_info() -> Any:
    """Create a simple schema info for testing."""
    validator = SchemaValidator()
    return validator.validate(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
            },
            "required": ["name", "age"],
        }
    )


@pytest.fixture
def extraction_result() -> ExtractionResult:
    """Create an extraction result for testing."""
    return ExtractionResult(
        extracted_data={"name": "John Doe", "age": 30},
        field_extractions=[
            FieldExtraction(field_path="name", value="John Doe", confidence=0.95),
            FieldExtraction(field_path="age", value=30, confidence=0.9),
        ],
        model_used="gpt-4o",
        total_tokens=150,
        prompt_tokens=100,
        completion_tokens=50,
        processing_time_seconds=1.5,
    )


@pytest.fixture
def output_service() -> OutputService:
    """Create an output service instance."""
    return OutputService(
        json_generator=JsonGenerator(),
        markdown_generator=MarkdownGenerator(use_llm=False),
    )


class TestExtractionMetadata:
    """Tests for ExtractionMetadata dataclass."""

    def test_basic_metadata(self) -> None:
        """Test creating basic metadata."""
        metadata = ExtractionMetadata(
            processing_time_seconds=1.5,
            model_used="gpt-4o",
            total_tokens=150,
        )

        assert metadata.processing_time_seconds == 1.5
        assert metadata.model_used == "gpt-4o"
        assert metadata.total_tokens == 150

    def test_metadata_defaults(self) -> None:
        """Test metadata default values."""
        metadata = ExtractionMetadata()

        assert metadata.processing_time_seconds == 0.0
        assert metadata.chunks_processed == 1
        assert metadata.is_chunked is False
        assert metadata.validation_passed is True

    def test_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        metadata = ExtractionMetadata(
            processing_time_seconds=1.5,
            model_used="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            chunks_processed=2,
            is_chunked=True,
        )

        result = metadata.to_dict()

        assert result["processing_time_seconds"] == 1.5
        assert result["model_used"] == "gpt-4o"
        assert result["token_usage"]["prompt_tokens"] == 100
        assert result["token_usage"]["completion_tokens"] == 50
        assert result["is_chunked"] is True


class TestExtractionOutput:
    """Tests for ExtractionOutput dataclass."""

    def test_basic_output(self) -> None:
        """Test creating basic extraction output."""
        json_result = JsonOutputResult(
            data={"name": "John"},
            validation_result=ValidationResult(is_valid=True),
        )
        markdown_result = MarkdownOutputResult(
            markdown="# Summary",
            generated_by_llm=False,
        )
        metadata = ExtractionMetadata()

        output = ExtractionOutput(
            json_result=json_result,
            markdown_result=markdown_result,
            metadata=metadata,
        )

        assert output.json_result.data == {"name": "John"}
        assert output.markdown_result.markdown == "# Summary"

    def test_to_dict(self) -> None:
        """Test converting output to dictionary."""
        json_result = JsonOutputResult(
            data={"name": "John"},
            validation_result=ValidationResult(is_valid=True),
        )
        markdown_result = MarkdownOutputResult(
            markdown="# Summary",
            source_references=[{"field": "name", "source_text": "John"}],
        )
        metadata = ExtractionMetadata(model_used="gpt-4o")

        output = ExtractionOutput(
            json_result=json_result,
            markdown_result=markdown_result,
            metadata=metadata,
        )

        result = output.to_dict()

        assert result["data"] == {"name": "John"}
        assert result["markdown"] == "# Summary"
        assert len(result["source_references"]) == 1

    def test_to_api_response(self) -> None:
        """Test converting output to API response format."""
        json_result = JsonOutputResult(
            data={"name": "John"},
            validation_result=ValidationResult(
                is_valid=True,
                errors=[],
                missing_required_fields=[],
            ),
        )
        markdown_result = MarkdownOutputResult(markdown="# Summary")
        metadata = ExtractionMetadata(
            processing_time_seconds=1.5,
            model_used="gpt-4o",
        )

        output = ExtractionOutput(
            json_result=json_result,
            markdown_result=markdown_result,
            metadata=metadata,
        )

        response = output.to_api_response()

        assert response["extracted_data"] == {"name": "John"}
        assert response["markdown_summary"] == "# Summary"
        assert response["is_valid"] is True
        assert response["validation_errors"] == []


class TestOutputServiceInit:
    """Tests for OutputService initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        service = OutputService()

        assert service.json_generator is not None
        assert service.markdown_generator is not None
        assert service.max_retries == 3

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        json_gen = JsonGenerator(max_retries=5)
        markdown_gen = MarkdownGenerator(use_llm=False)

        service = OutputService(
            json_generator=json_gen,
            markdown_generator=markdown_gen,
            max_retries=5,
        )

        assert service.json_generator is json_gen
        assert service.markdown_generator is markdown_gen
        assert service.max_retries == 5


class TestOutputServiceGenerateOutput:
    """Tests for OutputService generate_output method."""

    def test_generate_output_valid_data(
        self,
        output_service: OutputService,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test generating output from valid extraction result."""
        output = output_service.generate_output(
            extraction_result,
            simple_schema_info,
        )

        assert output.json_result.data["name"] == "John Doe"
        assert output.json_result.validation_result.is_valid is True
        assert "# Extraction Summary" in output.markdown_result.markdown

    def test_generate_output_includes_metadata(
        self,
        output_service: OutputService,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test that generated output includes metadata."""
        output = output_service.generate_output(
            extraction_result,
            simple_schema_info,
        )

        assert output.metadata.model_used == "gpt-4o"
        assert output.metadata.total_tokens > 0

    def test_generate_output_without_markdown(
        self,
        output_service: OutputService,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test generating output without Markdown."""
        output = output_service.generate_output(
            extraction_result,
            simple_schema_info,
            include_markdown=False,
        )

        assert output.markdown_result.markdown == ""
        assert output.json_result.data is not None

    def test_generate_output_with_validation_failure(
        self,
        output_service: OutputService,
        simple_schema_info: Any,
    ) -> None:
        """Test generating output when validation fails."""
        # Create extraction with missing required field
        extraction_result = ExtractionResult(
            extracted_data={"name": "John", "age": None},  # age is null
            model_used="gpt-4o",
        )

        output = output_service.generate_output(
            extraction_result,
            simple_schema_info,
        )

        assert output.json_result.validation_result.is_valid is False
        assert output.metadata.validation_passed is False


class TestOutputServiceExtractAndGenerate:
    """Tests for OutputService extract_and_generate method."""

    def test_extract_and_generate_requires_service(
        self,
        simple_schema_info: Any,
    ) -> None:
        """Test that extract_and_generate requires extraction service."""
        service = OutputService()

        with pytest.raises(ValueError, match="Extraction service not provided"):
            service.extract_and_generate("test text", simple_schema_info)

    def test_extract_and_generate_with_mock_service(
        self,
        simple_schema_info: Any,
    ) -> None:
        """Test extract_and_generate with mocked extraction service."""
        # Create mock extraction service
        mock_extraction_service = MagicMock()
        mock_extraction_service.extract.return_value = ExtractionResult(
            extracted_data={"name": "John", "age": 30},
            model_used="gpt-4o",
            total_tokens=100,
        )

        service = OutputService(
            extraction_service=mock_extraction_service,
            markdown_generator=MarkdownGenerator(use_llm=False),
        )

        output = service.extract_and_generate(
            "John is 30 years old.",
            simple_schema_info,
            retry_on_validation_failure=False,
        )

        assert output.json_result.data["name"] == "John"
        mock_extraction_service.extract.assert_called_once()

    def test_extract_and_generate_with_retry(
        self,
        simple_schema_info: Any,
    ) -> None:
        """Test extract_and_generate with retry on validation failure."""
        # Create mock extraction service that fails first then succeeds
        mock_extraction_service = MagicMock()
        mock_extraction_service.extract.side_effect = [
            ExtractionResult(
                extracted_data={"name": "John", "age": None},  # Invalid
                model_used="gpt-4o",
            ),
            ExtractionResult(
                extracted_data={"name": "John", "age": 30},  # Valid
                model_used="gpt-4o",
            ),
        ]

        service = OutputService(
            extraction_service=mock_extraction_service,
            markdown_generator=MarkdownGenerator(use_llm=False),
            max_retries=3,
        )

        output = service.extract_and_generate(
            "John is 30 years old.",
            simple_schema_info,
            retry_on_validation_failure=True,
        )

        assert output.json_result.validation_result.is_valid is True
        assert output.metadata.retry_count == 1


class TestOutputServiceFormatForApi:
    """Tests for OutputService format_for_api method."""

    def test_format_for_api(
        self,
        output_service: OutputService,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test formatting output for API."""
        output = output_service.generate_output(
            extraction_result,
            simple_schema_info,
        )

        api_response = output_service.format_for_api(output)

        assert "extracted_data" in api_response
        assert "markdown_summary" in api_response
        assert "is_valid" in api_response
        assert "metadata" in api_response


class TestOutputServiceTokenTracking:
    """Tests for token usage tracking."""

    def test_token_tracking_extraction_only(
        self,
        output_service: OutputService,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test token tracking with extraction only."""
        output = output_service.generate_output(
            extraction_result,
            simple_schema_info,
            include_markdown=False,
        )

        assert output.metadata.prompt_tokens == 100
        assert output.metadata.completion_tokens == 50
        assert output.metadata.total_tokens == 150

    def test_token_tracking_with_markdown_llm(
        self,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test token tracking includes markdown LLM usage."""
        # Create generator with mock LLM
        markdown_gen = MarkdownGenerator(use_llm=True, api_key="test")
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "# Summary"
        mock_response.usage_metadata = {
            "input_tokens": 50,
            "output_tokens": 25,
            "total_tokens": 75,
        }
        mock_llm.invoke.return_value = mock_response
        markdown_gen._llm = mock_llm

        service = OutputService(
            markdown_generator=markdown_gen,
        )

        output = service.generate_output(
            extraction_result,
            simple_schema_info,
        )

        # Should include both extraction and markdown tokens
        assert output.metadata.prompt_tokens == 150  # 100 + 50
        assert output.metadata.completion_tokens == 75  # 50 + 25


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_complete_extraction_workflow(
        self,
        simple_schema_info: Any,
    ) -> None:
        """Test complete extraction workflow."""
        # Create mock extraction service
        mock_extraction_service = MagicMock()
        mock_extraction_service.extract.return_value = ExtractionResult(
            extracted_data={"name": "Alice Smith", "age": 25},
            field_extractions=[
                FieldExtraction(
                    field_path="name",
                    value="Alice Smith",
                    confidence=0.95,
                    source_text="My name is Alice Smith",
                ),
                FieldExtraction(
                    field_path="age",
                    value=25,
                    confidence=0.9,
                ),
            ],
            model_used="gpt-4o",
            total_tokens=200,
            prompt_tokens=150,
            completion_tokens=50,
            processing_time_seconds=2.0,
        )

        service = OutputService(
            extraction_service=mock_extraction_service,
            markdown_generator=MarkdownGenerator(use_llm=False),
        )

        output = service.extract_and_generate(
            "My name is Alice Smith and I am 25 years old.",
            simple_schema_info,
        )

        # Verify JSON output
        assert output.json_result.data["name"] == "Alice Smith"
        assert output.json_result.data["age"] == 25
        assert output.json_result.validation_result.is_valid is True

        # Verify Markdown output
        assert "Alice Smith" in output.markdown_result.markdown
        assert "Source References" in output.markdown_result.markdown

        # Verify metadata
        assert output.metadata.model_used == "gpt-4o"
        assert output.metadata.validation_passed is True
