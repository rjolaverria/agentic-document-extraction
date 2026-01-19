"""Tests for the LangChain LLM text extraction service."""

import json
import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionError,
    ExtractionResult,
    FieldExtraction,
    TextExtractionService,
)
from agentic_document_extraction.services.schema_validator import (
    SchemaInfo,
    SchemaValidator,
)

# Check if OpenAI API key is available for integration tests
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
    "ADE_OPENAI_API_KEY", ""
)
SKIP_INTEGRATION = not OPENAI_API_KEY
INTEGRATION_SKIP_REASON = (
    "OpenAI API key not set (set OPENAI_API_KEY or ADE_OPENAI_API_KEY)"
)


@pytest.fixture
def simple_schema() -> dict[str, Any]:
    """Simple schema for testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's full name"},
            "age": {"type": "integer", "description": "Person's age"},
            "email": {"type": "string", "description": "Email address"},
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def simple_schema_info(simple_schema: dict[str, Any]) -> SchemaInfo:
    """Validated simple schema info."""
    validator = SchemaValidator()
    return validator.validate(simple_schema)


@pytest.fixture
def nested_schema() -> dict[str, Any]:
    """Nested schema for testing."""
    return {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                    },
                },
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["person"],
    }


@pytest.fixture
def nested_schema_info(nested_schema: dict[str, Any]) -> SchemaInfo:
    """Validated nested schema info."""
    validator = SchemaValidator()
    return validator.validate(nested_schema)


@pytest.fixture
def mock_llm_response() -> MagicMock:
    """Create a mock LLM response."""
    response = MagicMock()
    response.content = json.dumps(
        {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
        }
    )
    response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    return response


class TestFieldExtraction:
    """Tests for FieldExtraction dataclass."""

    def test_basic_field_extraction(self) -> None:
        """Test creating a basic field extraction."""
        extraction = FieldExtraction(
            field_path="name",
            value="John Doe",
            confidence=0.95,
        )

        assert extraction.field_path == "name"
        assert extraction.value == "John Doe"
        assert extraction.confidence == 0.95
        assert extraction.source_text is None
        assert extraction.reasoning is None

    def test_field_extraction_with_all_fields(self) -> None:
        """Test field extraction with all optional fields."""
        extraction = FieldExtraction(
            field_path="address.city",
            value="New York",
            confidence=0.8,
            source_text="located in New York City",
            reasoning="Extracted city from location mention",
        )

        assert extraction.field_path == "address.city"
        assert extraction.value == "New York"
        assert extraction.confidence == 0.8
        assert extraction.source_text == "located in New York City"
        assert extraction.reasoning == "Extracted city from location mention"


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_basic_extraction_result(self) -> None:
        """Test creating a basic extraction result."""
        result = ExtractionResult(
            extracted_data={"name": "John", "age": 30},
            model_used="gpt-4",
            total_tokens=150,
        )

        assert result.extracted_data == {"name": "John", "age": 30}
        assert result.model_used == "gpt-4"
        assert result.total_tokens == 150
        assert result.field_extractions == []
        assert result.chunks_processed == 1
        assert result.is_chunked is False

    def test_extraction_result_to_dict(self) -> None:
        """Test converting extraction result to dictionary."""
        result = ExtractionResult(
            extracted_data={"name": "John"},
            field_extractions=[
                FieldExtraction(field_path="name", value="John", confidence=0.9)
            ],
            model_used="gpt-4",
            total_tokens=100,
            prompt_tokens=60,
            completion_tokens=40,
            processing_time_seconds=1.5,
            chunks_processed=2,
            is_chunked=True,
        )

        result_dict = result.to_dict()

        assert result_dict["extracted_data"] == {"name": "John"}
        assert len(result_dict["field_extractions"]) == 1
        assert result_dict["field_extractions"][0]["field_path"] == "name"
        assert result_dict["metadata"]["model_used"] == "gpt-4"
        assert result_dict["metadata"]["total_tokens"] == 100
        assert result_dict["metadata"]["is_chunked"] is True


class TestTextExtractionServiceInit:
    """Tests for TextExtractionService initialization."""

    def test_default_initialization(self) -> None:
        """Test service initialization with defaults."""
        service = TextExtractionService()

        assert service.api_key == ""  # From settings default
        assert service.model == "gpt-4o"  # From settings default
        assert service.temperature == 0.0
        assert service._llm is None
        assert service._text_splitter is None

    def test_custom_initialization(self) -> None:
        """Test service initialization with custom values."""
        service = TextExtractionService(
            api_key="test-key",
            model="gpt-4-turbo",
            temperature=0.5,
            max_tokens=2000,
            chunk_size=2000,
            chunk_overlap=100,
        )

        assert service.api_key == "test-key"
        assert service.model == "gpt-4-turbo"
        assert service.temperature == 0.5
        assert service.max_tokens == 2000
        assert service.chunk_size == 2000
        assert service.chunk_overlap == 100


class TestTextExtractionServiceLLM:
    """Tests for LLM property and configuration."""

    def test_llm_raises_error_without_api_key(self) -> None:
        """Test that accessing llm without API key raises error."""
        service = TextExtractionService(api_key="")

        with pytest.raises(ExtractionError) as exc_info:
            _ = service.llm

        assert exc_info.value.error_type == "configuration_error"
        assert "API key" in str(exc_info.value)

    def test_llm_created_with_api_key(self) -> None:
        """Test that LLM is created when API key is provided."""
        service = TextExtractionService(api_key="test-key")

        # The LLM should be created lazily
        llm = service.llm

        assert llm is not None
        assert service._llm is llm  # Same instance returned


class TestTextExtractionServiceTextSplitter:
    """Tests for text splitter functionality."""

    def test_text_splitter_created_lazily(self) -> None:
        """Test that text splitter is created lazily."""
        service = TextExtractionService()

        assert service._text_splitter is None

        splitter = service.text_splitter

        assert splitter is not None
        assert service._text_splitter is splitter

    def test_text_splitter_uses_config(self) -> None:
        """Test that text splitter uses configured values."""
        service = TextExtractionService(chunk_size=1000, chunk_overlap=50)

        splitter = service.text_splitter

        assert splitter._chunk_size == 1000
        assert splitter._chunk_overlap == 50


class TestTextExtractionServiceNeedsChunking:
    """Tests for chunking decision logic."""

    def test_short_text_no_chunking(self) -> None:
        """Test that short text doesn't need chunking."""
        service = TextExtractionService()

        # Short text (less than half of max context)
        short_text = "This is a short text." * 100  # ~2100 chars

        assert service._needs_chunking(short_text) is False

    def test_long_text_needs_chunking(self) -> None:
        """Test that long text needs chunking."""
        service = TextExtractionService()

        # Long text (exceeds threshold)
        long_text = "This is a long text with many words. " * 5000  # ~185000 chars

        assert service._needs_chunking(long_text) is True


class TestTextExtractionServiceExtract:
    """Tests for the main extract method with mocked LLM."""

    def test_extract_simple_text(
        self,
        simple_schema_info: SchemaInfo,
        mock_llm_response: MagicMock,
    ) -> None:
        """Test extracting from simple text."""
        service = TextExtractionService(api_key="test-key")

        # Create mock LLM and set it directly
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        service._llm = mock_llm

        text = (
            "My name is John Doe and I am 30 years old. Contact me at john@example.com."
        )

        result = service.extract(text, simple_schema_info)

        assert result.extracted_data["name"] == "John Doe"
        assert result.extracted_data["age"] == 30
        assert result.extracted_data["email"] == "john@example.com"
        assert result.model_used == "gpt-4o"
        assert result.is_chunked is False
        assert result.chunks_processed == 1

    def test_extract_with_missing_optional_field(
        self,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test extraction with missing optional field."""
        service = TextExtractionService(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "name": "Jane Doe",
                "age": 25,
                "email": None,  # Optional field is null
            }
        )
        mock_response.usage_metadata = {
            "input_tokens": 80,
            "output_tokens": 40,
            "total_tokens": 120,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        text = "Jane Doe is 25 years old."

        result = service.extract(text, simple_schema_info)

        assert result.extracted_data["name"] == "Jane Doe"
        assert result.extracted_data["age"] == 25
        assert result.extracted_data["email"] is None

    def test_extract_records_token_usage(
        self,
        simple_schema_info: SchemaInfo,
        mock_llm_response: MagicMock,
    ) -> None:
        """Test that token usage is recorded."""
        service = TextExtractionService(api_key="test-key")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        service._llm = mock_llm

        result = service.extract("Test text", simple_schema_info)

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150


class TestTextExtractionServiceChunking:
    """Tests for chunked extraction."""

    def test_extract_with_chunking(
        self,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test extraction with chunking for large text."""
        service = TextExtractionService(
            api_key="test-key", chunk_size=500, chunk_overlap=50
        )

        # Create responses for chunks
        chunk_responses = [
            MagicMock(
                content=json.dumps({"name": "John Doe", "age": 30, "email": None}),
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 25,
                    "total_tokens": 75,
                },
            ),
            MagicMock(
                content=json.dumps(
                    {"name": None, "age": None, "email": "john@example.com"}
                ),
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 25,
                    "total_tokens": 75,
                },
            ),
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = chunk_responses
        service._llm = mock_llm

        # Create long text that will be chunked (needs to exceed 16000 chars to trigger)
        # The _needs_chunking threshold is estimated_tokens > 4000, which is ~16000 chars
        long_text = (
            "My name is John Doe and I am 30 years old. " * 500
            + "Contact me at john@example.com. " * 500
        )

        result = service.extract(long_text, simple_schema_info)

        # Results should be merged
        assert result.extracted_data["name"] == "John Doe"
        assert result.extracted_data["age"] == 30
        assert result.extracted_data["email"] == "john@example.com"
        assert result.is_chunked is True
        assert result.chunks_processed >= 1


class TestTextExtractionServiceMerging:
    """Tests for chunk result merging."""

    def test_merge_simple_values(self) -> None:
        """Test merging simple values from chunks."""
        service = TextExtractionService()

        chunk_results = [
            {"name": "John", "age": None},
            {"name": None, "age": 30},
        ]

        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            }
        )

        merged = service._merge_chunk_results(chunk_results, schema_info)

        assert merged["name"] == "John"
        assert merged["age"] == 30

    def test_merge_arrays(self) -> None:
        """Test merging array values from chunks."""
        service = TextExtractionService()

        chunk_results = [
            {"tags": ["python", "langchain"]},
            {"tags": ["openai", "langchain"]},  # langchain is duplicate
        ]

        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            }
        )

        merged = service._merge_chunk_results(chunk_results, schema_info)

        # Should have all unique tags
        assert "python" in merged["tags"]
        assert "openai" in merged["tags"]
        assert "langchain" in merged["tags"]
        # Should deduplicate
        assert merged["tags"].count("langchain") == 1

    def test_merge_nested_objects(self) -> None:
        """Test merging nested object values."""
        service = TextExtractionService()

        chunk_results = [
            {"person": {"name": "John", "city": None}},
            {"person": {"name": None, "city": "NYC"}},
        ]

        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "person": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "city": {"type": "string"},
                        },
                    },
                },
            }
        )

        merged = service._merge_chunk_results(chunk_results, schema_info)

        assert merged["person"]["name"] == "John"
        assert merged["person"]["city"] == "NYC"


class TestTextExtractionServiceJsonParsing:
    """Tests for JSON response parsing."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        service = TextExtractionService()

        validator = SchemaValidator()
        schema_info = validator.validate({"type": "object", "properties": {}})

        response = '{"name": "John", "age": 30}'
        result = service._parse_json_response(response, schema_info)

        assert result == {"name": "John", "age": 30}

    def test_parse_json_with_extra_text(self) -> None:
        """Test parsing JSON that has extra text around it."""
        service = TextExtractionService()

        validator = SchemaValidator()
        schema_info = validator.validate({"type": "object", "properties": {}})

        response = 'Here is the result: {"name": "John"} and some more text'
        result = service._parse_json_response(response, schema_info)

        assert result == {"name": "John"}

    def test_parse_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON raises ExtractionError."""
        service = TextExtractionService()

        validator = SchemaValidator()
        schema_info = validator.validate({"type": "object", "properties": {}})

        response = "This is not JSON at all"

        with pytest.raises(ExtractionError) as exc_info:
            service._parse_json_response(response, schema_info)

        assert exc_info.value.error_type == "parse_error"


class TestTextExtractionServiceFieldExtractions:
    """Tests for building field extractions."""

    def test_build_field_extractions(self) -> None:
        """Test building field extraction objects."""
        service = TextExtractionService()

        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name"],
            }
        )

        data = {"name": "John", "age": 30}

        extractions = service._build_field_extractions(data, schema_info)

        assert len(extractions) >= 2

        name_extraction = next(e for e in extractions if e.field_path == "name")
        assert name_extraction.value == "John"

        age_extraction = next(e for e in extractions if e.field_path == "age")
        assert age_extraction.value == 30


class TestTextExtractionServiceNestedValues:
    """Tests for getting nested values."""

    def test_get_simple_value(self) -> None:
        """Test getting a simple top-level value."""
        service = TextExtractionService()

        data = {"name": "John", "age": 30}
        value = service._get_nested_value(data, "name")

        assert value == "John"

    def test_get_nested_value(self) -> None:
        """Test getting a nested value."""
        service = TextExtractionService()

        data = {"person": {"address": {"city": "NYC"}}}
        value = service._get_nested_value(data, "person.address.city")

        assert value == "NYC"

    def test_get_missing_value_returns_none(self) -> None:
        """Test that missing values return None."""
        service = TextExtractionService()

        data = {"name": "John"}
        value = service._get_nested_value(data, "address.city")

        assert value is None


class TestTextExtractionServiceExtractFromDocuments:
    """Tests for extracting from LangChain documents."""

    def test_extract_from_documents(
        self,
        simple_schema_info: SchemaInfo,
        mock_llm_response: MagicMock,
    ) -> None:
        """Test extracting from LangChain documents."""
        service = TextExtractionService(api_key="test-key")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response
        service._llm = mock_llm

        documents = [
            Document(page_content="My name is John Doe.", metadata={"source": "doc1"}),
            Document(page_content="I am 30 years old.", metadata={"source": "doc2"}),
        ]

        result = service.extract_from_documents(documents, simple_schema_info)

        assert result.extracted_data["name"] == "John Doe"
        assert result.extracted_data["age"] == 30


class TestExtractionError:
    """Tests for ExtractionError class."""

    def test_basic_error(self) -> None:
        """Test creating a basic extraction error."""
        error = ExtractionError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.error_type == "extraction_error"
        assert error.details == {}

    def test_error_with_details(self) -> None:
        """Test creating error with details."""
        error = ExtractionError(
            "API error",
            error_type="api_error",
            details={"status_code": 500, "reason": "Server error"},
        )

        assert error.error_type == "api_error"
        assert error.details["status_code"] == 500
        assert error.details["reason"] == "Server error"


class TestIntegrationScenarios:
    """Integration tests for realistic extraction scenarios."""

    def test_extract_invoice_data(self) -> None:
        """Test extracting invoice-like data."""
        service = TextExtractionService(api_key="test-key")

        invoice_schema = {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "date": {"type": "string"},
                "total_amount": {"type": "number"},
                "vendor": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "price": {"type": "number"},
                        },
                    },
                },
            },
            "required": ["invoice_number", "total_amount"],
        }

        validator = SchemaValidator()
        schema_info = validator.validate(invoice_schema)

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "invoice_number": "INV-2024-001",
                "date": "2024-01-15",
                "total_amount": 1500.00,
                "vendor": "Acme Corp",
                "items": [
                    {"description": "Widget A", "quantity": 10, "price": 100.00},
                    {"description": "Widget B", "quantity": 5, "price": 100.00},
                ],
            }
        )
        mock_response.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        text = """
        INVOICE
        Invoice Number: INV-2024-001
        Date: January 15, 2024

        Vendor: Acme Corp

        Items:
        - Widget A x 10 @ $100.00 each
        - Widget B x 5 @ $100.00 each

        Total: $1,500.00
        """

        result = service.extract(text, schema_info)

        assert result.extracted_data["invoice_number"] == "INV-2024-001"
        assert result.extracted_data["total_amount"] == 1500.00
        assert len(result.extracted_data["items"]) == 2

    def test_extract_contact_info(self) -> None:
        """Test extracting contact information."""
        service = TextExtractionService(api_key="test-key")

        contact_schema = {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "company": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["full_name", "email"],
        }

        validator = SchemaValidator()
        schema_info = validator.validate(contact_schema)

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "full_name": "Alice Smith",
                "email": "alice.smith@techcorp.com",
                "phone": "+1-555-123-4567",
                "company": "TechCorp Inc.",
                "title": "Senior Engineer",
            }
        )
        mock_response.usage_metadata = {
            "input_tokens": 150,
            "output_tokens": 80,
            "total_tokens": 230,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        text = """
        Contact Card:
        Alice Smith
        Senior Engineer at TechCorp Inc.
        Email: alice.smith@techcorp.com
        Phone: +1-555-123-4567
        """

        result = service.extract(text, schema_info)

        assert result.extracted_data["full_name"] == "Alice Smith"
        assert result.extracted_data["email"] == "alice.smith@techcorp.com"
        assert result.extracted_data["company"] == "TechCorp Inc."


# Integration tests with real OpenAI API
# These tests are skipped if OPENAI_API_KEY or ADE_OPENAI_API_KEY is not set
@pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
class TestRealAPIIntegration:
    """Integration tests using the real OpenAI API.

    These tests verify end-to-end functionality with actual API calls.
    They are skipped if no API key is available.

    To run these tests, set the OPENAI_API_KEY or ADE_OPENAI_API_KEY
    environment variable before running pytest.
    """

    @pytest.fixture
    def service(self) -> TextExtractionService:
        """Create a service instance with real API key."""
        return TextExtractionService(api_key=OPENAI_API_KEY)

    def test_extract_simple_contact_info(self, service: TextExtractionService) -> None:
        """Test extracting simple contact information with real API."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's full name"},
                "email": {"type": "string", "description": "Email address"},
                "phone": {"type": "string", "description": "Phone number"},
            },
            "required": ["name", "email"],
        }

        validator = SchemaValidator()
        schema_info = validator.validate(schema)

        text = """
        Contact Information:
        Name: John Smith
        Email: john.smith@example.com
        Phone: (555) 123-4567
        """

        result = service.extract(text, schema_info)

        assert result.extracted_data is not None
        assert "name" in result.extracted_data
        assert "email" in result.extracted_data
        assert result.model_used == service.model
        assert result.total_tokens > 0
        assert result.processing_time_seconds > 0

    def test_extract_nested_data(self, service: TextExtractionService) -> None:
        """Test extracting nested data structures with real API."""
        schema = {
            "type": "object",
            "properties": {
                "company": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "country": {"type": "string"},
                            },
                        },
                    },
                },
                "employees": {"type": "integer", "description": "Number of employees"},
            },
            "required": ["company"],
        }

        validator = SchemaValidator()
        schema_info = validator.validate(schema)

        text = """
        Company Profile:
        Acme Corporation is headquartered at 123 Main Street, New York, USA.
        The company employs approximately 500 people worldwide.
        """

        result = service.extract(text, schema_info)

        assert result.extracted_data is not None
        assert "company" in result.extracted_data
        assert isinstance(result.extracted_data["company"], dict)
        assert result.is_chunked is False

    def test_extract_array_data(self, service: TextExtractionService) -> None:
        """Test extracting array data with real API."""
        schema = {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                        },
                    },
                },
                "total": {"type": "number"},
            },
            "required": ["products"],
        }

        validator = SchemaValidator()
        schema_info = validator.validate(schema)

        text = """
        Order Receipt:
        - Widget A: $19.99
        - Widget B: $29.99
        - Widget C: $9.99
        Total: $59.97
        """

        result = service.extract(text, schema_info)

        assert result.extracted_data is not None
        assert "products" in result.extracted_data
        assert isinstance(result.extracted_data["products"], list)
        assert len(result.extracted_data["products"]) >= 1

    def test_extraction_with_missing_optional_fields(
        self, service: TextExtractionService
    ) -> None:
        """Test that optional fields can be null when not in text."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "isbn": {"type": "string", "description": "ISBN number"},
                "publisher": {"type": "string"},
            },
            "required": ["title", "author"],
        }

        validator = SchemaValidator()
        schema_info = validator.validate(schema)

        text = """
        Book: The Great Adventure
        Written by Jane Doe
        """

        result = service.extract(text, schema_info)

        assert result.extracted_data is not None
        assert "title" in result.extracted_data
        assert "author" in result.extracted_data
        # ISBN and publisher may be null since not in text

    def test_token_usage_tracking(self, service: TextExtractionService) -> None:
        """Test that token usage is properly tracked."""
        schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        }

        validator = SchemaValidator()
        schema_info = validator.validate(schema)

        text = "Hello, World!"

        result = service.extract(text, schema_info)

        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.total_tokens == result.prompt_tokens + result.completion_tokens

    def test_field_extractions_populated(self, service: TextExtractionService) -> None:
        """Test that field extractions are properly populated."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

        validator = SchemaValidator()
        schema_info = validator.validate(schema)

        text = "John Doe is 30 years old."

        result = service.extract(text, schema_info)

        assert len(result.field_extractions) > 0

        name_extraction = next(
            (e for e in result.field_extractions if e.field_path == "name"), None
        )
        assert name_extraction is not None
        assert name_extraction.value is not None
