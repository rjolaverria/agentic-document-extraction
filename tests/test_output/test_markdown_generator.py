"""Tests for the Markdown output generator."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_document_extraction.output.markdown_generator import (
    MarkdownGenerator,
    MarkdownOutputResult,
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
                "email": {"type": "string", "description": "Email address"},
            },
            "required": ["name", "age"],
        }
    )


@pytest.fixture
def extraction_result() -> ExtractionResult:
    """Create an extraction result for testing."""
    return ExtractionResult(
        extracted_data={
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
        },
        field_extractions=[
            FieldExtraction(
                field_path="name",
                value="John Doe",
                confidence=0.95,
                source_text="My name is John Doe",
            ),
            FieldExtraction(
                field_path="age",
                value=30,
                confidence=0.9,
                source_text="I am 30 years old",
            ),
            FieldExtraction(
                field_path="email",
                value="john@example.com",
                confidence=0.85,
            ),
        ],
        model_used="gpt-4o",
        total_tokens=150,
        prompt_tokens=100,
        completion_tokens=50,
        processing_time_seconds=1.5,
    )


class TestMarkdownOutputResult:
    """Tests for MarkdownOutputResult dataclass."""

    def test_basic_result(self) -> None:
        """Test creating a basic Markdown output result."""
        result = MarkdownOutputResult(
            markdown="# Test\n\nContent",
            generated_by_llm=False,
        )

        assert result.markdown == "# Test\n\nContent"
        assert result.generated_by_llm is False
        assert result.source_references == []
        assert result.token_usage == {}

    def test_result_with_all_fields(self) -> None:
        """Test creating result with all fields."""
        result = MarkdownOutputResult(
            markdown="# Test",
            source_references=[{"field": "name", "source_text": "John"}],
            generated_by_llm=True,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        assert result.generated_by_llm is True
        assert len(result.source_references) == 1
        assert result.token_usage["prompt_tokens"] == 100

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = MarkdownOutputResult(
            markdown="# Test",
            source_references=[{"field": "name", "source_text": "John"}],
            generated_by_llm=True,
            token_usage={"prompt_tokens": 100},
        )

        result_dict = result.to_dict()

        assert result_dict["markdown"] == "# Test"
        assert result_dict["generated_by_llm"] is True
        assert len(result_dict["source_references"]) == 1


class TestMarkdownGeneratorInit:
    """Tests for MarkdownGenerator initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        generator = MarkdownGenerator(use_llm=False)

        assert generator.use_llm is False
        assert generator._llm is None

    def test_initialization_with_api_key(self) -> None:
        """Test initialization with API key."""
        generator = MarkdownGenerator(
            use_llm=True,
            api_key="test-key",
            model="gpt-4",
        )

        assert generator.use_llm is True
        assert generator.api_key == "test-key"
        assert generator.model == "gpt-4"

    def test_llm_property_without_key(self) -> None:
        """Test that LLM property returns None without API key."""
        generator = MarkdownGenerator(use_llm=True, api_key="")

        assert generator.llm is None


class TestMarkdownGeneratorTemplateGeneration:
    """Tests for template-based Markdown generation."""

    def test_generate_template_markdown(
        self,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test generating Markdown using templates."""
        generator = MarkdownGenerator(use_llm=False)

        result = generator.generate(
            extraction_result,
            simple_schema_info,
            include_source_refs=True,
            include_confidence=True,
        )

        assert "# Extraction Summary" in result.markdown
        assert "John Doe" in result.markdown
        assert "30" in result.markdown
        assert result.generated_by_llm is False

    def test_generate_includes_metadata(
        self,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test that generated Markdown includes metadata."""
        generator = MarkdownGenerator(use_llm=False)

        result = generator.generate(extraction_result, simple_schema_info)

        assert "Model Used" in result.markdown
        assert "gpt-4o" in result.markdown
        assert "Processing Time" in result.markdown
        assert "Total Tokens" in result.markdown

    def test_generate_includes_confidence(
        self,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test that generated Markdown includes confidence indicators."""
        generator = MarkdownGenerator(use_llm=False)

        result = generator.generate(
            extraction_result,
            simple_schema_info,
            include_confidence=True,
        )

        assert "confidence" in result.markdown.lower()

    def test_generate_includes_source_references(
        self,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test that generated Markdown includes source references."""
        generator = MarkdownGenerator(use_llm=False)

        result = generator.generate(
            extraction_result,
            simple_schema_info,
            include_source_refs=True,
        )

        assert "Source References" in result.markdown
        assert len(result.source_references) > 0


class TestMarkdownGeneratorSourceReferences:
    """Tests for source reference collection."""

    def test_collect_source_references(
        self,
        extraction_result: ExtractionResult,
    ) -> None:
        """Test collecting source references."""
        generator = MarkdownGenerator(use_llm=False)

        refs = generator._collect_source_references(extraction_result.field_extractions)

        # Should have 2 refs (name and age have source_text)
        assert len(refs) == 2
        assert refs[0]["field"] == "name"
        assert refs[0]["source_text"] == "My name is John Doe"

    def test_collect_source_references_empty(self) -> None:
        """Test collecting source references when none exist."""
        generator = MarkdownGenerator(use_llm=False)

        field_extractions = [
            FieldExtraction(field_path="name", value="John"),
        ]

        refs = generator._collect_source_references(field_extractions)

        assert len(refs) == 0


class TestMarkdownGeneratorSchemaDescription:
    """Tests for schema description building."""

    def test_build_schema_description(
        self,
        simple_schema_info: Any,
    ) -> None:
        """Test building schema description."""
        generator = MarkdownGenerator(use_llm=False)

        description = generator._build_schema_description(simple_schema_info)

        assert "object" in description
        assert "name" in description
        assert "age" in description


class TestMarkdownGeneratorFieldDetails:
    """Tests for field details building."""

    def test_build_field_details(
        self,
        extraction_result: ExtractionResult,
    ) -> None:
        """Test building field details."""
        generator = MarkdownGenerator(use_llm=False)

        details = generator._build_field_details(
            extraction_result.field_extractions,
            include_source_refs=True,
            include_confidence=True,
        )

        assert "name" in details
        assert "John Doe" in details
        assert "confidence" in details.lower()

    def test_build_field_details_empty(self) -> None:
        """Test building field details with empty extractions."""
        generator = MarkdownGenerator(use_llm=False)

        details = generator._build_field_details(
            [],
            include_source_refs=True,
            include_confidence=True,
        )

        assert "No detailed field information" in details


class TestMarkdownGeneratorDataFormatting:
    """Tests for data formatting."""

    def test_format_data_for_prompt(self) -> None:
        """Test formatting data for prompt."""
        generator = MarkdownGenerator(use_llm=False)

        data = {"name": "John", "age": 30}
        formatted = generator._format_data_for_prompt(data)

        assert "name" in formatted
        assert "John" in formatted
        assert "30" in formatted

    def test_format_dict_inline(self) -> None:
        """Test formatting dictionary inline."""
        generator = MarkdownGenerator(use_llm=False)

        data = {"a": 1, "b": 2, "c": None}
        formatted = generator._format_dict_inline(data)

        assert "a: 1" in formatted
        assert "b: 2" in formatted
        assert "c" not in formatted  # None values excluded


class TestMarkdownGeneratorNestedData:
    """Tests for nested data rendering."""

    def test_render_nested_data(self) -> None:
        """Test rendering nested data."""
        generator = MarkdownGenerator(use_llm=False)

        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "person": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                },
            }
        )

        extraction_result = ExtractionResult(
            extracted_data={"person": {"name": "John"}},
            model_used="gpt-4o",
        )

        result = generator.generate(extraction_result, schema_info)

        assert "person" in result.markdown
        assert "John" in result.markdown

    def test_render_array_data(self) -> None:
        """Test rendering array data."""
        generator = MarkdownGenerator(use_llm=False)

        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            }
        )

        extraction_result = ExtractionResult(
            extracted_data={"items": ["item1", "item2", "item3"]},
            model_used="gpt-4o",
        )

        result = generator.generate(extraction_result, schema_info)

        assert "items" in result.markdown
        assert "item1" in result.markdown
        assert "item2" in result.markdown


class TestMarkdownGeneratorLLM:
    """Tests for LLM-based Markdown generation."""

    def test_generate_with_mock_llm(
        self,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test generating Markdown with mocked LLM."""
        generator = MarkdownGenerator(use_llm=True, api_key="test-key")

        # Create mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "# Generated Summary\n\n**Name:** John Doe"
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        mock_llm.invoke.return_value = mock_response
        generator._llm = mock_llm

        result = generator.generate(extraction_result, simple_schema_info)

        assert result.generated_by_llm is True
        assert "Generated Summary" in result.markdown
        assert result.token_usage["total_tokens"] == 150

    def test_fallback_to_template_on_llm_failure(
        self,
        extraction_result: ExtractionResult,
        simple_schema_info: Any,
    ) -> None:
        """Test fallback to template when LLM fails."""
        generator = MarkdownGenerator(use_llm=True, api_key="test-key")

        # Create mock LLM that raises exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        generator._llm = mock_llm

        result = generator.generate(extraction_result, simple_schema_info)

        # Should fall back to template
        assert result.generated_by_llm is False
        assert "# Extraction Summary" in result.markdown


class TestMarkdownGeneratorMissingData:
    """Tests for handling missing data."""

    def test_handle_null_values(self) -> None:
        """Test handling null values in data."""
        generator = MarkdownGenerator(use_llm=False)

        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
            }
        )

        extraction_result = ExtractionResult(
            extracted_data={"name": "John", "email": None},
            model_used="gpt-4o",
        )

        result = generator.generate(extraction_result, schema_info)

        assert "Not found" in result.markdown or "null" in result.markdown.lower()

    def test_handle_empty_extraction(self) -> None:
        """Test handling empty extraction data."""
        generator = MarkdownGenerator(use_llm=False)

        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {},
            }
        )

        extraction_result = ExtractionResult(
            extracted_data={},
            model_used="gpt-4o",
        )

        result = generator.generate(extraction_result, schema_info)

        assert "# Extraction Summary" in result.markdown
