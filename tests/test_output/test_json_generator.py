"""Tests for the JSON output generator."""

from typing import Any

import pytest

from agentic_document_extraction.output.json_generator import (
    JsonGenerator,
    JsonOutputResult,
    ValidationError,
    ValidationResult,
    normalize_date,
    parse_date_string,
)
from agentic_document_extraction.services.schema_validator import SchemaValidator


@pytest.fixture
def json_generator() -> JsonGenerator:
    """Create a JSON generator instance."""
    return JsonGenerator(max_retries=3)


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
def nested_schema_info() -> Any:
    """Create a nested schema info for testing."""
    validator = SchemaValidator()
    return validator.validate(
        {
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
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["person"],
        }
    )


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid validation result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.missing_required_fields == []

    def test_invalid_result_with_errors(self) -> None:
        """Test creating an invalid validation result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["name: is required", "age: must be integer"],
            missing_required_fields=["name"],
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "name" in result.missing_required_fields

    def test_to_dict(self) -> None:
        """Test converting validation result to dictionary."""
        result = ValidationResult(
            is_valid=False,
            errors=["error1"],
            missing_required_fields=["field1"],
        )

        result_dict = result.to_dict()

        assert result_dict["is_valid"] is False
        assert result_dict["errors"] == ["error1"]
        assert result_dict["missing_required_fields"] == ["field1"]


class TestJsonOutputResult:
    """Tests for JsonOutputResult dataclass."""

    def test_basic_result(self) -> None:
        """Test creating a basic JSON output result."""
        validation = ValidationResult(is_valid=True)
        result = JsonOutputResult(
            data={"name": "John", "age": 30},
            validation_result=validation,
        )

        assert result.data == {"name": "John", "age": 30}
        assert result.validation_result.is_valid is True
        assert result.retry_count == 0

    def test_to_dict(self) -> None:
        """Test converting JSON output result to dictionary."""
        validation = ValidationResult(is_valid=True)
        result = JsonOutputResult(
            data={"name": "John"},
            validation_result=validation,
            retry_count=2,
        )

        result_dict = result.to_dict()

        assert result_dict["data"] == {"name": "John"}
        assert result_dict["validation"]["is_valid"] is True
        assert result_dict["retry_count"] == 2


class TestJsonGeneratorValidation:
    """Tests for JSON generator validation."""

    def test_validate_valid_data(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test validating valid data."""
        data = {"name": "John Doe", "age": 30, "email": "john@example.com"}

        result = json_generator.validate(data, simple_schema_info)

        assert result.is_valid is True
        assert result.errors == []
        assert result.missing_required_fields == []

    def test_validate_missing_required_field(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test validating data with missing required field."""
        data = {"name": "John Doe", "email": "john@example.com"}  # missing age

        result = json_generator.validate(data, simple_schema_info)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_null_required_field(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test validating data with null required field."""
        data = {"name": "John Doe", "age": None, "email": "john@example.com"}

        result = json_generator.validate(data, simple_schema_info)

        assert result.is_valid is False
        assert "age" in result.missing_required_fields

    def test_validate_wrong_type(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test validating data with wrong type."""
        data = {"name": "John Doe", "age": "thirty", "email": "john@example.com"}

        result = json_generator.validate(data, simple_schema_info)

        assert result.is_valid is False
        assert len(result.errors) > 0


class TestJsonGeneratorGenerate:
    """Tests for JSON generator generate method."""

    def test_generate_valid_data(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test generating output from valid data."""
        data = {"name": "John Doe", "age": 30, "email": "john@example.com"}

        result = json_generator.generate(data, simple_schema_info)

        assert result.data["name"] == "John Doe"
        assert result.data["age"] == 30
        assert result.validation_result.is_valid is True

    def test_generate_handles_missing_optional(
        self,
        json_generator: JsonGenerator,
    ) -> None:
        """Test generating output handles missing optional field."""
        # Create schema that allows null for optional fields
        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": ["string", "null"]},  # Allow null
                },
                "required": ["name", "age"],
            }
        )

        data = {"name": "John Doe", "age": 30}  # email is optional

        result = json_generator.generate(data, schema_info, handle_nulls=True)

        assert result.data["name"] == "John Doe"
        assert result.validation_result.is_valid is True

    def test_generate_with_extra_fields(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test generating output preserves extra fields."""
        data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "extra_field": "extra_value",
        }

        result = json_generator.generate(data, simple_schema_info)

        assert "extra_field" in result.data
        assert result.data["extra_field"] == "extra_value"


class TestJsonGeneratorTypeCoercion:
    """Tests for type coercion in JSON generator."""

    def test_coerce_string_to_integer(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test coercing string to integer."""
        data = {"name": "John", "age": "30", "email": "john@example.com"}

        result = json_generator.generate(data, simple_schema_info)

        assert result.data["age"] == 30
        assert isinstance(result.data["age"], int)

    def test_coerce_number_to_string(
        self,
        json_generator: JsonGenerator,
    ) -> None:
        """Test coercing number to string."""
        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                },
            }
        )

        data = {"code": 12345}

        result = json_generator.generate(data, schema_info)

        assert result.data["code"] == "12345"
        assert isinstance(result.data["code"], str)


class TestJsonGeneratorNestedData:
    """Tests for nested data handling."""

    def test_validate_nested_data(
        self,
        json_generator: JsonGenerator,
        nested_schema_info: Any,
    ) -> None:
        """Test validating nested data."""
        data = {
            "person": {
                "name": "John",
                "address": {"city": "NYC", "country": "USA"},
            },
            "items": ["item1", "item2"],
        }

        result = json_generator.validate(data, nested_schema_info)

        assert result.is_valid is True

    def test_get_nested_value(
        self,
        json_generator: JsonGenerator,
    ) -> None:
        """Test getting nested values."""
        data = {"person": {"address": {"city": "NYC"}}}

        value = json_generator._get_nested_value(data, "person.address.city")

        assert value == "NYC"

    def test_get_nested_value_missing(
        self,
        json_generator: JsonGenerator,
    ) -> None:
        """Test getting missing nested value."""
        data = {"person": {"name": "John"}}

        value = json_generator._get_nested_value(data, "person.address.city")

        assert value is None


class TestJsonGeneratorValidationFeedback:
    """Tests for validation feedback generation."""

    def test_generate_feedback_missing_fields(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test generating feedback for missing fields."""
        validation = ValidationResult(
            is_valid=False,
            errors=[],
            missing_required_fields=["name", "age"],
        )

        feedback = json_generator.get_validation_feedback(
            validation, simple_schema_info
        )

        assert "name" in feedback
        assert "age" in feedback
        assert "Missing required fields" in feedback

    def test_generate_feedback_validation_errors(
        self,
        json_generator: JsonGenerator,
        simple_schema_info: Any,
    ) -> None:
        """Test generating feedback for validation errors."""
        validation = ValidationResult(
            is_valid=False,
            errors=["age: 'thirty' is not of type 'integer'"],
            missing_required_fields=[],
        )

        feedback = json_generator.get_validation_feedback(
            validation, simple_schema_info
        )

        assert "Validation errors" in feedback
        assert "age" in feedback


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_basic_error(self) -> None:
        """Test creating a basic validation error."""
        error = ValidationError("Validation failed")

        assert str(error) == "Validation failed"
        assert error.errors == []
        assert error.data is None

    def test_error_with_details(self) -> None:
        """Test creating error with details."""
        error = ValidationError(
            "Validation failed",
            errors=["field1: required", "field2: wrong type"],
            data={"field1": None, "field2": "invalid"},
        )

        assert len(error.errors) == 2
        assert error.data is not None
        assert "field1" in error.data


class TestParseDateString:
    """Tests for parse_date_string function."""

    def test_parse_iso_format(self) -> None:
        """Test parsing ISO format date."""
        result = parse_date_string("2024-02-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 15

    def test_parse_month_name_format(self) -> None:
        """Test parsing full month name format."""
        result = parse_date_string("February 15, 2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 15

    def test_parse_abbreviated_month_format(self) -> None:
        """Test parsing abbreviated month name format."""
        result = parse_date_string("Feb 15, 2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 15

    def test_parse_us_format(self) -> None:
        """Test parsing US date format."""
        result = parse_date_string("02/15/2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 15

    def test_parse_eu_format_slash(self) -> None:
        """Test parsing EU date format with slash."""
        result = parse_date_string("15/02/2024")
        assert result is not None
        # Note: This will be parsed as EU format
        assert result.year == 2024

    def test_parse_eu_format_dot(self) -> None:
        """Test parsing EU date format with dots."""
        result = parse_date_string("15.02.2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 15

    def test_parse_none_returns_none(self) -> None:
        """Test that None input returns None."""
        result = parse_date_string(None)  # type: ignore[arg-type]
        assert result is None

    def test_parse_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        result = parse_date_string("")
        assert result is None

    def test_parse_invalid_date_returns_none(self) -> None:
        """Test that invalid date string returns None."""
        result = parse_date_string("not a date")
        assert result is None

    def test_parse_whitespace_handling(self) -> None:
        """Test that whitespace is handled correctly."""
        result = parse_date_string("  February 15, 2024  ")
        assert result is not None
        assert result.year == 2024

    def test_parse_month_name_no_comma(self) -> None:
        """Test parsing month name format without comma."""
        result = parse_date_string("February 15 2024")
        assert result is not None
        assert result.year == 2024


class TestNormalizeDate:
    """Tests for normalize_date function."""

    def test_normalize_month_name_to_iso(self) -> None:
        """Test normalizing month name format to ISO."""
        result = normalize_date("February 15, 2024")
        assert result == "2024-02-15"

    def test_normalize_already_iso(self) -> None:
        """Test that ISO format is preserved."""
        result = normalize_date("2024-02-15")
        assert result == "2024-02-15"

    def test_normalize_abbreviated_month(self) -> None:
        """Test normalizing abbreviated month name."""
        result = normalize_date("Feb 15, 2024")
        assert result == "2024-02-15"

    def test_normalize_us_format(self) -> None:
        """Test normalizing US date format."""
        result = normalize_date("02/15/2024")
        assert result == "2024-02-15"

    def test_normalize_none_returns_none(self) -> None:
        """Test that None returns None."""
        result = normalize_date(None)
        assert result is None

    def test_normalize_non_string_returns_none(self) -> None:
        """Test that non-string returns None."""
        result = normalize_date(12345)  # type: ignore[arg-type]
        assert result is None

    def test_normalize_unparseable_returns_original(self) -> None:
        """Test that unparseable string returns original."""
        result = normalize_date("some random text")
        assert result == "some random text"

    def test_normalize_eu_format(self) -> None:
        """Test normalizing EU date format."""
        result = normalize_date("15.02.2024")
        assert result == "2024-02-15"


class TestJsonGeneratorDateNormalization:
    """Tests for date normalization in JSON generator."""

    @pytest.fixture
    def date_schema_info(self) -> Any:
        """Create a schema info with date format fields."""
        validator = SchemaValidator()
        return validator.validate(
            {
                "type": "object",
                "properties": {
                    "invoice_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Invoice issue date",
                    },
                    "due_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Payment due date",
                    },
                    "name": {
                        "type": "string",
                        "description": "Customer name",
                    },
                },
                "required": ["invoice_date"],
            }
        )

    def test_normalizes_date_field(
        self,
        json_generator: JsonGenerator,
        date_schema_info: Any,
    ) -> None:
        """Test that date fields are normalized to ISO format."""
        data = {
            "invoice_date": "February 15, 2024",
            "due_date": "March 1, 2024",
            "name": "Test Customer",
        }

        result = json_generator.generate(data, date_schema_info)

        assert result.data["invoice_date"] == "2024-02-15"
        assert result.data["due_date"] == "2024-03-01"
        assert result.data["name"] == "Test Customer"  # Non-date field unchanged

    def test_preserves_iso_date(
        self,
        json_generator: JsonGenerator,
        date_schema_info: Any,
    ) -> None:
        """Test that ISO format dates are preserved."""
        data = {
            "invoice_date": "2024-02-15",
            "name": "Test Customer",
        }

        result = json_generator.generate(data, date_schema_info)

        assert result.data["invoice_date"] == "2024-02-15"

    def test_normalizes_various_date_formats(
        self,
        json_generator: JsonGenerator,
        date_schema_info: Any,
    ) -> None:
        """Test normalization of various date formats."""
        test_cases = [
            ("February 15, 2024", "2024-02-15"),
            ("Feb 15, 2024", "2024-02-15"),
            ("02/15/2024", "2024-02-15"),
            ("2024-02-15", "2024-02-15"),
            ("15.02.2024", "2024-02-15"),
        ]

        for input_date, expected in test_cases:
            data = {"invoice_date": input_date, "name": "Test"}
            result = json_generator.generate(data, date_schema_info)
            assert result.data["invoice_date"] == expected, (
                f"Failed for input: {input_date}"
            )

    def test_non_date_string_fields_unchanged(
        self,
        json_generator: JsonGenerator,
        date_schema_info: Any,
    ) -> None:
        """Test that non-date string fields are not modified."""
        data = {
            "invoice_date": "2024-02-15",
            "name": "February Company Inc.",  # Has month name but not a date field
        }

        result = json_generator.generate(data, date_schema_info)

        assert result.data["name"] == "February Company Inc."

    def test_handles_null_date_field(
        self,
        json_generator: JsonGenerator,
        date_schema_info: Any,
    ) -> None:
        """Test handling of null date fields."""
        data = {
            "invoice_date": "2024-02-15",
            "due_date": None,
            "name": "Test",
        }

        result = json_generator.generate(data, date_schema_info, handle_nulls=True)

        assert result.data["invoice_date"] == "2024-02-15"
        # due_date should be handled according to handle_nulls setting

    def test_invoice_schema_date_normalization(
        self,
        json_generator: JsonGenerator,
    ) -> None:
        """Test date normalization with invoice-like schema."""
        validator = SchemaValidator()
        schema_info = validator.validate(
            {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "invoice_date": {"type": "string", "format": "date"},
                    "due_date": {"type": "string", "format": "date"},
                    "total": {"type": "number"},
                },
                "required": ["invoice_number", "invoice_date", "total"],
            }
        )

        data = {
            "invoice_number": "INV-2024-001",
            "invoice_date": "January 15, 2024",
            "due_date": "February 15, 2024",
            "total": 1250.00,
        }

        result = json_generator.generate(data, schema_info)

        assert result.data["invoice_date"] == "2024-01-15"
        assert result.data["due_date"] == "2024-02-15"
        assert result.data["invoice_number"] == "INV-2024-001"
        assert result.data["total"] == 1250.00
