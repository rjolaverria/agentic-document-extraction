"""JSON output generator with schema validation.

This module provides functionality to validate and format extracted data
as JSON output according to a user-provided schema.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft7Validator

from agentic_document_extraction.services.schema_validator import SchemaInfo

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when JSON validation fails."""

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with message and validation errors.

        Args:
            message: Error message.
            errors: List of specific validation errors.
            data: The data that failed validation.
        """
        super().__init__(message)
        self.errors = errors or []
        self.data = data


@dataclass
class ValidationResult:
    """Result of JSON schema validation."""

    is_valid: bool
    """Whether the data is valid against the schema."""

    errors: list[str] = field(default_factory=list)
    """List of validation error messages."""

    missing_required_fields: list[str] = field(default_factory=list)
    """List of required fields that are missing or null."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with validation result information.
        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "missing_required_fields": self.missing_required_fields,
        }


@dataclass
class JsonOutputResult:
    """Result of JSON output generation."""

    data: dict[str, Any]
    """The validated JSON data."""

    validation_result: ValidationResult
    """Validation result for the data."""

    retry_count: int = 0
    """Number of retries performed to get valid data."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with JSON output result information.
        """
        return {
            "data": self.data,
            "validation": self.validation_result.to_dict(),
            "retry_count": self.retry_count,
        }


class JsonGenerator:
    """Generates validated JSON output from extraction results.

    Validates extracted data against the user-provided schema and
    handles missing or uncertain data gracefully.
    """

    def __init__(self, max_retries: int = 3) -> None:
        """Initialize the JSON generator.

        Args:
            max_retries: Maximum number of retries for validation failures.
        """
        self.max_retries = max_retries

    def validate(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
    ) -> ValidationResult:
        """Validate data against a JSON schema.

        Args:
            data: Data to validate.
            schema_info: Schema information.

        Returns:
            ValidationResult with validation status and errors.
        """
        validator = Draft7Validator(schema_info.schema)
        errors: list[str] = []
        missing_required: list[str] = []

        for error in validator.iter_errors(data):
            path = (
                ".".join(str(p) for p in error.absolute_path)
                if error.absolute_path
                else "root"
            )
            errors.append(f"{path}: {error.message}")

        # Check for null values in required fields
        for field_info in schema_info.required_fields:
            value = self._get_nested_value(data, field_info.path)
            if value is None:
                missing_required.append(field_info.path)

        is_valid = len(errors) == 0 and len(missing_required) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            missing_required_fields=missing_required,
        )

    def generate(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
        handle_nulls: bool = True,
    ) -> JsonOutputResult:
        """Generate validated JSON output.

        Args:
            data: Extracted data to validate and format.
            schema_info: Schema information for validation.
            handle_nulls: Whether to handle null values gracefully.

        Returns:
            JsonOutputResult with validated data.

        Raises:
            ValidationError: If data doesn't match schema after retries.
        """
        # First, clean and prepare the data
        cleaned_data = self._prepare_data(data, schema_info, handle_nulls)

        # Validate the data
        validation_result = self.validate(cleaned_data, schema_info)

        if not validation_result.is_valid:
            logger.warning(
                f"Validation failed: {len(validation_result.errors)} errors, "
                f"{len(validation_result.missing_required_fields)} missing required fields"
            )

        return JsonOutputResult(
            data=cleaned_data,
            validation_result=validation_result,
            retry_count=0,
        )

    def _prepare_data(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
        handle_nulls: bool,
    ) -> dict[str, Any]:
        """Prepare and clean data for output.

        Args:
            data: Raw extracted data.
            schema_info: Schema information.
            handle_nulls: Whether to handle null values gracefully.

        Returns:
            Cleaned data dictionary.
        """
        cleaned: dict[str, Any] = {}

        # Process all fields from schema
        for field_info in schema_info.all_fields:
            field_name = field_info.name
            value = data.get(field_name)

            if value is not None:
                cleaned[field_name] = self._clean_value(
                    value, field_info.field_type, handle_nulls
                )
            elif handle_nulls:
                # Set default values for missing fields
                cleaned[field_name] = self._get_default_value(field_info.field_type)
            else:
                cleaned[field_name] = None

        # Include any extra fields that were extracted but not in schema
        for key, value in data.items():
            if key not in cleaned:
                cleaned[key] = value

        return cleaned

    def _clean_value(
        self,
        value: Any,
        field_type: str | list[str],
        handle_nulls: bool,
    ) -> Any:
        """Clean a value according to its type.

        Args:
            value: Value to clean.
            field_type: Expected type of the value.
            handle_nulls: Whether to handle null values.

        Returns:
            Cleaned value.
        """
        if value is None:
            if handle_nulls:
                return self._get_default_value(field_type)
            return None

        # Handle type coercion if needed
        expected_type = field_type[0] if isinstance(field_type, list) else field_type

        if expected_type == "string" and not isinstance(value, str):
            return str(value)
        elif expected_type == "integer" and isinstance(value, (int, float, str)):
            try:
                return int(float(value)) if isinstance(value, str) else int(value)
            except (ValueError, TypeError):
                return value
        elif expected_type == "number" and isinstance(value, (int, float, str)):
            try:
                return float(value) if isinstance(value, str) else value
            except (ValueError, TypeError):
                return value
        elif expected_type == "boolean" and not isinstance(value, bool):
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1")
            return bool(value)
        elif expected_type == "array" and not isinstance(value, list):
            return [value]
        elif expected_type == "object" and isinstance(value, dict):
            return value

        return value

    def _get_default_value(self, field_type: str | list[str]) -> Any:
        """Get default value for a field type.

        Args:
            field_type: JSON Schema type.

        Returns:
            Default value for the type.
        """
        expected_type = field_type[0] if isinstance(field_type, list) else field_type

        defaults: dict[str, Any] = {
            "string": None,
            "integer": None,
            "number": None,
            "boolean": None,
            "array": [],
            "object": {},
            "null": None,
        }

        return defaults.get(expected_type)

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get a value from nested dictionary using dot notation.

        Args:
            data: Dictionary to get value from.
            path: Dot-separated path (e.g., 'address.city').

        Returns:
            Value at path or None if not found.
        """
        keys = path.split(".")
        current: Any = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def get_validation_feedback(
        self,
        validation_result: ValidationResult,
        schema_info: SchemaInfo,  # noqa: ARG002
    ) -> str:
        """Generate feedback for retry attempts.

        Args:
            validation_result: Result of validation.
            schema_info: Schema information (reserved for future use).

        Returns:
            Feedback string for improving extraction.
        """
        feedback_parts: list[str] = []

        if validation_result.missing_required_fields:
            fields_list = ", ".join(validation_result.missing_required_fields)
            feedback_parts.append(
                f"Missing required fields: {fields_list}. "
                "Please extract values for these fields from the text."
            )

        if validation_result.errors:
            feedback_parts.append(
                "Validation errors:\n"
                + "\n".join(f"- {e}" for e in validation_result.errors)
            )

        return "\n\n".join(feedback_parts)
