"""JSON Schema validation service.

This module provides functionality to validate user-provided JSON schemas
before processing, ensuring they conform to JSON Schema standards (Draft 7+)
and extracting field information for extraction planning.
"""

from typing import Any

from jsonschema import Draft7Validator, SchemaError
from jsonschema.validators import validator_for

from agentic_document_extraction.utils.exceptions import SchemaValidationError
from agentic_document_extraction.utils.logging import get_logger

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "FieldInfo",
    "SchemaInfo",
    "SchemaValidationError",
    "SchemaValidator",
]


class FieldInfo:
    """Information about a field extracted from a JSON schema."""

    def __init__(
        self,
        name: str,
        field_type: str | list[str],
        required: bool = False,
        description: str | None = None,
        path: str | None = None,
        nested_fields: list["FieldInfo"] | None = None,
        format_spec: str | None = None,
    ) -> None:
        """Initialize field information.

        Args:
            name: Field name.
            field_type: JSON Schema type (string, number, boolean, object, array).
            required: Whether this field is required.
            description: Optional field description from schema.
            path: JSON path to this field (e.g., "address.city").
            nested_fields: For object types, list of nested field info.
            format_spec: Optional JSON Schema format (e.g., "date", "email", "uri").
        """
        self.name = name
        self.field_type = field_type
        self.required = required
        self.description = description
        self.path = path or name
        self.nested_fields = nested_fields or []
        self.format_spec = format_spec

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with field information.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.field_type,
            "required": self.required,
            "path": self.path,
        }
        if self.description:
            result["description"] = self.description
        if self.nested_fields:
            result["nested_fields"] = [f.to_dict() for f in self.nested_fields]
        if self.format_spec:
            result["format"] = self.format_spec
        return result


class SchemaInfo:
    """Parsed information from a JSON schema for extraction planning."""

    def __init__(
        self,
        schema: dict[str, Any],
        required_fields: list[FieldInfo],
        optional_fields: list[FieldInfo],
        schema_type: str,
    ) -> None:
        """Initialize schema information.

        Args:
            schema: The original validated schema.
            required_fields: List of required field information.
            optional_fields: List of optional field information.
            schema_type: Root type of the schema (object, array, etc.).
        """
        self.schema = schema
        self.required_fields = required_fields
        self.optional_fields = optional_fields
        self.schema_type = schema_type

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with schema information.
        """
        return {
            "schema_type": self.schema_type,
            "required_fields": [f.to_dict() for f in self.required_fields],
            "optional_fields": [f.to_dict() for f in self.optional_fields],
            "required_count": len(self.required_fields),
            "optional_count": len(self.optional_fields),
            "total_fields": len(self.required_fields) + len(self.optional_fields),
        }

    @property
    def all_fields(self) -> list[FieldInfo]:
        """Get all fields (required and optional).

        Returns:
            Combined list of all fields.
        """
        return self.required_fields + self.optional_fields


class SchemaValidator:
    """Validates JSON schemas and extracts field information.

    Supports JSON Schema Draft 7 and later drafts. Validates schema syntax,
    checks for common issues, and extracts required/optional field information
    for extraction planning.
    """

    # Supported JSON Schema data types
    SUPPORTED_TYPES: set[str] = {
        "string",
        "number",
        "integer",
        "boolean",
        "object",
        "array",
        "null",
    }

    def __init__(self) -> None:
        """Initialize the schema validator."""
        pass

    def validate(self, schema: dict[str, Any]) -> SchemaInfo:
        """Validate a JSON schema and extract field information.

        Args:
            schema: JSON schema to validate.

        Returns:
            SchemaInfo with validated schema and extracted field information.

        Raises:
            SchemaValidationError: If the schema is invalid.
        """
        # Check basic schema structure
        if not isinstance(schema, dict):
            raise SchemaValidationError(
                "Schema must be a JSON object",
                errors=["Expected a dictionary/object at the root level"],
            )

        # Validate schema syntax using jsonschema
        self._validate_schema_syntax(schema)

        # Extract field information
        schema_type = schema.get("type", "object")
        if isinstance(schema_type, list):
            # Handle type arrays (e.g., ["string", "null"])
            schema_type = schema_type[0] if schema_type else "object"

        required_fields, optional_fields = self._extract_fields(schema)

        logger.info(
            "Schema validated",
            schema_type=schema_type,
            required_fields=len(required_fields),
            optional_fields=len(optional_fields),
        )

        return SchemaInfo(
            schema=schema,
            required_fields=required_fields,
            optional_fields=optional_fields,
            schema_type=schema_type,
        )

    def _validate_schema_syntax(self, schema: dict[str, Any]) -> None:
        """Validate JSON schema syntax.

        Args:
            schema: Schema to validate.

        Raises:
            SchemaValidationError: If schema syntax is invalid.
        """
        errors: list[str] = []

        # Determine the appropriate validator class based on $schema
        validator_cls = validator_for(schema)

        try:
            # Check if the schema itself is valid
            validator_cls.check_schema(schema)
        except SchemaError as e:
            errors.append(f"Schema structure error: {e.message}")
            if e.context:
                for context_error in e.context:
                    errors.append(f"  - {context_error.message}")

        # Additional validation checks
        self._check_type_definitions(schema, errors)

        if errors:
            raise SchemaValidationError(
                f"Invalid JSON Schema: {len(errors)} error(s) found",
                errors=errors,
            )

    def _check_type_definitions(
        self, schema: dict[str, Any], errors: list[str], path: str = ""
    ) -> None:
        """Recursively check type definitions in the schema.

        Args:
            schema: Schema or sub-schema to check.
            errors: List to append errors to.
            path: Current path in the schema for error messages.
        """
        if not isinstance(schema, dict):
            return

        # Check type field if present
        if "type" in schema:
            schema_type = schema["type"]
            if isinstance(schema_type, str):
                if schema_type not in self.SUPPORTED_TYPES:
                    errors.append(
                        f"Unsupported type '{schema_type}' at {path or 'root'}"
                    )
            elif isinstance(schema_type, list):
                for t in schema_type:
                    if t not in self.SUPPORTED_TYPES:
                        errors.append(
                            f"Unsupported type '{t}' in type array at {path or 'root'}"
                        )

        # Recursively check properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                prop_path = f"{path}.{prop_name}" if path else prop_name
                self._check_type_definitions(prop_schema, errors, prop_path)

        # Check items for arrays
        if "items" in schema:
            items = schema["items"]
            if isinstance(items, dict):
                items_path = f"{path}[]" if path else "[]"
                self._check_type_definitions(items, errors, items_path)
            elif isinstance(items, list):
                for i, item_schema in enumerate(items):
                    items_path = f"{path}[{i}]" if path else f"[{i}]"
                    self._check_type_definitions(item_schema, errors, items_path)

        # Check additionalProperties if it's a schema
        if "additionalProperties" in schema and isinstance(
            schema["additionalProperties"], dict
        ):
            add_path = (
                f"{path}.<additionalProperties>" if path else "<additionalProperties>"
            )
            self._check_type_definitions(
                schema["additionalProperties"], errors, add_path
            )

        # Check allOf, anyOf, oneOf
        for keyword in ("allOf", "anyOf", "oneOf"):
            if keyword in schema and isinstance(schema[keyword], list):
                for i, sub_schema in enumerate(schema[keyword]):
                    sub_path = f"{path}.{keyword}[{i}]" if path else f"{keyword}[{i}]"
                    self._check_type_definitions(sub_schema, errors, sub_path)

    def _extract_fields(
        self,
        schema: dict[str, Any],
        parent_path: str = "",
        parent_required: set[str] | None = None,
    ) -> tuple[list[FieldInfo], list[FieldInfo]]:
        """Extract required and optional fields from schema.

        Args:
            schema: Schema to extract fields from.
            parent_path: Path prefix for nested fields.
            parent_required: Set of required field names from parent schema.

        Returns:
            Tuple of (required_fields, optional_fields).
        """
        required_fields: list[FieldInfo] = []
        optional_fields: list[FieldInfo] = []

        if parent_required is None:
            parent_required = set()

        # Get required field names for this level
        required_names: set[str] = set(schema.get("required", []))

        # Extract properties
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return required_fields, optional_fields

        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue

            field_path = f"{parent_path}.{field_name}" if parent_path else field_name
            is_required = field_name in required_names

            field_type = field_schema.get("type", "any")
            description = field_schema.get("description")
            format_spec = field_schema.get("format")

            # Handle nested objects
            nested_fields: list[FieldInfo] = []
            if field_type == "object" and "properties" in field_schema:
                nested_req, nested_opt = self._extract_fields(
                    field_schema,
                    parent_path=field_path,
                    parent_required=set(field_schema.get("required", [])),
                )
                nested_fields = nested_req + nested_opt

            # Handle arrays with object items
            if field_type == "array" and "items" in field_schema:
                items = field_schema["items"]
                if isinstance(items, dict) and items.get("type") == "object":
                    nested_req, nested_opt = self._extract_fields(
                        items,
                        parent_path=f"{field_path}[]",
                        parent_required=set(items.get("required", [])),
                    )
                    nested_fields = nested_req + nested_opt

            field_info = FieldInfo(
                name=field_name,
                field_type=field_type,
                required=is_required,
                description=description,
                path=field_path,
                nested_fields=nested_fields,
                format_spec=format_spec,
            )

            if is_required:
                required_fields.append(field_info)
            else:
                optional_fields.append(field_info)

        return required_fields, optional_fields

    def validate_data_against_schema(
        self, data: Any, schema: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate data against a JSON schema.

        Args:
            data: Data to validate.
            schema: JSON schema to validate against.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        validator = Draft7Validator(schema)
        errors: list[str] = []

        for error in validator.iter_errors(data):
            path = (
                ".".join(str(p) for p in error.absolute_path)
                if error.absolute_path
                else "root"
            )
            errors.append(f"{path}: {error.message}")

        return len(errors) == 0, errors

    @staticmethod
    def get_supported_types() -> list[str]:
        """Get list of supported JSON Schema types.

        Returns:
            List of supported type names.
        """
        return sorted(SchemaValidator.SUPPORTED_TYPES)
