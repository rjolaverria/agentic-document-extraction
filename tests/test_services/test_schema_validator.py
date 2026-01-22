"""Tests for the JSON Schema validation service."""

from typing import Any

import pytest

from agentic_document_extraction.services.schema_validator import (
    FieldInfo,
    SchemaInfo,
    SchemaValidationError,
    SchemaValidator,
)


@pytest.fixture
def validator() -> SchemaValidator:
    """Create a SchemaValidator instance for testing."""
    return SchemaValidator()


class TestFieldInfo:
    """Tests for FieldInfo class."""

    def test_basic_field_info_creation(self) -> None:
        """Test creating basic field info."""
        field = FieldInfo(
            name="test_field",
            field_type="string",
            required=True,
            description="A test field",
            path="test_field",
        )

        assert field.name == "test_field"
        assert field.field_type == "string"
        assert field.required is True
        assert field.description == "A test field"
        assert field.path == "test_field"
        assert field.nested_fields == []

    def test_field_info_with_nested_fields(self) -> None:
        """Test creating field info with nested fields."""
        nested = FieldInfo(name="city", field_type="string", required=True)
        field = FieldInfo(
            name="address",
            field_type="object",
            required=False,
            nested_fields=[nested],
        )

        assert len(field.nested_fields) == 1
        assert field.nested_fields[0].name == "city"

    def test_field_info_to_dict(self) -> None:
        """Test converting field info to dictionary."""
        field = FieldInfo(
            name="email",
            field_type="string",
            required=True,
            description="User email address",
            path="user.email",
        )

        result = field.to_dict()

        assert result["name"] == "email"
        assert result["type"] == "string"
        assert result["required"] is True
        assert result["path"] == "user.email"
        assert result["description"] == "User email address"

    def test_field_info_to_dict_without_description(self) -> None:
        """Test that description is omitted when not set."""
        field = FieldInfo(name="count", field_type="integer", required=False)
        result = field.to_dict()

        assert "description" not in result

    def test_field_info_to_dict_with_nested_fields(self) -> None:
        """Test nested fields are included in dict output."""
        nested = FieldInfo(name="street", field_type="string", required=True)
        field = FieldInfo(
            name="address",
            field_type="object",
            nested_fields=[nested],
        )

        result = field.to_dict()

        assert "nested_fields" in result
        assert len(result["nested_fields"]) == 1
        assert result["nested_fields"][0]["name"] == "street"


class TestSchemaInfo:
    """Tests for SchemaInfo class."""

    def test_schema_info_creation(self) -> None:
        """Test creating SchemaInfo."""
        schema = {"type": "object"}
        required = [FieldInfo(name="name", field_type="string", required=True)]
        optional = [FieldInfo(name="age", field_type="integer", required=False)]

        info = SchemaInfo(
            schema=schema,
            required_fields=required,
            optional_fields=optional,
            schema_type="object",
        )

        assert info.schema == schema
        assert len(info.required_fields) == 1
        assert len(info.optional_fields) == 1
        assert info.schema_type == "object"

    def test_schema_info_to_dict(self) -> None:
        """Test converting SchemaInfo to dictionary."""
        schema = {"type": "object"}
        required = [FieldInfo(name="id", field_type="string", required=True)]
        optional = [
            FieldInfo(name="name", field_type="string", required=False),
            FieldInfo(name="value", field_type="number", required=False),
        ]

        info = SchemaInfo(
            schema=schema,
            required_fields=required,
            optional_fields=optional,
            schema_type="object",
        )

        result = info.to_dict()

        assert result["schema_type"] == "object"
        assert result["required_count"] == 1
        assert result["optional_count"] == 2
        assert result["total_fields"] == 3
        assert len(result["required_fields"]) == 1
        assert len(result["optional_fields"]) == 2

    def test_schema_info_all_fields_property(self) -> None:
        """Test all_fields property returns combined fields."""
        required = [FieldInfo(name="a", field_type="string", required=True)]
        optional = [FieldInfo(name="b", field_type="string", required=False)]

        info = SchemaInfo(
            schema={},
            required_fields=required,
            optional_fields=optional,
            schema_type="object",
        )

        all_fields = info.all_fields
        assert len(all_fields) == 2
        assert all_fields[0].name == "a"
        assert all_fields[1].name == "b"


class TestSchemaValidatorBasicValidation:
    """Tests for basic schema validation."""

    def test_validate_simple_object_schema(self, validator: SchemaValidator) -> None:
        """Test validating a simple object schema."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

        result = validator.validate(schema)

        assert result.schema_type == "object"
        assert len(result.required_fields) == 1
        assert len(result.optional_fields) == 1
        assert result.required_fields[0].name == "name"
        assert result.optional_fields[0].name == "age"

    def test_validate_schema_with_no_required_fields(
        self, validator: SchemaValidator
    ) -> None:
        """Test schema with no required fields."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "optional_field": {"type": "string"},
            },
        }

        result = validator.validate(schema)

        assert len(result.required_fields) == 0
        assert len(result.optional_fields) == 1

    def test_validate_schema_with_all_required_fields(
        self, validator: SchemaValidator
    ) -> None:
        """Test schema where all fields are required."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"},
            },
            "required": ["field1", "field2"],
        }

        result = validator.validate(schema)

        assert len(result.required_fields) == 2
        assert len(result.optional_fields) == 0


class TestSchemaValidatorDataTypes:
    """Tests for supported data types."""

    def test_string_type(self, validator: SchemaValidator) -> None:
        """Test string type is supported."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].field_type == "string"

    def test_number_type(self, validator: SchemaValidator) -> None:
        """Test number type is supported."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"amount": {"type": "number"}},
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].field_type == "number"

    def test_integer_type(self, validator: SchemaValidator) -> None:
        """Test integer type is supported."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].field_type == "integer"

    def test_boolean_type(self, validator: SchemaValidator) -> None:
        """Test boolean type is supported."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].field_type == "boolean"

    def test_object_type(self, validator: SchemaValidator) -> None:
        """Test object type is supported."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"metadata": {"type": "object"}},
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].field_type == "object"

    def test_array_type(self, validator: SchemaValidator) -> None:
        """Test array type is supported."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"items": {"type": "array"}},
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].field_type == "array"

    def test_null_type(self, validator: SchemaValidator) -> None:
        """Test null type is supported."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"empty": {"type": "null"}},
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].field_type == "null"

    def test_type_array_union(self, validator: SchemaValidator) -> None:
        """Test type array (union types) is supported."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"nullable_string": {"type": ["string", "null"]}},
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].field_type == ["string", "null"]


class TestSchemaValidatorNestedStructures:
    """Tests for nested schema structures."""

    def test_nested_object(self, validator: SchemaValidator) -> None:
        """Test nested object structures."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                }
            },
            "required": ["address"],
        }

        result = validator.validate(schema)

        assert len(result.required_fields) == 1
        address_field = result.required_fields[0]
        assert address_field.name == "address"
        assert len(address_field.nested_fields) == 3

        # Find the city field in nested fields
        city_field = next(
            (f for f in address_field.nested_fields if f.name == "city"), None
        )
        assert city_field is not None
        assert city_field.required is True

    def test_deeply_nested_objects(self, validator: SchemaValidator) -> None:
        """Test deeply nested object structures."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {"type": "string"},
                            },
                        }
                    },
                }
            },
        }

        result = validator.validate(schema)

        level1 = result.optional_fields[0]
        assert level1.name == "level1"
        assert level1.path == "level1"

        level2 = level1.nested_fields[0]
        assert level2.name == "level2"
        assert level2.path == "level1.level2"

        level3 = level2.nested_fields[0]
        assert level3.name == "level3"
        assert level3.path == "level1.level2.level3"

    def test_array_of_objects(self, validator: SchemaValidator) -> None:
        """Test array containing object items."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                        "required": ["id"],
                    },
                }
            },
        }

        result = validator.validate(schema)

        users_field = result.optional_fields[0]
        assert users_field.name == "users"
        assert users_field.field_type == "array"
        assert len(users_field.nested_fields) == 2

        # Check nested field paths include array notation
        id_field = next((f for f in users_field.nested_fields if f.name == "id"), None)
        assert id_field is not None
        assert id_field.path == "users[].id"
        assert id_field.required is True


class TestSchemaValidatorInvalidSchemas:
    """Tests for invalid schema handling."""

    def test_non_dict_schema_raises_error(self, validator: SchemaValidator) -> None:
        """Test that non-dict schemas raise error."""
        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate("not a schema")  # type: ignore[arg-type]

        assert "must be a JSON object" in str(exc_info.value)
        assert len(exc_info.value.errors) > 0

    def test_invalid_type_value_raises_error(self, validator: SchemaValidator) -> None:
        """Test that invalid type values raise error."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "field": {"type": "invalid_type"},
            },
        }

        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate(schema)

        # The error can come from jsonschema library or our custom check
        error_text = exc_info.value.errors[0]
        assert "invalid_type" in error_text or "Unsupported type" in error_text

    def test_invalid_nested_type_raises_error(self, validator: SchemaValidator) -> None:
        """Test that invalid types in nested schemas raise error."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "not_a_type"},
                    },
                }
            },
        }

        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate(schema)

        # The error can come from jsonschema library or our custom check
        error_text = exc_info.value.errors[0]
        assert "not_a_type" in error_text or "Unsupported type" in error_text

    def test_malformed_schema_structure(self, validator: SchemaValidator) -> None:
        """Test handling of malformed schema structure."""
        # Schema with invalid 'required' type
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": "name",  # Should be an array
        }

        with pytest.raises(SchemaValidationError):
            validator.validate(schema)


class TestSchemaValidatorFieldDescriptions:
    """Tests for field description extraction."""

    def test_extracts_field_descriptions(self, validator: SchemaValidator) -> None:
        """Test that field descriptions are extracted."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "User's email address",
                },
                "age": {
                    "type": "integer",
                    "description": "User's age in years",
                },
            },
        }

        result = validator.validate(schema)

        email_field = next(f for f in result.all_fields if f.name == "email")
        assert email_field.description == "User's email address"

        age_field = next(f for f in result.all_fields if f.name == "age")
        assert age_field.description == "User's age in years"

    def test_missing_description_is_none(self, validator: SchemaValidator) -> None:
        """Test that missing descriptions result in None."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "field_without_desc": {"type": "string"},
            },
        }

        result = validator.validate(schema)
        assert result.optional_fields[0].description is None


class TestSchemaValidatorDraft7Features:
    """Tests for JSON Schema Draft 7+ feature support."""

    def test_validates_draft7_schema(self, validator: SchemaValidator) -> None:
        """Test validation of explicit Draft 7 schema."""
        schema: dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = validator.validate(schema)
        assert result.schema_type == "object"

    def test_validates_draft201909_schema(self, validator: SchemaValidator) -> None:
        """Test validation of Draft 2019-09 schema."""
        schema: dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = validator.validate(schema)
        assert result.schema_type == "object"

    def test_validates_draft202012_schema(self, validator: SchemaValidator) -> None:
        """Test validation of Draft 2020-12 schema."""
        schema: dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = validator.validate(schema)
        assert result.schema_type == "object"

    def test_schema_with_if_then_else(self, validator: SchemaValidator) -> None:
        """Test schema with conditional (if/then/else) keywords."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "value": {"type": "number"},
            },
            "if": {"properties": {"type": {"const": "percentage"}}},
            "then": {
                "properties": {"value": {"maximum": 100}},
            },
        }

        # Should not raise - if/then/else is valid in Draft 7+
        result = validator.validate(schema)
        assert result.schema_type == "object"


class TestSchemaValidatorDataValidation:
    """Tests for validating data against schemas."""

    def test_valid_data_passes_validation(self, validator: SchemaValidator) -> None:
        """Test that valid data passes validation."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        data = {"name": "John", "age": 30}

        is_valid, errors = validator.validate_data_against_schema(data, schema)

        assert is_valid is True
        assert len(errors) == 0

    def test_missing_required_field_fails_validation(
        self, validator: SchemaValidator
    ) -> None:
        """Test that missing required fields fail validation."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        data: dict[str, Any] = {}

        is_valid, errors = validator.validate_data_against_schema(data, schema)

        assert is_valid is False
        assert len(errors) > 0
        assert "name" in errors[0].lower() or "required" in errors[0].lower()

    def test_wrong_type_fails_validation(self, validator: SchemaValidator) -> None:
        """Test that wrong types fail validation."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        data = {"count": "not an integer"}

        is_valid, errors = validator.validate_data_against_schema(data, schema)

        assert is_valid is False
        assert len(errors) > 0

    def test_nested_validation_errors_include_path(
        self, validator: SchemaValidator
    ) -> None:
        """Test that nested validation errors include the path."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                    },
                    "required": ["email"],
                }
            },
        }
        data = {"user": {}}

        is_valid, errors = validator.validate_data_against_schema(data, schema)

        assert is_valid is False
        # Error should reference the nested path
        assert any("user" in e or "email" in e for e in errors)


class TestSchemaValidatorStaticMethods:
    """Tests for static methods."""

    def test_get_supported_types(self) -> None:
        """Test getting list of supported types."""
        types = SchemaValidator.get_supported_types()

        assert "string" in types
        assert "number" in types
        assert "integer" in types
        assert "boolean" in types
        assert "object" in types
        assert "array" in types
        assert "null" in types
        # Should be sorted
        assert types == sorted(types)


class TestSchemaValidatorEdgeCases:
    """Tests for edge cases."""

    def test_empty_properties(self, validator: SchemaValidator) -> None:
        """Test schema with empty properties."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
        }

        result = validator.validate(schema)

        assert len(result.required_fields) == 0
        assert len(result.optional_fields) == 0

    def test_schema_without_properties(self, validator: SchemaValidator) -> None:
        """Test schema without properties key."""
        schema: dict[str, Any] = {
            "type": "object",
        }

        result = validator.validate(schema)

        assert len(result.required_fields) == 0
        assert len(result.optional_fields) == 0

    def test_schema_without_type(self, validator: SchemaValidator) -> None:
        """Test schema without explicit type defaults to object."""
        schema: dict[str, Any] = {
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = validator.validate(schema)

        assert result.schema_type == "object"

    def test_schema_with_definitions(self, validator: SchemaValidator) -> None:
        """Test schema with $defs/definitions."""
        schema: dict[str, Any] = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                    },
                }
            },
            "type": "object",
            "properties": {
                "home": {"$ref": "#/$defs/Address"},
            },
        }

        # Should validate without error (refs are valid)
        result = validator.validate(schema)
        assert result.schema_type == "object"

    def test_schema_with_additional_properties(
        self, validator: SchemaValidator
    ) -> None:
        """Test schema with additionalProperties."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "known": {"type": "string"},
            },
            "additionalProperties": {"type": "number"},
        }

        result = validator.validate(schema)
        assert len(result.optional_fields) == 1

    def test_schema_with_pattern_properties(self, validator: SchemaValidator) -> None:
        """Test schema with patternProperties."""
        schema: dict[str, Any] = {
            "type": "object",
            "patternProperties": {
                "^S_": {"type": "string"},
                "^I_": {"type": "integer"},
            },
        }

        result = validator.validate(schema)
        # patternProperties don't create named fields
        assert len(result.all_fields) == 0

    def test_array_schema_at_root(self, validator: SchemaValidator) -> None:
        """Test schema with array as root type."""
        schema: dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                },
            },
        }

        result = validator.validate(schema)
        assert result.schema_type == "array"

    def test_type_array_at_root(self, validator: SchemaValidator) -> None:
        """Test schema with type array at root level."""
        schema: dict[str, Any] = {
            "type": ["object", "null"],
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = validator.validate(schema)
        # Should use first type from array
        assert result.schema_type == "object"


class TestSchemaValidatorFormatSpec:
    """Tests for format specification extraction."""

    def test_extracts_date_format(self, validator: SchemaValidator) -> None:
        """Test that date format is extracted from schema."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "invoice_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Invoice issue date",
                },
            },
        }

        result = validator.validate(schema)

        date_field = result.optional_fields[0]
        assert date_field.name == "invoice_date"
        assert date_field.format_spec == "date"

    def test_extracts_email_format(self, validator: SchemaValidator) -> None:
        """Test that email format is extracted from schema."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                },
            },
        }

        result = validator.validate(schema)

        email_field = result.optional_fields[0]
        assert email_field.format_spec == "email"

    def test_format_spec_none_when_not_specified(
        self, validator: SchemaValidator
    ) -> None:
        """Test that format_spec is None when not specified in schema."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = validator.validate(schema)

        name_field = result.optional_fields[0]
        assert name_field.format_spec is None

    def test_format_spec_in_to_dict(self, validator: SchemaValidator) -> None:
        """Test that format_spec is included in to_dict output."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "due_date": {
                    "type": "string",
                    "format": "date",
                },
            },
        }

        result = validator.validate(schema)
        field_dict = result.optional_fields[0].to_dict()

        assert "format" in field_dict
        assert field_dict["format"] == "date"

    def test_format_spec_not_in_to_dict_when_none(
        self, validator: SchemaValidator
    ) -> None:
        """Test that format is not in to_dict when not specified."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = validator.validate(schema)
        field_dict = result.optional_fields[0].to_dict()

        assert "format" not in field_dict

    def test_multiple_fields_with_different_formats(
        self, validator: SchemaValidator
    ) -> None:
        """Test schema with multiple format specifications."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "invoice_date": {"type": "string", "format": "date"},
                "email": {"type": "string", "format": "email"},
                "website": {"type": "string", "format": "uri"},
                "name": {"type": "string"},
            },
        }

        result = validator.validate(schema)

        formats = {f.name: f.format_spec for f in result.all_fields}
        assert formats["invoice_date"] == "date"
        assert formats["email"] == "email"
        assert formats["website"] == "uri"
        assert formats["name"] is None
