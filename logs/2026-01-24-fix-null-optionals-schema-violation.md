# Log: Fix Null Optional Fields Schema Violation

**Date**: 2026-01-24
**Task**: 0032-issue-null-optionals-schema-violation
**Status**: Completed

## Problem

When extracting data from documents, optional string fields that couldn't be extracted were returned as `null` values. The JSON schema validation then flagged these as schema violations because the schema types were defined as just `string` without allowing `null`.

### Example

Schema:
```json
{
  "properties": {
    "signed_by": { "type": "string" },
    "media_name": { "type": "string" }
  },
  "required": []
}
```

Extracted data:
```json
{
  "signed_by": null,
  "media_name": null
}
```

Result: `schema_violation` issues reported for both fields because `null` is not a valid `string`.

### Root Cause

The verifier's `_validate_against_schema()` method validated the extracted data directly against the JSON schema. When the LLM couldn't extract a value for an optional field, it returned `null`, which caused jsonschema validation to fail.

## Solution

Added a `_strip_null_optional_fields()` helper method in the verifier that:
1. Identifies optional fields from the schema_info
2. Creates a copy of the extracted data
3. Removes keys that are optional and have null values
4. Uses the cleaned data only for schema validation

This ensures that:
- Null optional fields don't trigger spurious schema violations
- Null required fields still fail validation (as expected)
- Metrics computation still sees the original data (for accurate optional_field_coverage)
- Original data is preserved (no mutation)

## Key Changes

### `src/agentic_document_extraction/agents/verifier.py`

1. Added `_strip_null_optional_fields()` method:
   - Takes extracted data and schema_info
   - Returns a copy with null optional fields removed
   - Uses deep copy to avoid modifying original

2. Modified `_validate_against_schema()`:
   - Calls `_strip_null_optional_fields()` before validation
   - Uses cleaned data for jsonschema.validate()

### `tests/test_agents/test_verifier.py`

Added `TestNullOptionalFieldsHandling` test class with 6 tests:
- `test_null_optional_fields_stripped_before_validation`
- `test_null_required_fields_still_fail_validation`
- `test_strip_null_optional_preserves_non_null_values`
- `test_strip_null_optional_helper_method`
- `test_metrics_still_count_null_optional_fields`
- `test_coupon_form_scenario` - Tests exact reproduction scenario

## Files Changed

- `src/agentic_document_extraction/agents/verifier.py` (modified)
- `tests/test_agents/test_verifier.py` (modified)

## Testing

- All 871 tests pass (864 passed, 7 skipped)
- Ruff check: All checks passed
- Ruff format: Clean
- Mypy: No issues found in 29 source files

## Notes

- The fix uses deep copy to ensure the original extracted data is not modified
- Metrics like `optional_field_coverage` still correctly reflect null fields
- This approach allows schemas to remain strict (no need to add `null` to type definitions)
