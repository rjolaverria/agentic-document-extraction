### Feature: Date Format Normalization (2026-01-22)

**Issue**: The `due_date` field was extracted as "February 15, 2024" instead of the expected ISO format "YYYY-MM-DD" per the schema's `format: date` specification.

**Root Cause**: Two issues were identified:
1. The `FieldInfo` class in `schema_validator.py` didn't capture the `format` specification from JSON schemas
2. The `JsonGenerator._clean_value()` method didn't normalize date values to ISO format
3. The `extraction_processor.py` directly used raw extracted data without passing through JSON normalization

**Fix Applied**:

1. **Added `format_spec` to `FieldInfo` class** (`services/schema_validator.py`):
   - Added `format_spec: str | None = None` parameter to capture JSON Schema format specifications
   - Updated `_extract_fields()` to extract format specs from schema properties
   - Updated `to_dict()` to include format in serialization

2. **Added date normalization utilities** (`output/json_generator.py`):
   - Added `DATE_FORMATS` list with common date formats (ISO, US, EU, month names)
   - Added `parse_date_string()` function to parse various date formats
   - Added `normalize_date()` function to convert dates to ISO format (YYYY-MM-DD)
   - Updated `_clean_value()` to normalize dates when `format_spec == "date"`
   - Updated `_prepare_data()` to pass `format_spec` to `_clean_value()`

3. **Integrated normalization into extraction pipeline** (`services/extraction_processor.py`):
   - Added `JsonGenerator` import
   - Added JSON normalization step before saving extraction results
   - This ensures all extracted data goes through format normalization

**Supported Date Formats**:
- ISO: `2024-02-15`, `2024/02/15`
- US: `02/15/2024`, `02-15-2024`
- EU: `15/02/2024`, `15-02-2024`, `15.02.2024`
- Month names: `February 15, 2024`, `Feb 15, 2024`, `15 February 2024`
- Short year: `02/15/24`, `15/02/24`

**Files Changed**:
- `src/agentic_document_extraction/services/schema_validator.py` - Added `format_spec` attribute
- `src/agentic_document_extraction/output/json_generator.py` - Added date normalization utilities
- `src/agentic_document_extraction/services/extraction_processor.py` - Added JSON normalization step

**Tests Added**:
- `tests/test_services/test_schema_validator.py` - 6 new tests for `format_spec` extraction
- `tests/test_output/test_json_generator.py` - 25+ new tests for date parsing and normalization

**Validation**:
- All 830 tests passing (43 new tests added)
- Ruff: All checks passed
- Mypy: No issues found

**E2E Verification**:
Before fix:
- `invoice_date`: "January 15, 2024" (or sometimes normalized)
- `due_date`: "February 15, 2024"

After fix:
- `invoice_date`: "2024-01-15" ✓
- `due_date`: "2024-02-15" ✓
