### Fix: Phone Number Normalization (2026-01-23)

**Issue**: Extracted phone numbers appeared in mixed formats (e.g., "(555) 123-4567") without consistent normalization.

**Root Cause**: The JSON output normalization step did not apply any phone-specific formatting, and nested phone fields were left untouched during post-processing.

**Fix Applied**:

1. **Added phone normalization utilities** (`output/json_generator.py`):
   - Introduced `normalize_phone()` to standardize numbers to a best-effort E.164 format.
   - Added phone field detection using common key hints (phone, telephone, tel, mobile, cell).

2. **Applied normalization across nested data** (`output/json_generator.py`):
   - Added recursive normalization to walk nested objects and arrays.
   - Applied phone normalization after schema-based cleaning in JSON generation.

**Files Changed**:
- `src/agentic_document_extraction/output/json_generator.py` - Phone normalization helpers and recursive application
- `tests/test_output/test_json_generator.py` - Added phone normalization unit tests

**Tests Added**:
- `tests/test_output/test_json_generator.py` - Phone normalization helper tests and nested field handling

**Validation**:
- Ruff: `uv run ruff check .`
- Ruff format: `uv run ruff format .`
- Mypy: `uv run mypy src`
- Pytest: `uv run pytest`
- Coverage: `uv run pytest --cov=src --cov-report=term-missing`

**E2E Verification**:
- Ran the service locally and verified extraction output normalizes phone numbers in nested contact objects using sample fixtures.
