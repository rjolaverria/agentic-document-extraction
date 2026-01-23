### Fix: Spurious Null Value Warnings in CSV Extraction (2026-01-23)

**Issue**: The verifier reported null values for populated array fields like `employees[].name` and `employees[].email`, causing false positives in quality reports.

**Root Cause**: The verifier's nested field lookup did not handle array paths (e.g., `employees[].name`) and returned `None`, which was treated as a null value.

**Fix Applied**:
1. Updated `_get_nested_value()` in `verifier.py` to resolve array paths and return per-item values for `[]` segments.
2. Adjusted `_check_null_required_fields()` to treat list results correctly, only flagging nulls when any item is actually missing.

**Tests Added**:
- `tests/test_agents/test_verifier.py`:
  - Added `test_verify_array_nested_fields_no_false_nulls` to ensure populated array fields do not trigger null warnings.

**Validation**:
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest`
- `uv run pytest --cov=src --cov-report=term-missing`

**E2E Verification**:
- Started FastAPI service locally via `uvicorn` (programmatic, port 8001)
- `POST /extract` with:
  - Document: `tests/fixtures/sample_documents/sample_data.csv`
  - Schema: `tests/fixtures/sample_schemas/employee_data_schema.json`
- Result: `job_status=completed`, `null_value_issues=0`
