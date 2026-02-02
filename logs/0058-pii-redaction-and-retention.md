# 2026-02-02 Feature: PII Redaction & Retention Controls

## Summary
- Added PIIRedactor service with regex-based detection, field-level policies (`allow`, `mask`, `hash`, `drop`), and output-wide redaction flag on the API.
- Wired redaction through extraction pipeline, schema validation, and markdown generation; metrics logged for masked/hashed/dropped items.
- Introduced retention purge task and configurable retention settings with optional auto-deletion of uploads.
- Documented new env vars, API flag, and schema `pii` directive.

## Testing
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest --cov=src --cov-report=term-missing`
