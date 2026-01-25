# 2026-01-24 Feature: LangChain Agent Abstraction

## Summary
- Added a shared LangChain agent helper and refactored extraction, planning, and output services to invoke agents instead of direct LLM calls.
- Updated tests to mock agent invocation and supply usage metadata expectations.
- Escaped JSON braces in the reading-order system prompt to avoid template formatting errors.

## Tests
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest`

## Manual Verification
- Ran the FastAPI service locally and validated `/extract` and job result retrieval using `tests/fixtures/sample_documents/sample_resume.txt` and `tests/fixtures/sample_schemas/resume_schema.json`.

## Notes
- Coverage run omitted per user request.
