# 2026-01-25 Feature: LangChain Agentic Loop Tools

## Summary
- Added tool-driven orchestration for the plan/execute/verify/refine loop with shared loop state and memory.
- Swapped refinement memory to LangChain chat history and added agent-tool integration coverage.
- Adjusted loop memory handling to avoid tool-call history issues during orchestration.

## Tests
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest --cov=src --cov-report=term-missing`

## Manual Verification
- Ran the FastAPI service locally.
- `POST /extract` with `tests/fixtures/sample_documents/sample_invoice.txt` and `tests/fixtures/sample_schemas/invoice_schema.json` (schema sent as string).
- Polled `GET /jobs/{job_id}/result` and received a completed extraction with structured invoice data and markdown summary.
