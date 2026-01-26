# Feature: AnalyzeChart Tool (Function + @tool)

## Summary
- Replaced the AnalyzeChart implementation with a function-based LangChain tool that accepts a base64 image input.
- Aligned the chart analysis prompt and JSON schema with the reference notebook pattern.
- Updated exports and tests to use the new tool interface.

## Testing
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest --cov=src --cov-report=term-missing`

## Manual Verification
- Started Redis via Docker: `docker run --name ade-redis -p 6379:6379 -d redis` (existing container reused after recreation).
- Started Docket worker: `DOCKET_URL=redis://localhost:6379/0 DOCKET_NAME=agentic-document-extraction DOCKET_TASKS=agentic_document_extraction.docket_tasks:tasks uv run docket worker`.
- Started API: `uv run uvicorn agentic_document_extraction.api:app --port 8001`.
- Posted `tests/fixtures/sample_documents/sample_invoice.txt` with `tests/fixtures/sample_schemas/invoice_schema.json` to `/extract` and polled `/jobs/{job_id}` until `completed`.
- Retrieved `/jobs/{job_id}/result` and confirmed extracted invoice fields (invoice number, dates, totals, vendor, bill-to, line items).
- Stopped worker/API and Redis container after verification.
