# Feature: AnalyzeTable Tool (Region-Based)

## Summary
- Added shared region cropping and VLM helper utilities for tool-based visual analysis.
- Implemented `AnalyzeTableTool` as a BaseTool with region-id lookup, structured output parsing, and error handling.
- Refactored `AnalyzeChartTool` to use the shared region utilities and updated tests accordingly.

## Testing
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest --cov=src --cov-report=term-missing`

## Manual Verification
- Started Redis via Docker: `docker run -d --name agentic-document-redis -p 6379:6379 redis`.
- Started Docket worker: `uv run docket worker --tasks agentic_document_extraction.docket_tasks:tasks --docket agentic-document-extraction --url redis://localhost:6379/0 --concurrency 1`.
- Started API: `uv run uvicorn agentic_document_extraction.api:app --host 127.0.0.1 --port 8000`.
- Posted `tests/fixtures/sample_documents/sample_invoice.txt` with `tests/fixtures/sample_schemas/invoice_schema.json` to `/extract`, polled `/jobs/{job_id}` until `completed`, and fetched `/jobs/{job_id}/result`.
- Confirmed extraction returned invoice fields and markdown summary; stopped worker/API and removed the Redis container after verification.
