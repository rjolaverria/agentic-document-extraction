# Feature: Replace Custom Job Manager with Docket

**Date:** 2026-01-25
**Task:** 0036-replace-job-manager-docket
**Author:** Claude Opus 4.5

## Summary

Completed migration from the custom in-memory job manager to Docket for background job execution and status tracking. Docket provides a standardized async execution framework with Redis-backed persistence.

## Changes Made

### New Files

- `src/agentic_document_extraction/services/docket_client.py` - Factory for creating configured Docket instances
- `src/agentic_document_extraction/services/docket_jobs.py` - Job metadata storage and execution state helpers
- `src/agentic_document_extraction/docket_tasks.py` - Task registry for Docket worker CLI
- `tests/test_services/test_docket_jobs.py` - Tests for Docket job helpers

### Modified Files

- `src/agentic_document_extraction/api.py` - Updated to use Docket for job scheduling and status
- `src/agentic_document_extraction/config.py` - Added Docket configuration settings
- `src/agentic_document_extraction/services/extraction_processor.py` - Integrated Docket Progress for status updates
- `tests/test_api.py` - Updated test fixtures for proper lifespan handling with httpx 0.28+
- `tests/test_services/test_extraction_processor.py` - Updated assertions for JsonGenerator behavior
- `README.md` - Added Docket worker instructions and configuration documentation

### Removed Files

- `src/agentic_document_extraction/services/job_manager.py` - Replaced by Docket
- `tests/test_services/test_job_manager.py` - Tests for removed job manager

## Configuration

New environment variables added:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADE_DOCKET_NAME` | `agentic-document-extraction` | Shared Docket name for API and workers |
| `ADE_DOCKET_URL` | `redis://localhost:6379/0` | Redis backend URL |
| `ADE_DOCKET_RESULT_STORAGE_URL` | `None` | Optional separate Redis for results |
| `ADE_DOCKET_EXECUTION_TTL_SECONDS` | `None` | Override execution TTL (defaults to job_ttl_seconds) |
| `ADE_DOCKET_ENABLE_INTERNAL_INSTRUMENTATION` | `false` | Enable OpenTelemetry spans for Docket internals |

## Running the Worker

To process extraction jobs, run a Docket worker:

```bash
uv run docket worker \
  --tasks agentic_document_extraction.docket_tasks:tasks \
  --docket agentic-document-extraction \
  --url redis://localhost:6379/0 \
  --concurrency 2
```

## Test Fixes

Fixed test compatibility issues with httpx 0.28+:
- Removed deprecated `lifespan="on"` parameter from ASGITransport
- Created `create_test_client` helper that properly manages FastAPI lifespan context
- Updated assertions for JsonGenerator's handling of optional fields

## Verification

- All 848 tests passing
- ruff check clean
- ruff format clean
- mypy strict mode passing
- Manual API verification successful with memory:// backend

## Notes

- The `memory://` backend is suitable for single-process development/testing
- Production deployments require Redis for job persistence across API and worker processes
- Progress messages are updated during extraction and visible via GET /jobs/{job_id}
