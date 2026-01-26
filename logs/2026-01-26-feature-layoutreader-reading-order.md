# Feature: LayoutReader Reading Order Detection

**Date:** 2026-01-26
**Task:** 0037-implement-layoutreader-reading-order
**Author:** Codex (GPT-5)

## Summary

Replaced the LLM-based reading order detector with a LayoutReader-backed implementation, added model configuration, and updated tests and documentation to reflect the new reading order path.

## Changes Made

### Added

- `tests/test_services/test_reading_order_detector_performance.py` - Performance comparison between LayoutReader and simulated legacy LLM
- `ADE_LAYOUTREADER_MODEL` configuration entry in `src/agentic_document_extraction/config.py`

### Updated

- `src/agentic_document_extraction/services/reading_order_detector.py` - LayoutReader inference, box scaling, and heuristic fallback
- `tests/test_services/test_reading_order_detector.py` - New LayoutReader-focused unit coverage
- `README.md` - Added LayoutReader feature + environment variable documentation
- `AGENT.md` - Updated architecture references to LayoutReader
- `pyproject.toml` - Added pytest marker for performance tests

## Manual Verification

- Started Redis on port 6380, Docket worker, and API server
- Submitted `tests/fixtures/sample_documents/sample_coupon_code_form.png` with `tests/fixtures/sample_schemas/sample_coupon_code_form_schema.json` via `POST /extract`
- Job `f38980d5-18ff-400b-b152-f18a53ece66a` completed successfully
- Result returned with `extracted_data=null` (OCR produced no text elements for this fixture), but end-to-end job lifecycle completed successfully

Logs captured:
- `/tmp/ade-api.log`
- `/tmp/ade-worker.log`
- `/tmp/ade-result.json`

## Verification

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest --cov=src --cov-report=term-missing`
