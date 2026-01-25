# Log: Replace Tesseract with PaddleOCR-VL

**Date**: 2026-01-25
**Task**: 0035-replace-tesseract-paddleocr-vl
**Status**: Completed

## Summary

Replaced the Tesseract OCR integration with PaddleOCR-VL, added configurable runtime/model options, and updated tests and documentation to match the new OCR engine.

## Key Changes

- Implemented PaddleOCR-VL OCR flow and bounding box normalization in `src/agentic_document_extraction/services/visual_text_extractor.py`.
- Added PaddleOCR configuration settings in `src/agentic_document_extraction/config.py`.
- Updated dependencies and mypy overrides in `pyproject.toml`.
- Updated OCR-related tests and planner strategy expectations in `tests/`.
- Updated documentation references in `README.md` and `AGENT.md`.

## Testing

- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `ADE_SKIP_REAL_API_TESTS=1 uv run pytest --cov=src --cov-report=term-missing` (skipped real API tests due to OpenAI quota)
- Manual service run: `sample_coupon_code_form.png` + `sample_coupon_code_form_schema.json` reached LLM stage but failed with OpenAI quota (429)
