# BUG: PNG/Image File Support Not Working

- [x] **BUG: PNG/Image File Support Not Working**
  - **Severity**: Critical
  - **Error**: `TextExtractionError: Unsupported file extension for text extraction: .png`
  - **Description**: The API accepts PNG files via the `/extract` endpoint (returns 202 Accepted), but the background processor fails immediately because the text extraction service does not support image files.
  - **Expected Behavior**: Image files should be routed to the visual extraction pipeline (OCR + layout detection + VLM), not the text extraction pipeline.
  - **Root Cause**: The extraction processor was using `TextExtractor` for visual documents instead of `VisualTextExtractor`.
  - **Fix Applied**: Updated `services/extraction_processor.py` to use `VisualTextExtractor` for documents with `ProcessingCategory.VISUAL`
  - **Tests Added**: 12 new tests in `tests/test_services/test_extraction_processor.py` covering all visual and text format routing
