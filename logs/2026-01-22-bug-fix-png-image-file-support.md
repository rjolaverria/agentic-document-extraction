### Bug Fix: PNG/Image File Support (2026-01-22)

**Issue**: PNG and other image files were not being routed to the visual extraction pipeline. The extraction processor was attempting to use the text extractor for all documents, which only supports `.txt` and `.csv` files.

**Error Before Fix**: `TextExtractionError: Unsupported file extension for text extraction: .png`

**Root Cause**: In `services/extraction_processor.py`, the code was using `TextExtractor` for both text-based AND visual documents. The else branch for visual documents was incorrectly calling `text_extractor.extract_from_path()` instead of using the `VisualTextExtractor`.

**Fix Applied**:
1. Imported `VisualTextExtractor` in `extraction_processor.py`
2. Updated the else branch (visual documents) to use `VisualTextExtractor` instead of `TextExtractor`
3. The visual extractor uses OCR (pytesseract) for images and pdfplumber for PDFs

**Files Changed**:
- `src/agentic_document_extraction/services/extraction_processor.py` - Added import and routing logic

**Tests Added**:
- `tests/test_services/test_extraction_processor.py` - 12 new tests covering:
  - PNG files route to visual extractor
  - TXT files route to text extractor
  - All visual formats (png, jpg, jpeg, pdf, bmp, gif, webp, tiff) use visual extractor
  - All text formats (txt, csv) use text extractor

**Validation**:
- All 799 tests passing (12 new tests added)
- Ruff: All checks passed
- Mypy: No issues found
- Coverage: 93%

**E2E Verification**:
After the fix, PNG files are correctly routed to the visual extraction pipeline. The job status now shows `VisualExtractionError` (missing tesseract dependency) instead of `TextExtractionError`, confirming the routing fix works correctly.
