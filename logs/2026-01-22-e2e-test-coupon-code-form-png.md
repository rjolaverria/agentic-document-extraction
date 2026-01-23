### E2E Test: Coupon Code Form PNG Extraction (2026-01-22)

**Test Document**: `tests/fixtures/sample_documents/sample_coupon_code_form.png`
**Schema**: `tests/fixtures/sample_schemas/sample_coupon_code_form_schema.json`

**Result**: ❌ **FAILED**

**Job ID**: `7025ba12-e94d-4b56-bfe1-9fd274026b32`

**Error Message**:
```
VisualExtractionError: Failed to extract text from image: tesseract is not installed or it's not in your PATH. See README file for more information.
```

**Analysis**:
- The image file is correctly routed to the visual extraction pipeline (confirmed by `VisualExtractionError` rather than `TextExtractionError`)
- The failure occurs at the OCR step because `tesseract` is not installed on the system
- Tesseract is a required external dependency for image processing via `pytesseract`

**Root Cause**: Missing system dependency - tesseract-ocr is not installed

**Impact**: All image-based document extraction (PNG, JPG, BMP, TIFF, GIF, WEBP) will fail until tesseract is installed

**Resolution Required**:
1. Install tesseract-ocr system package:
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Verify installation: `tesseract --version`
3. Ensure tesseract is in system PATH

**Documentation Gap**: The README.md mentions `pytesseract` as a dependency but does not explicitly document the requirement to install the `tesseract-ocr` system package separately.
