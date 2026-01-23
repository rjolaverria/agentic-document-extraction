### Fix: Tesseract OCR Documentation and Verification (2026-01-22)

**Issue**: Visual extraction pipeline failed with "tesseract is not installed or it's not in your PATH" error for all image-based documents.

**Root Cause**: The `pytesseract` Python package is a wrapper that calls the external `tesseract` binary. Without the system-level tesseract-ocr installation, all image processing (PNG, JPG, BMP, TIFF, GIF, WEBP) fails.

**Fix Applied**:

1. **Installed tesseract-ocr on the system**:
   ```bash
   brew install tesseract
   ```
   Verified with `tesseract --version` → tesseract 5.5.2

2. **Updated README.md** to document tesseract as a prerequisite:
   - Added Tesseract OCR to Prerequisites section
   - Added installation instructions for macOS, Ubuntu/Debian, and Windows
   - Added verification command
   - Added note about language data (eng, osd, snum default; tesseract-lang for additional languages)

**Files Changed**:
- `README.md` - Added Tesseract OCR prerequisite documentation

**E2E Verification**:
Successfully extracted data from `sample_coupon_code_form.png`:
- Job status: **completed** ✓
- Document type: **visual** ✓
- Processing time: 58 seconds
- Extracted fields: form_title, coupon_issue_date, coupon_expiration_date, circulation, coupon_value
- Date normalization working: "10/4/99" → "1999-10-04", "3/31/00" → "2000-03-31"

**Validation**:
- All 830 tests passing
- Ruff: All checks passed
- Mypy: No issues found
- Coverage: 93%
