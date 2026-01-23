# ISSUE: Tesseract OCR Not Installed - Image Extraction Fails

- [x] **ISSUE: Tesseract OCR Not Installed - Image Extraction Fails**
  - **Severity**: Critical (blocks all image processing)
  - **Discovered**: 2026-01-22 via E2E test with `sample_coupon_code_form.png`
  - **Error**: `VisualExtractionError: Failed to extract text from image: tesseract is not installed or it's not in your PATH.`
  - **Description**: The visual extraction pipeline requires `tesseract-ocr` to be installed as a system dependency. The Python `pytesseract` package is a wrapper that calls the tesseract binary. Without it, all image-based document extraction (PNG, JPG, BMP, TIFF, GIF, WEBP) will fail.
  - **Impact**: Complete failure of image document processing
  - **Resolution Applied**:
    1. Installed tesseract-ocr via Homebrew (tesseract 5.5.2)
    2. Updated README.md with installation instructions for macOS, Ubuntu/Debian, and Windows
    3. Verified image extraction works with E2E test on `sample_coupon_code_form.png`
