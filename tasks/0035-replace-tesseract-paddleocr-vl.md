# Replace Tesseract with PaddleOCR-VL

- [x] Replace Tesseract with PaddleOCR-VL
  - As a developer, I want to replace Tesseract-based OCR with PaddleOCR-VL, so that visual text extraction is more accurate and consistent across document types.
  - **Acceptance Criteria**:
    - Remove Tesseract usage in visual OCR flows and replace with PaddleOCR-VL
    - Update dependencies to include PaddleOCR-VL and remove Tesseract-specific tooling
    - Ensure `text_extractor` and visual document pipeline use PaddleOCR-VL outputs
    - Update configuration to support PaddleOCR-VL model selection and runtime options
    - Add or update tests to validate OCR extraction using PaddleOCR-VL
    - Update documentation and logs to reflect the OCR engine change
