# Document Format Detection & Classification

- [x] Document Format Detection & Classification
  - As a system, I want to automatically detect and classify document format from uploaded files, so that the system knows whether to process it as text-based or visual.
  - **Acceptance Criteria**:
    - Detects format from file extension and/or MIME type
    - Supports: txt, pdf, doc, docx, odt, ppt, pptx, odp, csv, xlsx, jpeg, png, bmp, psd, tiff, gif, webp
    - Classifies documents into two categories:
      - **Text-based**: txt, csv
      - **Visual**: pdf, doc, docx, odt, ppt, pptx, odp, xlsx, jpeg, png, bmp, psd, tiff, gif, webp
    - Returns structured format information (MIME type, extension, format family, processing category)
    - Handles edge cases (missing extensions, incorrect extensions using magic bytes)
    - Uses `python-magic` or similar for robust detection
    - Unit tests for all supported formats
