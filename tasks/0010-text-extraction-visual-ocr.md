# Text Extraction from Visual Documents (OCR)

- [x] Text Extraction from Visual Documents (OCR)
  - As a system, I want to extract text from PDFs and images using OCR, so that I have the raw textual content before visual analysis.
  - **Acceptance Criteria**:
    - Extracts text from PDF documents using `pypdf` or `pdfplumber` (multi-page support)
    - Extracts text from image formats using OCR (`paddleocr` or cloud OCR)
    - Returns text with bounding box coordinates for each text element
    - Preserves page/image boundaries
    - Returns confidence scores for OCR results
    - Handles multi-page documents (PDFs with multiple pages, multi-page TIFFs)
    - Unit tests with sample PDFs and images
    - Performance considerations for large documents
