# Text Extraction from Text-Based Documents

- [x] Text Extraction from Text-Based Documents
  - As a system, I want to extract raw text from text-based documents, so that I can process them directly without visual conversion.
  - **Acceptance Criteria**:
    - Extracts text from .txt files preserving line breaks and structure
    - Detects and handles different encodings (UTF-8, Latin-1, etc.) using `chardet`
    - Extracts text from .csv files preserving tabular structure using `pandas`
    - Returns extracted text with metadata (encoding, line count, structure type)
    - Preserves natural reading order
    - Unit tests with various encodings and structures
