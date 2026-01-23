# Region-Based Visual Extraction with VLM

- [x] Region-Based Visual Extraction with VLM
  - As a system, I want to send individual layout regions to OpenAI's vision models for detailed analysis, so that I can extract information from complex visual elements.
  - **Acceptance Criteria**:
    - Integrates LangChain with OpenAI GPT-4V (Vision)
    - Sends segmented region images to VLM
    - Provides context for each region (type, position, surrounding regions, reading order)
    - Extracts structured information from tables, charts, figures using vision capabilities
    - Handles text-heavy regions with combined OCR + VLM analysis
    - Returns extraction results per region with confidence scores
    - Preserves spatial relationships between regions
    - Uses reasoning capabilities when analyzing complex visual structures
    - Batch processing for efficiency
    - Unit tests with mocked VLM responses
    - Integration tests with sample visual elements
