# Reading Order Detection with LLM

- [x] Reading Order Detection with LLM
  - As a system, I want to use GPT models to detect the reading order of text elements and layout regions, so that I understand the logical flow of the document.
  - **Acceptance Criteria**:
    - Takes text elements/regions with bounding box coordinates as input
    - Uses LangChain with OpenAI GPT-4 to analyze spatial relationships
    - Prompts model with coordinates and text snippets to determine reading order
    - Returns ordered sequence of elements/regions
    - Handles complex layouts (multi-column, sidebar, headers/footers)
    - Provides confidence scores for reading order decisions
    - Works across multi-page documents
    - Unit tests with mocked LLM responses
    - Integration tests with various layout patterns
