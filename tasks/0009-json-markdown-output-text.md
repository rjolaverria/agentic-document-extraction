# JSON + Markdown Output (Text Documents)

- [x] JSON + Markdown Output (Text Documents)
  - As a user, I want extracted information from text documents returned as valid JSON and Markdown via API response, so that I can use the results programmatically and review them easily.
  - **Acceptance Criteria**:
    - Maps extracted data to user-provided JSON schema
    - Validates output against schema before returning using `jsonschema`
    - Returns validation errors if extraction doesn't match schema (with retry logic)
    - API response includes both JSON and Markdown in structured format
    - Generates formatted Markdown summary using LangChain
    - Handles missing or uncertain data gracefully (null values, confidence indicators)
    - Includes source references (line numbers, sections) in Markdown
    - Response includes metadata (processing time, model used, token usage)
    - Unit tests for output generation and validation
