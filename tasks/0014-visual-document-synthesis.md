# Visual Document Synthesis

- [x] Visual Document Synthesis
  - As a system, I want to combine region-level extractions into a coherent document-level extraction using LangChain, so that I can produce final output matching the user's schema.
  - **Acceptance Criteria**:
    - Uses LangChain chains to orchestrate synthesis
    - Combines information from all regions respecting reading order
    - Resolves conflicts or redundancies between regions using LLM reasoning
    - Maps combined information to user's JSON schema
    - Validates against schema using `jsonschema`
    - Generates JSON + Markdown output similar to text-based extraction
    - Includes source references (page numbers, region types, coordinates)
    - Metadata includes processing pipeline details
    - Unit tests for synthesis logic
