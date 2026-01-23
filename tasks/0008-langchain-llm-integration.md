# LangChain LLM Integration for Text Extraction

- [x] LangChain LLM Integration for Text Extraction
  - As a system, I want to use LangChain with OpenAI models to extract structured information from text documents according to the schema.
  - **Acceptance Criteria**:
    - Integrates `langchain-openai` with GPT-4 or GPT-4 Turbo
    - Configures OpenAI API key from environment variables
    - Creates LangChain chain for extraction task
    - Uses structured output (JSON mode or function calling) to enforce schema compliance
    - Sends extracted text with JSON schema as context
    - Extracts information matching user's JSON schema
    - Returns extraction with confidence indicators where applicable
    - Handles documents that exceed token limits (chunking strategy with LangChain text splitters)
    - Includes reasoning in extraction process (using reasoning models when available)
    - Unit tests with mocked OpenAI responses
    - Integration tests with real API (marked as optional/skippable)
