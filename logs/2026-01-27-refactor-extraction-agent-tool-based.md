# 2026-01-27 Refactor: Tool-Based Extraction Agent

## Summary
- Implemented a tool-based `ExtractionAgent` that uses LangChain's agent
  framework and StructuredTool registration for AnalyzeChart/AnalyzeTable.
- Updated visual document extraction to supply OCR text, layout regions, and
  reading order metadata to the new agent, replacing direct VLM prompts.
- Added tests for tool registration, prompt composition, and integration
  handoff with VisualDocumentExtractionService.

## Notes
- Agent construction follows the official LangChain agent documentation for
  tool-enabled chat models and StructuredTool usage.
- Visual extraction now favors OCR text + layout metadata, invoking VLM tools
  only when region-specific analysis is required.
- Manual `/extract` verification used `ADE_DOCKET_URL=memory://` to avoid Redis,
  but job execution remained pending without a worker or OpenAI credentials.
