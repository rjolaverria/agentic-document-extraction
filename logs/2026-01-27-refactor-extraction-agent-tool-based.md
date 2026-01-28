# Refactor: Extraction Agent to Tool-Based Architecture

## Summary
Completed the ExtractionAgent implementation (task 0040) by adding exception handling,
integration tests, and extraction processor tool-agent path tests.

### Changes
- **Exception handling**: Wrapped `agent.invoke()` in try/except, catching any exception
  and re-raising as `DocumentProcessingError` with logged context. This prevents
  unhandled LangChain errors from crashing the pipeline.
- **Error handling tests**: Added `TestExtractErrorHandling` with 2 tests verifying that
  invoke failures raise `DocumentProcessingError` and log the error.
- **Integration tests**: Added `TestExtractionAgentIntegration` with 5 tests covering
  text-only documents, visual documents with chart/table/mixed regions, and empty
  extraction results.
- **Processor tests**: Added `TestExtractionProcessorToolAgent` with 3 tests verifying
  layout detection is called for visual docs, layout detection failure degrades gracefully,
  and text docs skip layout detection entirely.

### Performance Comparison vs Multi-Agent Approach
The tool-based architecture replaces a multi-agent orchestration loop (planner + extractor +
verifier + refiner) with a single `ExtractionAgent` that uses LangChain's `create_agent`
with `response_format` for structured output. Key differences:

| Aspect | Multi-Agent Loop | Tool-Based Agent |
|--------|-----------------|------------------|
| LLM calls per doc | 3-6+ (plan + extract + verify + refine) | 1-3 (extract + optional tool calls) |
| Agent coordination | Custom Python loop | LangChain agent runtime |
| Tool usage | Implicit (all regions processed) | Selective (agent decides) |
| Error recovery | Refinement iterations | Single-pass with error handling |
| Code complexity | ~4 agent classes | 1 agent class + 2 tools |

The new architecture reduces API costs by eliminating unnecessary LLM calls for planning
and verification while maintaining extraction quality through structured output and
selective tool usage.

## Testing
```
uv run pytest tests/test_agents/test_extraction_agent.py -v  # 28 passed
uv run pytest tests/test_services/test_extraction_processor.py -v  # 15 passed
uv run pytest --tb=short  # 885 passed
uv run ruff check .
uv run ruff format .
uv run mypy src/agentic_document_extraction/agents/extraction_agent.py
```
