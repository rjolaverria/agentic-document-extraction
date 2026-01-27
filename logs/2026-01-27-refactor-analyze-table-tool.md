# Refactor: AnalyzeTable Tool to Function-Based Input

## Summary
- Reworked AnalyzeTable into a function-based `@tool` implementation that accepts a region list with optional region images.
- Added RegionImage/TableRegion input models to pass `image` or `base64` per region.
- Updated exports and tests to match the new tool interface.

## References
- Reviewed LangChain tool decorator documentation for `@tool` and `args_schema` usage.

## Testing
- `uv run ruff format src/agentic_document_extraction/agents/tools/analyze_table.py tests/test_agents/test_analyze_table_tool.py`
- `uv run ruff check src/agentic_document_extraction/agents/tools/analyze_table.py tests/test_agents/test_analyze_table_tool.py`
- `uv run mypy src/agentic_document_extraction/agents/tools/analyze_table.py src/agentic_document_extraction/agents/__init__.py src/agentic_document_extraction/agents/tools/__init__.py`
- `uv run pytest tests/test_agents/test_analyze_table_tool.py`
