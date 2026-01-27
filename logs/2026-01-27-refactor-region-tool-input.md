# Refactor: Region-Input Tool Interface

## Summary
- Reused `LayoutRegion` for tool inputs and added `RegionImage` dataclass to carry optional image/base64 payloads.
- Converted AnalyzeTable and AnalyzeChart to function-based `@tool` wrappers using region lists and region images.
- Updated tool exports and tests to use the new interface.

## Testing
- `uv run ruff format src/agentic_document_extraction/agents/tools/analyze_chart.py src/agentic_document_extraction/agents/tools/analyze_table.py src/agentic_document_extraction/services/layout_detector.py tests/test_agents/test_analyze_chart_tool.py tests/test_agents/test_analyze_table_tool.py`
- `uv run ruff check src/agentic_document_extraction/agents/tools/analyze_chart.py src/agentic_document_extraction/agents/tools/analyze_table.py src/agentic_document_extraction/services/layout_detector.py tests/test_agents/test_analyze_chart_tool.py tests/test_agents/test_analyze_table_tool.py`
- `uv run mypy src/agentic_document_extraction/services/layout_detector.py src/agentic_document_extraction/agents/tools/analyze_chart.py src/agentic_document_extraction/agents/tools/analyze_table.py src/agentic_document_extraction/agents/__init__.py src/agentic_document_extraction/agents/tools/__init__.py`
- `uv run pytest tests/test_agents/test_analyze_chart_tool.py tests/test_agents/test_analyze_table_tool.py`
