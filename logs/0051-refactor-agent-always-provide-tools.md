# Log 0051: Refactor Agent to Always Provide All Tools

## Date
2026-01-29

## Task Reference
`tasks/0051-refactor-agent-always-provide-tools.md`

## Summary
Removed conditional tool logic from the extraction agent so all 9 tools are always available, allowing the LLM to decide which tools to use based on context.

## Changes Made

### 1. `src/agentic_document_extraction/agents/extraction_agent.py`

**Removed conditional logic:**
```python
# Before
tools: list[Any] = []
if is_visual and has_visual_regions:
    if analyze_chart is not None:
        tools.append(analyze_chart)
    # ... repeated for all 9 tools

# After
tools: list[Any] = []
for tool in [
    analyze_chart,
    analyze_diagram,
    analyze_form,
    analyze_handwriting,
    analyze_image,
    analyze_logo,
    analyze_math,
    analyze_signature,
    analyze_table,
]:
    if tool is not None:
        tools.append(tool)
```

**Removed unused variables and imports:**
- Removed `has_visual_regions` and `is_visual` variables (no longer needed)
- Removed `RegionType` import (only used for conditional check)
- Removed `ProcessingCategory` import (accessed through `format_info.processing_category` instead)

### 2. `tests/test_agents/test_extraction_agent.py`

**Updated test expectations:**
- `test_all_tools_registered_for_text_documents` (was `test_no_tools_for_text_documents`): Now expects 9 tools
- `test_all_tools_registered_for_visual_with_regions` (was `test_tools_registered_for_visual_with_regions`): Unchanged (still expects 9)
- `test_all_tools_registered_visual_without_visual_regions` (was `test_no_tools_visual_without_visual_regions`): Now expects 9 tools
- `test_text_only_document_all_tools_registered` (was `test_text_only_document_no_tools`): Now expects 9 tools
- `test_visual_document_with_chart_region`: Updated assertion to `assert len(tools) == 9`

## Rationale

Per LangGraph best practices:
1. All tools should be registered with the agent
2. The agent/LLM uses context to decide which tools are relevant
3. Tool descriptions serve as the primary mechanism for tool selection guidance
4. The region metadata table in the prompt provides context for tool selection

## Test Results

- All 1144 tests pass
- 87% code coverage
- No linting or type errors

## Verification

The changes were verified through:
1. `uv run ruff check .` - All checks passed
2. `uv run ruff format .` - No changes needed
3. `uv run mypy src` - No issues found
4. `uv run pytest --cov=src --cov-report=term-missing` - 1144 tests passed

## Notes

- The `if tool is not None` checks are kept to handle potential import failures gracefully
- This is the foundational task for the tool-based agent refactoring series (0051-0056)
- Tool descriptions (Task 0053) and system prompt updates (Task 0054) become more important now that all tools are always available
