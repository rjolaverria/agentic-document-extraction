# Log: Tool Instructions in System Prompt for All Documents

**Date:** 2026-01-30
**Task:** 0054 - Add Tool Usage to System Prompt for All Documents

## Summary

Updated the extraction agent's system prompt to always include tool instructions, with guidance on when to use and when NOT to use tools. This completes the tool-based architecture refactoring started in Tasks 0051-0053.

## Changes Made

### 1. Updated Tool Instructions Constant

Renamed `_TOOL_INSTRUCTIONS_VISUAL` to `_TOOL_INSTRUCTIONS` and rewrote it to:
- Explain that tools are available but optional based on document content
- Provide explicit "When to use tools" guidance (PICTURE/TABLE regions, insufficient OCR)
- Provide explicit "When NOT to use tools" guidance (OCR text sufficient, no visual regions)
- List all 9 available tools with descriptions
- Clarify how to use tools with `region_id`

### 2. Simplified Prompt Building Methods

Updated `_build_system_prompt()` and `_build_refinement_prompt()`:
- Removed `has_tools` parameter (no longer needed since tools are always provided)
- Tool instructions are now always included in the prompt
- Agent decides when to use tools based on document content

### 3. Updated Tests

- Updated test imports to remove unused `_TOOL_INSTRUCTIONS_VISUAL` reference
- Updated tests to verify tool instructions ARE included (not that they aren't)
- Added new tests for tool instruction content:
  - `test_tool_instructions_include_all_tools` - verifies all 9 tools documented
  - `test_tool_instructions_include_skip_guidance` - verifies "when NOT to use" guidance

## Key Decision

The tool instructions are included for ALL documents (text and visual) because:
1. Tools are always provided (per Task 0051)
2. The agent needs to understand tool capabilities to make informed decisions
3. The "When NOT to use tools" guidance prevents unnecessary VLM calls for text documents

## Files Modified

- `src/agentic_document_extraction/agents/extraction_agent.py`
  - Renamed `_TOOL_INSTRUCTIONS_VISUAL` to `_TOOL_INSTRUCTIONS`
  - Updated tool instructions content with use/skip guidance
  - Simplified `_build_system_prompt()` and `_build_refinement_prompt()`

- `tests/test_agents/test_extraction_agent.py`
  - Removed unused import
  - Updated tests for new prompt structure
  - Added tests for tool instruction content

## Verification

- `uv run ruff check .` - All checks passed
- `uv run ruff format .` - No changes needed
- `uv run mypy src` - No issues found
- `uv run pytest --cov=src` - 1146 tests passed, 87% coverage
