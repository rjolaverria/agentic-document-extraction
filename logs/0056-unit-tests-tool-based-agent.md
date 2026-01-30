# Log: Unit Tests for Tool-Based Agent

**Date:** 2026-01-30  
**Task:** 0056 - Add Unit Tests for Tool-Based Agent

## Summary

Added comprehensive fixtures and tool-selection tests to validate the refactored extraction agent that always exposes all nine visual tools. The new tests cover text-only guidance, region-driven tool availability, mixed-region handling, response_format title sanitization, and tool failure surfacing.

## Changes Made

- Created shared pytest fixtures for schemas, format metadata, and representative layout regions (text-only, chart, table, mixed) in `tests/conftest.py`.
- Added `tests/test_agents/test_tool_selection.py` with focused cases:
  - Ensures text-only documents pass empty regions and include skip-tool guidance.
  - Verifies TABLE/PICTURE/mixed regions are surfaced to the agent and all tool names are registered.
  - Confirms schema titles are sanitized before passing to `response_format`.
  - Asserts tool/agent failures raise `DocumentProcessingError`.
- Marked Task 0056 complete in `TASKS.md`.

## Verification

- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest --cov=src --cov-report=term-missing` (1152 passed; `extraction_agent.py` 96% coverage)
