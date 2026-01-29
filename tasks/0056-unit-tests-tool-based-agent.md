# Task 0056: Add Unit Tests for Tool-Based Agent with All Tools

## Objective
Create comprehensive unit tests to validate the refactored extraction agent that always provides all tools to the LLM.

## Context
After completing Tasks 0051-0054, the extraction agent will:
1. Always provide all 9 tools regardless of document type
2. Have simplified tool definitions (no try/except ImportError)
3. Have improved tool descriptions for autonomous selection
4. Always include tool instructions in the system prompt

We need tests to ensure:
- The agent receives all tools
- The agent correctly decides when to use/skip tools
- Existing extraction behavior is preserved (no regression)
- Edge cases are handled properly

## Acceptance Criteria
- [ ] Test agent receives all 9 tools for any document type
- [ ] Test agent skips tool usage for text-only documents
- [ ] Test agent uses appropriate tools for visual documents
- [ ] Test agent handles mixed document types
- [ ] Test backward compatibility with existing extraction behavior
- [ ] Test tool selection logic based on region types
- [ ] Test error handling when tools fail
- [ ] All tests pass with >90% coverage on extraction_agent.py

## Test Cases

### 1. Tool Registration Tests
```python
def test_agent_receives_all_tools_for_text_document():
    """Agent should have all 9 tools even for text-only documents."""
    ...

def test_agent_receives_all_tools_for_visual_document():
    """Agent should have all 9 tools for visual documents."""
    ...

def test_all_tools_are_not_none():
    """All tool imports should succeed (no None values)."""
    ...
```

### 2. Tool Selection Tests
```python
def test_agent_skips_tools_for_simple_text():
    """Agent should not call tools when OCR text is sufficient."""
    ...

def test_agent_calls_analyze_chart_for_chart_region():
    """Agent should call analyze_chart for PICTURE regions with charts."""
    ...

def test_agent_calls_analyze_table_for_table_region():
    """Agent should call analyze_table for TABLE regions."""
    ...

def test_agent_selects_correct_tool_for_region_type():
    """Agent should select the most appropriate tool based on region type."""
    ...
```

### 3. Integration Tests
```python
def test_extraction_text_only_document():
    """Full extraction pipeline for text-only document."""
    ...

def test_extraction_visual_document_with_chart():
    """Full extraction pipeline for document with chart."""
    ...

def test_extraction_mixed_document():
    """Full extraction pipeline for document with multiple region types."""
    ...
```

### 4. Regression Tests
```python
def test_invoice_extraction_unchanged():
    """Invoice extraction should produce same results as before refactor."""
    ...

def test_resume_extraction_unchanged():
    """Resume extraction should produce same results as before refactor."""
    ...
```

### 5. Error Handling Tests
```python
def test_tool_failure_does_not_crash_extraction():
    """If a tool fails, extraction should continue with fallback."""
    ...

def test_invalid_region_id_handled_gracefully():
    """Tool should handle invalid region_id without crashing."""
    ...
```

## Test Fixtures Needed

### Mock Regions
```python
@pytest.fixture
def text_only_regions():
    """Regions with only TEXT type."""
    return [
        LayoutRegion(region_id="r1", region_type=RegionType.TEXT, ...),
        LayoutRegion(region_id="r2", region_type=RegionType.TEXT, ...),
    ]

@pytest.fixture
def visual_regions_with_chart():
    """Regions including a PICTURE (chart)."""
    return [
        LayoutRegion(region_id="r1", region_type=RegionType.TEXT, ...),
        LayoutRegion(region_id="r2", region_type=RegionType.PICTURE, ...),
    ]

@pytest.fixture
def visual_regions_with_table():
    """Regions including a TABLE."""
    return [
        LayoutRegion(region_id="r1", region_type=RegionType.TEXT, ...),
        LayoutRegion(region_id="r2", region_type=RegionType.TABLE, ...),
    ]

@pytest.fixture
def mixed_regions():
    """Regions with multiple types."""
    return [
        LayoutRegion(region_id="r1", region_type=RegionType.TEXT, ...),
        LayoutRegion(region_id="r2", region_type=RegionType.PICTURE, ...),
        LayoutRegion(region_id="r3", region_type=RegionType.TABLE, ...),
        LayoutRegion(region_id="r4", region_type=RegionType.FORMULA, ...),
    ]
```

### Mock Documents
```python
@pytest.fixture
def simple_invoice_text():
    """OCR text from a simple invoice."""
    return """
    INVOICE #12345
    Date: 2024-01-15
    Total: $500.00
    ...
    """

@pytest.fixture
def invoice_with_chart_text():
    """OCR text from invoice with chart (chart data not in text)."""
    return """
    INVOICE #12345
    [Chart showing monthly sales - data not extractable via OCR]
    ...
    """
```

## Files to Create/Modify
- `tests/agents/test_extraction_agent.py` (add new test cases)
- `tests/agents/test_tool_selection.py` (new file for tool selection tests)
- `tests/conftest.py` (add fixtures)

## Dependencies
- Task 0051: Refactor agent to always provide all tools
- Task 0052: Remove try/except ImportError pattern
- Task 0053: Improve tool descriptions
- Task 0054: Add tool usage to system prompt

## Testing Strategy
- Use pytest with pytest-asyncio for async tests
- Mock LLM responses to test tool selection logic
- Mock VLM calls to avoid external API dependencies
- Use snapshot testing for regression tests where appropriate

## Notes
- Focus on testing the decision-making logic, not the tool implementations
- Tool implementations have their own unit tests
- Consider using `pytest.mark.parametrize` for testing multiple scenarios
- Add integration tests that run against real LLM (marked as slow/optional)
