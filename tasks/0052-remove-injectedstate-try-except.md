# Task 0052: Remove try/except ImportError Pattern for InjectedState

## Objective
Remove the defensive try/except blocks around InjectedState imports in all tool files, simplifying tool definitions and removing unnecessary None checks.

## Context
Each tool file wraps the agent-compatible tool in a try/except block:

```python
try:
    from langchain.tools import InjectedState

    @tool("analyze_chart_agent")
    def analyze_chart(
        region_id: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> dict[str, Any]:
        ...

except ImportError:
    analyze_chart = None
```

This pattern was added for backwards compatibility with older LangChain versions. However:
1. The project requires LangChain 1.x (1.2.x) which includes InjectedState
2. The `None` checks add unnecessary complexity in extraction_agent.py
3. All modern LangChain versions (1.x+) include InjectedState
4. This pattern makes the code harder to understand and maintain

## Acceptance Criteria
- [ ] Remove try/except blocks around InjectedState imports in all 9 tool files
- [ ] Import InjectedState directly at the top of each tool file
- [ ] Remove `if analyze_X is not None` checks in extraction_agent.py
- [ ] Verify minimum LangChain version in pyproject.toml includes InjectedState
- [ ] All unit tests pass
- [ ] Integration tests pass

## Files to Modify

### Tool Files (remove try/except, move import to top)
- `src/agentic_document_extraction/agents/tools/analyze_chart.py`
- `src/agentic_document_extraction/agents/tools/analyze_diagram.py`
- `src/agentic_document_extraction/agents/tools/analyze_form.py`
- `src/agentic_document_extraction/agents/tools/analyze_handwriting.py`
- `src/agentic_document_extraction/agents/tools/analyze_image.py`
- `src/agentic_document_extraction/agents/tools/analyze_logo.py`
- `src/agentic_document_extraction/agents/tools/analyze_math.py`
- `src/agentic_document_extraction/agents/tools/analyze_signature.py`
- `src/agentic_document_extraction/agents/tools/analyze_table.py`

### Agent File (remove None checks)
- `src/agentic_document_extraction/agents/extraction_agent.py`

### Dependency File (verify version)
- `pyproject.toml`

## Implementation

### Before (each tool file)
```python
from langchain_core.tools import ToolException, tool

# ... implementation code ...

@tool("analyze_chart")
def AnalyzeChart(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze chart/graph regions when OCR text is insufficient."""
    return analyze_chart_impl(region_id, regions)


try:
    from langchain.tools import InjectedState

    @tool("analyze_chart_agent")
    def analyze_chart(
        region_id: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> dict[str, Any]:
        """..."""
        regions: list[LayoutRegion] = state.get("regions", [])
        return analyze_chart_impl(region_id, regions)

except ImportError:
    analyze_chart = None
```

### After (each tool file)
```python
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt.chat_agent_executor import InjectedState

# ... implementation code ...

@tool("analyze_chart")
def AnalyzeChart(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze chart/graph regions when OCR text is insufficient."""
    return analyze_chart_impl(region_id, regions)


@tool("analyze_chart_agent")
def analyze_chart(
    region_id: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> dict[str, Any]:
    """..."""
    regions: list[LayoutRegion] = state.get("regions", [])
    return analyze_chart_impl(region_id, regions)
```

### Before (extraction_agent.py)
```python
tools: list[Any] = []
for tool in [analyze_chart, analyze_diagram, ...]:
    if tool is not None:
        tools.append(tool)
```

### After (extraction_agent.py)
```python
tools = [
    analyze_chart,
    analyze_diagram,
    analyze_form,
    analyze_handwriting,
    analyze_image,
    analyze_logo,
    analyze_math,
    analyze_signature,
    analyze_table,
]
```

## Dependencies
- Task 0051: Refactor agent to always provide all tools (must be completed first)

## Testing Strategy
- Verify all tool imports succeed
- Run unit tests for each tool
- Run extraction agent tests
- Verify no ImportError at runtime

## Notes
- InjectedState import path may be `langgraph.prebuilt.chat_agent_executor` or `langchain.tools` depending on version
- Check actual import path in current LangChain/LangGraph versions
- Consider adding a version check in pyproject.toml: `langchain >= 1.0.0`
