# Task 0051: Refactor Agent to Always Provide All Tools

## Objective
Remove conditional tool logic from the extraction agent so all tools are always available, allowing the LLM to decide which tools to use based on context.

## Context
The current extraction agent conditionally adds tools based on document type and region checks (lines 248-268 in extraction_agent.py):

```python
tools: list[Any] = []
if is_visual and has_visual_regions:
    if analyze_chart is not None:
        tools.append(analyze_chart)
    if analyze_diagram is not None:
        tools.append(analyze_diagram)
    # ... repeated for all 9 tools
```

This is problematic because:
1. The agent should have access to all tools and decide which to use based on context
2. LangGraph best practice is to give the agent all tools and let the LLM decide
3. The tool descriptions already guide when each tool should be used
4. The region metadata table in the prompt provides context for tool selection

## LangGraph Best Practice
From the LangGraph documentation on "Dynamically Select Tools Based on Runtime Context":
- All tools should be registered with the agent
- The agent/LLM uses context to decide which tools are relevant
- Tool descriptions serve as the primary mechanism for tool selection guidance

## Acceptance Criteria
- [ ] Remove `is_visual and has_visual_regions` conditional check for tool addition
- [ ] Always provide all 9 tools to the agent regardless of document type
- [ ] Keep `if tool is not None` checks (these handle import failures gracefully)
- [ ] Agent can process text-only documents without calling any tools
- [ ] Agent correctly uses tools when regions require visual analysis
- [ ] Existing extraction behavior unchanged (no regression)
- [ ] Unit tests pass

## Implementation

### Current Code (to remove)
```python
# Build tools list
tools: list[Any] = []
if is_visual and has_visual_regions:
    if analyze_chart is not None:
        tools.append(analyze_chart)
    if analyze_diagram is not None:
        tools.append(analyze_diagram)
    # ... etc
```

### New Code
```python
# Build tools list - always provide all tools, let agent decide
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

## Files to Modify
- `src/agentic_document_extraction/agents/extraction_agent.py` (lines 248-268)

## Dependencies
None - this is the foundational task.

## Testing Strategy
- Run existing unit tests to verify no regression
- Test extraction on text-only documents (agent should not call tools)
- Test extraction on visual documents (agent should call appropriate tools)
- Test extraction on mixed documents

## Notes
- The `if tool is not None` checks remain necessary until Task 0052 removes the try/except ImportError pattern
- Tool descriptions (Task 0053) become more important when agent has all tools available
- System prompt (Task 0054) should always include tool instructions when tools are provided
