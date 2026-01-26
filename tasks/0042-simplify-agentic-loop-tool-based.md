# Task 0042: Simplify Agentic Loop for Tool-Based Architecture

## Objective
Simplify or replace the current multi-step agentic loop (plan → execute → verify → refine) with a streamlined approach that leverages the tool-based extraction agent's built-in reasoning capabilities.

## Context
Current approach (from `refiner.py`, `AgenticLoop`):
- Separate planning agent creates extraction plan
- Execution step runs text/visual extraction
- Verification agent checks quality and identifies issues
- Refinement loop iterates until quality threshold met
- Multiple LLM calls per iteration

New approach (tool-based architecture):
- Single agent with tools makes autonomous decisions
- Agent uses tools (AnalyzeChart, AnalyzeTable) as needed
- Built-in LLM reasoning handles planning and verification
- Fewer LLM calls, more efficient

The agentic loop may still be useful for:
- Quality verification after agent completes extraction
- Iterative refinement if schema compliance fails
- Confidence threshold checking
- Error recovery

## Acceptance Criteria
- [ ] Evaluate if full agentic loop (planning, verification, refinement) is still needed
- [ ] Option A: Simplify to single quality verification step after agent extraction
- [ ] Option B: Keep lightweight refinement loop for schema compliance only
- [ ] Option C: Eliminate loop entirely, rely on agent's built-in reasoning
- [ ] Update `extraction_processor.py` to use simplified approach
- [ ] Maintain or improve extraction quality vs current approach
- [ ] Reduce number of LLM API calls
- [ ] Unit tests for new approach
- [ ] Integration tests comparing quality metrics
- [ ] Performance benchmarks

## Design Options

### Option A: Quality Verification Only
```python
# Agent extracts with tools
result = extraction_agent.extract()

# Single quality check
quality_report = verify_extraction(result, schema)

if quality_report.is_valid:
    return result
else:
    raise ExtractionError(quality_report.issues)
```

### Option B: Lightweight Refinement
```python
max_iterations = 3
for i in range(max_iterations):
    result = extraction_agent.extract()
    quality_report = verify_extraction(result, schema)
    
    if quality_report.is_valid:
        return result
    
    # Give agent feedback for next iteration
    extraction_agent.add_feedback(quality_report.issues)

return result  # Return best attempt
```

### Option C: No Loop (Pure Agent)
```python
# Agent handles everything internally
result = extraction_agent.extract()
return result
```

## Dependencies
- Task 0040 (Tool-based extraction agent)
- Current quality verification logic (can be reused)

## Implementation Notes
- Start with Option B (lightweight refinement) as it's safest
- Measure quality metrics compared to current full agentic loop
- Consider A/B testing different approaches
- Keep quality verification thresholds configurable
- Monitor API call counts and costs

## Testing Strategy
- Run all existing fixture tests with new approach
- Compare extraction quality metrics:
  - Schema compliance rate
  - Field accuracy
  - Required field completion
- Benchmark performance:
  - Total execution time
  - Number of API calls
  - Token usage
  - Cost per extraction
- Test edge cases:
  - Low confidence scenarios
  - Complex nested schemas
  - Missing required fields

## Success Metrics
- Maintain >= 95% extraction quality vs current approach
- Reduce LLM API calls by >= 30%
- Reduce average extraction time by >= 20%
- Maintain schema compliance rate

## Migration Strategy
- Implement new approach alongside existing loop
- Feature flag to switch between old and new
- A/B test on production fixtures
- Gradual rollout based on quality metrics
