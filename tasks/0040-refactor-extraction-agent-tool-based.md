# Task 0040: Refactor Extraction Agent to Tool-Based Architecture

## Objective
Refactor the main extraction agent to follow the new tool-based architecture where a single LangChain agent receives all OCR text (in reading order), layout region metadata, and selectively uses VLM tools (AnalyzeChart, AnalyzeTable) to process specific regions.

## Context
Current architecture (from AGENT.md):
- Separate agents for planning, verification, refinement
- Visual extraction service processes all regions automatically
- Multiple LLM calls for coordination

New architecture (from diagram):
- Single LangChain agent with rich system prompt
- Agent receives:
  - All OCR text in reading order
  - Layout region IDs and types (table, chart, text block)
  - Tool descriptions
- Agent decides when to use tools (AnalyzeChart, AnalyzeTable)
- Agent synthesizes final extraction from OCR + tool results

## Acceptance Criteria
- [ ] Create new `ExtractionAgent` class using LangChain agent framework
- [ ] System prompt includes:
  - All OCR text in reading order
  - Layout region metadata (IDs, types, bounding boxes)
  - Target JSON schema
  - Extraction instructions
  - Tool usage guidelines
- [ ] Register AnalyzeChart and AnalyzeTable tools with agent
- [ ] Agent makes autonomous decisions about tool usage
- [ ] Agent synthesizes final JSON output from OCR text + tool results
- [ ] Replace existing planning/verification/refinement agents with single agent
- [ ] Unit tests for agent initialization and tool registration
- [ ] Integration tests with various document types
- [ ] Performance comparison vs old multi-agent approach

## Agent Prompt Structure
```
You are an expert document extraction agent. Your task is to extract information 
from a document according to the provided JSON schema.

DOCUMENT OCR TEXT (in reading order):
{ordered_text}

LAYOUT REGIONS DETECTED:
{regions}  # List of region_id, type, bbox

TARGET SCHEMA:
{json_schema}

AVAILABLE TOOLS:
- AnalyzeChart: Use for chart/graph regions when you need structured data extraction
- AnalyzeTable: Use for complex table regions when OCR text is insufficient

INSTRUCTIONS:
1. Review the OCR text and layout regions
2. Identify which fields in the schema can be filled from OCR text alone
3. For chart/graph regions, use AnalyzeChart tool to extract structured data
4. For complex tables, use AnalyzeTable tool when needed
5. Synthesize all information into final JSON matching the schema
6. Return valid JSON only

Extract the information now:
```

## Dependencies
- Task 0038 (AnalyzeChart tool)
- Task 0039 (AnalyzeTable tool)
- Task 0037 (LayoutReader for reading order)
- PaddleOCR integration

## Implementation Notes
- Use LangChain's `create_react_agent` or similar for tool-using agent
- Store document image and layout results in agent context (via RunnableConfig or similar)
- Agent should be able to call tools multiple times for different regions
- Consider token limits - may need to truncate very long OCR text
- Implement retry logic for failed tool calls

## Testing Strategy
- Test with text-only documents (no tool usage)
- Test with documents containing charts (chart tool usage)
- Test with documents containing tables (table tool usage)
- Test with complex documents (multiple tool calls)
- Compare extraction quality vs old multi-agent approach
- Benchmark performance (execution time, API costs)

## Migration Strategy
- Keep old agents temporarily for comparison
- Add feature flag to switch between old and new architecture
- Run parallel tests on fixture documents
- Deprecate old agents once new architecture is validated
