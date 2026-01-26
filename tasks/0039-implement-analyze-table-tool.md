# Task 0039: Implement AnalyzeTable Tool for LangChain Agent

## Objective
Create a LangChain tool that allows the extraction agent to analyze table regions by sending cropped images to a Vision Language Model (VLM).

## Context
Similar to the AnalyzeChart tool, the AnalyzeTable tool enables the agent to selectively process table regions with VLM when needed. The tool:
- Takes a region ID (from layout detection)
- Crops the table region from the source document
- Sends the cropped image to GPT-4V
- Extracts structured table data: headers, rows, values, notes

This gives the agent flexibility to use OCR text for simple tables or VLM for complex tables with merged cells, colors, etc.

## Acceptance Criteria
- [ ] Create `AnalyzeTableTool` class implementing LangChain `BaseTool`
- [ ] Tool accepts region_id parameter
- [ ] Crops image region using bounding boxes from layout detection
- [ ] Sends cropped image to GPT-4V with table extraction prompt
- [ ] Returns structured output: headers, rows (list of dicts), notes
- [ ] Tool description clearly explains when to use (for table regions)
- [ ] Integration with PaddleOCR layout detection results
- [ ] Unit tests for tool functionality
- [ ] Integration tests with sample tables (simple and complex)
- [ ] Error handling for invalid region IDs or failed VLM calls

## Tool Schema
```python
class AnalyzeTableInput(BaseModel):
    region_id: str = Field(description="ID of the table region to analyze")

class AnalyzeTableOutput(BaseModel):
    headers: list[str]  # Column headers
    rows: list[dict[str, Any]]  # Each row as a dict mapping header to value
    notes: str | None  # Additional observations (merged cells, formatting notes)
```

## Dependencies
- PaddleOCR layout detection with region bounding boxes
- GPT-4V API access (already configured)
- Image cropping utilities (PIL/Pillow)

## Implementation Notes
- Share image cropping logic with AnalyzeChart tool (create common utility)
- Use structured output with GPT-4V for reliable table parsing
- Consider table complexity detection to auto-suggest VLM usage
- Tool should be stateless - all context passed via parameters or injected dependencies
- Handle tables that span multiple pages (return partial results with note)

## Testing Strategy
- Create test fixtures with various table types:
  - Simple grid tables
  - Tables with merged cells
  - Tables with colored regions
  - Tables with icons/symbols
- Mock VLM responses for unit tests
- Integration tests with real GPT-4V calls
- Test error cases: invalid region_id, non-table regions, VLM failures
