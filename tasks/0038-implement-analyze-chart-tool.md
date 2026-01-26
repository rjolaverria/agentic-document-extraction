# Task 0038: Implement AnalyzeChart Tool for LangChain Agent

## Objective
Create a LangChain tool that allows the extraction agent to analyze chart/graph regions by sending cropped images to a Vision Language Model (VLM).

## Context
The new architecture uses a tool-based approach where the agent decides when to use VLM analysis for specific regions. The AnalyzeChart tool:
- Takes a region ID (from layout detection)
- Crops the chart/graph region from the source document
- Sends the cropped image to GPT-4V
- Extracts structured data: chart type, axes labels, data points, trends

This replaces the current approach where all visual regions are processed automatically.

## Acceptance Criteria
- [ ] Create `AnalyzeChartTool` class implementing LangChain `BaseTool`
- [ ] Tool accepts region_id parameter
- [ ] Crops image region using bounding boxes from layout detection
- [ ] Sends cropped image to GPT-4V with structured extraction prompt
- [ ] Returns structured output: chart_type, x_axis, y_axis, data_points, trends, notes
- [ ] Tool description clearly explains when to use (for chart/graph regions)
- [ ] Integration with PaddleOCR layout detection results
- [ ] Unit tests for tool functionality
- [ ] Integration tests with sample charts
- [ ] Error handling for invalid region IDs or failed VLM calls

## Tool Schema
```python
class AnalyzeChartInput(BaseModel):
    region_id: str = Field(description="ID of the chart region to analyze")

class AnalyzeChartOutput(BaseModel):
    chart_type: str  # e.g., "bar", "line", "pie", "scatter"
    x_axis_label: str | None
    y_axis_label: str | None
    data_points: list[dict[str, Any]]
    trends: str  # Narrative description of trends
    notes: str | None  # Additional observations
```

## Dependencies
- PaddleOCR layout detection with region bounding boxes
- GPT-4V API access (already configured)
- Image cropping utilities (PIL/Pillow)

## Implementation Notes
- Store layout detection results in agent context
- Implement efficient image cropping (cache original image)
- Use structured output with GPT-4V for reliable parsing
- Consider caching chart analysis results to avoid redundant API calls
- Tool should be stateless - all context passed via parameters or injected dependencies

## Testing Strategy
- Create test fixtures with various chart types
- Mock VLM responses for unit tests
- Integration tests with real GPT-4V calls
- Test error cases: invalid region_id, non-chart regions, VLM failures
