# Task 0041: Refactor Visual Document Extraction Pipeline

## Objective
Refactor the visual document extraction pipeline to match the new architecture: PaddleOCR for text + layout → LayoutReader for reading order → Tool-based agent for extraction.

## Context
Current pipeline (`visual_text_extractor.py`, `visual_document_extraction.py`):
1. PaddleOCR for text extraction
2. LLM-based reading order detection
3. Region-based visual extraction service (processes all regions)
4. Synthesis service

New pipeline (from architecture diagram):
1. PaddleOCR for text extraction AND layout detection
2. LayoutReader for reading order
3. Single extraction agent with tools (processes regions selectively)

## Acceptance Criteria
- [ ] Update `VisualTextExtractor` to use PaddleOCR layout detection
- [ ] Integrate LayoutReader for reading order (Task 0037)
- [ ] Refactor to pass ordered OCR text + layout regions to new ExtractionAgent
- [ ] Remove old region-based visual extraction service
- [ ] Remove old synthesis service
- [ ] Update `extraction_processor.py` to use new pipeline
- [ ] Maintain backward compatibility for API response format
- [ ] Unit tests for new pipeline components
- [ ] Integration tests with visual documents
- [ ] Performance benchmarks

## Pipeline Flow
```python
# New visual extraction flow
visual_extractor = VisualTextExtractor()

# Step 1: Extract text + detect layout with PaddleOCR
ocr_result = visual_extractor.extract_with_layout(document_path)
# Returns: {
#   "text_regions": [...],  # OCR text with bounding boxes
#   "layout_regions": [...],  # Tables, charts, text blocks
#   "full_text": "...",
#   "confidence": 0.95
# }

# Step 2: Determine reading order with LayoutReader
reading_order = layout_reader.order_regions(ocr_result.text_regions)
ordered_text = "\n".join([region.text for region in reading_order])

# Step 3: Extract with tool-based agent
extraction_agent = ExtractionAgent(
    schema=schema,
    ordered_text=ordered_text,
    layout_regions=ocr_result.layout_regions,
    document_image=document_image,
    tools=[AnalyzeChartTool(), AnalyzeTableTool()]
)
result = extraction_agent.extract()
```

## Dependencies
- Task 0037 (LayoutReader integration)
- Task 0038 (AnalyzeChart tool)
- Task 0039 (AnalyzeTable tool)
- Task 0040 (Tool-based extraction agent)

## Implementation Notes
- Update PaddleOCR configuration to return layout detection results
- Ensure layout regions include region_id for tool reference
- Cache document image for tool-based cropping
- Update response models if needed to include layout metadata
- Consider memory usage for large documents (image caching)

## Testing Strategy
- Test with various visual document types (PDFs, images, scanned docs)
- Compare extraction quality vs old pipeline
- Benchmark processing time and API costs
- Test with fixtures: invoice.pdf, resume.pdf, coupon_form.png
- Verify no regressions in existing functionality

## Migration Strategy
- Keep old pipeline code for comparison
- Add feature flag to switch between old and new pipeline
- Run A/B tests on fixture documents
- Monitor extraction quality metrics
- Deprecate old pipeline components once validated
