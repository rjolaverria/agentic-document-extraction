# Task 0037: Implement LayoutReader for Reading Order Detection

## Objective
Replace the current LLM-based reading order detection with LayoutReader, a dedicated model for determining the reading order of text regions detected by PaddleOCR layout detection.

## Context
The new architecture (see architecture diagram) uses LayoutReader to determine reading order instead of relying on GPT-4 prompts. This provides:
- Faster processing (local model vs API calls)
- More consistent results
- Lower cost
- Better handling of complex multi-column layouts

Current implementation: `reading_order_detector.py` uses LangChain + GPT-4 with bounding boxes.
New implementation: Use LayoutReader model to sort text regions in natural reading order.

## Acceptance Criteria
- [ ] Research and integrate LayoutReader model (likely via Hugging Face transformers)
- [ ] Replace `ReadingOrderDetector` class to use LayoutReader instead of LLM
- [ ] Accept PaddleOCR layout detection results as input
- [ ] Return ordered list of text regions with reading sequence
- [ ] Maintain compatibility with existing `VisualTextExtractor` integration
- [ ] Unit tests for LayoutReader integration
- [ ] Performance tests comparing old vs new approach
- [ ] Update documentation in README and AGENT.md

## Dependencies
- Task 0035 (PaddleOCR-VL) - uses same layout detection results
- PaddleOCR layout detection (already implemented)
- LayoutReader model availability

## Implementation Notes
- LayoutReader typically works with bounding boxes and region types
- May need to convert PaddleOCR layout results to LayoutReader input format
- Consider caching model weights for faster initialization
- Test with multi-column documents, forms, and complex layouts

## Testing Strategy
- Test with various layout types (single column, multi-column, forms)
- Compare reading order accuracy vs LLM-based approach
- Benchmark processing time
- Test with fixtures: invoice.pdf, resume.pdf, coupon_form.png
