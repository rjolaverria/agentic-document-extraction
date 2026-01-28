# Refactor: Visual Document Extraction Pipeline

**Date:** 2026-01-28
**Task:** 0041 - Refactor Visual Document Extraction Pipeline
**Status:** Complete

## Summary

Refactored the visual document extraction pipeline to use a unified approach where PaddleOCR handles both text extraction AND layout detection in a single pass. Integrated the ReadingOrderDetector to determine reading order of layout regions before passing to the ExtractionAgent.

## Changes Made

### 1. VisualTextExtractor Updates (`visual_text_extractor.py`)

**New Data Structures:**
- Added `VisualExtractionWithLayoutResult` dataclass that combines:
  - `ocr_result`: Standard `VisualExtractionResult` with OCR text
  - `layout_result`: `LayoutDetectionResult` with detected regions
- Added `PADDLE_LAYOUT_LABEL_MAP` mapping PaddleOCR labels to `RegionType` enum

**New Methods:**
- `extract_with_layout(file_path)`: Unified extraction that returns both OCR text and layout regions
- `_detect_layout(file_path, ocr_result)`: Internal method using PaddleOCR's `LayoutDetection` module
- `_detect_layout_for_image(image, page_num, layout_engine, region_counter)`: Per-image layout detection
- `_parse_layout_item(item, page_num, region_counter)`: Parses PaddleOCR layout output to `LayoutRegion`
- `_get_layout_engine()`: Lazy initialization of PaddleOCR `LayoutDetection` engine

**Architecture:**
- Uses `PP-DocLayoutV2` model from PaddleOCR for layout detection
- Handles both single images and multi-page PDFs
- Graceful degradation: if layout detection fails, returns empty layout but valid OCR

### 2. ExtractionProcessor Updates (`extraction_processor.py`)

**Pipeline Changes:**
- Visual documents now use `extract_with_layout()` instead of separate OCR + LayoutDetector calls
- Integrated `ReadingOrderDetector` to order regions by reading sequence
- Layout regions from the unified extraction are passed directly to `ExtractionAgent`
- Fallback full-image PICTURE region still created when no regions detected

**Flow:**
```
Visual Document → extract_with_layout() → OCR + Layout Regions
                                       ↓
                           ReadingOrderDetector → Ordered Regions
                                       ↓
                           ExtractionAgent (with tools)
```

### 3. Test Updates

**test_visual_text_extractor.py:**
- Added `TestVisualExtractionWithLayoutResult` class testing the new combined result type
- Added `TestExtractWithLayout` class testing the new unified extraction method
- Tests cover: file not found, successful extraction, layout detection failure handling

**test_extraction_processor.py:**
- Updated `create_mock_visual_extractor()` to mock `extract_with_layout()`
- Changed assertions to verify `extract_with_layout` is called (not `extract_from_path`)
- Updated `TestExtractionProcessorToolAgent` tests for new pipeline:
  - `test_visual_doc_with_layout_regions_passes_to_agent`: Verifies regions passed to agent
  - `test_visual_doc_without_regions_uses_fallback`: Verifies fallback full-image region

## Benefits

1. **Unified Pipeline**: Single pass for OCR + layout reduces model loading and processing time
2. **PaddleOCR Consistency**: Using same library for both OCR and layout detection
3. **Reading Order Integration**: Regions are now ordered before extraction
4. **Graceful Degradation**: Layout detection failures don't break text extraction

## Technical Notes

- PaddleOCR `LayoutDetection` module uses `PP-DocLayoutV2` model
- Layout labels mapped: text, title, table, figure/picture, list, formula, header, footer, caption, footnote
- The `_layout_engine` is lazily initialized and cached on the extractor instance

## Verification

- All 890 tests pass
- mypy type checking passes
- ruff linting passes
- 86% code coverage maintained

## Files Modified

- `src/agentic_document_extraction/services/visual_text_extractor.py`
- `src/agentic_document_extraction/services/extraction_processor.py`
- `tests/test_services/test_visual_text_extractor.py`
- `tests/test_services/test_extraction_processor.py`
