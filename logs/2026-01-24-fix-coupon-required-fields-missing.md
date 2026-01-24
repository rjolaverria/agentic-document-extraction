# Log: Fix Coupon Form Extraction Missing Required Fields

**Date**: 2026-01-24
**Task**: 0031-issue-coupon-required-fields-missing
**Status**: Completed

## Problem

When extracting data from the coupon form image (`sample_coupon_code_form.png`), required fields like `brands_applicable` and `media_type` were either empty or incorrectly extracted.

### Root Cause Analysis

1. **OCR was missing underlined text values**: Tesseract OCR failed to recognize the underlined/filled-in form values. For example:
   - "OLD GOLD" (brand name) was completely absent from OCR output
   - "DIRECT MAIL" (media type) was completely absent from OCR output

2. **Label vs Value Confusion**: The extraction was getting the form labels (like "MEDIA TYPE") instead of the actual values ("DIRECT MAIL") because OCR output didn't preserve the spatial relationship between labels and values.

3. **Visual Pipeline Not Connected**: The codebase had visual extraction components (layout detection, region extraction, synthesis) but they weren't integrated into the extraction processor. Visual documents were being treated as text-only after OCR.

## Solution

Created a new `VisualDocumentExtractionService` that uses GPT-4V (Vision Language Model) to extract directly from document images, bypassing OCR limitations.

### Key Changes

1. **New Service**: `src/agentic_document_extraction/services/extraction/visual_document_extraction.py`
   - Uses GPT-4V to "see" the document image directly
   - Includes form-specific prompts that help the model understand label-value relationships
   - Combines image analysis with OCR text as reference
   - Returns standard `ExtractionResult` for compatibility with agentic loop

2. **Modified Extraction Processor**: `src/agentic_document_extraction/services/extraction_processor.py`
   - For visual documents (`ProcessingCategory.VISUAL`), now uses VLM-based extraction instead of text extraction
   - Passes both the image path and OCR text to the visual extraction service

### Results

| Field | Before (OCR) | After (VLM) |
|-------|--------------|-------------|
| brands_applicable | `[]` | `["OLD GOLD"]` |
| media_type | `"MEDIA TYPE"` | `"DIRECT MAIL"` |
| to | `[]` | `["KELLI SCRUGGS"]` |
| from | `[]` | `["LEONARD JONES"]` |
| cc | `[]` | 6 names extracted |
| signed_by | `null` | `"Leonard H Jones"` |

All required fields are now properly extracted with the VLM approach.

## Files Changed

- `src/agentic_document_extraction/services/extraction/visual_document_extraction.py` (new)
- `src/agentic_document_extraction/services/extraction_processor.py` (modified)
- `tests/test_services/test_extraction/test_visual_document_extraction.py` (new)

## Testing

- All 865 tests pass
- 93% code coverage
- Manual verification with coupon form shows correct extraction

## Notes

- The quality report still shows "failed" status due to unrelated issues (task 0032: null optionals causing schema violations)
- Date format issues are handled by existing normalization in JsonGenerator
- This approach provides a general improvement for all form-based visual documents
