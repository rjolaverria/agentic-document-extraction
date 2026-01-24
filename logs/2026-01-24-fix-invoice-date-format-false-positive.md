# Fix: Invoice Date Format False Positive in Quality Report

**Date**: 2026-01-24
**Task**: tasks/0030-issue-invoice-date-format-false-positive.md

## Summary

Fixed an issue where the quality report would show incorrect `current_value` for date fields, causing confusion when the LLM reports false positive format errors.

## Problem

When extracting invoice data:
1. The LLM extraction returns dates in human-readable format (e.g., "January 15, 2024")
2. The verification happens on this pre-normalized data
3. The JsonGenerator then normalizes dates to ISO format (e.g., "2024-01-15")
4. But the quality report issues still showed the pre-normalization values

This meant the quality report would show:
```json
{
  "field_path": "invoice_date",
  "current_value": "January 15, 2024",
  "message": "Date format is wrong"
}
```

When the actual `extracted_data` in the response had:
```json
{
  "invoice_date": "2024-01-15"
}
```

This made it impossible to tell if the issue was real or a false positive.

## Root Cause

The verification happens BEFORE date normalization in the pipeline:
1. Agentic loop runs extraction + verification (dates like "January 15, 2024")
2. JsonGenerator normalizes dates to ISO format ("2024-01-15")
3. Quality report still reflects pre-normalization state

## Solution

Two changes were made:

### 1. Updated verifier prompt (verifier.py)

Added explicit guidance about date format validation to reduce LLM false positives:
```
DATE FORMAT RULES (CRITICAL - DO NOT FLAG VALID DATES):
- JSON Schema "format": "date" expects ISO 8601 date format: YYYY-MM-DD
- Do NOT flag a format_error for date fields that correctly use YYYY-MM-DD format
```

### 2. Post-normalization quality report update (extraction_processor.py)

After normalization, the quality report issues are updated to reflect the actual normalized values:
```python
# Update quality report issues with normalized values
if quality_report.get("issues"):
    for issue in quality_report["issues"]:
        field_path = issue.get("field_path", "")
        if field_path:
            normalized_value = _get_nested_value(normalized_data, field_path)
            if normalized_value is not None:
                issue["current_value"] = normalized_value
```

This ensures the `current_value` in quality report issues matches the actual extracted data.

## Testing

- Added unit tests for LLM issue enrichment with current_value
- Added test to verify date format guidance in system prompt
- All 842 tests pass
- Manual verification confirms `current_value` now matches `extracted_data`

## Files Changed

- `src/agentic_document_extraction/agents/verifier.py`: Added date format guidance to LLM prompt, enriched LLM issues with current_value
- `src/agentic_document_extraction/services/extraction_processor.py`: Added post-normalization update for quality report issues
- `tests/test_agents/test_verifier.py`: Added tests for issue enrichment and date format validation
