# 2026-01-31 Feature: Native Excel Parsing Support

## Summary
- Added native Excel parsing pipeline using openpyxl with new `STRUCTURED` processing category.
- Introduced `ExcelExtractor` service, spreadsheet dataclasses, and `analyze_spreadsheet` tool for agent access.
- Routed `.xlsx` documents through structured extraction with preview text; configurable via `use_native_spreadsheet_extraction`.
- Added unit/integration tests for Excel extraction and routing.

## Notes
- Spreadsheet preview truncates to 30 rows for prompt budget.
- Formulas are captured alongside computed values; merged cells flagged per-cell.
