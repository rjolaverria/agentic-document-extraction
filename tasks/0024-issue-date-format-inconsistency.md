# ISSUE: Date Format Inconsistency in Invoice Extraction

- [x] **ISSUE: Date Format Inconsistency in Invoice Extraction**
  - **Severity**: Medium
  - **Description**: The `due_date` field is extracted as "February 15, 2024" instead of the expected ISO format "YYYY-MM-DD" per the schema's `format: date` specification.
  - **Current Value**: `"February 15, 2024"`
  - **Expected**: `"2024-02-15"`
  - **Impact**: Downstream systems expecting ISO dates will need to handle this inconsistency.
  - **Fix Applied**: Added date format normalization in the JSON generator with support for multiple date formats (ISO, US, EU, month names). The `FieldInfo` class now captures `format_spec` from JSON schemas, and dates are normalized to ISO format when `format: date` is specified.
