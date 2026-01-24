# ISSUE: Invoice Date Format False Positive in Quality Report

- [ ] **ISSUE: Invoice Date Format False Positive in Quality Report**
  - **Severity**: Medium
  - **Discovered**: 2026-01-24 via local run with `sample_invoice.txt`
  - **Reproduction**: POST `/extract` with `tests/fixtures/sample_documents/sample_invoice.txt` + `tests/fixtures/sample_schemas/invoice_schema.json`, then inspect `quality_report.issues` in `/jobs/{job_id}/result`.
  - **Observed**: `quality_report.issues` contains a `format_error` for `invoice_date` even though `extracted_data.invoice_date` is `2024-01-15`. The issue payload reports `current_value` as null.
  - **Expected**: No format error when the extracted invoice date matches the `YYYY-MM-DD` format.
