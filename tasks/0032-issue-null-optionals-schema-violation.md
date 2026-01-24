# ISSUE: Optional String Fields Returned as null Trigger Schema Violations

- [x] **ISSUE: Optional String Fields Returned as null Trigger Schema Violations**
  - **Severity**: Medium
  - **Discovered**: 2026-01-24 via local run with `sample_coupon_code_form.png`
  - **Reproduction**: POST `/extract` with `tests/fixtures/sample_documents/sample_coupon_code_form.png` + `tests/fixtures/sample_schemas/sample_coupon_code_form_schema.json`, then inspect `quality_report.issues` in `/jobs/{job_id}/result`.
  - **Observed**: Output includes nulls for optional string fields (e.g., `signed_by`, `media_name`, `issue_frequency`), and the verifier reports `schema_violation` issues because schema types are `string` without `null` allowance.
  - **Expected**: Optional fields should be omitted when unknown or the schema should allow `null` values for optional fields.
