# ISSUE: Coupon Form Extraction Missing Required Fields

- [ ] **ISSUE: Coupon Form Extraction Missing Required Fields**
  - **Severity**: High
  - **Discovered**: 2026-01-24 via local run with `sample_coupon_code_form.png`
  - **Reproduction**: POST `/extract` with `tests/fixtures/sample_documents/sample_coupon_code_form.png` + `tests/fixtures/sample_schemas/sample_coupon_code_form_schema.json`, then inspect `quality_report` in `/jobs/{job_id}/result`.
  - **Observed**: `media_type` is null and `brands_applicable` is an empty array; `quality_report.status` is `failed` with missing required field errors and low-confidence warnings.
  - **Expected**: Required fields should be populated (non-null, non-empty) or the system should surface a clearer extraction failure specific to OCR/layout capture.
