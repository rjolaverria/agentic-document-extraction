# 2026-01-24 Local Run: Fixture Extractions

## Scope
- Started FastAPI server locally with `uv run uvicorn agentic_document_extraction.api:app --host 127.0.0.1 --port 8000`.
- Exercised `/extract` and `/jobs/{job_id}/result` for all fixtures in `tests/fixtures`.

## Fixtures Tested
- `tests/fixtures/sample_documents/sample_invoice.txt` + `tests/fixtures/sample_schemas/invoice_schema.json`
- `tests/fixtures/sample_documents/sample_resume.txt` + `tests/fixtures/sample_schemas/resume_schema.json`
- `tests/fixtures/sample_documents/sample_data.csv` + `tests/fixtures/sample_schemas/employee_data_schema.json`
- `tests/fixtures/sample_documents/sample_coupon_code_form.png` + `tests/fixtures/sample_schemas/sample_coupon_code_form_schema.json`

## Findings
- Invoice run reports a `format_error` for `invoice_date` even though the extracted value is `2024-01-15`.
- Coupon form run fails quality checks due to missing required fields (`media_type`, `brands_applicable`) and low-confidence outputs.
- Coupon form run reports schema violations when optional string fields are returned as null.

## Artifacts
- Server log: `logs/local_server_test.log`
