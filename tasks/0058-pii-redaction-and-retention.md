# Task 0058: PII Redaction & Data Retention Controls

## Objective
Protect sensitive information by adding first-class PII redaction and retention controls across the document ingestion, processing, logging, and response layers.

## Context
The system now handles a wide range of documents (PDF, images, spreadsheets, text) and stores artifacts for later inspection. There is no systematic way to detect or redact PII, nor to automatically purge stored artifacts. This poses compliance and privacy risks for production use (HIPAA/PCI-lite/privacy-by-design).

## Acceptance Criteria
- [ ] Introduce a `PIIRedactor` service that detects common PII entities (SSN, TIN/EIN, phone, email, postal address, credit card, IBAN, date of birth) using configurable regex patterns plus optional LLM fallback for edge cases.
- [ ] Allow schema authors to mark fields with a `pii` policy (`allow`, `mask`, `hash`, `drop`) so redaction behavior is deterministic per field.
- [ ] Add an API flag (e.g., `redact_output=true`) that, when enabled, applies redaction to JSON + Markdown responses, stored artifacts, and job logs before persistence.
- [ ] Ensure raw uploads are stored only in a controlled location and are redacted or deleted according to policy before any downstream logging or tracing (OpenTelemetry spans, debug dumps, retries).
- [ ] Implement retention settings (`ADE_RETENTION_DAYS` env + config) with a scheduled purge task that removes stored uploads, intermediate artifacts, and logs older than the retention window.
- [ ] Provide metrics/log counters for redaction actions (masked, hashed, dropped) and purge outcomes for observability.
- [ ] Add unit tests for the `PIIRedactor`, policy application, and regression cases for overlapping patterns; add integration tests to verify end-to-end redaction and retention with a sample document containing PII.
- [ ] Update API docs and README with usage examples, configuration options, and security notes.

## Proposed Implementation Notes
- Place the redaction service in `src/agentic_document_extraction/services/pii_redactor.py` with pluggable detectors (regex first, optional LLM tool hook) and strategy helpers for mask/hash/drop.
- Add a small config model (e.g., `PIIRedactionConfig`) wired through dependency injection so FastAPI endpoints and background tasks share the same policy.
- For retention, add a Docket task (or cron-compatible runner) that scans storage paths and job logs, deleting items older than `retention_days`; make it idempotent and safe for concurrent runs.
- Keep defaults privacy-forward: mask PII in logs and responses when `ADE_ENV=production`; allow fully unredacted output only when explicitly requested in non-production.
