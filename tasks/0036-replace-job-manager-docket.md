# Replace Custom Job Manager with Docket

- [ ] Replace Custom Job Manager with Docket
  - As a developer, I want to replace the custom async job manager with the Docket library, so that long-running operations have standardized async execution and progress tracking.
  - **Acceptance Criteria**:
    - Replace custom job manager implementation with Docket in async processing flows
    - Integrate Docket progress tracking and status reporting in `/jobs/{job_id}` endpoints
    - Ensure existing job metadata and result storage continue to work with Docket
    - Update configuration to support Docket settings (backend, storage, timeouts)
    - Add or update tests to validate Docket-backed job lifecycle and progress reporting
    - Update documentation to describe Docket usage and operational requirements
