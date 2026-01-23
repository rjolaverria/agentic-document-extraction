# Error Handling & Logging

- [x] Error Handling & Logging
  - As a user/operator, I want comprehensive error handling and logging, so that I can debug issues and monitor the service.
  - **Acceptance Criteria**:
    - Structured logging with configurable levels (using `structlog` or Python logging)
    - Clear error messages for common failures (unsupported format, invalid schema, API errors, OCR failures)
    - Custom exception classes for different error types
    - Proper HTTP status codes for all error cases
    - Error responses include error code, message, and details
    - Progress tracking for long-running extractions (pages processed, regions analyzed)
    - Performance metrics logged (processing time, API calls, iterations, token usage)
    - Detailed logs for debugging visual pipeline (OCR quality, layout detection results, reading order)
    - Request ID tracking through entire pipeline
    - Unit tests for error scenarios
