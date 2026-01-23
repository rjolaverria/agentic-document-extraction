# Async Processing & Job Management

- [x] Async Processing & Job Management
  - As a user, I want long-running extractions to be processed asynchronously, so that the API doesn't timeout.
  - **Acceptance Criteria**:
    - `POST /extract` returns immediately with job ID for large documents
    - Background task processing using FastAPI BackgroundTasks or Celery
    - `GET /jobs/{job_id}` endpoint to check job status
    - `GET /jobs/{job_id}/result` endpoint to retrieve results when complete
    - Job states: pending, processing, completed, failed
    - Results stored temporarily with TTL (e.g., 24 hours)
    - WebSocket endpoint for real-time progress updates (optional)
    - Unit tests for job lifecycle
