# FastAPI Service Bootstrap

- [x] FastAPI Service Bootstrap
  - As a developer, I want a working FastAPI service with health check endpoint, so that I have the foundation for the extraction API.
  - **Acceptance Criteria**:
    - FastAPI application initialized with proper structure
    - Health check endpoint (`GET /health`) returns service status
    - Service can be started with `uvicorn`
    - Basic logging configured
    - CORS configured appropriately
    - OpenAPI docs available at `/docs`
    - Tests for health endpoint using `httpx`
