# Document Upload Endpoint

- [x] Document Upload Endpoint
  - As a user, I want to upload a document file along with a JSON schema via API, so that I can trigger the extraction process.
  - **Acceptance Criteria**:
    - `POST /extract` endpoint accepts multipart form data
    - Accepts file upload (various formats)
    - Accepts JSON schema as either file or JSON string in form field
    - Validates file size limits (configurable, e.g., 10MB default)
    - Returns 400 for missing required fields
    - Returns 413 for files exceeding size limit
    - Saves uploaded file temporarily with unique identifier
    - Tests for upload validation and error cases
