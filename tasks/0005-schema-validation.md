# Schema Validation

- [x] Schema Validation
  - As a system, I want to validate the user-provided JSON schema before processing, so that extraction has a valid target structure.
  - **Acceptance Criteria**:
    - Accepts JSON schema in standard JSON Schema format (Draft 7 or later)
    - Validates schema syntax before processing using `jsonschema`
    - Provides clear error messages for invalid schemas (returns 400 with details)
    - Supports common data types (string, number, boolean, object, array)
    - Supports nested structures
    - Extracts required fields and optional fields for extraction planning
    - Unit tests for valid and invalid schema cases
