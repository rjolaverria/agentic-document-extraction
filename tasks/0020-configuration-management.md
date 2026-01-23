# Configuration Management

- [x] Configuration Management
  - As an operator, I want centralized configuration management, so that I can easily configure the service for different environments.
  - **Acceptance Criteria**:
    - Environment-based configuration using `pydantic-settings`
    - Configuration for:
      - OpenAI API key and model selections
      - Quality thresholds
      - Iteration limits
      - File size limits
      - Job TTL
      - Logging levels
    - `.env` file support for local development
    - Configuration validation on startup
    - Sensitive values (API keys) never logged
    - Documentation for all configuration options
