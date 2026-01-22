## Tasks 

- Environment setup:
  - [x] uv project initialized
  - [x] pytest configured and passing
  - [x] ruff configured and passing
  - [x] mypy configured, targeting Python 3.12, and passing
  - [x] FastAPI dev server runnable


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

- [x] Document Format Detection & Classification
  - As a system, I want to automatically detect and classify document format from uploaded files, so that the system knows whether to process it as text-based or visual.
  - **Acceptance Criteria**:
    - Detects format from file extension and/or MIME type
    - Supports: txt, pdf, doc, docx, odt, ppt, pptx, odp, csv, xlsx, jpeg, png, bmp, psd, tiff, gif, webp
    - Classifies documents into two categories:
      - **Text-based**: txt, csv
      - **Visual**: pdf, doc, docx, odt, ppt, pptx, odp, xlsx, jpeg, png, bmp, psd, tiff, gif, webp
    - Returns structured format information (MIME type, extension, format family, processing category)
    - Handles edge cases (missing extensions, incorrect extensions using magic bytes)
    - Uses `python-magic` or similar for robust detection
    - Unit tests for all supported formats

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

- [x] Text Extraction from Text-Based Documents
  - As a system, I want to extract raw text from text-based documents, so that I can process them directly without visual conversion.
  - **Acceptance Criteria**:
    - Extracts text from .txt files preserving line breaks and structure
    - Detects and handles different encodings (UTF-8, Latin-1, etc.) using `chardet`
    - Extracts text from .csv files preserving tabular structure using `pandas`
    - Returns extracted text with metadata (encoding, line count, structure type)
    - Preserves natural reading order
    - Unit tests with various encodings and structures

- [x] Use LangChain Document Loaders for reading data from different sources.


- [x] LangChain LLM Integration for Text Extraction
  - As a system, I want to use LangChain with OpenAI models to extract structured information from text documents according to the schema.
  - **Acceptance Criteria**:
    - Integrates `langchain-openai` with GPT-4 or GPT-4 Turbo
    - Configures OpenAI API key from environment variables
    - Creates LangChain chain for extraction task
    - Uses structured output (JSON mode or function calling) to enforce schema compliance
    - Sends extracted text with JSON schema as context
    - Extracts information matching user's JSON schema
    - Returns extraction with confidence indicators where applicable
    - Handles documents that exceed token limits (chunking strategy with LangChain text splitters)
    - Includes reasoning in extraction process (using reasoning models when available)
    - Unit tests with mocked OpenAI responses
    - Integration tests with real API (marked as optional/skippable)

- [x] JSON + Markdown Output (Text Documents)
  - As a user, I want extracted information from text documents returned as valid JSON and Markdown via API response, so that I can use the results programmatically and review them easily.
  - **Acceptance Criteria**:
    - Maps extracted data to user-provided JSON schema
    - Validates output against schema before returning using `jsonschema`
    - Returns validation errors if extraction doesn't match schema (with retry logic)
    - API response includes both JSON and Markdown in structured format
    - Generates formatted Markdown summary using LangChain
    - Handles missing or uncertain data gracefully (null values, confidence indicators)
    - Includes source references (line numbers, sections) in Markdown
    - Response includes metadata (processing time, model used, token usage)
    - Unit tests for output generation and validation

### Core Stories - Phase 3 (Visual Document Processing Pipeline)

- [x] Text Extraction from Visual Documents (OCR)
  - As a system, I want to extract text from PDFs and images using OCR, so that I have the raw textual content before visual analysis.
  - **Acceptance Criteria**:
    - Extracts text from PDF documents using `pypdf` or `pdfplumber` (multi-page support)
    - Extracts text from image formats using OCR (`pytesseract` or cloud OCR)
    - Returns text with bounding box coordinates for each text element
    - Preserves page/image boundaries
    - Returns confidence scores for OCR results
    - Handles multi-page documents (PDFs with multiple pages, multi-page TIFFs)
    - Unit tests with sample PDFs and images
    - Performance considerations for large documents

- [x] Layout Detection with Hugging Face Transformers
  - As a system, I want to detect and segment different layout regions in visual documents using open-source models, so that I can process each region appropriately.
  - **Acceptance Criteria**:
    - Integrates Hugging Face `transformers` library
    - Uses pre-trained layout detection model (e.g., LayoutLMv3, Detectron2-based models from HF Model Hub)
    - Detects layout regions: headers, footers, body text, tables, figures, captions, sidebars, etc.
    - Returns bounding boxes and classification for each region
    - Segments document images into separate region images
    - Maintains metadata linking regions to source pages
    - Handles nested regions (e.g., caption within figure region)
    - Works across multi-page documents
    - Model caching for performance
    - Unit tests with sample document layouts

- [x] Reading Order Detection with LLM
  - As a system, I want to use GPT models to detect the reading order of text elements and layout regions, so that I understand the logical flow of the document.
  - **Acceptance Criteria**:
    - Takes text elements/regions with bounding box coordinates as input
    - Uses LangChain with OpenAI GPT-4 to analyze spatial relationships
    - Prompts model with coordinates and text snippets to determine reading order
    - Returns ordered sequence of elements/regions
    - Handles complex layouts (multi-column, sidebar, headers/footers)
    - Provides confidence scores for reading order decisions
    - Works across multi-page documents
    - Unit tests with mocked LLM responses
    - Integration tests with various layout patterns

- [x] Region-Based Visual Extraction with VLM
  - As a system, I want to send individual layout regions to OpenAI's vision models for detailed analysis, so that I can extract information from complex visual elements.
  - **Acceptance Criteria**:
    - Integrates LangChain with OpenAI GPT-4V (Vision)
    - Sends segmented region images to VLM
    - Provides context for each region (type, position, surrounding regions, reading order)
    - Extracts structured information from tables, charts, figures using vision capabilities
    - Handles text-heavy regions with combined OCR + VLM analysis
    - Returns extraction results per region with confidence scores
    - Preserves spatial relationships between regions
    - Uses reasoning capabilities when analyzing complex visual structures
    - Batch processing for efficiency
    - Unit tests with mocked VLM responses
    - Integration tests with sample visual elements

- [x] Visual Document Synthesis
  - As a system, I want to combine region-level extractions into a coherent document-level extraction using LangChain, so that I can produce final output matching the user's schema.
  - **Acceptance Criteria**:
    - Uses LangChain chains to orchestrate synthesis
    - Combines information from all regions respecting reading order
    - Resolves conflicts or redundancies between regions using LLM reasoning
    - Maps combined information to user's JSON schema
    - Validates against schema using `jsonschema`
    - Generates JSON + Markdown output similar to text-based extraction
    - Includes source references (page numbers, region types, coordinates)
    - Metadata includes processing pipeline details
    - Unit tests for synthesis logic

### Core Stories - Phase 4 (Agentic Loop with LangChain)

- [x] Extraction Planning with LangChain Agent
  - As a system, I want to use a LangChain agent to create an extraction plan before starting, so that I can optimize the extraction strategy.
  - **Acceptance Criteria**:
    - Creates LangChain agent with planning capabilities
    - Analyzes schema complexity and document characteristics
    - Determines extraction strategy based on document type (text vs. visual)
    - For visual documents: plans region processing order and prioritization
    - Identifies potential challenges (complex tables, multi-column, charts)
    - Generates step-by-step extraction plan using LLM reasoning
    - Estimates confidence and quality thresholds
    - Plan is logged and included in response metadata
    - Unit tests with various document/schema combinations

- [x] Quality Verification Agent
  - As a system, I want a LangChain agent to verify extraction quality against defined thresholds, so that I can determine if results are acceptable or need refinement.
  - **Acceptance Criteria**:
    - Creates verification agent with quality assessment capabilities
    - Defines quality metrics (schema coverage, confidence scores, completeness, consistency)
    - Evaluates extraction results against thresholds using LLM reasoning
    - Identifies specific issues (missing required fields, low confidence values, schema violations, logical inconsistencies)
    - Returns verification report with pass/fail status
    - Provides actionable feedback for improvement
    - Different verification strategies for text vs. visual documents
    - Configurable quality thresholds
    - Unit tests for verification logic

- [x] Iterative Refinement Loop
  - As a system, I want to automatically refine extractions that don't meet quality thresholds using LangChain agents, so that I can improve results without user intervention.
  - **Acceptance Criteria**:
    - Implements agentic loop: Plan → Execute → Verify → Refine
    - Re-attempts extraction with refined prompts based on verification feedback
    - For text documents: focuses on specific schema fields that were missed or incorrect
    - For visual documents: re-processes specific regions or uses different analysis strategies
    - Uses LangChain memory to maintain context across iterations
    - Limits iteration count to prevent infinite loops (configurable, default max 3-5 iterations)
    - Tracks improvement metrics across iterations
    - Returns best result even if threshold not met, with quality report
    - Logs entire agentic loop for debugging
    - Unit tests for refinement logic
    - Integration tests for full agentic loop

### Core Stories - Phase 5 (API Enhancement & Production Readiness)

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

- [x] API Documentation & Examples
  - As a user, I want comprehensive API documentation with examples, so that I can easily integrate the service.
  - **Acceptance Criteria**:
    - OpenAPI/Swagger documentation at `/docs`
    - Request/response schemas fully documented
    - Example requests in documentation
    - README with:
      - Quick start guide
      - API endpoint descriptions
      - Example curl commands
      - Example Python client code
      - Supported document formats
      - Schema specification guide
    - Postman collection or similar for testing
    - Sample documents and schemas in repository

### Maintenance & Bug Fixes

- [x] End-to-End Testing & Bug Fixes
  - As a developer, I want the test suite to pass completely with no warnings, so that the codebase is production-ready.
  - **Fixes Applied**:
    - Fixed API key fallback logic bug: Changed `api_key or settings...` to `api_key if api_key is not None else settings...` across 8 service files to properly handle empty string values in tests
    - Fixed Pydantic warning: Renamed `schema` parameter to `extraction_schema` with `alias="schema"` in API endpoint to avoid shadowing `BaseModel.schema`
    - Improved quality verification prompts: Added explicit guidance for numerical consistency checks to reduce false positives
    - Fixed test assertion: Updated `test_default_initialization` to correctly verify settings-based defaults
  - **Results**:
    - All 787 tests passing (was 778 passed, 9 failed)
    - No Pydantic warnings
    - Ruff: All checks passed
    - Mypy: No issues found

#### Critical Bugs Found

- [x] **BUG: PNG/Image File Support Not Working**
  - **Severity**: Critical
  - **Error**: `TextExtractionError: Unsupported file extension for text extraction: .png`
  - **Description**: The API accepts PNG files via the `/extract` endpoint (returns 202 Accepted), but the background processor fails immediately because the text extraction service does not support image files.
  - **Expected Behavior**: Image files should be routed to the visual extraction pipeline (OCR + layout detection + VLM), not the text extraction pipeline.
  - **Root Cause**: The extraction processor was using `TextExtractor` for visual documents instead of `VisualTextExtractor`.
  - **Fix Applied**: Updated `services/extraction_processor.py` to use `VisualTextExtractor` for documents with `ProcessingCategory.VISUAL`
  - **Tests Added**: 12 new tests in `tests/test_services/test_extraction_processor.py` covering all visual and text format routing

#### Quality Issues Found

- [x] **ISSUE: Date Format Inconsistency in Invoice Extraction**
  - **Severity**: Medium
  - **Description**: The `due_date` field is extracted as "February 15, 2024" instead of the expected ISO format "YYYY-MM-DD" per the schema's `format: date` specification.
  - **Current Value**: `"February 15, 2024"`
  - **Expected**: `"2024-02-15"`
  - **Impact**: Downstream systems expecting ISO dates will need to handle this inconsistency.
  - **Fix Applied**: Added date format normalization in the JSON generator with support for multiple date formats (ISO, US, EU, month names). The `FieldInfo` class now captures `format_spec` from JSON schemas, and dates are normalized to ISO format when `format: date` is specified.

- [ ] **ISSUE: Low Confidence Scores Prevent Convergence**
  - **Severity**: Medium
  - **Description**: Resume and CSV extractions complete with all data correctly extracted, but the agentic loop does not converge because confidence scores remain below the threshold (0.90).
  - **Resume**: Confidence 0.70, 4 issues flagged (contact info, phone format, skills format)
  - **CSV**: Confidence 0.50, 2 issues flagged
  - **Impact**: Extra iterations are wasted without improving results.
  - **Suggested Fix**: Review confidence threshold settings or improve confidence scoring logic.

- [ ] **ISSUE: Spurious Null Value Warnings in CSV Extraction**
  - **Severity**: Low
  - **Description**: The quality report flags `employees[].name` and `employees[].email` as having null values, but the actual extracted data contains all values correctly.
  - **Quality Report Issue**:
    ```json
    {
      "issue_type": "null_value",
      "field_path": "employees[].name",
      "message": "Nested required field 'employees[].name' has null value"
    }
    ```
  - **Actual Data**: All 5 employees have names and emails correctly populated.
  - **Impact**: False positive quality warnings reduce trust in the quality report.
  - **Suggested Fix**: Fix the verifier's array field null-checking logic.

- [ ] **ISSUE: Skills Extracted as Category-Prefixed Strings**
  - **Severity**: Low
  - **Description**: Resume skills are extracted as strings like `"Languages: Python, JavaScript, TypeScript, Go, SQL"` instead of individual skill items.
  - **Schema Expectation**: Array of individual skill strings
  - **Current Extraction**: Array of category-prefixed strings
  - **Impact**: Downstream parsing of individual skills is complicated.
  - **Suggested Fix**: Improve extraction prompt to split skills into individual items.

- [ ] **ISSUE: Phone Number Format Not Standardized**
  - **Severity**: Low
  - **Description**: Phone numbers are extracted in various formats like `(555) 987-6543` without standardization to E.164 or another consistent format.
  - **Impact**: Consistency issues for downstream systems.
  - **Suggested Fix**: Add optional phone number normalization in post-processing.

#### Environment/Setup Issues

- [x] **ISSUE: Tesseract OCR Not Installed - Image Extraction Fails**
  - **Severity**: Critical (blocks all image processing)
  - **Discovered**: 2026-01-22 via E2E test with `sample_coupon_code_form.png`
  - **Error**: `VisualExtractionError: Failed to extract text from image: tesseract is not installed or it's not in your PATH.`
  - **Description**: The visual extraction pipeline requires `tesseract-ocr` to be installed as a system dependency. The Python `pytesseract` package is a wrapper that calls the tesseract binary. Without it, all image-based document extraction (PNG, JPG, BMP, TIFF, GIF, WEBP) will fail.
  - **Impact**: Complete failure of image document processing
  - **Resolution Applied**:
    1. Installed tesseract-ocr via Homebrew (tesseract 5.5.2)
    2. Updated README.md with installation instructions for macOS, Ubuntu/Debian, and Windows
    3. Verified image extraction works with E2E test on `sample_coupon_code_form.png`
