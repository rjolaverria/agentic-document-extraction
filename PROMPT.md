# Project: Agentic Document Extraction

**Complete the next story according to the specifications below. Only complete one story.**

## Objective
Design and implement a vision-first, agentic document extraction system as a FastAPI service in modern Python 3.12 using a clean, test-first workflow. The system enables users to submit documents via HTTP API along with a JSON schema and extract structured information. The system intelligently processes text-based documents (txt, csv, etc.) directly while treating visual documents (PDF, images, presentations) as visual objects where meaning is encoded in layout, structure, and spatial relationships. The system uses agentic patterns (plan, decide, act, verify) with LangChain orchestration to iteratively improve extraction quality until output meets defined thresholds.

## Environment & Tooling

You are working in a Python 3.12 codebase that uses:

- uv for environment & dependency management
- pytest for testing
- ruff for linting and formatting
- mypy for static type checking

Assume:
- The project is managed by `pyproject.toml` with `uv`
- Code lives in `src/` (e.g., `src/agentic_document_extraction/...`)
- Tests live in `tests/`
- Environment is already bootstrapped with `uv init` and dependencies installed
- All environment variables (e.g., OpenAI API keys) are set already

If any of these tools or configurations are missing, you must:
1. Add them to `pyproject.toml` with the appropriate `uv` commands.
2. Generate minimal, working configuration files (e.g. `ruff.toml`, `mypy.ini` or `pyproject.toml` sections).
3. Run the tools at least once to confirm correct setup.

## Required Commands (Feedback Loops)

Always make sure that you are running in the virtual environment at `.venv/` created by `uv`.

Before you consider any story complete, you MUST ensure all of the following commands succeed:

1. Lint & format with ruff:
   - `uv run ruff check .`
   - `uv run ruff format .` (if formatting is configured via ruff)

2. Static type checking with mypy:
   - `uv run mypy src`

3. Tests with pytest (with coverage if configured):
   - `uv run pytest`
   - If coverage is configured: `uv run pytest --cov=src --cov-report=term-missing`

4. (Optional but preferred) Basic packaging sanity:
   - `uv run python -m compileall src`

If any of these commands fail, you must fix the issues and re-run them until they all pass.

## Tooling Setup Requirements

When bootstrapping a new repository, ensure the following steps are carried out (and codified in the repo):

1. Initialize a uv-based project:
   - `uv init agentic-document-extraction --package` (or equivalent command)
   - Confirm `pyproject.toml`, `uv.lock`, and `src/agentic_document_extraction/__init__.py` exist.

2. Add dev dependencies:
   - `uv add --dev pytest`
   - `uv add --dev ruff`
   - `uv add --dev mypy`
   - `uv add --dev pytest-cov`
   - `uv add --dev httpx` (for FastAPI testing)

3. Add production dependencies:
   - `uv add fastapi`
   - `uv add uvicorn[standard]`
   - `uv add langchain`
   - `uv add langchain-openai`
   - `uv add openai`
   - `uv add transformers` (for layout detection models)
   - `uv add torch` (PyTorch for transformers)
   - `uv add python-multipart` (for file uploads in FastAPI)
   - Additional dependencies as needed per story

4. Create or update configuration:
   - Configure `ruff` (either via `ruff.toml` or `[tool.ruff]` in `pyproject.toml`) for linting & formatting.
   - Configure `mypy` (via `mypy.ini`, `pyproject.toml`, or `mypy.cfg`) to:
     - Target Python 3.12
     - Enable strictness appropriate for this project (prefer strict or near-strict mode).
   - Configure `pytest` (e.g. `pytest.ini` or `pyproject.toml` `[tool.pytest.ini_options]`) so tests in `tests/` are discovered.

5. Add convenient scripts (if appropriate for this repo) in `pyproject.toml` or a `justfile` / `Makefile`:
   - `lint`: `uv run ruff check .`
   - `fmt`: `uv run ruff format .`
   - `typecheck`: `uv run mypy src`
   - `test`: `uv run pytest`
   - `test-cov`: `uv run pytest --cov=src --cov-report=term-missing`
   - `serve`: `uv run uvicorn agentic_document_extraction.api:app --reload`

Document these commands in the repository README so humans can reproduce them.

## User Stories

### Core Stories - Phase 1 (Foundation & API)

- [x] **Story 1.1**: FastAPI Service Bootstrap
  - As a developer, I want a working FastAPI service with health check endpoint, so that I have the foundation for the extraction API.
  - **Acceptance Criteria**:
    - FastAPI application initialized with proper structure
    - Health check endpoint (`GET /health`) returns service status
    - Service can be started with `uvicorn`
    - Basic logging configured
    - CORS configured appropriately
    - OpenAPI docs available at `/docs`
    - Tests for health endpoint using `httpx`

- [x] **Story 1.2**: Document Upload Endpoint
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

- [ ] **Story 1.3**: Document Format Detection & Classification
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

- [ ] **Story 1.4**: Schema Validation
  - As a system, I want to validate the user-provided JSON schema before processing, so that extraction has a valid target structure.
  - **Acceptance Criteria**:
    - Accepts JSON schema in standard JSON Schema format (Draft 7 or later)
    - Validates schema syntax before processing using `jsonschema`
    - Provides clear error messages for invalid schemas (returns 400 with details)
    - Supports common data types (string, number, boolean, object, array)
    - Supports nested structures
    - Extracts required fields and optional fields for extraction planning
    - Unit tests for valid and invalid schema cases

### Core Stories - Phase 2 (Text-Based Extraction)

- [ ] **Story 2.1**: Text Extraction from Text-Based Documents
  - As a system, I want to extract raw text from text-based documents, so that I can process them directly without visual conversion.
  - **Acceptance Criteria**:
    - Extracts text from .txt files preserving line breaks and structure
    - Detects and handles different encodings (UTF-8, Latin-1, etc.) using `chardet`
    - Extracts text from .csv files preserving tabular structure using `pandas`
    - Returns extracted text with metadata (encoding, line count, structure type)
    - Preserves natural reading order
    - Unit tests with various encodings and structures

- [ ] **Story 2.2**: LangChain LLM Integration for Text Extraction
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

- [ ] **Story 2.3**: JSON + Markdown Output (Text Documents)
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

- [ ] **Story 3.1**: Text Extraction from Visual Documents (OCR)
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

- [ ] **Story 3.2**: Layout Detection with Hugging Face Transformers
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

- [ ] **Story 3.3**: Reading Order Detection with LLM
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

- [ ] **Story 3.4**: Region-Based Visual Extraction with VLM
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

- [ ] **Story 3.5**: Visual Document Synthesis
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

- [ ] **Story 4.1**: Extraction Planning with LangChain Agent
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

- [ ] **Story 4.2**: Quality Verification Agent
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

- [ ] **Story 4.3**: Iterative Refinement Loop
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

- [ ] **Story 5.1**: Async Processing & Job Management
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

- [ ] **Story 5.2**: Error Handling & Logging
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

- [ ] **Story 5.3**: Configuration Management
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

- [ ] **Story 5.4**: API Documentation & Examples
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

## Technical Constraints

- Use **Python 3.12** syntax and stdlib features where appropriate.
- Write **fully typed** code with type hints everywhere (functions, methods, module boundaries, public APIs).
- Prefer small, composable functions and modules.
- Tests must be fast, deterministic, and isolated.
- **API Framework**: FastAPI with async endpoints where appropriate
- **LLM Orchestration**: LangChain with langchain-openai for all LLM/VLM interactions
- **LLM/VLM Provider**: OpenAI (GPT-4, GPT-4 Turbo, GPT-4V) with reasoning capabilities
- **Layout Detection**: Open-source models via Hugging Face `transformers` library
- **Processing Strategy**:
  - Text-based documents (txt, csv): Direct text extraction → LangChain LLM processing
  - Visual documents (pdf, images, office docs): OCR → Layout detection (HF transformers) → Reading order detection (LangChain + GPT-4) → Region segmentation → VLM processing (LangChain + GPT-4V) → Synthesis (LangChain)
- **Agentic architecture**: LangChain agents for planning, execution, verification, and refinement
- **Dependencies**:
  - API: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `pydantic-settings`
  - LLM/VLM: `langchain`, `langchain-openai`, `openai`
  - Layout detection: `transformers`, `torch`, `pillow`
  - Text extraction: `chardet`, `pandas`
  - OCR & PDF: `pypdf` or `pdfplumber`, `pytesseract`, `pdf2image`, `pillow`
  - Document conversion: `python-docx`, `python-pptx`, `openpyxl`
  - Schema validation: `jsonschema`
  - Testing: `pytest`, `pytest-cov`, `httpx`, `pytest-asyncio`
  - Logging: `structlog` (optional) or standard `logging`
- Avoid introducing unnecessary dependencies; if a dependency is justified, add it via `uv add` and explain its purpose in code comments or docstrings where relevant.

## Architecture Notes

### Service Architecture

```
FastAPI Service
    ↓
POST /extract (file + schema)
    ↓
Document Upload & Validation (Stories 1.2, 1.4)
    ↓
Format Detection (Story 1.3)
    ↓
    ├─→ TEXT-BASED PATH (Stories 2.1-2.3)
    │       ↓
    │   Text Extraction
    │       ↓
    │   LangChain + GPT-4 Extraction
    │       ↓
    │   JSON + Markdown Output
    │
    └─→ VISUAL PATH (Stories 3.1-3.5)
            ↓
        OCR / Text Extraction (3.1)
            ↓
        Layout Detection - HF Transformers (3.2)
            ↓
        Reading Order - LangChain + GPT-4 (3.3)
            ↓
        Region-Based VLM - LangChain + GPT-4V (3.4)
            ↓
        Synthesis - LangChain (3.5)
    ↓
LangChain Agentic Loop (Stories 4.1-4.3)
    ↓
Planning Agent → Execution → Verification Agent → Refinement (if needed)
    ↓
Response (JSON + Markdown + Metadata)
```

### Module Structure

```
src/agentic_document_extraction/
├── __init__.py
├── api.py                          # FastAPI application (Story 1.1, 1.2)
├── config.py                       # Configuration management (Story 5.3)
├── models.py                       # Pydantic models for requests/responses
├── services/
│   ├── __init__.py
│   ├── format_detector.py          # Story 1.3
│   ├── schema_validator.py         # Story 1.4
│   ├── text_extractor.py           # Story 2.1, 3.1
│   ├── layout_detector.py          # Story 3.2 (HF transformers)
│   ├── reading_order.py            # Story 3.3 (LangChain + GPT-4)
│   └── extraction/
│       ├── __init__.py
│       ├── text_extraction.py      # Story 2.2 (LangChain + GPT-4)
│       ├── visual_extraction.py    # Story 3.4 (LangChain + GPT-4V)
│       └── synthesis.py            # Story 3.5 (LangChain)
├── agents/
│   ├── __init__.py
│   ├── planner.py                  # Story 4.1
│   ├── verifier.py                 # Story 4.2
│   └── refiner.py                  # Story 4.3
├── output/
│   ├── __init__.py
│   ├── json_generator.py           # Story 2.3
│   └── markdown_generator.py       # Story 2.3
└── utils/
    ├── __init__.py
    ├── logging.py                  # Story 5.2
    └── exceptions.py               # Story 5.2

tests/
├── __init__.py
├── test_api.py
├── test_services/
│   ├── test_format_detector.py
│   ├── test_schema_validator.py
│   ├── test_text_extractor.py
│   ├── test_layout_detector.py
│   └── test_extraction/
├── test_agents/
├── test_output/
└── fixtures/
    ├── sample_documents/
    └── sample_schemas/
```

## Completion Criteria
You are done when you complete **EXACTLY ONE** story according to the following criteria.

For any story, it is **complete** only when:

1. All acceptance criteria for that story are satisfied
2. The following commands all succeed without errors:
   - `uv run ruff check .`
   - `uv run ruff format .` (no changes or only expected formatting changes)
   - `uv run mypy src`
   - `uv run pytest` (and, if configured, `uv run pytest --cov=src --cov-report=term-missing` with acceptable coverage)

3. There are no obvious TODO/FIXME comments left in critical paths for that story
4. Tests for the story are comprehensive and passing
5. The story is marked as complete (`[x]`) in the User Stories section
6. The completed story is committed to git with a descriptive commit message referencing the story number

## Current Progress

- Environment setup:
  - [x] uv project initialized
  - [x] pytest configured and passing
  - [x] ruff configured and passing
  - [x] mypy configured, targeting Python 3.12, and passing
  - [x] FastAPI dev server runnable

- Implemented stories:
  - [x] Story 1.1: FastAPI Service Bootstrap
  - [x] Story 1.2: Document Upload Endpoint
  - [ ] Story 1.3: Document Format Detection & Classification
  - [ ] Story 1.4: Schema Validation
  - [ ] Story 2.1: Text Extraction from Text-Based Documents
  - [ ] Story 2.2: LangChain LLM Integration for Text Extraction
  - [ ] Story 2.3: JSON + Markdown Output (Text Documents)
  - [ ] Story 3.1: Text Extraction from Visual Documents (OCR)
  - [ ] Story 3.2: Layout Detection with Hugging Face Transformers
  - [ ] Story 3.3: Reading Order Detection with LLM
  - [ ] Story 3.4: Region-Based Visual Extraction with VLM
  - [ ] Story 3.5: Visual Document Synthesis
  - [ ] Story 4.1: Extraction Planning with LangChain Agent
  - [ ] Story 4.2: Quality Verification Agent
  - [ ] Story 4.3: Iterative Refinement Loop
  - [ ] Story 5.1: Async Processing & Job Management
  - [ ] Story 5.2: Error Handling & Logging
  - [ ] Story 5.3: Configuration Management
  - [ ] Story 5.4: API Documentation & Examples
