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

## Task & Log Indexes

- Tasks are indexed in `TASKS.md` with individual task files stored in `tasks/`.
- Logs are indexed in `LOGS.md` with individual log files stored in `logs/`.
- When updating tasks or logs, add a new file and update the corresponding index.

If any of these tools or configurations are missing, you must:
1. Add them to `pyproject.toml` with the appropriate `uv` commands.
2. Generate minimal, working configuration files (e.g. `ruff.toml`, `mypy.ini` or `pyproject.toml` sections).
3. Run the tools at least once to confirm correct setup.

## Architecture Notes

Please refer to the architecture notes below for understanding the overall system design, module structure, and technical constraints. Update it as needed when making changes.

### Service Architecture

```
FastAPI Service
    ↓
POST /extract (file + schema)
    ↓
Document Upload & Validation
    ↓
Format Detection
    ↓
    ├─→ TEXT-BASED PATH
    │       ↓
    │   Text Extraction
    │       ↓
    │   LangChain + GPT-4 Extraction
    │       ↓
    │   JSON + Markdown Output
    │
    └─→ VISUAL PATH
            ↓
        PaddleOCR Text Extraction
        • Text strings with bounding boxes
        • Confidence scores
            ↓
        PaddleOCR Layout Detection
        • Tables, charts, text blocks
        • Region bounding boxes
            ↓
        LayoutReader Reading Order
        • Determines reading order
        • Handles multi-column layouts
            ↓
        Tool-Based Extraction Agent
        • Single LangChain agent
        • Receives OCR text + regions + schema
        • Invokes analyze_chart/analyze_table tools
            ↓
        Lightweight Verification Loop
        • Rule-based quality checks
        • Iterative refinement if needed
    ↓
Response (JSON + Markdown + Quality Report)
```

### Tool-Based Extraction Architecture

The extraction system uses a single LangChain agent with specialized tools:

```
┌─────────────────────────────────────────────────────────────┐
│  ExtractionAgent (LangChain)                                │
│                                                             │
│  System Prompt contains:                                    │
│  • All OCR text (reading order)                             │
│  • Layout region IDs and types                              │
│  • Target JSON schema                                       │
│  • Tool descriptions                                        │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ analyze_chart    │  │ analyze_table    │                │
│  │ Tool             │  │ Tool             │                │
│  │                  │  │                  │                │
│  │ Sends cropped    │  │ Sends cropped    │                │
│  │ image to VLM     │  │ image to VLM     │                │
│  │                  │  │                  │                │
│  │ Returns:         │  │ Returns:         │                │
│  │ • Chart type     │  │ • Headers        │                │
│  │ • Axes           │  │ • Rows           │                │
│  │ • Data points    │  │ • Values         │                │
│  │ • Trends         │  │ • Notes          │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  Lightweight Verification Loop:                             │
│  1. Extract using tool-based agent                          │
│  2. Verify quality (rule-based, no LLM call)                │
│  3. If issues found, provide feedback and re-extract        │
│  4. Return best result with quality metrics                 │
└─────────────────────────────────────────────────────────────┘
    ↓
Structured JSON Output
```

### Module Structure

```
src/agentic_document_extraction/
├── __init__.py
├── api.py                          # FastAPI application and endpoints
├── config.py                       # Configuration management
├── models.py                       # Pydantic models (FormatInfo, ProcessingCategory)
├── docket_tasks.py                 # Docket async task definitions
├── services/
│   ├── __init__.py
│   ├── format_detector.py          # Document format detection
│   ├── schema_validator.py         # JSON schema validation and metadata
│   ├── text_extractor.py           # Text-based document extraction
│   ├── visual_text_extractor.py    # PaddleOCR text extraction for visual docs
│   ├── layout_detector.py          # PaddleOCR layout region detection
│   ├── reading_order_detector.py   # LayoutReader reading order detection
│   ├── extraction_processor.py     # Main extraction orchestration
│   ├── docket_client.py            # Docket API client
│   ├── docket_jobs.py              # Job management
│   └── extraction/
│       ├── __init__.py
│       ├── text_extraction.py      # LLM-based text extraction (ExtractionResult)
│       ├── visual_document_extraction.py  # Visual document processing
│       ├── region_visual_extraction.py    # Region-level extraction
│       └── synthesis.py            # Multi-region result synthesis
├── agents/
│   ├── __init__.py
│   ├── extraction_agent.py         # PRIMARY: Tool-based extraction agent
│   ├── planner.py                  # Extraction planning (data models)
│   ├── verifier.py                 # Quality verification (rule-based + LLM)
│   ├── refiner.py                  # Refinement agent and AgenticLoopResult
│   └── tools/
│       ├── __init__.py
│       ├── analyze_chart.py        # Chart/graph analysis VLM tool
│       ├── analyze_table.py        # Table analysis VLM tool
│       ├── vlm_utils.py            # VLM integration utilities
│       └── region_utils.py         # Region processing utilities
├── output/
│   ├── __init__.py
│   ├── json_generator.py           # JSON output generation
│   ├── markdown_generator.py       # Markdown output generation
│   └── output_service.py           # Output service orchestration
└── utils/
    ├── __init__.py
    ├── agent_helpers.py            # LangChain agent utilities
    ├── logging.py                  # Structured logging
    └── exceptions.py               # Custom exception types

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

## Technical Constraints

- Use **Python 3.12** syntax and stdlib features where appropriate.
- Write **fully typed** code with type hints everywhere (functions, methods, module boundaries, public APIs).
- Prefer small, composable functions and modules.
- Tests must be fast, deterministic, and isolated.
- **API Framework**: FastAPI with async endpoints where appropriate
- **Document loading**: Use langchain `Document` and the document loaders for reading data from different sources.
- **LLM Orchestration**: LangChain with langchain-openai for all LLM/VLM interactions
- **LLM/VLM Provider**: OpenAI (GPT-4o, GPT-4 Turbo) for extraction and vision tasks
- **OCR & Layout Detection**: PaddleOCR for both text extraction and layout region detection
- **Reading Order**: LayoutReader model for determining reading order of detected regions
- **Processing Strategy**:
  - Text-based documents (txt, csv): Direct text extraction → LangChain LLM processing
  - Visual documents (pdf, images, office docs): PaddleOCR text extraction → PaddleOCR layout detection → LayoutReader reading order → Tool-based extraction agent with VLM tools
- **Agentic architecture**: Single tool-based LangChain agent with lightweight verification loop
  - `ExtractionAgent`: Primary agent with analyze_chart/analyze_table tools
  - `QualityVerificationAgent`: Rule-based quality verification
  - Iterative refinement based on quality feedback
- **Job Processing**: Docket for async job management with Redis backend
- **Dependencies**:
  - API: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `pydantic-settings`
  - LLM/VLM: `langchain`, `langchain-openai`, `openai`
  - OCR & Layout: `paddleocr`, `paddlepaddle`, `pillow`
  - Reading Order: `transformers`, `torch` (for LayoutReader model)
  - PDF Processing: `pypdf`, `pdfplumber`, `pdf2image`
  - Text extraction: `chardet`, `pandas`
  - Document conversion: `python-docx`, `python-pptx`, `openpyxl`
  - Schema validation: `jsonschema`
  - Job Management: `docket`
  - Testing: `pytest`, `pytest-cov`, `httpx`, `pytest-asyncio`
  - Logging: standard `logging` with structured output
- Avoid introducing unnecessary dependencies; if a dependency is justified, add it via `uv add` and explain its purpose in code comments or docstrings where relevant.
