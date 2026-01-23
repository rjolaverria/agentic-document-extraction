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

## Technical Constraints

- Use **Python 3.12** syntax and stdlib features where appropriate.
- Write **fully typed** code with type hints everywhere (functions, methods, module boundaries, public APIs).
- Prefer small, composable functions and modules.
- Tests must be fast, deterministic, and isolated.
- **API Framework**: FastAPI with async endpoints where appropriate
- **Document loading**: Use langchain `Document` and the document loaders for reading data from different sources.
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
