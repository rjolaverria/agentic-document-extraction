# Agentic Document Extraction

Vision-first, agentic document extraction system as a FastAPI service.

## Setup

```bash
# Install dependencies
uv sync

# Run the development server
uv run uvicorn agentic_document_extraction.api:app --reload
```

## Commands

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type checking
uv run mypy src

# Tests
uv run pytest

# Tests with coverage
uv run pytest --cov=src --cov-report=term-missing
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /extract` - Upload document and schema for extraction
- `GET /docs` - OpenAPI documentation
