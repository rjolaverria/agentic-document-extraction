# Agentic Document Extraction

A vision-first, agentic document extraction system built as a FastAPI service. This system intelligently processes documents and extracts structured information according to user-defined JSON schemas using AI-powered extraction with LangChain and OpenAI.

## Features

- **Multi-format support**: Process text files, CSVs, PDFs, images, Office documents, and more
- **Vision-first approach**: Treats visual documents (PDFs, images) as visual objects, understanding layout and spatial relationships
- **Agentic processing**: Uses plan→execute→verify→refine loops to iteratively improve extraction quality
- **Async job processing**: Long-running extractions run in the background with status tracking
- **Structured output**: Returns both JSON (matching your schema) and Markdown summaries
- **Quality verification**: Built-in quality checks with configurable confidence thresholds

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key
- **Tesseract OCR** (required for image/visual document processing)

#### Installing Tesseract OCR

Tesseract is required for processing image-based documents (PNG, JPG, TIFF, etc.). The Python `pytesseract` package is a wrapper that calls the tesseract binary.

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download the installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.

**Verify installation:**
```bash
tesseract --version
```

> **Note:** By default, Tesseract includes English (`eng`), orientation/script detection (`osd`), and `snum` language data. For additional languages, install `tesseract-lang` (macOS: `brew install tesseract-lang`).

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd agentic-document-extraction

# Install dependencies
uv sync

# Create .env file with your configuration
cat > .env << EOF
ADE_OPENAI_API_KEY=sk-your-api-key-here
ADE_LOG_LEVEL=INFO
ADE_DEBUG=false
EOF
```

### Running the Server

```bash
# Start the development server
uv run uvicorn agentic_document_extraction.api:app --reload

# Or with custom host/port
uv run uvicorn agentic_document_extraction.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive documentation is at `http://localhost:8000/docs`.

## API Endpoints

### Health Check

```
GET /health
```

Check if the service is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000000+00:00",
  "version": "0.1.0"
}
```

### Extract Document

```
POST /extract
```

Upload a document and JSON schema to start extraction. Returns immediately with a job ID for async processing.

**Request (multipart/form-data):**
- `file` (required): The document file to extract from
- `schema` (optional): JSON schema as a string
- `schema_file` (optional): JSON schema as a file upload

One of `schema` or `schema_file` must be provided.

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "invoice.pdf",
  "file_size": 102400,
  "status": "pending",
  "message": "Document uploaded successfully. Processing will begin shortly."
}
```

### Get Job Status

```
GET /jobs/{job_id}
```

Check the status of an extraction job.

**Response (200 OK):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "filename": "invoice.pdf",
  "created_at": "2024-01-15T10:30:00.000000+00:00",
  "updated_at": "2024-01-15T10:30:05.000000+00:00",
  "progress": "Analyzing document layout (page 2 of 3)",
  "error_message": null
}
```

**Job Status Values:**
- `pending`: Job created, waiting to start
- `processing`: Extraction in progress
- `completed`: Extraction finished successfully
- `failed`: Extraction failed (check error_message)

### Get Job Result

```
GET /jobs/{job_id}/result
```

Retrieve the extraction results for a completed job.

**Response (200 OK):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "extracted_data": {
    "invoice_number": "INV-2024-001",
    "date": "2024-01-15",
    "total_amount": 1250.00,
    "line_items": [
      {"description": "Widget A", "quantity": 5, "price": 100.00},
      {"description": "Widget B", "quantity": 10, "price": 75.00}
    ]
  },
  "markdown_summary": "# Invoice Extraction Summary\n\n**Invoice Number:** INV-2024-001\n...",
  "metadata": {
    "processing_time_seconds": 12.5,
    "model_used": "gpt-4o",
    "total_tokens": 2500,
    "iterations_completed": 2,
    "converged": true,
    "document_type": "visual"
  },
  "quality_report": {
    "overall_confidence": 0.92,
    "field_scores": {...},
    "issues": []
  },
  "error_message": null
}
```

## Example Usage

### Using curl

**Extract data from a text file:**
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@document.txt" \
  -F 'schema={"type":"object","properties":{"title":{"type":"string"},"summary":{"type":"string"}},"required":["title"]}'
```

**Extract data from a PDF with schema file:**
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@invoice.pdf" \
  -F "schema_file=@invoice_schema.json"
```

**Check job status:**
```bash
curl http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000
```

**Get extraction results:**
```bash
curl http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000/result
```

### Using Python

```python
import httpx
import time

BASE_URL = "http://localhost:8000"

# Define the extraction schema
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string", "format": "date"},
        "vendor_name": {"type": "string"},
        "total_amount": {"type": "number"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "unit_price": {"type": "number"},
                    "total": {"type": "number"}
                },
                "required": ["description", "quantity", "unit_price"]
            }
        }
    },
    "required": ["invoice_number", "date", "total_amount"]
}

# Upload document and start extraction
with open("invoice.pdf", "rb") as f:
    response = httpx.post(
        f"{BASE_URL}/extract",
        files={"file": ("invoice.pdf", f, "application/pdf")},
        data={"schema": json.dumps(schema)}
    )
    job = response.json()
    job_id = job["job_id"]
    print(f"Job started: {job_id}")

# Poll for completion
while True:
    status_response = httpx.get(f"{BASE_URL}/jobs/{job_id}")
    status = status_response.json()

    if status["status"] == "completed":
        print("Extraction complete!")
        break
    elif status["status"] == "failed":
        print(f"Extraction failed: {status['error_message']}")
        break
    else:
        print(f"Status: {status['status']} - {status.get('progress', 'Processing...')}")
        time.sleep(2)

# Get results
result_response = httpx.get(f"{BASE_URL}/jobs/{job_id}/result")
result = result_response.json()

print("Extracted data:")
print(json.dumps(result["extracted_data"], indent=2))

print("\nMarkdown summary:")
print(result["markdown_summary"])
```

### Async Python Client

```python
import asyncio
import httpx
import json

async def extract_document(file_path: str, schema: dict) -> dict:
    """Extract data from a document using the ADE API."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Upload and start extraction
        with open(file_path, "rb") as f:
            response = await client.post(
                "/extract",
                files={"file": (file_path, f)},
                data={"schema": json.dumps(schema)}
            )
            job = response.json()

        # Poll for completion
        job_id = job["job_id"]
        while True:
            status = (await client.get(f"/jobs/{job_id}")).json()

            if status["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(1)

        # Return results
        return (await client.get(f"/jobs/{job_id}/result")).json()

# Usage
schema = {"type": "object", "properties": {"title": {"type": "string"}}}
result = asyncio.run(extract_document("document.pdf", schema))
```

## Supported Document Formats

### Text-Based Documents (Direct Processing)
| Format | Extensions | Description |
|--------|-----------|-------------|
| Plain Text | `.txt` | UTF-8/Latin-1 text files |
| CSV | `.csv` | Comma-separated values |

### Visual Documents (OCR + Layout Analysis)
| Format | Extensions | Description |
|--------|-----------|-------------|
| PDF | `.pdf` | Portable Document Format |
| Word | `.doc`, `.docx` | Microsoft Word documents |
| OpenDocument | `.odt` | OpenDocument text |
| PowerPoint | `.ppt`, `.pptx` | Microsoft PowerPoint |
| OpenDocument | `.odp` | OpenDocument presentations |
| Excel | `.xlsx` | Microsoft Excel spreadsheets |
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.psd` | Various image formats |

## JSON Schema Specification

The extraction schema follows [JSON Schema Draft 7](https://json-schema.org/draft-07/json-schema-release-notes.html) specification.

### Basic Structure

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "field_name": {
      "type": "string",
      "description": "Description helps the AI understand what to extract"
    }
  },
  "required": ["field_name"]
}
```

### Supported Types

- `string`: Text values (supports `format`: `date`, `date-time`, `email`, `uri`)
- `number`: Decimal numbers
- `integer`: Whole numbers
- `boolean`: True/false values
- `array`: Lists of items
- `object`: Nested structures
- `null`: Null values

### Example Schemas

**Invoice Schema:**
```json
{
  "type": "object",
  "properties": {
    "invoice_number": {
      "type": "string",
      "description": "The unique invoice identifier"
    },
    "invoice_date": {
      "type": "string",
      "format": "date",
      "description": "Date the invoice was issued"
    },
    "due_date": {
      "type": "string",
      "format": "date",
      "description": "Payment due date"
    },
    "vendor": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "address": {"type": "string"},
        "tax_id": {"type": "string"}
      },
      "required": ["name"]
    },
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": {"type": "string"},
          "quantity": {"type": "number"},
          "unit_price": {"type": "number"},
          "total": {"type": "number"}
        },
        "required": ["description", "quantity", "unit_price"]
      }
    },
    "subtotal": {"type": "number"},
    "tax": {"type": "number"},
    "total": {"type": "number"}
  },
  "required": ["invoice_number", "invoice_date", "total"]
}
```

**Resume/CV Schema:**
```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string", "description": "Full name of the candidate"},
    "email": {"type": "string", "format": "email"},
    "phone": {"type": "string"},
    "summary": {"type": "string", "description": "Professional summary or objective"},
    "experience": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "company": {"type": "string"},
          "title": {"type": "string"},
          "start_date": {"type": "string"},
          "end_date": {"type": "string"},
          "description": {"type": "string"}
        },
        "required": ["company", "title"]
      }
    },
    "education": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "institution": {"type": "string"},
          "degree": {"type": "string"},
          "field": {"type": "string"},
          "graduation_date": {"type": "string"}
        },
        "required": ["institution", "degree"]
      }
    },
    "skills": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["name"]
}
```

### Schema Tips

1. **Use descriptions**: The `description` field helps the AI understand context
2. **Mark required fields**: Use `required` array to specify mandatory fields
3. **Use appropriate types**: Use `number` for decimals, `integer` for whole numbers
4. **Nest logically**: Group related fields into nested objects
5. **Use arrays for lists**: Line items, skills, etc. should be arrays

## Configuration

All configuration is done via environment variables with the `ADE_` prefix. Create a `.env` file in the project root or set variables directly.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ADE_OPENAI_API_KEY` | (required) | OpenAI API key for LLM features |
| `ADE_OPENAI_MODEL` | `gpt-4o` | Default model for text processing |
| `ADE_OPENAI_VLM_MODEL` | `gpt-4o` | Model for vision/multimodal processing |
| `ADE_OPENAI_TEMPERATURE` | `0.0` | LLM temperature (0.0 = deterministic) |
| `ADE_OPENAI_MAX_TOKENS` | `4096` | Max tokens for LLM responses |
| `ADE_MAX_FILE_SIZE_MB` | `10` | Maximum file upload size in MB |
| `ADE_TEMP_UPLOAD_DIR` | `/tmp/ade_uploads` | Temporary file storage directory |
| `ADE_CHUNK_SIZE` | `4000` | Token chunk size for large documents |
| `ADE_CHUNK_OVERLAP` | `200` | Token overlap between chunks |
| `ADE_MIN_OVERALL_CONFIDENCE` | `0.7` | Minimum overall confidence threshold |
| `ADE_MIN_FIELD_CONFIDENCE` | `0.5` | Minimum per-field confidence |
| `ADE_REQUIRED_FIELD_COVERAGE` | `0.9` | Required field coverage threshold |
| `ADE_MAX_REFINEMENT_ITERATIONS` | `3` | Max agentic loop iterations (1-10) |
| `ADE_JOB_TTL_HOURS` | `24` | Job result retention time in hours |
| `ADE_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `ADE_DEBUG` | `false` | Enable debug mode |
| `ADE_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `ADE_SERVER_HOST` | `0.0.0.0` | Server bind host |
| `ADE_SERVER_PORT` | `8000` | Server bind port |

### Example .env File

```bash
# Required
ADE_OPENAI_API_KEY=sk-your-api-key-here

# Recommended for production
ADE_LOG_LEVEL=INFO
ADE_DEBUG=false
ADE_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
ADE_MAX_FILE_SIZE_MB=50

# Quality tuning
ADE_MIN_OVERALL_CONFIDENCE=0.8
ADE_MAX_REFINEMENT_ITERATIONS=5
```

## Error Handling

The API returns structured error responses:

```json
{
  "detail": "Human-readable error message",
  "error_code": "E1001",
  "details": {"field": "additional context"},
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| E1001 | 400 | Validation error (missing/invalid fields) |
| E1002 | 400 | Schema parsing error (invalid JSON) |
| E1003 | 413 | File too large |
| E2001 | 500 | Document processing error |
| E2002 | 500 | OCR failure |
| E3001 | 500 | LLM API error |
| E5001 | 500 | Internal server error |

## API Testing

### HTTP Test Collection

The repository includes an `api.http` file for testing the API with popular HTTP clients:

- **VS Code REST Client**: Install the [REST Client extension](https://marketplace.visualstudio.com/items?itemName=humao.rest-client)
- **JetBrains HTTP Client**: Built into IntelliJ IDEA, PyCharm, and WebStorm
- **httpYac**: CLI and VS Code extension at [httpyac.github.io](https://httpyac.github.io/)

The collection includes tests for:
- Health check endpoint
- Document extraction with inline schemas
- Job status and result retrieval
- Error cases (missing schema, invalid JSON, job not found)

### Sample Fixtures

Sample documents and schemas are available in `tests/fixtures/`:

**Sample Documents** (`tests/fixtures/sample_documents/`):
- `sample_invoice.txt` - Text invoice for testing
- `sample_resume.txt` - Resume/CV document
- `sample_data.csv` - Employee data in CSV format

**Sample Schemas** (`tests/fixtures/sample_schemas/`):
- `invoice_schema.json` - Schema for invoice extraction
- `resume_schema.json` - Schema for resume/CV extraction
- `employee_data_schema.json` - Schema for employee data extraction
- `simple_schema.json` - Minimal schema for basic extraction

Use with curl:
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@tests/fixtures/sample_documents/sample_invoice.txt" \
  -F "schema_file=@tests/fixtures/sample_schemas/invoice_schema.json"
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_api.py -v
```

### Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type checking
uv run mypy src
```

### Project Structure

```
src/agentic_document_extraction/
├── api.py                     # FastAPI application
├── config.py                  # Configuration management
├── models.py                  # Pydantic models
├── services/
│   ├── format_detector.py     # Document format detection
│   ├── schema_validator.py    # JSON schema validation
│   ├── text_extractor.py      # Text extraction
│   ├── layout_detector.py     # Visual layout detection
│   ├── reading_order.py       # Reading order detection
│   └── extraction/
│       ├── text_extraction.py # LLM text extraction
│       ├── visual_extraction.py # VLM visual extraction
│       └── synthesis.py       # Result synthesis
├── agents/
│   ├── planner.py            # Extraction planning agent
│   ├── verifier.py           # Quality verification agent
│   └── refiner.py            # Iterative refinement agent
├── output/
│   ├── json_generator.py     # JSON output generation
│   └── markdown_generator.py # Markdown output generation
└── utils/
    ├── logging.py            # Structured logging
    └── exceptions.py         # Custom exceptions
```

## License

[Add your license here]
