# Agentic Document Extraction - Architecture

This document describes the architecture of the Agentic Document Extraction system.

## High-Level Overview

```
                    ┌─────────────────────────────────┐
                    │        FastAPI Service          │
                    │    POST /extract (file+schema)  │
                    └─────────────────┬───────────────┘
                                      │
                    ┌─────────────────▼───────────────┐
                    │   Document Upload & Validation   │
                    │   • File size check              │
                    │   • Schema validation            │
                    └─────────────────┬───────────────┘
                                      │
                    ┌─────────────────▼───────────────┐
                    │       Format Detection           │
                    │   • MIME type detection          │
                    │   • Processing category          │
                    └─────────────────┬───────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       │                       ▼
    ┌─────────────────┐               │         ┌─────────────────────┐
    │  TEXT-BASED     │               │         │     VISUAL PATH     │
    │  (txt, csv)     │               │         │ (pdf, images, docs) │
    └────────┬────────┘               │         └──────────┬──────────┘
             │                        │                    │
             ▼                        │                    ▼
    ┌─────────────────┐               │      ┌────────────────────────┐
    │ Text Extraction │               │      │ PaddleOCR Text Extract │
    │ (chardet, csv)  │               │      │ • Bounding boxes       │
    └────────┬────────┘               │      │ • Confidence scores    │
             │                        │      └───────────┬────────────┘
             │                        │                  │
             │                        │                  ▼
             │                        │      ┌────────────────────────┐
             │                        │      │ PaddleOCR Layout Detect│
             │                        │      │ • Tables               │
             │                        │      │ • Charts (PICTURE)     │
             │                        │      │ • Text blocks          │
             │                        │      └───────────┬────────────┘
             │                        │                  │
             │                        │                  ▼
             │                        │      ┌────────────────────────┐
             │                        │      │ LayoutReader Order     │
             │                        │      │ • Multi-column layout  │
             │                        │      │ • Reading sequence     │
             │                        │      └───────────┬────────────┘
             │                        │                  │
             └────────────────────────┼──────────────────┘
                                      │
                    ┌─────────────────▼───────────────┐
                    │   Tool-Based Extraction Agent    │
                    │                                  │
                    │   System Prompt:                 │
                    │   • OCR text (reading order)     │
                    │   • Region metadata              │
                    │   • Target JSON schema           │
                    │                                  │
                    │   Tools:                         │
                    │   • analyze_chart (VLM)          │
                    │   • analyze_table (VLM)          │
                    └─────────────────┬───────────────┘
                                      │
                    ┌─────────────────▼───────────────┐
                    │   Lightweight Verification Loop  │
                    │                                  │
                    │   1. Extract (tool-based agent)  │
                    │   2. Verify (rule-based checks)  │
                    │   3. If issues: feedback + retry │
                    │   4. Return best result          │
                    └─────────────────┬───────────────┘
                                      │
                    ┌─────────────────▼───────────────┐
                    │        Response Generation       │
                    │   • JSON (schema-compliant)      │
                    │   • Markdown summary             │
                    │   • Quality report               │
                    └─────────────────────────────────┘
```

## Component Details

### 1. Document Processing Pipeline

#### Text-Based Documents
- **Formats**: `.txt`, `.csv`
- **Processing**: Direct text extraction using chardet for encoding detection
- **Output**: Raw text fed to LLM extraction

#### Visual Documents
- **Formats**: `.pdf`, `.png`, `.jpg`, `.docx`, `.pptx`, etc.
- **Processing**:
  1. **PaddleOCR Text Extraction**: Extracts text with bounding boxes and confidence
  2. **PaddleOCR Layout Detection**: Identifies regions (tables, charts, text blocks)
  3. **LayoutReader**: Determines correct reading order for multi-column layouts
  4. **Tool-Based Extraction**: LangChain agent with visual analysis tools

### 2. Extraction Agent Architecture

The `ExtractionAgent` is a single LangChain agent that:
- Receives OCR text, layout regions, and target schema in its system prompt
- Autonomously invokes tools when visual regions need analysis
- Returns structured JSON matching the target schema

#### Available Tools

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `analyze_chart` | Analyze chart/graph regions | region_id | chart_type, axes, data_points, trends |
| `analyze_table` | Analyze table regions | region_id | headers, rows, notes |

### 3. Verification Loop

The lightweight verification loop performs:

1. **Rule-based checks** (fast, no LLM call):
   - Required field coverage
   - Schema type validation
   - Confidence thresholds
   - Null value detection

2. **Iterative refinement**:
   - Issues are formatted as feedback
   - Agent re-extracts with issue context
   - Best result tracked across iterations

3. **Quality metrics**:
   - Overall confidence
   - Required field coverage
   - Completeness score
   - Consistency score

### 4. Job Processing

- **Docket**: Async job management with Redis backend
- **Worker**: Separate process executes extraction jobs
- **API**: Returns job ID immediately, client polls for status

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `ExtractionAgent` | `agents/extraction_agent.py` | Primary tool-based extraction agent |
| `QualityVerificationAgent` | `agents/verifier.py` | Quality verification with metrics |
| `LayoutDetector` | `services/layout_detector.py` | PaddleOCR layout analysis |
| `ReadingOrderDetector` | `services/reading_order_detector.py` | LayoutReader integration |
| `ExtractionProcessor` | `services/extraction_processor.py` | Main orchestration |

## Data Flow

```
Input Document
    │
    ▼
FormatInfo (format_family, processing_category)
    │
    ▼
[Visual Path Only]
LayoutRegion[] (region_id, type, bbox, confidence, image)
    │
    ▼
OCR Text (reading order)
    │
    ▼
ExtractionAgent.extract(text, schema_info, format_info, regions)
    │
    ▼
AgenticLoopResult
    ├── final_result: ExtractionResult
    │   ├── extracted_data: dict
    │   └── field_extractions: list[FieldExtraction]
    ├── final_verification: VerificationReport
    │   ├── status: PASSED | FAILED | NEEDS_IMPROVEMENT
    │   ├── metrics: QualityMetrics
    │   └── issues: list[VerificationIssue]
    ├── iterations_completed: int
    └── converged: bool
```
