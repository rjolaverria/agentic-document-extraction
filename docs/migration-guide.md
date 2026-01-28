# Migration Guide: Multi-Agent to Tool-Based Architecture

This guide describes the architectural changes from the original multi-agent design to the new tool-based extraction architecture.

## Overview of Changes

The system has evolved from a multi-agent orchestration pattern to a simpler, more efficient tool-based approach:

| Aspect | Old Architecture | New Architecture |
|--------|------------------|------------------|
| **Agent Pattern** | Multi-agent (planner + verifier + refiner) | Single tool-using agent |
| **Verification** | LLM-based verification agent | Rule-based verification (fast) |
| **Visual Analysis** | Region-based VLM extraction | Tool-based VLM calls |
| **OCR** | Tesseract | PaddleOCR |
| **Layout Detection** | HuggingFace Transformers | PaddleOCR Layout |
| **Reading Order** | LLM-based | LayoutReader model |
| **Job Processing** | Custom job manager | Docket |

## Key Component Changes

### 1. Extraction Agent

**Before**: Three separate agents
- `ExtractionPlanningAgent`: Created extraction plans
- `QualityVerificationAgent`: Verified extraction quality with LLM
- `RefinementAgent`: Improved extractions iteratively
- `AgenticLoop`: Orchestrated all agents

**After**: Single `ExtractionAgent`
- Combines extraction + verification in one component
- Uses tools (`analyze_chart`, `analyze_table`) for visual regions
- Lightweight verification loop (rule-based, no extra LLM calls)
- Iterates with quality feedback until convergence

### 2. Visual Document Processing

**Before**:
```
OCR → Layout Detection → Region Extraction → VLM per region → Synthesis
```

**After**:
```
PaddleOCR (text + layout) → LayoutReader → Single Agent with tools
```

The agent receives all OCR text and region metadata upfront, and selectively calls tools when it needs to analyze specific regions.

### 3. Verification

**Before**: LLM-based verification
- Called LLM to analyze extraction quality
- Expensive and slow

**After**: Rule-based verification
- Schema compliance checks
- Required field coverage
- Confidence thresholds
- Optional LLM analysis (disabled by default)

## Code Migration

### Using ExtractionAgent

```python
from agentic_document_extraction.agents.extraction_agent import ExtractionAgent
from agentic_document_extraction.services.schema_validator import SchemaInfo
from agentic_document_extraction.models import FormatInfo

# Create agent
agent = ExtractionAgent(
    model="gpt-4o",
    max_iterations=3,
    use_llm_verification=False,  # Rule-based by default
)

# Run extraction with verification loop
result = agent.extract(
    text=ocr_text,
    schema_info=schema_info,
    format_info=format_info,
    layout_regions=regions,  # Optional for visual docs
)

# Access results
print(result.final_result.extracted_data)
print(result.final_verification.status)
print(result.converged)
```

### Deprecated Components

The following components are deprecated but kept for reference:

| Component | Status | Replacement |
|-----------|--------|-------------|
| `AgenticLoop` | Deprecated | `ExtractionAgent.extract()` |
| `RefinementAgent.refine()` | Deprecated | Built into ExtractionAgent |
| `RegionVisualExtractor` | Kept | Used internally by tools |
| `SynthesisService` | Kept | Used for multi-region merging |

### Tool Development

To add new visual analysis tools:

1. Create tool in `agents/tools/`:
```python
from langchain.tools import tool
from agentic_document_extraction.agents.tools.vlm_utils import call_vlm_with_image

@tool
def analyze_form(region_id: str) -> str:
    """Analyze a form region and extract field-value pairs."""
    # Implementation using VLM
    ...
```

2. Register in `ExtractionAgent`:
```python
# In extraction_agent.py, add to tools list
tools = [analyze_chart, analyze_table, analyze_form]
```

## Configuration Changes

### Environment Variables

New variables:
- `ADE_LAYOUTREADER_MODEL`: LayoutReader model name (default: `hantian/layoutreader`)
- `ADE_PADDLEOCR_USE_GPU`: Enable GPU for PaddleOCR (default: `false`)

Removed variables:
- `ADE_TESSERACT_PATH`: No longer needed (using PaddleOCR)
- `ADE_HF_LAYOUT_MODEL`: No longer needed (using PaddleOCR layout)

## Performance Improvements

| Metric | Before | After |
|--------|--------|-------|
| Verification calls | 1 LLM call per iteration | 0 LLM calls (rule-based) |
| Visual region analysis | Sequential VLM calls | On-demand tool calls |
| Reading order detection | LLM-based | LayoutReader (local model) |

## Backward Compatibility

The `AgenticLoopResult` response format is preserved for API compatibility:

```python
@dataclass
class AgenticLoopResult:
    final_result: ExtractionResult
    final_verification: VerificationReport
    plan: ExtractionPlan
    iterations_completed: int
    iteration_history: list[IterationMetrics]
    converged: bool
    best_iteration: int
    total_tokens: int
    total_processing_time_seconds: float
    loop_metadata: dict[str, Any]
```

The `ExtractionAgent` wraps its output in this format for seamless integration with existing code.
