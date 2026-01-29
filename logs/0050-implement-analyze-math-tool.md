# Log: Implement AnalyzeMath Tool for LangChain Agent

**Date:** 2026-01-29
**Task:** 0050-implement-analyze-math-tool.md

## Summary

Implemented the `AnalyzeMath` tool for the LangChain extraction agent. This tool enables accurate extraction of mathematical equations, chemical formulas, matrices, and scientific notation that standard OCR struggles to handle.

## Changes Made

### New Files

1. **`src/agentic_document_extraction/agents/tools/analyze_math.py`**
   - Created `AnalyzeMath` tool implementing LangChain `@tool` decorator
   - Accepts `region_id` parameter to identify the mathematical content region
   - Uses VLM (GPT-4V) to analyze cropped image regions
   - Returns structured output with:
     - `content_type`: equation, chemical_formula, matrix, notation, or mixed
     - `latex`: LaTeX representation of the mathematical content
     - `plain_text`: Human-readable description
     - `variables`: Dictionary mapping variable symbols to their meanings
     - `notes`: Additional observations or context
   - Includes `analyze_math_impl` function for direct calling
   - Includes `_normalize_math_result` helper for response validation

2. **`tests/test_agents/test_analyze_math_tool.py`**
   - 38 comprehensive unit tests covering:
     - Detection of equations (quadratic formula, integrals, derivatives, summations)
     - Detection of chemical formulas (simple and complex reactions)
     - Detection of matrices
     - Detection of scientific notation (E=mc², F=ma)
     - Detection of Greek letters and mixed content
     - Error handling (invalid region ID, VLM failure, missing images)
     - JSON response parsing (wrapped and direct)
     - Normalization of all output fields
     - Integration tests with real VLM (skipped by default)

### Modified Files

1. **`src/agentic_document_extraction/agents/tools/__init__.py`**
   - Added exports for `AnalyzeMath`, `analyze_math`, `analyze_math_impl`

2. **`src/agentic_document_extraction/agents/extraction_agent.py`**
   - Added import for `analyze_math`
   - Added `analyze_math` to the tools list (now 9 tools total)
   - Added tool instruction #10 for math content analysis

3. **`tests/test_agents/test_extraction_agent.py`**
   - Updated tool count assertion from 8 to 9

## Technical Details

### VLM Prompt Strategy

The tool uses a detailed prompt that instructs the VLM to:
1. Identify the content type (equation, chemical formula, matrix, notation, mixed)
2. Convert to standard LaTeX syntax with proper symbols and structure
3. Provide plain text description of what the content represents
4. Identify key variables and their likely meanings
5. Return structured JSON output

### Supported Content Types

- **equation**: Mathematical equations and formulas (quadratic formula, calculus)
- **chemical_formula**: Chemical formulas and reactions (H₂O, combustion reactions)
- **matrix**: Matrix and vector notation using `\begin{bmatrix}`
- **notation**: Scientific notation with physical units (E=mc², F=ma)
- **mixed**: Documents containing multiple types of mathematical content

### Expected Region Types

The tool adds a warning note if the region type is not one of:
- `FORMULA` (primary expected type)
- `TEXT` (inline equations)
- `PICTURE` (scanned equation images)

## Test Results

All 1144 tests pass with 87% overall coverage. The new `analyze_math.py` file has 95% coverage.

## Acceptance Criteria Status

- [x] Create `AnalyzeMathTool` class implementing LangChain `BaseTool`
- [x] Tool accepts region_id parameter
- [x] Crops image region using bounding boxes from layout detection
- [x] Sends cropped image to GPT-4V with math extraction prompt
- [x] Returns structured output (LaTeX, plain text, content type, variables)
- [x] Tool description clearly explains when to use
- [x] Integration with PaddleOCR layout detection results
- [x] Unit tests for tool functionality
- [x] Integration tests with sample documents (mocked + real API)
- [x] Error handling for invalid region IDs or failed VLM calls
