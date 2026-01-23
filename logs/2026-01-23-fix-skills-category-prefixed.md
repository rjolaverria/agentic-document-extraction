# Fix: Skills Extracted as Category-Prefixed Strings

**Date:** 2026-01-23
**Task:** tasks/0027-issue-skills-category-prefixed.md
**Status:** Completed

## Issue Description

When extracting skills from resumes, the system was producing category-prefixed strings like:
- `"Languages: Python, JavaScript, TypeScript, Go, SQL"`
- `"Frameworks: FastAPI, React, Node.js, Django"`

Instead of individual skill items:
- `["Python", "JavaScript", "TypeScript", "Go", "SQL", "FastAPI", "React", "Node.js", "Django"]`

This made downstream parsing of individual skills unnecessarily complicated.

## Root Cause

The extraction prompts did not include specific instructions for handling array fields that contain comma-separated or category-prefixed values. The LLM was preserving the original formatting from the source document rather than splitting values into individual array items.

## Solution

Enhanced the extraction prompts in three services to include explicit "Array Handling Rules":

### 1. Text Extraction (`text_extraction.py`)

Added rules 7-11 to `EXTRACTION_SYSTEM_PROMPT`:
```
ARRAY HANDLING RULES:
When extracting into an array of strings (e.g., skills, tags, keywords):
7. Split comma-separated, semicolon-separated, or line-separated lists into individual array items
8. Remove category prefixes (e.g., "Languages: Python, JavaScript" becomes ["Python", "JavaScript"])
9. Remove bullet points, dashes, or other list markers from individual items
10. Trim whitespace from each item
11. Each element should be a single, atomic value - not a grouped or prefixed string
```

Also included an explicit example:
```
Example: If text shows "Skills: Languages: Python, JavaScript - Frameworks: React, FastAPI"
The skills array should be: ["Python", "JavaScript", "React", "FastAPI"]
NOT: ["Languages: Python, JavaScript", "Frameworks: React, FastAPI"]
```

### 2. Synthesis Service (`synthesis.py`)

Added rules 10-14 to `SYNTHESIS_SYSTEM_PROMPT` with the same array handling rules.

### 3. Refinement Agent (`refiner.py`)

Added rules 8-12 to `REFINEMENT_SYSTEM_PROMPT` with the same array handling rules.

## Testing

### Unit Test Added

Added `test_extract_skills_as_individual_items` to `test_text_extraction.py`:
- Verifies skills are extracted as individual items without category prefixes
- Checks that no skill contains `:` (category separator)
- Checks that no skill contains `,` (list separator)
- Validates specific expected skills are present

### Integration Test

Manually verified end-to-end extraction with sample resume:
- Input skills from resume:
  ```
  - Languages: Python, JavaScript, TypeScript, Go, SQL
  - Frameworks: FastAPI, React, Node.js, Django
  - Cloud: AWS, GCP, Docker, Kubernetes
  - Databases: PostgreSQL, MongoDB, Redis
  - Tools: Git, GitHub Actions, Terraform
  ```
- Output skills array: 19 individual items correctly extracted

## Verification

All checks pass:
- `uv run ruff check .` - All checks passed
- `uv run ruff format .` - No changes (already formatted)
- `uv run mypy src` - No issues found
- `uv run pytest` - 833 tests passed
- `uv run pytest --cov=src --cov-report=term-missing` - 93% coverage
- Manual end-to-end test with sample resume - Skills correctly split

## Files Modified

1. `src/agentic_document_extraction/services/extraction/text_extraction.py` - Added array handling rules to EXTRACTION_SYSTEM_PROMPT
2. `src/agentic_document_extraction/services/extraction/synthesis.py` - Added array handling rules to SYNTHESIS_SYSTEM_PROMPT
3. `src/agentic_document_extraction/agents/refiner.py` - Added array handling rules to REFINEMENT_SYSTEM_PROMPT
4. `tests/test_services/test_extraction/test_text_extraction.py` - Added test_extract_skills_as_individual_items

## Impact

- Downstream systems can now reliably iterate over individual skills without parsing
- Skills are directly usable for matching, filtering, or categorization
- No breaking changes to the API contract
