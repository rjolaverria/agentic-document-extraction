# Log Entry: Improve Tool Descriptions for Autonomous Selection

**Date:** 2026-01-30
**Task:** 0053 - Improve Tool Descriptions for Autonomous Selection

## Summary

Enhanced all 9 tool descriptions to be self-documenting so the LLM agent can make informed decisions about which tools to use without external guidance.

## Changes Made

Updated the docstrings for all 9 agent tools in `src/agentic_document_extraction/agents/tools/`:

1. **analyze_chart.py** - Updated `analyze_chart` (715 chars)
2. **analyze_table.py** - Updated `analyze_table` (560 chars)
3. **analyze_diagram.py** - Updated `analyze_diagram` (678 chars)
4. **analyze_form.py** - Updated `analyze_form` (656 chars)
5. **analyze_handwriting.py** - Updated `analyze_handwriting` (693 chars)
6. **analyze_image.py** - Updated `analyze_image` (800 chars)
7. **analyze_logo.py** - Updated `analyze_logo` (626 chars)
8. **analyze_math.py** - Updated `analyze_math` (605 chars)
9. **analyze_signature.py** - Updated `analyze_signature` (624 chars)

## Tool Description Template

Each tool description now follows a consistent template:

```
[Brief one-line summary]

Use this tool when:
- [Condition 1]
- [Condition 2]
- [Condition 3]

Do NOT use when:
- [Anti-condition 1]
- [Anti-condition 2]

Args:
    region_id: The ID from the Document Regions table

Returns:
    JSON with [key fields]
```

## Example: analyze_chart

```python
"""Extract structured data from chart or graph images.

Use this tool when:
- The region_type is PICTURE and contains a chart, graph, or plot
- You need axis labels, data points, trends, or legend information
- The OCR text does not contain the numerical data visible in the chart

Do NOT use when:
- The region is a TABLE (use analyze_table_agent instead)
- The image is a diagram, flowchart, or org chart (use analyze_diagram_agent)
- The image is a photo or illustration (use analyze_image_agent)

Args:
    region_id: The ID from the Document Regions table (e.g., "region_3")

Returns:
    JSON with chart_type, title, x_axis, y_axis, key_data_points, trends, legend
"""
```

## Verification

- All 1144 tests pass
- Ruff linting: all checks passed
- Ruff formatting: no changes needed (85 files already formatted)
- Mypy: no issues found in 45 source files
- Tool descriptions verified accessible via LangChain tool interface

## Benefits

1. **Clearer Use Cases**: Each tool now explicitly states when it should be used
2. **Avoid Conflicts**: "Do NOT use when" sections prevent incorrect tool selection
3. **Reduced VLM Calls**: Clear guidance helps avoid unnecessary VLM invocations
4. **Self-Documenting**: Agent can make autonomous decisions based on descriptions
5. **Consistent Format**: All tools follow the same documentation pattern

## Notes

- Descriptions are concise (560-800 chars) to avoid excessive token usage
- Each description includes example region_id format for clarity
- Cross-references between tools help agent choose the right tool for edge cases
