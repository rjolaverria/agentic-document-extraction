# Task 0053: Improve Tool Descriptions for Autonomous Selection

## Objective
Enhance tool descriptions to be self-documenting so the LLM can make informed decisions about which tools to use without external guidance.

## Context
When the agent has access to all tools (after Task 0051), it relies on tool descriptions to decide when to use each tool. Current descriptions are minimal:

```python
def analyze_chart(...):
    """Analyze a chart/graph region by its ID to extract structured data
    (chart type, axes, data points, trends). Use when OCR text alone
    cannot capture chart information."""
```

LangChain best practices recommend detailed descriptions that include:
- Clear use cases (when to call this tool)
- Region types the tool handles
- What information can be extracted
- When NOT to use the tool (to avoid unnecessary VLM calls)

## Acceptance Criteria
- [ ] Update all 9 tool descriptions with comprehensive guidance
- [ ] Each description includes:
  - Primary use case
  - Region types it handles
  - What data it extracts
  - When NOT to use it
  - Input parameters explained
  - Output format summary
- [ ] Descriptions are concise but complete (avoid excessive length)
- [ ] Unit tests pass
- [ ] Agent correctly selects tools based on descriptions

## Tool Description Template

```python
"""[Brief one-line summary].

Use this tool when:
- [Condition 1]
- [Condition 2]
- [Condition 3]

Do NOT use when:
- [Anti-condition 1]
- [Anti-condition 2]

Args:
    region_id: The region ID from the Document Regions table

Returns:
    JSON object with [key fields]
"""
```

## Files to Modify
- `src/agentic_document_extraction/agents/tools/analyze_chart.py`
- `src/agentic_document_extraction/agents/tools/analyze_diagram.py`
- `src/agentic_document_extraction/agents/tools/analyze_form.py`
- `src/agentic_document_extraction/agents/tools/analyze_handwriting.py`
- `src/agentic_document_extraction/agents/tools/analyze_image.py`
- `src/agentic_document_extraction/agents/tools/analyze_logo.py`
- `src/agentic_document_extraction/agents/tools/analyze_math.py`
- `src/agentic_document_extraction/agents/tools/analyze_signature.py`
- `src/agentic_document_extraction/agents/tools/analyze_table.py`

## Improved Descriptions

### analyze_chart
```python
"""Extract structured data from chart or graph images.

Use this tool when:
- The region_type is PICTURE and contains a chart, graph, or plot
- You need axis labels, data points, trends, or legend information
- The OCR text does not contain the numerical data visible in the chart

Do NOT use when:
- The region is a TABLE (use analyze_table instead)
- The image is a diagram, flowchart, or org chart (use analyze_diagram)
- The image is a photo or illustration (use analyze_image)

Args:
    region_id: The ID from the Document Regions table (e.g., "region_3")

Returns:
    JSON with chart_type, title, x_axis, y_axis, key_data_points, trends, legend
"""
```

### analyze_table
```python
"""Extract structured tabular data from table images.

Use this tool when:
- The region_type is TABLE
- You need to extract rows, columns, headers, and cell values
- The OCR text shows table content but lacks structure

Do NOT use when:
- The region is a chart or graph (use analyze_chart)
- The content is a form with labels and fields (use analyze_form)

Args:
    region_id: The ID from the Document Regions table

Returns:
    JSON with headers (list), rows (list of lists), and notes
"""
```

### analyze_diagram
```python
"""Extract structure and relationships from diagrams, flowcharts, and org charts.

Use this tool when:
- The region_type is PICTURE and contains a flowchart, process diagram, or org chart
- You need to extract nodes, connections, flow direction, or hierarchy
- The OCR text captures labels but not the visual relationships

Do NOT use when:
- The image is a data chart or graph (use analyze_chart)
- The image is a photo or illustration (use analyze_image)

Args:
    region_id: The ID from the Document Regions table

Returns:
    JSON with diagram_type, nodes, connections, flow_direction, hierarchy
"""
```

### analyze_form
```python
"""Extract form fields, checkboxes, and filled values from form images.

Use this tool when:
- The region contains a form with labeled fields, checkboxes, or radio buttons
- You need to extract field labels paired with their values
- The form may have handwritten entries in fields

Do NOT use when:
- The content is a simple table without form elements (use analyze_table)
- The content is purely handwritten text (use analyze_handwriting)

Args:
    region_id: The ID from the Document Regions table

Returns:
    JSON with form_type, fields (list of label/value pairs), checkboxes, notes
"""
```

### analyze_handwriting
```python
"""Recognize and transcribe handwritten text from images.

Use this tool when:
- The region contains handwritten text that OCR could not read accurately
- You need to transcribe handwritten notes, annotations, or entries
- The text style is cursive, print, or mixed handwriting

Do NOT use when:
- The text is typed/printed (OCR should handle it)
- The content is a signature (use analyze_signature)

Args:
    region_id: The ID from the Document Regions table

Returns:
    JSON with transcription, confidence, writing_style, language
"""
```

### analyze_image
```python
"""Analyze photos, illustrations, or general images for content description.

Use this tool when:
- The region_type is PICTURE and contains a photo, illustration, or artwork
- You need to identify objects, people, scenes, or assess image content
- You need to count items or describe visual attributes

Do NOT use when:
- The image is a chart or graph (use analyze_chart)
- The image is a diagram or flowchart (use analyze_diagram)
- The image is a logo or seal (use analyze_logo)

Args:
    region_id: The ID from the Document Regions table
    focus: Optional specific aspect to analyze (e.g., "count items", "identify brand")

Returns:
    JSON with description, objects_identified, scene_type, attributes
"""
```

### analyze_logo
```python
"""Identify logos, certification badges, seals, and brand marks.

Use this tool when:
- The region contains a company logo, brand mark, or trademark
- You need to identify certification badges (ISO, FDA, CE, etc.)
- The region contains official seals or stamps (not signatures)

Do NOT use when:
- The image is a signature (use analyze_signature)
- The image is a general photo (use analyze_image)

Args:
    region_id: The ID from the Document Regions table

Returns:
    JSON with logo_type, organization, certification_type, text_in_logo
"""
```

### analyze_math
```python
"""Extract mathematical equations, formulas, and scientific notation.

Use this tool when:
- The region contains mathematical equations or formulas
- You need LaTeX representation of complex math notation
- The content includes chemical formulas, matrices, or scientific notation

Do NOT use when:
- The content is regular text with simple numbers
- The math is simple enough to read from OCR text (e.g., "2 + 2 = 4")

Args:
    region_id: The ID from the Document Regions table

Returns:
    JSON with content_type, latex, plain_text, variables
"""
```

### analyze_signature
```python
"""Extract signature blocks, stamps, and authentication marks.

Use this tool when:
- The region contains handwritten signatures
- You need to identify official stamps or seals on signatures
- The region has a signature block with printed name and date

Do NOT use when:
- The content is a logo or certification badge (use analyze_logo)
- The content is general handwriting (use analyze_handwriting)

Args:
    region_id: The ID from the Document Regions table

Returns:
    JSON with signature_type, signer_name, date, has_stamp, stamp_text
"""
```

## Dependencies
None - can be done in parallel with Task 0051.

## Testing Strategy
- Review each description for clarity and completeness
- Test that agent selects correct tools for various document types
- Verify no regression in extraction quality
- Test edge cases where tool selection could be ambiguous

## Notes
- Keep descriptions under ~300 characters for the summary line
- The detailed "Use this tool when" and "Do NOT use when" sections help the LLM make decisions
- Consider adding examples in the future if needed
