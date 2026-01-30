# Task 0054: Add Tool Usage to System Prompt for All Documents

## Objective
Update the system prompt to always include tool usage instructions when tools are provided, regardless of document type.

## Context
Currently, the tool instructions (`_TOOL_INSTRUCTIONS_VISUAL`) are only added to the system prompt when `has_tools=True`:

```python
tool_instructions = _TOOL_INSTRUCTIONS_VISUAL if has_tools else ""
```

After Task 0051, tools will always be provided to the agent. The system prompt should:
1. Always include tool instructions when tools are available
2. Explain that tools are available but may not be needed for all documents
3. Guide the agent on when to skip tool usage (e.g., simple text extraction)

## Acceptance Criteria
- [x] Always include tool instructions in system prompt when tools are provided
- [x] Update instructions to explain tools are optional based on document content
- [x] Add guidance for when to skip tool usage
- [x] Simplify the prompt logic (remove conditional has_tools checks where possible)
- [x] Unit tests pass
- [x] Agent correctly decides when to use/skip tools

## Current Implementation

### extraction_agent.py (lines 156-176)
```python
_TOOL_INSTRUCTIONS_VISUAL = """\
5. When a schema field likely comes from a chart/graph region, call
   `analyze_chart` with that region's `region_id`.
6. When a schema field likely comes from a diagram (flowchart, org chart,
   process diagram), call `analyze_diagram` with that region's `region_id`.
...
"""
```

### _build_system_prompt (lines 480-498)
```python
def _build_system_prompt(
    self,
    text: str,
    schema_info: SchemaInfo,
    regions: list[LayoutRegion],
    *,
    has_tools: bool,
) -> str:
    region_section = self._build_region_section(regions)
    tool_instructions = _TOOL_INSTRUCTIONS_VISUAL if has_tools else ""
    ...
```

## Proposed Changes

### Updated Tool Instructions
```python
_TOOL_INSTRUCTIONS = """\
## Visual Analysis Tools
You have access to visual analysis tools for extracting data from images.
Use these tools ONLY when the OCR text is insufficient for a schema field.

**When to use tools:**
- The Document Regions table shows PICTURE or TABLE regions
- A schema field requires data visible in an image (chart, table, form, etc.)
- OCR text is missing, garbled, or incomplete for visual content

**When NOT to use tools:**
- All required data is already in the OCR text
- The document has no PICTURE or TABLE regions
- Simple text extraction is sufficient

**Available tools:**
- `analyze_chart`: Extract data from charts and graphs
- `analyze_table`: Extract structured data from tables
- `analyze_diagram`: Extract flowcharts, org charts, process diagrams
- `analyze_form`: Extract form fields, checkboxes, values
- `analyze_handwriting`: Transcribe handwritten text
- `analyze_image`: Describe photos, illustrations, general images
- `analyze_logo`: Identify logos, certifications, brand marks
- `analyze_math`: Extract equations, formulas, scientific notation
- `analyze_signature`: Extract signature blocks and stamps

Call tools with the `region_id` from the Document Regions table.
You may call multiple tools for different regions as needed.
"""
```

### Simplified Prompt Building
```python
def _build_system_prompt(
    self,
    text: str,
    schema_info: SchemaInfo,
    regions: list[LayoutRegion],
) -> str:
    """Assemble the initial extraction system prompt."""
    region_section = self._build_region_section(regions)
    # Always include tool instructions - agent decides when to use
    tool_instructions = _TOOL_INSTRUCTIONS if regions else ""
    ocr_text = self._truncate_text(text)

    return _SYSTEM_PROMPT_TEMPLATE.format(
        ocr_text=ocr_text,
        region_section=region_section,
        schema_json=json.dumps(schema_info.schema, indent=2),
        tool_instructions=tool_instructions,
    )
```

## Files to Modify
- `src/agentic_document_extraction/agents/extraction_agent.py`
  - Update `_TOOL_INSTRUCTIONS_VISUAL` constant (or rename to `_TOOL_INSTRUCTIONS`)
  - Update `_build_system_prompt` method
  - Update `_build_refinement_prompt` method
  - Remove `has_tools` parameter if no longer needed

## Dependencies
- Task 0051: Refactor agent to always provide all tools (should be completed first)

## Testing Strategy
- Test extraction on text-only documents (agent should not call tools)
- Test extraction on documents with visual regions (agent should call tools)
- Verify prompt content includes tool instructions
- Check agent reasoning in logs for tool selection decisions

## Notes
- The tool instructions are now more of a "reference guide" rather than "instructions to follow"
- The agent should use its judgment based on the document content
- Clear "when to use" and "when NOT to use" guidance prevents unnecessary VLM calls
