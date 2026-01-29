# Task 0044: Implement AnalyzeForm Tool for LangChain Agent

## Objective
Create a LangChain tool that allows the extraction agent to analyze form regions, including checkboxes, radio buttons, dropdown selections, and handwritten entries.

## Context
Forms are ubiquitous in business documents and contain elements that standard OCR cannot handle:
- Checkbox states (checked/unchecked) are visual, not textual
- Radio button selections
- Handwritten entries in form fields
- Field labels and their associated values
- Form structure and field relationships

The AnalyzeForm tool enables the agent to extract structured data from forms by:
- Taking a region ID (from layout detection)
- Cropping the form region
- Sending to GPT-4V for visual analysis
- Extracting field labels, values, types, and states

## Acceptance Criteria
- [x] Create `AnalyzeFormTool` class implementing LangChain `BaseTool`
- [x] Tool accepts region_id parameter
- [x] Crops image region using bounding boxes from layout detection
- [x] Sends cropped image to GPT-4V with form extraction prompt
- [x] Returns structured output with fields array containing:
  - Field label/name
  - Field value (text, checkbox state, selection)
  - Field type (text, checkbox, radio, dropdown, signature)
  - Handwritten flag (boolean)
  - Required flag (if detectable)
- [x] Tool description clearly explains when to use (for form regions)
- [x] Integration with PaddleOCR layout detection results
- [x] Unit tests for tool functionality
- [x] Integration tests with sample forms
- [x] Error handling for invalid region IDs or failed VLM calls

## Tool Schema
```python
class AnalyzeFormInput(BaseModel):
    region_id: str = Field(description="ID of the form region to analyze")

class FormField(BaseModel):
    label: str  # Field label/question
    value: str | bool | None  # Field value (text, True/False for checkboxes)
    field_type: str  # "text", "checkbox", "radio", "dropdown", "signature", "date"
    is_handwritten: bool = False  # True if value appears handwritten
    is_required: bool | None = None  # True if marked as required (*)
    position: str | None = None  # e.g., "top-left", "middle-right" for spatial context

class AnalyzeFormOutput(BaseModel):
    fields: list[FormField]
    form_title: str | None  # Form title if present
    notes: str | None  # Additional observations
```

## Dependencies
- PaddleOCR layout detection with region bounding boxes
- GPT-4V API access (already configured)
- Image cropping utilities (shared with other tools)

## Implementation Notes
- Use detailed prompt instructing GPT-4V to identify all form elements
- Request specific checkbox state detection (checked/unchecked vs empty)
- Ask for handwriting detection on filled values
- Handle multi-column forms
- Preserve field order (top-to-bottom, left-to-right)
- Consider form field grouping (sections)

## VLM Prompt Strategy
```
You are analyzing a form region. Extract all form fields including:
- Text input fields (labels and filled values)
- Checkboxes (label and state: checked/unchecked)
- Radio buttons (options and selected value)
- Dropdown selections
- Signature fields (presence/absence)
- Date fields

For each field, identify:
1. The label/question text
2. The filled value or selection state
3. Whether the value is handwritten
4. The field type
5. Whether it appears required (marked with * or "required")

Return a structured list of all fields.
```

## Testing Strategy
- Create test fixtures with various form types:
  - Simple text forms
  - Forms with checkboxes and radio buttons
  - Forms with handwritten entries
  - Multi-column forms
  - Application forms
  - Survey forms
- Mock VLM responses for unit tests
- Integration tests with real GPT-4V calls
- Test error cases: invalid region_id, non-form regions, VLM failures
- Test accuracy: checkbox state detection, handwriting recognition

## Use Cases
- Job applications
- Tax forms
- Survey responses
- Medical intake forms
- Registration forms
- Questionnaires
- Insurance claim forms
