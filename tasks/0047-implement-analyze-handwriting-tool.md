# Task 0047: Implement AnalyzeHandwriting Tool for LangChain Agent

## Objective
Create a LangChain tool that allows the extraction agent to accurately transcribe handwritten text, annotations, and margin notes that standard OCR cannot handle reliably.

## Context
Handwritten content is common in real-world documents:
- Handwritten forms and applications
- Margin notes and annotations
- Corrections and edits
- Signatures with accompanying text
- Medical prescriptions
- Field notes

While PaddleOCR has some handwriting support, GPT-4V often performs better on difficult handwriting, especially with context awareness. The AnalyzeHandwriting tool provides:
- Better accuracy for challenging handwriting
- Context-aware transcription
- Annotation type classification
- Position/placement information

## Acceptance Criteria
- [ ] Create `AnalyzeHandwritingTool` class implementing LangChain `BaseTool`
- [ ] Tool accepts region_id parameter
- [ ] Crops image region using bounding boxes from layout detection
- [ ] Sends cropped image to GPT-4V with handwriting transcription prompt
- [ ] Returns structured output with:
  - Transcribed text
  - Confidence level (high/medium/low)
  - Annotation type (note, correction, answer, signature text, etc.)
  - Position/context information
  - Legibility assessment
- [ ] Tool description clearly explains when to use (for handwritten regions)
- [ ] Integration with PaddleOCR layout detection results
- [ ] Unit tests for tool functionality
- [ ] Integration tests with sample handwritten documents
- [ ] Error handling for invalid region IDs or failed VLM calls

## Tool Schema
```python
class AnalyzeHandwritingInput(BaseModel):
    region_id: str = Field(description="ID of the handwritten region to analyze")
    context: str | None = Field(
        default=None, 
        description="Optional surrounding context to help with transcription"
    )

class AnalyzeHandwritingOutput(BaseModel):
    transcribed_text: str
    confidence: str  # "high", "medium", "low"
    annotation_type: str | None  # "margin_note", "correction", "answer", "signature_text", "comment"
    position: str | None  # "top_margin", "bottom", "inline", etc.
    is_legible: bool  # False if handwriting is too unclear
    alternative_readings: list[str] | None  # Possible alternative transcriptions if uncertain
    notes: str | None  # Additional observations (e.g., "cursive", "printed", "mixed")
```

## Dependencies
- PaddleOCR layout detection with region bounding boxes
- GPT-4V API access (already configured)
- Image cropping utilities (shared with other tools)

## Implementation Notes
- Provide surrounding context to VLM when available (helps with ambiguous characters)
- Handle different handwriting styles: cursive, print, mixed
- Different languages and scripts
- Medical/technical terminology in context
- Consider multiple possible readings for unclear text
- May need higher resolution crops for small handwriting

## VLM Prompt Strategy
```
You are transcribing handwritten text. 

IMAGE: [cropped handwriting region]

SURROUNDING CONTEXT (if available): {context}

TASK:
1. Transcribe the handwritten text as accurately as possible
2. If uncertain about specific words/letters, provide alternative readings
3. Assess the legibility (is the handwriting clear enough to read?)
4. Identify the type of annotation (margin note, answer to question, correction, etc.)
5. Note the handwriting style (cursive, print, mixed)

Be especially careful with:
- Ambiguous letters (e.g., 'a' vs 'o', 'u' vs 'v', 'n' vs 'u')
- Numbers vs letters (e.g., '1' vs 'l', '0' vs 'O')
- Common abbreviations

If the handwriting is illegible, indicate this clearly rather than guessing.

Return structured transcription with confidence assessment.
```

## Testing Strategy
- Create test fixtures with various handwriting samples:
  - Clear printed handwriting
  - Cursive handwriting
  - Messy/hurried handwriting
  - Medical prescriptions
  - Form answers
  - Margin annotations
  - Different languages
  - Numbers and symbols
- Mock VLM responses for unit tests
- Integration tests with real GPT-4V calls
- Compare accuracy vs PaddleOCR on same samples
- Test edge cases: illegible text, mixed print/cursive, faded writing

## Use Cases
- Handwritten form submissions
- Annotated documents
- Medical prescriptions
- Field inspection reports
- Student assignments
- Surveys with written responses
- Historical document transcription
- Signed contracts with handwritten clauses
