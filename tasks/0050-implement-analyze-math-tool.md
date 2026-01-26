# Task 0050: Implement AnalyzeMath Tool for LangChain Agent

## Objective
Create a LangChain tool that allows the extraction agent to accurately extract mathematical equations, chemical formulas, and scientific notation that standard OCR cannot handle reliably.

## Context
Mathematical and scientific content has special formatting and symbols:
- Mathematical equations and formulas
- Chemical formulas and reactions
- Scientific notation
- Subscripts and superscripts
- Special mathematical symbols (integrals, summations, Greek letters)
- Matrix notation

Standard OCR struggles with:
- Complex mathematical layout (fractions, exponents)
- Special symbols and operators
- Spatial relationships (subscripts, superscripts)
- Multi-line equations

The AnalyzeMath tool enables the agent to:
- Accurately transcribe equations
- Convert to LaTeX for machine-readable format
- Preserve mathematical structure
- Extract chemical formulas correctly
- Provide plain-text descriptions

## Acceptance Criteria
- [ ] Create `AnalyzeMathTool` class implementing LangChain `BaseTool`
- [ ] Tool accepts region_id parameter
- [ ] Crops image region using bounding boxes from layout detection
- [ ] Sends cropped image to GPT-4V with math extraction prompt
- [ ] Returns structured output with:
  - LaTeX representation
  - Plain text description
  - Content type (equation, formula, notation)
  - Variables and their meanings (if inferable)
- [ ] Tool description clearly explains when to use (for math regions)
- [ ] Integration with PaddleOCR layout detection results
- [ ] Unit tests for tool functionality
- [ ] Integration tests with sample scientific documents
- [ ] Error handling for invalid region IDs or failed VLM calls

## Tool Schema
```python
class AnalyzeMathInput(BaseModel):
    region_id: str = Field(description="ID of the mathematical content region to analyze")

class AnalyzeMathOutput(BaseModel):
    content_type: str  # "equation", "chemical_formula", "matrix", "notation", "mixed"
    latex: str  # LaTeX representation
    plain_text: str  # Human-readable description
    variables: dict[str, str] | None  # Variable definitions if inferable (e.g., {"x": "unknown", "E": "energy"})
    notes: str | None  # Additional context or observations
```

## Dependencies
- PaddleOCR layout detection with region bounding boxes
- GPT-4V API access (already configured)
- Image cropping utilities (shared with other tools)

## Implementation Notes
- Prioritize LaTeX output for machine-readable format
- Handle different math notation styles (inline, display)
- Support chemical formulas with proper subscripts/superscripts
- Recognize common scientific notation patterns
- Handle multi-line equations and equation arrays
- Consider equation numbering and labels

## VLM Prompt Strategy
```
You are transcribing mathematical or scientific content.

IMAGE: [cropped math region]

TASK:
1. Identify the content type:
   - Mathematical equation
   - Chemical formula
   - Matrix/vector notation
   - Scientific notation
   - Mixed content

2. Convert to LaTeX:
   - Use standard LaTeX math syntax
   - Preserve structure (fractions, exponents, subscripts, etc.)
   - Include special symbols correctly
   - For chemical formulas, use proper notation

3. Provide plain text description:
   - Describe what the equation represents
   - Explain in human-readable terms

4. Identify variables:
   - List key variables and their likely meanings (if clear from context)

EXAMPLES:
- Quadratic formula: \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
- Water: H_2O or \ce{H2O}
- Pythagorean theorem: a^2 + b^2 = c^2

Be precise with LaTeX syntax. If uncertain, indicate ambiguity.

Return structured transcription with LaTeX and description.
```

## Testing Strategy
- Create test fixtures with various mathematical content:
  - Simple equations (linear, quadratic)
  - Complex equations (integrals, derivatives, summations)
  - Chemical formulas (simple and complex)
  - Matrices and vectors
  - Greek letters and special symbols
  - Multi-line equation systems
  - Physical formulas with units
- Mock VLM responses for unit tests
- Integration tests with real GPT-4V calls
- Validate LaTeX output (can it be rendered?)
- Test accuracy: symbol recognition, structure preservation
- Test edge cases: handwritten equations, low-quality scans

## Use Cases
- Scientific paper extraction
- Patent document processing
- Academic textbook digitization
- Chemistry lab reports
- Physics problem sets
- Engineering documentation
- Educational material extraction
- Research data extraction
