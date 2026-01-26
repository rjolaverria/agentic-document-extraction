# Task 0045: Implement AnalyzeSignature Tool for LangChain Agent

## Objective
Create a LangChain tool that allows the extraction agent to analyze signature blocks, stamps, seals, and watermarks for document validation and authentication.

## Context
Signatures and official stamps are critical for document verification but cannot be extracted with standard OCR:
- Signature presence/absence detection
- Printed signer names and titles
- Date and location of signing
- Official stamps and company seals
- Certification marks and watermarks
- Notary stamps

The AnalyzeSignature tool enables the agent to:
- Detect signature presence
- Extract associated text (printed names, dates, titles)
- Identify stamp types and content
- Validate signature block completeness

## Acceptance Criteria
- [ ] Create `AnalyzeSignatureTool` class implementing LangChain `BaseTool`
- [ ] Tool accepts region_id parameter
- [ ] Crops image region using bounding boxes from layout detection
- [ ] Sends cropped image to GPT-4V with signature analysis prompt
- [ ] Returns structured output with:
  - Signature present (boolean)
  - Signer name (if printed)
  - Title/role (if present)
  - Date signed (if present)
  - Stamp/seal present (boolean)
  - Stamp text/content
  - Certification marks
  - Completeness assessment
- [ ] Tool description clearly explains when to use (for signature regions)
- [ ] Integration with PaddleOCR layout detection results
- [ ] Unit tests for tool functionality
- [ ] Integration tests with sample signed documents
- [ ] Error handling for invalid region IDs or failed VLM calls

## Tool Schema
```python
class AnalyzeSignatureInput(BaseModel):
    region_id: str = Field(description="ID of the signature/stamp region to analyze")

class AnalyzeSignatureOutput(BaseModel):
    signature_present: bool
    signer_name: str | None  # Printed name if present
    signer_title: str | None  # Title/role (e.g., "CEO", "Witness")
    date_signed: str | None  # Date if present
    location: str | None  # Location if present (e.g., "New York, NY")
    
    stamp_present: bool
    stamp_type: str | None  # "company seal", "notary", "certification", "watermark"
    stamp_text: str | None  # Text content of stamp
    
    certification_marks: list[str] | None  # e.g., ["ISO 9001", "Certified Copy"]
    
    is_complete: bool  # All expected elements present
    missing_elements: list[str] | None  # e.g., ["date", "printed name"]
    notes: str | None  # Additional observations
```

## Dependencies
- PaddleOCR layout detection with region bounding boxes
- GPT-4V API access (already configured)
- Image cropping utilities (shared with other tools)

## Implementation Notes
- Signature detection is visual pattern recognition
- Look for handwritten signatures vs digital signatures
- Extract associated printed information
- Identify official stamps by visual characteristics (circular, rectangular, ornate borders)
- Check for standard signature block elements (name line, date line, title line)
- Handle multiple signatures in one region (e.g., dual signers, witness signatures)

## VLM Prompt Strategy
```
You are analyzing a signature block region. Identify:

1. SIGNATURE:
   - Is a handwritten signature present?
   - Is there a printed name below/near the signature?
   - Is there a title or role indicated?
   - Is there a date?
   - Is there a location?

2. STAMPS/SEALS:
   - Are there any official stamps or seals?
   - What type? (company seal, notary stamp, certification mark)
   - What text appears on the stamp?

3. COMPLETENESS:
   - Are all expected signature block elements present?
   - What is missing (if anything)?

Return structured information about all elements found.
```

## Testing Strategy
- Create test fixtures with various signature types:
  - Simple signature with printed name
  - Signature with date and title
  - Notarized documents with notary stamp
  - Documents with company seals
  - Multi-signer documents
  - Electronic/digital signatures
  - Incomplete signature blocks
- Mock VLM responses for unit tests
- Integration tests with real GPT-4V calls
- Test accuracy: presence detection, text extraction from stamps
- Test edge cases: faded signatures, overlapping stamps

## Use Cases
- Contracts and agreements
- Legal documents
- Financial documents (checks, promissory notes)
- Official certificates
- Notarized documents
- Government forms
- Employment agreements
- Real estate documents
