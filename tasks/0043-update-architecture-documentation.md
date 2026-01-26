# Task 0043: Update Architecture Documentation

## Objective
Update all architecture documentation to reflect the new tool-based agentic architecture and remove references to deprecated multi-agent approach.

## Context
After implementing the new architecture (Tasks 0037-0042), the documentation needs comprehensive updates:
- AGENT.md - Architecture notes and module structure
- README.md - System description and features
- Code docstrings and module-level documentation
- Architecture diagrams (add the new diagram)

## Acceptance Criteria
- [ ] Update AGENT.md with new architecture diagram and flow
- [ ] Update module structure in AGENT.md to reflect new components
- [ ] Update README.md system description
- [ ] Update README.md features to highlight tool-based approach
- [ ] Add architecture diagram image to repository
- [ ] Update technical constraints in AGENT.md
- [ ] Remove references to deprecated components:
  - Old planning/verification/refinement agents
  - Region-based visual extraction service
  - Synthesis service
  - LLM-based reading order detection
- [ ] Update code docstrings in new components
- [ ] Create migration guide for developers
- [ ] Update API documentation if response formats changed
- [ ] Review and update all task files for consistency

## New Architecture Diagram
```
Input Document
    ↓
┌─────────────────────────────────────┐
│  Text Extraction (PaddleOCR)        │
│  • Text strings                     │
│  • Bounding boxes                   │
│  • Confidence scores                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Region Detection                   │
│  (PaddleOCR Layout Detect)          │
│  • Tables                           │
│  • Charts                           │
│  • Text blocks                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Order Detection (LayoutReader)     │
│  • Determines reading order of      │
│    text regions                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  LangChain Extraction Agent                         │
│                                                      │
│  System Prompt:                                     │
│  • All OCR text (ordered)                           │
│  • Layout region IDs and types                      │
│  • Target JSON schema                               │
│  • Tool descriptions                                │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────┐       │
│  │ AnalyzeChart     │  │ AnalyzeTable     │       │
│  │ Tool             │  │ Tool             │       │
│  │                  │  │                  │       │
│  │ Sends cropped    │  │ Sends cropped    │       │
│  │ image to VLM     │  │ image to VLM     │       │
│  │                  │  │                  │       │
│  │ Returns:         │  │ Returns:         │       │
│  │ • Chart type     │  │ • Headers        │       │
│  │ • Axes           │  │ • Rows           │       │
│  │ • Data points    │  │ • Values         │       │
│  │ • Trends         │  │ • Notes          │       │
│  └──────────────────┘  └──────────────────┘       │
└─────────────────────────────────────────────────────┘
    ↓
Structured JSON Output
```

## Updated Module Structure
```
src/agentic_document_extraction/
├── services/
│   ├── text_extractor.py          # Text-based document extraction
│   ├── visual_text_extractor.py   # PaddleOCR with layout detection
│   ├── reading_order_detector.py  # LayoutReader integration
│   └── extraction/
│       └── extraction_agent.py    # Tool-based extraction agent
├── tools/
│   ├── __init__.py
│   ├── analyze_chart.py           # Chart analysis VLM tool
│   ├── analyze_table.py           # Table analysis VLM tool
│   └── utils.py                   # Shared tool utilities (image cropping)
├── agents/
│   └── verifier.py                # Quality verification (simplified)
└── ...
```

## Dependencies
- All implementation tasks (0037-0042) completed
- New architecture validated and tested

## Implementation Notes
- Use diff format to show before/after changes clearly
- Archive old architecture documentation for reference
- Update inline code comments
- Ensure consistency across all documentation files
- Add examples demonstrating new architecture

## Deliverables
- [ ] Updated AGENT.md
- [ ] Updated README.md
- [ ] Architecture diagram image file
- [ ] Migration guide document
- [ ] Updated API documentation
- [ ] Updated code docstrings
- [ ] Changelog entry
