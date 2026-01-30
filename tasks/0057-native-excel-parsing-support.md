# Task 0057: Native Excel Parsing Support

## Objective
Add native Excel parsing capabilities to handle complex spreadsheet data that the visual pipeline cannot reliably extract.

## Context
Currently, Excel files (`.xlsx`) are detected and classified as `SPREADSHEET` format but processed through the visual pipeline:
1. Excel files are converted to images
2. PaddleOCR extracts text from the visual representation
3. The LLM agent extracts structured data based on the schema

This approach has limitations for complex Excel documents:
- Multi-sheet workbooks (only visible sheet is processed)
- Hidden rows/columns are not accessible
- Cell formulas and computed values may be missed
- Large spreadsheets with many rows lose data
- Merged cells and complex formatting confuse OCR
- Numeric precision can be lost in visual conversion

## Acceptance Criteria
- [ ] Add `openpyxl` dependency for native `.xlsx` parsing
- [ ] Create `ExcelExtractor` service class for spreadsheet processing
- [ ] Support multi-sheet extraction with sheet selection
- [ ] Preserve cell data types (numbers, dates, strings, formulas)
- [ ] Handle merged cells appropriately
- [ ] Support both native and visual extraction modes (configurable)
- [ ] Update `FormatDetector` to route Excel files to native extractor
- [ ] Add `ProcessingCategory.STRUCTURED` for native spreadsheet processing
- [ ] Create `AnalyzeSpreadsheet` tool for agent-driven Excel analysis
- [ ] Unit tests for Excel extraction scenarios
- [ ] Integration tests with sample Excel files

## Current Implementation

### Format Detection (format_detector.py)
```python
# Lines 47, 71: Extension mapping
".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Lines 101-103: Format family classification
if mime_type in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",):
    return FormatFamily.SPREADSHEET

# Lines 133-135: Processing category (VISUAL)
FormatFamily.SPREADSHEET: ProcessingCategory.VISUAL
```

### Text Extractor (text_extractor.py)
```python
# Line 657: Only supports .txt and .csv
SUPPORTED_EXTENSIONS = {".txt", ".csv"}
```

## Proposed Changes

### 1. Add Dependencies
```toml
# pyproject.toml
dependencies = [
    ...
    "openpyxl>=3.1.0",
]
```

### 2. New ExcelExtractor Service
```python
# src/agentic_document_extraction/services/excel_extractor.py

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

class ExcelExtractor:
    """Native Excel file parser for structured data extraction."""

    def extract(
        self,
        file_path: Path,
        *,
        sheet_name: str | None = None,
        include_formulas: bool = False,
        max_rows: int | None = None,
    ) -> ExcelDocument:
        """Extract data from Excel file."""
        ...

    def get_sheet_names(self, file_path: Path) -> list[str]:
        """List all sheet names in workbook."""
        ...

    def extract_as_dataframe(
        self,
        file_path: Path,
        sheet_name: str | None = None,
    ) -> pd.DataFrame:
        """Extract sheet as pandas DataFrame."""
        ...
```

### 3. ExcelDocument Model
```python
# src/agentic_document_extraction/models/excel_document.py

@dataclass
class ExcelCell:
    value: Any
    data_type: str  # "string", "number", "date", "formula", "boolean"
    formula: str | None = None
    is_merged: bool = False

@dataclass
class ExcelSheet:
    name: str
    rows: list[list[ExcelCell]]
    headers: list[str] | None = None
    row_count: int
    column_count: int

@dataclass
class ExcelDocument:
    sheets: list[ExcelSheet]
    active_sheet: str
    metadata: dict[str, Any]
```

### 4. Update Format Detector
```python
# Add new processing category
class ProcessingCategory(str, Enum):
    TEXT = "text"
    VISUAL = "visual"
    STRUCTURED = "structured"  # New: for native spreadsheet/data files

# Update category mapping
_FORMAT_TO_CATEGORY = {
    FormatFamily.SPREADSHEET: ProcessingCategory.STRUCTURED,
    ...
}
```

### 5. New AnalyzeSpreadsheet Tool
```python
# src/agentic_document_extraction/tools/analyze_spreadsheet.py

@tool
def analyze_spreadsheet(
    sheet_name: Annotated[str | None, "Sheet name to analyze (None for active sheet)"] = None,
    row_range: Annotated[str | None, "Row range like '1:100' or '5:50'"] = None,
    columns: Annotated[list[str] | None, "Column letters to include, e.g. ['A', 'B', 'C']"] = None,
    state: Annotated[ExtractorState, InjectedState],
) -> str:
    """
    Analyze spreadsheet data from Excel files.

    Use this tool when extracting data from Excel spreadsheets.
    Provides direct access to cell values, formulas, and structure.

    Returns formatted table data with headers and values.
    """
    ...
```

### 6. Update Processing Pipeline
```python
# In extraction orchestrator or pipeline
async def process_document(self, file_path: Path, ...) -> ExtractionResult:
    format_info = self.format_detector.detect(file_path)

    if format_info.processing_category == ProcessingCategory.STRUCTURED:
        return await self._process_structured(file_path, ...)
    elif format_info.processing_category == ProcessingCategory.VISUAL:
        return await self._process_visual(file_path, ...)
    else:
        return await self._process_text(file_path, ...)
```

## Files to Create
- `src/agentic_document_extraction/services/excel_extractor.py`
- `src/agentic_document_extraction/models/excel_document.py`
- `src/agentic_document_extraction/tools/analyze_spreadsheet.py`
- `tests/test_services/test_excel_extractor.py`
- `tests/fixtures/sample_spreadsheets/` (test files)

## Files to Modify
- `pyproject.toml` - Add openpyxl dependency
- `src/agentic_document_extraction/services/format_detector.py` - Add STRUCTURED category
- `src/agentic_document_extraction/agents/extraction_agent.py` - Register new tool
- `src/agentic_document_extraction/services/extraction_orchestrator.py` - Route to Excel extractor

## Dependencies
- None (can be implemented independently)

## Testing Strategy
1. **Unit Tests**
   - Parse single-sheet Excel file
   - Parse multi-sheet workbook
   - Handle merged cells
   - Preserve data types (numbers, dates, strings)
   - Extract formulas vs computed values
   - Handle empty cells and sparse data

2. **Integration Tests**
   - End-to-end extraction from Excel invoice
   - Extract tabular data matching schema
   - Handle large spreadsheets (1000+ rows)

3. **Test Files Needed**
   - Simple single-sheet spreadsheet
   - Multi-sheet workbook
   - Spreadsheet with formulas
   - Spreadsheet with merged cells
   - Large data spreadsheet
   - Mixed data types spreadsheet

## Notes
- Consider adding support for `.xls` (legacy format) via `xlrd` if needed
- Visual fallback should remain available for scanned/image-based spreadsheets
- The agent should be able to choose between native parsing and visual analysis
- Performance consideration: native parsing is significantly faster than visual pipeline
