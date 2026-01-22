---

### Feature: Date Format Normalization (2026-01-22)

**Issue**: The `due_date` field was extracted as "February 15, 2024" instead of the expected ISO format "YYYY-MM-DD" per the schema's `format: date` specification.

**Root Cause**: Two issues were identified:
1. The `FieldInfo` class in `schema_validator.py` didn't capture the `format` specification from JSON schemas
2. The `JsonGenerator._clean_value()` method didn't normalize date values to ISO format
3. The `extraction_processor.py` directly used raw extracted data without passing through JSON normalization

**Fix Applied**:

1. **Added `format_spec` to `FieldInfo` class** (`services/schema_validator.py`):
   - Added `format_spec: str | None = None` parameter to capture JSON Schema format specifications
   - Updated `_extract_fields()` to extract format specs from schema properties
   - Updated `to_dict()` to include format in serialization

2. **Added date normalization utilities** (`output/json_generator.py`):
   - Added `DATE_FORMATS` list with common date formats (ISO, US, EU, month names)
   - Added `parse_date_string()` function to parse various date formats
   - Added `normalize_date()` function to convert dates to ISO format (YYYY-MM-DD)
   - Updated `_clean_value()` to normalize dates when `format_spec == "date"`
   - Updated `_prepare_data()` to pass `format_spec` to `_clean_value()`

3. **Integrated normalization into extraction pipeline** (`services/extraction_processor.py`):
   - Added `JsonGenerator` import
   - Added JSON normalization step before saving extraction results
   - This ensures all extracted data goes through format normalization

**Supported Date Formats**:
- ISO: `2024-02-15`, `2024/02/15`
- US: `02/15/2024`, `02-15-2024`
- EU: `15/02/2024`, `15-02-2024`, `15.02.2024`
- Month names: `February 15, 2024`, `Feb 15, 2024`, `15 February 2024`
- Short year: `02/15/24`, `15/02/24`

**Files Changed**:
- `src/agentic_document_extraction/services/schema_validator.py` - Added `format_spec` attribute
- `src/agentic_document_extraction/output/json_generator.py` - Added date normalization utilities
- `src/agentic_document_extraction/services/extraction_processor.py` - Added JSON normalization step

**Tests Added**:
- `tests/test_services/test_schema_validator.py` - 6 new tests for `format_spec` extraction
- `tests/test_output/test_json_generator.py` - 25+ new tests for date parsing and normalization

**Validation**:
- All 830 tests passing (43 new tests added)
- Ruff: All checks passed
- Mypy: No issues found

**E2E Verification**:
Before fix:
- `invoice_date`: "January 15, 2024" (or sometimes normalized)
- `due_date`: "February 15, 2024"

After fix:
- `invoice_date`: "2024-01-15" ✓
- `due_date`: "2024-02-15" ✓

---

### Bug Fix: PNG/Image File Support (2026-01-22)

**Issue**: PNG and other image files were not being routed to the visual extraction pipeline. The extraction processor was attempting to use the text extractor for all documents, which only supports `.txt` and `.csv` files.

**Error Before Fix**: `TextExtractionError: Unsupported file extension for text extraction: .png`

**Root Cause**: In `services/extraction_processor.py`, the code was using `TextExtractor` for both text-based AND visual documents. The else branch for visual documents was incorrectly calling `text_extractor.extract_from_path()` instead of using the `VisualTextExtractor`.

**Fix Applied**:
1. Imported `VisualTextExtractor` in `extraction_processor.py`
2. Updated the else branch (visual documents) to use `VisualTextExtractor` instead of `TextExtractor`
3. The visual extractor uses OCR (pytesseract) for images and pdfplumber for PDFs

**Files Changed**:
- `src/agentic_document_extraction/services/extraction_processor.py` - Added import and routing logic

**Tests Added**:
- `tests/test_services/test_extraction_processor.py` - 12 new tests covering:
  - PNG files route to visual extractor
  - TXT files route to text extractor
  - All visual formats (png, jpg, jpeg, pdf, bmp, gif, webp, tiff) use visual extractor
  - All text formats (txt, csv) use text extractor

**Validation**:
- All 799 tests passing (12 new tests added)
- Ruff: All checks passed
- Mypy: No issues found
- Coverage: 93%

**E2E Verification**:
After the fix, PNG files are correctly routed to the visual extraction pipeline. The job status now shows `VisualExtractionError` (missing tesseract dependency) instead of `TextExtractionError`, confirming the routing fix works correctly.

---

### E2E Testing Results - `/extract` Endpoint (2026-01-22)

The following tests were run against the `/extract` endpoint using fixtures from `tests/fixtures/`:

#### Test Scenarios Executed

| Scenario | Document | Schema | Status | Converged | Confidence | Issues |
|----------|----------|--------|--------|-----------|------------|--------|
| Invoice Extraction | `sample_invoice.txt` | `invoice_schema.json` | ✅ Completed | ✅ Yes | 0.70 | 2 (date format) |
| Resume Extraction | `sample_resume.txt` | `resume_schema.json` | ✅ Completed | ❌ No | 0.70 | 4 (format issues) |
| CSV Employee Data | `sample_data.csv` | `employee_data_schema.json` | ✅ Completed | ❌ No | 0.50 | 2 (spurious nulls) |
| Simple Schema Test | `sample_invoice.txt` | `simple_schema.json` | ✅ Completed | ✅ Yes | 0.70 | 0 |
| PNG Image Extraction | `sample_coupon_code_form.png` | `sample_coupon_code_form_schema.json` | ❌ **FAILED** | N/A | N/A | Critical Error |

#### Tasks to Address

1. **[Critical]** Fix image file routing to visual extraction pipeline
   - Investigate `services/extraction_processor.py`
   - Ensure `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.webp`, `.tiff` files use visual pipeline
   - Add integration tests for image extraction

2. **[Medium] ✅ COMPLETED** Add date format normalization
   - Parse dates extracted in natural language formats
   - Convert to ISO 8601 format when schema specifies `format: date`

3. **[Medium]** Review and adjust confidence thresholds
   - Current threshold (0.90) may be too strict
   - Consider making thresholds configurable per-field

4. **[Low]** Fix verifier's array field null-checking logic
   - The verifier incorrectly reports null values for populated array fields

5. **[Low]** Improve skills extraction prompt
   - Guide the LLM to extract individual skills rather than category strings

6. **[Low]** Add optional phone number normalization
   - Consider E.164 format standardization as a post-processing step
