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
