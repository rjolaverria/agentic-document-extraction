### Fix: Low Confidence Scores Prevent Convergence (2026-01-23)

**Issue**: Resume and CSV extractions completed with all data correctly extracted, but the agentic loop did not converge because confidence scores remained below the threshold (0.90). Extra iterations were wasted without improving results.

- Resume: Confidence 0.70, 4 issues flagged (though data was correct)
- CSV: Confidence 0.50, 2 issues flagged (though data was correct)

**Root Cause**: Three related issues in the confidence scoring system:

1. **Initial extraction assigns `confidence=None`** to all fields (text_extraction.py:670)
   - The LLM doesn't provide explicit confidence scores during extraction

2. **Verifier defaults to 0.5 when no confidence scores exist** (verifier.py:869-873)
   - This arbitrary default was below the 0.7 threshold, triggering unnecessary refinement

3. **Refinement uses hardcoded confidence values** (refiner.py:662-676)
   - Values 0.60-0.70 based on whether fields changed, not actual extraction quality
   - Fields that didn't change kept their low/null confidence

**Fix Applied**:

Added `_derive_confidence_from_completeness()` method to `QualityVerificationAgent`:

```python
def _derive_confidence_from_completeness(
    self,
    required_field_coverage: float,
    extracted_fields: int,
    total_fields: int,
) -> float:
    """Derive confidence from completeness when no explicit scores exist."""
    # Weight required fields heavily - they're the primary quality indicator
    required_weight = 0.60
    coverage_weight = 0.30
    base_weight = 0.10

    field_coverage = extracted_fields / total_fields if total_fields > 0 else 0.0

    derived_confidence = (
        required_field_coverage * required_weight
        + field_coverage * coverage_weight
        + base_weight
    )

    # Boost for 100% required field coverage
    if required_field_coverage >= 1.0:
        derived_confidence = min(1.0, derived_confidence + 0.05)

    return max(0.0, min(1.0, derived_confidence))
```

**How It Works**:
- When no explicit confidence scores exist, confidence is now derived from completeness metrics
- Formula: `(required_coverage * 0.6) + (overall_coverage * 0.3) + 0.1 + (0.05 bonus if 100% required)`
- With 100% field coverage: confidence = 0.6 + 0.3 + 0.1 + 0.05 = 1.0 (capped)
- With 50% coverage: confidence = 0.3 + 0.15 + 0.1 = 0.55

**Files Changed**:
- `src/agentic_document_extraction/agents/verifier.py` - Added `_derive_confidence_from_completeness()` method and updated `_compute_metrics()` to use it

**Tests Added/Updated**:
- `tests/test_agents/test_verifier.py`:
  - Updated `test_compute_metrics_no_confidence_scores` to verify new derived confidence behavior
  - Added `test_compute_metrics_derived_confidence_partial_coverage` for partial coverage scenarios

**Validation**:
- All 831 tests passing
- Ruff: All checks passed
- Mypy: No issues found
- Coverage: 93%

**E2E Verification**:

Before fix:
- Invoice: iterations=2+, converged=sometimes, confidence=0.70
- Resume: iterations=3+, converged=no, confidence=0.70
- CSV: iterations=3+, converged=no, confidence=0.50

After fix:
- Invoice: iterations=1, converged=true, confidence=1.0 ✓
- Resume: iterations=1, converged=true, confidence=1.0 ✓
- CSV: iterations=1, converged=true, confidence=1.0 ✓

**Impact**: Extractions with complete data now converge on the first iteration instead of wasting iterations trying to improve already-correct results.
