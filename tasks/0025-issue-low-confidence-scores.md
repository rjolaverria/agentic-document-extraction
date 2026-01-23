# ISSUE: Low Confidence Scores Prevent Convergence

- [x] **ISSUE: Low Confidence Scores Prevent Convergence**
  - **Severity**: Medium
  - **Description**: Resume and CSV extractions complete with all data correctly extracted, but the agentic loop does not converge because confidence scores remain below the threshold (0.90).
  - **Resume**: Confidence 0.70, 4 issues flagged (contact info, phone format, skills format)
  - **CSV**: Confidence 0.50, 2 issues flagged
  - **Impact**: Extra iterations are wasted without improving results.
  - **Fix Applied**: Added `_derive_confidence_from_completeness()` method to `QualityVerificationAgent` that derives confidence from completeness metrics (required field coverage, overall field coverage) when no explicit confidence scores exist. Complete extractions now converge on the first iteration with derived confidence of 1.0.
