# ISSUE: Phone Number Format Not Standardized

- [ ] **ISSUE: Phone Number Format Not Standardized**
  - **Severity**: Low
  - **Description**: Phone numbers are extracted in various formats like `(555) 987-6543` without standardization to E.164 or another consistent format.
  - **Impact**: Consistency issues for downstream systems.
  - **Suggested Fix**: Add optional phone number normalization in post-processing.
