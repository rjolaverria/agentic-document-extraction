# ISSUE: Skills Extracted as Category-Prefixed Strings

- [ ] **ISSUE: Skills Extracted as Category-Prefixed Strings**
  - **Severity**: Low
  - **Description**: Resume skills are extracted as strings like "Languages: Python, JavaScript, TypeScript, Go, SQL" instead of individual skill items.
  - **Schema Expectation**: Array of individual skill strings
  - **Current Extraction**: Array of category-prefixed strings
  - **Impact**: Downstream parsing of individual skills is complicated.
  - **Suggested Fix**: Improve extraction prompt to split skills into individual items.
