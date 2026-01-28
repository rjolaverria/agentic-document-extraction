# Refactor: Tool-Based Agent Lightweight Verification Loop

**Date:** 2026-01-28
**Task:** 0042 - Simplify Agentic Loop for Tool-Based Architecture
**Status:** Complete

## Summary

Added a lightweight verification and refinement loop to the `ExtractionAgent` so it can verify quality and iterate on extractions when needed, without requiring the full multi-agent orchestration loop (planner + verifier + refiner). This implements "Option B: Lightweight Refinement" from the task specification.

## Problem

Previously, the tool-based `ExtractionAgent` had two limitations:

1. **No quality verification**: It created a fake `VerificationReport` with hardcoded 0.85 confidence and always PASSED status
2. **No refinement capability**: If extraction quality was poor, there was no way to improve it without switching to the legacy multi-agent loop (4-10 LLM calls)

## Solution

Integrated quality verification directly into `ExtractionAgent.extract()` with a lightweight feedback loop:

1. **Extract** using the tool-based agent (1 LLM call)
2. **Verify** quality using `QualityVerificationAgent` (rule-based by default, no extra LLM call)
3. **If issues found**, provide feedback and **re-extract** (up to `max_iterations`)
4. **Track best result** across iterations and return it

## Changes Made

### 1. ExtractionAgent Updates (`extraction_agent.py`)

**New Attributes:**
- `max_iterations: int`: Maximum extraction iterations (defaults to settings)
- `use_llm_verification: bool`: Whether to use LLM for deep verification (defaults to False for speed)

**New Methods:**
- `_build_refinement_prompt()`: Builds a system prompt that includes previous extraction issues
- `_format_issues_for_refinement()`: Formats verification issues as feedback for the agent
- `_calculate_result_score()`: Scores extraction results for best-result tracking

**Modified `extract()` Method:**
- Added iteration loop with quality verification after each extraction
- Tracks best result across iterations using score calculation
- Uses refinement prompt with issues when re-extracting
- Breaks early when quality thresholds are met
- Returns actual `VerificationReport` from the verifier (not fake)

**New Refinement Prompt Template:**
```python
_REFINEMENT_PROMPT_TEMPLATE = """
Your previous extraction had the following quality issues:

## Issues to Fix
{issues}

## Previous Extraction
{previous_extraction}

## Document Text
{ocr_text}
...
"""
```

### 2. Test Updates (`test_extraction_agent.py`)

**New Test Classes:**
- `TestRefinementPrompt`: Tests for the refinement prompt generation
- `TestVerificationLoop`: Tests for the lightweight verification loop
- `TestFormatIssuesForRefinement`: Tests for issue formatting
- `TestCalculateResultScore`: Tests for result scoring

**Key Tests Added:**
- `test_single_pass_when_quality_passes`: Verifies no extra iterations when quality is good
- `test_multiple_iterations_until_quality_passes`: Verifies refinement works
- `test_max_iterations_reached`: Verifies loop stops at max iterations
- `test_best_result_tracked_across_iterations`: Verifies best result is returned even if later iterations regress
- `test_iteration_history_tracked`: Verifies iteration metrics are captured
- `test_invoke_failure_returns_best_if_available`: Verifies graceful handling of failures

**Updated Existing Tests:**
- All tests now mock `QualityVerificationAgent` since it's used in the loop
- Updated assertions for new `loop_metadata["agent_type"]` value: `"tool_agent_with_verification"`

## Architecture

### Before (Single Pass)
```
ExtractionAgent.extract()
  └─ LLM Extraction (1 call)
     └─ Fake VerificationReport (hardcoded)
```

### After (Lightweight Loop)
```
ExtractionAgent.extract()
  └─ For iteration 1..max_iterations:
       ├─ Build prompt (initial or refinement)
       ├─ LLM Extraction (1 call)
       ├─ QualityVerificationAgent.verify() (rule-based, 0 LLM calls)
       ├─ Track best result
       ├─ If PASSED → break
       └─ Format issues for next iteration
```

### LLM Call Comparison

| Mode | Planning | Extraction | Verification | Refinement | Total |
|------|----------|------------|--------------|------------|-------|
| Legacy Multi-Agent | 1 | 1-3 | 1-3 | 1-3 | 4-10 |
| Tool Agent (before) | 0 | 1 | 0 | 0 | 1 |
| Tool Agent (after) | 0 | 1-3 | 0* | 0 | 1-3 |

*Rule-based verification by default. LLM analysis can be enabled via `use_llm_verification=True`.

## Benefits

1. **Quality Assurance**: Extractions are now verified with real quality metrics
2. **Iterative Improvement**: Poor extractions can be refined without switching to legacy loop
3. **Efficient**: Uses rule-based verification by default (no extra LLM calls)
4. **Best Result Tracking**: Returns the best result even if later iterations regress
5. **Backward Compatible**: Result structure remains `AgenticLoopResult` for pipeline compatibility

## Technical Notes

- Issue severity order for feedback: CRITICAL → HIGH → MEDIUM → LOW
- Limits feedback to top 10 issues to avoid overwhelming the agent
- Score calculation: `confidence*0.3 + required_coverage*0.4 + completeness*0.2 + consistency*0.1`
- Critical issues penalize score by -0.2 each
- Passing verification adds +0.1 bonus to score

## Verification

- All 900 tests pass
- mypy type checking passes
- ruff linting passes
- 86% code coverage maintained

## Files Modified

- `src/agentic_document_extraction/agents/extraction_agent.py`
- `tests/test_agents/test_extraction_agent.py`
