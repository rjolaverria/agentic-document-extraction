"""Quality verification agent using LangChain.

This module provides a LangChain agent that verifies extraction quality
against defined thresholds. The agent evaluates extraction results using
various quality metrics and provides actionable feedback for improvement.

Key features:
- Evaluates extraction results against quality thresholds
- Computes quality metrics (schema coverage, confidence, completeness, consistency)
- Identifies specific issues (missing fields, low confidence, schema violations)
- Returns verification report with pass/fail status
- Provides actionable feedback for improvement
- Different verification strategies for text vs. visual documents
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jsonschema
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agentic_document_extraction.agents.planner import QualityThreshold
from agentic_document_extraction.config import settings
from agentic_document_extraction.models import ProcessingCategory
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.schema_validator import FieldInfo, SchemaInfo

logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised when verification fails due to an error."""

    def __init__(
        self,
        message: str,
        error_type: str = "verification_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with message and error details.

        Args:
            message: Error message.
            error_type: Type of error for categorization.
            details: Optional additional error details.
        """
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class VerificationStatus(str, Enum):
    """Status of the verification result."""

    PASSED = "passed"
    FAILED = "failed"
    NEEDS_IMPROVEMENT = "needs_improvement"


class IssueType(str, Enum):
    """Types of issues that can be identified during verification."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    LOW_CONFIDENCE = "low_confidence"
    SCHEMA_VIOLATION = "schema_violation"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    INCOMPLETE_VALUE = "incomplete_value"
    TYPE_MISMATCH = "type_mismatch"
    FORMAT_ERROR = "format_error"
    EMPTY_ARRAY = "empty_array"
    NULL_VALUE = "null_value"
    POTENTIAL_HALLUCINATION = "potential_hallucination"


class IssueSeverity(str, Enum):
    """Severity levels for identified issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class VerificationIssue:
    """A specific issue identified during verification."""

    issue_type: IssueType
    """Type of the issue."""

    field_path: str
    """Path to the field with the issue."""

    message: str
    """Human-readable description of the issue."""

    severity: IssueSeverity
    """Severity of the issue."""

    current_value: Any = None
    """Current value of the field (if applicable)."""

    expected: str | None = None
    """Description of expected value or behavior."""

    suggestion: str | None = None
    """Suggested action to resolve the issue."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with issue information.
        """
        return {
            "issue_type": self.issue_type.value,
            "field_path": self.field_path,
            "message": self.message,
            "severity": self.severity.value,
            "current_value": self.current_value,
            "expected": self.expected,
            "suggestion": self.suggestion,
        }


@dataclass
class QualityMetrics:
    """Quality metrics computed during verification."""

    overall_confidence: float
    """Average confidence across all extracted fields (0.0-1.0)."""

    schema_coverage: float
    """Percentage of schema fields that have non-null values (0.0-1.0)."""

    required_field_coverage: float
    """Percentage of required fields that have non-null values (0.0-1.0)."""

    optional_field_coverage: float
    """Percentage of optional fields that have non-null values (0.0-1.0)."""

    completeness_score: float
    """Overall completeness score (0.0-1.0)."""

    consistency_score: float
    """Score for logical consistency (0.0-1.0)."""

    min_field_confidence: float
    """Minimum confidence among all fields."""

    fields_with_low_confidence: int
    """Number of fields with confidence below threshold."""

    total_fields: int
    """Total number of fields in schema."""

    extracted_fields: int
    """Number of fields with non-null values."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with quality metrics.
        """
        return {
            "overall_confidence": self.overall_confidence,
            "schema_coverage": self.schema_coverage,
            "required_field_coverage": self.required_field_coverage,
            "optional_field_coverage": self.optional_field_coverage,
            "completeness_score": self.completeness_score,
            "consistency_score": self.consistency_score,
            "min_field_confidence": self.min_field_confidence,
            "fields_with_low_confidence": self.fields_with_low_confidence,
            "total_fields": self.total_fields,
            "extracted_fields": self.extracted_fields,
        }


@dataclass
class VerificationReport:
    """Complete verification report for an extraction result."""

    status: VerificationStatus
    """Overall verification status."""

    metrics: QualityMetrics
    """Computed quality metrics."""

    issues: list[VerificationIssue] = field(default_factory=list)
    """List of identified issues."""

    passed_checks: list[str] = field(default_factory=list)
    """List of checks that passed."""

    recommendations: list[str] = field(default_factory=list)
    """List of recommendations for improvement."""

    llm_analysis: str | None = None
    """LLM reasoning and analysis."""

    model_used: str = ""
    """Model used for verification analysis."""

    total_tokens: int = 0
    """Total tokens used in verification."""

    prompt_tokens: int = 0
    """Prompt tokens used."""

    completion_tokens: int = 0
    """Completion tokens used."""

    processing_time_seconds: float = 0.0
    """Time taken for verification."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with verification report information.
        """
        return {
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "passed_checks": self.passed_checks,
            "recommendations": self.recommendations,
            "llm_analysis": self.llm_analysis,
            "metadata": {
                "model_used": self.model_used,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "processing_time_seconds": self.processing_time_seconds,
            },
        }

    @property
    def passed(self) -> bool:
        """Check if verification passed.

        Returns:
            True if status is PASSED.
        """
        return self.status == VerificationStatus.PASSED

    @property
    def critical_issues(self) -> list[VerificationIssue]:
        """Get critical issues.

        Returns:
            List of critical severity issues.
        """
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def high_priority_issues(self) -> list[VerificationIssue]:
        """Get high priority issues (critical + high severity).

        Returns:
            List of critical and high severity issues.
        """
        return [
            i
            for i in self.issues
            if i.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)
        ]


class QualityVerificationAgent:
    """Agent for verifying extraction quality using LangChain.

    Uses LLM reasoning to evaluate extraction results against quality
    thresholds and identify issues. Combines rule-based checks with
    LLM-powered analysis for comprehensive verification.
    """

    VERIFICATION_SYSTEM_PROMPT = """You are an expert quality verification assistant.
Your task is to analyze extracted data and verify its quality against the original schema.

IMPORTANT RESPONSIBILITIES:
1. Check if all required fields have been extracted with reasonable values
2. Identify any logical inconsistencies in the extracted data
3. Detect potential hallucinations or fabricated information
4. Verify that extracted values make sense in context
5. Check for completeness and accuracy of complex fields (arrays, nested objects)
6. Identify any formatting issues or type mismatches
7. Assess overall confidence in the extraction quality

ANALYSIS GUIDELINES:
- Be thorough but fair - minor issues shouldn't fail otherwise good extractions
- Focus on semantic correctness, not just syntactic validity
- Consider the context and expected patterns for each field type
- Flag suspicious values that might be hallucinated
- Provide specific, actionable feedback for improvements
- When checking numerical consistency (e.g., sums, totals), calculate the actual values before claiming inconsistency
- Do NOT report an inconsistency if the values actually match - verify by computing the sum yourself

You must respond with ONLY valid JSON matching this structure:
{{
    "consistency_score": 0.0-1.0,
    "issues": [
        {{
            "issue_type": "logical_inconsistency" | "potential_hallucination" | "incomplete_value" | "format_error",
            "field_path": "path.to.field",
            "message": "Description of the issue",
            "severity": "critical" | "high" | "medium" | "low",
            "suggestion": "How to fix this issue"
        }}
    ],
    "passed_checks": ["List of checks that passed"],
    "recommendations": ["List of improvement recommendations"],
    "analysis": "Detailed reasoning about the extraction quality"
}}"""

    VERIFICATION_USER_PROMPT = """Verify the quality of the following extraction result.

## Target JSON Schema:
```json
{schema}
```

## Required Fields:
{required_fields}

## Optional Fields:
{optional_fields}

## Extracted Data:
```json
{extracted_data}
```

## Field Extractions with Confidence:
{field_extractions}

## Additional Context:
- Processing type: {processing_type}
- Total fields in schema: {total_fields}
- Fields extracted: {extracted_fields}
- Current overall confidence: {overall_confidence:.2f}

Analyze the extraction quality and identify any issues. Focus on:
1. Logical consistency between related fields
2. Completeness of arrays and nested objects
3. Plausibility of extracted values
4. Any potential hallucinations or fabricated data
5. Format and type correctness

IMPORTANT: For numerical fields like totals and subtotals:
- Calculate the sum yourself before claiming a mismatch
- Only report an inconsistency if you have verified the numbers actually don't add up
- If line item totals sum to exactly the subtotal, that is CORRECT, not an inconsistency

Respond with ONLY the JSON structure specified."""

    @staticmethod
    def get_default_thresholds() -> QualityThreshold:
        """Get default quality thresholds from settings.

        Returns:
            QualityThreshold with values from configuration.
        """
        return QualityThreshold(
            min_overall_confidence=settings.min_overall_confidence,
            min_field_confidence=settings.min_field_confidence,
            required_field_coverage=settings.required_field_coverage,
            max_iterations=settings.max_refinement_iterations,
        )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the verification agent.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Model name to use. Defaults to settings.openai_model.
            temperature: Sampling temperature. Defaults to settings.openai_temperature.
            max_tokens: Maximum tokens for response. Defaults to settings.openai_max_tokens.
        """
        self.api_key = api_key if api_key is not None else settings.get_openai_api_key()
        self.model = model or settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self.max_tokens = max_tokens or settings.openai_max_tokens

        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LangChain ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            VerificationError: If API key is not configured.
        """
        if self._llm is None:
            if not self.api_key:
                raise VerificationError(
                    "OpenAI API key not configured",
                    error_type="configuration_error",
                    details={"missing": "openai_api_key"},
                )

            self._llm = ChatOpenAI(
                api_key=self.api_key,  # type: ignore[arg-type]
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

        return self._llm

    def verify(
        self,
        extraction_result: ExtractionResult,
        schema_info: SchemaInfo,
        thresholds: QualityThreshold | None = None,
        processing_category: ProcessingCategory = ProcessingCategory.TEXT_BASED,
        use_llm_analysis: bool = True,
    ) -> VerificationReport:
        """Verify extraction quality against thresholds.

        Args:
            extraction_result: The extraction result to verify.
            schema_info: Schema information for validation.
            thresholds: Quality thresholds. Defaults to settings-based thresholds.
            processing_category: Whether text or visual processing was used.
            use_llm_analysis: Whether to use LLM for additional analysis.

        Returns:
            VerificationReport with verification results.

        Raises:
            VerificationError: If verification encounters an error.
        """
        start_time = time.time()
        thresholds = thresholds or self.get_default_thresholds()

        # Perform rule-based checks
        issues = self._perform_rule_based_checks(
            extraction_result, schema_info, thresholds
        )

        # Compute metrics
        metrics = self._compute_metrics(
            extraction_result, schema_info, thresholds, issues
        )

        # Perform schema validation
        schema_issues = self._validate_against_schema(
            extraction_result.extracted_data, schema_info
        )
        issues.extend(schema_issues)

        # Initialize report variables
        llm_analysis: str | None = None
        model_used = ""
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        passed_checks: list[str] = []
        recommendations: list[str] = []

        # Use LLM for deeper analysis if enabled
        if use_llm_analysis:
            try:
                llm_result = self._perform_llm_analysis(
                    extraction_result, schema_info, metrics, processing_category
                )
                llm_issues, llm_passed_checks, llm_recommendations, llm_analysis = (
                    llm_result["issues"],
                    llm_result["passed_checks"],
                    llm_result["recommendations"],
                    llm_result["analysis"],
                )

                # Add LLM-identified issues (avoiding duplicates)
                existing_issue_keys = {(i.issue_type, i.field_path) for i in issues}
                for issue in llm_issues:
                    if (issue.issue_type, issue.field_path) not in existing_issue_keys:
                        issues.append(issue)

                passed_checks = llm_passed_checks
                recommendations = llm_recommendations
                model_used = self.model
                total_tokens = llm_result["total_tokens"]
                prompt_tokens = llm_result["prompt_tokens"]
                completion_tokens = llm_result["completion_tokens"]

                # Update consistency score from LLM
                metrics = QualityMetrics(
                    overall_confidence=metrics.overall_confidence,
                    schema_coverage=metrics.schema_coverage,
                    required_field_coverage=metrics.required_field_coverage,
                    optional_field_coverage=metrics.optional_field_coverage,
                    completeness_score=metrics.completeness_score,
                    consistency_score=llm_result["consistency_score"],
                    min_field_confidence=metrics.min_field_confidence,
                    fields_with_low_confidence=metrics.fields_with_low_confidence,
                    total_fields=metrics.total_fields,
                    extracted_fields=metrics.extracted_fields,
                )

            except Exception as e:
                logger.warning(f"LLM analysis failed, using rule-based only: {e}")
                llm_analysis = f"LLM analysis unavailable: {e}"

        # Add rule-based passed checks
        rule_passed_checks = self._get_rule_based_passed_checks(
            metrics, thresholds, issues
        )
        passed_checks = list(set(passed_checks + rule_passed_checks))

        # Determine overall status
        status = self._determine_status(metrics, thresholds, issues)

        processing_time = time.time() - start_time

        logger.info(
            f"Verification completed: "
            f"status={status.value}, "
            f"issues={len(issues)}, "
            f"confidence={metrics.overall_confidence:.2f}, "
            f"coverage={metrics.required_field_coverage:.2f}, "
            f"tokens={total_tokens}, "
            f"time={processing_time:.2f}s"
        )

        return VerificationReport(
            status=status,
            metrics=metrics,
            issues=issues,
            passed_checks=passed_checks,
            recommendations=recommendations,
            llm_analysis=llm_analysis,
            model_used=model_used,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            processing_time_seconds=processing_time,
        )

    def _perform_rule_based_checks(
        self,
        extraction_result: ExtractionResult,
        schema_info: SchemaInfo,
        thresholds: QualityThreshold,
    ) -> list[VerificationIssue]:
        """Perform rule-based verification checks.

        Args:
            extraction_result: Extraction result to check.
            schema_info: Schema information.
            thresholds: Quality thresholds.

        Returns:
            List of identified issues.
        """
        issues: list[VerificationIssue] = []

        # Check required fields
        issues.extend(
            self._check_required_fields(extraction_result, schema_info.required_fields)
        )

        # Check field confidence
        issues.extend(
            self._check_field_confidence(
                extraction_result.field_extractions, thresholds.min_field_confidence
            )
        )

        # Check for empty arrays in required fields
        issues.extend(
            self._check_empty_arrays(extraction_result, schema_info.required_fields)
        )

        # Check for null values in required fields
        issues.extend(
            self._check_null_required_fields(
                extraction_result, schema_info.required_fields
            )
        )

        return issues

    def _check_required_fields(
        self,
        extraction_result: ExtractionResult,
        required_fields: list[FieldInfo],
    ) -> list[VerificationIssue]:
        """Check that all required fields have values.

        Args:
            extraction_result: Extraction result to check.
            required_fields: List of required field info.

        Returns:
            List of issues for missing required fields.
        """
        issues: list[VerificationIssue] = []
        data = extraction_result.extracted_data

        for field_info in required_fields:
            value = self._get_nested_value(data, field_info.path)

            if value is None:
                issues.append(
                    VerificationIssue(
                        issue_type=IssueType.MISSING_REQUIRED_FIELD,
                        field_path=field_info.path,
                        message=f"Required field '{field_info.path}' is missing",
                        severity=IssueSeverity.CRITICAL,
                        expected=f"A value of type {field_info.field_type}",
                        suggestion=f"Re-extract with focus on finding {field_info.path}",
                    )
                )

        return issues

    def _check_field_confidence(
        self,
        field_extractions: list[FieldExtraction],
        min_confidence: float,
    ) -> list[VerificationIssue]:
        """Check field confidence scores against threshold.

        Args:
            field_extractions: Field extractions with confidence scores.
            min_confidence: Minimum acceptable confidence.

        Returns:
            List of issues for low confidence fields.
        """
        issues: list[VerificationIssue] = []

        for extraction in field_extractions:
            if (
                extraction.confidence is not None
                and extraction.confidence < min_confidence
            ):
                severity = (
                    IssueSeverity.HIGH
                    if extraction.confidence < min_confidence * 0.5
                    else IssueSeverity.MEDIUM
                )

                issues.append(
                    VerificationIssue(
                        issue_type=IssueType.LOW_CONFIDENCE,
                        field_path=extraction.field_path,
                        message=(
                            f"Field '{extraction.field_path}' has low confidence "
                            f"({extraction.confidence:.2f} < {min_confidence:.2f})"
                        ),
                        severity=severity,
                        current_value=extraction.value,
                        expected=f"Confidence >= {min_confidence:.2f}",
                        suggestion="Re-extract this field or verify manually",
                    )
                )

        return issues

    def _check_empty_arrays(
        self,
        extraction_result: ExtractionResult,
        required_fields: list[FieldInfo],
    ) -> list[VerificationIssue]:
        """Check for empty arrays in required fields.

        Args:
            extraction_result: Extraction result to check.
            required_fields: List of required field info.

        Returns:
            List of issues for empty required arrays.
        """
        issues: list[VerificationIssue] = []
        data = extraction_result.extracted_data

        for field_info in required_fields:
            if field_info.field_type == "array":
                value = self._get_nested_value(data, field_info.path)
                if isinstance(value, list) and len(value) == 0:
                    issues.append(
                        VerificationIssue(
                            issue_type=IssueType.EMPTY_ARRAY,
                            field_path=field_info.path,
                            message=f"Required array field '{field_info.path}' is empty",
                            severity=IssueSeverity.HIGH,
                            current_value=value,
                            expected="Array with at least one element",
                            suggestion=f"Re-extract to populate {field_info.path} array",
                        )
                    )

        return issues

    def _check_null_required_fields(
        self,
        extraction_result: ExtractionResult,
        required_fields: list[FieldInfo],
    ) -> list[VerificationIssue]:
        """Check for explicit null values in required fields.

        Args:
            extraction_result: Extraction result to check.
            required_fields: List of required field info.

        Returns:
            List of issues for null required fields.
        """
        issues: list[VerificationIssue] = []
        data = extraction_result.extracted_data

        for field_info in required_fields:
            # Check if the key exists but value is None
            value = self._get_nested_value(data, field_info.path)
            if value is None:
                # Already handled in _check_required_fields
                continue

            # Check nested fields if this is an object
            if field_info.nested_fields:
                for nested_field in field_info.nested_fields:
                    nested_value = self._get_nested_value(data, nested_field.path)
                    if nested_value is None and nested_field.required:
                        issues.append(
                            VerificationIssue(
                                issue_type=IssueType.NULL_VALUE,
                                field_path=nested_field.path,
                                message=(
                                    f"Nested required field '{nested_field.path}' "
                                    "has null value"
                                ),
                                severity=IssueSeverity.HIGH,
                                expected=f"A value of type {nested_field.field_type}",
                                suggestion=(
                                    f"Re-extract with focus on {nested_field.path}"
                                ),
                            )
                        )

        return issues

    def _validate_against_schema(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
    ) -> list[VerificationIssue]:
        """Validate extracted data against JSON schema.

        Args:
            data: Extracted data to validate.
            schema_info: Schema information.

        Returns:
            List of schema violation issues.
        """
        issues: list[VerificationIssue] = []

        try:
            jsonschema.validate(instance=data, schema=schema_info.schema)
        except jsonschema.ValidationError as e:
            # Extract field path from the error
            field_path = ".".join(str(p) for p in e.absolute_path) or "root"

            issues.append(
                VerificationIssue(
                    issue_type=IssueType.SCHEMA_VIOLATION,
                    field_path=field_path,
                    message=f"Schema validation failed: {e.message}",
                    severity=IssueSeverity.CRITICAL,
                    current_value=e.instance,
                    expected=str(e.schema),
                    suggestion="Fix the data to match schema requirements",
                )
            )

        return issues

    def _compute_metrics(
        self,
        extraction_result: ExtractionResult,
        schema_info: SchemaInfo,
        thresholds: QualityThreshold,
        issues: list[VerificationIssue],
    ) -> QualityMetrics:
        """Compute quality metrics from extraction result.

        Args:
            extraction_result: Extraction result to analyze.
            schema_info: Schema information.
            thresholds: Quality thresholds for comparison.
            issues: Already identified issues.

        Returns:
            QualityMetrics with computed values.
        """
        data = extraction_result.extracted_data
        field_extractions = extraction_result.field_extractions

        # Count fields
        total_fields = len(schema_info.all_fields)
        required_count = len(schema_info.required_fields)
        optional_count = len(schema_info.optional_fields)

        # Count extracted fields (non-null values)
        extracted_fields = sum(
            1
            for f in schema_info.all_fields
            if self._get_nested_value(data, f.path) is not None
        )

        # Count required fields extracted
        required_extracted = sum(
            1
            for f in schema_info.required_fields
            if self._get_nested_value(data, f.path) is not None
        )

        # Count optional fields extracted
        optional_extracted = sum(
            1
            for f in schema_info.optional_fields
            if self._get_nested_value(data, f.path) is not None
        )

        # Compute coverage scores
        schema_coverage = extracted_fields / total_fields if total_fields > 0 else 0.0
        required_field_coverage = (
            required_extracted / required_count if required_count > 0 else 1.0
        )
        optional_field_coverage = (
            optional_extracted / optional_count if optional_count > 0 else 1.0
        )

        # Compute confidence metrics
        confidences = [
            fe.confidence for fe in field_extractions if fe.confidence is not None
        ]

        if confidences:
            overall_confidence = sum(confidences) / len(confidences)
            min_field_confidence = min(confidences)
            fields_with_low_confidence = sum(
                1 for c in confidences if c < thresholds.min_field_confidence
            )
        else:
            # No explicit confidence scores available - derive from completeness
            # This addresses the issue where initial extractions have no confidence
            # but the data may be complete and correct
            overall_confidence = self._derive_confidence_from_completeness(
                required_field_coverage=required_field_coverage,
                extracted_fields=extracted_fields,
                total_fields=total_fields,
            )
            min_field_confidence = overall_confidence
            fields_with_low_confidence = 0

        # Compute completeness score (weighted average of coverage)
        completeness_score = (
            required_field_coverage * 0.7 + optional_field_coverage * 0.3
        )

        # Compute consistency score based on issues
        critical_issues = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        high_issues = sum(1 for i in issues if i.severity == IssueSeverity.HIGH)
        medium_issues = sum(1 for i in issues if i.severity == IssueSeverity.MEDIUM)

        # Deduct from consistency based on issue severity
        consistency_score = max(
            0.0,
            1.0 - (critical_issues * 0.3 + high_issues * 0.15 + medium_issues * 0.05),
        )

        return QualityMetrics(
            overall_confidence=overall_confidence,
            schema_coverage=schema_coverage,
            required_field_coverage=required_field_coverage,
            optional_field_coverage=optional_field_coverage,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            min_field_confidence=min_field_confidence,
            fields_with_low_confidence=fields_with_low_confidence,
            total_fields=total_fields,
            extracted_fields=extracted_fields,
        )

    def _derive_confidence_from_completeness(
        self,
        required_field_coverage: float,
        extracted_fields: int,
        total_fields: int,
    ) -> float:
        """Derive a confidence score when no explicit field confidence is available.

        This method addresses the issue where initial extractions have no
        per-field confidence scores, causing the system to default to 0.5
        and waste iterations on unnecessary refinement.

        The derived confidence is based on:
        - Required field coverage (60% weight): Most important for quality
        - Overall field coverage (30% weight): How many fields have values
        - Base confidence (10%): Minimum confidence for any extraction

        This ensures that complete, correct extractions can converge on the
        first iteration if all required fields are present.

        Args:
            required_field_coverage: Percentage of required fields with values (0-1).
            extracted_fields: Number of fields with non-null values.
            total_fields: Total number of fields in schema.

        Returns:
            Derived confidence score between 0.0 and 1.0.
        """
        # Weight required fields heavily - they're the primary quality indicator
        required_weight = 0.60
        coverage_weight = 0.30
        base_weight = 0.10

        # Calculate overall field coverage
        field_coverage = extracted_fields / total_fields if total_fields > 0 else 0.0

        # Derive confidence from completeness metrics
        derived_confidence = (
            required_field_coverage * required_weight
            + field_coverage * coverage_weight
            + base_weight
        )

        # If all required fields are present, boost confidence slightly
        # This reflects the fact that complete extractions are more trustworthy
        if required_field_coverage >= 1.0:
            derived_confidence = min(1.0, derived_confidence + 0.05)

        # Clamp to valid range
        return max(0.0, min(1.0, derived_confidence))

    def _perform_llm_analysis(
        self,
        extraction_result: ExtractionResult,
        schema_info: SchemaInfo,
        metrics: QualityMetrics,
        processing_category: ProcessingCategory,
    ) -> dict[str, Any]:
        """Perform LLM-based quality analysis.

        Args:
            extraction_result: Extraction result to analyze.
            schema_info: Schema information.
            metrics: Already computed metrics.
            processing_category: Type of document processing used.

        Returns:
            Dictionary with LLM analysis results.

        Raises:
            VerificationError: If LLM analysis fails.
        """
        # Build prompt
        prompt = self._build_verification_prompt(
            extraction_result, schema_info, metrics, processing_category
        )

        try:
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)

            # Parse response
            content = response.content
            if not isinstance(content, str):
                content = str(content)

            result = self._parse_llm_response(content)

            # Extract token usage
            usage_metadata = getattr(response, "usage_metadata", None) or {}
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            result["total_tokens"] = total_tokens
            result["prompt_tokens"] = prompt_tokens
            result["completion_tokens"] = completion_tokens

            return result

        except Exception as e:
            if isinstance(e, VerificationError):
                raise
            raise VerificationError(
                f"LLM analysis failed: {e}",
                error_type="llm_error",
                details={"original_error": str(e)},
            ) from e

    def _build_verification_prompt(
        self,
        extraction_result: ExtractionResult,
        schema_info: SchemaInfo,
        metrics: QualityMetrics,
        processing_category: ProcessingCategory,
    ) -> ChatPromptTemplate:
        """Build the verification prompt.

        Args:
            extraction_result: Extraction result to analyze.
            schema_info: Schema information.
            metrics: Already computed metrics.
            processing_category: Type of document processing.

        Returns:
            Configured ChatPromptTemplate.
        """
        # Format field extractions
        field_extractions_str = self._format_field_extractions(
            extraction_result.field_extractions
        )

        # Format required and optional fields
        required_fields_str = self._format_fields(schema_info.required_fields)
        optional_fields_str = self._format_fields(schema_info.optional_fields)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.VERIFICATION_SYSTEM_PROMPT),
                ("human", self.VERIFICATION_USER_PROMPT),
            ]
        )

        return prompt.partial(
            schema=json.dumps(schema_info.schema, indent=2),
            required_fields=required_fields_str,
            optional_fields=optional_fields_str,
            extracted_data=json.dumps(extraction_result.extracted_data, indent=2),
            field_extractions=field_extractions_str,
            processing_type=processing_category.value,
            total_fields=str(metrics.total_fields),
            extracted_fields=str(metrics.extracted_fields),
            overall_confidence=metrics.overall_confidence,
        )

    def _format_field_extractions(
        self,
        field_extractions: list[FieldExtraction],
    ) -> str:
        """Format field extractions for the prompt.

        Args:
            field_extractions: List of field extractions.

        Returns:
            Formatted string of field extractions.
        """
        if not field_extractions:
            return "No field extraction details available."

        parts: list[str] = []
        for fe in field_extractions:
            confidence_str = (
                f"{fe.confidence:.2f}" if fe.confidence is not None else "N/A"
            )
            value_preview = str(fe.value)[:100] + (
                "..." if len(str(fe.value)) > 100 else ""
            )
            parts.append(
                f"- {fe.field_path}: confidence={confidence_str}, value={value_preview}"
            )

        return "\n".join(parts)

    def _format_fields(self, fields: list[FieldInfo]) -> str:
        """Format field information for the prompt.

        Args:
            fields: List of field info objects.

        Returns:
            Formatted string of field information.
        """
        if not fields:
            return "None"

        return "\n".join(
            f"- {f.path}: {f.field_type}"
            + (f" - {f.description}" if f.description else "")
            for f in fields
        )

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM verification response.

        Args:
            response: Raw response string.

        Returns:
            Parsed response data.

        Raises:
            VerificationError: If parsing fails.
        """
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                raise VerificationError(
                    "Expected JSON object response",
                    error_type="parse_error",
                    details={"received_type": type(data).__name__},
                )

            # Parse issues
            issues: list[VerificationIssue] = []
            for issue_data in data.get("issues", []):
                try:
                    issue_type = IssueType(issue_data.get("issue_type", "format_error"))
                except ValueError:
                    issue_type = IssueType.FORMAT_ERROR

                try:
                    severity = IssueSeverity(
                        issue_data.get("severity", "medium").lower()
                    )
                except ValueError:
                    severity = IssueSeverity.MEDIUM

                issues.append(
                    VerificationIssue(
                        issue_type=issue_type,
                        field_path=issue_data.get("field_path", "unknown"),
                        message=issue_data.get("message", "Unknown issue"),
                        severity=severity,
                        suggestion=issue_data.get("suggestion"),
                    )
                )

            return {
                "consistency_score": float(data.get("consistency_score", 0.5)),
                "issues": issues,
                "passed_checks": data.get("passed_checks", []),
                "recommendations": data.get("recommendations", []),
                "analysis": data.get("analysis", ""),
            }

        except json.JSONDecodeError:
            # Try to extract JSON from response
            try:
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return self._parse_llm_response(json_str)
            except Exception:
                pass

            raise VerificationError(
                "Failed to parse verification response as JSON",
                error_type="parse_error",
                details={"response_preview": response[:500]},
            ) from None

    def _get_rule_based_passed_checks(
        self,
        metrics: QualityMetrics,
        thresholds: QualityThreshold,
        issues: list[VerificationIssue],
    ) -> list[str]:
        """Get list of rule-based checks that passed.

        Args:
            metrics: Quality metrics.
            thresholds: Quality thresholds.
            issues: Identified issues.

        Returns:
            List of passed check descriptions.
        """
        passed: list[str] = []

        if metrics.required_field_coverage >= thresholds.required_field_coverage:
            passed.append(
                f"Required field coverage ({metrics.required_field_coverage:.0%}) "
                f"meets threshold ({thresholds.required_field_coverage:.0%})"
            )

        if metrics.overall_confidence >= thresholds.min_overall_confidence:
            passed.append(
                f"Overall confidence ({metrics.overall_confidence:.2f}) "
                f"meets threshold ({thresholds.min_overall_confidence:.2f})"
            )

        schema_violations = [
            i for i in issues if i.issue_type == IssueType.SCHEMA_VIOLATION
        ]
        if not schema_violations:
            passed.append("Schema validation passed")

        missing_required = [
            i for i in issues if i.issue_type == IssueType.MISSING_REQUIRED_FIELD
        ]
        if not missing_required:
            passed.append("All required fields present")

        return passed

    def _determine_status(
        self,
        metrics: QualityMetrics,
        thresholds: QualityThreshold,
        issues: list[VerificationIssue],
    ) -> VerificationStatus:
        """Determine overall verification status.

        Args:
            metrics: Quality metrics.
            thresholds: Quality thresholds.
            issues: Identified issues.

        Returns:
            VerificationStatus based on analysis.
        """
        # Check for critical issues
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        if critical_issues:
            return VerificationStatus.FAILED

        # Check required field coverage
        if metrics.required_field_coverage < thresholds.required_field_coverage:
            return VerificationStatus.FAILED

        # Check overall confidence
        if metrics.overall_confidence < thresholds.min_overall_confidence:
            return VerificationStatus.NEEDS_IMPROVEMENT

        # Check for high severity issues
        high_issues = [i for i in issues if i.severity == IssueSeverity.HIGH]
        if len(high_issues) >= 3:
            return VerificationStatus.NEEDS_IMPROVEMENT

        # Check consistency score
        if metrics.consistency_score < 0.5:
            return VerificationStatus.NEEDS_IMPROVEMENT

        return VerificationStatus.PASSED

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get a value from nested dictionary using dot notation.

        Args:
            data: Dictionary to get value from.
            path: Dot-separated path (e.g., 'address.city').

        Returns:
            Value at path or None if not found.
        """
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def verify_quick(
        self,
        extraction_result: ExtractionResult,
        schema_info: SchemaInfo,
        thresholds: QualityThreshold | None = None,
    ) -> VerificationReport:
        """Perform quick verification without LLM analysis.

        This is faster but less thorough. Use for initial checks or
        when LLM analysis is not needed.

        Args:
            extraction_result: The extraction result to verify.
            schema_info: Schema information for validation.
            thresholds: Quality thresholds. Defaults to settings-based thresholds.

        Returns:
            VerificationReport with verification results.
        """
        return self.verify(
            extraction_result=extraction_result,
            schema_info=schema_info,
            thresholds=thresholds,
            use_llm_analysis=False,
        )
