"""Tests for the quality verification agent."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from agentic_document_extraction.agents.planner import QualityThreshold
from agentic_document_extraction.agents.verifier import (
    IssueSeverity,
    IssueType,
    QualityMetrics,
    QualityVerificationAgent,
    VerificationError,
    VerificationIssue,
    VerificationReport,
    VerificationStatus,
)
from agentic_document_extraction.models import ProcessingCategory
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.schema_validator import FieldInfo, SchemaInfo


# Test fixtures
@pytest.fixture
def simple_schema_info() -> SchemaInfo:
    """Create a simple schema for testing."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's name"},
            "age": {"type": "integer", "description": "Person's age"},
        },
        "required": ["name"],
    }
    return SchemaInfo(
        schema=schema,
        required_fields=[
            FieldInfo(name="name", field_type="string", required=True, path="name")
        ],
        optional_fields=[
            FieldInfo(name="age", field_type="integer", required=False, path="age")
        ],
        schema_type="object",
    )


@pytest.fixture
def complex_schema_info() -> SchemaInfo:
    """Create a complex schema for testing."""
    schema = {
        "type": "object",
        "properties": {
            "company": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "employees": {"type": "integer"},
                },
                "required": ["name"],
            },
            "products": {
                "type": "array",
                "items": {"type": "string"},
            },
            "revenue": {"type": "number"},
        },
        "required": ["company", "products"],
    }
    return SchemaInfo(
        schema=schema,
        required_fields=[
            FieldInfo(
                name="company",
                field_type="object",
                required=True,
                path="company",
                nested_fields=[
                    FieldInfo(
                        name="name",
                        field_type="string",
                        required=True,
                        path="company.name",
                    ),
                    FieldInfo(
                        name="employees",
                        field_type="integer",
                        required=False,
                        path="company.employees",
                    ),
                ],
            ),
            FieldInfo(
                name="products",
                field_type="array",
                required=True,
                path="products",
            ),
        ],
        optional_fields=[
            FieldInfo(
                name="revenue",
                field_type="number",
                required=False,
                path="revenue",
            ),
        ],
        schema_type="object",
    )


@pytest.fixture
def good_extraction_result() -> ExtractionResult:
    """Create a good extraction result for testing."""
    return ExtractionResult(
        extracted_data={"name": "John Doe", "age": 30},
        field_extractions=[
            FieldExtraction(
                field_path="name",
                value="John Doe",
                confidence=0.95,
            ),
            FieldExtraction(
                field_path="age",
                value=30,
                confidence=0.85,
            ),
        ],
        model_used="gpt-4o",
        total_tokens=500,
    )


@pytest.fixture
def poor_extraction_result() -> ExtractionResult:
    """Create a poor extraction result with missing required fields."""
    return ExtractionResult(
        extracted_data={"age": 30},  # Missing required "name"
        field_extractions=[
            FieldExtraction(
                field_path="name",
                value=None,
                confidence=0.2,
            ),
            FieldExtraction(
                field_path="age",
                value=30,
                confidence=0.4,
            ),
        ],
        model_used="gpt-4o",
        total_tokens=300,
    )


@pytest.fixture
def low_confidence_result() -> ExtractionResult:
    """Create an extraction result with low confidence scores."""
    return ExtractionResult(
        extracted_data={"name": "Maybe John", "age": 25},
        field_extractions=[
            FieldExtraction(
                field_path="name",
                value="Maybe John",
                confidence=0.35,
            ),
            FieldExtraction(
                field_path="age",
                value=25,
                confidence=0.40,
            ),
        ],
        model_used="gpt-4o",
        total_tokens=400,
    )


@pytest.fixture
def complex_extraction_result() -> ExtractionResult:
    """Create a complex extraction result."""
    return ExtractionResult(
        extracted_data={
            "company": {"name": "Acme Corp", "employees": 100},
            "products": ["Widget", "Gadget"],
            "revenue": 1000000.0,
        },
        field_extractions=[
            FieldExtraction(
                field_path="company",
                value={"name": "Acme Corp", "employees": 100},
                confidence=0.90,
            ),
            FieldExtraction(
                field_path="company.name",
                value="Acme Corp",
                confidence=0.92,
            ),
            FieldExtraction(
                field_path="company.employees",
                value=100,
                confidence=0.85,
            ),
            FieldExtraction(
                field_path="products",
                value=["Widget", "Gadget"],
                confidence=0.88,
            ),
            FieldExtraction(
                field_path="revenue",
                value=1000000.0,
                confidence=0.80,
            ),
        ],
        model_used="gpt-4o",
        total_tokens=800,
    )


@pytest.fixture
def default_thresholds() -> QualityThreshold:
    """Create default quality thresholds for testing."""
    return QualityThreshold(
        min_overall_confidence=0.7,
        min_field_confidence=0.5,
        required_field_coverage=0.9,
        max_iterations=3,
    )


@pytest.fixture
def strict_thresholds() -> QualityThreshold:
    """Create strict quality thresholds for testing."""
    return QualityThreshold(
        min_overall_confidence=0.9,
        min_field_confidence=0.8,
        required_field_coverage=1.0,
        max_iterations=5,
    )


class TestVerificationIssue:
    """Tests for VerificationIssue dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        issue = VerificationIssue(
            issue_type=IssueType.MISSING_REQUIRED_FIELD,
            field_path="name",
            message="Required field 'name' is missing",
            severity=IssueSeverity.CRITICAL,
            current_value=None,
            expected="A string value",
            suggestion="Re-extract with focus on finding name",
        )

        result = issue.to_dict()

        assert result["issue_type"] == "missing_required_field"
        assert result["field_path"] == "name"
        assert result["message"] == "Required field 'name' is missing"
        assert result["severity"] == "critical"
        assert result["current_value"] is None
        assert result["expected"] == "A string value"
        assert result["suggestion"] == "Re-extract with focus on finding name"


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = QualityMetrics(
            overall_confidence=0.85,
            schema_coverage=0.9,
            required_field_coverage=1.0,
            optional_field_coverage=0.5,
            completeness_score=0.85,
            consistency_score=0.95,
            min_field_confidence=0.75,
            fields_with_low_confidence=1,
            total_fields=4,
            extracted_fields=3,
        )

        result = metrics.to_dict()

        assert result["overall_confidence"] == 0.85
        assert result["schema_coverage"] == 0.9
        assert result["required_field_coverage"] == 1.0
        assert result["optional_field_coverage"] == 0.5
        assert result["completeness_score"] == 0.85
        assert result["consistency_score"] == 0.95
        assert result["min_field_confidence"] == 0.75
        assert result["fields_with_low_confidence"] == 1
        assert result["total_fields"] == 4
        assert result["extracted_fields"] == 3


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = QualityMetrics(
            overall_confidence=0.85,
            schema_coverage=0.9,
            required_field_coverage=1.0,
            optional_field_coverage=0.5,
            completeness_score=0.85,
            consistency_score=0.95,
            min_field_confidence=0.75,
            fields_with_low_confidence=0,
            total_fields=4,
            extracted_fields=3,
        )

        report = VerificationReport(
            status=VerificationStatus.PASSED,
            metrics=metrics,
            issues=[],
            passed_checks=["All required fields present", "Schema validation passed"],
            recommendations=[],
            llm_analysis="Extraction looks good",
            model_used="gpt-4o",
            total_tokens=200,
            processing_time_seconds=1.5,
        )

        result = report.to_dict()

        assert result["status"] == "passed"
        assert result["metrics"]["overall_confidence"] == 0.85
        assert len(result["issues"]) == 0
        assert "All required fields present" in result["passed_checks"]
        assert result["llm_analysis"] == "Extraction looks good"
        assert result["metadata"]["model_used"] == "gpt-4o"

    def test_passed_property(self) -> None:
        """Test passed property."""
        metrics = QualityMetrics(
            overall_confidence=0.85,
            schema_coverage=0.9,
            required_field_coverage=1.0,
            optional_field_coverage=0.5,
            completeness_score=0.85,
            consistency_score=0.95,
            min_field_confidence=0.75,
            fields_with_low_confidence=0,
            total_fields=4,
            extracted_fields=3,
        )

        passed_report = VerificationReport(
            status=VerificationStatus.PASSED,
            metrics=metrics,
        )
        failed_report = VerificationReport(
            status=VerificationStatus.FAILED,
            metrics=metrics,
        )

        assert passed_report.passed is True
        assert failed_report.passed is False

    def test_critical_issues_property(self) -> None:
        """Test critical_issues property."""
        metrics = QualityMetrics(
            overall_confidence=0.5,
            schema_coverage=0.5,
            required_field_coverage=0.5,
            optional_field_coverage=0.5,
            completeness_score=0.5,
            consistency_score=0.5,
            min_field_confidence=0.5,
            fields_with_low_confidence=2,
            total_fields=4,
            extracted_fields=2,
        )

        issues = [
            VerificationIssue(
                issue_type=IssueType.MISSING_REQUIRED_FIELD,
                field_path="name",
                message="Missing required field",
                severity=IssueSeverity.CRITICAL,
            ),
            VerificationIssue(
                issue_type=IssueType.LOW_CONFIDENCE,
                field_path="age",
                message="Low confidence",
                severity=IssueSeverity.MEDIUM,
            ),
        ]

        report = VerificationReport(
            status=VerificationStatus.FAILED,
            metrics=metrics,
            issues=issues,
        )

        critical = report.critical_issues
        assert len(critical) == 1
        assert critical[0].field_path == "name"

    def test_high_priority_issues_property(self) -> None:
        """Test high_priority_issues property."""
        metrics = QualityMetrics(
            overall_confidence=0.5,
            schema_coverage=0.5,
            required_field_coverage=0.5,
            optional_field_coverage=0.5,
            completeness_score=0.5,
            consistency_score=0.5,
            min_field_confidence=0.5,
            fields_with_low_confidence=3,
            total_fields=4,
            extracted_fields=2,
        )

        issues = [
            VerificationIssue(
                issue_type=IssueType.MISSING_REQUIRED_FIELD,
                field_path="name",
                message="Critical issue",
                severity=IssueSeverity.CRITICAL,
            ),
            VerificationIssue(
                issue_type=IssueType.LOW_CONFIDENCE,
                field_path="age",
                message="High severity issue",
                severity=IssueSeverity.HIGH,
            ),
            VerificationIssue(
                issue_type=IssueType.FORMAT_ERROR,
                field_path="email",
                message="Medium issue",
                severity=IssueSeverity.MEDIUM,
            ),
        ]

        report = VerificationReport(
            status=VerificationStatus.FAILED,
            metrics=metrics,
            issues=issues,
        )

        high_priority = report.high_priority_issues
        assert len(high_priority) == 2


class TestRuleBasedVerification:
    """Tests for rule-based verification checks."""

    def test_verify_good_extraction_quick(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test quick verification of good extraction."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        assert report.status == VerificationStatus.PASSED
        assert report.metrics.required_field_coverage == 1.0
        assert len(report.critical_issues) == 0

    def test_verify_missing_required_field(
        self,
        poor_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test verification detects missing required fields."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=poor_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        assert report.status == VerificationStatus.FAILED
        assert report.metrics.required_field_coverage == 0.0

        # Should have a missing required field issue
        missing_issues = [
            i for i in report.issues if i.issue_type == IssueType.MISSING_REQUIRED_FIELD
        ]
        assert len(missing_issues) == 1
        assert missing_issues[0].field_path == "name"
        assert missing_issues[0].severity == IssueSeverity.CRITICAL

    def test_verify_low_confidence(
        self,
        low_confidence_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test verification detects low confidence scores."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=low_confidence_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        # Should have low confidence issues
        low_confidence_issues = [
            i for i in report.issues if i.issue_type == IssueType.LOW_CONFIDENCE
        ]
        assert len(low_confidence_issues) == 2

        # Should need improvement due to low confidence
        assert report.status == VerificationStatus.NEEDS_IMPROVEMENT

    def test_verify_complex_extraction(
        self,
        complex_extraction_result: ExtractionResult,
        complex_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test verification of complex extraction."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=complex_extraction_result,
            schema_info=complex_schema_info,
            thresholds=default_thresholds,
        )

        assert report.status == VerificationStatus.PASSED
        assert report.metrics.required_field_coverage == 1.0

    def test_verify_empty_required_array(
        self,
        complex_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test verification detects empty required arrays."""
        extraction_result = ExtractionResult(
            extracted_data={
                "company": {"name": "Acme Corp"},
                "products": [],  # Empty required array
            },
            field_extractions=[
                FieldExtraction(field_path="company", value={"name": "Acme Corp"}),
                FieldExtraction(field_path="products", value=[]),
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=complex_schema_info,
            thresholds=default_thresholds,
        )

        empty_array_issues = [
            i for i in report.issues if i.issue_type == IssueType.EMPTY_ARRAY
        ]
        assert len(empty_array_issues) == 1
        assert empty_array_issues[0].field_path == "products"

    def test_verify_array_nested_fields_no_false_nulls(
        self,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test array item fields do not trigger null warnings when populated."""
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "employees": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                            "required": ["name", "email"],
                        },
                    }
                },
                "required": ["employees"],
            },
            required_fields=[
                FieldInfo(
                    name="employees",
                    path="employees",
                    field_type="array",
                    required=True,
                    nested_fields=[
                        FieldInfo(
                            name="name",
                            path="employees[].name",
                            field_type="string",
                            required=True,
                        ),
                        FieldInfo(
                            name="email",
                            path="employees[].email",
                            field_type="string",
                            required=True,
                        ),
                    ],
                )
            ],
            optional_fields=[],
            schema_type="object",
        )

        extraction_result = ExtractionResult(
            extracted_data={
                "employees": [
                    {"name": "Ada Lovelace", "email": "ada@example.com"},
                    {"name": "Grace Hopper", "email": "grace@example.com"},
                ]
            },
            field_extractions=[],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=schema_info,
            thresholds=default_thresholds,
        )

        null_value_issues = [
            i for i in report.issues if i.issue_type == IssueType.NULL_VALUE
        ]
        assert len(null_value_issues) == 0


class TestSchemaValidation:
    """Tests for JSON schema validation."""

    def test_schema_validation_passes(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test that valid data passes schema validation."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
        )

        schema_violations = [
            i for i in report.issues if i.issue_type == IssueType.SCHEMA_VIOLATION
        ]
        assert len(schema_violations) == 0
        assert "Schema validation passed" in report.passed_checks

    def test_schema_validation_fails_wrong_type(
        self,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test that invalid data type fails schema validation."""
        extraction_result = ExtractionResult(
            extracted_data={"name": "John", "age": "not a number"},
            field_extractions=[
                FieldExtraction(field_path="name", value="John"),
                FieldExtraction(field_path="age", value="not a number"),
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=simple_schema_info,
        )

        schema_violations = [
            i for i in report.issues if i.issue_type == IssueType.SCHEMA_VIOLATION
        ]
        assert len(schema_violations) == 1
        assert schema_violations[0].severity == IssueSeverity.CRITICAL


class TestMetricsComputation:
    """Tests for quality metrics computation."""

    def test_compute_metrics_full_coverage(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test metrics computation with full coverage."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        metrics = report.metrics
        assert metrics.required_field_coverage == 1.0
        assert metrics.schema_coverage == 1.0
        assert metrics.overall_confidence == pytest.approx(0.9)  # (0.95 + 0.85) / 2
        assert metrics.min_field_confidence == 0.85
        assert metrics.total_fields == 2
        assert metrics.extracted_fields == 2

    def test_compute_metrics_partial_coverage(
        self,
        poor_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test metrics computation with partial coverage."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=poor_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        metrics = report.metrics
        assert metrics.required_field_coverage == 0.0  # name is missing
        assert metrics.optional_field_coverage == 1.0  # age is present
        assert metrics.schema_coverage == 0.5  # 1 of 2 fields

    def test_compute_metrics_no_confidence_scores(
        self,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test metrics derived from completeness when no confidence scores available.

        When no explicit field confidence scores exist (initial extraction),
        confidence is derived from completeness metrics:
        - 60% weight on required field coverage
        - 30% weight on overall field coverage
        - 10% base confidence
        - +5% bonus for 100% required field coverage

        With 100% coverage, derived confidence should be 1.0 (capped).
        """
        extraction_result = ExtractionResult(
            extracted_data={"name": "John", "age": 30},
            field_extractions=[
                FieldExtraction(field_path="name", value="John"),  # No confidence
                FieldExtraction(field_path="age", value=30),  # No confidence
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        # Derived confidence: 1.0*0.6 + 1.0*0.3 + 0.1 + 0.05 (bonus) = 1.05 → capped at 1.0
        assert report.metrics.overall_confidence == 1.0
        assert report.metrics.fields_with_low_confidence == 0
        # Verification should pass since derived confidence (1.0) >= threshold (0.7)
        assert report.status == VerificationStatus.PASSED

    def test_compute_metrics_derived_confidence_partial_coverage(
        self,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test derived confidence with partial field coverage.

        With 50% required field coverage and 50% overall coverage:
        - Derived confidence = 0.5*0.6 + 0.5*0.3 + 0.1 = 0.55
        - No bonus (required coverage < 100%)
        - Should be below 0.7 threshold
        """
        # Create a schema with 2 required fields
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["name", "email"],
            },
            required_fields=[
                FieldInfo(name="name", path="name", field_type="string", required=True),
                FieldInfo(
                    name="email", path="email", field_type="string", required=True
                ),
            ],
            optional_fields=[],
            schema_type="object",
        )

        # Only one field extracted (50% coverage)
        extraction_result = ExtractionResult(
            extracted_data={"name": "John", "email": None},
            field_extractions=[
                FieldExtraction(field_path="name", value="John"),  # No confidence
                FieldExtraction(field_path="email", value=None),  # No confidence
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=schema_info,
            thresholds=default_thresholds,
        )

        # Derived confidence: 0.5*0.6 + 0.5*0.3 + 0.1 = 0.55
        assert report.metrics.overall_confidence == pytest.approx(0.55)
        assert report.metrics.required_field_coverage == 0.5


class TestStatusDetermination:
    """Tests for verification status determination."""

    def test_status_passed(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test PASSED status for good extraction."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        assert report.status == VerificationStatus.PASSED

    def test_status_failed_critical_issue(
        self,
        poor_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test FAILED status when critical issues exist."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=poor_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        assert report.status == VerificationStatus.FAILED

    def test_status_failed_low_coverage(
        self,
        simple_schema_info: SchemaInfo,
        strict_thresholds: QualityThreshold,
    ) -> None:
        """Test FAILED status when required field coverage is low."""
        extraction_result = ExtractionResult(
            extracted_data={"name": None, "age": 30},
            field_extractions=[
                FieldExtraction(field_path="name", value=None, confidence=0.9),
                FieldExtraction(field_path="age", value=30, confidence=0.9),
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=simple_schema_info,
            thresholds=strict_thresholds,
        )

        assert report.status == VerificationStatus.FAILED

    def test_status_needs_improvement_low_confidence(
        self,
        low_confidence_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test NEEDS_IMPROVEMENT status for low confidence."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=low_confidence_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        assert report.status == VerificationStatus.NEEDS_IMPROVEMENT


class TestLLMAnalysis:
    """Tests for LLM-based analysis."""

    def test_verify_with_llm_analysis(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test verification with LLM analysis enabled."""
        llm_response = {
            "consistency_score": 0.95,
            "issues": [
                {
                    "issue_type": "format_error",
                    "field_path": "name",
                    "message": "Name format could be improved",
                    "severity": "low",
                    "suggestion": "Consider using full name format",
                }
            ],
            "passed_checks": [
                "All fields are logically consistent",
                "No potential hallucinations detected",
            ],
            "recommendations": [
                "Consider adding middle name if available",
            ],
            "analysis": "The extraction looks accurate and complete.",
        }

        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 500,
                "output_tokens": 200,
                "total_tokens": 700,
            },
        )

        agent = QualityVerificationAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        report = agent.verify(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
            use_llm_analysis=True,
        )

        assert report.status == VerificationStatus.PASSED
        assert report.metrics.consistency_score == 0.95
        assert report.llm_analysis == "The extraction looks accurate and complete."
        assert "All fields are logically consistent" in report.passed_checks
        assert len(report.recommendations) > 0
        assert report.total_tokens == 700

    def test_verify_llm_detects_issues(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test that LLM can detect additional issues."""
        llm_response = {
            "consistency_score": 0.6,
            "issues": [
                {
                    "issue_type": "logical_inconsistency",
                    "field_path": "age",
                    "message": "Age seems inconsistent with context",
                    "severity": "medium",
                    "suggestion": "Verify age against other document context",
                },
                {
                    "issue_type": "potential_hallucination",
                    "field_path": "name",
                    "message": "Name might be fabricated",
                    "severity": "high",
                    "suggestion": "Verify name exists in source document",
                },
            ],
            "passed_checks": [],
            "recommendations": [
                "Re-verify the source document",
            ],
            "analysis": "Several inconsistencies detected.",
        }

        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 150,
                "total_tokens": 550,
            },
        )

        agent = QualityVerificationAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        report = agent.verify(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
            use_llm_analysis=True,
        )

        # Should have LLM-detected issues
        llm_issues = [
            i
            for i in report.issues
            if i.issue_type
            in (IssueType.LOGICAL_INCONSISTENCY, IssueType.POTENTIAL_HALLUCINATION)
        ]
        assert len(llm_issues) == 2

    def test_verify_llm_failure_fallback(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test that verification continues when LLM fails."""
        agent = QualityVerificationAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("LLM API error")
        agent._agent = mock_agent

        # Should not raise - falls back to rule-based only
        report = agent.verify(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
            use_llm_analysis=True,
        )

        assert report.status == VerificationStatus.PASSED
        assert "LLM analysis unavailable" in report.llm_analysis


class TestVerificationStrategies:
    """Tests for different verification strategies."""

    def test_text_processing_category(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test verification with text processing category."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
        )

        # Should complete successfully for text-based
        assert report.status == VerificationStatus.PASSED

    def test_visual_processing_category(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test verification with visual processing category."""
        llm_response = {
            "consistency_score": 0.9,
            "issues": [],
            "passed_checks": ["Visual extraction verified"],
            "recommendations": [],
            "analysis": "Visual document extraction looks good.",
        }

        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 300,
                "output_tokens": 100,
                "total_tokens": 400,
            },
        )

        agent = QualityVerificationAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        report = agent.verify(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            processing_category=ProcessingCategory.VISUAL,
            use_llm_analysis=True,
        )

        assert report.status == VerificationStatus.PASSED


class TestVerificationErrors:
    """Tests for error handling."""

    def test_no_api_key_error(self) -> None:
        """Test error when API key is missing."""
        agent = QualityVerificationAgent(api_key="")

        with pytest.raises(VerificationError) as exc_info:
            _ = agent.llm

        assert exc_info.value.error_type == "configuration_error"
        assert "openai_api_key" in str(exc_info.value.details)

    def test_verification_error_attributes(self) -> None:
        """Test VerificationError attributes."""
        error = VerificationError(
            "Test error",
            error_type="test_error",
            details={"key": "value"},
        )

        assert str(error) == "Test error"
        assert error.error_type == "test_error"
        assert error.details == {"key": "value"}


class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_response(self) -> None:
        """Test parsing valid JSON response."""
        agent = QualityVerificationAgent(api_key="test-key")

        response = json.dumps(
            {
                "consistency_score": 0.85,
                "issues": [
                    {
                        "issue_type": "format_error",
                        "field_path": "email",
                        "message": "Invalid format",
                        "severity": "medium",
                    }
                ],
                "passed_checks": ["Check 1"],
                "recommendations": ["Rec 1"],
                "analysis": "Analysis text",
            }
        )

        result = agent._parse_llm_response(response)

        assert result["consistency_score"] == 0.85
        assert len(result["issues"]) == 1
        assert result["issues"][0].issue_type == IssueType.FORMAT_ERROR
        assert result["passed_checks"] == ["Check 1"]
        assert result["recommendations"] == ["Rec 1"]
        assert result["analysis"] == "Analysis text"

    def test_parse_response_with_wrapper(self) -> None:
        """Test parsing JSON with extra text around it."""
        agent = QualityVerificationAgent(api_key="test-key")

        response = (
            'Here is my analysis: {"consistency_score": 0.9, '
            '"issues": [], "passed_checks": [], "recommendations": [], '
            '"analysis": "Good"} That is all.'
        )

        result = agent._parse_llm_response(response)
        assert result["consistency_score"] == 0.9

    def test_parse_response_unknown_issue_type(self) -> None:
        """Test parsing with unknown issue type falls back to format_error."""
        agent = QualityVerificationAgent(api_key="test-key")

        response = json.dumps(
            {
                "consistency_score": 0.8,
                "issues": [
                    {
                        "issue_type": "unknown_type",
                        "field_path": "field",
                        "message": "Test",
                        "severity": "low",
                    }
                ],
                "passed_checks": [],
                "recommendations": [],
                "analysis": "",
            }
        )

        result = agent._parse_llm_response(response)
        assert result["issues"][0].issue_type == IssueType.FORMAT_ERROR

    def test_parse_invalid_response(self) -> None:
        """Test parsing invalid JSON raises error."""
        agent = QualityVerificationAgent(api_key="test-key")

        with pytest.raises(VerificationError) as exc_info:
            agent._parse_llm_response("This is not JSON at all!")

        assert exc_info.value.error_type == "parse_error"


class TestNestedValueAccess:
    """Tests for nested value access utility."""

    def test_get_nested_value_simple(self) -> None:
        """Test getting simple nested value."""
        agent = QualityVerificationAgent(api_key="test-key")

        data = {"name": "John", "age": 30}
        assert agent._get_nested_value(data, "name") == "John"
        assert agent._get_nested_value(data, "age") == 30

    def test_get_nested_value_deep(self) -> None:
        """Test getting deeply nested value."""
        agent = QualityVerificationAgent(api_key="test-key")

        data = {
            "company": {
                "address": {
                    "city": "San Francisco",
                }
            }
        }
        assert agent._get_nested_value(data, "company.address.city") == "San Francisco"

    def test_get_nested_value_missing(self) -> None:
        """Test getting missing nested value returns None."""
        agent = QualityVerificationAgent(api_key="test-key")

        data = {"name": "John"}
        assert agent._get_nested_value(data, "address.city") is None
        assert agent._get_nested_value(data, "nonexistent") is None


class TestPassedChecks:
    """Tests for passed checks generation."""

    def test_passed_checks_full_coverage(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test passed checks for full coverage."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
        )

        assert "All required fields present" in report.passed_checks
        assert "Schema validation passed" in report.passed_checks
        assert any("coverage" in check.lower() for check in report.passed_checks)
        assert any("confidence" in check.lower() for check in report.passed_checks)


class TestConfigurableThresholds:
    """Tests for configurable quality thresholds."""

    def test_default_thresholds_used(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test that default thresholds are used when none provided."""
        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=good_extraction_result,
            schema_info=simple_schema_info,
            # No thresholds provided - uses defaults
        )

        # Should use default thresholds
        assert report.status == VerificationStatus.PASSED

    def test_strict_thresholds_fail_marginal(
        self,
        simple_schema_info: SchemaInfo,
        strict_thresholds: QualityThreshold,
    ) -> None:
        """Test that strict thresholds fail marginal extractions."""
        # Good enough for default, but not for strict
        extraction_result = ExtractionResult(
            extracted_data={"name": "John", "age": 30},
            field_extractions=[
                FieldExtraction(field_path="name", value="John", confidence=0.75),
                FieldExtraction(field_path="age", value=30, confidence=0.75),
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=simple_schema_info,
            thresholds=strict_thresholds,
        )

        # Should fail due to low confidence with strict thresholds
        assert report.status in (
            VerificationStatus.FAILED,
            VerificationStatus.NEEDS_IMPROVEMENT,
        )

        low_conf_issues = [
            i for i in report.issues if i.issue_type == IssueType.LOW_CONFIDENCE
        ]
        assert len(low_conf_issues) == 2  # Both fields below 0.8 threshold


class TestLLMIssueEnrichment:
    """Tests for enriching LLM-reported issues with current values."""

    def test_llm_issues_include_current_value(
        self,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test that LLM-reported issues are enriched with current_value.

        This addresses the bug where format_error issues from the LLM
        had current_value=null even when the extracted data had a value.
        """
        extraction_result = ExtractionResult(
            extracted_data={"name": "John Doe", "age": 30},
            field_extractions=[
                FieldExtraction(field_path="name", value="John Doe", confidence=0.9),
                FieldExtraction(field_path="age", value=30, confidence=0.9),
            ],
        )

        # LLM reports an issue but doesn't include current_value
        llm_response = {
            "consistency_score": 0.8,
            "issues": [
                {
                    "issue_type": "format_error",
                    "field_path": "name",
                    "message": "Some format concern",
                    "severity": "low",
                    "suggestion": "Consider reformatting",
                }
            ],
            "passed_checks": [],
            "recommendations": [],
            "analysis": "Minor formatting concern.",
        }

        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 300,
                "output_tokens": 100,
                "total_tokens": 400,
            },
        )

        agent = QualityVerificationAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        report = agent.verify(
            extraction_result=extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
            use_llm_analysis=True,
        )

        # Find the format_error issue from LLM
        format_issues = [
            i for i in report.issues if i.issue_type == IssueType.FORMAT_ERROR
        ]
        assert len(format_issues) == 1

        # The issue should have current_value populated from extraction result
        assert format_issues[0].current_value == "John Doe"

    def test_llm_issue_current_value_overrides_wrong_llm_value(
        self,
        simple_schema_info: SchemaInfo,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test that actual extracted value overrides wrong LLM-reported value.

        This addresses the case where the LLM confuses the extracted value
        with another representation (e.g., markdown formatted vs ISO date).
        """
        extraction_result = ExtractionResult(
            extracted_data={"name": "2024-01-15", "age": 30},  # ISO date as name
            field_extractions=[
                FieldExtraction(field_path="name", value="2024-01-15", confidence=0.9),
                FieldExtraction(field_path="age", value=30, confidence=0.9),
            ],
        )

        # LLM reports wrong current_value (e.g., from markdown representation)
        llm_response = {
            "consistency_score": 0.8,
            "issues": [
                {
                    "issue_type": "format_error",
                    "field_path": "name",
                    "message": "Date format is wrong",
                    "severity": "high",
                    "current_value": "January 15, 2024",  # Wrong! LLM confused
                    "suggestion": "Use ISO format",
                }
            ],
            "passed_checks": [],
            "recommendations": [],
            "analysis": "Date format issue.",
        }

        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 300,
                "output_tokens": 100,
                "total_tokens": 400,
            },
        )

        agent = QualityVerificationAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        report = agent.verify(
            extraction_result=extraction_result,
            schema_info=simple_schema_info,
            thresholds=default_thresholds,
            use_llm_analysis=True,
        )

        format_issues = [
            i for i in report.issues if i.issue_type == IssueType.FORMAT_ERROR
        ]
        assert len(format_issues) == 1

        # Should use actual extracted value, not the wrong LLM-reported value
        assert format_issues[0].current_value == "2024-01-15"

    def test_llm_issues_nested_field_current_value(
        self,
        complex_schema_info: SchemaInfo,
        complex_extraction_result: ExtractionResult,
        default_thresholds: QualityThreshold,
    ) -> None:
        """Test that nested field paths get correct current_value."""
        llm_response = {
            "consistency_score": 0.9,
            "issues": [
                {
                    "issue_type": "format_error",
                    "field_path": "company.name",
                    "message": "Name format concern",
                    "severity": "low",
                }
            ],
            "passed_checks": [],
            "recommendations": [],
            "analysis": "Analysis.",
        }

        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 300,
                "output_tokens": 100,
                "total_tokens": 400,
            },
        )

        agent = QualityVerificationAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        report = agent.verify(
            extraction_result=complex_extraction_result,
            schema_info=complex_schema_info,
            thresholds=default_thresholds,
            use_llm_analysis=True,
        )

        format_issues = [
            i for i in report.issues if i.issue_type == IssueType.FORMAT_ERROR
        ]
        assert len(format_issues) == 1
        # Should resolve nested path company.name to "Acme Corp"
        assert format_issues[0].current_value == "Acme Corp"


class TestDateFormatValidation:
    """Tests for date format validation to prevent false positives."""

    def test_valid_iso_date_no_false_positive(self) -> None:
        """Test that valid ISO 8601 dates don't trigger false positive format errors.

        This is a regression test for the invoice_date false positive issue
        where dates like '2024-01-15' were incorrectly flagged.
        """
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "invoice_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Invoice date in YYYY-MM-DD format",
                    },
                    "invoice_number": {"type": "string"},
                },
                "required": ["invoice_date", "invoice_number"],
            },
            required_fields=[
                FieldInfo(
                    name="invoice_date",
                    path="invoice_date",
                    field_type="string",
                    required=True,
                ),
                FieldInfo(
                    name="invoice_number",
                    path="invoice_number",
                    field_type="string",
                    required=True,
                ),
            ],
            optional_fields=[],
            schema_type="object",
        )

        extraction_result = ExtractionResult(
            extracted_data={
                "invoice_date": "2024-01-15",
                "invoice_number": "INV-2024-001",
            },
            field_extractions=[
                FieldExtraction(
                    field_path="invoice_date", value="2024-01-15", confidence=0.95
                ),
                FieldExtraction(
                    field_path="invoice_number", value="INV-2024-001", confidence=0.95
                ),
            ],
        )

        # LLM returns no issues for valid date format
        llm_response = {
            "consistency_score": 1.0,
            "issues": [],
            "passed_checks": [
                "All date fields use valid ISO 8601 format (YYYY-MM-DD)",
                "All required fields present",
            ],
            "recommendations": [],
            "analysis": "Invoice extraction is complete and accurate.",
        }

        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 150,
                "total_tokens": 550,
            },
        )

        agent = QualityVerificationAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        report = agent.verify(
            extraction_result=extraction_result,
            schema_info=schema_info,
            use_llm_analysis=True,
        )

        # Should pass with no format errors
        assert report.status == VerificationStatus.PASSED
        format_issues = [
            i for i in report.issues if i.issue_type == IssueType.FORMAT_ERROR
        ]
        assert len(format_issues) == 0

    def test_system_prompt_includes_date_format_guidance(self) -> None:
        """Verify the system prompt includes guidance about valid date formats."""
        agent = QualityVerificationAgent(api_key="test-key")

        # Check that the prompt includes date format guidance
        assert "YYYY-MM-DD" in agent.VERIFICATION_SYSTEM_PROMPT
        assert "format_error" in agent.VERIFICATION_SYSTEM_PROMPT.lower()
        assert "date" in agent.VERIFICATION_SYSTEM_PROMPT.lower()


class TestNullOptionalFieldsHandling:
    """Tests for handling null values in optional fields.

    This addresses the issue where optional string fields returned as null
    trigger schema violations because the schema type is 'string' without
    null allowance.
    """

    def test_null_optional_fields_stripped_before_validation(self) -> None:
        """Test that null optional fields are stripped before schema validation.

        When an LLM returns null for optional fields it can't extract,
        these should be omitted from the data before schema validation
        to avoid spurious schema_violation issues.
        """
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "signed_by": {"type": "string"},
                    "media_name": {"type": "string"},
                },
                "required": ["title"],
            },
            required_fields=[
                FieldInfo(
                    name="title", path="title", field_type="string", required=True
                )
            ],
            optional_fields=[
                FieldInfo(
                    name="signed_by",
                    path="signed_by",
                    field_type="string",
                    required=False,
                ),
                FieldInfo(
                    name="media_name",
                    path="media_name",
                    field_type="string",
                    required=False,
                ),
            ],
            schema_type="object",
        )

        # LLM returned null for optional fields it couldn't extract
        extraction_result = ExtractionResult(
            extracted_data={
                "title": "Coupon Form",
                "signed_by": None,
                "media_name": None,
            },
            field_extractions=[
                FieldExtraction(
                    field_path="title", value="Coupon Form", confidence=0.9
                ),
                FieldExtraction(field_path="signed_by", value=None, confidence=0.0),
                FieldExtraction(field_path="media_name", value=None, confidence=0.0),
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=schema_info,
        )

        # Should NOT have schema violations for null optional fields
        schema_violations = [
            i for i in report.issues if i.issue_type == IssueType.SCHEMA_VIOLATION
        ]
        assert len(schema_violations) == 0
        assert "Schema validation passed" in report.passed_checks

    def test_null_required_fields_still_fail_validation(self) -> None:
        """Test that null required fields still trigger validation issues.

        Stripping null optional fields should not affect required field validation.
        """
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["title", "code"],
            },
            required_fields=[
                FieldInfo(
                    name="title", path="title", field_type="string", required=True
                ),
                FieldInfo(name="code", path="code", field_type="string", required=True),
            ],
            optional_fields=[],
            schema_type="object",
        )

        # Required field is null
        extraction_result = ExtractionResult(
            extracted_data={
                "title": "Coupon Form",
                "code": None,  # Required field is null
            },
            field_extractions=[
                FieldExtraction(
                    field_path="title", value="Coupon Form", confidence=0.9
                ),
                FieldExtraction(field_path="code", value=None, confidence=0.0),
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=schema_info,
        )

        # Should have issues for missing required field
        missing_issues = [
            i for i in report.issues if i.issue_type == IssueType.MISSING_REQUIRED_FIELD
        ]
        assert len(missing_issues) == 1
        assert missing_issues[0].field_path == "code"

    def test_strip_null_optional_preserves_non_null_values(self) -> None:
        """Test that stripping null optionals doesn't affect non-null optional values."""
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "signed_by": {"type": "string"},
                    "media_name": {"type": "string"},
                },
                "required": ["title"],
            },
            required_fields=[
                FieldInfo(
                    name="title", path="title", field_type="string", required=True
                )
            ],
            optional_fields=[
                FieldInfo(
                    name="signed_by",
                    path="signed_by",
                    field_type="string",
                    required=False,
                ),
                FieldInfo(
                    name="media_name",
                    path="media_name",
                    field_type="string",
                    required=False,
                ),
            ],
            schema_type="object",
        )

        # One optional field has a value, one is null
        # All fields have high confidence for those that have values
        extraction_result = ExtractionResult(
            extracted_data={
                "title": "Coupon Form",
                "signed_by": "John Doe",  # Has value
                "media_name": None,  # Null
            },
            field_extractions=[
                FieldExtraction(
                    field_path="title", value="Coupon Form", confidence=0.9
                ),
                FieldExtraction(
                    field_path="signed_by", value="John Doe", confidence=0.85
                ),
                # Don't include field extraction for null field - realistic scenario
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=schema_info,
        )

        # Should have no schema violations
        schema_violations = [
            i for i in report.issues if i.issue_type == IssueType.SCHEMA_VIOLATION
        ]
        assert len(schema_violations) == 0
        # Schema validation should pass
        assert "Schema validation passed" in report.passed_checks

    def test_strip_null_optional_helper_method(self) -> None:
        """Test the _strip_null_optional_fields helper method directly."""
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"},
                    "optional_with_value": {"type": "string"},
                    "optional_null": {"type": "string"},
                },
                "required": ["required_field"],
            },
            required_fields=[
                FieldInfo(
                    name="required_field",
                    path="required_field",
                    field_type="string",
                    required=True,
                )
            ],
            optional_fields=[
                FieldInfo(
                    name="optional_with_value",
                    path="optional_with_value",
                    field_type="string",
                    required=False,
                ),
                FieldInfo(
                    name="optional_null",
                    path="optional_null",
                    field_type="string",
                    required=False,
                ),
            ],
            schema_type="object",
        )

        data = {
            "required_field": "value",
            "optional_with_value": "has value",
            "optional_null": None,
        }

        agent = QualityVerificationAgent(api_key="test-key")
        cleaned = agent._strip_null_optional_fields(data, schema_info)

        # Should remove optional_null but keep other fields
        assert "required_field" in cleaned
        assert cleaned["required_field"] == "value"
        assert "optional_with_value" in cleaned
        assert cleaned["optional_with_value"] == "has value"
        assert "optional_null" not in cleaned

        # Original data should be unchanged
        assert "optional_null" in data
        assert data["optional_null"] is None

    def test_metrics_still_count_null_optional_fields(self) -> None:
        """Test that metrics computation still sees null optional fields.

        The stripping should only affect schema validation, not metrics
        like optional_field_coverage.
        """
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "optional1": {"type": "string"},
                    "optional2": {"type": "string"},
                },
                "required": ["title"],
            },
            required_fields=[
                FieldInfo(
                    name="title", path="title", field_type="string", required=True
                )
            ],
            optional_fields=[
                FieldInfo(
                    name="optional1",
                    path="optional1",
                    field_type="string",
                    required=False,
                ),
                FieldInfo(
                    name="optional2",
                    path="optional2",
                    field_type="string",
                    required=False,
                ),
            ],
            schema_type="object",
        )

        # One optional has value, one is null
        extraction_result = ExtractionResult(
            extracted_data={
                "title": "Test",
                "optional1": "value",
                "optional2": None,
            },
            field_extractions=[
                FieldExtraction(field_path="title", value="Test", confidence=0.9),
                FieldExtraction(field_path="optional1", value="value", confidence=0.8),
                FieldExtraction(field_path="optional2", value=None, confidence=0.0),
            ],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=schema_info,
        )

        # Optional field coverage should be 50% (1 of 2 optional fields)
        assert report.metrics.optional_field_coverage == 0.5
        # Schema coverage: 2 of 3 fields have values
        assert report.metrics.schema_coverage == pytest.approx(2 / 3)

    def test_coupon_form_scenario(self) -> None:
        """Test the exact scenario from the bug report.

        POST /extract with sample_coupon_code_form.png + schema should not
        produce schema_violation issues for optional null fields.
        """
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "brands_applicable": {"type": "array", "items": {"type": "string"}},
                    "media_type": {"type": "string"},
                    "media_name": {"type": "string"},
                    "issue_frequency": {"type": "string"},
                    "issue_date": {"type": "string"},
                    "expiration_date": {"type": "string"},
                    "coupon_value": {"type": "string"},
                    "signed_by": {"type": "string"},
                    "code_assigned": {"type": "string"},
                },
                "required": [
                    "brands_applicable",
                    "media_type",
                    "issue_date",
                    "expiration_date",
                    "coupon_value",
                    "code_assigned",
                ],
            },
            required_fields=[
                FieldInfo(
                    name="brands_applicable",
                    path="brands_applicable",
                    field_type="array",
                    required=True,
                ),
                FieldInfo(
                    name="media_type",
                    path="media_type",
                    field_type="string",
                    required=True,
                ),
                FieldInfo(
                    name="issue_date",
                    path="issue_date",
                    field_type="string",
                    required=True,
                ),
                FieldInfo(
                    name="expiration_date",
                    path="expiration_date",
                    field_type="string",
                    required=True,
                ),
                FieldInfo(
                    name="coupon_value",
                    path="coupon_value",
                    field_type="string",
                    required=True,
                ),
                FieldInfo(
                    name="code_assigned",
                    path="code_assigned",
                    field_type="string",
                    required=True,
                ),
            ],
            optional_fields=[
                FieldInfo(
                    name="title", path="title", field_type="string", required=False
                ),
                FieldInfo(
                    name="media_name",
                    path="media_name",
                    field_type="string",
                    required=False,
                ),
                FieldInfo(
                    name="issue_frequency",
                    path="issue_frequency",
                    field_type="string",
                    required=False,
                ),
                FieldInfo(
                    name="signed_by",
                    path="signed_by",
                    field_type="string",
                    required=False,
                ),
            ],
            schema_type="object",
        )

        # Simulated extraction with required fields present and optional nulls
        extraction_result = ExtractionResult(
            extracted_data={
                "brands_applicable": ["KOOL", "SALEM"],
                "media_type": "FSI",
                "issue_date": "2024-03-15",
                "expiration_date": "2024-06-15",
                "coupon_value": "$0.75",
                "code_assigned": "ABC123",
                "title": "Coupon Registration Form",
                # Optional fields are null
                "media_name": None,
                "issue_frequency": None,
                "signed_by": None,
            },
            field_extractions=[],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=schema_info,
        )

        # Should NOT have schema violations for null optional fields
        schema_violations = [
            i for i in report.issues if i.issue_type == IssueType.SCHEMA_VIOLATION
        ]
        assert len(schema_violations) == 0

        # All required fields are present
        assert report.metrics.required_field_coverage == 1.0

        # Should pass verification
        assert report.status == VerificationStatus.PASSED

    def test_strip_null_optional_nested_fields(self) -> None:
        """Test that nested optional fields with null values are stripped.

        This addresses PR feedback about nested optional fields not being
        handled, e.g., company.employees where employees is optional.
        """
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "company": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "employees": {"type": "integer"},
                            "revenue": {"type": "number"},
                        },
                        "required": ["name"],
                    },
                },
                "required": ["company"],
            },
            required_fields=[
                FieldInfo(
                    name="company",
                    path="company",
                    field_type="object",
                    required=True,
                    nested_fields=[
                        FieldInfo(
                            name="name",
                            path="company.name",
                            field_type="string",
                            required=True,
                        ),
                        FieldInfo(
                            name="employees",
                            path="company.employees",
                            field_type="integer",
                            required=False,
                        ),
                        FieldInfo(
                            name="revenue",
                            path="company.revenue",
                            field_type="number",
                            required=False,
                        ),
                    ],
                )
            ],
            optional_fields=[],
            schema_type="object",
        )

        # Nested optional fields are null
        extraction_result = ExtractionResult(
            extracted_data={
                "company": {
                    "name": "Acme Corp",
                    "employees": None,  # Optional nested field is null
                    "revenue": None,  # Optional nested field is null
                }
            },
            field_extractions=[],
        )

        agent = QualityVerificationAgent(api_key="test-key")

        report = agent.verify_quick(
            extraction_result=extraction_result,
            schema_info=schema_info,
        )

        # Should NOT have schema violations for null nested optional fields
        schema_violations = [
            i for i in report.issues if i.issue_type == IssueType.SCHEMA_VIOLATION
        ]
        assert len(schema_violations) == 0

    def test_strip_null_optional_handles_non_dict_data(self) -> None:
        """Test that non-dict data is returned unchanged.

        This addresses PR feedback about non-dict schema handling where
        calling .items() on non-dict would raise AttributeError.
        """
        schema_info = SchemaInfo(
            schema={"type": "array", "items": {"type": "string"}},
            required_fields=[],
            optional_fields=[],
            schema_type="array",
        )

        # Non-dict data (array)
        data = ["item1", "item2", "item3"]

        agent = QualityVerificationAgent(api_key="test-key")
        result = agent._strip_null_optional_fields(data, schema_info)

        # Should return the data unchanged (same object since no modification needed)
        assert result == data

    def test_strip_null_optional_handles_primitive_data(self) -> None:
        """Test that primitive data types are handled gracefully."""
        schema_info = SchemaInfo(
            schema={"type": "string"},
            required_fields=[],
            optional_fields=[],
            schema_type="string",
        )

        agent = QualityVerificationAgent(api_key="test-key")

        # Test with string
        assert agent._strip_null_optional_fields("test", schema_info) == "test"

        # Test with number
        assert agent._strip_null_optional_fields(42, schema_info) == 42

        # Test with None
        assert agent._strip_null_optional_fields(None, schema_info) is None

    def test_strip_null_optional_array_of_objects(self) -> None:
        """Test that optional fields in array items are handled."""
        schema_info = SchemaInfo(
            schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    }
                },
                "required": ["items"],
            },
            required_fields=[
                FieldInfo(
                    name="items",
                    path="items",
                    field_type="array",
                    required=True,
                    nested_fields=[
                        FieldInfo(
                            name="name",
                            path="items[].name",
                            field_type="string",
                            required=True,
                        ),
                        FieldInfo(
                            name="description",
                            path="items[].description",
                            field_type="string",
                            required=False,
                        ),
                    ],
                )
            ],
            optional_fields=[],
            schema_type="object",
        )

        data = {
            "items": [
                {"name": "Item 1", "description": None},
                {"name": "Item 2", "description": "Has description"},
            ]
        }

        agent = QualityVerificationAgent(api_key="test-key")
        cleaned = agent._strip_null_optional_fields(data, schema_info)

        # Null description should be stripped from first item
        assert "description" not in cleaned["items"][0]
        # Non-null description should remain in second item
        assert cleaned["items"][1]["description"] == "Has description"
