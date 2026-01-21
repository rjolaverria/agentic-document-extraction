"""Tests for the quality verification agent."""

import json
from unittest.mock import MagicMock

import pytest

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
        """Test metrics computation when no confidence scores available."""
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

        # Should use defaults when no confidence available
        assert report.metrics.overall_confidence == 0.5
        assert report.metrics.fields_with_low_confidence == 0


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

        mock_response = MagicMock()
        mock_response.content = json.dumps(llm_response)
        mock_response.usage_metadata = {
            "input_tokens": 500,
            "output_tokens": 200,
        }

        agent = QualityVerificationAgent(api_key="test-key")
        agent._llm = MagicMock()
        agent._llm.invoke.return_value = mock_response

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

        mock_response = MagicMock()
        mock_response.content = json.dumps(llm_response)
        mock_response.usage_metadata = {"input_tokens": 400, "output_tokens": 150}

        agent = QualityVerificationAgent(api_key="test-key")
        agent._llm = MagicMock()
        agent._llm.invoke.return_value = mock_response

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
        agent._llm = MagicMock()
        agent._llm.invoke.side_effect = Exception("LLM API error")

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

        mock_response = MagicMock()
        mock_response.content = json.dumps(llm_response)
        mock_response.usage_metadata = {"input_tokens": 300, "output_tokens": 100}

        agent = QualityVerificationAgent(api_key="test-key")
        agent._llm = MagicMock()
        agent._llm.invoke.return_value = mock_response

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
