"""Tests for the iterative refinement agent and agentic loop."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from agentic_document_extraction.agents.planner import (
    DocumentCharacteristics,
    ExtractionPlan,
    QualityThreshold,
    SchemaComplexity,
)
from agentic_document_extraction.agents.refiner import (
    AgenticLoop,
    AgenticLoopResult,
    IterationMetrics,
    RefinementAgent,
    RefinementError,
    RefinementFeedback,
)
from agentic_document_extraction.agents.verifier import (
    IssueSeverity,
    IssueType,
    QualityMetrics,
    VerificationIssue,
    VerificationReport,
    VerificationStatus,
)
from agentic_document_extraction.models import (
    FormatFamily,
    FormatInfo,
    ProcessingCategory,
)
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.schema_validator import FieldInfo, SchemaInfo


class ToolCallingFakeModel:
    """Minimal chat model stub that supports tool calling."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._index = 0

    def bind_tools(self, _tools: object, **_kwargs: object) -> "ToolCallingFakeModel":
        return self

    def invoke(self, _messages: object) -> AIMessage:
        response = self._responses[self._index]
        self._index += 1
        return response


# Test fixtures
@pytest.fixture
def simple_schema_info() -> SchemaInfo:
    """Create a simple schema for testing."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's name"},
            "age": {"type": "integer", "description": "Person's age"},
            "email": {"type": "string", "description": "Email address"},
        },
        "required": ["name", "email"],
    }
    return SchemaInfo(
        schema=schema,
        required_fields=[
            FieldInfo(
                name="name",
                field_type="string",
                required=True,
                path="name",
                description="Person's name",
            ),
            FieldInfo(
                name="email",
                field_type="string",
                required=True,
                path="email",
                description="Email address",
            ),
        ],
        optional_fields=[
            FieldInfo(
                name="age",
                field_type="integer",
                required=False,
                path="age",
                description="Person's age",
            ),
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
        optional_fields=[],
        schema_type="object",
    )


@pytest.fixture
def good_extraction_result() -> ExtractionResult:
    """Create a good extraction result for testing."""
    return ExtractionResult(
        extracted_data={"name": "John Doe", "age": 30, "email": "john@example.com"},
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
            FieldExtraction(
                field_path="email",
                value="john@example.com",
                confidence=0.90,
            ),
        ],
        model_used="gpt-4o",
        total_tokens=500,
    )


@pytest.fixture
def poor_extraction_result() -> ExtractionResult:
    """Create a poor extraction result with issues."""
    return ExtractionResult(
        extracted_data={"name": None, "age": 30, "email": "invalid"},
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
            FieldExtraction(
                field_path="email",
                value="invalid",
                confidence=0.3,
            ),
        ],
        model_used="gpt-4o",
        total_tokens=300,
    )


@pytest.fixture
def verification_report_failed() -> VerificationReport:
    """Create a failed verification report."""
    metrics = QualityMetrics(
        overall_confidence=0.35,
        schema_coverage=0.67,
        required_field_coverage=0.5,
        optional_field_coverage=1.0,
        completeness_score=0.55,
        consistency_score=0.5,
        min_field_confidence=0.2,
        fields_with_low_confidence=3,
        total_fields=3,
        extracted_fields=2,
    )

    issues = [
        VerificationIssue(
            issue_type=IssueType.MISSING_REQUIRED_FIELD,
            field_path="name",
            message="Required field 'name' is missing",
            severity=IssueSeverity.CRITICAL,
            expected="A string value",
            suggestion="Search the text more carefully for a name",
        ),
        VerificationIssue(
            issue_type=IssueType.LOW_CONFIDENCE,
            field_path="email",
            message="Field 'email' has low confidence (0.30)",
            severity=IssueSeverity.HIGH,
            current_value="invalid",
            expected="Confidence >= 0.5",
            suggestion="Re-extract this field or verify manually",
        ),
        VerificationIssue(
            issue_type=IssueType.LOW_CONFIDENCE,
            field_path="age",
            message="Field 'age' has low confidence (0.40)",
            severity=IssueSeverity.MEDIUM,
            current_value=30,
            expected="Confidence >= 0.5",
            suggestion="Re-extract this field",
        ),
    ]

    return VerificationReport(
        status=VerificationStatus.FAILED,
        metrics=metrics,
        issues=issues,
        passed_checks=[],
        recommendations=["Focus on finding the person's name in the text"],
        llm_analysis="The extraction is missing critical information.",
    )


@pytest.fixture
def verification_report_needs_improvement() -> VerificationReport:
    """Create a needs improvement verification report."""
    metrics = QualityMetrics(
        overall_confidence=0.55,
        schema_coverage=1.0,
        required_field_coverage=1.0,
        optional_field_coverage=1.0,
        completeness_score=0.85,
        consistency_score=0.7,
        min_field_confidence=0.45,
        fields_with_low_confidence=2,
        total_fields=3,
        extracted_fields=3,
    )

    issues = [
        VerificationIssue(
            issue_type=IssueType.LOW_CONFIDENCE,
            field_path="email",
            message="Field 'email' has low confidence",
            severity=IssueSeverity.MEDIUM,
            current_value="john@example",
            suggestion="Verify email format",
        ),
    ]

    return VerificationReport(
        status=VerificationStatus.NEEDS_IMPROVEMENT,
        metrics=metrics,
        issues=issues,
        passed_checks=["All required fields present"],
        recommendations=["Verify the email address format"],
    )


@pytest.fixture
def verification_report_passed() -> VerificationReport:
    """Create a passed verification report."""
    metrics = QualityMetrics(
        overall_confidence=0.9,
        schema_coverage=1.0,
        required_field_coverage=1.0,
        optional_field_coverage=1.0,
        completeness_score=0.95,
        consistency_score=0.95,
        min_field_confidence=0.85,
        fields_with_low_confidence=0,
        total_fields=3,
        extracted_fields=3,
    )

    return VerificationReport(
        status=VerificationStatus.PASSED,
        metrics=metrics,
        issues=[],
        passed_checks=["All required fields present", "Schema validation passed"],
        recommendations=[],
    )


@pytest.fixture
def default_thresholds() -> QualityThreshold:
    """Create default quality thresholds."""
    return QualityThreshold(
        min_overall_confidence=0.7,
        min_field_confidence=0.5,
        required_field_coverage=0.9,
        max_iterations=3,
    )


@pytest.fixture
def text_format_info() -> FormatInfo:
    """Create format info for text document."""
    return FormatInfo(
        mime_type="text/plain",
        extension=".txt",
        format_family=FormatFamily.PLAIN_TEXT,
        processing_category=ProcessingCategory.TEXT_BASED,
    )


@pytest.fixture
def extraction_plan(default_thresholds: QualityThreshold) -> ExtractionPlan:
    """Create a basic extraction plan."""
    doc_chars = DocumentCharacteristics(
        processing_category="text_based",
        format_family="text",
        estimated_complexity="simple",
    )
    return ExtractionPlan(
        document_characteristics=doc_chars,
        schema_complexity=SchemaComplexity.SIMPLE,
        extraction_strategy="text_direct",
        steps=[],
        region_priorities=[],
        challenges=[],
        quality_thresholds=default_thresholds,
        reasoning="Test plan",
        estimated_confidence=0.8,
    )


class TestRefinementFeedback:
    """Tests for RefinementFeedback dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        issues = [
            VerificationIssue(
                issue_type=IssueType.MISSING_REQUIRED_FIELD,
                field_path="name",
                message="Missing name",
                severity=IssueSeverity.CRITICAL,
            )
        ]

        feedback = RefinementFeedback(
            focus_fields=["name", "email"],
            issues_to_address=issues,
            suggested_strategy="text_comprehensive_search",
            additional_context="Focus on finding names",
        )

        result = feedback.to_dict()

        assert result["focus_fields"] == ["name", "email"]
        assert len(result["issues_to_address"]) == 1
        assert result["suggested_strategy"] == "text_comprehensive_search"
        assert result["additional_context"] == "Focus on finding names"


class TestIterationMetrics:
    """Tests for IterationMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = IterationMetrics(
            iteration_number=2,
            verification_status=VerificationStatus.NEEDS_IMPROVEMENT,
            overall_confidence=0.65,
            required_field_coverage=0.8,
            issues_count=3,
            critical_issues_count=1,
            improvements_from_previous={"confidence_delta": 0.15},
            total_tokens=500,
            processing_time_seconds=2.5,
        )

        result = metrics.to_dict()

        assert result["iteration_number"] == 2
        assert result["verification_status"] == "needs_improvement"
        assert result["overall_confidence"] == 0.65
        assert result["required_field_coverage"] == 0.8
        assert result["issues_count"] == 3
        assert result["critical_issues_count"] == 1
        assert result["improvements"]["confidence_delta"] == 0.15
        assert result["total_tokens"] == 500
        assert result["processing_time_seconds"] == 2.5


class TestAgenticLoopResult:
    """Tests for AgenticLoopResult dataclass."""

    def test_to_dict(
        self,
        good_extraction_result: ExtractionResult,
        verification_report_passed: VerificationReport,
        extraction_plan: ExtractionPlan,
    ) -> None:
        """Test conversion to dictionary."""
        iteration_metrics = [
            IterationMetrics(
                iteration_number=1,
                verification_status=VerificationStatus.PASSED,
                overall_confidence=0.9,
                required_field_coverage=1.0,
                issues_count=0,
                critical_issues_count=0,
            )
        ]

        result = AgenticLoopResult(
            final_result=good_extraction_result,
            final_verification=verification_report_passed,
            plan=extraction_plan,
            iterations_completed=1,
            iteration_history=iteration_metrics,
            converged=True,
            best_iteration=1,
            total_tokens=1000,
            total_processing_time_seconds=5.0,
        )

        dict_result = result.to_dict()

        assert dict_result["converged"] is True
        assert dict_result["best_iteration"] == 1
        assert dict_result["iterations_completed"] == 1
        assert dict_result["total_tokens"] == 1000
        assert len(dict_result["iteration_history"]) == 1


class TestRefinementAgent:
    """Tests for RefinementAgent."""

    def test_generate_feedback_from_failed_verification(
        self,
        verification_report_failed: VerificationReport,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test feedback generation from failed verification."""
        agent = RefinementAgent(api_key="test-key")

        feedback = agent.generate_feedback(
            verification_report=verification_report_failed,
            schema_info=simple_schema_info,
            processing_category=ProcessingCategory.TEXT_BASED,
        )

        # Should prioritize critical issues
        assert "name" in feedback.focus_fields
        assert len(feedback.issues_to_address) == 3

        # First issue should be critical
        assert feedback.issues_to_address[0].severity == IssueSeverity.CRITICAL

        # Should determine appropriate strategy
        assert feedback.suggested_strategy == "text_comprehensive_search"

    def test_generate_feedback_for_visual_document(
        self,
        verification_report_failed: VerificationReport,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test feedback generation for visual documents."""
        agent = RefinementAgent(api_key="test-key")

        feedback = agent.generate_feedback(
            verification_report=verification_report_failed,
            schema_info=simple_schema_info,
            processing_category=ProcessingCategory.VISUAL,
        )

        # Should use visual strategy
        assert "visual" in feedback.suggested_strategy

    def test_generate_feedback_for_low_confidence(
        self,
        verification_report_needs_improvement: VerificationReport,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test feedback generation for low confidence issues."""
        agent = RefinementAgent(api_key="test-key")

        feedback = agent.generate_feedback(
            verification_report=verification_report_needs_improvement,
            schema_info=simple_schema_info,
            processing_category=ProcessingCategory.TEXT_BASED,
        )

        # Should focus on confidence boost
        assert (
            "confidence" in feedback.suggested_strategy
            or "targeted" in feedback.suggested_strategy
        )

    def test_refine_extraction(
        self,
        poor_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test refinement of extraction."""
        refined_data = {
            "name": "John Smith",
            "age": 30,
            "email": "john.smith@example.com",
        }

        mock_response = AIMessage(
            content=json.dumps(refined_data),
            usage_metadata={
                "input_tokens": 800,
                "output_tokens": 100,
                "total_tokens": 900,
            },
        )

        agent = RefinementAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        feedback = RefinementFeedback(
            focus_fields=["name", "email"],
            issues_to_address=[
                VerificationIssue(
                    issue_type=IssueType.MISSING_REQUIRED_FIELD,
                    field_path="name",
                    message="Missing name",
                    severity=IssueSeverity.CRITICAL,
                )
            ],
            suggested_strategy="text_comprehensive_search",
            additional_context="Look for person names",
        )

        result = agent.refine(
            text="John Smith is 30 years old. Email: john.smith@example.com",
            previous_result=poor_extraction_result,
            feedback=feedback,
            schema_info=simple_schema_info,
        )

        assert result.extracted_data["name"] == "John Smith"
        assert result.extracted_data["email"] == "john.smith@example.com"
        assert result.total_tokens == 900

    def test_refine_preserves_unchanged_fields(
        self,
        good_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
    ) -> None:
        """Test that refinement preserves unchanged fields."""
        refined_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@newdomain.com",  # Only email changed
        }

        mock_response = AIMessage(
            content=json.dumps(refined_data),
            usage_metadata={
                "input_tokens": 500,
                "output_tokens": 50,
                "total_tokens": 550,
            },
        )

        agent = RefinementAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        feedback = RefinementFeedback(
            focus_fields=["email"],
            issues_to_address=[],
            suggested_strategy="text_targeted_refinement",
            additional_context="",
        )

        result = agent.refine(
            text="Test text",
            previous_result=good_extraction_result,
            feedback=feedback,
            schema_info=simple_schema_info,
        )

        # Name unchanged - should preserve confidence
        name_extraction = next(
            fe for fe in result.field_extractions if fe.field_path == "name"
        )
        assert name_extraction.confidence == 0.95  # Preserved from previous

        # Email changed - should have moderate confidence
        email_extraction = next(
            fe for fe in result.field_extractions if fe.field_path == "email"
        )
        assert email_extraction.confidence == 0.70  # Changed value

    def test_reset_memory(self) -> None:
        """Test resetting conversation memory."""
        agent = RefinementAgent(api_key="test-key")
        agent._message_history.add_message(MagicMock())
        agent._message_history.add_message(MagicMock())

        agent.reset_memory()

        assert len(agent._message_history.messages) == 0

    def test_no_api_key_error(self) -> None:
        """Test error when API key is missing."""
        agent = RefinementAgent(api_key="")

        with pytest.raises(RefinementError) as exc_info:
            _ = agent.llm

        assert exc_info.value.error_type == "configuration_error"

    def test_parse_json_with_wrapper(self) -> None:
        """Test parsing JSON with extra text."""
        agent = RefinementAgent(api_key="test-key")

        response = 'Here is the result: {"name": "John"} end.'

        # Create minimal schema_info
        schema_info = MagicMock()

        result = agent._parse_json_response(response, schema_info)
        assert result["name"] == "John"

    def test_parse_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON raises error."""
        agent = RefinementAgent(api_key="test-key")
        schema_info = MagicMock()

        with pytest.raises(RefinementError) as exc_info:
            agent._parse_json_response("not json at all", schema_info)

        assert exc_info.value.error_type == "parse_error"


class TestRefinementStrategies:
    """Tests for refinement strategy determination."""

    def test_strategy_for_missing_required(self) -> None:
        """Test strategy determination for missing required fields."""
        agent = RefinementAgent(api_key="test-key")

        issues = [
            VerificationIssue(
                issue_type=IssueType.MISSING_REQUIRED_FIELD,
                field_path="name",
                message="Missing",
                severity=IssueSeverity.CRITICAL,
            )
        ]

        strategy = agent._determine_refinement_strategy(
            issues, ProcessingCategory.TEXT_BASED
        )
        assert strategy == "text_comprehensive_search"

    def test_strategy_for_schema_violations(self) -> None:
        """Test strategy determination for schema violations."""
        agent = RefinementAgent(api_key="test-key")

        issues = [
            VerificationIssue(
                issue_type=IssueType.SCHEMA_VIOLATION,
                field_path="age",
                message="Type mismatch",
                severity=IssueSeverity.HIGH,
            )
        ]

        strategy = agent._determine_refinement_strategy(
            issues, ProcessingCategory.TEXT_BASED
        )
        assert strategy == "text_type_correction"

    def test_strategy_for_low_confidence(self) -> None:
        """Test strategy determination for low confidence."""
        agent = RefinementAgent(api_key="test-key")

        issues = [
            VerificationIssue(
                issue_type=IssueType.LOW_CONFIDENCE,
                field_path="email",
                message="Low confidence",
                severity=IssueSeverity.MEDIUM,
            )
        ]

        strategy = agent._determine_refinement_strategy(
            issues, ProcessingCategory.TEXT_BASED
        )
        assert strategy == "text_confidence_boost"

    def test_strategy_for_visual_missing_required(self) -> None:
        """Test strategy for visual document with missing fields."""
        agent = RefinementAgent(api_key="test-key")

        issues = [
            VerificationIssue(
                issue_type=IssueType.MISSING_REQUIRED_FIELD,
                field_path="name",
                message="Missing",
                severity=IssueSeverity.CRITICAL,
            )
        ]

        strategy = agent._determine_refinement_strategy(
            issues, ProcessingCategory.VISUAL
        )
        assert strategy == "visual_region_reprocessing"


class TestAgenticLoop:
    """Tests for AgenticLoop orchestrator."""

    def test_loop_converges_on_first_iteration(
        self,
        good_extraction_result: ExtractionResult,
        verification_report_passed: VerificationReport,
        simple_schema_info: SchemaInfo,
        text_format_info: FormatInfo,
        extraction_plan: ExtractionPlan,
    ) -> None:
        """Test loop that converges on first iteration."""
        loop = AgenticLoop(api_key="test-key")

        # Mock verifier to return passed
        loop._verifier = MagicMock()
        loop._verifier.verify.return_value = verification_report_passed

        # Mock planner
        loop._planner = MagicMock()
        loop._planner.create_plan.return_value = extraction_plan

        result = loop.run(
            text="John Doe, 30 years old, john@example.com",
            schema_info=simple_schema_info,
            format_info=text_format_info,
            initial_result=good_extraction_result,
            plan=extraction_plan,
            use_agent_orchestration=False,
        )

        assert result.converged is True
        assert result.iterations_completed == 1
        assert result.best_iteration == 1
        assert result.final_verification.status == VerificationStatus.PASSED

    def test_loop_improves_over_iterations(
        self,
        poor_extraction_result: ExtractionResult,
        verification_report_failed: VerificationReport,
        verification_report_needs_improvement: VerificationReport,
        verification_report_passed: VerificationReport,
        simple_schema_info: SchemaInfo,
        text_format_info: FormatInfo,
        extraction_plan: ExtractionPlan,
    ) -> None:
        """Test loop that improves over multiple iterations."""
        loop = AgenticLoop(api_key="test-key")

        # Mock verifier to return progressively better results
        loop._verifier = MagicMock()
        loop._verifier.verify.side_effect = [
            verification_report_failed,
            verification_report_needs_improvement,
            verification_report_passed,
        ]

        # Mock refiner
        refined_result = ExtractionResult(
            extracted_data={"name": "John", "age": 30, "email": "john@example.com"},
            field_extractions=[],
            total_tokens=100,
        )

        loop._refiner = MagicMock()
        loop._refiner.generate_feedback.return_value = RefinementFeedback(
            focus_fields=["name"],
            issues_to_address=[],
            suggested_strategy="text_comprehensive_search",
            additional_context="",
        )
        loop._refiner.refine.return_value = refined_result
        loop._refiner.reset_memory = MagicMock()

        result = loop.run(
            text="John Doe, 30, john@example.com",
            schema_info=simple_schema_info,
            format_info=text_format_info,
            initial_result=poor_extraction_result,
            plan=extraction_plan,
            use_agent_orchestration=False,
        )

        assert result.converged is True
        assert result.iterations_completed == 3
        assert len(result.iteration_history) == 3

        # Should track improvements
        assert result.iteration_history[1].improvements_from_previous is not None

    def test_loop_reaches_max_iterations(
        self,
        poor_extraction_result: ExtractionResult,
        verification_report_failed: VerificationReport,
        simple_schema_info: SchemaInfo,
        text_format_info: FormatInfo,
        extraction_plan: ExtractionPlan,
    ) -> None:
        """Test loop that reaches max iterations without converging."""
        loop = AgenticLoop(api_key="test-key", default_max_iterations=2)

        # Mock verifier to always return failed
        loop._verifier = MagicMock()
        loop._verifier.verify.return_value = verification_report_failed

        # Mock refiner
        loop._refiner = MagicMock()
        loop._refiner.generate_feedback.return_value = RefinementFeedback(
            focus_fields=["name"],
            issues_to_address=[],
            suggested_strategy="text_comprehensive_search",
            additional_context="",
        )
        loop._refiner.refine.return_value = poor_extraction_result
        loop._refiner.reset_memory = MagicMock()

        # Override max iterations in plan
        extraction_plan.quality_thresholds.max_iterations = 2

        result = loop.run(
            text="Some text",
            schema_info=simple_schema_info,
            format_info=text_format_info,
            initial_result=poor_extraction_result,
            plan=extraction_plan,
            use_agent_orchestration=False,
        )

        assert result.converged is False
        assert result.iterations_completed == 2
        assert result.loop_metadata["max_iterations_allowed"] == 2

    def test_loop_tracks_best_result(
        self,
        poor_extraction_result: ExtractionResult,
        simple_schema_info: SchemaInfo,
        text_format_info: FormatInfo,
        extraction_plan: ExtractionPlan,
    ) -> None:
        """Test that loop tracks and returns the best result."""
        loop = AgenticLoop(api_key="test-key")

        # Create verification reports with different scores
        good_metrics = QualityMetrics(
            overall_confidence=0.8,
            schema_coverage=1.0,
            required_field_coverage=1.0,
            optional_field_coverage=1.0,
            completeness_score=0.9,
            consistency_score=0.9,
            min_field_confidence=0.7,
            fields_with_low_confidence=0,
            total_fields=3,
            extracted_fields=3,
        )
        good_verification = VerificationReport(
            status=VerificationStatus.NEEDS_IMPROVEMENT,
            metrics=good_metrics,
            issues=[],
        )

        worse_metrics = QualityMetrics(
            overall_confidence=0.5,
            schema_coverage=0.67,
            required_field_coverage=0.5,
            optional_field_coverage=1.0,
            completeness_score=0.6,
            consistency_score=0.7,
            min_field_confidence=0.3,
            fields_with_low_confidence=2,
            total_fields=3,
            extracted_fields=2,
        )
        worse_verification = VerificationReport(
            status=VerificationStatus.FAILED,
            metrics=worse_metrics,
            issues=[],
        )

        # Second iteration is best, third is worse
        loop._verifier = MagicMock()
        loop._verifier.verify.side_effect = [
            worse_verification,
            good_verification,
            worse_verification,
        ]

        better_result = ExtractionResult(
            extracted_data={"name": "John", "age": 30, "email": "john@test.com"},
            field_extractions=[],
            total_tokens=100,
        )

        loop._refiner = MagicMock()
        loop._refiner.generate_feedback.return_value = RefinementFeedback(
            focus_fields=[],
            issues_to_address=[],
            suggested_strategy="",
            additional_context="",
        )
        loop._refiner.refine.side_effect = [better_result, poor_extraction_result]
        loop._refiner.reset_memory = MagicMock()

        result = loop.run(
            text="Test",
            schema_info=simple_schema_info,
            format_info=text_format_info,
            initial_result=poor_extraction_result,
            plan=extraction_plan,
            use_agent_orchestration=False,
        )

        # Should return iteration 2's result as best
        assert result.best_iteration == 2
        assert result.final_result == better_result

    def test_loop_no_initial_result_no_func_raises_error(
        self,
        simple_schema_info: SchemaInfo,
        text_format_info: FormatInfo,
        extraction_plan: ExtractionPlan,
    ) -> None:
        """Test error when no initial result and no extraction function."""
        loop = AgenticLoop(api_key="test-key")

        with pytest.raises(RefinementError) as exc_info:
            loop.run(
                text="Test",
                schema_info=simple_schema_info,
                format_info=text_format_info,
                initial_result=None,
                extraction_func=None,
                plan=extraction_plan,
                use_agent_orchestration=False,
            )

        assert exc_info.value.error_type == "configuration_error"

    def test_loop_uses_extraction_function(
        self,
        good_extraction_result: ExtractionResult,
        verification_report_passed: VerificationReport,
        simple_schema_info: SchemaInfo,
        text_format_info: FormatInfo,
        extraction_plan: ExtractionPlan,
    ) -> None:
        """Test loop calls extraction function when no initial result."""
        loop = AgenticLoop(api_key="test-key")

        loop._verifier = MagicMock()
        loop._verifier.verify.return_value = verification_report_passed

        extraction_func = MagicMock(return_value=good_extraction_result)

        result = loop.run(
            text="Test text",
            schema_info=simple_schema_info,
            format_info=text_format_info,
            initial_result=None,
            extraction_func=extraction_func,
            plan=extraction_plan,
            use_agent_orchestration=False,
        )

        extraction_func.assert_called_once_with("Test text", simple_schema_info)
        assert result.converged is True

    def test_loop_uses_agents_with_tools(
        self,
        good_extraction_result: ExtractionResult,
        verification_report_passed: VerificationReport,
        simple_schema_info: SchemaInfo,
        text_format_info: FormatInfo,
        extraction_plan: ExtractionPlan,
    ) -> None:
        """Test loop orchestration through agent tool calls."""
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"id": "plan", "name": "plan_action", "args": {}}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"id": "execute", "name": "execute_action", "args": {}}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"id": "verify", "name": "verify_action", "args": {}}],
            ),
        ]
        loop = AgenticLoop(
            api_key="test-key",
            loop_model=ToolCallingFakeModel(responses),
        )

        loop._planner = MagicMock()
        loop._planner.create_plan.return_value = extraction_plan
        loop._verifier = MagicMock()
        loop._verifier.verify.return_value = verification_report_passed

        extraction_func = MagicMock(return_value=good_extraction_result)

        result = loop.run(
            text="Test text",
            schema_info=simple_schema_info,
            format_info=text_format_info,
            initial_result=None,
            extraction_func=extraction_func,
            plan=None,
            use_llm_verification=False,
            use_agent_orchestration=True,
        )

        assert result.converged is True
        assert loop._loop_agent is not None
        assert len(loop._loop_history.messages) > 0
        loop._planner.create_plan.assert_called_once()
        loop._verifier.verify.assert_called_once()
        extraction_func.assert_called_once_with("Test text", simple_schema_info)

    def test_calculate_result_score(
        self,
        verification_report_passed: VerificationReport,
        verification_report_failed: VerificationReport,
    ) -> None:
        """Test result score calculation."""
        loop = AgenticLoop(api_key="test-key")

        passed_score = loop._calculate_result_score(verification_report_passed)
        failed_score = loop._calculate_result_score(verification_report_failed)

        # Passed should have higher score
        assert passed_score > failed_score

        # Passed gets bonus
        assert passed_score > 0.9

    def test_lazy_agent_initialization(self) -> None:
        """Test that agents are initialized lazily."""
        loop = AgenticLoop(api_key="test-key")

        assert loop._planner is None
        assert loop._verifier is None
        assert loop._refiner is None

        # Access properties
        _ = loop.planner
        _ = loop.verifier
        _ = loop.refiner

        assert loop._planner is not None
        assert loop._verifier is not None
        assert loop._refiner is not None


class TestNestedValueAccess:
    """Tests for nested value access utility."""

    def test_get_simple_value(self) -> None:
        """Test getting simple value."""
        agent = RefinementAgent(api_key="test-key")

        data = {"name": "John", "age": 30}
        assert agent._get_nested_value(data, "name") == "John"

    def test_get_nested_value(self) -> None:
        """Test getting nested value."""
        agent = RefinementAgent(api_key="test-key")

        data = {"company": {"address": {"city": "NYC"}}}
        assert agent._get_nested_value(data, "company.address.city") == "NYC"

    def test_get_missing_value(self) -> None:
        """Test getting missing value returns None."""
        agent = RefinementAgent(api_key="test-key")

        data = {"name": "John"}
        assert agent._get_nested_value(data, "missing") is None
        assert agent._get_nested_value(data, "a.b.c") is None


class TestRefinementError:
    """Tests for RefinementError."""

    def test_error_attributes(self) -> None:
        """Test error attributes."""
        error = RefinementError(
            "Test message",
            error_type="test_type",
            details={"key": "value"},
        )

        assert str(error) == "Test message"
        assert error.error_type == "test_type"
        assert error.details == {"key": "value"}

    def test_error_default_values(self) -> None:
        """Test error default values."""
        error = RefinementError("Simple error")

        assert error.error_type == "refinement_error"
        assert error.details == {}


class TestComplexSchemaRefinement:
    """Tests for refinement with complex schemas."""

    def test_refine_nested_fields(
        self,
        complex_schema_info: SchemaInfo,
    ) -> None:
        """Test refinement handles nested fields."""
        previous_result = ExtractionResult(
            extracted_data={
                "company": {"name": None, "employees": 100},
                "products": [],
            },
            field_extractions=[
                FieldExtraction(
                    field_path="company", value={"name": None}, confidence=0.5
                ),
                FieldExtraction(field_path="company.name", value=None, confidence=0.2),
                FieldExtraction(field_path="products", value=[], confidence=0.4),
            ],
        )

        refined_data = {
            "company": {"name": "Acme Corp", "employees": 100},
            "products": ["Widget", "Gadget"],
        }

        mock_response = AIMessage(
            content=json.dumps(refined_data),
            usage_metadata={
                "input_tokens": 300,
                "output_tokens": 50,
                "total_tokens": 350,
            },
        )

        agent = RefinementAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        feedback = RefinementFeedback(
            focus_fields=["company.name", "products"],
            issues_to_address=[],
            suggested_strategy="text_comprehensive_search",
            additional_context="",
        )

        result = agent.refine(
            text="Acme Corp with 100 employees makes Widget and Gadget",
            previous_result=previous_result,
            feedback=feedback,
            schema_info=complex_schema_info,
        )

        assert result.extracted_data["company"]["name"] == "Acme Corp"
        assert len(result.extracted_data["products"]) == 2

        # Check nested field extractions
        company_name_fe = next(
            (fe for fe in result.field_extractions if fe.field_path == "company.name"),
            None,
        )
        assert company_name_fe is not None
        assert company_name_fe.value == "Acme Corp"
