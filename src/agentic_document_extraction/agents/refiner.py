"""Iterative refinement agent for document extraction.

This module provides agents and orchestration for iteratively refining
document extractions that don't meet quality thresholds. It implements
the agentic loop: Plan → Execute → Verify → Refine.

Key features:
- Re-attempts extraction with refined prompts based on verification feedback
- For text documents: focuses on specific fields that were missed or incorrect
- For visual documents: re-processes specific regions or uses different strategies
- Uses LangChain memory to maintain context across iterations
- Limits iteration count to prevent infinite loops
- Tracks improvement metrics across iterations
- Returns best result even if threshold not met, with quality report
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agentic_document_extraction.agents.planner import (
    ExtractionPlan,
    ExtractionPlanningAgent,
)
from agentic_document_extraction.agents.verifier import (
    IssueSeverity,
    IssueType,
    QualityVerificationAgent,
    VerificationIssue,
    VerificationReport,
    VerificationStatus,
)
from agentic_document_extraction.config import settings
from agentic_document_extraction.models import FormatInfo, ProcessingCategory
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.schema_validator import SchemaInfo

logger = logging.getLogger(__name__)


class RefinementError(Exception):
    """Raised when refinement fails."""

    def __init__(
        self,
        message: str,
        error_type: str = "refinement_error",
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


@dataclass
class RefinementFeedback:
    """Feedback for improving extraction based on verification."""

    focus_fields: list[str]
    """Fields that need specific attention in re-extraction."""

    issues_to_address: list[VerificationIssue]
    """Specific issues from verification to address."""

    suggested_strategy: str
    """Suggested approach for refinement."""

    additional_context: str
    """Additional context or hints for the LLM."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with refinement feedback information.
        """
        return {
            "focus_fields": self.focus_fields,
            "issues_to_address": [i.to_dict() for i in self.issues_to_address],
            "suggested_strategy": self.suggested_strategy,
            "additional_context": self.additional_context,
        }


@dataclass
class IterationMetrics:
    """Metrics for a single iteration of the refinement loop."""

    iteration_number: int
    """Which iteration this represents (1-indexed)."""

    verification_status: VerificationStatus
    """Verification status after this iteration."""

    overall_confidence: float
    """Overall confidence score."""

    required_field_coverage: float
    """Required field coverage percentage."""

    issues_count: int
    """Number of issues identified."""

    critical_issues_count: int
    """Number of critical issues."""

    improvements_from_previous: dict[str, float] = field(default_factory=dict)
    """Metric improvements compared to previous iteration."""

    total_tokens: int = 0
    """Tokens used in this iteration."""

    processing_time_seconds: float = 0.0
    """Time taken for this iteration."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with iteration metrics.
        """
        return {
            "iteration_number": self.iteration_number,
            "verification_status": self.verification_status.value,
            "overall_confidence": self.overall_confidence,
            "required_field_coverage": self.required_field_coverage,
            "issues_count": self.issues_count,
            "critical_issues_count": self.critical_issues_count,
            "improvements": self.improvements_from_previous,
            "total_tokens": self.total_tokens,
            "processing_time_seconds": self.processing_time_seconds,
        }


@dataclass
class AgenticLoopResult:
    """Complete result from the agentic extraction loop."""

    final_result: ExtractionResult
    """The best extraction result achieved."""

    final_verification: VerificationReport
    """Verification report for the final result."""

    plan: ExtractionPlan
    """The extraction plan used."""

    iterations_completed: int
    """Number of iterations completed."""

    iteration_history: list[IterationMetrics]
    """Metrics from each iteration."""

    converged: bool
    """Whether the loop converged (met quality thresholds)."""

    best_iteration: int
    """Which iteration produced the best result."""

    total_tokens: int = 0
    """Total tokens used across all iterations."""

    total_processing_time_seconds: float = 0.0
    """Total time taken for the entire loop."""

    loop_metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the loop execution."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with complete loop result information.
        """
        return {
            "final_result": self.final_result.to_dict(),
            "final_verification": self.final_verification.to_dict(),
            "plan": self.plan.to_dict(),
            "iterations_completed": self.iterations_completed,
            "iteration_history": [m.to_dict() for m in self.iteration_history],
            "converged": self.converged,
            "best_iteration": self.best_iteration,
            "total_tokens": self.total_tokens,
            "total_processing_time_seconds": self.total_processing_time_seconds,
            "loop_metadata": self.loop_metadata,
        }


class RefinementAgent:
    """Agent for refining extractions based on verification feedback.

    Uses LLM to generate refined extractions focusing on specific issues
    identified during verification. Maintains context across iterations
    to avoid repeating the same mistakes.
    """

    REFINEMENT_SYSTEM_PROMPT = """You are an expert information extraction assistant.
Your task is to refine a previous extraction based on specific feedback about issues found.

IMPORTANT RULES:
1. Focus specifically on the fields and issues mentioned in the feedback
2. Use the original text carefully - do NOT fabricate information
3. If a field cannot be found in the text, use null
4. Maintain the exact structure of the requested JSON schema
5. Preserve correct extractions from the previous attempt
6. Only modify fields that had issues
7. Provide higher confidence scores only when you are more certain

REFINEMENT GUIDELINES:
- For MISSING_REQUIRED_FIELD: Search more carefully for this information
- For LOW_CONFIDENCE: Re-analyze the source text for this field
- For SCHEMA_VIOLATION: Ensure the value matches the expected type
- For LOGICAL_INCONSISTENCY: Check that related fields are consistent
- For INCOMPLETE_VALUE: Provide the complete value from the source

You must respond with ONLY valid JSON matching the schema."""

    REFINEMENT_USER_PROMPT = """Refine the following extraction based on the issues identified.

## Previous Extraction:
```json
{previous_extraction}
```

## Target JSON Schema:
```json
{schema}
```

## Issues to Address:
{issues}

## Fields Requiring Attention:
{focus_fields}

## Additional Guidance:
{guidance}

## Source Text:
```
{text}
```

Provide an improved extraction addressing the issues above. Respond with ONLY the corrected JSON data."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the refinement agent.

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
        self._conversation_history: list[BaseMessage] = []

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LangChain ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            RefinementError: If API key is not configured.
        """
        if self._llm is None:
            if not self.api_key:
                raise RefinementError(
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

    def reset_memory(self) -> None:
        """Reset the conversation history for a new extraction task."""
        self._conversation_history = []

    def generate_feedback(
        self,
        verification_report: VerificationReport,
        schema_info: SchemaInfo,
        processing_category: ProcessingCategory = ProcessingCategory.TEXT_BASED,
    ) -> RefinementFeedback:
        """Generate refinement feedback from a verification report.

        Args:
            verification_report: The verification report with issues.
            schema_info: Schema information.
            processing_category: Whether text or visual processing was used.

        Returns:
            RefinementFeedback with guidance for the next iteration.
        """
        # Identify fields that need attention (prioritize by severity)
        focus_fields: list[str] = []
        issues_to_address: list[VerificationIssue] = []

        # Sort issues by severity
        sorted_issues = sorted(
            verification_report.issues,
            key=lambda i: (
                0
                if i.severity == IssueSeverity.CRITICAL
                else 1
                if i.severity == IssueSeverity.HIGH
                else 2
                if i.severity == IssueSeverity.MEDIUM
                else 3
            ),
        )

        for issue in sorted_issues:
            if issue.field_path not in focus_fields:
                focus_fields.append(issue.field_path)
            issues_to_address.append(issue)

        # Determine strategy based on issues and processing category
        strategy = self._determine_refinement_strategy(
            issues_to_address, processing_category
        )

        # Generate additional context
        additional_context = self._generate_additional_context(
            verification_report, schema_info
        )

        return RefinementFeedback(
            focus_fields=focus_fields,
            issues_to_address=issues_to_address,
            suggested_strategy=strategy,
            additional_context=additional_context,
        )

    def _determine_refinement_strategy(
        self,
        issues: list[VerificationIssue],
        processing_category: ProcessingCategory,
    ) -> str:
        """Determine the refinement strategy based on issues.

        Args:
            issues: List of issues to address.
            processing_category: Processing category.

        Returns:
            Strategy name for refinement.
        """
        # Count issue types
        missing_required = sum(
            1 for i in issues if i.issue_type == IssueType.MISSING_REQUIRED_FIELD
        )
        low_confidence = sum(
            1 for i in issues if i.issue_type == IssueType.LOW_CONFIDENCE
        )
        schema_violations = sum(
            1 for i in issues if i.issue_type == IssueType.SCHEMA_VIOLATION
        )
        logical_issues = sum(
            1 for i in issues if i.issue_type == IssueType.LOGICAL_INCONSISTENCY
        )

        if processing_category == ProcessingCategory.VISUAL:
            if missing_required > 0:
                return "visual_region_reprocessing"
            elif logical_issues > 0:
                return "visual_context_enhanced"
            else:
                return "visual_targeted_refinement"
        else:
            if missing_required > 0:
                return "text_comprehensive_search"
            elif schema_violations > 0:
                return "text_type_correction"
            elif low_confidence > 0:
                return "text_confidence_boost"
            elif logical_issues > 0:
                return "text_consistency_check"
            else:
                return "text_targeted_refinement"

    def _generate_additional_context(
        self,
        verification_report: VerificationReport,
        schema_info: SchemaInfo,
    ) -> str:
        """Generate additional context for the refinement prompt.

        Args:
            verification_report: Verification report.
            schema_info: Schema information.

        Returns:
            Additional context string.
        """
        parts: list[str] = []

        # Add recommendations from verification
        if verification_report.recommendations:
            parts.append("Recommendations from verification:")
            for rec in verification_report.recommendations:
                parts.append(f"- {rec}")

        # Add field descriptions for focus fields
        focus_paths = {i.field_path for i in verification_report.issues}
        relevant_fields = [f for f in schema_info.all_fields if f.path in focus_paths]

        if relevant_fields:
            parts.append("\nField descriptions:")
            for field in relevant_fields:
                desc = field.description or "No description"
                parts.append(f"- {field.path} ({field.field_type}): {desc}")

        # Add LLM analysis if available
        if verification_report.llm_analysis:
            parts.append(f"\nPrevious analysis: {verification_report.llm_analysis}")

        return "\n".join(parts) if parts else "Focus on accuracy and completeness."

    def refine(
        self,
        text: str,
        previous_result: ExtractionResult,
        feedback: RefinementFeedback,
        schema_info: SchemaInfo,
    ) -> ExtractionResult:
        """Refine an extraction based on feedback.

        Args:
            text: Original source text.
            previous_result: Previous extraction result to improve.
            feedback: Feedback guiding the refinement.
            schema_info: Schema information.

        Returns:
            Refined ExtractionResult.

        Raises:
            RefinementError: If refinement fails.
        """
        start_time = time.time()

        # Format issues for the prompt
        issues_str = self._format_issues_for_prompt(feedback.issues_to_address)
        focus_fields_str = ", ".join(feedback.focus_fields) or "all fields"

        # Build prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.REFINEMENT_SYSTEM_PROMPT),
                ("human", self.REFINEMENT_USER_PROMPT),
            ]
        )

        formatted = prompt.partial(
            previous_extraction=json.dumps(previous_result.extracted_data, indent=2),
            schema=json.dumps(schema_info.schema, indent=2),
            issues=issues_str,
            focus_fields=focus_fields_str,
            guidance=feedback.additional_context,
            text=text[:8000],  # Truncate to avoid token limits
        )

        try:
            messages = formatted.format_messages()

            # Add to conversation history for context
            self._conversation_history.extend(messages)

            response = self.llm.invoke(messages)

            # Parse response
            content = response.content
            if not isinstance(content, str):
                content = str(content)

            refined_data = self._parse_json_response(content, schema_info)

            # Extract token usage
            usage_metadata = getattr(response, "usage_metadata", None) or {}
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            # Build field extractions with confidence
            field_extractions = self._build_field_extractions(
                refined_data, schema_info, previous_result.field_extractions
            )

            processing_time = time.time() - start_time

            logger.info(
                f"Refinement completed: "
                f"fields_focused={len(feedback.focus_fields)}, "
                f"strategy={feedback.suggested_strategy}, "
                f"tokens={total_tokens}, "
                f"time={processing_time:.2f}s"
            )

            return ExtractionResult(
                extracted_data=refined_data,
                field_extractions=field_extractions,
                model_used=self.model,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                processing_time_seconds=processing_time,
                chunks_processed=previous_result.chunks_processed,
                is_chunked=previous_result.is_chunked,
                raw_response=content,
            )

        except Exception as e:
            if isinstance(e, RefinementError):
                raise
            raise RefinementError(
                f"Refinement failed: {e}",
                error_type="llm_error",
                details={"original_error": str(e)},
            ) from e

    def _format_issues_for_prompt(
        self,
        issues: list[VerificationIssue],
    ) -> str:
        """Format issues for the refinement prompt.

        Args:
            issues: List of issues to format.

        Returns:
            Formatted string of issues.
        """
        if not issues:
            return "No specific issues identified - general improvement needed."

        parts: list[str] = []
        for issue in issues:
            severity = issue.severity.value.upper()
            suggestion = issue.suggestion or "Please address this issue"
            parts.append(
                f"[{severity}] {issue.field_path}: {issue.message}\n"
                f"   Suggestion: {suggestion}"
            )

        return "\n\n".join(parts)

    def _parse_json_response(
        self,
        response: str,
        schema_info: SchemaInfo,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Parse JSON response from the model.

        Args:
            response: Raw response string.
            schema_info: Schema information for validation.

        Returns:
            Parsed JSON data.

        Raises:
            RefinementError: If parsing fails.
        """
        try:
            data = json.loads(response)

            if not isinstance(data, dict):
                raise RefinementError(
                    "Expected JSON object response",
                    error_type="parse_error",
                    details={"received_type": type(data).__name__},
                )

            return data

        except json.JSONDecodeError:
            # Try to extract JSON from the response
            try:
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        return data
            except json.JSONDecodeError:
                pass

            raise RefinementError(
                "Failed to parse refinement response as JSON",
                error_type="parse_error",
                details={"response_preview": response[:500]},
            ) from None

    def _build_field_extractions(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
        previous_extractions: list[FieldExtraction],
    ) -> list[FieldExtraction]:
        """Build field extractions with estimated confidence.

        Args:
            data: Refined extracted data.
            schema_info: Schema information.
            previous_extractions: Previous field extractions for reference.

        Returns:
            List of FieldExtraction objects.
        """
        # Build lookup for previous extractions
        prev_lookup = {fe.field_path: fe for fe in previous_extractions}

        extractions: list[FieldExtraction] = []

        for field_info in schema_info.all_fields:
            field_path = field_info.path
            value = self._get_nested_value(data, field_path)
            prev = prev_lookup.get(field_path)

            # Estimate confidence based on whether value changed and previous confidence
            if prev is not None:
                if value == prev.value:
                    # Value unchanged - keep previous confidence
                    confidence = prev.confidence
                elif prev.value is None and value is not None:
                    # Found a missing value - moderate confidence
                    confidence = 0.65
                else:
                    # Value changed - moderate confidence for the correction
                    confidence = 0.70
            else:
                # New field - default confidence
                confidence = 0.60 if value is not None else None

            extractions.append(
                FieldExtraction(
                    field_path=field_path,
                    value=value,
                    confidence=confidence,
                    source_text=None,
                    reasoning="Refined based on verification feedback",
                )
            )

            # Handle nested fields
            if field_info.nested_fields and isinstance(value, dict):
                for nested_field in field_info.nested_fields:
                    nested_path = nested_field.path
                    nested_value = self._get_nested_value(data, nested_path)
                    nested_prev = prev_lookup.get(nested_path)

                    if nested_prev is not None:
                        if nested_value == nested_prev.value:
                            nested_confidence = nested_prev.confidence
                        else:
                            nested_confidence = 0.70
                    else:
                        nested_confidence = 0.60 if nested_value is not None else None

                    extractions.append(
                        FieldExtraction(
                            field_path=nested_path,
                            value=nested_value,
                            confidence=nested_confidence,
                            source_text=None,
                            reasoning="Refined based on verification feedback",
                        )
                    )

        return extractions

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


class AgenticLoop:
    """Orchestrator for the Plan → Execute → Verify → Refine loop.

    Coordinates the extraction planning, execution, verification, and
    refinement agents to iteratively improve extraction quality until
    thresholds are met or max iterations reached.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        default_max_iterations: int | None = None,
    ) -> None:
        """Initialize the agentic loop orchestrator.

        Args:
            api_key: OpenAI API key for all agents.
            model: Model name for all agents.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for responses.
            default_max_iterations: Default maximum iterations if not in plan.
                Defaults to settings.max_refinement_iterations.
        """
        self.api_key = api_key if api_key is not None else settings.get_openai_api_key()
        self.model = model or settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self.max_tokens = max_tokens or settings.openai_max_tokens
        self.default_max_iterations = (
            default_max_iterations
            if default_max_iterations is not None
            else settings.max_refinement_iterations
        )

        # Initialize agents lazily
        self._planner: ExtractionPlanningAgent | None = None
        self._verifier: QualityVerificationAgent | None = None
        self._refiner: RefinementAgent | None = None

    @property
    def planner(self) -> ExtractionPlanningAgent:
        """Get or create the planning agent."""
        if self._planner is None:
            self._planner = ExtractionPlanningAgent(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._planner

    @property
    def verifier(self) -> QualityVerificationAgent:
        """Get or create the verification agent."""
        if self._verifier is None:
            self._verifier = QualityVerificationAgent(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._verifier

    @property
    def refiner(self) -> RefinementAgent:
        """Get or create the refinement agent."""
        if self._refiner is None:
            self._refiner = RefinementAgent(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._refiner

    def run(
        self,
        text: str,
        schema_info: SchemaInfo,
        format_info: FormatInfo,
        initial_result: ExtractionResult | None = None,
        extraction_func: Any | None = None,
        plan: ExtractionPlan | None = None,
        use_llm_verification: bool = True,
    ) -> AgenticLoopResult:
        """Run the complete agentic extraction loop.

        Args:
            text: Source text to extract from.
            schema_info: Schema for extraction.
            format_info: Format information about the document.
            initial_result: Optional initial extraction result (skip first extraction).
            extraction_func: Function to call for extraction (receives text, schema_info).
            plan: Optional pre-computed extraction plan.
            use_llm_verification: Whether to use LLM for verification.

        Returns:
            AgenticLoopResult with the best extraction and loop metadata.

        Raises:
            RefinementError: If the loop encounters a critical error.
        """
        start_time = time.time()
        total_tokens = 0
        iteration_history: list[IterationMetrics] = []

        # Reset refiner memory for new extraction
        self.refiner.reset_memory()

        # Phase 1: Planning
        logger.info("Starting agentic loop: Phase 1 - Planning")
        if plan is None:
            plan = self.planner.create_plan(
                schema_info=schema_info,
                format_info=format_info,
                content_summary=text[:500] if text else None,
            )
            total_tokens += plan.total_tokens

        max_iterations = (
            plan.quality_thresholds.max_iterations or self.default_max_iterations
        )
        thresholds = plan.quality_thresholds
        processing_category = ProcessingCategory(
            plan.document_characteristics.processing_category
        )

        # Track best result
        best_result: ExtractionResult | None = None
        best_verification: VerificationReport | None = None
        best_score: float = -1.0
        best_iteration: int = 0

        # Phase 2-4: Execute → Verify → Refine loop
        current_result = initial_result
        previous_metrics: IterationMetrics | None = None

        for iteration in range(1, max_iterations + 1):
            iteration_start = time.time()
            iteration_tokens = 0

            logger.info(f"Agentic loop: Iteration {iteration}/{max_iterations}")

            # Phase 2: Execute (only if no current result)
            if current_result is None:
                if extraction_func is not None:
                    current_result = extraction_func(text, schema_info)
                    iteration_tokens += current_result.total_tokens
                else:
                    raise RefinementError(
                        "No initial result and no extraction function provided",
                        error_type="configuration_error",
                    )

            # Phase 3: Verify
            logger.info(f"Iteration {iteration}: Verifying extraction")
            verification = self.verifier.verify(
                extraction_result=current_result,
                schema_info=schema_info,
                thresholds=thresholds,
                processing_category=processing_category,
                use_llm_analysis=use_llm_verification,
            )
            iteration_tokens += verification.total_tokens

            # Calculate improvement from previous iteration
            improvements: dict[str, float] = {}
            if previous_metrics is not None:
                improvements = {
                    "confidence_delta": (
                        verification.metrics.overall_confidence
                        - previous_metrics.overall_confidence
                    ),
                    "coverage_delta": (
                        verification.metrics.required_field_coverage
                        - previous_metrics.required_field_coverage
                    ),
                    "issues_delta": (
                        previous_metrics.issues_count - len(verification.issues)
                    ),
                }

            # Record iteration metrics
            iteration_time = time.time() - iteration_start
            metrics = IterationMetrics(
                iteration_number=iteration,
                verification_status=verification.status,
                overall_confidence=verification.metrics.overall_confidence,
                required_field_coverage=verification.metrics.required_field_coverage,
                issues_count=len(verification.issues),
                critical_issues_count=len(verification.critical_issues),
                improvements_from_previous=improvements,
                total_tokens=iteration_tokens,
                processing_time_seconds=iteration_time,
            )
            iteration_history.append(metrics)
            total_tokens += iteration_tokens

            # Calculate score for tracking best result
            score = self._calculate_result_score(verification)
            if score > best_score:
                best_score = score
                best_result = current_result
                best_verification = verification
                best_iteration = iteration

            logger.info(
                f"Iteration {iteration}: "
                f"status={verification.status.value}, "
                f"confidence={verification.metrics.overall_confidence:.2f}, "
                f"coverage={verification.metrics.required_field_coverage:.2f}, "
                f"issues={len(verification.issues)}"
            )

            # Check if we've met the quality thresholds
            if verification.status == VerificationStatus.PASSED:
                logger.info(f"Quality thresholds met at iteration {iteration}")
                break

            # Check if we've reached max iterations
            if iteration >= max_iterations:
                logger.info(
                    f"Max iterations ({max_iterations}) reached without convergence"
                )
                break

            # Phase 4: Refine (prepare for next iteration)
            if verification.status in (
                VerificationStatus.FAILED,
                VerificationStatus.NEEDS_IMPROVEMENT,
            ):
                logger.info(f"Iteration {iteration}: Generating refinement")

                feedback = self.refiner.generate_feedback(
                    verification_report=verification,
                    schema_info=schema_info,
                    processing_category=processing_category,
                )

                current_result = self.refiner.refine(
                    text=text,
                    previous_result=current_result,
                    feedback=feedback,
                    schema_info=schema_info,
                )

            previous_metrics = metrics

        # Ensure we have a result
        if best_result is None or best_verification is None:
            raise RefinementError(
                "No extraction result produced",
                error_type="execution_error",
            )

        total_time = time.time() - start_time
        converged = best_verification.status == VerificationStatus.PASSED

        logger.info(
            f"Agentic loop completed: "
            f"iterations={len(iteration_history)}, "
            f"converged={converged}, "
            f"best_iteration={best_iteration}, "
            f"total_tokens={total_tokens}, "
            f"time={total_time:.2f}s"
        )

        return AgenticLoopResult(
            final_result=best_result,
            final_verification=best_verification,
            plan=plan,
            iterations_completed=len(iteration_history),
            iteration_history=iteration_history,
            converged=converged,
            best_iteration=best_iteration,
            total_tokens=total_tokens,
            total_processing_time_seconds=total_time,
            loop_metadata={
                "max_iterations_allowed": max_iterations,
                "thresholds": thresholds.to_dict(),
                "processing_category": processing_category.value,
            },
        )

    def _calculate_result_score(
        self,
        verification: VerificationReport,
    ) -> float:
        """Calculate a score for ranking extraction results.

        Args:
            verification: Verification report.

        Returns:
            Score value (higher is better).
        """
        metrics = verification.metrics

        # Weight different metrics
        score = (
            metrics.overall_confidence * 0.3
            + metrics.required_field_coverage * 0.4
            + metrics.completeness_score * 0.2
            + metrics.consistency_score * 0.1
        )

        # Penalize for critical issues
        score -= len(verification.critical_issues) * 0.2

        # Bonus for passing
        if verification.status == VerificationStatus.PASSED:
            score += 0.1

        return max(0.0, min(1.0, score))

    def run_with_visual_refinement(
        self,
        text: str,
        schema_info: SchemaInfo,
        format_info: FormatInfo,
        initial_result: ExtractionResult,
        region_reprocess_func: Any | None = None,  # noqa: ARG002
        plan: ExtractionPlan | None = None,
        use_llm_verification: bool = True,
    ) -> AgenticLoopResult:
        """Run agentic loop with visual document refinement support.

        This variant supports re-processing specific regions for visual
        documents when certain fields have issues.

        Args:
            text: OCR text from the document.
            schema_info: Schema for extraction.
            format_info: Format information.
            initial_result: Initial visual extraction result.
            region_reprocess_func: Optional function to reprocess specific regions.
                Reserved for future use with region-level re-processing.
            plan: Optional pre-computed extraction plan.
            use_llm_verification: Whether to use LLM for verification.

        Returns:
            AgenticLoopResult with the best extraction and loop metadata.
        """
        # For visual documents, we use the same loop but the refinement
        # uses a different strategy (visual_region_reprocessing)
        # Note: region_reprocess_func is reserved for future region-level
        # re-processing support

        return self.run(
            text=text,
            schema_info=schema_info,
            format_info=format_info,
            initial_result=initial_result,
            extraction_func=None,  # Don't need extraction func for visual
            plan=plan,
            use_llm_verification=use_llm_verification,
        )
