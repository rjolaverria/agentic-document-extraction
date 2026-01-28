"""Agentic components for document extraction.

This package contains LangChain agents for planning, verification, and
refinement of document extraction tasks.
"""

from agentic_document_extraction.agents.extraction_agent import ExtractionAgent
from agentic_document_extraction.agents.planner import (
    DocumentCharacteristics,
    ExtractionChallenge,
    ExtractionPlan,
    ExtractionPlanningAgent,
    ExtractionStep,
    PlanningError,
    QualityThreshold,
    RegionPriority,
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
from agentic_document_extraction.agents.tools import (
    AnalyzeChart,
    AnalyzeTable,
    analyze_chart,
    analyze_chart_impl,
    analyze_table,
    analyze_table_impl,
)
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

__all__ = [
    # Extraction Agent (new)
    "ExtractionAgent",
    # Planner
    "DocumentCharacteristics",
    "ExtractionChallenge",
    "ExtractionPlan",
    "ExtractionPlanningAgent",
    "ExtractionStep",
    "PlanningError",
    "QualityThreshold",
    "RegionPriority",
    "SchemaComplexity",
    # Refiner
    "AgenticLoop",
    "AgenticLoopResult",
    "IterationMetrics",
    "RefinementAgent",
    "RefinementError",
    "RefinementFeedback",
    # Verifier
    "IssueSeverity",
    "IssueType",
    "QualityMetrics",
    "QualityVerificationAgent",
    "VerificationError",
    "VerificationIssue",
    "VerificationReport",
    "VerificationStatus",
    # Tools
    "AnalyzeChart",
    "AnalyzeTable",
    "analyze_chart",
    "analyze_chart_impl",
    "analyze_table",
    "analyze_table_impl",
]
