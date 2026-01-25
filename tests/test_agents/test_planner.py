"""Tests for the extraction planning agent."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

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
from agentic_document_extraction.models import (
    FormatFamily,
    FormatInfo,
    ProcessingCategory,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionType,
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
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                    },
                },
            },
            "employees": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "salary": {"type": "number"},
                    },
                },
            },
            "departments": {
                "type": "array",
                "items": {"type": "string"},
            },
            "financials": {
                "type": "object",
                "properties": {
                    "revenue": {"type": "number"},
                    "expenses": {"type": "number"},
                    "profit": {"type": "number"},
                    "year": {"type": "integer"},
                },
            },
        },
        "required": ["company", "employees", "financials"],
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
                    FieldInfo(name="name", field_type="string", path="company.name"),
                    FieldInfo(
                        name="address",
                        field_type="object",
                        path="company.address",
                        nested_fields=[
                            FieldInfo(
                                name="street",
                                field_type="string",
                                path="company.address.street",
                            ),
                            FieldInfo(
                                name="city",
                                field_type="string",
                                path="company.address.city",
                            ),
                        ],
                    ),
                ],
            ),
            FieldInfo(
                name="employees",
                field_type="array",
                required=True,
                path="employees",
                nested_fields=[
                    FieldInfo(
                        name="name", field_type="string", path="employees[].name"
                    ),
                    FieldInfo(
                        name="role", field_type="string", path="employees[].role"
                    ),
                    FieldInfo(
                        name="salary", field_type="number", path="employees[].salary"
                    ),
                ],
            ),
            FieldInfo(
                name="financials",
                field_type="object",
                required=True,
                path="financials",
                nested_fields=[
                    FieldInfo(
                        name="revenue", field_type="number", path="financials.revenue"
                    ),
                    FieldInfo(
                        name="expenses", field_type="number", path="financials.expenses"
                    ),
                ],
            ),
        ],
        optional_fields=[
            FieldInfo(
                name="departments",
                field_type="array",
                required=False,
                path="departments",
            ),
        ],
        schema_type="object",
    )


@pytest.fixture
def text_format_info() -> FormatInfo:
    """Create text-based format info for testing."""
    return FormatInfo(
        mime_type="text/plain",
        extension=".txt",
        format_family=FormatFamily.PLAIN_TEXT,
        processing_category=ProcessingCategory.TEXT_BASED,
        detected_from_content=True,
    )


@pytest.fixture
def pdf_format_info() -> FormatInfo:
    """Create PDF format info for testing."""
    return FormatInfo(
        mime_type="application/pdf",
        extension=".pdf",
        format_family=FormatFamily.PDF,
        processing_category=ProcessingCategory.VISUAL,
        detected_from_content=True,
    )


@pytest.fixture
def image_format_info() -> FormatInfo:
    """Create image format info for testing."""
    return FormatInfo(
        mime_type="image/png",
        extension=".png",
        format_family=FormatFamily.IMAGE,
        processing_category=ProcessingCategory.VISUAL,
        detected_from_content=True,
    )


@pytest.fixture
def sample_layout_regions() -> list[LayoutRegion]:
    """Create sample layout regions for testing."""
    return [
        LayoutRegion(
            region_id="region_0_0",
            region_type=RegionType.TITLE,
            bbox=RegionBoundingBox(x0=100, y0=50, x1=500, y1=100),
            confidence=0.95,
            page_number=1,
        ),
        LayoutRegion(
            region_id="region_0_1",
            region_type=RegionType.TEXT,
            bbox=RegionBoundingBox(x0=50, y0=150, x1=300, y1=400),
            confidence=0.92,
            page_number=1,
        ),
        LayoutRegion(
            region_id="region_0_2",
            region_type=RegionType.TEXT,
            bbox=RegionBoundingBox(x0=350, y0=150, x1=550, y1=400),
            confidence=0.90,
            page_number=1,
        ),
        LayoutRegion(
            region_id="region_0_3",
            region_type=RegionType.TABLE,
            bbox=RegionBoundingBox(x0=50, y0=450, x1=550, y1=700),
            confidence=0.88,
            page_number=1,
        ),
    ]


@pytest.fixture
def complex_layout_regions() -> list[LayoutRegion]:
    """Create complex layout regions with multiple content types."""
    return [
        LayoutRegion(
            region_id="r1",
            region_type=RegionType.TITLE,
            bbox=RegionBoundingBox(x0=100, y0=50, x1=500, y1=100),
            confidence=0.95,
            page_number=1,
        ),
        LayoutRegion(
            region_id="r2",
            region_type=RegionType.SECTION_HEADER,
            bbox=RegionBoundingBox(x0=50, y0=120, x1=300, y1=150),
            confidence=0.93,
            page_number=1,
        ),
        LayoutRegion(
            region_id="r3",
            region_type=RegionType.TEXT,
            bbox=RegionBoundingBox(x0=50, y0=160, x1=300, y1=350),
            confidence=0.91,
            page_number=1,
        ),
        LayoutRegion(
            region_id="r4",
            region_type=RegionType.TABLE,
            bbox=RegionBoundingBox(x0=320, y0=160, x1=550, y1=350),
            confidence=0.89,
            page_number=1,
        ),
        LayoutRegion(
            region_id="r5",
            region_type=RegionType.PICTURE,
            bbox=RegionBoundingBox(x0=50, y0=380, x1=250, y1=550),
            confidence=0.87,
            page_number=1,
        ),
        LayoutRegion(
            region_id="r6",
            region_type=RegionType.CAPTION,
            bbox=RegionBoundingBox(x0=50, y0=560, x1=250, y1=580),
            confidence=0.85,
            page_number=1,
        ),
        LayoutRegion(
            region_id="r7",
            region_type=RegionType.FORMULA,
            bbox=RegionBoundingBox(x0=300, y0=400, x1=550, y1=500),
            confidence=0.82,
            page_number=1,
        ),
    ]


class TestDocumentCharacteristics:
    """Tests for DocumentCharacteristics dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        chars = DocumentCharacteristics(
            processing_category="visual",
            format_family="pdf",
            estimated_complexity="moderate",
            page_count=5,
            has_tables=True,
            has_images=True,
            has_charts=False,
            is_multi_column=True,
            text_density="normal",
            region_count=10,
            region_types=["text", "table", "title"],
        )

        result = chars.to_dict()

        assert result["processing_category"] == "visual"
        assert result["format_family"] == "pdf"
        assert result["estimated_complexity"] == "moderate"
        assert result["page_count"] == 5
        assert result["has_tables"] is True
        assert result["has_images"] is True
        assert result["is_multi_column"] is True
        assert result["region_count"] == 10
        assert result["region_types"] == ["text", "table", "title"]


class TestExtractionStep:
    """Tests for ExtractionStep dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        step = ExtractionStep(
            step_number=1,
            action="Extract text",
            target="page_1",
            strategy="ocr",
            expected_output="Raw text content",
            fallback="use_vlm",
            depends_on=[],
        )

        result = step.to_dict()

        assert result["step_number"] == 1
        assert result["action"] == "Extract text"
        assert result["target"] == "page_1"
        assert result["strategy"] == "ocr"
        assert result["expected_output"] == "Raw text content"
        assert result["fallback"] == "use_vlm"
        assert result["depends_on"] == []

    def test_to_dict_no_fallback(self) -> None:
        """Test conversion without fallback."""
        step = ExtractionStep(
            step_number=2,
            action="Validate",
            target="extracted_data",
            strategy="schema_validation",
            expected_output="Validated data",
        )

        result = step.to_dict()

        assert result["fallback"] is None
        assert result["depends_on"] is None


class TestQualityThreshold:
    """Tests for QualityThreshold dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        threshold = QualityThreshold(
            min_overall_confidence=0.8,
            min_field_confidence=0.7,
            required_field_coverage=0.95,
            max_iterations=3,
        )

        result = threshold.to_dict()

        assert result["min_overall_confidence"] == 0.8
        assert result["min_field_confidence"] == 0.7
        assert result["required_field_coverage"] == 0.95
        assert result["max_iterations"] == 3


class TestRegionPriority:
    """Tests for RegionPriority dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        priority = RegionPriority(
            region_type="table",
            priority=1,
            reason="Contains financial data",
            schema_fields=["revenue", "expenses", "profit"],
        )

        result = priority.to_dict()

        assert result["region_type"] == "table"
        assert result["priority"] == 1
        assert result["reason"] == "Contains financial data"
        assert result["schema_fields"] == ["revenue", "expenses", "profit"]


class TestExtractionPlan:
    """Tests for ExtractionPlan dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        doc_chars = DocumentCharacteristics(
            processing_category="text_based",
            format_family="plain_text",
            estimated_complexity="simple",
        )

        steps = [
            ExtractionStep(
                step_number=1,
                action="Extract text",
                target="document",
                strategy="direct",
                expected_output="Text content",
            )
        ]

        thresholds = QualityThreshold(
            min_overall_confidence=0.8,
            min_field_confidence=0.7,
            required_field_coverage=0.95,
            max_iterations=2,
        )

        plan = ExtractionPlan(
            document_characteristics=doc_chars,
            schema_complexity=SchemaComplexity.SIMPLE,
            extraction_strategy="text_direct",
            steps=steps,
            region_priorities=[],
            challenges=[ExtractionChallenge.SPARSE_CONTENT],
            quality_thresholds=thresholds,
            reasoning="Simple document extraction plan.",
            estimated_confidence=0.9,
            model_used="fast_path",
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            processing_time_seconds=0.1,
        )

        result = plan.to_dict()

        assert result["schema_complexity"] == "simple"
        assert result["extraction_strategy"] == "text_direct"
        assert len(result["steps"]) == 1
        assert result["challenges"] == ["sparse_content"]
        assert result["estimated_confidence"] == 0.9
        assert result["metadata"]["model_used"] == "fast_path"


class TestSchemaComplexityAnalysis:
    """Tests for schema complexity analysis."""

    def test_simple_schema(self, simple_schema_info: SchemaInfo) -> None:
        """Test detection of simple schema."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        complexity = agent._analyze_schema_complexity(simple_schema_info)
        assert complexity == SchemaComplexity.SIMPLE

    def test_complex_schema(self, complex_schema_info: SchemaInfo) -> None:
        """Test detection of complex schema."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        complexity = agent._analyze_schema_complexity(complex_schema_info)
        assert complexity == SchemaComplexity.COMPLEX

    def test_moderate_schema(self) -> None:
        """Test detection of moderate schema."""
        # 8 fields, 2 levels deep, no arrays of objects
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "date": {"type": "string"},
                "summary": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "pages": {"type": "integer"},
                        "format": {"type": "string"},
                    },
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }

        # Build schema info with 8 fields
        schema_info = SchemaInfo(
            schema=schema,
            required_fields=[
                FieldInfo(
                    name="title", field_type="string", required=True, path="title"
                ),
                FieldInfo(
                    name="author", field_type="string", required=True, path="author"
                ),
            ],
            optional_fields=[
                FieldInfo(
                    name="date", field_type="string", required=False, path="date"
                ),
                FieldInfo(
                    name="summary", field_type="string", required=False, path="summary"
                ),
                FieldInfo(
                    name="metadata",
                    field_type="object",
                    required=False,
                    path="metadata",
                    nested_fields=[
                        FieldInfo(
                            name="pages",
                            field_type="integer",
                            required=False,
                            path="metadata.pages",
                        ),
                        FieldInfo(
                            name="format",
                            field_type="string",
                            required=False,
                            path="metadata.format",
                        ),
                    ],
                ),
                FieldInfo(name="tags", field_type="array", required=False, path="tags"),
            ],
            schema_type="object",
        )

        agent = ExtractionPlanningAgent(api_key="test-key")
        complexity = agent._analyze_schema_complexity(schema_info)
        assert complexity == SchemaComplexity.MODERATE


class TestDocumentAnalysis:
    """Tests for document analysis."""

    def test_analyze_text_document(self, text_format_info: FormatInfo) -> None:
        """Test analysis of text document."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        chars = agent._analyze_document(text_format_info, None, 1)

        assert chars.processing_category == "text_based"
        assert chars.format_family == "plain_text"
        assert chars.estimated_complexity == "simple"
        assert chars.has_tables is False
        assert chars.region_count == 0

    def test_analyze_visual_document_simple(
        self, pdf_format_info: FormatInfo, sample_layout_regions: list[LayoutRegion]
    ) -> None:
        """Test analysis of visual document with tables."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        chars = agent._analyze_document(pdf_format_info, sample_layout_regions, 1)

        assert chars.processing_category == "visual"
        assert chars.format_family == "pdf"
        # Has tables so should be complex
        assert chars.estimated_complexity == "complex"
        assert chars.has_tables is True
        assert chars.region_count == 4
        assert "table" in chars.region_types

    def test_analyze_visual_document_complex(
        self, pdf_format_info: FormatInfo, complex_layout_regions: list[LayoutRegion]
    ) -> None:
        """Test analysis of complex visual document."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        chars = agent._analyze_document(pdf_format_info, complex_layout_regions, 3)

        assert chars.processing_category == "visual"
        assert chars.has_tables is True
        assert chars.has_images is True
        assert chars.estimated_complexity == "complex"
        assert "table" in chars.region_types
        assert "picture" in chars.region_types
        assert "formula" in chars.region_types

    def test_detect_multi_column(
        self, sample_layout_regions: list[LayoutRegion]
    ) -> None:
        """Test multi-column detection."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        # Sample regions have two text regions side by side at similar y positions
        is_multi_column = agent._detect_multi_column(sample_layout_regions)
        assert is_multi_column is True

    def test_detect_single_column(self) -> None:
        """Test single column detection."""
        single_column_regions = [
            LayoutRegion(
                region_id="r1",
                region_type=RegionType.TITLE,
                bbox=RegionBoundingBox(x0=100, y0=50, x1=500, y1=100),
                confidence=0.95,
                page_number=1,
            ),
            LayoutRegion(
                region_id="r2",
                region_type=RegionType.TEXT,
                bbox=RegionBoundingBox(x0=50, y0=150, x1=550, y1=400),
                confidence=0.92,
                page_number=1,
            ),
            LayoutRegion(
                region_id="r3",
                region_type=RegionType.TEXT,
                bbox=RegionBoundingBox(x0=50, y0=420, x1=550, y1=600),
                confidence=0.90,
                page_number=1,
            ),
        ]

        agent = ExtractionPlanningAgent(api_key="test-key")
        is_multi_column = agent._detect_multi_column(single_column_regions)
        assert is_multi_column is False


class TestFastPathPlanning:
    """Tests for fast path (non-LLM) planning."""

    def test_fast_path_simple_text(
        self, text_format_info: FormatInfo, simple_schema_info: SchemaInfo
    ) -> None:
        """Test fast path planning for simple text document."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        plan = agent.create_plan(
            schema_info=simple_schema_info,
            format_info=text_format_info,
            page_count=1,
        )

        assert plan.extraction_strategy == "text_direct"
        assert plan.schema_complexity == SchemaComplexity.SIMPLE
        assert plan.model_used == "fast_path"
        assert len(plan.steps) == 3
        assert plan.steps[0].action == "Extract text content"
        assert plan.quality_thresholds.min_overall_confidence == 0.8
        assert plan.estimated_confidence == 0.85

    def test_fast_path_large_text(
        self, text_format_info: FormatInfo, simple_schema_info: SchemaInfo
    ) -> None:
        """Test fast path with chunked strategy for large documents."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        plan = agent.create_plan(
            schema_info=simple_schema_info,
            format_info=text_format_info,
            page_count=20,  # Large document triggers chunked strategy
        )

        assert plan.extraction_strategy == "text_chunked"
        assert plan.model_used == "fast_path"

    def test_can_use_fast_path_text_simple(self) -> None:
        """Test fast path eligibility for simple text document."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        doc_chars = DocumentCharacteristics(
            processing_category=ProcessingCategory.TEXT_BASED.value,
            format_family="plain_text",
            estimated_complexity="simple",
        )

        can_use = agent._can_use_fast_path(doc_chars, SchemaComplexity.SIMPLE)
        assert can_use is True

    def test_cannot_use_fast_path_visual(self) -> None:
        """Test fast path not used for visual documents."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        doc_chars = DocumentCharacteristics(
            processing_category=ProcessingCategory.VISUAL.value,
            format_family="pdf",
            estimated_complexity="moderate",
        )

        can_use = agent._can_use_fast_path(doc_chars, SchemaComplexity.SIMPLE)
        assert can_use is False


class TestLLMPlanning:
    """Tests for LLM-based planning."""

    def test_llm_planning_visual_document(
        self,
        pdf_format_info: FormatInfo,
        complex_schema_info: SchemaInfo,
        complex_layout_regions: list[LayoutRegion],
    ) -> None:
        """Test LLM planning for complex visual document."""
        # Create mock response
        llm_response = {
            "schema_complexity": "complex",
            "extraction_strategy": "visual_vlm",
            "steps": [
                {
                    "step_number": 1,
                    "action": "OCR text extraction",
                    "target": "all_pages",
                    "strategy": "pytesseract",
                    "expected_output": "Raw text with bounding boxes",
                    "fallback": "cloud_ocr",
                    "depends_on": None,
                },
                {
                    "step_number": 2,
                    "action": "Table extraction",
                    "target": "table_regions",
                    "strategy": "vlm_table_specialized",
                    "expected_output": "Structured table data",
                    "fallback": "ocr_table_heuristics",
                    "depends_on": [1],
                },
            ],
            "region_priorities": [
                {
                    "region_type": "table",
                    "priority": 1,
                    "reason": "Contains financial data",
                    "schema_fields": ["financials.revenue", "financials.expenses"],
                },
                {
                    "region_type": "text",
                    "priority": 2,
                    "reason": "Contains company and employee info",
                    "schema_fields": ["company.name", "employees"],
                },
            ],
            "challenges": ["complex_table", "multi_column", "chart_graph"],
            "quality_thresholds": {
                "min_overall_confidence": 0.65,
                "min_field_confidence": 0.55,
                "required_field_coverage": 0.85,
                "max_iterations": 4,
            },
            "reasoning": "Complex visual document requires full VLM pipeline.",
            "estimated_confidence": 0.72,
        }

        # Setup mock
        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 500,
                "output_tokens": 300,
                "total_tokens": 800,
            },
        )

        agent = ExtractionPlanningAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        plan = agent.create_plan(
            schema_info=complex_schema_info,
            format_info=pdf_format_info,
            content_summary="Financial report with tables and charts.",
            layout_regions=complex_layout_regions,
            page_count=5,
        )

        assert plan.extraction_strategy == "visual_vlm"
        assert plan.schema_complexity == SchemaComplexity.COMPLEX
        assert len(plan.steps) == 2
        assert len(plan.region_priorities) == 2
        assert ExtractionChallenge.COMPLEX_TABLE in plan.challenges
        assert plan.quality_thresholds.min_overall_confidence == 0.65
        assert plan.estimated_confidence == 0.72
        assert plan.total_tokens == 800

    def test_llm_planning_parses_challenges(self, pdf_format_info: FormatInfo) -> None:
        """Test that challenges are properly parsed."""
        llm_response = {
            "schema_complexity": "moderate",
            "extraction_strategy": "visual_ocr_simple",
            "steps": [
                {
                    "step_number": 1,
                    "action": "Extract",
                    "target": "document",
                    "strategy": "ocr",
                    "expected_output": "Text",
                }
            ],
            "region_priorities": [],
            "challenges": [
                "complex_table",
                "multi_column",
                "dense_text",
                "unknown_challenge",  # Should be skipped with warning
            ],
            "quality_thresholds": {
                "min_overall_confidence": 0.7,
                "min_field_confidence": 0.6,
                "required_field_coverage": 0.9,
                "max_iterations": 3,
            },
            "reasoning": "Test reasoning",
            "estimated_confidence": 0.75,
        }

        mock_response = AIMessage(
            content=json.dumps(llm_response),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 100,
                "total_tokens": 200,
            },
        )

        schema_info = SchemaInfo(
            schema={"type": "object", "properties": {"test": {"type": "string"}}},
            required_fields=[],
            optional_fields=[],
            schema_type="object",
        )

        agent = ExtractionPlanningAgent(api_key="test-key")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_response]}
        agent._agent = mock_agent

        plan = agent.create_plan(
            schema_info=schema_info,
            format_info=pdf_format_info,
        )

        # Should have 3 valid challenges (unknown_challenge is skipped)
        assert len(plan.challenges) == 3
        assert ExtractionChallenge.COMPLEX_TABLE in plan.challenges
        assert ExtractionChallenge.MULTI_COLUMN in plan.challenges
        assert ExtractionChallenge.DENSE_TEXT in plan.challenges


class TestPlanningPrompts:
    """Tests for planning prompt construction."""

    def test_format_fields(self, simple_schema_info: SchemaInfo) -> None:
        """Test field formatting for prompts."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        required_str = agent._format_fields(simple_schema_info.required_fields)
        assert "name" in required_str
        assert "string" in required_str

        optional_str = agent._format_fields(simple_schema_info.optional_fields)
        assert "age" in optional_str
        assert "integer" in optional_str

    def test_format_fields_empty(self) -> None:
        """Test formatting empty fields list."""
        agent = ExtractionPlanningAgent(api_key="test-key")
        result = agent._format_fields([])
        assert result == "None"

    def test_format_fields_with_description(self) -> None:
        """Test formatting fields with descriptions."""
        fields = [
            FieldInfo(
                name="email",
                field_type="string",
                required=True,
                path="email",
                description="Contact email address",
            )
        ]
        agent = ExtractionPlanningAgent(api_key="test-key")
        result = agent._format_fields(fields)

        assert "email" in result
        assert "string" in result
        assert "Contact email address" in result


class TestParseResponses:
    """Tests for response parsing."""

    def test_parse_plan_response_valid(self) -> None:
        """Test parsing valid JSON response."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        response = json.dumps(
            {
                "schema_complexity": "simple",
                "extraction_strategy": "text_direct",
                "steps": [],
                "region_priorities": [],
                "challenges": [],
                "quality_thresholds": {},
                "reasoning": "Test",
                "estimated_confidence": 0.8,
            }
        )

        result = agent._parse_plan_response(response)
        assert result["schema_complexity"] == "simple"

    def test_parse_plan_response_with_wrapper(self) -> None:
        """Test parsing JSON with extra text around it."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        response = 'Here is the plan: {"extraction_strategy": "test"} That is the plan.'

        result = agent._parse_plan_response(response)
        assert result["extraction_strategy"] == "test"

    def test_parse_plan_response_invalid(self) -> None:
        """Test parsing invalid JSON raises error."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        with pytest.raises(PlanningError) as exc_info:
            agent._parse_plan_response("This is not JSON at all!")

        assert exc_info.value.error_type == "parse_error"

    def test_build_steps_from_response(self) -> None:
        """Test building steps from response data."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        steps_data = [
            {
                "step_number": 1,
                "action": "Extract",
                "target": "document",
                "strategy": "ocr",
                "expected_output": "Text",
                "fallback": "vlm",
                "depends_on": None,
            },
            {
                "step_number": 2,
                "action": "Validate",
                "target": "data",
                "strategy": "schema",
                "expected_output": "Valid data",
                "depends_on": [1],
            },
        ]

        steps = agent._build_steps_from_response(steps_data)

        assert len(steps) == 2
        assert steps[0].step_number == 1
        assert steps[0].fallback == "vlm"
        assert steps[1].depends_on == [1]

    def test_build_quality_thresholds_defaults(self) -> None:
        """Test quality thresholds with defaults."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        thresholds = agent._build_quality_thresholds({})

        assert thresholds.min_overall_confidence == 0.7
        assert thresholds.min_field_confidence == 0.6
        assert thresholds.required_field_coverage == 0.9
        assert thresholds.max_iterations == 3

    def test_parse_challenges_mapping(self) -> None:
        """Test challenge parsing with common variations."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        challenges = agent._parse_challenges(
            [
                "table",
                "multi-column",
                "chart",
                "dense",
            ]
        )

        assert ExtractionChallenge.COMPLEX_TABLE in challenges
        assert ExtractionChallenge.MULTI_COLUMN in challenges
        assert ExtractionChallenge.CHART_GRAPH in challenges
        assert ExtractionChallenge.DENSE_TEXT in challenges


class TestNestingDepthCalculation:
    """Tests for nesting depth calculation."""

    def test_flat_schema(self) -> None:
        """Test depth of flat schema."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        depth = agent._calculate_nesting_depth(schema)
        assert depth == 1

    def test_nested_schema(self) -> None:
        """Test depth of nested schema."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }

        depth = agent._calculate_nesting_depth(schema)
        assert depth == 3

    def test_array_schema(self) -> None:
        """Test depth with arrays."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        }

        depth = agent._calculate_nesting_depth(schema)
        assert depth == 3


class TestTextDensityEstimation:
    """Tests for text density estimation."""

    def test_sparse_text(self) -> None:
        """Test detection of sparse text."""
        # Small text regions in large area
        regions = [
            LayoutRegion(
                region_id="r1",
                region_type=RegionType.TEXT,
                bbox=RegionBoundingBox(x0=100, y0=100, x1=200, y1=150),  # Small region
                confidence=0.9,
                page_number=1,
            ),
            LayoutRegion(
                region_id="r2",
                region_type=RegionType.PICTURE,  # Large non-text region
                bbox=RegionBoundingBox(x0=0, y0=0, x1=1000, y1=1000),
                confidence=0.9,
                page_number=1,
            ),
        ]

        agent = ExtractionPlanningAgent(api_key="test-key")
        density = agent._estimate_text_density(regions)
        assert density == "sparse"

    def test_dense_text(self) -> None:
        """Test detection of dense text."""
        # Text regions covering most of the area
        regions = [
            LayoutRegion(
                region_id="r1",
                region_type=RegionType.TEXT,
                bbox=RegionBoundingBox(x0=0, y0=0, x1=500, y1=400),
                confidence=0.9,
                page_number=1,
            ),
            LayoutRegion(
                region_id="r2",
                region_type=RegionType.TEXT,
                bbox=RegionBoundingBox(x0=0, y0=400, x1=500, y1=800),
                confidence=0.9,
                page_number=1,
            ),
        ]

        agent = ExtractionPlanningAgent(api_key="test-key")
        density = agent._estimate_text_density(regions)
        assert density == "dense"

    def test_normal_text(self) -> None:
        """Test detection of normal text density."""
        regions = [
            LayoutRegion(
                region_id="r1",
                region_type=RegionType.TEXT,
                bbox=RegionBoundingBox(x0=50, y0=50, x1=450, y1=300),
                confidence=0.9,
                page_number=1,
            ),
            LayoutRegion(
                region_id="r2",
                region_type=RegionType.PICTURE,
                bbox=RegionBoundingBox(x0=50, y0=320, x1=450, y1=500),
                confidence=0.9,
                page_number=1,
            ),
        ]

        agent = ExtractionPlanningAgent(api_key="test-key")
        density = agent._estimate_text_density(regions)
        assert density == "normal"


class TestPlanningErrors:
    """Tests for error handling."""

    def test_no_api_key_error(self) -> None:
        """Test error when API key is missing."""
        agent = ExtractionPlanningAgent(api_key="")

        with pytest.raises(PlanningError) as exc_info:
            _ = agent.llm

        assert exc_info.value.error_type == "configuration_error"
        assert "openai_api_key" in str(exc_info.value.details)

    def test_planning_error_attributes(self) -> None:
        """Test PlanningError attributes."""
        error = PlanningError(
            "Test error",
            error_type="test_error",
            details={"key": "value"},
        )

        assert str(error) == "Test error"
        assert error.error_type == "test_error"
        assert error.details == {"key": "value"}


class TestSchemaArrayDetection:
    """Tests for array detection in schemas."""

    def test_has_arrays_true(self) -> None:
        """Test detection of arrays in schema."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }

        assert agent._schema_has_arrays(schema) is True

    def test_has_arrays_false(self) -> None:
        """Test no arrays in schema."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        assert agent._schema_has_arrays(schema) is False

    def test_has_nested_arrays(self) -> None:
        """Test detection of nested arrays."""
        agent = ExtractionPlanningAgent(api_key="test-key")

        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
        }

        assert agent._schema_has_arrays(schema) is True
