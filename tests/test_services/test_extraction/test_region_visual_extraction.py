"""Tests for the region visual extraction service."""

import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_document_extraction.services.extraction.region_visual_extraction import (
    REGION_TYPE_STRATEGIES,
    DocumentRegionExtractionResult,
    ExtractionStrategy,
    RegionContext,
    RegionExtractionResult,
    RegionVisualExtractionError,
    RegionVisualExtractor,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionType,
)
from agentic_document_extraction.services.reading_order_detector import OrderedRegion


# Helper functions for creating test objects
def create_region(
    region_id: str,
    region_type: RegionType,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_number: int = 1,
    confidence: float = 0.9,
    parent_region_id: str | None = None,
) -> LayoutRegion:
    """Helper to create a layout region for testing."""
    return LayoutRegion(
        region_type=region_type,
        bbox=RegionBoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        confidence=confidence,
        page_number=page_number,
        region_id=region_id,
        parent_region_id=parent_region_id,
    )


def create_test_image(
    width: int = 100, height: int = 100, color: str = "white"
) -> Image.Image:
    """Helper to create a test image."""
    return Image.new("RGB", (width, height), color=color)


def create_ordered_region(
    region: LayoutRegion,
    order_index: int,
    confidence: float = 0.9,
    skip_in_reading: bool = False,
) -> OrderedRegion:
    """Helper to create an ordered region for testing."""
    return OrderedRegion(
        region=region,
        order_index=order_index,
        confidence=confidence,
        skip_in_reading=skip_in_reading,
    )


@pytest.fixture
def extractor() -> RegionVisualExtractor:
    """Create a RegionVisualExtractor instance for testing."""
    return RegionVisualExtractor(api_key="test-key")


@pytest.fixture
def sample_region() -> LayoutRegion:
    """Create a sample layout region."""
    return create_region("r1", RegionType.TEXT, 100, 100, 500, 300)


@pytest.fixture
def sample_table_region() -> LayoutRegion:
    """Create a sample table region."""
    return create_region("r_table", RegionType.TABLE, 100, 400, 800, 700)


@pytest.fixture
def sample_figure_region() -> LayoutRegion:
    """Create a sample figure region."""
    return create_region("r_figure", RegionType.PICTURE, 200, 500, 600, 800)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample test image."""
    return create_test_image(100, 100, "white")


@pytest.fixture
def page_image() -> Image.Image:
    """Create a sample full page image."""
    return create_test_image(1000, 1200, "white")


@pytest.fixture
def mock_vlm_response_text() -> str:
    """Mock VLM response for text extraction."""
    return json.dumps(
        {
            "extracted_content": {
                "text": "This is sample extracted text.",
                "text_type": "paragraph",
                "formatting": {
                    "has_bold": False,
                    "has_italic": False,
                    "has_bullet_points": False,
                },
                "line_count": 1,
            },
            "confidence": 0.92,
            "reasoning": "Clear text visible in the region.",
        }
    )


@pytest.fixture
def mock_vlm_response_table() -> str:
    """Mock VLM response for table extraction."""
    return json.dumps(
        {
            "extracted_content": {
                "headers": ["Name", "Age", "City"],
                "rows": [
                    ["Alice", "30", "New York"],
                    ["Bob", "25", "Los Angeles"],
                ],
                "has_header_row": True,
                "total_rows": 2,
                "total_columns": 3,
                "notes": "Standard 3-column table with header row.",
            },
            "confidence": 0.88,
            "reasoning": "Table structure clearly visible with distinct rows and columns.",
        }
    )


@pytest.fixture
def mock_vlm_response_figure() -> str:
    """Mock VLM response for figure extraction."""
    return json.dumps(
        {
            "extracted_content": {
                "visual_type": "chart",
                "description": "Bar chart showing sales by quarter.",
                "labels": ["Q1", "Q2", "Q3", "Q4"],
                "data_points": [
                    {"label": "Q1", "value": "100"},
                    {"label": "Q2", "value": "150"},
                    {"label": "Q3", "value": "120"},
                    {"label": "Q4", "value": "180"},
                ],
                "colors_used": ["blue"],
                "annotations": ["Sales 2024"],
            },
            "confidence": 0.85,
            "reasoning": "Bar chart with clear labels and values.",
        }
    )


class TestRegionVisualExtractionError:
    """Tests for RegionVisualExtractionError exception."""

    def test_error_with_message(self) -> None:
        """Test error with just message."""
        error = RegionVisualExtractionError("Test error")

        assert str(error) == "Test error"
        assert error.region_id is None
        assert error.details == {}

    def test_error_with_region_id(self) -> None:
        """Test error with region ID."""
        error = RegionVisualExtractionError("Test error", region_id="r1")

        assert error.region_id == "r1"

    def test_error_with_details(self) -> None:
        """Test error with details."""
        error = RegionVisualExtractionError(
            "Test error",
            region_id="r1",
            details={"error_type": "parse_error"},
        )

        assert error.details == {"error_type": "parse_error"}


class TestExtractionStrategy:
    """Tests for ExtractionStrategy enum."""

    def test_all_strategies_defined(self) -> None:
        """Test that all expected strategies are defined."""
        assert ExtractionStrategy.VLM_ONLY.value == "vlm_only"
        assert ExtractionStrategy.OCR_ENHANCED.value == "ocr_enhanced"
        assert ExtractionStrategy.TABLE_SPECIALIZED.value == "table_specialized"
        assert ExtractionStrategy.FIGURE_ANALYSIS.value == "figure_analysis"

    def test_region_type_strategies_mapping(self) -> None:
        """Test that region types map to appropriate strategies."""
        assert (
            REGION_TYPE_STRATEGIES[RegionType.TEXT] == ExtractionStrategy.OCR_ENHANCED
        )
        assert (
            REGION_TYPE_STRATEGIES[RegionType.TABLE]
            == ExtractionStrategy.TABLE_SPECIALIZED
        )
        assert (
            REGION_TYPE_STRATEGIES[RegionType.PICTURE]
            == ExtractionStrategy.FIGURE_ANALYSIS
        )
        assert (
            REGION_TYPE_STRATEGIES[RegionType.TITLE] == ExtractionStrategy.OCR_ENHANCED
        )
        assert REGION_TYPE_STRATEGIES[RegionType.UNKNOWN] == ExtractionStrategy.VLM_ONLY


class TestRegionContext:
    """Tests for RegionContext dataclass."""

    def test_create_context(self) -> None:
        """Test creating a RegionContext."""
        context = RegionContext(
            region_type=RegionType.TEXT,
            page_number=1,
            page_dimensions=(1000.0, 1200.0),
            bbox_normalized=(0.1, 0.1, 0.5, 0.3),
            reading_order_position=0,
            total_regions_on_page=5,
            preceding_region_types=["title"],
            following_region_types=["text", "table"],
            parent_region_type=None,
            ocr_text="Sample OCR text",
        )

        assert context.region_type == RegionType.TEXT
        assert context.page_number == 1
        assert context.page_dimensions == (1000.0, 1200.0)
        assert context.reading_order_position == 0
        assert context.total_regions_on_page == 5
        assert context.ocr_text == "Sample OCR text"

    def test_to_dict(self) -> None:
        """Test converting RegionContext to dictionary."""
        context = RegionContext(
            region_type=RegionType.TABLE,
            page_number=2,
            page_dimensions=(800.0, 1000.0),
            bbox_normalized=(0.2, 0.3, 0.8, 0.6),
            reading_order_position=3,
            total_regions_on_page=8,
        )

        result = context.to_dict()

        assert result["region_type"] == "table"
        assert result["page_number"] == 2
        assert result["page_dimensions"]["width"] == 800.0
        assert result["page_dimensions"]["height"] == 1000.0
        assert result["bbox_normalized"]["x0"] == 0.2
        assert result["reading_order_position"] == 3
        assert result["has_ocr_text"] is False


class TestRegionExtractionResult:
    """Tests for RegionExtractionResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a RegionExtractionResult."""
        result = RegionExtractionResult(
            region_id="r1",
            region_type=RegionType.TEXT,
            extracted_content={"text": "Hello world"},
            confidence=0.95,
            strategy_used=ExtractionStrategy.OCR_ENHANCED,
            reasoning="Clear text",
            raw_response='{"extracted_content": {"text": "Hello world"}}',
            processing_time_seconds=0.5,
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert result.region_id == "r1"
        assert result.region_type == RegionType.TEXT
        assert result.extracted_content["text"] == "Hello world"
        assert result.confidence == 0.95
        assert result.strategy_used == ExtractionStrategy.OCR_ENHANCED
        assert result.processing_time_seconds == 0.5

    def test_to_dict(self) -> None:
        """Test converting RegionExtractionResult to dictionary."""
        result = RegionExtractionResult(
            region_id="r_table",
            region_type=RegionType.TABLE,
            extracted_content={"headers": ["A", "B"]},
            confidence=0.88,
            strategy_used=ExtractionStrategy.TABLE_SPECIALIZED,
            processing_time_seconds=1.2,
            prompt_tokens=200,
            completion_tokens=100,
        )

        data = result.to_dict()

        assert data["region_id"] == "r_table"
        assert data["region_type"] == "table"
        assert data["confidence"] == 0.88
        assert data["strategy_used"] == "table_specialized"
        assert data["tokens"]["prompt"] == 200
        assert data["tokens"]["completion"] == 100


class TestDocumentRegionExtractionResult:
    """Tests for DocumentRegionExtractionResult dataclass."""

    def test_create_document_result(self) -> None:
        """Test creating a DocumentRegionExtractionResult."""
        region_results = [
            RegionExtractionResult(
                region_id="r1",
                region_type=RegionType.TEXT,
                extracted_content={"text": "Text 1"},
                confidence=0.9,
                strategy_used=ExtractionStrategy.OCR_ENHANCED,
            ),
            RegionExtractionResult(
                region_id="r2",
                region_type=RegionType.TABLE,
                extracted_content={"headers": ["A"]},
                confidence=0.85,
                strategy_used=ExtractionStrategy.TABLE_SPECIALIZED,
            ),
        ]

        doc_result = DocumentRegionExtractionResult(
            region_results=region_results,
            total_regions=2,
            successful_extractions=2,
            failed_extractions=0,
            model_used="gpt-4o",
            total_tokens=300,
            prompt_tokens=200,
            completion_tokens=100,
            processing_time_seconds=2.5,
        )

        assert len(doc_result.region_results) == 2
        assert doc_result.total_regions == 2
        assert doc_result.successful_extractions == 2
        assert doc_result.failed_extractions == 0
        assert doc_result.model_used == "gpt-4o"

    def test_to_dict(self) -> None:
        """Test converting DocumentRegionExtractionResult to dictionary."""
        doc_result = DocumentRegionExtractionResult(
            region_results=[
                RegionExtractionResult(
                    region_id="r1",
                    region_type=RegionType.TEXT,
                    extracted_content={},
                    confidence=0.9,
                    strategy_used=ExtractionStrategy.OCR_ENHANCED,
                ),
            ],
            total_regions=1,
            successful_extractions=1,
            failed_extractions=0,
            model_used="gpt-4o",
            processing_time_seconds=1.0,
        )

        data = doc_result.to_dict()

        assert data["summary"]["total_regions"] == 1
        assert data["summary"]["successful_extractions"] == 1
        assert data["metadata"]["model_used"] == "gpt-4o"
        assert len(data["region_results"]) == 1

    def test_get_results_by_type(self) -> None:
        """Test filtering results by region type."""
        region_results = [
            RegionExtractionResult(
                region_id="r1",
                region_type=RegionType.TEXT,
                extracted_content={},
                confidence=0.9,
                strategy_used=ExtractionStrategy.OCR_ENHANCED,
            ),
            RegionExtractionResult(
                region_id="r2",
                region_type=RegionType.TABLE,
                extracted_content={},
                confidence=0.85,
                strategy_used=ExtractionStrategy.TABLE_SPECIALIZED,
            ),
            RegionExtractionResult(
                region_id="r3",
                region_type=RegionType.TEXT,
                extracted_content={},
                confidence=0.88,
                strategy_used=ExtractionStrategy.OCR_ENHANCED,
            ),
        ]

        doc_result = DocumentRegionExtractionResult(
            region_results=region_results,
            total_regions=3,
            successful_extractions=3,
            failed_extractions=0,
            model_used="gpt-4o",
        )

        text_results = doc_result.get_results_by_type(RegionType.TEXT)
        assert len(text_results) == 2
        assert all(r.region_type == RegionType.TEXT for r in text_results)

        table_results = doc_result.get_results_by_type(RegionType.TABLE)
        assert len(table_results) == 1

    def test_get_result_by_id(self) -> None:
        """Test getting a result by region ID."""
        region_results = [
            RegionExtractionResult(
                region_id="r1",
                region_type=RegionType.TEXT,
                extracted_content={"text": "Test"},
                confidence=0.9,
                strategy_used=ExtractionStrategy.OCR_ENHANCED,
            ),
        ]

        doc_result = DocumentRegionExtractionResult(
            region_results=region_results,
            total_regions=1,
            successful_extractions=1,
            failed_extractions=0,
            model_used="gpt-4o",
        )

        result = doc_result.get_result_by_id("r1")
        assert result is not None
        assert result.region_id == "r1"

        not_found = doc_result.get_result_by_id("nonexistent")
        assert not_found is None


class TestRegionVisualExtractorInit:
    """Tests for RegionVisualExtractor initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with defaults."""
        with patch(
            "agentic_document_extraction.services.extraction.region_visual_extraction.settings"
        ) as mock_settings:
            mock_settings.openai_api_key = "env-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.0
            mock_settings.openai_max_tokens = 4096

            extractor = RegionVisualExtractor()

            assert extractor.api_key == "env-key"
            assert extractor.model == "gpt-4o"
            assert extractor.temperature == 0.0
            assert extractor.max_tokens == 4096

    def test_custom_initialization(self) -> None:
        """Test initialization with custom values."""
        extractor = RegionVisualExtractor(
            api_key="custom-key",
            model="gpt-4-turbo",
            temperature=0.5,
            max_tokens=2048,
        )

        assert extractor.api_key == "custom-key"
        assert extractor.model == "gpt-4-turbo"
        assert extractor.temperature == 0.5
        assert extractor.max_tokens == 2048

    def test_llm_property_raises_without_key(self) -> None:
        """Test that accessing llm property without API key raises error."""
        extractor = RegionVisualExtractor(api_key="")

        with pytest.raises(RegionVisualExtractionError) as exc_info:
            _ = extractor.llm

        assert "API key not configured" in str(exc_info.value)


class TestRegionVisualExtractorImageEncoding:
    """Tests for image encoding functionality."""

    def test_encode_rgb_image(self, extractor: RegionVisualExtractor) -> None:
        """Test encoding an RGB image to base64."""
        image = create_test_image(50, 50, "red")

        encoded = extractor._encode_image_to_base64(image)

        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        # Verify it can be loaded as an image
        loaded = Image.open(io.BytesIO(decoded))
        assert loaded.size == (50, 50)

    def test_encode_rgba_image(self, extractor: RegionVisualExtractor) -> None:
        """Test encoding an RGBA image to base64."""
        image = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))

        encoded = extractor._encode_image_to_base64(image)

        decoded = base64.b64decode(encoded)
        loaded = Image.open(io.BytesIO(decoded))
        # Should be converted to RGB
        assert loaded.mode == "RGB"


class TestRegionVisualExtractorSystemPrompts:
    """Tests for system prompt selection."""

    def test_get_system_prompt_for_text(self, extractor: RegionVisualExtractor) -> None:
        """Test system prompt selection for text strategy."""
        prompt = extractor._get_system_prompt(ExtractionStrategy.OCR_ENHANCED)
        assert "text from document images" in prompt.lower()

    def test_get_system_prompt_for_table(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test system prompt selection for table strategy."""
        prompt = extractor._get_system_prompt(ExtractionStrategy.TABLE_SPECIALIZED)
        assert "table" in prompt.lower()
        assert "headers" in prompt.lower()

    def test_get_system_prompt_for_figure(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test system prompt selection for figure strategy."""
        prompt = extractor._get_system_prompt(ExtractionStrategy.FIGURE_ANALYSIS)
        assert "chart" in prompt.lower() or "graph" in prompt.lower()

    def test_get_system_prompt_for_vlm_only(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test system prompt selection for VLM-only strategy."""
        prompt = extractor._get_system_prompt(ExtractionStrategy.VLM_ONLY)
        assert "document analyst" in prompt.lower()


class TestRegionVisualExtractorStrategySelection:
    """Tests for extraction strategy selection."""

    def test_get_strategy_for_text_region(
        self, extractor: RegionVisualExtractor, sample_region: LayoutRegion
    ) -> None:
        """Test strategy selection for text region."""
        strategy = extractor._get_extraction_strategy(sample_region)
        assert strategy == ExtractionStrategy.OCR_ENHANCED

    def test_get_strategy_for_table_region(
        self, extractor: RegionVisualExtractor, sample_table_region: LayoutRegion
    ) -> None:
        """Test strategy selection for table region."""
        strategy = extractor._get_extraction_strategy(sample_table_region)
        assert strategy == ExtractionStrategy.TABLE_SPECIALIZED

    def test_get_strategy_for_figure_region(
        self, extractor: RegionVisualExtractor, sample_figure_region: LayoutRegion
    ) -> None:
        """Test strategy selection for figure region."""
        strategy = extractor._get_extraction_strategy(sample_figure_region)
        assert strategy == ExtractionStrategy.FIGURE_ANALYSIS


class TestRegionVisualExtractorContextBuilding:
    """Tests for context building functionality."""

    def test_build_context_basic(
        self, extractor: RegionVisualExtractor, sample_region: LayoutRegion
    ) -> None:
        """Test building basic context without ordered regions."""
        context = extractor._build_context(
            region=sample_region,
            page_width=1000.0,
            page_height=1200.0,
        )

        assert context.region_type == RegionType.TEXT
        assert context.page_number == 1
        assert context.page_dimensions == (1000.0, 1200.0)
        # Check normalized bbox
        assert context.bbox_normalized[0] == 0.1  # 100/1000
        assert context.bbox_normalized[1] == pytest.approx(0.0833, rel=0.01)  # 100/1200
        assert context.reading_order_position is None

    def test_build_context_with_ordered_regions(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test building context with ordered regions."""
        regions = [
            create_region("r1", RegionType.TITLE, 100, 50, 900, 100),
            create_region("r2", RegionType.TEXT, 100, 150, 900, 300),
            create_region("r3", RegionType.TABLE, 100, 350, 900, 550),
        ]

        ordered_regions = [
            create_ordered_region(regions[0], 0),
            create_ordered_region(regions[1], 1),
            create_ordered_region(regions[2], 2),
        ]

        context = extractor._build_context(
            region=regions[1],  # The text region
            page_width=1000.0,
            page_height=1200.0,
            ordered_regions=ordered_regions,
        )

        assert context.reading_order_position == 1
        assert context.total_regions_on_page == 3
        assert "title" in context.preceding_region_types
        assert "table" in context.following_region_types

    def test_build_context_with_ocr_text(
        self, extractor: RegionVisualExtractor, sample_region: LayoutRegion
    ) -> None:
        """Test building context with OCR text."""
        context = extractor._build_context(
            region=sample_region,
            page_width=1000.0,
            page_height=1200.0,
            ocr_text="Sample OCR extracted text",
        )

        assert context.ocr_text == "Sample OCR extracted text"

    def test_build_context_with_parent_region(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test building context with parent region."""
        parent = create_region("r_parent", RegionType.PICTURE, 100, 100, 500, 500)
        child = create_region(
            "r_child",
            RegionType.CAPTION,
            120,
            510,
            480,
            550,
            parent_region_id="r_parent",
        )

        context = extractor._build_context(
            region=child,
            page_width=1000.0,
            page_height=1200.0,
            parent_region=parent,
        )

        assert context.parent_region_type == "picture"


class TestRegionVisualExtractorUserPrompt:
    """Tests for user prompt building."""

    def test_build_user_prompt_basic(self, extractor: RegionVisualExtractor) -> None:
        """Test building a basic user prompt."""
        context = RegionContext(
            region_type=RegionType.TEXT,
            page_number=1,
            page_dimensions=(1000.0, 1200.0),
            bbox_normalized=(0.1, 0.1, 0.5, 0.3),
            reading_order_position=2,
            total_regions_on_page=5,
            preceding_region_types=["title"],
            following_region_types=["table"],
        )

        prompt = extractor._build_user_prompt(context)

        assert "Region Type: text" in prompt
        assert "Page Number: 1" in prompt
        assert "Reading Order Position: 3 of 5" in prompt  # 0-indexed + 1
        assert "Preceded by: title" in prompt
        assert "Followed by: table" in prompt

    def test_build_user_prompt_with_ocr_text(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test building user prompt with OCR text."""
        context = RegionContext(
            region_type=RegionType.TEXT,
            page_number=1,
            page_dimensions=(1000.0, 1200.0),
            bbox_normalized=(0.1, 0.1, 0.5, 0.3),
            ocr_text="This is the OCR text.",
        )

        prompt = extractor._build_user_prompt(context)

        assert "OCR TEXT (for reference):" in prompt
        assert "This is the OCR text." in prompt


class TestRegionVisualExtractorResponseParsing:
    """Tests for response parsing functionality."""

    def test_parse_response_valid_json(self, extractor: RegionVisualExtractor) -> None:
        """Test parsing valid JSON response."""
        response = json.dumps(
            {
                "extracted_content": {"text": "Hello"},
                "confidence": 0.9,
                "reasoning": "Clear text",
            }
        )

        content, confidence, reasoning = extractor._parse_response(response, "r1")

        assert content == {"text": "Hello"}
        assert confidence == 0.9
        assert reasoning == "Clear text"

    def test_parse_response_with_extra_text(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test parsing response with extra text before JSON."""
        response = 'Here is the result: {"extracted_content": {"text": "Test"}, "confidence": 0.85}'

        content, confidence, _ = extractor._parse_response(response, "r1")

        assert content == {"text": "Test"}
        assert confidence == 0.85

    def test_parse_response_invalid_json(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test parsing invalid JSON response."""
        with pytest.raises(RegionVisualExtractionError) as exc_info:
            extractor._parse_response("not valid json at all", "r1")

        assert "Failed to parse VLM response" in str(exc_info.value)
        assert exc_info.value.region_id == "r1"

    def test_parse_response_clamps_confidence(
        self, extractor: RegionVisualExtractor
    ) -> None:
        """Test that confidence is clamped to valid range."""
        # Confidence > 1.0
        response = json.dumps(
            {
                "extracted_content": {},
                "confidence": 1.5,
            }
        )

        _, confidence, _ = extractor._parse_response(response, "r1")
        assert confidence == 1.0

        # Confidence < 0.0
        response = json.dumps(
            {
                "extracted_content": {},
                "confidence": -0.5,
            }
        )

        _, confidence, _ = extractor._parse_response(response, "r1")
        assert confidence == 0.0


class TestRegionVisualExtractorSingleExtraction:
    """Tests for single region extraction with mocked VLM."""

    def test_extract_from_text_region(
        self,
        extractor: RegionVisualExtractor,
        sample_region: LayoutRegion,
        sample_image: Image.Image,
        mock_vlm_response_text: str,
    ) -> None:
        """Test extracting from a text region."""
        mock_response = MagicMock()
        mock_response.content = mock_vlm_response_text
        mock_response.usage_metadata = {
            "input_tokens": 150,
            "output_tokens": 50,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        result = extractor.extract_from_region(
            region=sample_region,
            region_image=sample_image,
            page_width=1000.0,
            page_height=1200.0,
        )

        assert result.region_id == "r1"
        assert result.region_type == RegionType.TEXT
        assert result.confidence == 0.92
        assert result.strategy_used == ExtractionStrategy.OCR_ENHANCED
        assert result.extracted_content["text"] == "This is sample extracted text."
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 50

    def test_extract_from_table_region(
        self,
        extractor: RegionVisualExtractor,
        sample_table_region: LayoutRegion,
        sample_image: Image.Image,
        mock_vlm_response_table: str,
    ) -> None:
        """Test extracting from a table region."""
        mock_response = MagicMock()
        mock_response.content = mock_vlm_response_table
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        result = extractor.extract_from_region(
            region=sample_table_region,
            region_image=sample_image,
            page_width=1000.0,
            page_height=1200.0,
        )

        assert result.region_id == "r_table"
        assert result.region_type == RegionType.TABLE
        assert result.strategy_used == ExtractionStrategy.TABLE_SPECIALIZED
        assert result.extracted_content["headers"] == ["Name", "Age", "City"]
        assert len(result.extracted_content["rows"]) == 2

    def test_extract_from_figure_region(
        self,
        extractor: RegionVisualExtractor,
        sample_figure_region: LayoutRegion,
        sample_image: Image.Image,
        mock_vlm_response_figure: str,
    ) -> None:
        """Test extracting from a figure region."""
        mock_response = MagicMock()
        mock_response.content = mock_vlm_response_figure
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        result = extractor.extract_from_region(
            region=sample_figure_region,
            region_image=sample_image,
            page_width=1000.0,
            page_height=1200.0,
        )

        assert result.region_id == "r_figure"
        assert result.strategy_used == ExtractionStrategy.FIGURE_ANALYSIS
        assert result.extracted_content["visual_type"] == "chart"
        assert "Q1" in result.extracted_content["labels"]

    def test_extract_with_context(
        self,
        extractor: RegionVisualExtractor,
        sample_region: LayoutRegion,
        sample_image: Image.Image,
        mock_vlm_response_text: str,
    ) -> None:
        """Test extraction with ordered regions context."""
        regions = [
            create_region("r0", RegionType.TITLE, 100, 50, 900, 90),
            sample_region,
            create_region("r2", RegionType.TABLE, 100, 350, 900, 550),
        ]

        ordered_regions = [
            create_ordered_region(regions[0], 0),
            create_ordered_region(regions[1], 1),
            create_ordered_region(regions[2], 2),
        ]

        mock_response = MagicMock()
        mock_response.content = mock_vlm_response_text
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        extractor.extract_from_region(
            region=sample_region,
            region_image=sample_image,
            page_width=1000.0,
            page_height=1200.0,
            ordered_regions=ordered_regions,
            ocr_text="Sample OCR text",
        )

        # Verify the invoke was called
        mock_llm.invoke.assert_called_once()
        call_messages = mock_llm.invoke.call_args[0][0]

        # Check that context is in the user message
        user_message = call_messages[1]
        assert "title" in str(user_message.content).lower()  # Preceding region
        assert "table" in str(user_message.content).lower()  # Following region
        assert "OCR TEXT" in str(user_message.content)  # OCR text section

    def test_extract_with_schema(
        self,
        extractor: RegionVisualExtractor,
        sample_region: LayoutRegion,
        sample_image: Image.Image,
        mock_vlm_response_text: str,
    ) -> None:
        """Test extraction with custom schema."""
        schema = {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "total_amount": {"type": "number"},
            },
        }

        mock_response = MagicMock()
        mock_response.content = mock_vlm_response_text
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        extractor.extract_from_region(
            region=sample_region,
            region_image=sample_image,
            page_width=1000.0,
            page_height=1200.0,
            schema=schema,
        )

        # Verify schema is included in the system prompt
        call_messages = mock_llm.invoke.call_args[0][0]
        system_message = call_messages[0]
        assert "invoice_number" in str(system_message.content)

    def test_extract_handles_llm_error(
        self,
        extractor: RegionVisualExtractor,
        sample_region: LayoutRegion,
        sample_image: Image.Image,
    ) -> None:
        """Test error handling when LLM call fails."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API Error")
        extractor._llm = mock_llm

        with pytest.raises(RegionVisualExtractionError) as exc_info:
            extractor.extract_from_region(
                region=sample_region,
                region_image=sample_image,
                page_width=1000.0,
                page_height=1200.0,
            )

        assert "VLM extraction failed" in str(exc_info.value)
        assert exc_info.value.region_id == "r1"


class TestRegionVisualExtractorMultiExtraction:
    """Tests for multi-region extraction."""

    def test_extract_from_multiple_regions(
        self,
        extractor: RegionVisualExtractor,
        mock_vlm_response_text: str,
        mock_vlm_response_table: str,
    ) -> None:
        """Test extracting from multiple regions."""
        regions = [
            create_region("r1", RegionType.TEXT, 100, 100, 500, 200),
            create_region("r2", RegionType.TABLE, 100, 250, 800, 500),
        ]

        region_images = {
            "r1": create_test_image(400, 100),
            "r2": create_test_image(700, 250),
        }

        mock_response_text = MagicMock()
        mock_response_text.content = mock_vlm_response_text
        mock_response_text.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

        mock_response_table = MagicMock()
        mock_response_table.content = mock_vlm_response_table
        mock_response_table.usage_metadata = {"input_tokens": 150, "output_tokens": 75}

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [mock_response_text, mock_response_table]
        extractor._llm = mock_llm

        result = extractor.extract_from_regions(
            regions=regions,
            region_images=region_images,
            page_width=1000.0,
            page_height=1200.0,
        )

        assert result.total_regions == 2
        assert result.successful_extractions == 2
        assert result.failed_extractions == 0
        assert result.total_tokens == 375  # 100+50 + 150+75
        assert len(result.region_results) == 2

    def test_extract_continues_on_error(
        self,
        extractor: RegionVisualExtractor,
        mock_vlm_response_text: str,
    ) -> None:
        """Test that extraction continues when one region fails."""
        regions = [
            create_region("r1", RegionType.TEXT, 100, 100, 500, 200),
            create_region("r2", RegionType.TEXT, 100, 250, 500, 400),
        ]

        region_images = {
            "r1": create_test_image(400, 100),
            "r2": create_test_image(400, 150),
        }

        mock_response_success = MagicMock()
        mock_response_success.content = mock_vlm_response_text
        mock_response_success.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            Exception("API Error"),  # First call fails
            mock_response_success,  # Second call succeeds
        ]
        extractor._llm = mock_llm

        result = extractor.extract_from_regions(
            regions=regions,
            region_images=region_images,
            page_width=1000.0,
            page_height=1200.0,
            continue_on_error=True,
        )

        assert result.total_regions == 2
        assert result.successful_extractions == 1
        assert result.failed_extractions == 1

        # Check the failed result
        failed_result = result.get_result_by_id("r1")
        assert failed_result is not None
        assert failed_result.confidence == 0.0
        assert "failed" in (failed_result.reasoning or "").lower()

    def test_extract_stops_on_error_when_configured(
        self,
        extractor: RegionVisualExtractor,
    ) -> None:
        """Test that extraction stops on error when continue_on_error=False."""
        regions = [
            create_region("r1", RegionType.TEXT, 100, 100, 500, 200),
            create_region("r2", RegionType.TEXT, 100, 250, 500, 400),
        ]

        region_images = {
            "r1": create_test_image(400, 100),
            "r2": create_test_image(400, 150),
        }

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API Error")
        extractor._llm = mock_llm

        with pytest.raises(RegionVisualExtractionError):
            extractor.extract_from_regions(
                regions=regions,
                region_images=region_images,
                page_width=1000.0,
                page_height=1200.0,
                continue_on_error=False,
            )

    def test_extract_handles_missing_image(
        self,
        extractor: RegionVisualExtractor,
        mock_vlm_response_text: str,
    ) -> None:
        """Test handling when region image is missing."""
        regions = [
            create_region("r1", RegionType.TEXT, 100, 100, 500, 200),
            create_region("r2", RegionType.TEXT, 100, 250, 500, 400),
        ]

        # Only provide image for r1
        region_images = {
            "r1": create_test_image(400, 100),
            # r2 is missing
        }

        mock_response = MagicMock()
        mock_response.content = mock_vlm_response_text
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        result = extractor.extract_from_regions(
            regions=regions,
            region_images=region_images,
            page_width=1000.0,
            page_height=1200.0,
        )

        assert result.successful_extractions == 1
        assert result.failed_extractions == 1


class TestRegionVisualExtractorPageExtraction:
    """Tests for full page extraction."""

    def test_extract_from_page(
        self,
        extractor: RegionVisualExtractor,
        page_image: Image.Image,
        mock_vlm_response_text: str,
        mock_vlm_response_table: str,
    ) -> None:
        """Test extracting all regions from a page."""
        regions = [
            create_region("r1", RegionType.TEXT, 100, 100, 500, 200),
            create_region("r2", RegionType.TABLE, 100, 250, 800, 500),
        ]

        mock_response_text = MagicMock()
        mock_response_text.content = mock_vlm_response_text
        mock_response_text.usage_metadata = {}

        mock_response_table = MagicMock()
        mock_response_table.content = mock_vlm_response_table
        mock_response_table.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [mock_response_text, mock_response_table]
        extractor._llm = mock_llm

        result = extractor.extract_from_page(
            page_image=page_image,
            regions=regions,
            page_width=1000.0,
            page_height=1200.0,
        )

        assert result.total_regions == 2
        assert result.successful_extractions == 2

    def test_extract_from_page_with_nested_regions(
        self,
        extractor: RegionVisualExtractor,
        page_image: Image.Image,
        mock_vlm_response_figure: str,
        mock_vlm_response_text: str,
    ) -> None:
        """Test extracting with parent-child region relationships."""
        parent = create_region("r_parent", RegionType.PICTURE, 100, 100, 600, 500)
        child = create_region(
            "r_child",
            RegionType.CAPTION,
            120,
            510,
            580,
            550,
            parent_region_id="r_parent",
        )
        regions = [parent, child]

        mock_response_figure = MagicMock()
        mock_response_figure.content = mock_vlm_response_figure
        mock_response_figure.usage_metadata = {}

        mock_response_caption = MagicMock()
        mock_response_caption.content = mock_vlm_response_text
        mock_response_caption.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [mock_response_figure, mock_response_caption]
        extractor._llm = mock_llm

        result = extractor.extract_from_page(
            page_image=page_image,
            regions=regions,
            page_width=1000.0,
            page_height=1200.0,
        )

        assert result.total_regions == 2
        assert result.successful_extractions == 2


class TestRegionVisualExtractorStaticMethods:
    """Tests for static methods."""

    def test_get_supported_strategies(self) -> None:
        """Test getting supported strategies."""
        strategies = RegionVisualExtractor.get_supported_strategies()

        assert "vlm_only" in strategies
        assert "ocr_enhanced" in strategies
        assert "table_specialized" in strategies
        assert "figure_analysis" in strategies
        assert len(strategies) == 4

    def test_get_strategy_for_region_type(self) -> None:
        """Test getting strategy for specific region types."""
        assert (
            RegionVisualExtractor.get_strategy_for_region_type(RegionType.TEXT)
            == ExtractionStrategy.OCR_ENHANCED
        )
        assert (
            RegionVisualExtractor.get_strategy_for_region_type(RegionType.TABLE)
            == ExtractionStrategy.TABLE_SPECIALIZED
        )
        assert (
            RegionVisualExtractor.get_strategy_for_region_type(RegionType.PICTURE)
            == ExtractionStrategy.FIGURE_ANALYSIS
        )
        assert (
            RegionVisualExtractor.get_strategy_for_region_type(RegionType.UNKNOWN)
            == ExtractionStrategy.VLM_ONLY
        )


class TestRegionVisualExtractorIntegration:
    """Integration tests for the visual extraction service."""

    def test_full_extraction_pipeline(
        self,
        extractor: RegionVisualExtractor,
        page_image: Image.Image,
        mock_vlm_response_text: str,
        mock_vlm_response_table: str,
        mock_vlm_response_figure: str,
    ) -> None:
        """Test a complete extraction pipeline with various region types."""
        # Create regions of different types
        regions = [
            create_region("r_title", RegionType.TITLE, 100, 50, 900, 100),
            create_region("r_text", RegionType.TEXT, 100, 150, 600, 300),
            create_region("r_table", RegionType.TABLE, 100, 350, 800, 600),
            create_region("r_figure", RegionType.PICTURE, 100, 650, 500, 900),
        ]

        # Create ordered regions
        ordered_regions = [
            create_ordered_region(regions[0], 0),
            create_ordered_region(regions[1], 1),
            create_ordered_region(regions[2], 2),
            create_ordered_region(regions[3], 3),
        ]

        # Mock responses
        mock_responses = [
            MagicMock(content=mock_vlm_response_text, usage_metadata={}),
            MagicMock(content=mock_vlm_response_text, usage_metadata={}),
            MagicMock(content=mock_vlm_response_table, usage_metadata={}),
            MagicMock(content=mock_vlm_response_figure, usage_metadata={}),
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_responses
        extractor._llm = mock_llm

        result = extractor.extract_from_page(
            page_image=page_image,
            regions=regions,
            page_width=1000.0,
            page_height=1200.0,
            ordered_regions=ordered_regions,
        )

        assert result.total_regions == 4
        assert result.successful_extractions == 4
        assert result.failed_extractions == 0
        assert result.model_used == "gpt-4o"

        # Verify each region type was processed with correct strategy
        title_result = result.get_result_by_id("r_title")
        assert title_result is not None
        assert title_result.strategy_used == ExtractionStrategy.OCR_ENHANCED

        table_result = result.get_result_by_id("r_table")
        assert table_result is not None
        assert table_result.strategy_used == ExtractionStrategy.TABLE_SPECIALIZED

        figure_result = result.get_result_by_id("r_figure")
        assert figure_result is not None
        assert figure_result.strategy_used == ExtractionStrategy.FIGURE_ANALYSIS

    def test_extraction_with_ocr_text_context(
        self,
        extractor: RegionVisualExtractor,
        mock_vlm_response_text: str,
    ) -> None:
        """Test extraction with OCR text provided as context."""
        region = create_region("r1", RegionType.TEXT, 100, 100, 500, 200)
        region_image = create_test_image(400, 100)

        ocr_text = "Invoice Number: INV-2024-001\nDate: January 15, 2024"

        mock_response = MagicMock()
        mock_response.content = mock_vlm_response_text
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm

        result = extractor.extract_from_region(
            region=region,
            region_image=region_image,
            page_width=1000.0,
            page_height=1200.0,
            ocr_text=ocr_text,
        )

        # Verify OCR text was included in the prompt
        call_messages = mock_llm.invoke.call_args[0][0]
        user_message_content = str(call_messages[1].content)
        assert "Invoice Number" in user_message_content
        assert "INV-2024-001" in user_message_content

        assert result.confidence > 0
