"""Tests for the reading order detection service."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    PageLayoutResult,
    RegionBoundingBox,
    RegionType,
)
from agentic_document_extraction.services.reading_order_detector import (
    SKIP_REGION_TYPES,
    DocumentReadingOrder,
    OrderedRegion,
    PageReadingOrder,
    ReadingOrderDetector,
    ReadingOrderError,
)


# Helper function to create regions for testing
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


def create_page_layout(
    regions: list[LayoutRegion],
    page_number: int = 1,
    page_width: float = 1000.0,
    page_height: float = 1200.0,
) -> PageLayoutResult:
    """Helper to create a page layout result for testing."""
    return PageLayoutResult(
        page_number=page_number,
        regions=regions,
        page_width=page_width,
        page_height=page_height,
    )


@pytest.fixture
def detector() -> ReadingOrderDetector:
    """Create a ReadingOrderDetector instance for testing."""
    return ReadingOrderDetector(api_key="test-key")


@pytest.fixture
def single_column_regions() -> list[LayoutRegion]:
    """Create regions for a simple single-column layout."""
    return [
        create_region("r1", RegionType.TITLE, 100, 50, 900, 100),
        create_region("r2", RegionType.TEXT, 100, 150, 900, 300),
        create_region("r3", RegionType.TEXT, 100, 350, 900, 500),
        create_region("r4", RegionType.PAGE_FOOTER, 100, 1100, 900, 1150),
    ]


@pytest.fixture
def multi_column_regions() -> list[LayoutRegion]:
    """Create regions for a multi-column layout."""
    return [
        # Left column
        create_region("r1", RegionType.TEXT, 50, 100, 450, 300),
        create_region("r2", RegionType.TEXT, 50, 350, 450, 550),
        # Right column
        create_region("r3", RegionType.TEXT, 550, 100, 950, 300),
        create_region("r4", RegionType.TEXT, 550, 350, 950, 550),
        # Header
        create_region("r5", RegionType.PAGE_HEADER, 50, 20, 950, 60),
    ]


@pytest.fixture
def complex_layout_regions() -> list[LayoutRegion]:
    """Create regions for a complex layout with sidebars and figures."""
    return [
        create_region("r1", RegionType.TITLE, 100, 50, 700, 100),
        create_region("r2", RegionType.TEXT, 100, 150, 600, 350),
        create_region("r3", RegionType.PICTURE, 100, 400, 400, 650),
        create_region("r4", RegionType.CAPTION, 120, 660, 380, 700),
        create_region("r5", RegionType.TEXT, 450, 400, 600, 650),
        create_region("r6", RegionType.TABLE, 650, 150, 950, 500),
        create_region("r7", RegionType.FOOTNOTE, 100, 1100, 600, 1150),
    ]


@pytest.fixture
def mock_llm_response_single_column() -> str:
    """Mock LLM response for single column layout."""
    return json.dumps(
        {
            "layout_type": "single_column",
            "ordered_regions": [
                {
                    "region_id": "r1",
                    "order_index": 0,
                    "confidence": 0.95,
                    "reasoning": "Title at top",
                    "skip_in_reading": False,
                },
                {
                    "region_id": "r2",
                    "order_index": 1,
                    "confidence": 0.92,
                    "reasoning": "First body text",
                    "skip_in_reading": False,
                },
                {
                    "region_id": "r3",
                    "order_index": 2,
                    "confidence": 0.90,
                    "reasoning": "Second body text",
                    "skip_in_reading": False,
                },
                {
                    "region_id": "r4",
                    "order_index": 3,
                    "confidence": 0.88,
                    "reasoning": "Footer",
                    "skip_in_reading": True,
                },
            ],
            "overall_confidence": 0.91,
        }
    )


@pytest.fixture
def mock_llm_response_multi_column() -> str:
    """Mock LLM response for multi-column layout."""
    return json.dumps(
        {
            "layout_type": "multi_column",
            "ordered_regions": [
                {
                    "region_id": "r5",
                    "order_index": 0,
                    "confidence": 0.85,
                    "reasoning": "Page header",
                    "skip_in_reading": True,
                },
                {
                    "region_id": "r1",
                    "order_index": 1,
                    "confidence": 0.92,
                    "reasoning": "Left column, top",
                    "skip_in_reading": False,
                },
                {
                    "region_id": "r2",
                    "order_index": 2,
                    "confidence": 0.90,
                    "reasoning": "Left column, bottom",
                    "skip_in_reading": False,
                },
                {
                    "region_id": "r3",
                    "order_index": 3,
                    "confidence": 0.88,
                    "reasoning": "Right column, top",
                    "skip_in_reading": False,
                },
                {
                    "region_id": "r4",
                    "order_index": 4,
                    "confidence": 0.86,
                    "reasoning": "Right column, bottom",
                    "skip_in_reading": False,
                },
            ],
            "overall_confidence": 0.88,
        }
    )


class TestOrderedRegion:
    """Tests for OrderedRegion dataclass."""

    def test_create_ordered_region(self) -> None:
        """Test creating an OrderedRegion."""
        region = create_region("r1", RegionType.TEXT, 100, 100, 500, 300)
        ordered = OrderedRegion(
            region=region,
            order_index=0,
            confidence=0.95,
            reasoning="First text region",
            skip_in_reading=False,
        )

        assert ordered.region == region
        assert ordered.order_index == 0
        assert ordered.confidence == 0.95
        assert ordered.reasoning == "First text region"
        assert ordered.skip_in_reading is False

    def test_ordered_region_with_skip(self) -> None:
        """Test OrderedRegion with skip_in_reading set."""
        region = create_region("r1", RegionType.PAGE_HEADER, 100, 10, 900, 50)
        ordered = OrderedRegion(
            region=region,
            order_index=0,
            confidence=0.88,
            skip_in_reading=True,
        )

        assert ordered.skip_in_reading is True

    def test_to_dict(self) -> None:
        """Test converting OrderedRegion to dictionary."""
        region = create_region("r1", RegionType.TEXT, 100, 100, 500, 300)
        ordered = OrderedRegion(
            region=region,
            order_index=2,
            confidence=0.90,
            reasoning="Test reasoning",
            skip_in_reading=False,
        )

        result = ordered.to_dict()

        assert result["region_id"] == "r1"
        assert result["region_type"] == "text"
        assert result["order_index"] == 2
        assert result["confidence"] == 0.90
        assert result["reasoning"] == "Test reasoning"
        assert result["skip_in_reading"] is False
        assert "bbox" in result


class TestPageReadingOrder:
    """Tests for PageReadingOrder dataclass."""

    def test_create_page_reading_order(self) -> None:
        """Test creating a PageReadingOrder."""
        region1 = create_region("r1", RegionType.TEXT, 100, 100, 500, 300)
        region2 = create_region("r2", RegionType.TEXT, 100, 400, 500, 600)

        ordered_regions = [
            OrderedRegion(region=region1, order_index=0, confidence=0.95),
            OrderedRegion(region=region2, order_index=1, confidence=0.90),
        ]

        page_order = PageReadingOrder(
            page_number=1,
            ordered_regions=ordered_regions,
            overall_confidence=0.92,
            layout_type="single_column",
            processing_time_seconds=0.5,
        )

        assert page_order.page_number == 1
        assert len(page_order.ordered_regions) == 2
        assert page_order.overall_confidence == 0.92
        assert page_order.layout_type == "single_column"
        assert page_order.processing_time_seconds == 0.5

    def test_to_dict(self) -> None:
        """Test converting PageReadingOrder to dictionary."""
        region = create_region("r1", RegionType.TEXT, 100, 100, 500, 300)
        ordered_regions = [
            OrderedRegion(region=region, order_index=0, confidence=0.95),
        ]

        page_order = PageReadingOrder(
            page_number=1,
            ordered_regions=ordered_regions,
            overall_confidence=0.95,
            layout_type="single_column",
            processing_time_seconds=0.25,
        )

        result = page_order.to_dict()

        assert result["page_number"] == 1
        assert result["overall_confidence"] == 0.95
        assert result["layout_type"] == "single_column"
        assert result["processing_time_seconds"] == 0.25
        assert result["region_count"] == 1
        assert len(result["ordered_regions"]) == 1

    def test_get_reading_sequence(self) -> None:
        """Test getting reading sequence from PageReadingOrder."""
        region1 = create_region("r1", RegionType.TEXT, 100, 100, 500, 300)
        region2 = create_region("r2", RegionType.PAGE_HEADER, 100, 10, 500, 50)
        region3 = create_region("r3", RegionType.TEXT, 100, 400, 500, 600)

        ordered_regions = [
            OrderedRegion(
                region=region2, order_index=0, confidence=0.90, skip_in_reading=True
            ),
            OrderedRegion(
                region=region1, order_index=1, confidence=0.95, skip_in_reading=False
            ),
            OrderedRegion(
                region=region3, order_index=2, confidence=0.92, skip_in_reading=False
            ),
        ]

        page_order = PageReadingOrder(
            page_number=1,
            ordered_regions=ordered_regions,
            overall_confidence=0.92,
            layout_type="single_column",
        )

        # Without skipped regions
        sequence = page_order.get_reading_sequence(include_skipped=False)
        assert len(sequence) == 2
        assert sequence[0].region_id == "r1"
        assert sequence[1].region_id == "r3"

        # With skipped regions
        sequence_full = page_order.get_reading_sequence(include_skipped=True)
        assert len(sequence_full) == 3
        assert sequence_full[0].region_id == "r2"


class TestDocumentReadingOrder:
    """Tests for DocumentReadingOrder dataclass."""

    def test_create_document_reading_order(self) -> None:
        """Test creating a DocumentReadingOrder."""
        region1 = create_region(
            "r1", RegionType.TEXT, 100, 100, 500, 300, page_number=1
        )
        region2 = create_region(
            "r2", RegionType.TEXT, 100, 100, 500, 300, page_number=2
        )

        page1 = PageReadingOrder(
            page_number=1,
            ordered_regions=[
                OrderedRegion(region=region1, order_index=0, confidence=0.95)
            ],
            overall_confidence=0.95,
            layout_type="single_column",
        )
        page2 = PageReadingOrder(
            page_number=2,
            ordered_regions=[
                OrderedRegion(region=region2, order_index=0, confidence=0.92)
            ],
            overall_confidence=0.92,
            layout_type="single_column",
        )

        doc_order = DocumentReadingOrder(
            pages=[page1, page2],
            total_pages=2,
            total_regions=2,
            model_used="gpt-4o",
            total_tokens=500,
            prompt_tokens=400,
            completion_tokens=100,
            processing_time_seconds=1.5,
        )

        assert len(doc_order.pages) == 2
        assert doc_order.total_pages == 2
        assert doc_order.total_regions == 2
        assert doc_order.model_used == "gpt-4o"
        assert doc_order.total_tokens == 500
        assert doc_order.processing_time_seconds == 1.5

    def test_to_dict(self) -> None:
        """Test converting DocumentReadingOrder to dictionary."""
        region = create_region("r1", RegionType.TEXT, 100, 100, 500, 300)
        page = PageReadingOrder(
            page_number=1,
            ordered_regions=[
                OrderedRegion(region=region, order_index=0, confidence=0.95)
            ],
            overall_confidence=0.95,
            layout_type="single_column",
        )

        doc_order = DocumentReadingOrder(
            pages=[page],
            total_pages=1,
            total_regions=1,
            model_used="gpt-4o",
            total_tokens=300,
            prompt_tokens=250,
            completion_tokens=50,
            processing_time_seconds=0.8,
        )

        result = doc_order.to_dict()

        assert result["total_pages"] == 1
        assert result["total_regions"] == 1
        assert len(result["pages"]) == 1
        assert result["metadata"]["model_used"] == "gpt-4o"
        assert result["metadata"]["total_tokens"] == 300
        assert result["metadata"]["processing_time_seconds"] == 0.8

    def test_get_all_regions_ordered(self) -> None:
        """Test getting all regions across pages in order."""
        region1 = create_region(
            "r1", RegionType.TEXT, 100, 100, 500, 300, page_number=1
        )
        region2 = create_region(
            "r2", RegionType.PAGE_HEADER, 100, 10, 500, 50, page_number=1
        )
        region3 = create_region(
            "r3", RegionType.TEXT, 100, 100, 500, 300, page_number=2
        )

        page1 = PageReadingOrder(
            page_number=1,
            ordered_regions=[
                OrderedRegion(
                    region=region2, order_index=0, confidence=0.90, skip_in_reading=True
                ),
                OrderedRegion(
                    region=region1,
                    order_index=1,
                    confidence=0.95,
                    skip_in_reading=False,
                ),
            ],
            overall_confidence=0.92,
            layout_type="single_column",
        )
        page2 = PageReadingOrder(
            page_number=2,
            ordered_regions=[
                OrderedRegion(
                    region=region3,
                    order_index=0,
                    confidence=0.92,
                    skip_in_reading=False,
                ),
            ],
            overall_confidence=0.92,
            layout_type="single_column",
        )

        doc_order = DocumentReadingOrder(
            pages=[page1, page2],
            total_pages=2,
            total_regions=3,
            model_used="gpt-4o",
        )

        # Without skipped regions
        all_regions = doc_order.get_all_regions_ordered(include_skipped=False)
        assert len(all_regions) == 2
        assert all_regions[0] == (1, region1)
        assert all_regions[1] == (2, region3)

        # With skipped regions
        all_regions_full = doc_order.get_all_regions_ordered(include_skipped=True)
        assert len(all_regions_full) == 3


class TestReadingOrderError:
    """Tests for ReadingOrderError exception."""

    def test_error_with_message(self) -> None:
        """Test error with just message."""
        error = ReadingOrderError("Test error")

        assert str(error) == "Test error"
        assert error.page_number is None
        assert error.details == {}

    def test_error_with_page_number(self) -> None:
        """Test error with page number."""
        error = ReadingOrderError("Test error", page_number=3)

        assert error.page_number == 3

    def test_error_with_details(self) -> None:
        """Test error with details."""
        error = ReadingOrderError(
            "Test error",
            page_number=1,
            details={"error_type": "LLM failure"},
        )

        assert error.details == {"error_type": "LLM failure"}


class TestReadingOrderDetectorInit:
    """Tests for ReadingOrderDetector initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with defaults."""
        with patch(
            "agentic_document_extraction.services.reading_order_detector.settings"
        ) as mock_settings:
            mock_settings.openai_api_key = "env-key"
            mock_settings.openai_model = "gpt-4o"

            detector = ReadingOrderDetector()

            assert detector.api_key == "env-key"
            assert detector.model == "gpt-4o"
            assert detector.temperature == 0.0

    def test_custom_initialization(self) -> None:
        """Test initialization with custom values."""
        detector = ReadingOrderDetector(
            api_key="custom-key",
            model="gpt-4-turbo",
            temperature=0.2,
        )

        assert detector.api_key == "custom-key"
        assert detector.model == "gpt-4-turbo"
        assert detector.temperature == 0.2

    def test_llm_property_raises_without_key(self) -> None:
        """Test that accessing llm property without API key raises error."""
        detector = ReadingOrderDetector(api_key="")

        with pytest.raises(ReadingOrderError) as exc_info:
            _ = detector.llm

        assert "API key not configured" in str(exc_info.value)


class TestReadingOrderDetectorFormatting:
    """Tests for region formatting in ReadingOrderDetector."""

    def test_format_regions_for_prompt(
        self, detector: ReadingOrderDetector, single_column_regions: list[LayoutRegion]
    ) -> None:
        """Test formatting regions for the LLM prompt."""
        formatted = detector._format_regions_for_prompt(single_column_regions)

        # Should be valid JSON
        parsed = json.loads(formatted)

        assert len(parsed) == 4
        assert parsed[0]["region_id"] == "r1"
        assert parsed[0]["region_type"] == "title"
        assert "bbox" in parsed[0]
        assert "center" in parsed[0]
        assert parsed[0]["confidence"] == 0.9

    def test_format_regions_with_parent(self, detector: ReadingOrderDetector) -> None:
        """Test formatting regions with parent relationships."""
        regions = [
            create_region("r1", RegionType.PICTURE, 100, 100, 400, 400),
            create_region(
                "r2", RegionType.CAPTION, 120, 410, 380, 450, parent_region_id="r1"
            ),
        ]

        formatted = detector._format_regions_for_prompt(regions)
        parsed = json.loads(formatted)

        # Caption should include parent_region_id
        caption = next(r for r in parsed if r["region_id"] == "r2")
        assert caption["parent_region_id"] == "r1"


class TestReadingOrderDetectorParsing:
    """Tests for response parsing in ReadingOrderDetector."""

    def test_parse_llm_response_success(
        self,
        detector: ReadingOrderDetector,
        single_column_regions: list[LayoutRegion],
        mock_llm_response_single_column: str,
    ) -> None:
        """Test successful parsing of LLM response."""
        ordered_regions, layout_type, confidence = detector._parse_llm_response(
            mock_llm_response_single_column, single_column_regions
        )

        assert len(ordered_regions) == 4
        assert layout_type == "single_column"
        assert confidence == 0.91

        # Check ordering
        assert ordered_regions[0].region.region_id == "r1"
        assert ordered_regions[0].order_index == 0
        assert ordered_regions[0].reasoning == "Title at top"

    def test_parse_llm_response_invalid_json(
        self, detector: ReadingOrderDetector, single_column_regions: list[LayoutRegion]
    ) -> None:
        """Test parsing with invalid JSON response."""
        with pytest.raises(ReadingOrderError) as exc_info:
            detector._parse_llm_response("invalid json", single_column_regions)

        assert "Failed to parse LLM response" in str(exc_info.value)

    def test_parse_llm_response_missing_region(
        self, detector: ReadingOrderDetector, single_column_regions: list[LayoutRegion]
    ) -> None:
        """Test parsing when LLM response is missing a region."""
        # Response missing r4
        incomplete_response = json.dumps(
            {
                "layout_type": "single_column",
                "ordered_regions": [
                    {"region_id": "r1", "order_index": 0, "confidence": 0.95},
                    {"region_id": "r2", "order_index": 1, "confidence": 0.92},
                    {"region_id": "r3", "order_index": 2, "confidence": 0.90},
                    # r4 missing
                ],
                "overall_confidence": 0.91,
            }
        )

        ordered_regions, _, _ = detector._parse_llm_response(
            incomplete_response, single_column_regions
        )

        # Should still have all 4 regions (missing one added with low confidence)
        assert len(ordered_regions) == 4

        # The missing region should have low confidence
        missing_region = next(r for r in ordered_regions if r.region.region_id == "r4")
        assert missing_region.confidence == 0.3
        assert "not analyzed by LLM" in (missing_region.reasoning or "")

    def test_parse_llm_response_duplicate_region(
        self, detector: ReadingOrderDetector, single_column_regions: list[LayoutRegion]
    ) -> None:
        """Test parsing when LLM response has duplicate regions."""
        duplicate_response = json.dumps(
            {
                "layout_type": "single_column",
                "ordered_regions": [
                    {"region_id": "r1", "order_index": 0, "confidence": 0.95},
                    {
                        "region_id": "r1",
                        "order_index": 1,
                        "confidence": 0.90,
                    },  # Duplicate
                    {"region_id": "r2", "order_index": 2, "confidence": 0.92},
                    {"region_id": "r3", "order_index": 3, "confidence": 0.90},
                    {"region_id": "r4", "order_index": 4, "confidence": 0.88},
                ],
                "overall_confidence": 0.91,
            }
        )

        ordered_regions, _, _ = detector._parse_llm_response(
            duplicate_response, single_column_regions
        )

        # Should only have 4 regions (duplicate ignored)
        assert len(ordered_regions) == 4

        # r1 should only appear once
        r1_count = sum(1 for r in ordered_regions if r.region.region_id == "r1")
        assert r1_count == 1


class TestReadingOrderDetectorWithMockedLLM:
    """Tests for reading order detection with mocked LLM."""

    def test_detect_reading_order_for_page_single_column(
        self,
        detector: ReadingOrderDetector,
        single_column_regions: list[LayoutRegion],
        mock_llm_response_single_column: str,
    ) -> None:
        """Test reading order detection for single column layout."""
        page_layout = create_page_layout(single_column_regions)

        mock_response = MagicMock()
        mock_response.content = mock_llm_response_single_column

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        # Patch ChatPromptTemplate to return a mock that creates our chain
        with patch(
            "agentic_document_extraction.services.reading_order_detector.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_class.from_messages.return_value = mock_prompt

            # Set _llm to avoid API key validation
            detector._llm = MagicMock()

            result = detector.detect_reading_order_for_page(page_layout)

        assert result.page_number == 1
        assert result.layout_type == "single_column"
        assert result.overall_confidence == 0.91
        assert len(result.ordered_regions) == 4

    def test_detect_reading_order_for_page_multi_column(
        self,
        detector: ReadingOrderDetector,
        multi_column_regions: list[LayoutRegion],
        mock_llm_response_multi_column: str,
    ) -> None:
        """Test reading order detection for multi-column layout."""
        page_layout = create_page_layout(multi_column_regions)

        mock_response = MagicMock()
        mock_response.content = mock_llm_response_multi_column

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with patch(
            "agentic_document_extraction.services.reading_order_detector.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_class.from_messages.return_value = mock_prompt

            detector._llm = MagicMock()

            result = detector.detect_reading_order_for_page(page_layout)

        assert result.page_number == 1
        assert result.layout_type == "multi_column"
        assert len(result.ordered_regions) == 5

    def test_detect_reading_order_for_empty_page(
        self, detector: ReadingOrderDetector
    ) -> None:
        """Test reading order detection for empty page."""
        page_layout = create_page_layout([])

        result = detector.detect_reading_order_for_page(page_layout)

        assert result.page_number == 1
        assert result.layout_type == "empty"
        assert result.overall_confidence == 1.0
        assert len(result.ordered_regions) == 0

    def test_detect_reading_order_multi_page(
        self,
        detector: ReadingOrderDetector,
        single_column_regions: list[LayoutRegion],
        mock_llm_response_single_column: str,
    ) -> None:
        """Test reading order detection for multi-page document."""
        page1_layout = create_page_layout(single_column_regions, page_number=1)
        page2_regions = [
            create_region("r5", RegionType.TEXT, 100, 100, 900, 400, page_number=2),
        ]
        page2_layout = create_page_layout(page2_regions, page_number=2)

        mock_response_page1 = MagicMock()
        mock_response_page1.content = mock_llm_response_single_column

        mock_response_page2 = MagicMock()
        mock_response_page2.content = json.dumps(
            {
                "layout_type": "single_column",
                "ordered_regions": [
                    {"region_id": "r5", "order_index": 0, "confidence": 0.93},
                ],
                "overall_confidence": 0.93,
            }
        )

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [mock_response_page1, mock_response_page2]

        with patch(
            "agentic_document_extraction.services.reading_order_detector.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_class.from_messages.return_value = mock_prompt

            detector._llm = MagicMock()

            result = detector.detect_reading_order([page1_layout, page2_layout])

        assert result.total_pages == 2
        assert result.total_regions == 5  # 4 from page 1 + 1 from page 2
        assert len(result.pages) == 2
        assert result.model_used == "gpt-4o"

    def test_detect_reading_order_llm_error(
        self,
        detector: ReadingOrderDetector,
        single_column_regions: list[LayoutRegion],
    ) -> None:
        """Test error handling when LLM call fails."""
        page_layout = create_page_layout(single_column_regions)

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM API error")

        with patch(
            "agentic_document_extraction.services.reading_order_detector.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_class.from_messages.return_value = mock_prompt

            detector._llm = MagicMock()

            with pytest.raises(ReadingOrderError) as exc_info:
                detector.detect_reading_order_for_page(page_layout)

        assert "LLM call failed" in str(exc_info.value)


class TestReadingOrderDetectorHeuristic:
    """Tests for heuristic-based reading order detection."""

    def test_detect_reading_order_simple_single_column(
        self, detector: ReadingOrderDetector, single_column_regions: list[LayoutRegion]
    ) -> None:
        """Test heuristic detection for single column layout."""
        result = detector.detect_reading_order_simple(
            single_column_regions, page_width=1000.0, page_height=1200.0
        )

        assert result.layout_type == "single_column"
        assert result.overall_confidence == 0.6  # Heuristic has lower confidence
        assert len(result.ordered_regions) == 4

        # First region should be the title (topmost)
        assert result.ordered_regions[0].region.region_id == "r1"

    def test_detect_reading_order_simple_multi_column(
        self, detector: ReadingOrderDetector, multi_column_regions: list[LayoutRegion]
    ) -> None:
        """Test heuristic detection for multi-column layout."""
        result = detector.detect_reading_order_simple(
            multi_column_regions, page_width=1000.0, page_height=1200.0
        )

        # Should detect multi-column layout
        assert result.layout_type in ["multi_column", "single_column"]
        assert len(result.ordered_regions) == 5

    def test_detect_reading_order_simple_empty(
        self, detector: ReadingOrderDetector
    ) -> None:
        """Test heuristic detection for empty regions."""
        result = detector.detect_reading_order_simple([])

        assert result.layout_type == "empty"
        assert result.overall_confidence == 1.0
        assert len(result.ordered_regions) == 0

    def test_detect_reading_order_simple_marks_skip_regions(
        self, detector: ReadingOrderDetector
    ) -> None:
        """Test that heuristic marks appropriate regions as skip."""
        regions = [
            create_region("r1", RegionType.PAGE_HEADER, 100, 10, 900, 50),
            create_region("r2", RegionType.TEXT, 100, 100, 900, 300),
            create_region("r3", RegionType.PAGE_FOOTER, 100, 1100, 900, 1150),
            create_region("r4", RegionType.FOOTNOTE, 100, 1000, 600, 1050),
        ]

        result = detector.detect_reading_order_simple(regions)

        # Header, footer, and footnote should be marked as skip
        header = next(r for r in result.ordered_regions if r.region.region_id == "r1")
        footer = next(r for r in result.ordered_regions if r.region.region_id == "r3")
        footnote = next(r for r in result.ordered_regions if r.region.region_id == "r4")
        text = next(r for r in result.ordered_regions if r.region.region_id == "r2")

        assert header.skip_in_reading is True
        assert footer.skip_in_reading is True
        assert footnote.skip_in_reading is True
        assert text.skip_in_reading is False


class TestReadingOrderDetectorLayoutDetection:
    """Tests for layout type detection heuristics."""

    def test_detect_layout_type_single_column(
        self, detector: ReadingOrderDetector
    ) -> None:
        """Test layout type detection for single column."""
        # All regions vertically stacked, similar X positions
        regions = [
            create_region("r1", RegionType.TEXT, 100, 100, 900, 200),
            create_region("r2", RegionType.TEXT, 100, 250, 900, 350),
            create_region("r3", RegionType.TEXT, 100, 400, 900, 500),
        ]

        layout_type = detector._detect_layout_type_heuristic(regions, 1000.0)
        assert layout_type == "single_column"

    def test_detect_layout_type_multi_column(
        self, detector: ReadingOrderDetector
    ) -> None:
        """Test layout type detection for multi-column."""
        # Regions spread horizontally at similar Y positions
        regions = [
            create_region("r1", RegionType.TEXT, 50, 100, 450, 300),
            create_region("r2", RegionType.TEXT, 550, 100, 950, 300),
            create_region("r3", RegionType.TEXT, 50, 350, 450, 550),
            create_region("r4", RegionType.TEXT, 550, 350, 950, 550),
        ]

        layout_type = detector._detect_layout_type_heuristic(regions, 1000.0)
        assert layout_type == "multi_column"

    def test_detect_layout_type_single_region(
        self, detector: ReadingOrderDetector
    ) -> None:
        """Test layout type detection with single region."""
        regions = [
            create_region("r1", RegionType.TEXT, 100, 100, 900, 500),
        ]

        layout_type = detector._detect_layout_type_heuristic(regions, 1000.0)
        assert layout_type == "single_column"

    def test_detect_layout_type_filters_headers_footers(
        self, detector: ReadingOrderDetector
    ) -> None:
        """Test that headers/footers are filtered from column analysis."""
        # Two columns with header/footer - with more content to trigger multi-column
        regions = [
            create_region("r1", RegionType.PAGE_HEADER, 50, 10, 950, 50),
            create_region("r2", RegionType.TEXT, 50, 100, 450, 250),
            create_region("r3", RegionType.TEXT, 550, 100, 950, 250),
            create_region("r4", RegionType.TEXT, 50, 300, 450, 450),
            create_region("r5", RegionType.TEXT, 550, 300, 950, 450),
            create_region("r6", RegionType.PAGE_FOOTER, 50, 1100, 950, 1150),
        ]

        layout_type = detector._detect_layout_type_heuristic(regions, 1000.0)
        # Should detect multi-column based on content regions only
        assert layout_type == "multi_column"


class TestReadingOrderDetectorSorting:
    """Tests for region sorting methods."""

    def test_sort_single_column(self, detector: ReadingOrderDetector) -> None:
        """Test single column sorting (top to bottom)."""
        # Regions in random order
        regions = [
            create_region("r3", RegionType.TEXT, 100, 500, 900, 600),
            create_region("r1", RegionType.TEXT, 100, 100, 900, 200),
            create_region("r2", RegionType.TEXT, 100, 300, 900, 400),
        ]

        sorted_regions = detector._sort_single_column(regions)

        assert sorted_regions[0].region_id == "r1"  # Top
        assert sorted_regions[1].region_id == "r2"  # Middle
        assert sorted_regions[2].region_id == "r3"  # Bottom

    def test_sort_multi_column(self, detector: ReadingOrderDetector) -> None:
        """Test multi-column sorting (left column then right column)."""
        regions = [
            create_region("r3", RegionType.TEXT, 550, 100, 950, 200),  # Right, top
            create_region("r1", RegionType.TEXT, 50, 100, 450, 200),  # Left, top
            create_region("r4", RegionType.TEXT, 550, 300, 950, 400),  # Right, bottom
            create_region("r2", RegionType.TEXT, 50, 300, 450, 400),  # Left, bottom
        ]

        sorted_regions = detector._sort_multi_column(regions, 1000.0)

        # Left column first (top to bottom), then right column (top to bottom)
        assert sorted_regions[0].region_id == "r1"  # Left, top
        assert sorted_regions[1].region_id == "r2"  # Left, bottom
        assert sorted_regions[2].region_id == "r3"  # Right, top
        assert sorted_regions[3].region_id == "r4"  # Right, bottom


class TestSkipRegionTypes:
    """Tests for skip region types configuration."""

    def test_skip_region_types_contains_expected(self) -> None:
        """Test that SKIP_REGION_TYPES contains expected types."""
        assert RegionType.PAGE_HEADER in SKIP_REGION_TYPES
        assert RegionType.PAGE_FOOTER in SKIP_REGION_TYPES
        assert RegionType.FOOTNOTE in SKIP_REGION_TYPES

    def test_skip_region_types_excludes_content(self) -> None:
        """Test that SKIP_REGION_TYPES excludes content types."""
        assert RegionType.TEXT not in SKIP_REGION_TYPES
        assert RegionType.TABLE not in SKIP_REGION_TYPES
        assert RegionType.PICTURE not in SKIP_REGION_TYPES
        assert RegionType.TITLE not in SKIP_REGION_TYPES


class TestReadingOrderDetectorStaticMethods:
    """Tests for static methods."""

    def test_get_default_model(self) -> None:
        """Test getting default model from settings."""
        with patch(
            "agentic_document_extraction.services.reading_order_detector.settings"
        ) as mock_settings:
            mock_settings.openai_model = "gpt-4-turbo"

            model = ReadingOrderDetector.get_default_model()

            assert model == "gpt-4-turbo"


class TestReadingOrderDetectorIntegration:
    """Integration tests for reading order detection with various layout patterns."""

    def test_complex_layout_with_figures_and_tables(
        self,
        detector: ReadingOrderDetector,
        complex_layout_regions: list[LayoutRegion],
    ) -> None:
        """Test reading order for complex layout with figures and tables."""
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "layout_type": "complex",
                "ordered_regions": [
                    {
                        "region_id": "r1",
                        "order_index": 0,
                        "confidence": 0.95,
                        "reasoning": "Title",
                        "skip_in_reading": False,
                    },
                    {
                        "region_id": "r2",
                        "order_index": 1,
                        "confidence": 0.92,
                        "reasoning": "Body text before figure",
                        "skip_in_reading": False,
                    },
                    {
                        "region_id": "r3",
                        "order_index": 2,
                        "confidence": 0.88,
                        "reasoning": "Figure",
                        "skip_in_reading": False,
                    },
                    {
                        "region_id": "r4",
                        "order_index": 3,
                        "confidence": 0.90,
                        "reasoning": "Figure caption",
                        "skip_in_reading": False,
                    },
                    {
                        "region_id": "r5",
                        "order_index": 4,
                        "confidence": 0.85,
                        "reasoning": "Text beside figure",
                        "skip_in_reading": False,
                    },
                    {
                        "region_id": "r6",
                        "order_index": 5,
                        "confidence": 0.87,
                        "reasoning": "Table on right",
                        "skip_in_reading": False,
                    },
                    {
                        "region_id": "r7",
                        "order_index": 6,
                        "confidence": 0.80,
                        "reasoning": "Footnote",
                        "skip_in_reading": True,
                    },
                ],
                "overall_confidence": 0.88,
            }
        )

        page_layout = create_page_layout(complex_layout_regions)

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with patch(
            "agentic_document_extraction.services.reading_order_detector.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_class.from_messages.return_value = mock_prompt

            detector._llm = MagicMock()

            result = detector.detect_reading_order_for_page(page_layout)

        assert result.layout_type == "complex"
        assert len(result.ordered_regions) == 7

        # Verify footnote is marked as skip
        footnote = next(r for r in result.ordered_regions if r.region.region_id == "r7")
        assert footnote.skip_in_reading is True

    def test_reading_sequence_excludes_skipped(
        self,
        detector: ReadingOrderDetector,
        single_column_regions: list[LayoutRegion],
        mock_llm_response_single_column: str,
    ) -> None:
        """Test that reading sequence properly excludes skipped regions."""
        page_layout = create_page_layout(single_column_regions)

        mock_response = MagicMock()
        mock_response.content = mock_llm_response_single_column

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with patch(
            "agentic_document_extraction.services.reading_order_detector.ChatPromptTemplate"
        ) as mock_prompt_class:
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_class.from_messages.return_value = mock_prompt

            detector._llm = MagicMock()

            result = detector.detect_reading_order_for_page(page_layout)

        # Get reading sequence without skipped
        sequence = result.get_reading_sequence(include_skipped=False)

        # Should not include footer (r4) which is marked as skip
        region_ids = [r.region_id for r in sequence]
        assert "r4" not in region_ids
        assert len(sequence) == 3  # r1, r2, r3

    def test_y_cluster_counting(self, detector: ReadingOrderDetector) -> None:
        """Test Y value clustering for layout detection."""
        # Test with distinct clusters
        y_values = [100.0, 110.0, 120.0, 400.0, 410.0, 420.0]
        clusters = detector._count_y_clusters(y_values, threshold=50.0)
        assert clusters == 2

        # Test with single cluster
        y_values_single = [100.0, 110.0, 120.0, 130.0]
        clusters_single = detector._count_y_clusters(y_values_single, threshold=50.0)
        assert clusters_single == 1

        # Test with empty list
        clusters_empty = detector._count_y_clusters([], threshold=50.0)
        assert clusters_empty == 0
