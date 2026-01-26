"""Tests for the reading order detection service."""

from unittest.mock import patch

import pytest

from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    PageLayoutResult,
    RegionBoundingBox,
    RegionType,
)
from agentic_document_extraction.services.reading_order_detector import (
    MAX_LEN,
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
    return ReadingOrderDetector(model_name="test-layoutreader", device="cpu")


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

        sequence = page_order.get_reading_sequence(include_skipped=False)
        assert len(sequence) == 2
        assert sequence[0].region_id == "r1"
        assert sequence[1].region_id == "r3"

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
            model_used="layoutreader",
            processing_time_seconds=1.5,
        )

        assert len(doc_order.pages) == 2
        assert doc_order.total_pages == 2
        assert doc_order.total_regions == 2
        assert doc_order.model_used == "layoutreader"
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
            model_used="layoutreader",
            processing_time_seconds=0.8,
        )

        result = doc_order.to_dict()

        assert result["total_pages"] == 1
        assert result["total_regions"] == 1
        assert len(result["pages"]) == 1
        assert result["metadata"]["model_used"] == "layoutreader"
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
            model_used="layoutreader",
        )

        all_regions = doc_order.get_all_regions_ordered(include_skipped=False)
        assert len(all_regions) == 2
        assert all_regions[0] == (1, region1)
        assert all_regions[1] == (2, region3)

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
            details={"error_type": "model failure"},
        )

        assert error.details == {"error_type": "model failure"}


class TestReadingOrderDetectorInit:
    """Tests for ReadingOrderDetector initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with defaults."""
        with patch(
            "agentic_document_extraction.services.reading_order_detector.settings"
        ) as mock_settings:
            mock_settings.layoutreader_model = "hantian/layoutreader"

            detector = ReadingOrderDetector()

            assert detector.model_name == "hantian/layoutreader"
            assert detector.device in {"cpu", "cuda"}

    def test_custom_initialization(self) -> None:
        """Test initialization with custom values."""
        detector = ReadingOrderDetector(
            model_name="custom-layoutreader",
            device="cpu",
            use_bfloat16=False,
        )

        assert detector.model_name == "custom-layoutreader"
        assert detector.device == "cpu"
        assert detector.use_bfloat16 is False


class TestReadingOrderDetectorScale:
    """Tests for LayoutReader box scaling."""

    def test_scale_boxes_clamps_to_bounds(self, detector: ReadingOrderDetector) -> None:
        """Ensure scaled boxes are clamped between 0 and 1000."""
        regions = [
            create_region("r1", RegionType.TEXT, -10, -5, 1200, 1400),
        ]
        boxes = detector._scale_boxes(regions, page_width=1000.0, page_height=1000.0)
        assert boxes == [[0, 0, 1000, 1000]]


class TestReadingOrderDetectorLayoutReader:
    """Tests for LayoutReader-driven reading order detection."""

    def test_detect_reading_order_for_page(
        self,
        detector: ReadingOrderDetector,
        single_column_regions: list[LayoutRegion],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test reading order detection using LayoutReader output."""
        page_layout = create_page_layout(single_column_regions)

        def fake_predict_orders(
            _boxes: list[list[int]],
        ) -> tuple[list[int], list[float]]:
            return [0, 1, 2, 3], [0.95, 0.92, 0.90, 0.88]

        monkeypatch.setattr(detector, "_predict_orders", fake_predict_orders)

        result = detector.detect_reading_order_for_page(page_layout)

        assert result.page_number == 1
        assert result.layout_type == "single_column"
        assert len(result.ordered_regions) == 4
        assert result.ordered_regions[0].region.region_id == "r1"
        assert result.ordered_regions[-1].region.region_id == "r4"
        assert result.ordered_regions[-1].skip_in_reading is True

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

    def test_detect_reading_order_fallback_for_large_pages(
        self,
        detector: ReadingOrderDetector,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pages exceeding MAX_LEN should fall back to heuristic ordering."""
        regions = [
            create_region(f"r{i}", RegionType.TEXT, 10, 10 + i, 200, 20 + i)
            for i in range(MAX_LEN + 1)
        ]
        page_layout = create_page_layout(regions)

        def fail_predict(_: list[list[int]]) -> tuple[list[int], list[float]]:
            raise AssertionError("LayoutReader should not be called")

        monkeypatch.setattr(detector, "_predict_orders", fail_predict)

        result = detector.detect_reading_order_for_page(page_layout)
        assert result.layout_type in {"single_column", "multi_column"}
        assert len(result.ordered_regions) == len(regions)

    def test_detect_reading_order_model_error(
        self,
        detector: ReadingOrderDetector,
        single_column_regions: list[LayoutRegion],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Model errors should surface as ReadingOrderError."""
        page_layout = create_page_layout(single_column_regions)

        def fail_predict(_: list[list[int]]) -> tuple[list[int], list[float]]:
            raise RuntimeError("model failed")

        monkeypatch.setattr(detector, "_predict_orders", fail_predict)

        with pytest.raises(ReadingOrderError) as exc_info:
            detector.detect_reading_order_for_page(page_layout)

        assert "LayoutReader inference failed" in str(exc_info.value)

    def test_detect_reading_order_multi_page(
        self,
        detector: ReadingOrderDetector,
        single_column_regions: list[LayoutRegion],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test reading order detection for multi-page document."""
        page1_layout = create_page_layout(single_column_regions, page_number=1)
        page2_regions = [
            create_region("r5", RegionType.TEXT, 100, 100, 900, 400, page_number=2),
        ]
        page2_layout = create_page_layout(page2_regions, page_number=2)

        calls = {"count": 0}

        def fake_predict_orders(
            _boxes: list[list[int]],
        ) -> tuple[list[int], list[float]]:
            calls["count"] += 1
            return list(range(len(_boxes))), [0.9] * len(_boxes)

        monkeypatch.setattr(detector, "_predict_orders", fake_predict_orders)

        result = detector.detect_reading_order([page1_layout, page2_layout])

        assert result.total_pages == 2
        assert result.total_regions == 5  # 4 from page 1 + 1 from page 2
        assert len(result.pages) == 2
        assert result.model_used == detector.model_name
        assert calls["count"] == 2


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
        assert result.overall_confidence == 0.6
        assert len(result.ordered_regions) == 4
        assert result.ordered_regions[0].region.region_id == "r1"

    def test_detect_reading_order_simple_multi_column(
        self, detector: ReadingOrderDetector, multi_column_regions: list[LayoutRegion]
    ) -> None:
        """Test heuristic detection for multi-column layout."""
        result = detector.detect_reading_order_simple(
            multi_column_regions, page_width=1000.0, page_height=1200.0
        )

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
        regions = [
            create_region("r1", RegionType.PAGE_HEADER, 50, 10, 950, 50),
            create_region("r2", RegionType.TEXT, 50, 100, 450, 250),
            create_region("r3", RegionType.TEXT, 550, 100, 950, 250),
            create_region("r4", RegionType.TEXT, 50, 300, 450, 450),
            create_region("r5", RegionType.TEXT, 550, 300, 950, 450),
            create_region("r6", RegionType.PAGE_FOOTER, 50, 1100, 950, 1150),
        ]

        layout_type = detector._detect_layout_type_heuristic(regions, 1000.0)
        assert layout_type == "multi_column"


class TestReadingOrderDetectorSorting:
    """Tests for region sorting methods."""

    def test_sort_single_column(self, detector: ReadingOrderDetector) -> None:
        """Test single column sorting (top to bottom)."""
        regions = [
            create_region("r3", RegionType.TEXT, 100, 500, 900, 600),
            create_region("r1", RegionType.TEXT, 100, 100, 900, 200),
            create_region("r2", RegionType.TEXT, 100, 300, 900, 400),
        ]

        sorted_regions = detector._sort_single_column(regions)

        assert sorted_regions[0].region_id == "r1"
        assert sorted_regions[1].region_id == "r2"
        assert sorted_regions[2].region_id == "r3"

    def test_sort_multi_column(self, detector: ReadingOrderDetector) -> None:
        """Test multi-column sorting (left column then right column)."""
        regions = [
            create_region("r3", RegionType.TEXT, 550, 100, 950, 200),
            create_region("r1", RegionType.TEXT, 50, 100, 450, 200),
            create_region("r4", RegionType.TEXT, 550, 300, 950, 400),
            create_region("r2", RegionType.TEXT, 50, 300, 450, 400),
        ]

        sorted_regions = detector._sort_multi_column(regions, 1000.0)

        assert sorted_regions[0].region_id == "r1"
        assert sorted_regions[1].region_id == "r2"
        assert sorted_regions[2].region_id == "r3"
        assert sorted_regions[3].region_id == "r4"


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
            mock_settings.layoutreader_model = "hantian/layoutreader"

            model = ReadingOrderDetector.get_default_model()

            assert model == "hantian/layoutreader"
