"""Tests for the layout detection service."""

import io
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_document_extraction.services.layout_detector import (
    DOCLAYNET_LABEL_MAP,
    LayoutDetectionError,
    LayoutDetectionResult,
    LayoutDetector,
    LayoutRegion,
    PageLayoutResult,
    RegionBoundingBox,
    RegionType,
)


@pytest.fixture
def detector() -> LayoutDetector:
    """Create a LayoutDetector instance for testing."""
    return LayoutDetector()


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample image for testing."""
    return Image.new("RGB", (800, 1000), color="white")


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Create sample image bytes for testing."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def mock_model_outputs() -> dict[str, Any]:
    """Create mock model outputs for testing."""
    import torch

    return {
        "logits": torch.randn(1, 100, 12),  # 12 classes
        "pred_boxes": torch.rand(1, 100, 4),
    }


@pytest.fixture
def mock_detection_results() -> dict[str, Any]:
    """Create mock post-processed detection results."""
    import torch

    return {
        "scores": torch.tensor([0.95, 0.87, 0.76]),
        "labels": torch.tensor([9, 8, 6]),  # TEXT, TABLE, PICTURE
        "boxes": torch.tensor(
            [
                [100.0, 50.0, 700.0, 150.0],  # TEXT region
                [100.0, 200.0, 700.0, 500.0],  # TABLE region
                [100.0, 550.0, 400.0, 850.0],  # PICTURE region
            ]
        ),
    }


class TestRegionBoundingBox:
    """Tests for RegionBoundingBox class."""

    def test_create_bounding_box(self) -> None:
        """Test creating a bounding box."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)

        assert bbox.x0 == 10.0
        assert bbox.y0 == 20.0
        assert bbox.x1 == 110.0
        assert bbox.y1 == 70.0

    def test_width_property(self) -> None:
        """Test width calculation."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        assert bbox.width == 100.0

    def test_height_property(self) -> None:
        """Test height calculation."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        assert bbox.height == 50.0

    def test_center_property(self) -> None:
        """Test center point calculation."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        center = bbox.center
        assert center == (60.0, 45.0)

    def test_area_property(self) -> None:
        """Test area calculation."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        assert bbox.area == 5000.0  # 100 * 50

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        result = bbox.to_dict()

        assert result["x0"] == 10.0
        assert result["y0"] == 20.0
        assert result["x1"] == 110.0
        assert result["y1"] == 70.0
        assert result["width"] == 100.0
        assert result["height"] == 50.0

    def test_contains_point_inside(self) -> None:
        """Test point containment - inside."""
        bbox = RegionBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        assert bbox.contains_point(50.0, 50.0) is True

    def test_contains_point_outside(self) -> None:
        """Test point containment - outside."""
        bbox = RegionBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        assert bbox.contains_point(150.0, 50.0) is False

    def test_contains_point_on_edge(self) -> None:
        """Test point containment - on edge."""
        bbox = RegionBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        assert bbox.contains_point(100.0, 50.0) is True

    def test_intersects_true(self) -> None:
        """Test intersection - overlapping boxes."""
        bbox1 = RegionBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        bbox2 = RegionBoundingBox(x0=50.0, y0=50.0, x1=150.0, y1=150.0)
        assert bbox1.intersects(bbox2) is True

    def test_intersects_false(self) -> None:
        """Test intersection - non-overlapping boxes."""
        bbox1 = RegionBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        bbox2 = RegionBoundingBox(x0=200.0, y0=200.0, x1=300.0, y1=300.0)
        assert bbox1.intersects(bbox2) is False

    def test_intersection_area_overlapping(self) -> None:
        """Test intersection area - overlapping boxes."""
        bbox1 = RegionBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        bbox2 = RegionBoundingBox(x0=50.0, y0=50.0, x1=150.0, y1=150.0)
        # Intersection: (50, 50) to (100, 100) = 50 * 50 = 2500
        assert bbox1.intersection_area(bbox2) == 2500.0

    def test_intersection_area_non_overlapping(self) -> None:
        """Test intersection area - non-overlapping boxes."""
        bbox1 = RegionBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        bbox2 = RegionBoundingBox(x0=200.0, y0=200.0, x1=300.0, y1=300.0)
        assert bbox1.intersection_area(bbox2) == 0.0


class TestRegionType:
    """Tests for RegionType enum."""

    def test_region_type_values(self) -> None:
        """Test region type values."""
        assert RegionType.TEXT.value == "text"
        assert RegionType.TABLE.value == "table"
        assert RegionType.PICTURE.value == "picture"
        assert RegionType.SECTION_HEADER.value == "section_header"
        assert RegionType.PAGE_HEADER.value == "page_header"
        assert RegionType.PAGE_FOOTER.value == "page_footer"
        assert RegionType.CAPTION.value == "caption"
        assert RegionType.TITLE.value == "title"
        assert RegionType.FOOTNOTE.value == "footnote"
        assert RegionType.FORMULA.value == "formula"
        assert RegionType.LIST_ITEM.value == "list_item"
        assert RegionType.UNKNOWN.value == "unknown"


class TestLayoutRegion:
    """Tests for LayoutRegion class."""

    def test_create_layout_region(self) -> None:
        """Test creating a layout region."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        region = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )

        assert region.region_type == RegionType.TEXT
        assert region.bbox == bbox
        assert region.confidence == 0.95
        assert region.page_number == 1
        assert region.region_id == "region_p1_1"
        assert region.parent_region_id is None
        assert region.metadata == {}

    def test_layout_region_with_parent(self) -> None:
        """Test creating a layout region with parent."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        region = LayoutRegion(
            region_type=RegionType.CAPTION,
            bbox=bbox,
            confidence=0.88,
            page_number=1,
            region_id="region_p1_2",
            parent_region_id="region_p1_1",
        )

        assert region.parent_region_id == "region_p1_1"

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        region = LayoutRegion(
            region_type=RegionType.TABLE,
            bbox=bbox,
            confidence=0.92,
            page_number=2,
            region_id="region_p2_1",
            metadata={"row_count": 5},
        )

        result = region.to_dict()

        assert result["region_type"] == "table"
        assert result["confidence"] == 0.92
        assert result["page_number"] == 2
        assert result["region_id"] == "region_p2_1"
        assert result["parent_region_id"] is None
        assert result["metadata"] == {"row_count": 5}
        assert "bbox" in result


class TestPageLayoutResult:
    """Tests for PageLayoutResult class."""

    def test_create_page_layout_result(self) -> None:
        """Test creating a page layout result."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        region = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )

        result = PageLayoutResult(
            page_number=1,
            regions=[region],
            page_width=800.0,
            page_height=1000.0,
        )

        assert result.page_number == 1
        assert len(result.regions) == 1
        assert result.page_width == 800.0
        assert result.page_height == 1000.0

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        region = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )

        result = PageLayoutResult(
            page_number=1,
            regions=[region],
            page_width=800.0,
            page_height=1000.0,
        )

        result_dict = result.to_dict()

        assert result_dict["page_number"] == 1
        assert result_dict["region_count"] == 1
        assert result_dict["page_width"] == 800.0
        assert result_dict["page_height"] == 1000.0
        assert len(result_dict["regions"]) == 1

    def test_get_regions_by_type(self) -> None:
        """Test getting regions by type."""
        bbox1 = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        bbox2 = RegionBoundingBox(x0=10.0, y0=100.0, x1=110.0, y1=150.0)

        region1 = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox1,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )
        region2 = LayoutRegion(
            region_type=RegionType.TABLE,
            bbox=bbox2,
            confidence=0.90,
            page_number=1,
            region_id="region_p1_2",
        )

        result = PageLayoutResult(
            page_number=1,
            regions=[region1, region2],
            page_width=800.0,
            page_height=1000.0,
        )

        text_regions = result.get_regions_by_type(RegionType.TEXT)
        table_regions = result.get_regions_by_type(RegionType.TABLE)

        assert len(text_regions) == 1
        assert text_regions[0].region_id == "region_p1_1"
        assert len(table_regions) == 1
        assert table_regions[0].region_id == "region_p1_2"


class TestLayoutDetectionResult:
    """Tests for LayoutDetectionResult class."""

    def test_create_layout_detection_result(self) -> None:
        """Test creating a layout detection result."""
        bbox = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        region = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )
        page = PageLayoutResult(
            page_number=1,
            regions=[region],
            page_width=800.0,
            page_height=1000.0,
        )

        result = LayoutDetectionResult(
            pages=[page],
            total_pages=1,
            total_regions=1,
            model_name="test-model",
        )

        assert len(result.pages) == 1
        assert result.total_pages == 1
        assert result.total_regions == 1
        assert result.model_name == "test-model"
        assert result.metadata == {}

    def test_get_all_regions(self) -> None:
        """Test getting all regions across pages."""
        bbox1 = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        bbox2 = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)

        region1 = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox1,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )
        region2 = LayoutRegion(
            region_type=RegionType.TABLE,
            bbox=bbox2,
            confidence=0.90,
            page_number=2,
            region_id="region_p2_1",
        )

        page1 = PageLayoutResult(
            page_number=1,
            regions=[region1],
            page_width=800.0,
            page_height=1000.0,
        )
        page2 = PageLayoutResult(
            page_number=2,
            regions=[region2],
            page_width=800.0,
            page_height=1000.0,
        )

        result = LayoutDetectionResult(
            pages=[page1, page2],
            total_pages=2,
            total_regions=2,
            model_name="test-model",
        )

        all_regions = result.get_all_regions()
        assert len(all_regions) == 2

    def test_get_regions_by_type(self) -> None:
        """Test getting regions by type across all pages."""
        bbox1 = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        bbox2 = RegionBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)

        region1 = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox1,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )
        region2 = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox2,
            confidence=0.90,
            page_number=2,
            region_id="region_p2_1",
        )

        page1 = PageLayoutResult(
            page_number=1,
            regions=[region1],
            page_width=800.0,
            page_height=1000.0,
        )
        page2 = PageLayoutResult(
            page_number=2,
            regions=[region2],
            page_width=800.0,
            page_height=1000.0,
        )

        result = LayoutDetectionResult(
            pages=[page1, page2],
            total_pages=2,
            total_regions=2,
            model_name="test-model",
        )

        text_regions = result.get_regions_by_type(RegionType.TEXT)
        assert len(text_regions) == 2


class TestLayoutDetectorInit:
    """Tests for LayoutDetector initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        detector = LayoutDetector()

        assert detector.model_name == LayoutDetector.DEFAULT_MODEL
        assert detector.confidence_threshold == 0.5
        assert detector._device is None

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        detector = LayoutDetector(
            model_name="custom/model",
            confidence_threshold=0.7,
            device="cpu",
        )

        assert detector.model_name == "custom/model"
        assert detector.confidence_threshold == 0.7
        assert detector._device == "cpu"


class TestLayoutDetectorStaticMethods:
    """Tests for LayoutDetector static methods."""

    def test_get_supported_region_types(self) -> None:
        """Test getting supported region types."""
        types = LayoutDetector.get_supported_region_types()

        assert "text" in types
        assert "table" in types
        assert "picture" in types
        assert "section_header" in types
        assert "caption" in types
        assert "unknown" not in types  # UNKNOWN should be excluded

    def test_get_default_model(self) -> None:
        """Test getting default model."""
        model = LayoutDetector.get_default_model()
        assert model == "pascalrai/Deformable-DETR-Document-Layout-Analysis"


class TestLayoutDetectorErrors:
    """Tests for error handling."""

    def test_nonexistent_file_raises(self, detector: LayoutDetector) -> None:
        """Test that nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            detector.detect_from_path("/nonexistent/path/image.png")


class TestLayoutDetectorMocked:
    """Tests with mocked model for controlled testing."""

    def test_detect_from_image_mocked(
        self,
        detector: LayoutDetector,
        sample_image: Image.Image,
        mock_detection_results: dict[str, Any],
    ) -> None:
        """Test layout detection with mocked model."""
        import torch

        # Create mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Configure mock processor
        mock_processor.return_value = {
            "pixel_values": torch.rand(1, 3, 800, 1000),
            "pixel_mask": torch.ones(1, 800, 1000),
        }
        mock_processor.post_process_object_detection.return_value = [
            mock_detection_results
        ]

        # Configure mock model
        mock_model.return_value = MagicMock()

        with (
            patch.object(detector, "_model", mock_model),
            patch.object(detector, "_processor", mock_processor),
        ):
            result = detector.detect_from_image(sample_image)

            assert result.page_number == 1
            assert result.page_width == 800.0
            assert result.page_height == 1000.0
            assert len(result.regions) == 3

            # Check region types based on mock labels
            region_types = [r.region_type for r in result.regions]
            assert RegionType.TEXT in region_types
            assert RegionType.TABLE in region_types
            assert RegionType.PICTURE in region_types

    def test_detect_from_content_mocked(
        self,
        detector: LayoutDetector,
        sample_image_bytes: bytes,
        mock_detection_results: dict[str, Any],
    ) -> None:
        """Test layout detection from content with mocked model."""
        import torch

        mock_model = MagicMock()
        mock_processor = MagicMock()

        mock_processor.return_value = {
            "pixel_values": torch.rand(1, 3, 800, 1000),
            "pixel_mask": torch.ones(1, 800, 1000),
        }
        mock_processor.post_process_object_detection.return_value = [
            mock_detection_results
        ]

        with (
            patch.object(detector, "_model", mock_model),
            patch.object(detector, "_processor", mock_processor),
        ):
            result = detector.detect_from_content(sample_image_bytes)

            assert result.total_pages == 1
            assert result.total_regions == 3
            assert result.model_name == LayoutDetector.DEFAULT_MODEL

    def test_detect_from_images_mocked(
        self,
        detector: LayoutDetector,
        sample_image: Image.Image,
        mock_detection_results: dict[str, Any],
    ) -> None:
        """Test multi-page layout detection with mocked model."""
        import torch

        mock_model = MagicMock()
        mock_processor = MagicMock()

        mock_processor.return_value = {
            "pixel_values": torch.rand(1, 3, 800, 1000),
            "pixel_mask": torch.ones(1, 800, 1000),
        }
        mock_processor.post_process_object_detection.return_value = [
            mock_detection_results
        ]

        with (
            patch.object(detector, "_model", mock_model),
            patch.object(detector, "_processor", mock_processor),
        ):
            images = [sample_image, sample_image]
            result = detector.detect_from_images(images)

            assert result.total_pages == 2
            assert result.total_regions == 6  # 3 regions per page
            assert len(result.pages) == 2

    def test_regions_sorted_by_position(
        self,
        detector: LayoutDetector,
        sample_image: Image.Image,
    ) -> None:
        """Test that detected regions are sorted by vertical position."""
        import torch

        # Create mock results with regions in reverse order
        mock_results = {
            "scores": torch.tensor([0.9, 0.9, 0.9]),
            "labels": torch.tensor([9, 9, 9]),  # All TEXT
            "boxes": torch.tensor(
                [
                    [100.0, 500.0, 700.0, 600.0],  # Middle
                    [100.0, 700.0, 700.0, 800.0],  # Bottom
                    [100.0, 100.0, 700.0, 200.0],  # Top
                ]
            ),
        }

        mock_model = MagicMock()
        mock_processor = MagicMock()

        mock_processor.return_value = {
            "pixel_values": torch.rand(1, 3, 800, 1000),
            "pixel_mask": torch.ones(1, 800, 1000),
        }
        mock_processor.post_process_object_detection.return_value = [mock_results]

        with (
            patch.object(detector, "_model", mock_model),
            patch.object(detector, "_processor", mock_processor),
        ):
            result = detector.detect_from_image(sample_image)

            # Verify regions are sorted top to bottom
            y_positions = [r.bbox.y0 for r in result.regions]
            assert y_positions == sorted(y_positions)


class TestLayoutDetectorCropRegion:
    """Tests for region cropping functionality."""

    def test_crop_region(self, detector: LayoutDetector) -> None:
        """Test cropping a region from an image."""
        image = Image.new("RGB", (800, 1000), color="white")
        bbox = RegionBoundingBox(x0=100.0, y0=200.0, x1=400.0, y1=500.0)
        region = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )

        cropped = detector.crop_region(image, region)

        assert cropped.width == 300  # 400 - 100
        assert cropped.height == 300  # 500 - 200

    def test_crop_region_with_padding(self, detector: LayoutDetector) -> None:
        """Test cropping a region with padding."""
        image = Image.new("RGB", (800, 1000), color="white")
        bbox = RegionBoundingBox(x0=100.0, y0=200.0, x1=400.0, y1=500.0)
        region = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )

        cropped = detector.crop_region(image, region, padding=10)

        assert cropped.width == 320  # 300 + 2*10
        assert cropped.height == 320  # 300 + 2*10

    def test_crop_region_at_edge(self, detector: LayoutDetector) -> None:
        """Test cropping a region at the image edge with padding."""
        image = Image.new("RGB", (800, 1000), color="white")
        bbox = RegionBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        region = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )

        # Padding should be clamped to image boundaries
        cropped = detector.crop_region(image, region, padding=50)

        # Should not exceed original image boundaries
        assert cropped.width <= 150  # 100 + padding on right only
        assert cropped.height <= 150  # 100 + padding on bottom only


class TestDocLayNetLabelMap:
    """Tests for DocLayNet label mapping."""

    def test_label_map_contains_expected_labels(self) -> None:
        """Test that label map contains expected labels."""
        assert DOCLAYNET_LABEL_MAP[0] == RegionType.CAPTION
        assert DOCLAYNET_LABEL_MAP[1] == RegionType.FOOTNOTE
        assert DOCLAYNET_LABEL_MAP[2] == RegionType.FORMULA
        assert DOCLAYNET_LABEL_MAP[3] == RegionType.LIST_ITEM
        assert DOCLAYNET_LABEL_MAP[4] == RegionType.PAGE_FOOTER
        assert DOCLAYNET_LABEL_MAP[5] == RegionType.PAGE_HEADER
        assert DOCLAYNET_LABEL_MAP[6] == RegionType.PICTURE
        assert DOCLAYNET_LABEL_MAP[7] == RegionType.SECTION_HEADER
        assert DOCLAYNET_LABEL_MAP[8] == RegionType.TABLE
        assert DOCLAYNET_LABEL_MAP[9] == RegionType.TEXT
        assert DOCLAYNET_LABEL_MAP[10] == RegionType.TITLE

    def test_label_map_count(self) -> None:
        """Test that label map has expected number of entries."""
        assert len(DOCLAYNET_LABEL_MAP) == 11


class TestLayoutDetectionError:
    """Tests for LayoutDetectionError."""

    def test_error_with_message(self) -> None:
        """Test error with just message."""
        error = LayoutDetectionError("Test error")

        assert str(error) == "Test error"
        assert error.file_path is None
        assert error.page_number is None

    def test_error_with_file_path(self) -> None:
        """Test error with file path."""
        error = LayoutDetectionError("Test error", file_path="/path/to/file.png")

        assert error.file_path == "/path/to/file.png"

    def test_error_with_page_number(self) -> None:
        """Test error with page number."""
        error = LayoutDetectionError(
            "Test error", file_path="/path/to/file.pdf", page_number=5
        )

        assert error.page_number == 5


class TestNestedRegionDetection:
    """Tests for nested region detection."""

    def test_caption_inside_picture_detected(self, detector: LayoutDetector) -> None:
        """Test that caption nested inside picture is detected."""
        # Picture region
        picture_bbox = RegionBoundingBox(x0=100.0, y0=100.0, x1=400.0, y1=400.0)
        picture_region = LayoutRegion(
            region_type=RegionType.PICTURE,
            bbox=picture_bbox,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )

        # Caption inside picture
        caption_bbox = RegionBoundingBox(x0=120.0, y0=350.0, x1=380.0, y1=390.0)
        caption_region = LayoutRegion(
            region_type=RegionType.CAPTION,
            bbox=caption_bbox,
            confidence=0.90,
            page_number=1,
            region_id="region_p1_2",
        )

        regions = [picture_region, caption_region]
        updated_regions = detector._detect_nested_regions(regions)

        # Caption should have picture as parent
        caption = next(
            r for r in updated_regions if r.region_type == RegionType.CAPTION
        )
        assert caption.parent_region_id == "region_p1_1"

    def test_non_nested_regions_unchanged(self, detector: LayoutDetector) -> None:
        """Test that non-nested regions don't get parent assignment."""
        # Two separate TEXT regions
        bbox1 = RegionBoundingBox(x0=100.0, y0=100.0, x1=400.0, y1=200.0)
        region1 = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox1,
            confidence=0.95,
            page_number=1,
            region_id="region_p1_1",
        )

        bbox2 = RegionBoundingBox(x0=100.0, y0=300.0, x1=400.0, y1=400.0)
        region2 = LayoutRegion(
            region_type=RegionType.TEXT,
            bbox=bbox2,
            confidence=0.90,
            page_number=1,
            region_id="region_p1_2",
        )

        regions = [region1, region2]
        updated_regions = detector._detect_nested_regions(regions)

        # Neither should have a parent
        for region in updated_regions:
            assert region.parent_region_id is None


class TestSegmentDocument:
    """Tests for document segmentation functionality."""

    def test_segment_document_mocked(
        self,
        detector: LayoutDetector,
        sample_image: Image.Image,
        mock_detection_results: dict[str, Any],
    ) -> None:
        """Test document segmentation with mocked model."""
        import torch

        mock_model = MagicMock()
        mock_processor = MagicMock()

        mock_processor.return_value = {
            "pixel_values": torch.rand(1, 3, 800, 1000),
            "pixel_mask": torch.ones(1, 800, 1000),
        }
        mock_processor.post_process_object_detection.return_value = [
            mock_detection_results
        ]

        with (
            patch.object(detector, "_model", mock_model),
            patch.object(detector, "_processor", mock_processor),
        ):
            segmented = detector.segment_document([sample_image])

            # Should have entries for detected region types
            assert "text" in segmented
            assert "table" in segmented
            assert "picture" in segmented

            # Each entry should contain (region, cropped_image) tuples
            for _region_type, items in segmented.items():
                for region, cropped_image in items:
                    assert isinstance(region, LayoutRegion)
                    assert isinstance(cropped_image, Image.Image)
