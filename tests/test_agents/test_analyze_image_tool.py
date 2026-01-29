"""Tests for the AnalyzeImage tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_image import (
    AnalyzeImage,
    _normalize_image_result,
    _normalize_object,
    analyze_image_impl,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionImage,
    RegionType,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
    "ADE_OPENAI_API_KEY", ""
)
SKIP_INTEGRATION = not OPENAI_API_KEY or os.environ.get("ADE_SKIP_REAL_API_TESTS")
INTEGRATION_SKIP_REASON = (
    "OpenAI API key not set (set OPENAI_API_KEY or ADE_OPENAI_API_KEY) "
    "or ADE_SKIP_REAL_API_TESTS is enabled"
)


def create_product_image() -> Image.Image:
    """Create a simple image simulating a product photo."""
    image = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(image)

    # Simulate a product box
    draw.rectangle((100, 50, 300, 250), outline="blue", width=3)

    # Add some product-like details
    draw.rectangle((120, 80, 180, 140), fill="red")  # Logo area
    draw.text((130, 160), "Product", fill="black")
    draw.text((130, 180), "Brand X", fill="blue")

    return image


def create_multiple_items_image() -> Image.Image:
    """Create an image with multiple items for counting."""
    image = Image.new("RGB", (500, 300), color="white")
    draw = ImageDraw.Draw(image)

    # Draw multiple circles (representing items like fruits)
    positions = [(80, 80), (180, 80), (280, 80), (380, 80), (130, 180), (230, 180)]
    for x, y in positions:
        draw.ellipse((x - 30, y - 30, x + 30, y + 30), fill="orange", outline="brown")

    return image


def create_damage_assessment_image() -> Image.Image:
    """Create an image simulating damage assessment."""
    image = Image.new("RGB", (400, 300), color="gray")
    draw = ImageDraw.Draw(image)

    # Simulate a damaged surface with cracks
    draw.line((50, 100, 200, 150), fill="black", width=3)
    draw.line((200, 150, 180, 200), fill="black", width=2)
    draw.line((200, 150, 250, 180), fill="black", width=2)

    # Scratch marks
    draw.line((300, 50, 350, 120), fill="darkgray", width=2)
    draw.line((310, 60, 360, 130), fill="darkgray", width=2)

    return image


def create_equipment_image() -> Image.Image:
    """Create an image simulating equipment/machinery."""
    image = Image.new("RGB", (400, 300), color="lightgray")
    draw = ImageDraw.Draw(image)

    # Simulate equipment with rectangles and circles
    draw.rectangle((50, 50, 200, 200), fill="silver", outline="black", width=2)

    # Control panel
    draw.rectangle((60, 60, 100, 100), fill="darkgray")

    # Dials/buttons
    draw.ellipse((70, 70, 90, 90), fill="red")

    # Serial number area
    draw.rectangle((110, 70, 190, 90), fill="white")
    draw.text((115, 72), "SN: 12345", fill="black")

    # Gauge
    draw.ellipse((120, 120, 180, 180), outline="black", width=2)
    draw.line((150, 150, 150, 125), fill="red", width=2)

    return image


def create_inventory_items_image() -> Image.Image:
    """Create an image simulating inventory items on shelves."""
    image = Image.new("RGB", (500, 400), color="white")
    draw = ImageDraw.Draw(image)

    # Draw shelf lines
    draw.line((20, 150, 480, 150), fill="brown", width=5)
    draw.line((20, 300, 480, 300), fill="brown", width=5)

    # Items on top shelf
    for i in range(4):
        x = 60 + i * 110
        draw.rectangle((x, 80, x + 80, 145), fill="lightblue", outline="blue")
        draw.text((x + 10, 100), f"Item {i + 1}", fill="black")

    # Items on bottom shelf
    for i in range(3):
        x = 80 + i * 140
        draw.rectangle((x, 180, x + 100, 295), fill="lightgreen", outline="green")
        draw.text((x + 10, 220), f"Box {i + 1}", fill="black")

    return image


def create_vehicle_image() -> Image.Image:
    """Create a simple vehicle image."""
    image = Image.new("RGB", (400, 250), color="lightblue")
    draw = ImageDraw.Draw(image)

    # Simple car shape
    draw.rectangle((80, 150, 320, 200), fill="red")  # Body
    draw.polygon([(120, 150), (160, 100), (280, 100), (280, 150)], fill="red")  # Top

    # Windows
    draw.polygon([(165, 145), (195, 105), (265, 105), (265, 145)], fill="lightblue")

    # Wheels
    draw.ellipse((100, 185, 150, 235), fill="black")
    draw.ellipse((250, 185, 300, 235), fill="black")

    # License plate
    draw.rectangle((170, 175, 230, 195), fill="white", outline="black")
    draw.text((178, 178), "ABC123", fill="black")

    return image


def create_property_image() -> Image.Image:
    """Create a simple property/building image."""
    image = Image.new("RGB", (400, 300), color="skyblue")
    draw = ImageDraw.Draw(image)

    # Ground
    draw.rectangle((0, 250, 400, 300), fill="green")

    # Building
    draw.rectangle((100, 100, 300, 250), fill="beige", outline="brown", width=2)

    # Roof
    draw.polygon([(80, 100), (200, 30), (320, 100)], fill="brown")

    # Door
    draw.rectangle((180, 180, 220, 250), fill="brown")

    # Windows
    draw.rectangle((120, 130, 160, 170), fill="lightblue", outline="brown")
    draw.rectangle((240, 130, 280, 170), fill="lightblue", outline="brown")

    return image


def create_image_with_text() -> Image.Image:
    """Create an image with visible text labels."""
    image = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(image)

    # Product with text labels
    draw.rectangle((50, 50, 350, 250), fill="lightgray", outline="black")

    draw.text((100, 80), "Model: XYZ-500", fill="black")
    draw.text((100, 110), "Serial: SN12345ABC", fill="black")
    draw.text((100, 140), "Warning: High Voltage", fill="red")
    draw.text((100, 170), "Made in USA", fill="blue")
    draw.text((100, 200), "Warranty: 2 Years", fill="black")

    return image


def create_low_quality_image() -> Image.Image:
    """Create a low quality/blurry image."""
    image = Image.new("RGB", (200, 150), color="gray")
    draw = ImageDraw.Draw(image)

    # Draw some blurry shapes
    for i in range(10):
        draw.rectangle(
            (20 + i * 5, 30 + i * 3, 50 + i * 5, 60 + i * 3),
            fill=(100 + i * 10, 100 + i * 10, 100 + i * 10),
        )

    return image


def create_empty_scene_image() -> Image.Image:
    """Create an image with minimal content."""
    image = Image.new("RGB", (300, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Just a simple empty table
    draw.rectangle((50, 50, 250, 150), outline="gray", width=1)

    return image


def create_regions(
    image: Image.Image,
    *,
    region_id: str = "region-1",
    region_type: RegionType = RegionType.PICTURE,
) -> list[LayoutRegion]:
    """Create layout regions for testing."""
    bbox = RegionBoundingBox(x0=0, y0=0, x1=image.width, y1=image.height)
    return [
        LayoutRegion(
            region_type=region_type,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id=region_id,
            region_image=RegionImage(image=image),
        )
    ]


class TestAnalyzeImageTool:
    """Tests for the AnalyzeImage tool function."""

    def test_analyze_image_identifies_single_product(self) -> None:
        """Test identification of a single product in image."""
        response_payload: dict[str, Any] = {
            "description": "A product box from Brand X",
            "objects": [
                {
                    "object_type": "product",
                    "description": "Product box with red logo",
                    "count": 1,
                    "attributes": {"brand": "Brand X", "color": "blue"},
                    "confidence": "high",
                }
            ],
            "total_items": 1,
            "condition_assessment": "good",
            "extracted_text": ["Product", "Brand X"],
            "notes": "Product appears to be in good condition.",
        }

        regions = create_regions(create_product_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["description"] == "A product box from Brand X"
        assert len(result["objects"]) == 1
        assert result["objects"][0]["object_type"] == "product"
        assert result["total_items"] == 1
        assert result["condition_assessment"] == "good"

    def test_analyze_image_counts_multiple_items(self) -> None:
        """Test counting multiple items in image."""
        response_payload: dict[str, Any] = {
            "description": "Multiple oranges arranged in rows",
            "objects": [
                {
                    "object_type": "item",
                    "description": "Orange fruits",
                    "count": 6,
                    "attributes": {"color": "orange", "type": "fruit"},
                    "confidence": "high",
                }
            ],
            "total_items": 6,
            "condition_assessment": "good",
            "extracted_text": None,
            "notes": None,
        }

        regions = create_regions(create_multiple_items_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["total_items"] == 6
        assert result["objects"][0]["count"] == 6

    def test_analyze_image_assesses_damage(self) -> None:
        """Test damage assessment in image."""
        response_payload: dict[str, Any] = {
            "description": "A surface showing signs of damage",
            "objects": [
                {
                    "object_type": "damage",
                    "description": "Cracks on surface",
                    "count": 3,
                    "attributes": {"type": "crack", "severity": "moderate"},
                    "confidence": "high",
                },
                {
                    "object_type": "damage",
                    "description": "Scratch marks",
                    "count": 2,
                    "attributes": {"type": "scratch", "severity": "minor"},
                    "confidence": "medium",
                },
            ],
            "total_items": None,
            "condition_assessment": "poor",
            "extracted_text": None,
            "notes": "Surface shows moderate wear with visible cracks.",
        }

        regions = create_regions(create_damage_assessment_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["condition_assessment"] == "poor"
        assert len(result["objects"]) == 2
        assert any(obj["object_type"] == "damage" for obj in result["objects"])

    def test_analyze_image_extracts_equipment_info(self) -> None:
        """Test extraction of equipment information."""
        response_payload: dict[str, Any] = {
            "description": "Industrial equipment with control panel",
            "objects": [
                {
                    "object_type": "equipment",
                    "description": "Control panel unit",
                    "count": 1,
                    "attributes": {
                        "model": "Unknown",
                        "serial_number": "12345",
                        "condition": "operational",
                    },
                    "confidence": "high",
                }
            ],
            "total_items": 1,
            "condition_assessment": "good",
            "extracted_text": ["SN: 12345"],
            "notes": "Equipment appears functional.",
        }

        regions = create_regions(create_equipment_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["objects"][0]["object_type"] == "equipment"
        assert "12345" in str(result["extracted_text"])

    def test_analyze_image_handles_inventory(self) -> None:
        """Test handling of inventory items."""
        response_payload: dict[str, Any] = {
            "description": "Inventory items displayed on shelves",
            "objects": [
                {
                    "object_type": "item",
                    "description": "Blue boxes on top shelf",
                    "count": 4,
                    "attributes": {"color": "lightblue", "location": "top shelf"},
                    "confidence": "high",
                },
                {
                    "object_type": "item",
                    "description": "Green boxes on bottom shelf",
                    "count": 3,
                    "attributes": {"color": "lightgreen", "location": "bottom shelf"},
                    "confidence": "high",
                },
            ],
            "total_items": 7,
            "condition_assessment": "good",
            "extracted_text": [
                "Item 1",
                "Item 2",
                "Item 3",
                "Item 4",
                "Box 1",
                "Box 2",
                "Box 3",
            ],
            "notes": None,
        }

        regions = create_regions(create_inventory_items_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["total_items"] == 7
        assert len(result["objects"]) == 2

    def test_analyze_image_identifies_vehicle(self) -> None:
        """Test identification of a vehicle."""
        response_payload: dict[str, Any] = {
            "description": "A red car with visible license plate",
            "objects": [
                {
                    "object_type": "vehicle",
                    "description": "Red sedan",
                    "count": 1,
                    "attributes": {"color": "red", "type": "sedan"},
                    "confidence": "high",
                }
            ],
            "total_items": 1,
            "condition_assessment": "good",
            "extracted_text": ["ABC123"],
            "notes": "Vehicle license plate visible.",
        }

        regions = create_regions(create_vehicle_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["objects"][0]["object_type"] == "vehicle"
        assert "ABC123" in result["extracted_text"]

    def test_analyze_image_with_focus_parameter(self) -> None:
        """Test that focus parameter is included in prompt."""
        response_payload: dict[str, Any] = {
            "description": "Items for counting",
            "objects": [
                {
                    "object_type": "item",
                    "description": "Orange items",
                    "count": 6,
                    "attributes": None,
                    "confidence": "high",
                }
            ],
            "total_items": 6,
            "condition_assessment": None,
            "extracted_text": None,
            "notes": "Focus was on counting items.",
        }

        regions = create_regions(create_multiple_items_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ) as mock_vlm:
            result = AnalyzeImage.invoke(
                {
                    "region_id": "region-1",
                    "regions": regions,
                    "focus": "count all items",
                }
            )

            # Verify focus was included in the prompt
            call_args = mock_vlm.call_args
            prompt = call_args[0][1]
            assert "count all items" in prompt.lower()

        assert result["total_items"] == 6

    def test_analyze_image_extracts_text(self) -> None:
        """Test extraction of visible text in image."""
        response_payload: dict[str, Any] = {
            "description": "Equipment with multiple labels",
            "objects": [
                {
                    "object_type": "equipment",
                    "description": "Labeled equipment panel",
                    "count": 1,
                    "attributes": {"model": "XYZ-500"},
                    "confidence": "high",
                }
            ],
            "total_items": 1,
            "condition_assessment": "good",
            "extracted_text": [
                "Model: XYZ-500",
                "Serial: SN12345ABC",
                "Warning: High Voltage",
                "Made in USA",
                "Warranty: 2 Years",
            ],
            "notes": None,
        }

        regions = create_regions(create_image_with_text())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert len(result["extracted_text"]) >= 3
        assert any("XYZ-500" in text for text in result["extracted_text"])

    def test_analyze_image_handles_wrapped_json(self) -> None:
        """Test handling of wrapped JSON in VLM response."""
        wrapped = (
            "Here is the analysis:\n"
            + json.dumps(
                {
                    "description": "Test image",
                    "objects": [],
                    "total_items": None,
                }
            )
            + "\nEnd of analysis."
        )
        regions = create_regions(create_product_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["description"] == "Test image"

    def test_analyze_image_invalid_region_id(self) -> None:
        """Test error handling for invalid region ID."""
        regions = create_regions(create_product_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            AnalyzeImage.invoke({"region_id": "nonexistent", "regions": regions})

    def test_analyze_image_vlm_failure(self) -> None:
        """Test error handling when VLM call fails."""
        regions = create_regions(create_product_image())
        with (
            patch(
                "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
                side_effect=RuntimeError("VLM service unavailable"),
            ),
            pytest.raises(ToolException, match="AnalyzeImage VLM call failed"),
        ):
            AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

    def test_analyze_image_adds_note_for_unexpected_region_type(self) -> None:
        """Test that a note is added for unexpected region types."""
        response_payload: dict[str, Any] = {
            "description": "Test",
            "objects": [],
        }

        regions = create_regions(
            create_product_image(),
            region_type=RegionType.TABLE,  # Unexpected type for image
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["notes"] is not None
        assert "table" in result["notes"].lower()

    def test_analyze_image_property_photo(self) -> None:
        """Test analysis of a property photo."""
        response_payload: dict[str, Any] = {
            "description": "A residential building with yard",
            "objects": [
                {
                    "object_type": "property",
                    "description": "Single-family house",
                    "count": 1,
                    "attributes": {"type": "residential", "stories": "1"},
                    "confidence": "high",
                }
            ],
            "total_items": 1,
            "condition_assessment": "good",
            "extracted_text": None,
            "notes": "Building appears well-maintained.",
        }

        regions = create_regions(create_property_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["objects"][0]["object_type"] == "property"
        assert result["condition_assessment"] == "good"

    def test_analyze_image_low_quality(self) -> None:
        """Test handling of low quality images."""
        response_payload: dict[str, Any] = {
            "description": "Blurry image, difficult to discern details",
            "objects": [
                {
                    "object_type": "other",
                    "description": "Indistinct shapes",
                    "count": None,
                    "attributes": None,
                    "confidence": "low",
                }
            ],
            "total_items": None,
            "condition_assessment": None,
            "extracted_text": None,
            "notes": "Image quality is too low for reliable analysis.",
        }

        regions = create_regions(create_low_quality_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["objects"][0]["confidence"] == "low"
        assert result["notes"] is not None

    def test_analyze_image_empty_scene(self) -> None:
        """Test handling of empty/minimal scene."""
        response_payload: dict[str, Any] = {
            "description": "An empty table surface",
            "objects": [],
            "total_items": 0,
            "condition_assessment": None,
            "extracted_text": None,
            "notes": "Scene appears to be empty.",
        }

        regions = create_regions(create_empty_scene_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert result["objects"] == []
        assert result["total_items"] == 0

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_image_integration_product(self) -> None:
        """Integration test with product image."""
        regions = create_regions(create_product_image())
        result = AnalyzeImage.invoke({"region_id": "region-1", "regions": regions})

        assert "description" in result
        assert "objects" in result
        assert isinstance(result["objects"], list)

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_image_integration_with_focus(self) -> None:
        """Integration test with focus parameter."""
        regions = create_regions(create_multiple_items_image())
        result = AnalyzeImage.invoke(
            {"region_id": "region-1", "regions": regions, "focus": "count all items"}
        )

        assert "description" in result
        assert "total_items" in result

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_image_integration_damage(self) -> None:
        """Integration test with damage assessment image."""
        regions = create_regions(create_damage_assessment_image())
        result = AnalyzeImage.invoke(
            {"region_id": "region-1", "regions": regions, "focus": "assess damage"}
        )

        assert "description" in result
        assert "condition_assessment" in result


class TestAnalyzeImageImpl:
    """Tests for the analyze_image_impl helper function."""

    def test_impl_parses_response(self) -> None:
        """Test that impl function parses VLM response correctly."""
        response_payload: dict[str, Any] = {
            "description": "A product photo",
            "objects": [
                {
                    "object_type": "product",
                    "description": "Red box",
                    "count": 1,
                    "attributes": {"color": "red"},
                    "confidence": "high",
                }
            ],
            "total_items": 1,
            "condition_assessment": "excellent",
            "extracted_text": ["Brand X"],
            "notes": "Product in new condition.",
        }
        regions = create_regions(create_product_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = analyze_image_impl("region-1", regions)

        assert result["description"] == "A product photo"
        assert len(result["objects"]) == 1
        assert result["objects"][0]["object_type"] == "product"
        assert result["condition_assessment"] == "excellent"

    def test_impl_with_focus(self) -> None:
        """Test impl function with focus parameter."""
        response_payload: dict[str, Any] = {
            "description": "Items to count",
            "objects": [],
            "total_items": 5,
        }
        regions = create_regions(create_multiple_items_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_image.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ) as mock_vlm:
            result = analyze_image_impl("region-1", regions, focus="count items")

            # Verify focus prompt was used
            prompt = mock_vlm.call_args[0][1]
            assert "count items" in prompt

        assert result["total_items"] == 5

    def test_impl_raises_on_missing_region(self) -> None:
        """Test error when region ID not found."""
        regions = create_regions(create_product_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            analyze_image_impl("nonexistent", regions)

    def test_impl_raises_on_missing_image(self) -> None:
        """Test error when region has no image."""
        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.PICTURE,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="no-img",
                region_image=None,
            )
        ]
        with pytest.raises(ToolException, match="Region image not provided"):
            analyze_image_impl("no-img", regions)

    def test_impl_raises_on_empty_image_data(self) -> None:
        """Test error when region image has no data."""
        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.PICTURE,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="empty-img",
                region_image=RegionImage(image=None, base64=None),
            )
        ]
        with pytest.raises(ToolException, match="Region image missing"):
            analyze_image_impl("empty-img", regions)


class TestNormalizeImageResult:
    """Tests for the _normalize_image_result helper function."""

    def test_normalize_empty_dict(self) -> None:
        """Test normalization of empty result dict."""
        result = _normalize_image_result({})
        assert result["description"] == ""
        assert result["objects"] == []
        assert result["total_items"] is None
        assert result["condition_assessment"] is None
        assert result["extracted_text"] is None
        assert result["notes"] is None

    def test_normalize_description(self) -> None:
        """Test normalization of description."""
        result = _normalize_image_result({"description": "  A test image  "})
        assert result["description"] == "A test image"

    def test_normalize_description_empty(self) -> None:
        """Test that empty description becomes empty string."""
        result = _normalize_image_result({"description": ""})
        assert result["description"] == ""

        result = _normalize_image_result({"description": None})
        assert result["description"] == ""

    def test_normalize_objects_list(self) -> None:
        """Test normalization of objects list."""
        result = _normalize_image_result(
            {
                "objects": [
                    {"object_type": "product", "description": "Test"},
                    {"object_type": "item", "description": "Another"},
                ]
            }
        )
        assert len(result["objects"]) == 2
        assert result["objects"][0]["object_type"] == "product"

    def test_normalize_objects_filters_non_dicts(self) -> None:
        """Test that non-dict objects are filtered out."""
        result = _normalize_image_result(
            {
                "objects": [
                    {"object_type": "product", "description": "Test"},
                    "not a dict",
                    123,
                ]
            }
        )
        assert len(result["objects"]) == 1

    def test_normalize_objects_non_list(self) -> None:
        """Test that non-list objects becomes empty list."""
        result = _normalize_image_result({"objects": "not a list"})
        assert result["objects"] == []

    def test_normalize_total_items_integer(self) -> None:
        """Test total_items normalization to integer."""
        result = _normalize_image_result({"total_items": 5})
        assert result["total_items"] == 5

        result = _normalize_image_result({"total_items": "10"})
        assert result["total_items"] == 10

    def test_normalize_total_items_invalid(self) -> None:
        """Test invalid total_items becomes None."""
        result = _normalize_image_result({"total_items": "not a number"})
        assert result["total_items"] is None

    def test_normalize_condition_assessment_validation(self) -> None:
        """Test condition assessment validation."""
        valid_conditions = ["excellent", "good", "fair", "poor", "damaged"]
        for condition in valid_conditions:
            result = _normalize_image_result({"condition_assessment": condition})
            assert result["condition_assessment"] == condition

        # Case insensitive
        result = _normalize_image_result({"condition_assessment": "EXCELLENT"})
        assert result["condition_assessment"] == "excellent"

        # Invalid becomes None
        result = _normalize_image_result({"condition_assessment": "bad"})
        assert result["condition_assessment"] is None

    def test_normalize_extracted_text_list(self) -> None:
        """Test normalization of extracted text list."""
        result = _normalize_image_result({"extracted_text": ["text1", "  text2  ", ""]})
        assert result["extracted_text"] == ["text1", "text2"]

    def test_normalize_extracted_text_empty_list(self) -> None:
        """Test that empty extracted text list becomes None."""
        result = _normalize_image_result({"extracted_text": []})
        assert result["extracted_text"] is None

        result = _normalize_image_result({"extracted_text": ["", "  "]})
        assert result["extracted_text"] is None

    def test_normalize_extracted_text_non_list(self) -> None:
        """Test that non-list extracted text becomes None."""
        result = _normalize_image_result({"extracted_text": "single"})
        assert result["extracted_text"] is None

    def test_normalize_notes_strips_whitespace(self) -> None:
        """Test that notes are stripped of whitespace."""
        result = _normalize_image_result({"notes": "  Some notes here.  "})
        assert result["notes"] == "Some notes here."

    def test_normalize_notes_empty_becomes_none(self) -> None:
        """Test that empty notes become None."""
        result = _normalize_image_result({"notes": ""})
        assert result["notes"] is None

        result = _normalize_image_result({"notes": "   "})
        assert result["notes"] is None


class TestNormalizeObject:
    """Tests for the _normalize_object helper function."""

    def test_normalize_object_empty(self) -> None:
        """Test normalization of empty object."""
        result = _normalize_object({})
        assert result["object_type"] == "other"
        assert result["description"] == ""
        assert result["count"] is None
        assert result["attributes"] is None
        assert result["confidence"] == "low"

    def test_normalize_object_type_validation(self) -> None:
        """Test object type validation."""
        valid_types = [
            "product",
            "item",
            "equipment",
            "damage",
            "vehicle",
            "property",
            "document",
            "person",
            "other",
        ]
        for obj_type in valid_types:
            result = _normalize_object({"object_type": obj_type})
            assert result["object_type"] == obj_type

        # Case insensitive
        result = _normalize_object({"object_type": "PRODUCT"})
        assert result["object_type"] == "product"

        # Invalid becomes "other"
        result = _normalize_object({"object_type": "unknown_type"})
        assert result["object_type"] == "other"

    def test_normalize_object_description(self) -> None:
        """Test object description normalization."""
        result = _normalize_object({"description": "  Test object  "})
        assert result["description"] == "Test object"

        result = _normalize_object({"description": ""})
        assert result["description"] == ""

    def test_normalize_object_count_integer(self) -> None:
        """Test object count normalization."""
        result = _normalize_object({"count": 5})
        assert result["count"] == 5

        result = _normalize_object({"count": "10"})
        assert result["count"] == 10

        result = _normalize_object({"count": "not a number"})
        assert result["count"] is None

    def test_normalize_object_attributes_dict(self) -> None:
        """Test object attributes normalization."""
        result = _normalize_object(
            {"attributes": {"color": "red", "size": "large", "empty": None}}
        )
        # None values should be filtered out
        assert result["attributes"] == {"color": "red", "size": "large"}

    def test_normalize_object_attributes_empty(self) -> None:
        """Test that empty attributes becomes None."""
        result = _normalize_object({"attributes": {}})
        assert result["attributes"] is None

    def test_normalize_object_attributes_non_dict(self) -> None:
        """Test that non-dict attributes becomes None."""
        result = _normalize_object({"attributes": "not a dict"})
        assert result["attributes"] is None

    def test_normalize_object_confidence_validation(self) -> None:
        """Test object confidence validation."""
        for level in ("high", "medium", "low"):
            result = _normalize_object({"confidence": level})
            assert result["confidence"] == level

        # Case insensitive
        result = _normalize_object({"confidence": "HIGH"})
        assert result["confidence"] == "high"

        # Invalid becomes low
        result = _normalize_object({"confidence": "very_high"})
        assert result["confidence"] == "low"
