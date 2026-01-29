"""Tests for the AnalyzeForm tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_form import (
    AnalyzeForm,
    _normalize_fields,
    analyze_form_impl,
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


def create_simple_form_image() -> Image.Image:
    """Create a simple form image with text fields."""
    image = Image.new("RGB", (500, 400), color="white")
    draw = ImageDraw.Draw(image)

    # Title
    draw.text((150, 20), "Registration Form", fill="black")

    # Name field
    draw.text((30, 70), "Name: *", fill="black")
    draw.rectangle((120, 65, 450, 90), outline="black")
    draw.text((130, 70), "John Doe", fill="blue")

    # Email field
    draw.text((30, 120), "Email:", fill="black")
    draw.rectangle((120, 115, 450, 140), outline="black")
    draw.text((130, 120), "john@example.com", fill="blue")

    # Date field
    draw.text((30, 170), "Date: *", fill="black")
    draw.rectangle((120, 165, 250, 190), outline="black")
    draw.text((130, 170), "01/15/2024", fill="blue")

    return image


def create_checkbox_form_image() -> Image.Image:
    """Create a form image with checkboxes."""
    image = Image.new("RGB", (500, 350), color="white")
    draw = ImageDraw.Draw(image)

    # Title
    draw.text((150, 20), "Preferences", fill="black")

    # Checkbox 1 - checked
    draw.rectangle((30, 70, 50, 90), outline="black")
    draw.line((32, 80, 40, 88), fill="black", width=2)
    draw.line((40, 88, 48, 72), fill="black", width=2)
    draw.text((60, 70), "Subscribe to newsletter", fill="black")

    # Checkbox 2 - unchecked
    draw.rectangle((30, 110, 50, 130), outline="black")
    draw.text((60, 110), "Enable notifications", fill="black")

    # Checkbox 3 - checked with X
    draw.rectangle((30, 150, 50, 170), outline="black")
    draw.line((32, 152, 48, 168), fill="black", width=2)
    draw.line((32, 168, 48, 152), fill="black", width=2)
    draw.text((60, 150), "I agree to terms *", fill="black")

    return image


def create_radio_form_image() -> Image.Image:
    """Create a form image with radio buttons."""
    image = Image.new("RGB", (500, 300), color="white")
    draw = ImageDraw.Draw(image)

    # Title
    draw.text((150, 20), "Select Payment Method", fill="black")

    # Radio 1 - selected
    draw.ellipse((30, 70, 50, 90), outline="black")
    draw.ellipse((35, 75, 45, 85), fill="black")
    draw.text((60, 70), "Credit Card", fill="black")

    # Radio 2 - unselected
    draw.ellipse((30, 110, 50, 130), outline="black")
    draw.text((60, 110), "PayPal", fill="black")

    # Radio 3 - unselected
    draw.ellipse((30, 150, 50, 170), outline="black")
    draw.text((60, 150), "Bank Transfer", fill="black")

    return image


def create_mixed_form_image() -> Image.Image:
    """Create a complex form with mixed field types."""
    image = Image.new("RGB", (600, 500), color="white")
    draw = ImageDraw.Draw(image)

    # Title
    draw.text((200, 20), "Application Form", fill="black")

    # Text field
    draw.text((30, 70), "Full Name: *", fill="black")
    draw.rectangle((150, 65, 550, 95), outline="black")
    draw.text((160, 72), "Jane Smith", fill="blue")

    # Date field
    draw.text((30, 120), "Birth Date:", fill="black")
    draw.rectangle((150, 115, 300, 145), outline="black")
    draw.text((160, 122), "03/22/1990", fill="blue")

    # Dropdown (simulated)
    draw.text((30, 170), "Country:", fill="black")
    draw.rectangle((150, 165, 350, 195), outline="black")
    draw.text((160, 172), "United States", fill="black")
    draw.polygon([(330, 175), (340, 175), (335, 185)], fill="black")

    # Checkbox
    draw.rectangle((30, 220, 50, 240), outline="black")
    draw.line((32, 230, 40, 238), fill="black", width=2)
    draw.line((40, 238, 48, 222), fill="black", width=2)
    draw.text((60, 220), "I certify the information is correct *", fill="black")

    # Signature field
    draw.text((30, 280), "Signature:", fill="black")
    draw.rectangle((150, 275, 400, 340), outline="black")
    draw.text((160, 300), "~signature~", fill="blue")

    return image


def create_regions(
    image: Image.Image,
    *,
    region_id: str = "region-1",
    region_type: RegionType = RegionType.TEXT,
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


class TestAnalyzeFormTool:
    """Tests for the AnalyzeForm tool function."""

    def test_analyze_form_parses_text_fields(self) -> None:
        """Test parsing form response with text fields."""
        response_payload: dict[str, Any] = {
            "fields": [
                {
                    "label": "Name",
                    "value": "John Doe",
                    "field_type": "text",
                    "is_handwritten": True,
                    "is_required": True,
                    "position": "top-left",
                },
                {
                    "label": "Email",
                    "value": "john@example.com",
                    "field_type": "text",
                    "is_handwritten": False,
                    "is_required": False,
                    "position": "middle-left",
                },
            ],
            "form_title": "Registration Form",
            "notes": None,
        }

        regions = create_regions(create_simple_form_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_form.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

        assert len(result["fields"]) == 2
        assert result["fields"][0]["label"] == "Name"
        assert result["fields"][0]["value"] == "John Doe"
        assert result["fields"][0]["is_handwritten"] is True
        assert result["form_title"] == "Registration Form"

    def test_analyze_form_parses_checkboxes(self) -> None:
        """Test parsing form response with checkbox fields."""
        response_payload: dict[str, Any] = {
            "fields": [
                {
                    "label": "Subscribe to newsletter",
                    "value": True,
                    "field_type": "checkbox",
                    "is_handwritten": False,
                    "is_required": False,
                },
                {
                    "label": "Enable notifications",
                    "value": False,
                    "field_type": "checkbox",
                    "is_handwritten": False,
                    "is_required": False,
                },
                {
                    "label": "I agree to terms",
                    "value": True,
                    "field_type": "checkbox",
                    "is_handwritten": False,
                    "is_required": True,
                },
            ],
            "form_title": "Preferences",
            "notes": None,
        }

        regions = create_regions(create_checkbox_form_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_form.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

        assert len(result["fields"]) == 3
        assert result["fields"][0]["field_type"] == "checkbox"
        assert result["fields"][0]["value"] is True
        assert result["fields"][1]["value"] is False
        assert result["fields"][2]["is_required"] is True

    def test_analyze_form_parses_radio_buttons(self) -> None:
        """Test parsing form response with radio button fields."""
        response_payload: dict[str, Any] = {
            "fields": [
                {
                    "label": "Payment Method",
                    "value": "Credit Card",
                    "field_type": "radio",
                    "is_handwritten": False,
                    "is_required": True,
                },
            ],
            "form_title": "Select Payment Method",
            "notes": None,
        }

        regions = create_regions(create_radio_form_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_form.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

        assert len(result["fields"]) == 1
        assert result["fields"][0]["field_type"] == "radio"
        assert result["fields"][0]["value"] == "Credit Card"

    def test_analyze_form_handles_wrapped_json(self) -> None:
        """Test handling of wrapped JSON in VLM response."""
        wrapped = (
            "Here is the form analysis:\n"
            + json.dumps(
                {
                    "fields": [
                        {"label": "Name", "value": "Test", "field_type": "text"}
                    ],
                    "form_title": "Test Form",
                }
            )
            + "\nEnd of analysis."
        )
        regions = create_regions(create_simple_form_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_form.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

        assert len(result["fields"]) == 1
        assert result["fields"][0]["label"] == "Name"
        assert result["form_title"] == "Test Form"

    def test_analyze_form_invalid_region_id(self) -> None:
        """Test error handling for invalid region ID."""
        regions = create_regions(create_simple_form_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            AnalyzeForm.invoke({"region_id": "nonexistent", "regions": regions})

    def test_analyze_form_vlm_failure(self) -> None:
        """Test error handling when VLM call fails."""
        regions = create_regions(create_simple_form_image())
        with (
            patch(
                "agentic_document_extraction.agents.tools.analyze_form.call_vlm_with_image",
                side_effect=RuntimeError("VLM service unavailable"),
            ),
            pytest.raises(ToolException, match="AnalyzeForm VLM call failed"),
        ):
            AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

    def test_analyze_form_adds_note_for_unexpected_region_type(self) -> None:
        """Test that a note is added for unexpected region types."""
        response_payload: dict[str, Any] = {
            "fields": [{"label": "Test", "value": "Value", "field_type": "text"}],
            "form_title": None,
            "notes": None,
        }

        regions = create_regions(
            create_simple_form_image(),
            region_type=RegionType.FORMULA,  # Unexpected type for a form
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_form.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

        assert result["notes"] is not None
        assert "formula" in result["notes"].lower()

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_form_integration_simple(self) -> None:
        """Integration test with a simple form image."""
        regions = create_regions(create_simple_form_image())
        result = AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

        assert "fields" in result
        assert isinstance(result["fields"], list)

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_form_integration_checkboxes(self) -> None:
        """Integration test with checkbox form image."""
        regions = create_regions(create_checkbox_form_image())
        result = AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

        assert "fields" in result
        assert isinstance(result["fields"], list)

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_form_integration_mixed(self) -> None:
        """Integration test with mixed form fields."""
        regions = create_regions(create_mixed_form_image())
        result = AnalyzeForm.invoke({"region_id": "region-1", "regions": regions})

        assert "fields" in result
        assert isinstance(result["fields"], list)


class TestAnalyzeFormImpl:
    """Tests for the analyze_form_impl helper function."""

    def test_impl_parses_response(self) -> None:
        """Test that impl function parses VLM response correctly."""
        response_payload: dict[str, Any] = {
            "fields": [
                {
                    "label": "Full Name",
                    "value": "Test User",
                    "field_type": "text",
                    "is_handwritten": True,
                    "is_required": True,
                }
            ],
            "form_title": "Test Form",
            "notes": "Test notes",
        }
        regions = create_regions(create_simple_form_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_form.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = analyze_form_impl("region-1", regions)

        assert result["form_title"] == "Test Form"
        assert len(result["fields"]) == 1
        assert result["fields"][0]["label"] == "Full Name"
        assert result["notes"] == "Test notes"

    def test_impl_raises_on_missing_region(self) -> None:
        """Test error when region ID not found."""
        regions = create_regions(create_simple_form_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            analyze_form_impl("nonexistent", regions)

    def test_impl_raises_on_missing_image(self) -> None:
        """Test error when region has no image."""
        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.TEXT,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="no-img",
                region_image=None,
            )
        ]
        with pytest.raises(ToolException, match="Region image not provided"):
            analyze_form_impl("no-img", regions)

    def test_impl_raises_on_empty_image_data(self) -> None:
        """Test error when region image has no data."""
        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.TEXT,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="empty-img",
                region_image=RegionImage(image=None, base64=None),
            )
        ]
        with pytest.raises(ToolException, match="Region image missing"):
            analyze_form_impl("empty-img", regions)


class TestNormalizeFields:
    """Tests for the _normalize_fields helper function."""

    def test_normalize_empty_list(self) -> None:
        """Test normalization of empty field list."""
        result = _normalize_fields([])
        assert result == []

    def test_normalize_invalid_input(self) -> None:
        """Test normalization of invalid input types."""
        assert _normalize_fields(None) == []
        assert _normalize_fields("not a list") == []
        assert _normalize_fields(123) == []

    def test_normalize_filters_invalid_items(self) -> None:
        """Test that non-dict items are filtered out."""
        fields = [
            {"label": "Valid", "value": "test"},
            "invalid string",
            123,
            None,
        ]
        result = _normalize_fields(fields)
        assert len(result) == 1
        assert result[0]["label"] == "Valid"

    def test_normalize_provides_defaults(self) -> None:
        """Test that missing fields get default values."""
        fields = [{"label": "Test"}]  # Minimal field
        result = _normalize_fields(fields)

        assert result[0]["label"] == "Test"
        assert result[0]["value"] is None
        assert result[0]["field_type"] == "text"
        assert result[0]["is_handwritten"] is False
        assert result[0]["is_required"] is None
        assert result[0]["position"] is None

    def test_normalize_invalid_field_type(self) -> None:
        """Test that invalid field types default to 'text'."""
        fields = [{"label": "Test", "field_type": "invalid_type"}]
        result = _normalize_fields(fields)
        assert result[0]["field_type"] == "text"

    def test_normalize_checkbox_string_values(self) -> None:
        """Test checkbox value string normalization."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("checked", True),
            ("yes", True),
            ("selected", True),
            ("x", True),
            ("false", False),
            ("False", False),
            ("unchecked", False),
            ("no", False),
            ("", False),
        ]

        for string_val, expected in test_cases:
            fields = [{"label": "CB", "value": string_val, "field_type": "checkbox"}]
            result = _normalize_fields(fields)
            assert result[0]["value"] is expected, (
                f"Expected {expected} for '{string_val}'"
            )

    def test_normalize_preserves_checkbox_boolean(self) -> None:
        """Test that boolean checkbox values are preserved."""
        fields = [
            {"label": "CB1", "value": True, "field_type": "checkbox"},
            {"label": "CB2", "value": False, "field_type": "checkbox"},
        ]
        result = _normalize_fields(fields)
        assert result[0]["value"] is True
        assert result[1]["value"] is False

    def test_normalize_radio_with_option_text(self) -> None:
        """Test that radio button string values are preserved for option text."""
        fields = [{"label": "Payment", "value": "Credit Card", "field_type": "radio"}]
        result = _normalize_fields(fields)
        # Option text should be preserved as-is
        assert result[0]["value"] == "Credit Card"

    def test_normalize_valid_field_types(self) -> None:
        """Test that all valid field types are accepted."""
        valid_types = ["text", "checkbox", "radio", "dropdown", "signature", "date"]
        for field_type in valid_types:
            fields = [{"label": "Test", "field_type": field_type}]
            result = _normalize_fields(fields)
            assert result[0]["field_type"] == field_type
