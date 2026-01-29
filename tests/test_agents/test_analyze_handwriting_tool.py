"""Tests for the AnalyzeHandwriting tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_handwriting import (
    AnalyzeHandwriting,
    _normalize_handwriting_result,
    analyze_handwriting_impl,
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


def create_cursive_handwriting_image() -> Image.Image:
    """Create a simple image simulating cursive handwriting."""
    image = Image.new("RGB", (400, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Simulate cursive text with connected wavy lines
    # "Hello World" in cursive style
    points = [
        (30, 80),
        (40, 60),
        (50, 80),
        (60, 60),
        (70, 80),  # H
        (80, 60),
        (90, 80),
        (100, 60),  # e
        (110, 40),
        (120, 80),  # l
        (130, 40),
        (140, 80),  # l
        (150, 60),
        (160, 80),
        (170, 60),  # o
        (200, 70),  # space
        (210, 60),
        (220, 80),
        (230, 60),
        (240, 80),  # W
        (250, 60),
        (260, 80),
        (270, 60),  # o
        (280, 40),
        (290, 80),
        (300, 60),  # r
        (310, 40),
        (320, 80),  # l
        (330, 60),
        (340, 80),
        (350, 60),  # d
    ]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="blue", width=2)

    return image


def create_print_handwriting_image() -> Image.Image:
    """Create an image simulating printed handwriting."""
    image = Image.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Simulate printed text: "TEST"
    # T
    draw.line((30, 30, 70, 30), fill="black", width=2)
    draw.line((50, 30, 50, 70), fill="black", width=2)
    # E
    draw.line((90, 30, 90, 70), fill="black", width=2)
    draw.line((90, 30, 120, 30), fill="black", width=2)
    draw.line((90, 50, 110, 50), fill="black", width=2)
    draw.line((90, 70, 120, 70), fill="black", width=2)
    # S
    draw.arc((140, 30, 170, 50), 180, 0, fill="black", width=2)
    draw.arc((140, 50, 170, 70), 0, 180, fill="black", width=2)
    # T
    draw.line((190, 30, 230, 30), fill="black", width=2)
    draw.line((210, 30, 210, 70), fill="black", width=2)

    return image


def create_margin_note_image() -> Image.Image:
    """Create an image with a margin note style annotation."""
    image = Image.new("RGB", (150, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Simulate margin note with arrow
    draw.line((10, 50, 50, 50), fill="black", width=1)
    draw.polygon([(50, 45), (60, 50), (50, 55)], fill="black")

    # Handwritten note
    points = [(20, 80), (30, 70), (40, 80), (50, 70), (60, 80)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="red", width=2)

    # Text representing "See pg 5"
    draw.text((20, 100), "See pg 5", fill="red")

    return image


def create_correction_image() -> Image.Image:
    """Create an image with a correction/strikethrough and replacement."""
    image = Image.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Original text with strikethrough
    draw.text((30, 30), "incorrect", fill="gray")
    draw.line((30, 40, 120, 40), fill="red", width=2)

    # Handwritten correction above
    points = [(40, 20), (50, 15), (60, 20), (70, 15), (80, 20)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="blue", width=2)

    return image


def create_form_answer_image() -> Image.Image:
    """Create an image with a handwritten form answer."""
    image = Image.new("RGB", (350, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Form field line
    draw.text((20, 20), "Name:", fill="black")
    draw.line((80, 50, 320, 50), fill="black", width=1)

    # Handwritten answer
    points = [
        (90, 45),
        (100, 35),
        (110, 45),
        (120, 35),
        (130, 45),
        (150, 45),
        (160, 35),
        (170, 45),
    ]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="blue", width=2)

    return image


def create_illegible_handwriting_image() -> Image.Image:
    """Create an image with illegible/messy handwriting."""
    image = Image.new("RGB", (300, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Random scribbles that are hard to read
    import random

    random.seed(42)  # For reproducibility
    for _ in range(20):
        x1 = random.randint(20, 280)
        y1 = random.randint(30, 120)
        x2 = x1 + random.randint(-30, 30)
        y2 = y1 + random.randint(-20, 20)
        draw.line((x1, y1, x2, y2), fill="black", width=random.randint(1, 3))

    return image


def create_mixed_style_image() -> Image.Image:
    """Create an image with mixed cursive and print handwriting."""
    image = Image.new("RGB", (400, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Printed text on top
    draw.text((30, 30), "DATE:", fill="black")

    # Cursive answer
    points = [
        (100, 45),
        (110, 35),
        (120, 45),
        (130, 35),
        (140, 45),
        (150, 35),
        (160, 45),
        (170, 35),
    ]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="blue", width=2)

    # More printed text
    draw.text((200, 30), "NAME:", fill="black")

    # More cursive
    points2 = [(270, 45), (280, 35), (290, 45), (300, 35), (310, 45)]
    for i in range(len(points2) - 1):
        draw.line([points2[i], points2[i + 1]], fill="blue", width=2)

    return image


def create_numbers_image() -> Image.Image:
    """Create an image with handwritten numbers."""
    image = Image.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Simulate handwritten numbers: "12345"
    draw.text((50, 40), "12345", fill="blue")
    # Add some wavy underlines to make it look handwritten
    draw.line((48, 65, 58, 63, 68, 67, 78, 63, 88, 65), fill="blue", width=1)

    return image


def create_signature_text_image() -> Image.Image:
    """Create an image with signature and accompanying text."""
    image = Image.new("RGB", (400, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Signature (wavy line)
    points = [(50, 60), (70, 50), (100, 65), (130, 45), (160, 60)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="blue", width=2)

    # Printed name below signature
    draw.text((80, 80), "John Doe", fill="black")

    # Date
    draw.text((80, 100), "Jan 15, 2024", fill="blue")

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


class TestAnalyzeHandwritingTool:
    """Tests for the AnalyzeHandwriting tool function."""

    def test_analyze_handwriting_transcribes_cursive(self) -> None:
        """Test transcription of cursive handwriting."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "Hello World",
            "confidence": "high",
            "annotation_type": "other",
            "position": "full_page",
            "is_legible": True,
            "alternative_readings": None,
            "style": "cursive",
            "notes": "Neat cursive handwriting in blue ink.",
        }

        regions = create_regions(create_cursive_handwriting_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["transcribed_text"] == "Hello World"
        assert result["confidence"] == "high"
        assert result["style"] == "cursive"
        assert result["is_legible"] is True

    def test_analyze_handwriting_transcribes_print(self) -> None:
        """Test transcription of printed handwriting."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "TEST",
            "confidence": "high",
            "annotation_type": "other",
            "position": "inline",
            "is_legible": True,
            "alternative_readings": None,
            "style": "print",
            "notes": "Clear printed letters.",
        }

        regions = create_regions(create_print_handwriting_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["transcribed_text"] == "TEST"
        assert result["style"] == "print"
        assert result["is_legible"] is True

    def test_analyze_handwriting_identifies_margin_note(self) -> None:
        """Test identification of margin note annotation type."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "See pg 5",
            "confidence": "high",
            "annotation_type": "margin_note",
            "position": "right_margin",
            "is_legible": True,
            "alternative_readings": None,
            "style": "cursive",
            "notes": "Red ink margin note with arrow.",
        }

        regions = create_regions(create_margin_note_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["transcribed_text"] == "See pg 5"
        assert result["annotation_type"] == "margin_note"
        assert result["position"] == "right_margin"

    def test_analyze_handwriting_identifies_correction(self) -> None:
        """Test identification of correction annotation type."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "correct",
            "confidence": "medium",
            "annotation_type": "correction",
            "position": "inline",
            "is_legible": True,
            "alternative_readings": None,
            "style": "cursive",
            "notes": "Blue ink correction above struck-through text.",
        }

        regions = create_regions(create_correction_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["annotation_type"] == "correction"
        assert result["confidence"] == "medium"

    def test_analyze_handwriting_identifies_form_answer(self) -> None:
        """Test identification of form answer annotation type."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "John Smith",
            "confidence": "high",
            "annotation_type": "answer",
            "position": "inline",
            "is_legible": True,
            "alternative_readings": None,
            "style": "cursive",
            "notes": "Handwritten name on form field line.",
        }

        regions = create_regions(create_form_answer_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["transcribed_text"] == "John Smith"
        assert result["annotation_type"] == "answer"

    def test_analyze_handwriting_handles_illegible_text(self) -> None:
        """Test handling of illegible handwriting."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "unclear scribbles",
            "confidence": "low",
            "annotation_type": "other",
            "position": "full_page",
            "is_legible": False,
            "alternative_readings": ["unclear scribbles", "messy writing"],
            "style": "unknown",
            "notes": "Handwriting is too messy to reliably transcribe.",
        }

        regions = create_regions(create_illegible_handwriting_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["is_legible"] is False
        assert result["confidence"] == "low"
        assert result["alternative_readings"] is not None

    def test_analyze_handwriting_with_alternative_readings(self) -> None:
        """Test that alternative readings are returned for uncertain text."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "Dr. Johnson",
            "confidence": "medium",
            "annotation_type": "signature_text",
            "position": "bottom",
            "is_legible": True,
            "alternative_readings": ["Dr. Johnston", "Dr. Jonson"],
            "style": "cursive",
            "notes": "Surname is slightly unclear.",
        }

        regions = create_regions(create_signature_text_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["alternative_readings"] == ["Dr. Johnston", "Dr. Jonson"]
        assert result["confidence"] == "medium"

    def test_analyze_handwriting_mixed_style(self) -> None:
        """Test detection of mixed handwriting style."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "DATE: January 15\nNAME: John",
            "confidence": "high",
            "annotation_type": "answer",
            "position": "full_page",
            "is_legible": True,
            "alternative_readings": None,
            "style": "mixed",
            "notes": "Printed labels with cursive answers.",
        }

        regions = create_regions(create_mixed_style_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["style"] == "mixed"

    def test_analyze_handwriting_with_context(self) -> None:
        """Test that context is used in the prompt when provided."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "Rx: Amoxicillin 500mg",
            "confidence": "high",
            "annotation_type": "other",
            "position": "full_page",
            "is_legible": True,
            "alternative_readings": None,
            "style": "cursive",
            "notes": "Medical prescription context helped identify drug name.",
        }

        regions = create_regions(create_cursive_handwriting_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ) as mock_vlm:
            result = AnalyzeHandwriting.invoke(
                {
                    "region_id": "region-1",
                    "regions": regions,
                    "context": "This is a medical prescription from Dr. Smith",
                }
            )

            # Verify context was included in the prompt
            call_args = mock_vlm.call_args
            prompt = call_args[0][1]
            assert "medical prescription" in prompt.lower()

        assert result["transcribed_text"] == "Rx: Amoxicillin 500mg"

    def test_analyze_handwriting_handles_wrapped_json(self) -> None:
        """Test handling of wrapped JSON in VLM response."""
        wrapped = (
            "Here is the transcription:\n"
            + json.dumps(
                {
                    "transcribed_text": "Test text",
                    "confidence": "high",
                    "is_legible": True,
                }
            )
            + "\nEnd of transcription."
        )
        regions = create_regions(create_cursive_handwriting_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["transcribed_text"] == "Test text"
        assert result["confidence"] == "high"

    def test_analyze_handwriting_invalid_region_id(self) -> None:
        """Test error handling for invalid region ID."""
        regions = create_regions(create_cursive_handwriting_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            AnalyzeHandwriting.invoke({"region_id": "nonexistent", "regions": regions})

    def test_analyze_handwriting_vlm_failure(self) -> None:
        """Test error handling when VLM call fails."""
        regions = create_regions(create_cursive_handwriting_image())
        with (
            patch(
                "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
                side_effect=RuntimeError("VLM service unavailable"),
            ),
            pytest.raises(ToolException, match="AnalyzeHandwriting VLM call failed"),
        ):
            AnalyzeHandwriting.invoke({"region_id": "region-1", "regions": regions})

    def test_analyze_handwriting_adds_note_for_unexpected_region_type(self) -> None:
        """Test that a note is added for unexpected region types."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "Test",
            "confidence": "high",
            "is_legible": True,
        }

        regions = create_regions(
            create_cursive_handwriting_image(),
            region_type=RegionType.TABLE,  # Unexpected type for handwriting
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeHandwriting.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["notes"] is not None
        assert "table" in result["notes"].lower()

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_handwriting_integration_cursive(self) -> None:
        """Integration test with cursive handwriting image."""
        regions = create_regions(create_cursive_handwriting_image())
        result = AnalyzeHandwriting.invoke(
            {"region_id": "region-1", "regions": regions}
        )

        assert "transcribed_text" in result
        assert "confidence" in result
        assert result["confidence"] in ("high", "medium", "low")

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_handwriting_integration_print(self) -> None:
        """Integration test with printed handwriting image."""
        regions = create_regions(create_print_handwriting_image())
        result = AnalyzeHandwriting.invoke(
            {"region_id": "region-1", "regions": regions}
        )

        assert "transcribed_text" in result
        assert "style" in result

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_handwriting_integration_with_context(self) -> None:
        """Integration test with context provided."""
        regions = create_regions(create_form_answer_image())
        result = AnalyzeHandwriting.invoke(
            {
                "region_id": "region-1",
                "regions": regions,
                "context": "This is a name field on a job application form.",
            }
        )

        assert "transcribed_text" in result
        assert "annotation_type" in result


class TestAnalyzeHandwritingImpl:
    """Tests for the analyze_handwriting_impl helper function."""

    def test_impl_parses_response(self) -> None:
        """Test that impl function parses VLM response correctly."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "Hello World",
            "confidence": "high",
            "annotation_type": "margin_note",
            "position": "right_margin",
            "is_legible": True,
            "alternative_readings": ["Hello Word"],
            "style": "cursive",
            "notes": "Blue ink cursive.",
        }
        regions = create_regions(create_cursive_handwriting_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = analyze_handwriting_impl("region-1", regions)

        assert result["transcribed_text"] == "Hello World"
        assert result["confidence"] == "high"
        assert result["annotation_type"] == "margin_note"
        assert result["style"] == "cursive"
        assert result["alternative_readings"] == ["Hello Word"]

    def test_impl_with_context(self) -> None:
        """Test impl function with context parameter."""
        response_payload: dict[str, Any] = {
            "transcribed_text": "500mg",
            "confidence": "high",
            "is_legible": True,
        }
        regions = create_regions(create_cursive_handwriting_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_handwriting.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ) as mock_vlm:
            result = analyze_handwriting_impl(
                "region-1", regions, context="Medical dosage"
            )

            # Verify context prompt was used
            prompt = mock_vlm.call_args[0][1]
            assert "Medical dosage" in prompt

        assert result["transcribed_text"] == "500mg"

    def test_impl_raises_on_missing_region(self) -> None:
        """Test error when region ID not found."""
        regions = create_regions(create_cursive_handwriting_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            analyze_handwriting_impl("nonexistent", regions)

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
            analyze_handwriting_impl("no-img", regions)

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
            analyze_handwriting_impl("empty-img", regions)


class TestNormalizeHandwritingResult:
    """Tests for the _normalize_handwriting_result helper function."""

    def test_normalize_empty_dict(self) -> None:
        """Test normalization of empty result dict."""
        result = _normalize_handwriting_result({})
        assert result["transcribed_text"] == ""
        assert result["confidence"] == "low"
        assert result["annotation_type"] is None
        assert result["position"] is None
        assert result["is_legible"] is False
        assert result["alternative_readings"] is None
        assert result["style"] is None
        assert result["notes"] is None

    def test_normalize_transcribed_text(self) -> None:
        """Test normalization of transcribed text."""
        result = _normalize_handwriting_result({"transcribed_text": "  Hello World  "})
        assert result["transcribed_text"] == "Hello World"

    def test_normalize_transcribed_text_empty(self) -> None:
        """Test that empty transcribed text becomes empty string."""
        result = _normalize_handwriting_result({"transcribed_text": ""})
        assert result["transcribed_text"] == ""

        result = _normalize_handwriting_result({"transcribed_text": None})
        assert result["transcribed_text"] == ""

    def test_normalize_confidence_validation(self) -> None:
        """Test confidence level validation."""
        # Valid confidence levels
        for level in ("high", "medium", "low"):
            result = _normalize_handwriting_result({"confidence": level})
            assert result["confidence"] == level

        # Case insensitive
        result = _normalize_handwriting_result({"confidence": "HIGH"})
        assert result["confidence"] == "high"

        # Invalid becomes low
        result = _normalize_handwriting_result({"confidence": "very_high"})
        assert result["confidence"] == "low"

    def test_normalize_annotation_type_validation(self) -> None:
        """Test annotation type validation."""
        valid_types = [
            "margin_note",
            "correction",
            "answer",
            "signature_text",
            "comment",
            "label",
            "other",
        ]
        for atype in valid_types:
            result = _normalize_handwriting_result({"annotation_type": atype})
            assert result["annotation_type"] == atype

        # Case insensitive
        result = _normalize_handwriting_result({"annotation_type": "MARGIN_NOTE"})
        assert result["annotation_type"] == "margin_note"

        # Invalid becomes None
        result = _normalize_handwriting_result({"annotation_type": "invalid"})
        assert result["annotation_type"] is None

    def test_normalize_position_strips_whitespace(self) -> None:
        """Test that position is stripped of whitespace."""
        result = _normalize_handwriting_result({"position": "  right_margin  "})
        assert result["position"] == "right_margin"

    def test_normalize_position_empty_becomes_none(self) -> None:
        """Test that empty position becomes None."""
        result = _normalize_handwriting_result({"position": ""})
        assert result["position"] is None

        result = _normalize_handwriting_result({"position": "   "})
        assert result["position"] is None

    def test_normalize_is_legible_boolean(self) -> None:
        """Test is_legible normalization to boolean."""
        result = _normalize_handwriting_result({"is_legible": True})
        assert result["is_legible"] is True

        result = _normalize_handwriting_result({"is_legible": False})
        assert result["is_legible"] is False

        result = _normalize_handwriting_result({"is_legible": "yes"})
        assert result["is_legible"] is True

        result = _normalize_handwriting_result({"is_legible": ""})
        assert result["is_legible"] is False

    def test_normalize_alternative_readings_list(self) -> None:
        """Test normalization of alternative readings list."""
        result = _normalize_handwriting_result(
            {"alternative_readings": ["option1", "  option2  ", ""]}
        )
        # Empty strings should be filtered out
        assert result["alternative_readings"] == ["option1", "option2"]

    def test_normalize_alternative_readings_empty_list(self) -> None:
        """Test that empty alternative readings list becomes None."""
        result = _normalize_handwriting_result({"alternative_readings": []})
        assert result["alternative_readings"] is None

        result = _normalize_handwriting_result({"alternative_readings": ["", "  "]})
        assert result["alternative_readings"] is None

    def test_normalize_alternative_readings_non_list(self) -> None:
        """Test that non-list alternative readings becomes None."""
        result = _normalize_handwriting_result({"alternative_readings": "single"})
        assert result["alternative_readings"] is None

    def test_normalize_style_validation(self) -> None:
        """Test style validation."""
        valid_styles = ["cursive", "print", "mixed", "unknown"]
        for style in valid_styles:
            result = _normalize_handwriting_result({"style": style})
            assert result["style"] == style

        # Case insensitive
        result = _normalize_handwriting_result({"style": "CURSIVE"})
        assert result["style"] == "cursive"

        # Invalid becomes None
        result = _normalize_handwriting_result({"style": "italic"})
        assert result["style"] is None

    def test_normalize_notes_strips_whitespace(self) -> None:
        """Test that notes are stripped of whitespace."""
        result = _normalize_handwriting_result({"notes": "  Some notes here.  "})
        assert result["notes"] == "Some notes here."

    def test_normalize_notes_empty_becomes_none(self) -> None:
        """Test that empty notes become None."""
        result = _normalize_handwriting_result({"notes": ""})
        assert result["notes"] is None

        result = _normalize_handwriting_result({"notes": "   "})
        assert result["notes"] is None
