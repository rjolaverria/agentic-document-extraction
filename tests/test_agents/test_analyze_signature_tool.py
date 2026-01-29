"""Tests for the AnalyzeSignature tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_signature import (
    AnalyzeSignature,
    _normalize_signature_result,
    analyze_signature_impl,
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


def create_simple_signature_image() -> Image.Image:
    """Create a simple signature block image with signature and printed name."""
    image = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Signature line
    draw.line((50, 100, 350, 100), fill="black", width=1)

    # Simulated signature (wavy line)
    points = [(80, 85), (100, 75), (130, 90), (160, 70), (200, 85), (230, 75)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="blue", width=2)

    # Printed name
    draw.text((130, 110), "John Smith", fill="black")

    # Title
    draw.text((150, 130), "CEO", fill="black")

    # Date
    draw.text((50, 160), "Date: January 15, 2024", fill="black")

    return image


def create_notary_stamp_image() -> Image.Image:
    """Create an image with a notary stamp and signature."""
    image = Image.new("RGB", (500, 300), color="white")
    draw = ImageDraw.Draw(image)

    # Signature area
    draw.line((50, 100, 250, 100), fill="black", width=1)
    points = [(70, 85), (100, 75), (130, 90), (160, 70), (180, 85)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="blue", width=2)
    draw.text((100, 110), "Jane Doe", fill="black")

    # Notary stamp (circular)
    draw.ellipse((300, 50, 450, 200), outline="blue", width=2)
    draw.ellipse((310, 60, 440, 190), outline="blue", width=1)
    draw.text((330, 100), "NOTARY", fill="blue")
    draw.text((325, 120), "PUBLIC", fill="blue")
    draw.text((310, 145), "State of NY", fill="blue")

    # Notary date and commission
    draw.text((300, 220), "My Commission Expires: 12/31/2025", fill="black")

    return image


def create_company_seal_image() -> Image.Image:
    """Create an image with a company seal."""
    image = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(image)

    # Company seal (circular with star pattern)
    cx, cy = 200, 150
    r1, r2 = 80, 60
    draw.ellipse((cx - r1, cy - r1, cx + r1, cy + r1), outline="red", width=3)
    draw.ellipse((cx - r2, cy - r2, cx + r2, cy + r2), outline="red", width=2)

    # Seal text
    draw.text((155, 125), "ACME", fill="red")
    draw.text((120, 145), "CORPORATION", fill="red")
    draw.text((140, 175), "Official Seal", fill="red")

    return image


def create_multi_signature_image() -> Image.Image:
    """Create an image with multiple signatures (e.g., dual signers)."""
    image = Image.new("RGB", (600, 250), color="white")
    draw = ImageDraw.Draw(image)

    # First signer
    draw.line((50, 100, 250, 100), fill="black", width=1)
    points1 = [(70, 85), (100, 75), (130, 90), (160, 70)]
    for i in range(len(points1) - 1):
        draw.line([points1[i], points1[i + 1]], fill="blue", width=2)
    draw.text((100, 110), "John Smith", fill="black")
    draw.text((120, 130), "Seller", fill="black")

    # Second signer
    draw.line((350, 100, 550, 100), fill="black", width=1)
    points2 = [(370, 85), (400, 75), (430, 90), (460, 70)]
    for i in range(len(points2) - 1):
        draw.line([points2[i], points2[i + 1]], fill="blue", width=2)
    draw.text((400, 110), "Jane Doe", fill="black")
    draw.text((420, 130), "Buyer", fill="black")

    # Date
    draw.text((250, 180), "Date: March 1, 2024", fill="black")

    return image


def create_incomplete_signature_image() -> Image.Image:
    """Create an image with incomplete signature block (missing date)."""
    image = Image.new("RGB", (400, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Signature line only
    draw.line((50, 80, 350, 80), fill="black", width=1)

    # Wavy signature
    points = [(80, 65), (100, 55), (130, 70), (160, 50), (200, 65)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill="blue", width=2)

    # Label only (no name filled in)
    draw.text((50, 100), "Signature: _______________", fill="black")

    return image


def create_certification_marks_image() -> Image.Image:
    """Create an image with certification marks."""
    image = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(image)

    # ISO certification box
    draw.rectangle((30, 30, 130, 80), outline="black", width=2)
    draw.text((50, 45), "ISO", fill="black")
    draw.text((45, 60), "9001", fill="black")

    # Certified copy stamp
    draw.rectangle((150, 30, 300, 80), outline="green", width=2)
    draw.text((165, 45), "CERTIFIED", fill="green")
    draw.text((185, 60), "COPY", fill="green")

    # Watermark simulation
    draw.text((100, 120), "ORIGINAL", fill="lightgray")

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


class TestAnalyzeSignatureTool:
    """Tests for the AnalyzeSignature tool function."""

    def test_analyze_signature_detects_signature_presence(self) -> None:
        """Test detection of signature presence with signer details."""
        response_payload: dict[str, Any] = {
            "signature_present": True,
            "signer_name": "John Smith",
            "signer_title": "CEO",
            "date_signed": "January 15, 2024",
            "location": None,
            "stamp_present": False,
            "stamp_type": None,
            "stamp_text": None,
            "certification_marks": None,
            "is_complete": True,
            "missing_elements": None,
            "notes": "Standard signature block with printed name and title.",
        }

        regions = create_regions(create_simple_signature_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeSignature.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["signature_present"] is True
        assert result["signer_name"] == "John Smith"
        assert result["signer_title"] == "CEO"
        assert result["date_signed"] == "January 15, 2024"
        assert result["is_complete"] is True

    def test_analyze_signature_detects_notary_stamp(self) -> None:
        """Test detection of notary stamp and signature."""
        response_payload: dict[str, Any] = {
            "signature_present": True,
            "signer_name": "Jane Doe",
            "signer_title": "Notary Public",
            "date_signed": None,
            "location": "State of NY",
            "stamp_present": True,
            "stamp_type": "notary",
            "stamp_text": "NOTARY PUBLIC State of NY Commission Expires 12/31/2025",
            "certification_marks": None,
            "is_complete": True,
            "missing_elements": None,
            "notes": "Notarized signature with valid notary stamp.",
        }

        regions = create_regions(create_notary_stamp_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeSignature.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["signature_present"] is True
        assert result["stamp_present"] is True
        assert result["stamp_type"] == "notary"
        assert result["location"] == "State of NY"

    def test_analyze_signature_detects_company_seal(self) -> None:
        """Test detection of company seal."""
        response_payload: dict[str, Any] = {
            "signature_present": False,
            "signer_name": None,
            "signer_title": None,
            "date_signed": None,
            "location": None,
            "stamp_present": True,
            "stamp_type": "company_seal",
            "stamp_text": "ACME CORPORATION Official Seal",
            "certification_marks": None,
            "is_complete": False,
            "missing_elements": ["signature", "date"],
            "notes": "Company seal present but no signature.",
        }

        regions = create_regions(create_company_seal_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeSignature.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["stamp_present"] is True
        assert result["stamp_type"] == "company_seal"
        assert result["stamp_text"] == "ACME CORPORATION Official Seal"
        assert result["signature_present"] is False

    def test_analyze_signature_detects_certification_marks(self) -> None:
        """Test detection of certification marks."""
        response_payload: dict[str, Any] = {
            "signature_present": False,
            "signer_name": None,
            "signer_title": None,
            "date_signed": None,
            "location": None,
            "stamp_present": False,
            "stamp_type": None,
            "stamp_text": None,
            "certification_marks": ["ISO 9001", "Certified Copy"],
            "is_complete": False,
            "missing_elements": ["signature"],
            "notes": "Certification marks present, no signature.",
        }

        regions = create_regions(create_certification_marks_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeSignature.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["certification_marks"] == ["ISO 9001", "Certified Copy"]
        assert result["signature_present"] is False

    def test_analyze_signature_detects_incomplete_block(self) -> None:
        """Test detection of incomplete signature block."""
        response_payload: dict[str, Any] = {
            "signature_present": True,
            "signer_name": None,
            "signer_title": None,
            "date_signed": None,
            "location": None,
            "stamp_present": False,
            "stamp_type": None,
            "stamp_text": None,
            "certification_marks": None,
            "is_complete": False,
            "missing_elements": ["printed name", "date", "title"],
            "notes": "Signature present but block is incomplete.",
        }

        regions = create_regions(create_incomplete_signature_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeSignature.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["signature_present"] is True
        assert result["is_complete"] is False
        assert result["missing_elements"] == ["printed name", "date", "title"]

    def test_analyze_signature_handles_wrapped_json(self) -> None:
        """Test handling of wrapped JSON in VLM response."""
        wrapped = (
            "Here is the signature analysis:\n"
            + json.dumps(
                {
                    "signature_present": True,
                    "signer_name": "Test User",
                    "stamp_present": False,
                    "is_complete": True,
                }
            )
            + "\nEnd of analysis."
        )
        regions = create_regions(create_simple_signature_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeSignature.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["signature_present"] is True
        assert result["signer_name"] == "Test User"

    def test_analyze_signature_invalid_region_id(self) -> None:
        """Test error handling for invalid region ID."""
        regions = create_regions(create_simple_signature_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            AnalyzeSignature.invoke({"region_id": "nonexistent", "regions": regions})

    def test_analyze_signature_vlm_failure(self) -> None:
        """Test error handling when VLM call fails."""
        regions = create_regions(create_simple_signature_image())
        with (
            patch(
                "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
                side_effect=RuntimeError("VLM service unavailable"),
            ),
            pytest.raises(ToolException, match="AnalyzeSignature VLM call failed"),
        ):
            AnalyzeSignature.invoke({"region_id": "region-1", "regions": regions})

    def test_analyze_signature_adds_note_for_unexpected_region_type(self) -> None:
        """Test that a note is added for unexpected region types."""
        response_payload: dict[str, Any] = {
            "signature_present": True,
            "signer_name": "Test User",
            "stamp_present": False,
            "is_complete": True,
        }

        regions = create_regions(
            create_simple_signature_image(),
            region_type=RegionType.TABLE,  # Unexpected type for a signature
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeSignature.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["notes"] is not None
        assert "table" in result["notes"].lower()

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_signature_integration_simple(self) -> None:
        """Integration test with a simple signature image."""
        regions = create_regions(create_simple_signature_image())
        result = AnalyzeSignature.invoke({"region_id": "region-1", "regions": regions})

        assert "signature_present" in result
        assert isinstance(result["signature_present"], bool)

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_signature_integration_notary(self) -> None:
        """Integration test with notary stamp image."""
        regions = create_regions(create_notary_stamp_image())
        result = AnalyzeSignature.invoke({"region_id": "region-1", "regions": regions})

        assert "stamp_present" in result
        assert isinstance(result["stamp_present"], bool)

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_signature_integration_company_seal(self) -> None:
        """Integration test with company seal image."""
        regions = create_regions(create_company_seal_image())
        result = AnalyzeSignature.invoke({"region_id": "region-1", "regions": regions})

        assert "stamp_present" in result
        assert "stamp_type" in result


class TestAnalyzeSignatureImpl:
    """Tests for the analyze_signature_impl helper function."""

    def test_impl_parses_response(self) -> None:
        """Test that impl function parses VLM response correctly."""
        response_payload: dict[str, Any] = {
            "signature_present": True,
            "signer_name": "Test User",
            "signer_title": "Manager",
            "date_signed": "2024-01-15",
            "location": "Boston, MA",
            "stamp_present": True,
            "stamp_type": "certification",
            "stamp_text": "Approved",
            "certification_marks": ["ISO 14001"],
            "is_complete": True,
            "missing_elements": None,
            "notes": "Complete signature block.",
        }
        regions = create_regions(create_simple_signature_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_signature.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = analyze_signature_impl("region-1", regions)

        assert result["signature_present"] is True
        assert result["signer_name"] == "Test User"
        assert result["signer_title"] == "Manager"
        assert result["stamp_type"] == "certification"
        assert result["certification_marks"] == ["ISO 14001"]
        assert result["is_complete"] is True

    def test_impl_raises_on_missing_region(self) -> None:
        """Test error when region ID not found."""
        regions = create_regions(create_simple_signature_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            analyze_signature_impl("nonexistent", regions)

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
            analyze_signature_impl("no-img", regions)

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
            analyze_signature_impl("empty-img", regions)


class TestNormalizeSignatureResult:
    """Tests for the _normalize_signature_result helper function."""

    def test_normalize_empty_dict(self) -> None:
        """Test normalization of empty result dict."""
        result = _normalize_signature_result({})
        assert result["signature_present"] is False
        assert result["stamp_present"] is False
        assert result["is_complete"] is False
        assert result["signer_name"] is None
        assert result["stamp_type"] is None

    def test_normalize_boolean_fields(self) -> None:
        """Test normalization of boolean fields."""
        result = _normalize_signature_result(
            {
                "signature_present": True,
                "stamp_present": "yes",  # String that should become True
                "is_complete": 1,  # Int that should become True
            }
        )
        assert result["signature_present"] is True
        assert result["stamp_present"] is True
        assert result["is_complete"] is True

    def test_normalize_string_fields_strips_whitespace(self) -> None:
        """Test that string fields are stripped of whitespace."""
        result = _normalize_signature_result(
            {
                "signer_name": "  John Smith  ",
                "signer_title": "\tCEO\n",
                "date_signed": " January 15 ",
                "location": "  New York  ",
            }
        )
        assert result["signer_name"] == "John Smith"
        assert result["signer_title"] == "CEO"
        assert result["date_signed"] == "January 15"
        assert result["location"] == "New York"

    def test_normalize_empty_strings_become_none(self) -> None:
        """Test that empty strings become None."""
        result = _normalize_signature_result(
            {
                "signer_name": "",
                "signer_title": "   ",
                "date_signed": None,
            }
        )
        assert result["signer_name"] is None
        assert result["signer_title"] is None
        assert result["date_signed"] is None

    def test_normalize_validates_stamp_type(self) -> None:
        """Test stamp type validation."""
        # Valid types
        for stamp_type in ["company_seal", "notary", "certification", "watermark"]:
            result = _normalize_signature_result({"stamp_type": stamp_type})
            assert result["stamp_type"] == stamp_type

        # Invalid type becomes None
        result = _normalize_signature_result({"stamp_type": "invalid_type"})
        assert result["stamp_type"] is None

        # Case insensitive
        result = _normalize_signature_result({"stamp_type": "NOTARY"})
        assert result["stamp_type"] == "notary"

    def test_normalize_certification_marks_list(self) -> None:
        """Test normalization of certification marks list."""
        result = _normalize_signature_result(
            {"certification_marks": ["ISO 9001", "  ISO 14001  ", ""]}
        )
        # Empty strings should be filtered out
        assert result["certification_marks"] == ["ISO 9001", "ISO 14001"]

    def test_normalize_certification_marks_empty_list(self) -> None:
        """Test that empty certification marks list becomes None."""
        result = _normalize_signature_result({"certification_marks": []})
        assert result["certification_marks"] is None

        result = _normalize_signature_result({"certification_marks": ["", "  "]})
        assert result["certification_marks"] is None

    def test_normalize_certification_marks_non_list(self) -> None:
        """Test that non-list certification marks becomes None."""
        result = _normalize_signature_result({"certification_marks": "ISO 9001"})
        assert result["certification_marks"] is None

    def test_normalize_missing_elements_list(self) -> None:
        """Test normalization of missing elements list."""
        result = _normalize_signature_result(
            {"missing_elements": ["date", "  title  ", ""]}
        )
        assert result["missing_elements"] == ["date", "title"]

    def test_normalize_missing_elements_empty_list(self) -> None:
        """Test that empty missing elements list becomes None."""
        result = _normalize_signature_result({"missing_elements": []})
        assert result["missing_elements"] is None

    def test_normalize_notes_strips_whitespace(self) -> None:
        """Test that notes are stripped of whitespace."""
        result = _normalize_signature_result({"notes": "  Some notes here.  "})
        assert result["notes"] == "Some notes here."

    def test_normalize_notes_empty_becomes_none(self) -> None:
        """Test that empty notes become None."""
        result = _normalize_signature_result({"notes": ""})
        assert result["notes"] is None

        result = _normalize_signature_result({"notes": "   "})
        assert result["notes"] is None
