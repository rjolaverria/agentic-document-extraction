"""Tests for the AnalyzeLogo tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_logo import (
    AnalyzeLogo,
    _normalize_logo_result,
    analyze_logo_impl,
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


def create_company_logo_image() -> Image.Image:
    """Create a simple company logo image."""
    image = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Draw a simple apple-like shape
    draw.ellipse((50, 50, 150, 160), fill="gray", outline="black")

    # Add a "bite" (white rectangle)
    draw.rectangle((130, 80, 160, 120), fill="white")

    # Add stem
    draw.rectangle((95, 30, 105, 55), fill="brown")

    # Add company name text
    draw.text((60, 170), "ACME Corp", fill="black")

    return image


def create_iso_certification_image() -> Image.Image:
    """Create an ISO certification badge image."""
    image = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Draw circular badge
    draw.ellipse((20, 20, 180, 180), outline="blue", width=3)
    draw.ellipse((30, 30, 170, 170), outline="blue", width=2)

    # Add ISO text
    draw.text((70, 70), "ISO", fill="blue")
    draw.text((60, 100), "9001", fill="blue")
    draw.text((40, 130), "CERTIFIED", fill="blue")

    return image


def create_government_seal_image() -> Image.Image:
    """Create a government/official seal image."""
    image = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Draw double circle seal
    draw.ellipse((10, 10, 190, 190), outline="gold", width=4)
    draw.ellipse((25, 25, 175, 175), outline="gold", width=2)

    # Add eagle-like shape (simplified as triangle)
    draw.polygon([(100, 40), (60, 100), (140, 100)], fill="gold", outline="black")

    # Add text
    draw.text((30, 120), "DEPARTMENT", fill="black")
    draw.text((50, 140), "OF STATE", fill="black")
    draw.text((35, 160), "OFFICIAL SEAL", fill="black")

    return image


def create_brand_mark_image() -> Image.Image:
    """Create a brand mark image (symbol without text)."""
    image = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Draw Nike-like swoosh
    points = [(30, 130), (80, 100), (130, 70), (170, 90), (80, 140)]
    draw.polygon(points, fill="black")

    return image


def create_trademark_image() -> Image.Image:
    """Create a trademark symbol image."""
    image = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Draw TM symbol with brand name
    draw.text((50, 80), "BrandX", fill="black")
    draw.text((140, 75), "TM", fill="black")  # Trademark symbol position

    # Add tagline
    draw.text((40, 120), "Quality First", fill="gray")

    return image


def create_fda_approval_image() -> Image.Image:
    """Create an FDA approval badge image."""
    image = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Draw rectangular badge
    draw.rectangle((10, 10, 190, 90), outline="green", width=2)

    # Add FDA text
    draw.text((70, 25), "FDA", fill="green")
    draw.text((40, 50), "APPROVED", fill="green")

    return image


def create_usda_organic_image() -> Image.Image:
    """Create a USDA Organic certification image."""
    image = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Draw circular badge
    draw.ellipse((20, 20, 180, 180), fill="lightgreen", outline="darkgreen", width=3)

    # Add USDA text
    draw.text((65, 60), "USDA", fill="darkgreen")
    draw.text((50, 100), "ORGANIC", fill="darkgreen")

    return image


def create_ce_mark_image() -> Image.Image:
    """Create a CE mark image."""
    image = Image.new("RGB", (150, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Draw CE letters
    draw.text((40, 30), "CE", fill="black")

    return image


def create_letterhead_logo_image() -> Image.Image:
    """Create a letterhead-style image with company logo and text."""
    image = Image.new("RGB", (400, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Draw simple logo (rectangle with text)
    draw.rectangle((20, 30, 100, 120), fill="navy", outline="black")
    draw.text((35, 60), "ABC", fill="white")

    # Add company name
    draw.text((120, 50), "ABC Corporation", fill="navy")

    # Add tagline
    draw.text((120, 80), "Excellence in Innovation", fill="gray")

    # Add contact info
    draw.text((120, 110), "www.abc-corp.com", fill="gray")

    return image


def create_low_quality_logo_image() -> Image.Image:
    """Create a low quality/blurry logo image."""
    image = Image.new("RGB", (100, 100), color="lightgray")
    draw = ImageDraw.Draw(image)

    # Draw very simple, unclear shape
    draw.ellipse((20, 20, 80, 80), fill="darkgray", outline="gray")

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


class TestAnalyzeLogoTool:
    """Tests for the AnalyzeLogo tool function."""

    def test_analyze_logo_detects_company_logo(self) -> None:
        """Test detection of company logo."""
        response_payload: dict[str, Any] = {
            "logo_type": "company_logo",
            "organization_name": "ACME Corporation",
            "description": "Company logo with stylized shape and text",
            "certification_type": None,
            "associated_text": ["ACME Corp"],
            "confidence": "high",
            "notes": "Clear company logo with brand name.",
        }

        regions = create_regions(create_company_logo_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "company_logo"
        assert result["organization_name"] == "ACME Corporation"
        assert result["confidence"] == "high"
        assert result["associated_text"] == ["ACME Corp"]

    def test_analyze_logo_detects_iso_certification(self) -> None:
        """Test detection of ISO certification badge."""
        response_payload: dict[str, Any] = {
            "logo_type": "certification_badge",
            "organization_name": "ISO",
            "description": "ISO 9001 quality certification badge",
            "certification_type": "ISO 9001",
            "associated_text": ["ISO", "9001", "CERTIFIED"],
            "confidence": "high",
            "notes": "Standard ISO quality certification mark.",
        }

        regions = create_regions(create_iso_certification_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "certification_badge"
        assert result["certification_type"] == "ISO 9001"
        assert result["confidence"] == "high"

    def test_analyze_logo_detects_official_seal(self) -> None:
        """Test detection of government/official seal."""
        response_payload: dict[str, Any] = {
            "logo_type": "official_seal",
            "organization_name": "Department of State",
            "description": "Official government seal with eagle emblem",
            "certification_type": None,
            "associated_text": ["DEPARTMENT", "OF STATE", "OFFICIAL SEAL"],
            "confidence": "high",
            "notes": "Government department official seal.",
        }

        regions = create_regions(create_government_seal_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "official_seal"
        assert result["organization_name"] == "Department of State"
        assert "OFFICIAL SEAL" in result["associated_text"]

    def test_analyze_logo_detects_brand_mark(self) -> None:
        """Test detection of brand mark (symbol without text)."""
        response_payload: dict[str, Any] = {
            "logo_type": "brand_mark",
            "organization_name": None,
            "description": "Swoosh-style brand mark",
            "certification_type": None,
            "associated_text": None,
            "confidence": "medium",
            "notes": "Brand symbol without visible company name.",
        }

        regions = create_regions(create_brand_mark_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "brand_mark"
        assert result["organization_name"] is None
        assert result["associated_text"] is None

    def test_analyze_logo_detects_trademark(self) -> None:
        """Test detection of trademark symbol."""
        response_payload: dict[str, Any] = {
            "logo_type": "trade_mark",
            "organization_name": "BrandX",
            "description": "Brand name with trademark symbol",
            "certification_type": None,
            "associated_text": ["BrandX", "TM", "Quality First"],
            "confidence": "high",
            "notes": "Registered trademark with tagline.",
        }

        regions = create_regions(create_trademark_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "trade_mark"
        assert result["organization_name"] == "BrandX"
        assert "Quality First" in result["associated_text"]

    def test_analyze_logo_detects_fda_approval(self) -> None:
        """Test detection of FDA approval badge."""
        response_payload: dict[str, Any] = {
            "logo_type": "certification_badge",
            "organization_name": "FDA",
            "description": "FDA approval certification badge",
            "certification_type": "FDA Approved",
            "associated_text": ["FDA", "APPROVED"],
            "confidence": "high",
            "notes": "US Food and Drug Administration approval mark.",
        }

        regions = create_regions(create_fda_approval_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "certification_badge"
        assert result["certification_type"] == "FDA Approved"
        assert result["organization_name"] == "FDA"

    def test_analyze_logo_detects_usda_organic(self) -> None:
        """Test detection of USDA Organic certification."""
        response_payload: dict[str, Any] = {
            "logo_type": "certification_badge",
            "organization_name": "USDA",
            "description": "USDA Organic certification seal",
            "certification_type": "USDA Organic",
            "associated_text": ["USDA", "ORGANIC"],
            "confidence": "high",
            "notes": "USDA organic certification mark.",
        }

        regions = create_regions(create_usda_organic_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "certification_badge"
        assert result["certification_type"] == "USDA Organic"

    def test_analyze_logo_detects_ce_mark(self) -> None:
        """Test detection of CE mark."""
        response_payload: dict[str, Any] = {
            "logo_type": "certification_badge",
            "organization_name": "European Commission",
            "description": "CE conformity marking",
            "certification_type": "CE",
            "associated_text": ["CE"],
            "confidence": "high",
            "notes": "European conformity marking for product safety.",
        }

        regions = create_regions(create_ce_mark_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "certification_badge"
        assert result["certification_type"] == "CE"

    def test_analyze_logo_handles_letterhead(self) -> None:
        """Test handling of letterhead with logo and company info."""
        response_payload: dict[str, Any] = {
            "logo_type": "company_logo",
            "organization_name": "ABC Corporation",
            "description": "Corporate letterhead logo with company name",
            "certification_type": None,
            "associated_text": [
                "ABC",
                "ABC Corporation",
                "Excellence in Innovation",
                "www.abc-corp.com",
            ],
            "confidence": "high",
            "notes": "Complete letterhead with logo, company name, and contact info.",
        }

        regions = create_regions(create_letterhead_logo_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "company_logo"
        assert result["organization_name"] == "ABC Corporation"
        assert len(result["associated_text"]) >= 2

    def test_analyze_logo_handles_low_quality(self) -> None:
        """Test handling of low quality/unclear logo."""
        response_payload: dict[str, Any] = {
            "logo_type": "brand_mark",
            "organization_name": None,
            "description": "Unclear circular shape, possibly a logo",
            "certification_type": None,
            "associated_text": None,
            "confidence": "low",
            "notes": "Image quality too low for reliable identification.",
        }

        regions = create_regions(create_low_quality_logo_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["confidence"] == "low"
        assert result["organization_name"] is None

    def test_analyze_logo_handles_wrapped_json(self) -> None:
        """Test handling of wrapped JSON in VLM response."""
        wrapped = (
            "Here is the logo analysis:\n"
            + json.dumps(
                {
                    "logo_type": "company_logo",
                    "organization_name": "Test Corp",
                    "description": "Test logo",
                    "confidence": "high",
                }
            )
            + "\nEnd of analysis."
        )
        regions = create_regions(create_company_logo_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["logo_type"] == "company_logo"
        assert result["organization_name"] == "Test Corp"

    def test_analyze_logo_invalid_region_id(self) -> None:
        """Test error handling for invalid region ID."""
        regions = create_regions(create_company_logo_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            AnalyzeLogo.invoke({"region_id": "nonexistent", "regions": regions})

    def test_analyze_logo_vlm_failure(self) -> None:
        """Test error handling when VLM call fails."""
        regions = create_regions(create_company_logo_image())
        with (
            patch(
                "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
                side_effect=RuntimeError("VLM service unavailable"),
            ),
            pytest.raises(ToolException, match="AnalyzeLogo VLM call failed"),
        ):
            AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

    def test_analyze_logo_adds_note_for_unexpected_region_type(self) -> None:
        """Test that a note is added for unexpected region types."""
        response_payload: dict[str, Any] = {
            "logo_type": "company_logo",
            "organization_name": "Test Corp",
            "description": "Test logo",
            "confidence": "high",
        }

        regions = create_regions(
            create_company_logo_image(),
            region_type=RegionType.TABLE,  # Unexpected type for a logo
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert result["notes"] is not None
        assert "table" in result["notes"].lower()

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_logo_integration_company_logo(self) -> None:
        """Integration test with a company logo image."""
        regions = create_regions(create_company_logo_image())
        result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert "logo_type" in result
        assert result["logo_type"] in {
            "company_logo",
            "certification_badge",
            "official_seal",
            "brand_mark",
            "trade_mark",
        }

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_logo_integration_certification(self) -> None:
        """Integration test with certification badge image."""
        regions = create_regions(create_iso_certification_image())
        result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert "logo_type" in result
        assert "confidence" in result

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_logo_integration_official_seal(self) -> None:
        """Integration test with official seal image."""
        regions = create_regions(create_government_seal_image())
        result = AnalyzeLogo.invoke({"region_id": "region-1", "regions": regions})

        assert "logo_type" in result
        assert "description" in result


class TestAnalyzeLogoImpl:
    """Tests for the analyze_logo_impl helper function."""

    def test_impl_parses_response(self) -> None:
        """Test that impl function parses VLM response correctly."""
        response_payload: dict[str, Any] = {
            "logo_type": "company_logo",
            "organization_name": "Test Corp",
            "description": "A corporate logo",
            "certification_type": None,
            "associated_text": ["Test", "Corp"],
            "confidence": "high",
            "notes": "Clear company logo.",
        }
        regions = create_regions(create_company_logo_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_logo.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = analyze_logo_impl("region-1", regions)

        assert result["logo_type"] == "company_logo"
        assert result["organization_name"] == "Test Corp"
        assert result["description"] == "A corporate logo"
        assert result["confidence"] == "high"
        assert result["associated_text"] == ["Test", "Corp"]

    def test_impl_raises_on_missing_region(self) -> None:
        """Test error when region ID not found."""
        regions = create_regions(create_company_logo_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            analyze_logo_impl("nonexistent", regions)

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
            analyze_logo_impl("no-img", regions)

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
            analyze_logo_impl("empty-img", regions)


class TestNormalizeLogoResult:
    """Tests for the _normalize_logo_result helper function."""

    def test_normalize_empty_dict(self) -> None:
        """Test normalization of empty result dict."""
        result = _normalize_logo_result({})
        assert result["logo_type"] == "brand_mark"
        assert result["organization_name"] is None
        assert result["description"] == ""
        assert result["certification_type"] is None
        assert result["associated_text"] is None
        assert result["confidence"] == "low"
        assert result["notes"] is None

    def test_normalize_validates_logo_type(self) -> None:
        """Test logo type validation."""
        # Valid types
        for logo_type in [
            "company_logo",
            "certification_badge",
            "official_seal",
            "brand_mark",
            "trade_mark",
        ]:
            result = _normalize_logo_result({"logo_type": logo_type})
            assert result["logo_type"] == logo_type

        # Invalid type defaults to brand_mark
        result = _normalize_logo_result({"logo_type": "invalid_type"})
        assert result["logo_type"] == "brand_mark"

        # Case insensitive
        result = _normalize_logo_result({"logo_type": "COMPANY_LOGO"})
        assert result["logo_type"] == "company_logo"

    def test_normalize_validates_confidence(self) -> None:
        """Test confidence level validation."""
        # Valid levels
        for confidence in ["high", "medium", "low"]:
            result = _normalize_logo_result({"confidence": confidence})
            assert result["confidence"] == confidence

        # Invalid level defaults to low
        result = _normalize_logo_result({"confidence": "invalid"})
        assert result["confidence"] == "low"

        # Case insensitive
        result = _normalize_logo_result({"confidence": "HIGH"})
        assert result["confidence"] == "high"

    def test_normalize_organization_name_strips_whitespace(self) -> None:
        """Test that organization name is stripped of whitespace."""
        result = _normalize_logo_result({"organization_name": "  Test Corp  "})
        assert result["organization_name"] == "Test Corp"

    def test_normalize_organization_name_empty_becomes_none(self) -> None:
        """Test that empty organization name becomes None."""
        result = _normalize_logo_result({"organization_name": ""})
        assert result["organization_name"] is None

        result = _normalize_logo_result({"organization_name": "   "})
        assert result["organization_name"] is None

    def test_normalize_description_strips_whitespace(self) -> None:
        """Test that description is stripped of whitespace."""
        result = _normalize_logo_result({"description": "  A logo description  "})
        assert result["description"] == "A logo description"

    def test_normalize_description_empty_returns_empty_string(self) -> None:
        """Test that empty description returns empty string (not None)."""
        result = _normalize_logo_result({"description": ""})
        assert result["description"] == ""

        result = _normalize_logo_result({"description": None})
        assert result["description"] == ""

    def test_normalize_certification_type_strips_whitespace(self) -> None:
        """Test that certification type is stripped of whitespace."""
        result = _normalize_logo_result({"certification_type": "  ISO 9001  "})
        assert result["certification_type"] == "ISO 9001"

    def test_normalize_certification_type_empty_becomes_none(self) -> None:
        """Test that empty certification type becomes None."""
        result = _normalize_logo_result({"certification_type": ""})
        assert result["certification_type"] is None

    def test_normalize_associated_text_list(self) -> None:
        """Test normalization of associated text list."""
        result = _normalize_logo_result(
            {"associated_text": ["Apple", "  iPhone  ", ""]}
        )
        # Empty strings should be filtered out
        assert result["associated_text"] == ["Apple", "iPhone"]

    def test_normalize_associated_text_empty_list(self) -> None:
        """Test that empty associated text list becomes None."""
        result = _normalize_logo_result({"associated_text": []})
        assert result["associated_text"] is None

        result = _normalize_logo_result({"associated_text": ["", "  "]})
        assert result["associated_text"] is None

    def test_normalize_associated_text_non_list(self) -> None:
        """Test that non-list associated text becomes None."""
        result = _normalize_logo_result({"associated_text": "Apple"})
        assert result["associated_text"] is None

    def test_normalize_notes_strips_whitespace(self) -> None:
        """Test that notes are stripped of whitespace."""
        result = _normalize_logo_result({"notes": "  Some notes here.  "})
        assert result["notes"] == "Some notes here."

    def test_normalize_notes_empty_becomes_none(self) -> None:
        """Test that empty notes become None."""
        result = _normalize_logo_result({"notes": ""})
        assert result["notes"] is None

        result = _normalize_logo_result({"notes": "   "})
        assert result["notes"] is None
