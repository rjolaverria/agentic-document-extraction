"""Tests for the AnalyzeMath tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_math import (
    AnalyzeMath,
    _normalize_math_result,
    analyze_math_impl,
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


def create_quadratic_formula_image() -> Image.Image:
    """Create an image with quadratic formula representation."""
    image = Image.new("RGB", (400, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Draw quadratic formula representation
    # x = (-b ± √(b²-4ac)) / 2a
    draw.text((50, 30), "x = ", fill="black")
    draw.text((90, 20), "-b ± sqrt(b² - 4ac)", fill="black")
    draw.line((90, 55, 260, 55), fill="black", width=2)  # fraction line
    draw.text((150, 65), "2a", fill="black")

    return image


def create_chemical_formula_image() -> Image.Image:
    """Create an image with a chemical formula."""
    image = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Draw water formation reaction
    draw.text((50, 40), "2H2 + O2 -> 2H2O", fill="black")

    return image


def create_matrix_image() -> Image.Image:
    """Create an image with a matrix."""
    image = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Draw matrix brackets
    draw.line((40, 40, 40, 160), fill="black", width=2)  # left bracket top
    draw.line((40, 40, 50, 40), fill="black", width=2)
    draw.line((40, 160, 50, 160), fill="black", width=2)

    draw.line((160, 40, 160, 160), fill="black", width=2)  # right bracket
    draw.line((150, 40, 160, 40), fill="black", width=2)
    draw.line((150, 160, 160, 160), fill="black", width=2)

    # Draw matrix elements
    draw.text((70, 60), "1  2", fill="black")
    draw.text((70, 100), "3  4", fill="black")

    return image


def create_einstein_equation_image() -> Image.Image:
    """Create an image with E=mc²."""
    image = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(image)

    draw.text((50, 35), "E = mc²", fill="black")

    return image


def create_integral_image() -> Image.Image:
    """Create an image with an integral."""
    image = Image.new("RGB", (300, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Draw integral symbol (simplified)
    draw.text((30, 20), "a", fill="black")  # upper limit
    draw.text((40, 40), "∫", fill="black")  # integral symbol
    draw.text((30, 80), "b", fill="black")  # lower limit
    draw.text((60, 50), "f(x) dx", fill="black")

    return image


def create_summation_image() -> Image.Image:
    """Create an image with a summation."""
    image = Image.new("RGB", (300, 150), color="white")
    draw = ImageDraw.Draw(image)

    # Draw summation
    draw.text((60, 20), "n", fill="black")  # upper limit
    draw.text((50, 50), "Σ", fill="black")  # sigma
    draw.text((45, 90), "i=1", fill="black")  # lower limit
    draw.text((80, 55), "i²", fill="black")

    return image


def create_derivative_image() -> Image.Image:
    """Create an image with a derivative."""
    image = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(image)

    draw.text((30, 30), "dy", fill="black")
    draw.line((30, 55, 60, 55), fill="black", width=2)
    draw.text((30, 60), "dx", fill="black")
    draw.text((70, 40), "= 2x", fill="black")

    return image


def create_complex_chemical_image() -> Image.Image:
    """Create an image with a complex chemical reaction."""
    image = Image.new("RGB", (500, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Combustion of methane
    draw.text((30, 40), "CH4 + 2O2 -> CO2 + 2H2O", fill="black")

    return image


def create_greek_letters_image() -> Image.Image:
    """Create an image with Greek letters in an equation."""
    image = Image.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(image)

    # α + β = γ
    draw.text((50, 40), "α + β = γ", fill="black")

    return image


def create_physics_formula_image() -> Image.Image:
    """Create an image with a physics formula with units."""
    image = Image.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(image)

    # F = ma
    draw.text((50, 40), "F = ma  [N = kg·m/s²]", fill="black")

    return image


def create_equation_array_image() -> Image.Image:
    """Create an image with multiple equations."""
    image = Image.new("RGB", (200, 150), color="white")
    draw = ImageDraw.Draw(image)

    draw.text((30, 30), "x + y = 5", fill="black")
    draw.text((30, 60), "x - y = 1", fill="black")
    draw.text((30, 90), "∴ x = 3", fill="black")

    return image


def create_mixed_content_image() -> Image.Image:
    """Create an image with mixed mathematical content."""
    image = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(image)

    # Equation
    draw.text((30, 30), "v = d/t", fill="black")
    # Chemical
    draw.text((30, 70), "NaCl", fill="black")
    # Matrix reference
    draw.text((30, 110), "det(A) = ad - bc", fill="black")

    return image


def create_low_quality_math_image() -> Image.Image:
    """Create a low quality mathematical content image."""
    image = Image.new("RGB", (100, 50), color="lightgray")
    draw = ImageDraw.Draw(image)

    # Very small, unclear text
    draw.text((20, 20), "x=?", fill="darkgray")

    return image


def create_regions(
    image: Image.Image,
    *,
    region_id: str = "region-1",
    region_type: RegionType = RegionType.FORMULA,
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


class TestAnalyzeMathTool:
    """Tests for the AnalyzeMath tool function."""

    def test_analyze_math_detects_equation(self) -> None:
        """Test detection of mathematical equation."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
            "plain_text": "The quadratic formula for finding roots of a quadratic equation",
            "variables": {
                "a": "coefficient of x squared",
                "b": "coefficient of x",
                "c": "constant term",
                "x": "unknown variable",
            },
            "notes": "Standard quadratic formula.",
        }

        regions = create_regions(create_quadratic_formula_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "equation"
        assert "frac" in result["latex"]
        assert result["variables"] is not None
        assert "a" in result["variables"]

    def test_analyze_math_detects_chemical_formula(self) -> None:
        """Test detection of chemical formula."""
        response_payload: dict[str, Any] = {
            "content_type": "chemical_formula",
            "latex": "2H_2 + O_2 \\rightarrow 2H_2O",
            "plain_text": "Two molecules of hydrogen react with oxygen to form water",
            "variables": {
                "H_2": "hydrogen gas",
                "O_2": "oxygen gas",
                "H_2O": "water",
            },
            "notes": "Water formation reaction.",
        }

        regions = create_regions(create_chemical_formula_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "chemical_formula"
        assert "H_2" in result["latex"]
        assert result["variables"] is not None

    def test_analyze_math_detects_matrix(self) -> None:
        """Test detection of matrix."""
        response_payload: dict[str, Any] = {
            "content_type": "matrix",
            "latex": "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}",
            "plain_text": "A 2x2 matrix with elements 1, 2, 3, 4",
            "variables": None,
            "notes": None,
        }

        regions = create_regions(create_matrix_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "matrix"
        assert "bmatrix" in result["latex"]

    def test_analyze_math_detects_notation(self) -> None:
        """Test detection of scientific notation."""
        response_payload: dict[str, Any] = {
            "content_type": "notation",
            "latex": "E = mc^2",
            "plain_text": "Einstein's mass-energy equivalence formula",
            "variables": {
                "E": "energy",
                "m": "mass",
                "c": "speed of light",
            },
            "notes": "Famous equation from special relativity.",
        }

        regions = create_regions(create_einstein_equation_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "notation"
        assert "mc^2" in result["latex"]
        assert result["variables"]["E"] == "energy"

    def test_analyze_math_detects_integral(self) -> None:
        """Test detection of integral."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "\\int_a^b f(x) \\, dx",
            "plain_text": "Definite integral of f(x) from a to b",
            "variables": {
                "a": "lower limit",
                "b": "upper limit",
                "f(x)": "integrand function",
            },
            "notes": "Standard definite integral notation.",
        }

        regions = create_regions(create_integral_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "equation"
        assert "int" in result["latex"]

    def test_analyze_math_detects_summation(self) -> None:
        """Test detection of summation."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "\\sum_{i=1}^{n} i^2",
            "plain_text": "Sum of squares from 1 to n",
            "variables": {
                "i": "index variable",
                "n": "upper limit",
            },
            "notes": "Common summation formula.",
        }

        regions = create_regions(create_summation_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "equation"
        assert "sum" in result["latex"]

    def test_analyze_math_detects_derivative(self) -> None:
        """Test detection of derivative."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "\\frac{dy}{dx} = 2x",
            "plain_text": "Derivative of y with respect to x equals 2x",
            "variables": {
                "y": "dependent variable",
                "x": "independent variable",
            },
            "notes": "First derivative notation.",
        }

        regions = create_regions(create_derivative_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "equation"
        assert "frac" in result["latex"]

    def test_analyze_math_detects_complex_chemical(self) -> None:
        """Test detection of complex chemical reaction."""
        response_payload: dict[str, Any] = {
            "content_type": "chemical_formula",
            "latex": "CH_4 + 2O_2 \\rightarrow CO_2 + 2H_2O",
            "plain_text": "Combustion of methane producing carbon dioxide and water",
            "variables": {
                "CH_4": "methane",
                "O_2": "oxygen",
                "CO_2": "carbon dioxide",
                "H_2O": "water",
            },
            "notes": "Complete combustion reaction of methane.",
        }

        regions = create_regions(create_complex_chemical_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "chemical_formula"
        assert "CH_4" in result["latex"]
        assert len(result["variables"]) >= 3

    def test_analyze_math_detects_greek_letters(self) -> None:
        """Test detection of equations with Greek letters."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "\\alpha + \\beta = \\gamma",
            "plain_text": "Alpha plus beta equals gamma",
            "variables": {
                "α": "first angle",
                "β": "second angle",
                "γ": "third angle",
            },
            "notes": "Equation with Greek letters.",
        }

        regions = create_regions(create_greek_letters_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "equation"
        assert "alpha" in result["latex"] or "β" in result["plain_text"]

    def test_analyze_math_detects_physics_formula(self) -> None:
        """Test detection of physics formula with units."""
        response_payload: dict[str, Any] = {
            "content_type": "notation",
            "latex": "F = ma",
            "plain_text": "Force equals mass times acceleration",
            "variables": {
                "F": "force (Newtons)",
                "m": "mass (kilograms)",
                "a": "acceleration (m/s²)",
            },
            "notes": "Newton's second law of motion.",
        }

        regions = create_regions(create_physics_formula_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "notation"
        assert result["latex"] == "F = ma"

    def test_analyze_math_detects_mixed_content(self) -> None:
        """Test detection of mixed mathematical content."""
        response_payload: dict[str, Any] = {
            "content_type": "mixed",
            "latex": "v = \\frac{d}{t}, \\text{NaCl}, \\det(A) = ad - bc",
            "plain_text": "Multiple mathematical expressions including velocity, sodium chloride, and matrix determinant",
            "variables": {
                "v": "velocity",
                "d": "distance",
                "t": "time",
                "A": "matrix",
            },
            "notes": "Contains equation, chemical formula, and matrix formula.",
        }

        regions = create_regions(create_mixed_content_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "mixed"

    def test_analyze_math_handles_equation_array(self) -> None:
        """Test handling of equation arrays/systems."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "\\begin{aligned} x + y &= 5 \\\\ x - y &= 1 \\\\ \\therefore x &= 3 \\end{aligned}",
            "plain_text": "System of linear equations with solution x=3",
            "variables": {
                "x": "first unknown",
                "y": "second unknown",
            },
            "notes": "Linear equation system.",
        }

        regions = create_regions(create_equation_array_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "equation"
        assert "aligned" in result["latex"] or "x + y" in result["latex"]

    def test_analyze_math_handles_low_quality(self) -> None:
        """Test handling of low quality mathematical content."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "x = ?",
            "plain_text": "Unclear mathematical expression",
            "variables": None,
            "notes": "Image quality too low for accurate transcription.",
        }

        regions = create_regions(create_low_quality_math_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["notes"] is not None
        assert "low" in result["notes"].lower() or "unclear" in result["notes"].lower()

    def test_analyze_math_handles_wrapped_json(self) -> None:
        """Test handling of wrapped JSON in VLM response."""
        wrapped = (
            "Here is the math analysis:\n"
            + json.dumps(
                {
                    "content_type": "equation",
                    "latex": "x^2 + y^2 = r^2",
                    "plain_text": "Circle equation",
                    "variables": {"r": "radius"},
                }
            )
            + "\nEnd of analysis."
        )
        regions = create_regions(create_einstein_equation_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["content_type"] == "equation"
        assert result["latex"] == "x^2 + y^2 = r^2"

    def test_analyze_math_invalid_region_id(self) -> None:
        """Test error handling for invalid region ID."""
        regions = create_regions(create_quadratic_formula_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            AnalyzeMath.invoke({"region_id": "nonexistent", "regions": regions})

    def test_analyze_math_vlm_failure(self) -> None:
        """Test error handling when VLM call fails."""
        regions = create_regions(create_quadratic_formula_image())
        with (
            patch(
                "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
                side_effect=RuntimeError("VLM service unavailable"),
            ),
            pytest.raises(ToolException, match="AnalyzeMath VLM call failed"),
        ):
            AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

    def test_analyze_math_adds_note_for_unexpected_region_type(self) -> None:
        """Test that a note is added for unexpected region types."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "x = 1",
            "plain_text": "Simple equation",
            "variables": None,
        }

        regions = create_regions(
            create_quadratic_formula_image(),
            region_type=RegionType.TABLE,  # Unexpected type for math
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert result["notes"] is not None
        assert "table" in result["notes"].lower()

    def test_analyze_math_accepts_text_region_type(self) -> None:
        """Test that TEXT region type doesn't trigger warning."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "x = 1",
            "plain_text": "Simple equation",
            "variables": None,
            "notes": None,
        }

        regions = create_regions(
            create_quadratic_formula_image(),
            region_type=RegionType.TEXT,
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        # Notes should be None since TEXT is an expected region type
        assert result["notes"] is None

    def test_analyze_math_accepts_picture_region_type(self) -> None:
        """Test that PICTURE region type doesn't trigger warning."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "x = 1",
            "plain_text": "Simple equation",
            "variables": None,
            "notes": None,
        }

        regions = create_regions(
            create_quadratic_formula_image(),
            region_type=RegionType.PICTURE,
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        # Notes should be None since PICTURE is an expected region type
        assert result["notes"] is None

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_math_integration_equation(self) -> None:
        """Integration test with an equation image."""
        regions = create_regions(create_quadratic_formula_image())
        result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert "content_type" in result
        assert result["content_type"] in {
            "equation",
            "chemical_formula",
            "matrix",
            "notation",
            "mixed",
        }
        assert "latex" in result
        assert "plain_text" in result

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_math_integration_chemical(self) -> None:
        """Integration test with a chemical formula image."""
        regions = create_regions(create_chemical_formula_image())
        result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert "content_type" in result
        assert "latex" in result

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_math_integration_matrix(self) -> None:
        """Integration test with a matrix image."""
        regions = create_regions(create_matrix_image())
        result = AnalyzeMath.invoke({"region_id": "region-1", "regions": regions})

        assert "content_type" in result
        assert "latex" in result


class TestAnalyzeMathImpl:
    """Tests for the analyze_math_impl helper function."""

    def test_impl_parses_response(self) -> None:
        """Test that impl function parses VLM response correctly."""
        response_payload: dict[str, Any] = {
            "content_type": "equation",
            "latex": "a^2 + b^2 = c^2",
            "plain_text": "Pythagorean theorem",
            "variables": {
                "a": "first leg",
                "b": "second leg",
                "c": "hypotenuse",
            },
            "notes": "Right triangle relationship.",
        }
        regions = create_regions(create_quadratic_formula_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_math.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = analyze_math_impl("region-1", regions)

        assert result["content_type"] == "equation"
        assert result["latex"] == "a^2 + b^2 = c^2"
        assert result["plain_text"] == "Pythagorean theorem"
        assert result["variables"]["a"] == "first leg"
        assert result["notes"] == "Right triangle relationship."

    def test_impl_raises_on_missing_region(self) -> None:
        """Test error when region ID not found."""
        regions = create_regions(create_quadratic_formula_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            analyze_math_impl("nonexistent", regions)

    def test_impl_raises_on_missing_image(self) -> None:
        """Test error when region has no image."""
        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.FORMULA,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="no-img",
                region_image=None,
            )
        ]
        with pytest.raises(ToolException, match="Region image not provided"):
            analyze_math_impl("no-img", regions)

    def test_impl_raises_on_empty_image_data(self) -> None:
        """Test error when region image has no data."""
        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.FORMULA,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="empty-img",
                region_image=RegionImage(image=None, base64=None),
            )
        ]
        with pytest.raises(ToolException, match="Region image missing"):
            analyze_math_impl("empty-img", regions)


class TestNormalizeMathResult:
    """Tests for the _normalize_math_result helper function."""

    def test_normalize_empty_dict(self) -> None:
        """Test normalization of empty result dict."""
        result = _normalize_math_result({})
        assert result["content_type"] == "equation"
        assert result["latex"] == ""
        assert result["plain_text"] == ""
        assert result["variables"] is None
        assert result["notes"] is None

    def test_normalize_validates_content_type(self) -> None:
        """Test content type validation."""
        # Valid types
        for content_type in [
            "equation",
            "chemical_formula",
            "matrix",
            "notation",
            "mixed",
        ]:
            result = _normalize_math_result({"content_type": content_type})
            assert result["content_type"] == content_type

        # Invalid type defaults to equation
        result = _normalize_math_result({"content_type": "invalid_type"})
        assert result["content_type"] == "equation"

        # Case insensitive
        result = _normalize_math_result({"content_type": "EQUATION"})
        assert result["content_type"] == "equation"

    def test_normalize_latex_strips_whitespace(self) -> None:
        """Test that latex is stripped of whitespace."""
        result = _normalize_math_result({"latex": "  x^2 + y^2  "})
        assert result["latex"] == "x^2 + y^2"

    def test_normalize_latex_empty_becomes_empty_string(self) -> None:
        """Test that empty latex becomes empty string."""
        result = _normalize_math_result({"latex": ""})
        assert result["latex"] == ""

        result = _normalize_math_result({"latex": None})
        assert result["latex"] == ""

    def test_normalize_plain_text_strips_whitespace(self) -> None:
        """Test that plain text is stripped of whitespace."""
        result = _normalize_math_result({"plain_text": "  A description  "})
        assert result["plain_text"] == "A description"

    def test_normalize_plain_text_empty_becomes_empty_string(self) -> None:
        """Test that empty plain text becomes empty string."""
        result = _normalize_math_result({"plain_text": ""})
        assert result["plain_text"] == ""

        result = _normalize_math_result({"plain_text": None})
        assert result["plain_text"] == ""

    def test_normalize_variables_dict(self) -> None:
        """Test normalization of variables dictionary."""
        result = _normalize_math_result(
            {"variables": {"x": "unknown", "  y  ": "  second variable  "}}
        )
        assert result["variables"] == {"x": "unknown", "y": "second variable"}

    def test_normalize_variables_empty_dict(self) -> None:
        """Test that empty variables dict becomes None."""
        result = _normalize_math_result({"variables": {}})
        assert result["variables"] is None

    def test_normalize_variables_with_empty_keys_values(self) -> None:
        """Test that empty keys/values are filtered out."""
        result = _normalize_math_result(
            {"variables": {"x": "valid", "": "empty key", "y": ""}}
        )
        assert result["variables"] == {"x": "valid"}

    def test_normalize_variables_non_dict(self) -> None:
        """Test that non-dict variables becomes None."""
        result = _normalize_math_result({"variables": "not a dict"})
        assert result["variables"] is None

        result = _normalize_math_result({"variables": ["x", "y"]})
        assert result["variables"] is None

    def test_normalize_notes_strips_whitespace(self) -> None:
        """Test that notes are stripped of whitespace."""
        result = _normalize_math_result({"notes": "  Some notes here.  "})
        assert result["notes"] == "Some notes here."

    def test_normalize_notes_empty_becomes_none(self) -> None:
        """Test that empty notes become None."""
        result = _normalize_math_result({"notes": ""})
        assert result["notes"] is None

        result = _normalize_math_result({"notes": "   "})
        assert result["notes"] is None
