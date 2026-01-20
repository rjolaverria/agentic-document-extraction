"""Tests for the visual text extraction service."""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from agentic_document_extraction.services.visual_text_extractor import (
    BoundingBox,
    ExtractionMethod,
    PageExtractionResult,
    TextElement,
    VisualExtractionError,
    VisualExtractionResult,
    VisualTextExtractor,
)


@pytest.fixture
def extractor() -> VisualTextExtractor:
    """Create a VisualTextExtractor instance for testing."""
    return VisualTextExtractor()


@pytest.fixture
def sample_image_with_text() -> bytes:
    """Create a sample image with text for testing OCR."""
    # Create a simple image with text
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", 24
            )
        except OSError:
            font = ImageFont.load_default()

    draw.text((10, 30), "Hello World", fill="black", font=font)

    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Create a simple PDF content for testing.

    Note: This creates a minimal PDF with embedded text.
    """
    # Minimal PDF with text "Hello World"
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 24 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF
"""
    return pdf_content


@pytest.fixture
def mock_ocr_data() -> dict[str, list[int | str]]:
    """Mock OCR data for tesseract."""
    return {
        "text": ["Hello", "World", ""],
        "conf": [95, 90, -1],
        "left": [10, 100, 0],
        "top": [10, 10, 0],
        "width": [50, 60, 0],
        "height": [30, 30, 0],
    }


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_create_bounding_box(self) -> None:
        """Test creating a bounding box."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)

        assert bbox.x0 == 10.0
        assert bbox.y0 == 20.0
        assert bbox.x1 == 110.0
        assert bbox.y1 == 70.0

    def test_width_property(self) -> None:
        """Test width calculation."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        assert bbox.width == 100.0

    def test_height_property(self) -> None:
        """Test height calculation."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        assert bbox.height == 50.0

    def test_center_property(self) -> None:
        """Test center point calculation."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        center = bbox.center
        assert center == (60.0, 45.0)

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        result = bbox.to_dict()

        assert result["x0"] == 10.0
        assert result["y0"] == 20.0
        assert result["x1"] == 110.0
        assert result["y1"] == 70.0
        assert result["width"] == 100.0
        assert result["height"] == 50.0


class TestTextElement:
    """Tests for TextElement class."""

    def test_create_text_element(self) -> None:
        """Test creating a text element."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        element = TextElement(
            text="Hello",
            bbox=bbox,
            confidence=0.95,
            page_number=1,
        )

        assert element.text == "Hello"
        assert element.bbox == bbox
        assert element.confidence == 0.95
        assert element.page_number == 1

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0)
        element = TextElement(
            text="World",
            bbox=bbox,
            confidence=0.87,
            page_number=2,
        )

        result = element.to_dict()

        assert result["text"] == "World"
        assert result["confidence"] == 0.87
        assert result["page_number"] == 2
        assert "bbox" in result
        assert result["bbox"]["x0"] == 10.0


class TestPageExtractionResult:
    """Tests for PageExtractionResult class."""

    def test_create_page_result(self) -> None:
        """Test creating a page extraction result."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        element = TextElement(text="Test", bbox=bbox, confidence=0.9, page_number=1)

        result = PageExtractionResult(
            page_number=1,
            text_elements=[element],
            full_text="Test",
            extraction_method=ExtractionMethod.OCR,
            page_width=612.0,
            page_height=792.0,
            average_confidence=0.9,
        )

        assert result.page_number == 1
        assert len(result.text_elements) == 1
        assert result.full_text == "Test"
        assert result.extraction_method == ExtractionMethod.OCR
        assert result.page_width == 612.0
        assert result.page_height == 792.0
        assert result.average_confidence == 0.9

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        element = TextElement(text="Test", bbox=bbox, confidence=0.9, page_number=1)

        result = PageExtractionResult(
            page_number=1,
            text_elements=[element],
            full_text="Test",
            extraction_method=ExtractionMethod.PDF_NATIVE,
            page_width=612.0,
            page_height=792.0,
            average_confidence=0.9,
        )

        result_dict = result.to_dict()

        assert result_dict["page_number"] == 1
        assert len(result_dict["text_elements"]) == 1
        assert result_dict["extraction_method"] == "pdf_native"


class TestVisualExtractionResult:
    """Tests for VisualExtractionResult class."""

    def test_create_result(self) -> None:
        """Test creating a visual extraction result."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        element = TextElement(text="Test", bbox=bbox, confidence=0.9, page_number=1)
        page = PageExtractionResult(
            page_number=1,
            text_elements=[element],
            full_text="Test",
            extraction_method=ExtractionMethod.OCR,
            page_width=612.0,
            page_height=792.0,
            average_confidence=0.9,
        )

        result = VisualExtractionResult(
            pages=[page],
            full_text="Test",
            total_pages=1,
            extraction_method=ExtractionMethod.OCR,
            average_confidence=0.9,
        )

        assert len(result.pages) == 1
        assert result.full_text == "Test"
        assert result.total_pages == 1
        assert result.extraction_method == ExtractionMethod.OCR
        assert result.average_confidence == 0.9
        assert result.metadata == {}

    def test_result_with_metadata(self) -> None:
        """Test result with metadata."""
        result = VisualExtractionResult(
            pages=[],
            full_text="",
            total_pages=0,
            extraction_method=ExtractionMethod.OCR,
            average_confidence=0.0,
            metadata={"source": "test.pdf"},
        )

        assert result.metadata == {"source": "test.pdf"}

    def test_get_all_text_elements(self) -> None:
        """Test getting all text elements from all pages."""
        bbox1 = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        bbox2 = BoundingBox(x0=0, y0=60, x1=100, y1=110)
        element1 = TextElement(text="Page1", bbox=bbox1, confidence=0.9, page_number=1)
        element2 = TextElement(text="Page2", bbox=bbox2, confidence=0.8, page_number=2)

        page1 = PageExtractionResult(
            page_number=1,
            text_elements=[element1],
            full_text="Page1",
            extraction_method=ExtractionMethod.OCR,
            page_width=612.0,
            page_height=792.0,
            average_confidence=0.9,
        )
        page2 = PageExtractionResult(
            page_number=2,
            text_elements=[element2],
            full_text="Page2",
            extraction_method=ExtractionMethod.OCR,
            page_width=612.0,
            page_height=792.0,
            average_confidence=0.8,
        )

        result = VisualExtractionResult(
            pages=[page1, page2],
            full_text="Page1\n\nPage2",
            total_pages=2,
            extraction_method=ExtractionMethod.OCR,
            average_confidence=0.85,
        )

        all_elements = result.get_all_text_elements()
        assert len(all_elements) == 2
        assert all_elements[0].text == "Page1"
        assert all_elements[1].text == "Page2"

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        result = VisualExtractionResult(
            pages=[],
            full_text="Test content",
            total_pages=1,
            extraction_method=ExtractionMethod.HYBRID,
            average_confidence=0.85,
            metadata={"key": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["full_text"] == "Test content"
        assert result_dict["total_pages"] == 1
        assert result_dict["extraction_method"] == "hybrid"
        assert result_dict["average_confidence"] == 0.85
        assert result_dict["metadata"] == {"key": "value"}


class TestVisualTextExtractorInit:
    """Tests for VisualTextExtractor initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        extractor = VisualTextExtractor()

        assert extractor.ocr_language == "eng"
        assert extractor.pdf_dpi == 300

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        extractor = VisualTextExtractor(ocr_language="deu", pdf_dpi=150)

        assert extractor.ocr_language == "deu"
        assert extractor.pdf_dpi == 150


class TestVisualTextExtractorStaticMethods:
    """Tests for static methods."""

    def test_get_supported_extensions(self) -> None:
        """Test getting supported extensions."""
        extensions = VisualTextExtractor.get_supported_extensions()

        assert ".pdf" in extensions
        assert ".png" in extensions
        assert ".jpg" in extensions
        assert ".jpeg" in extensions
        assert ".tiff" in extensions
        assert ".gif" in extensions
        assert ".webp" in extensions
        assert ".bmp" in extensions

    def test_is_pdf(self) -> None:
        """Test PDF extension check."""
        assert VisualTextExtractor.is_pdf(".pdf") is True
        assert VisualTextExtractor.is_pdf("pdf") is True
        assert VisualTextExtractor.is_pdf(".PDF") is True
        assert VisualTextExtractor.is_pdf(".png") is False

    def test_is_image(self) -> None:
        """Test image extension check."""
        assert VisualTextExtractor.is_image(".png") is True
        assert VisualTextExtractor.is_image(".jpg") is True
        assert VisualTextExtractor.is_image(".jpeg") is True
        assert VisualTextExtractor.is_image("png") is True
        assert VisualTextExtractor.is_image(".PNG") is True
        assert VisualTextExtractor.is_image(".pdf") is False
        assert VisualTextExtractor.is_image(".txt") is False


class TestVisualTextExtractorErrors:
    """Tests for error handling."""

    def test_unsupported_extension_raises(self, extractor: VisualTextExtractor) -> None:
        """Test that unsupported extension raises error."""
        with pytest.raises(VisualExtractionError) as exc_info:
            extractor.extract_from_content(b"test", ".txt")

        assert "Unsupported file extension" in str(exc_info.value)

    def test_nonexistent_file_raises(self, extractor: VisualTextExtractor) -> None:
        """Test that extracting from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            extractor.extract_from_path("/nonexistent/path/file.pdf")

    def test_error_includes_file_path(self, extractor: VisualTextExtractor) -> None:
        """Test that error includes file path when available."""
        with pytest.raises(VisualExtractionError) as exc_info:
            extractor.extract_from_content(b"test", ".xyz", filename="test.xyz")

        assert exc_info.value.file_path == "test.xyz"


class TestVisualTextExtractorImage:
    """Tests for image extraction with OCR (using mocks)."""

    def test_extract_png_image(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test extracting text from PNG image."""
        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_from_content(sample_image_with_text, ".png")

            assert result.total_pages == 1
            assert result.extraction_method == ExtractionMethod.OCR
            assert len(result.pages) == 1
            assert result.pages[0].extraction_method == ExtractionMethod.OCR
            assert "Hello World" in result.full_text

    def test_extract_jpg_image(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test extracting text from JPG image."""
        # Convert PNG to JPG
        img = Image.open(io.BytesIO(sample_image_with_text))
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG")
        jpg_content = buffer.getvalue()

        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_from_content(jpg_content, ".jpg")

            assert result.total_pages == 1
            assert result.extraction_method == ExtractionMethod.OCR

    def test_extract_image_from_path(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test extracting from image file path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(sample_image_with_text)
            temp_path = f.name

        try:
            with (
                patch("pytesseract.image_to_data") as mock_data,
                patch("pytesseract.image_to_string") as mock_string,
            ):
                mock_data.return_value = mock_ocr_data
                mock_string.return_value = "Hello World"

                result = extractor.extract_from_path(temp_path)
                assert result.total_pages == 1
                assert result.extraction_method == ExtractionMethod.OCR
        finally:
            Path(temp_path).unlink()

    def test_extract_image_public_method(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test public extract_image method."""
        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_image(sample_image_with_text)

            assert result.total_pages == 1
            assert result.extraction_method == ExtractionMethod.OCR

    def test_image_metadata_included(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test that image metadata is included in result."""
        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_from_content(sample_image_with_text, ".png")

            assert "image_width" in result.metadata
            assert "image_height" in result.metadata
            assert result.metadata["image_width"] == 400
            assert result.metadata["image_height"] == 100

    def test_text_elements_have_bounding_boxes(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test that text elements have bounding boxes."""
        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_from_content(sample_image_with_text, ".png")

            # Check that if we have elements, they have proper bounding boxes
            for page in result.pages:
                for element in page.text_elements:
                    assert isinstance(element.bbox, BoundingBox)
                    assert element.bbox.x1 >= element.bbox.x0
                    assert element.bbox.y1 >= element.bbox.y0

    def test_confidence_scores_in_range(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test that confidence scores are in valid range."""
        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_from_content(sample_image_with_text, ".png")

            assert 0.0 <= result.average_confidence <= 1.0
            for page in result.pages:
                assert 0.0 <= page.average_confidence <= 1.0
                for element in page.text_elements:
                    assert 0.0 <= element.confidence <= 1.0


class TestVisualTextExtractorPDF:
    """Tests for PDF extraction."""

    def test_extract_pdf_basic(
        self,
        extractor: VisualTextExtractor,
        sample_pdf_content: bytes,
    ) -> None:
        """Test basic PDF extraction."""
        result = extractor.extract_from_content(sample_pdf_content, ".pdf")

        assert result.total_pages >= 1
        assert len(result.pages) >= 1

    def test_extract_pdf_public_method(
        self,
        extractor: VisualTextExtractor,
        sample_pdf_content: bytes,
    ) -> None:
        """Test public extract_pdf method."""
        result = extractor.extract_pdf(sample_pdf_content)

        assert result.total_pages >= 1

    def test_extract_pdf_from_path(
        self,
        extractor: VisualTextExtractor,
        sample_pdf_content: bytes,
    ) -> None:
        """Test extracting from PDF file path."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(sample_pdf_content)
            temp_path = f.name

        try:
            result = extractor.extract_from_path(temp_path)
            assert result.total_pages >= 1
        finally:
            Path(temp_path).unlink()

    def test_pdf_metadata_included(
        self,
        extractor: VisualTextExtractor,
        sample_pdf_content: bytes,
    ) -> None:
        """Test that PDF metadata is included in result."""
        result = extractor.extract_from_content(
            sample_pdf_content, ".pdf", filename="test.pdf"
        )

        assert "pdf_dpi" in result.metadata
        assert result.metadata["pdf_dpi"] == 300


class TestVisualTextExtractorExtensionHandling:
    """Tests for extension handling."""

    def test_extension_without_dot(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test that extension without dot is handled."""
        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_from_content(sample_image_with_text, "png")

            assert result.total_pages == 1

    def test_uppercase_extension(
        self,
        extractor: VisualTextExtractor,
        sample_image_with_text: bytes,
        mock_ocr_data: dict[str, list[int | str]],
    ) -> None:
        """Test that uppercase extension is handled."""
        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_from_content(sample_image_with_text, ".PNG")

            assert result.total_pages == 1


class TestExtractionMethod:
    """Tests for ExtractionMethod enum."""

    def test_extraction_methods(self) -> None:
        """Test extraction method values."""
        assert ExtractionMethod.PDF_NATIVE.value == "pdf_native"
        assert ExtractionMethod.OCR.value == "ocr"
        assert ExtractionMethod.HYBRID.value == "hybrid"


class TestVisualExtractionError:
    """Tests for VisualExtractionError."""

    def test_error_with_message(self) -> None:
        """Test error with just message."""
        error = VisualExtractionError("Test error")

        assert str(error) == "Test error"
        assert error.file_path is None
        assert error.page_number is None

    def test_error_with_file_path(self) -> None:
        """Test error with file path."""
        error = VisualExtractionError("Test error", file_path="/path/to/file.pdf")

        assert error.file_path == "/path/to/file.pdf"

    def test_error_with_page_number(self) -> None:
        """Test error with page number."""
        error = VisualExtractionError(
            "Test error", file_path="/path/to/file.pdf", page_number=5
        )

        assert error.page_number == 5


class TestOCRMocking:
    """Tests with mocked OCR for controlled testing."""

    def test_ocr_with_mocked_tesseract(self, extractor: VisualTextExtractor) -> None:
        """Test OCR extraction with mocked Tesseract response."""
        # Create a simple test image
        img = Image.new("RGB", (200, 50), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_content = buffer.getvalue()

        # Mock Tesseract responses
        mock_ocr_data = {
            "text": ["Hello", "World", ""],
            "conf": [95, 90, -1],
            "left": [10, 100, 0],
            "top": [10, 10, 0],
            "width": [50, 60, 0],
            "height": [30, 30, 0],
        }

        with (
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Hello World"

            result = extractor.extract_from_content(img_content, ".png")

            assert result.total_pages == 1
            assert "Hello World" in result.full_text

            # Verify text elements were created
            page = result.pages[0]
            assert len(page.text_elements) == 2

            # Check first element
            assert page.text_elements[0].text == "Hello"
            assert page.text_elements[0].confidence == 0.95
            assert page.text_elements[0].bbox.x0 == 10
            assert page.text_elements[0].bbox.y0 == 10

            # Check second element
            assert page.text_elements[1].text == "World"
            assert page.text_elements[1].confidence == 0.90


class TestPDFMocking:
    """Tests with mocked PDF extraction for controlled testing."""

    def test_pdf_native_extraction_mocked(self, extractor: VisualTextExtractor) -> None:
        """Test PDF native extraction with mocked pdfplumber."""
        # Create mock page
        mock_page = MagicMock()
        mock_page.width = 612
        mock_page.height = 792
        mock_page.extract_words.return_value = [
            {
                "text": "Hello",
                "x0": 100,
                "top": 700,
                "x1": 150,
                "bottom": 724,
                "fontname": "Helvetica",
                "size": 24,
            },
            {
                "text": "World",
                "x0": 160,
                "top": 700,
                "x1": 220,
                "bottom": 724,
                "fontname": "Helvetica",
                "size": 24,
            },
        ]
        mock_page.extract_text.return_value = "Hello World"

        # Create mock PDF
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value = mock_pdf

            result = extractor.extract_from_content(b"fake pdf content", ".pdf")

            assert result.total_pages == 1
            assert result.extraction_method == ExtractionMethod.PDF_NATIVE
            assert "Hello World" in result.full_text

            # Verify text elements
            page = result.pages[0]
            assert len(page.text_elements) == 2
            assert page.text_elements[0].text == "Hello"
            assert page.text_elements[1].text == "World"

    def test_pdf_ocr_fallback_mocked(self, extractor: VisualTextExtractor) -> None:
        """Test PDF falls back to OCR when native extraction yields no text."""
        # Create mock page with no text
        mock_page = MagicMock()
        mock_page.width = 612
        mock_page.height = 792
        mock_page.extract_words.return_value = []  # No native text
        mock_page.extract_text.return_value = ""  # Empty text

        # Create mock PDF
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        # Create a test image for OCR
        img = Image.new("RGB", (612, 792), color="white")

        # Mock OCR responses
        mock_ocr_data = {
            "text": ["OCR", "Text", ""],
            "conf": [90, 85, -1],
            "left": [100, 100, 0],
            "top": [100, 130, 0],
            "width": [50, 50, 0],
            "height": [20, 20, 0],
        }

        with (
            patch("pdfplumber.open") as mock_open,
            patch("pdf2image.convert_from_bytes") as mock_convert,
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_open.return_value = mock_pdf
            mock_convert.return_value = [img]
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "OCR Text"

            result = extractor.extract_from_content(b"fake pdf content", ".pdf")

            assert result.total_pages == 1
            assert result.extraction_method == ExtractionMethod.OCR
            assert "OCR Text" in result.full_text


class TestMultiPageDocument:
    """Tests for multi-page document handling."""

    def test_multipage_pdf_mocked(self, extractor: VisualTextExtractor) -> None:
        """Test multi-page PDF extraction."""
        # Create mock pages
        mock_page1 = MagicMock()
        mock_page1.width = 612
        mock_page1.height = 792
        mock_page1.extract_words.return_value = [
            {"text": "Page1", "x0": 100, "top": 700, "x1": 150, "bottom": 724}
        ]
        mock_page1.extract_text.return_value = "Page 1 content"

        mock_page2 = MagicMock()
        mock_page2.width = 612
        mock_page2.height = 792
        mock_page2.extract_words.return_value = [
            {"text": "Page2", "x0": 100, "top": 700, "x1": 150, "bottom": 724}
        ]
        mock_page2.extract_text.return_value = "Page 2 content"

        # Create mock PDF with two pages
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value = mock_pdf

            result = extractor.extract_from_content(b"fake pdf content", ".pdf")

            assert result.total_pages == 2
            assert len(result.pages) == 2
            assert result.pages[0].page_number == 1
            assert result.pages[1].page_number == 2
            assert "Page 1 content" in result.full_text
            assert "Page 2 content" in result.full_text

    def test_multipage_tiff_mocked(self, extractor: VisualTextExtractor) -> None:
        """Test multi-page TIFF image extraction."""
        # Create a mock multi-frame image
        mock_image = MagicMock(spec=Image.Image)
        mock_image.n_frames = 2
        mock_image.mode = "RGB"
        mock_image.width = 400
        mock_image.height = 300
        mock_image.format = "TIFF"
        mock_image.copy.return_value = mock_image
        mock_image.convert.return_value = mock_image

        # Mock OCR responses
        mock_ocr_data = {
            "text": ["Frame", ""],
            "conf": [90, -1],
            "left": [10, 0],
            "top": [10, 0],
            "width": [50, 0],
            "height": [20, 0],
        }

        with (
            patch("PIL.Image.open") as mock_open,
            patch("pytesseract.image_to_data") as mock_data,
            patch("pytesseract.image_to_string") as mock_string,
        ):
            mock_open.return_value = mock_image
            mock_data.return_value = mock_ocr_data
            mock_string.return_value = "Frame text"

            result = extractor.extract_from_content(b"fake tiff content", ".tiff")

            assert result.total_pages == 2
            assert len(result.pages) == 2
            assert result.metadata.get("multipage") is True


class TestReadingOrderPreservation:
    """Tests for reading order preservation."""

    def test_pdf_preserves_text_order(self, extractor: VisualTextExtractor) -> None:
        """Test that PDF extraction preserves text order."""
        # Create mock page with ordered text
        mock_page = MagicMock()
        mock_page.width = 612
        mock_page.height = 792
        mock_page.extract_words.return_value = [
            {"text": "First", "x0": 100, "top": 100, "x1": 150, "bottom": 124},
            {"text": "Second", "x0": 100, "top": 150, "x1": 160, "bottom": 174},
            {"text": "Third", "x0": 100, "top": 200, "x1": 150, "bottom": 224},
        ]
        mock_page.extract_text.return_value = "First Second Third"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value = mock_pdf

            result = extractor.extract_from_content(b"fake pdf content", ".pdf")

            # Verify order is preserved
            elements = result.pages[0].text_elements
            assert elements[0].text == "First"
            assert elements[1].text == "Second"
            assert elements[2].text == "Third"
