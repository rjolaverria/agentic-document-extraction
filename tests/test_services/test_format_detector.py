"""Tests for the document format detection service."""

import tempfile
from pathlib import Path

import pytest

from agentic_document_extraction.models import (
    FormatFamily,
    FormatInfo,
    ProcessingCategory,
)
from agentic_document_extraction.services.format_detector import (
    EXTENSION_TO_MIME,
    MIME_TO_FORMAT_FAMILY,
    MIME_TO_PROCESSING_CATEGORY,
    SUPPORTED_MIME_TYPES,
    FormatDetector,
    UnsupportedFormatError,
)

# Magic bytes for common file formats
# These are proper minimal file structures that libmagic will recognize
MAGIC_BYTES: dict[str, bytes] = {
    # PDF: %PDF-1.x header
    "pdf": b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n",
    # PNG: Full PNG header with IHDR chunk (minimal 1x1 transparent PNG)
    "png": (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR"  # IHDR chunk length (13) and type
        b"\x00\x00\x00\x01"  # width = 1
        b"\x00\x00\x00\x01"  # height = 1
        b"\x08\x06"  # bit depth = 8, color type = 6 (RGBA)
        b"\x00\x00\x00"  # compression, filter, interlace
        b"\x1f\x15\xc4\x89"  # CRC
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"  # IDAT
        b"\x0d\n\x2d\xb4"  # CRC
        b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
    ),
    # JPEG: Full JFIF header
    "jpeg": (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xd9"  # EOI marker
    ),
    # GIF: GIF89a header with proper structure
    "gif": (
        b"GIF89a"  # GIF signature
        b"\x01\x00\x01\x00"  # width=1, height=1
        b"\x00\x00\x00"  # packed byte, bg color, aspect ratio
    ),
    # BMP: Proper BMP file header for a 1x1 pixel image
    "bmp": (
        b"BM"  # Signature
        b"\x46\x00\x00\x00"  # File size (70 bytes)
        b"\x00\x00"  # Reserved
        b"\x00\x00"  # Reserved
        b"\x36\x00\x00\x00"  # Offset to pixel data (54)
        b"\x28\x00\x00\x00"  # DIB header size (40)
        b"\x01\x00\x00\x00"  # Width (1)
        b"\x01\x00\x00\x00"  # Height (1)
        b"\x01\x00"  # Planes (1)
        b"\x18\x00"  # Bits per pixel (24)
        b"\x00\x00\x00\x00"  # Compression (none)
        b"\x10\x00\x00\x00"  # Image size
        b"\x13\x0b\x00\x00"  # X pixels per meter
        b"\x13\x0b\x00\x00"  # Y pixels per meter
        b"\x00\x00\x00\x00"  # Colors in color table
        b"\x00\x00\x00\x00"  # Important colors
        b"\x00\x00\xff\x00"  # Pixel data (1 red pixel + padding)
    ),
    # TIFF (little-endian): Proper TIFF header
    "tiff": (
        b"II"  # Little-endian
        b"\x2a\x00"  # TIFF magic number
        b"\x08\x00\x00\x00"  # Offset to first IFD
        b"\x00\x00"  # Number of directory entries (placeholder for detection)
    ),
    # WebP: Proper RIFF/WebP header
    "webp": (
        b"RIFF"
        b"\x24\x00\x00\x00"  # File size - 8
        b"WEBP"
        b"VP8 "  # VP8 chunk
        b"\x18\x00\x00\x00" + b"\x00" * 24  # Chunk size  # VP8 bitstream placeholder
    ),
    # Plain text (ASCII)
    "txt": b"This is a plain text file.\nWith multiple lines.\n",
    # CSV text
    "csv": b"name,age,city\nJohn,30,NYC\nJane,25,LA\n",
}


@pytest.fixture
def detector() -> FormatDetector:
    """Create a FormatDetector instance for testing."""
    return FormatDetector()


class TestFormatDetectorInit:
    """Tests for FormatDetector initialization."""

    def test_creates_instance(self) -> None:
        """Test that FormatDetector can be instantiated."""
        detector = FormatDetector()
        assert detector is not None


class TestDetectFromContent:
    """Tests for detect_from_content method."""

    def test_detects_pdf_from_content(self, detector: FormatDetector) -> None:
        """Test PDF detection from magic bytes."""
        result = detector.detect_from_content(MAGIC_BYTES["pdf"])

        assert result.mime_type == "application/pdf"
        assert result.extension == ".pdf"
        assert result.format_family == FormatFamily.PDF
        assert result.processing_category == ProcessingCategory.VISUAL
        assert result.detected_from_content is True

    def test_detects_png_from_content(self, detector: FormatDetector) -> None:
        """Test PNG detection from magic bytes."""
        result = detector.detect_from_content(MAGIC_BYTES["png"])

        assert result.mime_type == "image/png"
        assert result.extension == ".png"
        assert result.format_family == FormatFamily.IMAGE
        assert result.processing_category == ProcessingCategory.VISUAL
        assert result.detected_from_content is True

    def test_detects_jpeg_from_content(self, detector: FormatDetector) -> None:
        """Test JPEG detection from magic bytes."""
        result = detector.detect_from_content(MAGIC_BYTES["jpeg"])

        assert result.mime_type == "image/jpeg"
        assert result.extension == ".jpg"
        assert result.format_family == FormatFamily.IMAGE
        assert result.processing_category == ProcessingCategory.VISUAL
        assert result.detected_from_content is True

    def test_detects_gif_from_content(self, detector: FormatDetector) -> None:
        """Test GIF detection from magic bytes."""
        result = detector.detect_from_content(MAGIC_BYTES["gif"])

        assert result.mime_type == "image/gif"
        assert result.extension == ".gif"
        assert result.format_family == FormatFamily.IMAGE
        assert result.processing_category == ProcessingCategory.VISUAL
        assert result.detected_from_content is True

    def test_detects_bmp_from_content(self, detector: FormatDetector) -> None:
        """Test BMP detection from magic bytes."""
        result = detector.detect_from_content(MAGIC_BYTES["bmp"])

        assert result.mime_type == "image/bmp"
        assert result.extension == ".bmp"
        assert result.format_family == FormatFamily.IMAGE
        assert result.processing_category == ProcessingCategory.VISUAL
        assert result.detected_from_content is True

    def test_detects_tiff_from_content(self, detector: FormatDetector) -> None:
        """Test TIFF detection from magic bytes."""
        result = detector.detect_from_content(MAGIC_BYTES["tiff"])

        assert result.mime_type == "image/tiff"
        assert result.extension == ".tiff"
        assert result.format_family == FormatFamily.IMAGE
        assert result.processing_category == ProcessingCategory.VISUAL
        assert result.detected_from_content is True

    def test_detects_webp_from_content(self, detector: FormatDetector) -> None:
        """Test WebP detection from magic bytes."""
        result = detector.detect_from_content(MAGIC_BYTES["webp"])

        assert result.mime_type == "image/webp"
        assert result.extension == ".webp"
        assert result.format_family == FormatFamily.IMAGE
        assert result.processing_category == ProcessingCategory.VISUAL
        assert result.detected_from_content is True

    def test_detects_txt_from_extension_fallback(
        self, detector: FormatDetector
    ) -> None:
        """Test TXT detection with extension fallback."""
        # Plain text is detected as text/plain by magic
        result = detector.detect_from_content(MAGIC_BYTES["txt"], filename="test.txt")

        assert result.mime_type == "text/plain"
        assert result.extension == ".txt"
        assert result.format_family == FormatFamily.PLAIN_TEXT
        assert result.processing_category == ProcessingCategory.TEXT_BASED

    def test_detects_csv_from_extension_fallback(
        self, detector: FormatDetector
    ) -> None:
        """Test CSV detection with extension when content is ambiguous."""
        # CSV content might be detected as text/plain, so we use extension
        result = detector.detect_from_content(MAGIC_BYTES["csv"], filename="data.csv")

        # Should be text/csv based on extension, or text/plain from content
        assert result.mime_type in ("text/csv", "text/plain")
        assert result.format_family in (
            FormatFamily.SPREADSHEET,
            FormatFamily.PLAIN_TEXT,
        )
        assert result.processing_category == ProcessingCategory.TEXT_BASED


class TestDetectFromPath:
    """Tests for detect_from_path method."""

    def test_detects_pdf_from_file(self, detector: FormatDetector) -> None:
        """Test PDF detection from actual file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(MAGIC_BYTES["pdf"])
            f.flush()

            result = detector.detect_from_path(f.name)

            assert result.mime_type == "application/pdf"
            assert result.format_family == FormatFamily.PDF

            Path(f.name).unlink()

    def test_detects_png_from_file(self, detector: FormatDetector) -> None:
        """Test PNG detection from actual file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(MAGIC_BYTES["png"])
            f.flush()

            result = detector.detect_from_path(f.name)

            assert result.mime_type == "image/png"
            assert result.format_family == FormatFamily.IMAGE

            Path(f.name).unlink()

    def test_raises_for_nonexistent_file(self, detector: FormatDetector) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            detector.detect_from_path("/nonexistent/path/file.pdf")

    def test_detects_format_with_wrong_extension(
        self, detector: FormatDetector
    ) -> None:
        """Test detection when file extension doesn't match content."""
        # Create a PNG file with .txt extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(MAGIC_BYTES["png"])
            f.flush()

            result = detector.detect_from_path(f.name)

            # Should detect as PNG from content, note the extension mismatch
            assert result.mime_type == "image/png"
            assert result.detected_from_content is True
            assert result.original_extension == ".txt"

            Path(f.name).unlink()

    def test_path_object_support(self, detector: FormatDetector) -> None:
        """Test that Path objects are supported."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(MAGIC_BYTES["pdf"])
            f.flush()

            path = Path(f.name)
            result = detector.detect_from_path(path)

            assert result.mime_type == "application/pdf"

            path.unlink()


class TestExtensionBasedDetection:
    """Tests for extension-based format detection."""

    @pytest.mark.parametrize(
        "extension,expected_mime",
        [
            (".txt", "text/plain"),
            (".csv", "text/csv"),
            (".pdf", "application/pdf"),
            (".doc", "application/msword"),
            (
                ".docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            (".odt", "application/vnd.oasis.opendocument.text"),
            (".ppt", "application/vnd.ms-powerpoint"),
            (
                ".pptx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
            (".odp", "application/vnd.oasis.opendocument.presentation"),
            (
                ".xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            (".jpeg", "image/jpeg"),
            (".jpg", "image/jpeg"),
            (".png", "image/png"),
            (".bmp", "image/bmp"),
            (".psd", "image/vnd.adobe.photoshop"),
            (".tiff", "image/tiff"),
            (".tif", "image/tiff"),
            (".gif", "image/gif"),
            (".webp", "image/webp"),
        ],
    )
    def test_all_supported_extensions(
        self, detector: FormatDetector, extension: str, expected_mime: str
    ) -> None:
        """Test that all supported extensions are correctly mapped."""
        # Use generic binary content that won't be detected as specific format
        generic_content = b"\x00" * 10

        result = detector.detect_from_content(
            generic_content, filename=f"test{extension}"
        )

        assert result.mime_type == expected_mime


class TestProcessingCategoryClassification:
    """Tests for processing category classification."""

    @pytest.mark.parametrize(
        "mime_type,expected_category",
        [
            ("text/plain", ProcessingCategory.TEXT_BASED),
            ("text/csv", ProcessingCategory.TEXT_BASED),
            (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ProcessingCategory.STRUCTURED,
            ),
            ("application/pdf", ProcessingCategory.VISUAL),
            ("application/msword", ProcessingCategory.VISUAL),
            (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ProcessingCategory.VISUAL,
            ),
            ("image/png", ProcessingCategory.VISUAL),
            ("image/jpeg", ProcessingCategory.VISUAL),
        ],
    )
    def test_processing_category_assignment(
        self, mime_type: str, expected_category: ProcessingCategory
    ) -> None:
        """Test that processing categories are correctly assigned."""
        assert MIME_TO_PROCESSING_CATEGORY.get(mime_type) == expected_category


class TestFormatFamilyClassification:
    """Tests for format family classification."""

    @pytest.mark.parametrize(
        "mime_type,expected_family",
        [
            ("text/plain", FormatFamily.PLAIN_TEXT),
            ("text/csv", FormatFamily.SPREADSHEET),
            ("application/pdf", FormatFamily.PDF),
            ("application/msword", FormatFamily.DOCUMENT),
            (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                FormatFamily.DOCUMENT,
            ),
            ("application/vnd.oasis.opendocument.text", FormatFamily.DOCUMENT),
            ("application/vnd.ms-powerpoint", FormatFamily.PRESENTATION),
            (
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                FormatFamily.PRESENTATION,
            ),
            (
                "application/vnd.oasis.opendocument.presentation",
                FormatFamily.PRESENTATION,
            ),
            (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                FormatFamily.SPREADSHEET,
            ),
            ("image/png", FormatFamily.IMAGE),
            ("image/jpeg", FormatFamily.IMAGE),
            ("image/gif", FormatFamily.IMAGE),
            ("image/bmp", FormatFamily.IMAGE),
            ("image/tiff", FormatFamily.IMAGE),
            ("image/webp", FormatFamily.IMAGE),
            ("image/vnd.adobe.photoshop", FormatFamily.IMAGE),
        ],
    )
    def test_format_family_assignment(
        self, mime_type: str, expected_family: FormatFamily
    ) -> None:
        """Test that format families are correctly assigned."""
        assert MIME_TO_FORMAT_FAMILY.get(mime_type) == expected_family


class TestUnsupportedFormats:
    """Tests for handling unsupported formats."""

    def test_raises_for_unsupported_content(self, detector: FormatDetector) -> None:
        """Test that unsupported formats raise UnsupportedFormatError."""
        # Use a format we don't support (e.g., executable)
        exe_magic = b"MZ" + b"\x00" * 100  # DOS/Windows executable signature

        with pytest.raises(UnsupportedFormatError) as exc_info:
            detector.detect_from_content(exe_magic)

        assert exc_info.value.detected_mime is not None

    def test_raises_for_empty_content_no_extension(
        self, detector: FormatDetector
    ) -> None:
        """Test that empty content with no extension raises error."""
        with pytest.raises(UnsupportedFormatError):
            detector.detect_from_content(b"")

    def test_raises_for_unknown_extension_no_content(
        self, detector: FormatDetector
    ) -> None:
        """Test that unknown extension with undetectable content raises error."""
        # Binary content that won't be detected as any specific format
        with pytest.raises(UnsupportedFormatError):
            detector.detect_from_content(b"\x00" * 10, filename="test.xyz")


class TestEdgeCases:
    """Tests for edge cases in format detection."""

    def test_handles_missing_extension(self, detector: FormatDetector) -> None:
        """Test detection when filename has no extension."""
        result = detector.detect_from_content(MAGIC_BYTES["pdf"], filename="document")

        assert result.mime_type == "application/pdf"
        assert result.detected_from_content is True

    def test_handles_uppercase_extension(self, detector: FormatDetector) -> None:
        """Test that uppercase extensions are handled correctly."""
        # Create a file with uppercase extension
        with tempfile.NamedTemporaryFile(suffix=".PDF", delete=False) as f:
            f.write(MAGIC_BYTES["pdf"])
            f.flush()

            result = detector.detect_from_path(f.name)

            assert result.mime_type == "application/pdf"

            Path(f.name).unlink()

    def test_handles_mixed_case_extension(self, detector: FormatDetector) -> None:
        """Test that mixed case extensions are handled correctly."""
        with tempfile.NamedTemporaryFile(suffix=".PdF", delete=False) as f:
            f.write(MAGIC_BYTES["pdf"])
            f.flush()

            result = detector.detect_from_path(f.name)

            assert result.mime_type == "application/pdf"

            Path(f.name).unlink()

    def test_handles_double_extension(self, detector: FormatDetector) -> None:
        """Test files with double extensions like .tar.gz."""
        # PDF content with .tar.gz extension (should detect from content)
        result = detector.detect_from_content(
            MAGIC_BYTES["pdf"], filename="file.tar.gz"
        )

        # Should detect PDF from content regardless of extension
        assert result.mime_type == "application/pdf"
        assert result.detected_from_content is True


class TestStaticMethods:
    """Tests for static utility methods."""

    def test_get_supported_extensions(self) -> None:
        """Test get_supported_extensions returns all supported extensions."""
        extensions = FormatDetector.get_supported_extensions()

        assert isinstance(extensions, list)
        assert len(extensions) > 0
        assert all(ext.startswith(".") for ext in extensions)
        assert ".pdf" in extensions
        assert ".txt" in extensions
        assert ".png" in extensions

    def test_get_supported_mime_types(self) -> None:
        """Test get_supported_mime_types returns all supported MIME types."""
        mime_types = FormatDetector.get_supported_mime_types()

        assert isinstance(mime_types, list)
        assert len(mime_types) > 0
        assert "application/pdf" in mime_types
        assert "text/plain" in mime_types
        assert "image/png" in mime_types

    def test_is_text_based(self) -> None:
        """Test is_text_based static method."""
        text_info = FormatInfo(
            mime_type="text/plain",
            extension=".txt",
            format_family=FormatFamily.PLAIN_TEXT,
            processing_category=ProcessingCategory.TEXT_BASED,
        )
        visual_info = FormatInfo(
            mime_type="application/pdf",
            extension=".pdf",
            format_family=FormatFamily.PDF,
            processing_category=ProcessingCategory.VISUAL,
        )

        assert FormatDetector.is_text_based(text_info) is True
        assert FormatDetector.is_text_based(visual_info) is False

    def test_is_visual(self) -> None:
        """Test is_visual static method."""
        text_info = FormatInfo(
            mime_type="text/plain",
            extension=".txt",
            format_family=FormatFamily.PLAIN_TEXT,
            processing_category=ProcessingCategory.TEXT_BASED,
        )
        visual_info = FormatInfo(
            mime_type="application/pdf",
            extension=".pdf",
            format_family=FormatFamily.PDF,
            processing_category=ProcessingCategory.VISUAL,
        )

        assert FormatDetector.is_visual(text_info) is False
        assert FormatDetector.is_visual(visual_info) is True

    def test_is_structured(self) -> None:
        """Test is_structured static method."""
        structured_info = FormatInfo(
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            extension=".xlsx",
            format_family=FormatFamily.SPREADSHEET,
            processing_category=ProcessingCategory.STRUCTURED,
        )
        assert FormatDetector.is_structured(structured_info) is True


class TestSupportedFormatsCompleteness:
    """Tests to verify all required formats are supported."""

    def test_all_required_extensions_supported(self) -> None:
        """Verify all required extensions from the spec are supported."""
        required_extensions = [
            ".txt",
            ".pdf",
            ".doc",
            ".docx",
            ".odt",
            ".ppt",
            ".pptx",
            ".odp",
            ".csv",
            ".xlsx",
            ".jpeg",
            ".jpg",
            ".png",
            ".bmp",
            ".psd",
            ".tiff",
            ".tif",
            ".gif",
            ".webp",
        ]

        supported = set(EXTENSION_TO_MIME.keys())
        for ext in required_extensions:
            assert ext in supported, f"Extension {ext} is not supported"

    def test_text_based_formats_correct(self) -> None:
        """Verify text-based formats are correctly categorized."""
        text_based_mimes = ["text/plain", "text/csv"]

        for mime in text_based_mimes:
            assert mime in SUPPORTED_MIME_TYPES
            assert MIME_TO_PROCESSING_CATEGORY[mime] == ProcessingCategory.TEXT_BASED

    def test_visual_formats_correct(self) -> None:
        """Verify visual formats are correctly categorized."""
        visual_mimes = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.oasis.opendocument.text",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.oasis.opendocument.presentation",
            "image/jpeg",
            "image/png",
            "image/bmp",
            "image/vnd.adobe.photoshop",
            "image/tiff",
            "image/gif",
            "image/webp",
        ]

        for mime in visual_mimes:
            assert mime in SUPPORTED_MIME_TYPES
            assert MIME_TO_PROCESSING_CATEGORY[mime] == ProcessingCategory.VISUAL

    def test_structured_formats_correct(self) -> None:
        """Verify structured formats are correctly categorized."""
        structured_mimes = [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]
        for mime in structured_mimes:
            assert mime in SUPPORTED_MIME_TYPES
            assert MIME_TO_PROCESSING_CATEGORY[mime] == ProcessingCategory.STRUCTURED
