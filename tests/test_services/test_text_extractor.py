"""Tests for the text extraction service."""

import tempfile
from pathlib import Path

import pytest

from agentic_document_extraction.services.text_extractor import (
    CSVMetadata,
    StructureType,
    TextExtractionError,
    TextExtractionResult,
    TextExtractor,
)


@pytest.fixture
def extractor() -> TextExtractor:
    """Create a TextExtractor instance for testing."""
    return TextExtractor()


class TestTextExtractionResult:
    """Tests for TextExtractionResult class."""

    def test_basic_result_creation(self) -> None:
        """Test creating a basic extraction result."""
        result = TextExtractionResult(
            text="Hello, World!",
            encoding="utf-8",
            encoding_confidence=0.99,
            line_count=1,
            structure_type=StructureType.PLAIN_TEXT,
        )

        assert result.text == "Hello, World!"
        assert result.encoding == "utf-8"
        assert result.encoding_confidence == 0.99
        assert result.line_count == 1
        assert result.structure_type == StructureType.PLAIN_TEXT
        assert result.metadata == {}

    def test_result_with_metadata(self) -> None:
        """Test creating a result with metadata."""
        result = TextExtractionResult(
            text="test",
            encoding="utf-8",
            encoding_confidence=0.9,
            line_count=1,
            structure_type=StructureType.TABULAR,
            metadata={"key": "value"},
        )

        assert result.metadata == {"key": "value"}

    def test_result_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = TextExtractionResult(
            text="Hello",
            encoding="utf-8",
            encoding_confidence=0.95,
            line_count=1,
            structure_type=StructureType.PLAIN_TEXT,
            metadata={"char_count": 5},
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Hello"
        assert result_dict["encoding"] == "utf-8"
        assert result_dict["encoding_confidence"] == 0.95
        assert result_dict["line_count"] == 1
        assert result_dict["structure_type"] == "plain_text"
        assert result_dict["metadata"] == {"char_count": 5}


class TestCSVMetadata:
    """Tests for CSVMetadata class."""

    def test_csv_metadata_creation(self) -> None:
        """Test creating CSV metadata."""
        metadata = CSVMetadata(
            row_count=10,
            column_count=3,
            column_names=["id", "name", "value"],
            delimiter=",",
            has_header=True,
        )

        assert metadata.row_count == 10
        assert metadata.column_count == 3
        assert metadata.column_names == ["id", "name", "value"]
        assert metadata.delimiter == ","
        assert metadata.has_header is True

    def test_csv_metadata_to_dict(self) -> None:
        """Test converting CSV metadata to dictionary."""
        metadata = CSVMetadata(
            row_count=5,
            column_count=2,
            column_names=["a", "b"],
            delimiter=";",
            has_header=True,
        )

        result = metadata.to_dict()

        assert result["row_count"] == 5
        assert result["column_count"] == 2
        assert result["column_names"] == ["a", "b"]
        assert result["delimiter"] == ";"
        assert result["has_header"] is True


class TestTextExtractorTXT:
    """Tests for TXT file extraction."""

    def test_extract_simple_txt(self, extractor: TextExtractor) -> None:
        """Test extracting text from a simple TXT file."""
        content = b"Hello, World!\nThis is a test."
        result = extractor.extract_txt(content)

        assert "Hello, World!" in result.text
        assert "This is a test." in result.text
        assert result.encoding == "ascii"
        assert result.line_count == 2
        assert result.structure_type == StructureType.PLAIN_TEXT

    def test_extract_utf8_txt(self, extractor: TextExtractor) -> None:
        """Test extracting UTF-8 encoded text."""
        content = "Hello, 世界!\nПривет мир!".encode()
        result = extractor.extract_txt(content)

        assert "Hello, 世界!" in result.text
        assert "Привет мир!" in result.text
        assert result.encoding == "utf-8"
        assert result.line_count == 2

    def test_extract_utf8_with_bom(self, extractor: TextExtractor) -> None:
        """Test extracting UTF-8 text with BOM."""
        content = b"\xef\xbb\xbfHello with BOM"
        result = extractor.extract_txt(content)

        # BOM should be handled
        assert "Hello with BOM" in result.text
        assert result.metadata.get("has_bom") is True

    def test_extract_latin1_txt(self, extractor: TextExtractor) -> None:
        """Test extracting Latin-1 encoded text."""
        # Latin-1 specific characters: é, ñ, ü
        content = "Café, España, Münich".encode("latin-1")
        result = extractor.extract_txt(content)

        assert "Café" in result.text or "Caf" in result.text
        assert result.line_count == 1

    def test_extract_multiline_txt(self, extractor: TextExtractor) -> None:
        """Test extracting multiline text with various line endings."""
        content = b"Line 1\nLine 2\nLine 3\nLine 4"
        result = extractor.extract_txt(content)

        assert result.line_count == 4
        assert "Line 1" in result.text
        assert "Line 4" in result.text

    def test_extract_windows_line_endings(self, extractor: TextExtractor) -> None:
        """Test handling Windows-style line endings."""
        content = b"Line 1\r\nLine 2\r\nLine 3"
        result = extractor.extract_txt(content)

        # Line endings should be normalized to \n
        assert "\r\n" not in result.text
        assert "\n" in result.text
        assert result.line_count == 3

    def test_extract_old_mac_line_endings(self, extractor: TextExtractor) -> None:
        """Test handling old Mac-style line endings."""
        content = b"Line 1\rLine 2\rLine 3"
        result = extractor.extract_txt(content)

        # Line endings should be normalized to \n
        assert "\r" not in result.text or result.text.count("\r") == 0
        assert result.line_count == 3

    def test_extract_empty_txt(self, extractor: TextExtractor) -> None:
        """Test extracting empty TXT file."""
        content = b""
        result = extractor.extract_txt(content)

        assert result.text == ""
        assert result.line_count == 0

    def test_extract_single_line_txt(self, extractor: TextExtractor) -> None:
        """Test extracting single line without newline."""
        content = b"Single line without newline"
        result = extractor.extract_txt(content)

        assert result.text == "Single line without newline"
        assert result.line_count == 1

    def test_extract_txt_with_special_chars(self, extractor: TextExtractor) -> None:
        """Test extracting text with special characters."""
        content = b'Tab:\tNewline:\nQuote:"'
        result = extractor.extract_txt(content)

        assert "\t" in result.text
        assert '"' in result.text

    def test_extract_preserves_whitespace(self, extractor: TextExtractor) -> None:
        """Test that whitespace is preserved."""
        content = b"  Leading spaces\nTrailing spaces  \n  Both  "
        result = extractor.extract_txt(content)

        assert "  Leading spaces" in result.text
        assert "Trailing spaces  " in result.text

    def test_encoding_confidence_included(self, extractor: TextExtractor) -> None:
        """Test that encoding confidence is included in result."""
        content = b"Simple ASCII text"
        result = extractor.extract_txt(content)

        assert 0.0 <= result.encoding_confidence <= 1.0


class TestTextExtractorCSV:
    """Tests for CSV file extraction."""

    def test_extract_simple_csv(self, extractor: TextExtractor) -> None:
        """Test extracting a simple CSV file."""
        content = b"name,age,city\nAlice,30,NYC\nBob,25,LA"
        result = extractor.extract_csv(content)

        assert result.structure_type == StructureType.TABULAR
        assert "name" in result.text
        assert "Alice" in result.text
        assert "Bob" in result.text
        assert result.line_count == 3  # header + 2 data rows

    def test_csv_metadata_extracted(self, extractor: TextExtractor) -> None:
        """Test that CSV metadata is properly extracted."""
        content = b"id,name,value\n1,foo,100\n2,bar,200\n3,baz,300"
        result = extractor.extract_csv(content)

        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["row_count"] == 3
        assert csv_meta["column_count"] == 3
        assert csv_meta["column_names"] == ["id", "name", "value"]
        assert csv_meta["delimiter"] == ","
        assert csv_meta["has_header"] is True

    def test_extract_semicolon_delimited(self, extractor: TextExtractor) -> None:
        """Test extracting semicolon-delimited CSV."""
        content = b"name;age;city\nAlice;30;NYC\nBob;25;LA"
        result = extractor.extract_csv(content)

        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["delimiter"] == ";"
        assert "Alice" in result.text
        assert "Bob" in result.text

    def test_extract_tab_delimited(self, extractor: TextExtractor) -> None:
        """Test extracting tab-delimited CSV (TSV)."""
        content = b"name\tage\tcity\nAlice\t30\tNYC"
        result = extractor.extract_csv(content)

        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["delimiter"] == "\t"
        assert "Alice" in result.text

    def test_extract_csv_with_quotes(self, extractor: TextExtractor) -> None:
        """Test extracting CSV with quoted fields."""
        content = b'name,description\nFoo,"A description with, comma"\nBar,"Another ""quoted"" value"'
        result = extractor.extract_csv(content)

        assert "Foo" in result.text
        # Check that the complex values are handled
        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["row_count"] == 2

    def test_extract_csv_with_empty_fields(self, extractor: TextExtractor) -> None:
        """Test extracting CSV with empty fields."""
        content = b"a,b,c\n1,,3\n,2,\n1,2,3"
        result = extractor.extract_csv(content)

        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["row_count"] == 3
        assert csv_meta["column_count"] == 3

    def test_extract_csv_utf8(self, extractor: TextExtractor) -> None:
        """Test extracting UTF-8 encoded CSV."""
        content = "名前,年齢\n田中,30\n山田,25".encode()
        result = extractor.extract_csv(content)

        assert result.encoding == "utf-8"
        assert "名前" in result.text
        assert "田中" in result.text

    def test_extract_empty_csv(self, extractor: TextExtractor) -> None:
        """Test extracting empty CSV file."""
        content = b""
        result = extractor.extract_csv(content)

        assert result.text == ""
        assert result.line_count == 0
        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["row_count"] == 0
        assert csv_meta["column_count"] == 0

    def test_extract_csv_single_column(self, extractor: TextExtractor) -> None:
        """Test extracting single-column CSV."""
        content = b"values\n1\n2\n3\n4\n5"
        result = extractor.extract_csv(content)

        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["column_count"] == 1
        assert csv_meta["row_count"] == 5

    def test_extract_csv_many_columns(self, extractor: TextExtractor) -> None:
        """Test extracting CSV with many columns."""
        header = ",".join([f"col{i}" for i in range(20)])
        row = ",".join([str(i) for i in range(20)])
        content = f"{header}\n{row}".encode()
        result = extractor.extract_csv(content)

        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["column_count"] == 20

    def test_csv_preserves_numeric_as_string(self, extractor: TextExtractor) -> None:
        """Test that numeric values are preserved as strings."""
        content = b"id,code\n001,00123\n002,00456"
        result = extractor.extract_csv(content)

        # Leading zeros should be preserved
        assert "001" in result.text
        assert "00123" in result.text


class TestTextExtractorFromPath:
    """Tests for extracting from file paths."""

    def test_extract_txt_from_path(self, extractor: TextExtractor) -> None:
        """Test extracting TXT from file path."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            f.write(b"Test content from file")
            temp_path = f.name

        try:
            result = extractor.extract_from_path(temp_path)
            assert "Test content from file" in result.text
            assert result.structure_type == StructureType.PLAIN_TEXT
        finally:
            Path(temp_path).unlink()

    def test_extract_csv_from_path(self, extractor: TextExtractor) -> None:
        """Test extracting CSV from file path."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
            f.write(b"a,b\n1,2")
            temp_path = f.name

        try:
            result = extractor.extract_from_path(temp_path)
            assert result.structure_type == StructureType.TABULAR
        finally:
            Path(temp_path).unlink()

    def test_extract_from_path_object(self, extractor: TextExtractor) -> None:
        """Test extracting using Path object."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            f.write(b"Path object test")
            temp_path = Path(f.name)

        try:
            result = extractor.extract_from_path(temp_path)
            assert "Path object test" in result.text
        finally:
            temp_path.unlink()

    def test_extract_nonexistent_file_raises(self, extractor: TextExtractor) -> None:
        """Test that extracting from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            extractor.extract_from_path("/nonexistent/path/file.txt")


class TestTextExtractorFromContent:
    """Tests for extracting from content bytes."""

    def test_extract_from_content_txt(self, extractor: TextExtractor) -> None:
        """Test extracting TXT from content."""
        content = b"Content extraction test"
        result = extractor.extract_from_content(content, ".txt")

        assert "Content extraction test" in result.text
        assert result.structure_type == StructureType.PLAIN_TEXT

    def test_extract_from_content_csv(self, extractor: TextExtractor) -> None:
        """Test extracting CSV from content."""
        content = b"x,y\n1,2"
        result = extractor.extract_from_content(content, ".csv")

        assert result.structure_type == StructureType.TABULAR

    def test_extract_from_content_extension_without_dot(
        self, extractor: TextExtractor
    ) -> None:
        """Test that extension without dot is handled."""
        content = b"Test"
        result = extractor.extract_from_content(content, "txt")

        assert "Test" in result.text

    def test_extract_from_content_uppercase_extension(
        self, extractor: TextExtractor
    ) -> None:
        """Test that uppercase extension is handled."""
        content = b"Test"
        result = extractor.extract_from_content(content, ".TXT")

        assert "Test" in result.text


class TestTextExtractorErrors:
    """Tests for error handling."""

    def test_unsupported_extension_raises(self, extractor: TextExtractor) -> None:
        """Test that unsupported extension raises error."""
        with pytest.raises(TextExtractionError) as exc_info:
            extractor.extract_from_content(b"test", ".pdf")

        assert "Unsupported file extension" in str(exc_info.value)

    def test_error_includes_file_path(self, extractor: TextExtractor) -> None:
        """Test that error includes file path when available."""
        with pytest.raises(TextExtractionError) as exc_info:
            extractor.extract_from_content(b"test", ".xyz", filename="test.xyz")

        assert exc_info.value.file_path == "test.xyz"


class TestTextExtractorEncodings:
    """Tests for various encoding scenarios."""

    def test_detect_utf8(self, extractor: TextExtractor) -> None:
        """Test detection of UTF-8 encoding."""
        content = "UTF-8: αβγδ".encode()
        result = extractor.extract_txt(content)

        assert result.encoding == "utf-8"
        assert "αβγδ" in result.text

    def test_detect_utf16(self, extractor: TextExtractor) -> None:
        """Test detection of UTF-16 encoding."""
        content = "UTF-16 text".encode("utf-16")
        result = extractor.extract_txt(content)

        assert "UTF-16 text" in result.text

    def test_fallback_for_unknown_encoding(self, extractor: TextExtractor) -> None:
        """Test fallback when encoding detection is uncertain."""
        # ASCII-compatible content should work
        content = b"Simple ASCII text"
        result = extractor.extract_txt(content)

        assert "Simple ASCII text" in result.text
        assert result.encoding_confidence > 0


class TestTextExtractorStaticMethods:
    """Tests for static methods."""

    def test_get_supported_extensions(self) -> None:
        """Test getting supported extensions."""
        extensions = TextExtractor.get_supported_extensions()

        assert ".txt" in extensions
        assert ".csv" in extensions
        assert len(extensions) == 2


class TestTextExtractorReadingOrder:
    """Tests for natural reading order preservation."""

    def test_txt_preserves_reading_order(self, extractor: TextExtractor) -> None:
        """Test that TXT preserves natural reading order."""
        content = b"First line\nSecond line\nThird line"
        result = extractor.extract_txt(content)

        lines = result.text.strip().split("\n")
        assert lines[0] == "First line"
        assert lines[1] == "Second line"
        assert lines[2] == "Third line"

    def test_csv_preserves_row_order(self, extractor: TextExtractor) -> None:
        """Test that CSV preserves row order."""
        content = b"order\n1\n2\n3\n4\n5"
        result = extractor.extract_csv(content)

        lines = result.text.strip().split("\n")
        # First line is header
        assert lines[0] == "order"
        # Data rows in order
        assert lines[1] == "1"
        assert lines[5] == "5"

    def test_csv_preserves_column_order(self, extractor: TextExtractor) -> None:
        """Test that CSV preserves column order."""
        content = b"first,second,third\na,b,c"
        result = extractor.extract_csv(content)

        csv_meta = result.metadata.get("csv", {})
        assert csv_meta["column_names"] == ["first", "second", "third"]
