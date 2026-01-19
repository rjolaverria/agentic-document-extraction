"""Text extraction service for text-based documents.

This module provides functionality to extract raw text from text-based documents
(TXT and CSV files) using LangChain document loaders. It handles encoding detection
and preserves document structure while providing a unified interface for text
extraction.

Uses LangChain's TextLoader and CSVLoader for loading documents, following
the project's technical constraint to use LangChain Document and document loaders
for reading data from different sources.
"""

import csv
import io
import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import chardet
import pandas as pd
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextExtractionError(Exception):
    """Raised when text extraction fails."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        """Initialize with message and optional file path.

        Args:
            message: Error message.
            file_path: Optional path to the file that failed.
        """
        super().__init__(message)
        self.file_path = file_path


class StructureType(str, Enum):
    """Type of structure detected in the document."""

    PLAIN_TEXT = "plain_text"
    TABULAR = "tabular"


@dataclass
class TextExtractionResult:
    """Result of text extraction from a document."""

    text: str
    """The extracted text content."""

    encoding: str
    """Detected or used encoding (e.g., 'utf-8', 'latin-1')."""

    encoding_confidence: float
    """Confidence score for encoding detection (0.0-1.0)."""

    line_count: int
    """Number of lines in the extracted text."""

    structure_type: StructureType
    """Type of structure detected (plain_text or tabular)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the extraction."""

    documents: list[Document] = field(default_factory=list)
    """LangChain Document objects from the extraction."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with extraction result information.
        """
        return {
            "text": self.text,
            "encoding": self.encoding,
            "encoding_confidence": self.encoding_confidence,
            "line_count": self.line_count,
            "structure_type": self.structure_type.value,
            "metadata": self.metadata,
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self.documents
            ],
        }


@dataclass
class CSVMetadata:
    """Metadata specific to CSV extraction."""

    row_count: int
    """Number of data rows (excluding header)."""

    column_count: int
    """Number of columns."""

    column_names: list[str]
    """List of column names."""

    delimiter: str
    """Detected delimiter character."""

    has_header: bool
    """Whether the CSV has a header row."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with CSV metadata.
        """
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
        }


class TextExtractor:
    """Extracts text from text-based documents.

    Supports TXT and CSV files with automatic encoding detection
    and structure preservation.
    """

    # Common encodings to try if chardet fails
    FALLBACK_ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "ascii"]

    # Minimum confidence threshold for encoding detection
    MIN_ENCODING_CONFIDENCE = 0.5

    def __init__(self) -> None:
        """Initialize the text extractor."""
        pass

    def extract_from_path(self, file_path: str | Path) -> TextExtractionResult:
        """Extract text from a file path.

        Args:
            file_path: Path to the file to extract text from.

        Returns:
            TextExtractionResult with extracted text and metadata.

        Raises:
            TextExtractionError: If extraction fails.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_bytes()
        extension = path.suffix.lower()

        return self._extract(content, extension, str(file_path))

    def extract_from_content(
        self,
        content: bytes,
        extension: str,
        filename: str | None = None,
    ) -> TextExtractionResult:
        """Extract text from file content bytes.

        Args:
            content: File content as bytes.
            extension: File extension (e.g., '.txt', '.csv').
            filename: Optional filename for error messages.

        Returns:
            TextExtractionResult with extracted text and metadata.

        Raises:
            TextExtractionError: If extraction fails.
        """
        ext = extension.lower() if not extension.startswith(".") else extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        return self._extract(content, ext, filename)

    def _extract(
        self,
        content: bytes,
        extension: str,
        source: str | None,
    ) -> TextExtractionResult:
        """Internal extraction logic.

        Args:
            content: File content as bytes.
            extension: File extension (lowercase, with dot).
            source: Source identifier for error messages.

        Returns:
            TextExtractionResult with extracted text and metadata.

        Raises:
            TextExtractionError: If extraction fails.
        """
        if extension == ".csv":
            return self._extract_csv(content, source)
        elif extension == ".txt":
            return self._extract_txt(content, source)
        else:
            raise TextExtractionError(
                f"Unsupported file extension for text extraction: {extension}",
                file_path=source,
            )

    def _detect_encoding(self, content: bytes) -> tuple[str, float]:
        """Detect the encoding of byte content.

        Args:
            content: File content as bytes.

        Returns:
            Tuple of (encoding_name, confidence_score).
        """
        if not content:
            return "utf-8", 1.0

        # Use chardet for detection
        result = chardet.detect(content)
        encoding = result.get("encoding")
        confidence = result.get("confidence", 0.0) or 0.0

        if encoding and confidence >= self.MIN_ENCODING_CONFIDENCE:
            # Normalize encoding name
            encoding = self._normalize_encoding(encoding)
            logger.debug(
                f"Detected encoding: {encoding} (confidence: {confidence:.2f})"
            )
            return encoding, confidence

        # Try fallback encodings
        for fallback in self.FALLBACK_ENCODINGS:
            try:
                content.decode(fallback)
                logger.debug(f"Using fallback encoding: {fallback}")
                return fallback, 0.5  # Moderate confidence for fallback
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort: use latin-1 which accepts any byte sequence
        logger.warning("Could not detect encoding, falling back to latin-1")
        return "latin-1", 0.3

    def _normalize_encoding(self, encoding: str) -> str:
        """Normalize encoding name to Python-compatible form.

        Args:
            encoding: Encoding name from chardet.

        Returns:
            Normalized encoding name.
        """
        encoding = encoding.lower().replace("-", "_").replace(" ", "_")

        # Common normalizations
        normalizations = {
            "utf_8": "utf-8",
            "utf_16": "utf-16",
            "utf_32": "utf-32",
            "ascii": "ascii",
            "iso_8859_1": "iso-8859-1",
            "iso8859_1": "iso-8859-1",
            "latin_1": "latin-1",
            "latin1": "latin-1",
            "cp1252": "cp1252",
            "windows_1252": "cp1252",
        }

        return normalizations.get(encoding, encoding.replace("_", "-"))

    def _decode_content(self, content: bytes, encoding: str, source: str | None) -> str:
        """Decode byte content to string.

        Args:
            content: File content as bytes.
            encoding: Encoding to use.
            source: Source identifier for error messages.

        Returns:
            Decoded string content.

        Raises:
            TextExtractionError: If decoding fails.
        """
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError) as e:
            # Try fallback encodings
            for fallback in self.FALLBACK_ENCODINGS:
                try:
                    return content.decode(fallback)
                except (UnicodeDecodeError, LookupError):
                    continue

            raise TextExtractionError(
                f"Failed to decode content with encoding {encoding}: {e}",
                file_path=source,
            ) from e

    def _extract_txt(self, content: bytes, source: str | None) -> TextExtractionResult:
        """Extract text from a TXT file using LangChain TextLoader.

        Args:
            content: File content as bytes.
            source: Source identifier for error messages.

        Returns:
            TextExtractionResult with extracted text and LangChain documents.

        Raises:
            TextExtractionError: If extraction fails.
        """
        encoding, confidence = self._detect_encoding(content)
        text = self._decode_content(content, encoding, source)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Count lines
        lines = text.split("\n")
        line_count = len(lines)

        # Remove trailing empty line if present (common in text files)
        if lines and lines[-1] == "":
            line_count -= 1

        # Use LangChain TextLoader to create Document objects
        documents = self._load_text_with_langchain(content, encoding, source)

        logger.info(
            f"Extracted TXT: {line_count} lines, encoding={encoding}, "
            f"confidence={confidence:.2f}, documents={len(documents)}"
        )

        return TextExtractionResult(
            text=text,
            encoding=encoding,
            encoding_confidence=confidence,
            line_count=line_count,
            structure_type=StructureType.PLAIN_TEXT,
            metadata={
                "char_count": len(text),
                "has_bom": content.startswith(b"\xef\xbb\xbf"),  # UTF-8 BOM
            },
            documents=documents,
        )

    def _load_text_with_langchain(
        self,
        content: bytes,
        encoding: str,
        source: str | None,
    ) -> list[Document]:
        """Load text content using LangChain TextLoader.

        Args:
            content: File content as bytes.
            encoding: Detected encoding.
            source: Source identifier for metadata.

        Returns:
            List of LangChain Document objects.
        """
        # Write content to a temporary file for TextLoader
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="wb"
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Use LangChain TextLoader with detected encoding
            loader = TextLoader(temp_path, encoding=encoding)
            documents = loader.load()

            # Update document metadata with source information
            for doc in documents:
                if source:
                    doc.metadata["original_source"] = source
                doc.metadata["encoding"] = encoding

            return documents
        except Exception as e:
            logger.warning(
                f"LangChain TextLoader failed: {e}, creating document manually"
            )
            # Fallback: create Document manually if loader fails
            text = self._decode_content(content, encoding, source)
            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": source or "unknown",
                        "encoding": encoding,
                    },
                )
            ]
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

    def _extract_csv(self, content: bytes, source: str | None) -> TextExtractionResult:
        """Extract text from a CSV file using LangChain CSVLoader.

        Args:
            content: File content as bytes.
            source: Source identifier for error messages.

        Returns:
            TextExtractionResult with extracted text in tabular format and LangChain documents.

        Raises:
            TextExtractionError: If extraction fails.
        """
        encoding, confidence = self._detect_encoding(content)
        text_content = self._decode_content(content, encoding, source)

        # Detect delimiter
        delimiter = self._detect_csv_delimiter(text_content)

        # Parse CSV using pandas for robust handling
        try:
            # Use StringIO to parse from string
            df = pd.read_csv(
                io.StringIO(text_content),
                delimiter=delimiter,
                dtype=str,  # Keep all values as strings
                keep_default_na=False,  # Don't convert empty strings to NaN
            )
        except pd.errors.EmptyDataError:
            # Handle empty CSV
            return TextExtractionResult(
                text="",
                encoding=encoding,
                encoding_confidence=confidence,
                line_count=0,
                structure_type=StructureType.TABULAR,
                metadata={
                    "csv": CSVMetadata(
                        row_count=0,
                        column_count=0,
                        column_names=[],
                        delimiter=delimiter,
                        has_header=False,
                    ).to_dict(),
                },
                documents=[],
            )
        except pd.errors.ParserError as e:
            raise TextExtractionError(
                f"Failed to parse CSV: {e}",
                file_path=source,
            ) from e

        # Get column names
        column_names = [str(col) for col in df.columns.tolist()]

        # Format as readable tabular text
        formatted_text = self._format_dataframe_as_text(df, delimiter)

        # Count lines (header + data rows)
        line_count = len(df) + 1  # +1 for header

        csv_metadata = CSVMetadata(
            row_count=len(df),
            column_count=len(column_names),
            column_names=column_names,
            delimiter=delimiter,
            has_header=True,  # pandas assumes header by default
        )

        # Use LangChain CSVLoader to create Document objects
        documents = self._load_csv_with_langchain(
            content, encoding, delimiter, source, df
        )

        logger.info(
            f"Extracted CSV: {csv_metadata.row_count} rows, "
            f"{csv_metadata.column_count} columns, delimiter='{delimiter}', "
            f"documents={len(documents)}"
        )

        return TextExtractionResult(
            text=formatted_text,
            encoding=encoding,
            encoding_confidence=confidence,
            line_count=line_count,
            structure_type=StructureType.TABULAR,
            metadata={
                "csv": csv_metadata.to_dict(),
            },
            documents=documents,
        )

    def _load_csv_with_langchain(
        self,
        content: bytes,
        encoding: str,
        delimiter: str,
        source: str | None,
        df: pd.DataFrame,
    ) -> list[Document]:
        """Load CSV content using LangChain CSVLoader.

        Args:
            content: File content as bytes.
            encoding: Detected encoding.
            delimiter: Detected CSV delimiter.
            source: Source identifier for metadata.
            df: Parsed DataFrame for fallback.

        Returns:
            List of LangChain Document objects (one per row).
        """
        # Write content to a temporary file for CSVLoader
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="wb"
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Use LangChain CSVLoader with detected delimiter
            loader = CSVLoader(
                file_path=temp_path,
                csv_args={"delimiter": delimiter},
                encoding=encoding,
            )
            documents = loader.load()

            # Update document metadata with source information
            for doc in documents:
                if source:
                    doc.metadata["original_source"] = source
                doc.metadata["encoding"] = encoding
                doc.metadata["delimiter"] = delimiter

            return documents
        except Exception as e:
            logger.warning(
                f"LangChain CSVLoader failed: {e}, creating documents manually"
            )
            # Fallback: create Documents manually from DataFrame
            documents = []
            for idx, row in df.iterrows():
                row_content = "\n".join(f"{col}: {val}" for col, val in row.items())
                documents.append(
                    Document(
                        page_content=row_content,
                        metadata={
                            "source": source or "unknown",
                            "row": int(idx) if isinstance(idx, int) else 0,
                            "encoding": encoding,
                            "delimiter": delimiter,
                        },
                    )
                )
            return documents
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

    def _detect_csv_delimiter(self, content: str) -> str:
        """Detect the delimiter used in a CSV file.

        Args:
            content: CSV content as string.

        Returns:
            Detected delimiter character.
        """
        # Use csv.Sniffer for detection
        try:
            # Take a sample for sniffing
            sample = content[:8192]  # First 8KB
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            # Default to comma if detection fails
            logger.debug("CSV delimiter detection failed, defaulting to comma")
            return ","

    def _format_dataframe_as_text(
        self,
        df: pd.DataFrame,
        delimiter: str,  # noqa: ARG002
    ) -> str:
        """Format a DataFrame as readable text preserving structure.

        Args:
            df: DataFrame to format.
            delimiter: Original delimiter for reference (kept for potential future use).

        Returns:
            Formatted text representation.
        """
        # Use tab-separated format for readability while preserving structure
        lines: list[str] = []

        # Header row
        header = "\t".join(str(col) for col in df.columns)
        lines.append(header)

        # Data rows
        for _, row in df.iterrows():
            row_text = "\t".join(str(val) for val in row.values)
            lines.append(row_text)

        return "\n".join(lines)

    def extract_txt(self, content: bytes) -> TextExtractionResult:
        """Public method to extract text from TXT content.

        Args:
            content: TXT file content as bytes.

        Returns:
            TextExtractionResult with extracted text.
        """
        return self._extract_txt(content, None)

    def extract_csv(self, content: bytes) -> TextExtractionResult:
        """Public method to extract text from CSV content.

        Args:
            content: CSV file content as bytes.

        Returns:
            TextExtractionResult with extracted text in tabular format.
        """
        return self._extract_csv(content, None)

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions (with dots).
        """
        return [".txt", ".csv"]
