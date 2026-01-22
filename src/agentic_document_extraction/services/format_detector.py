"""Document format detection and classification service.

This module provides functionality to detect document formats from file extensions
and magic bytes (file content signatures), and classify them into processing categories.
"""

from pathlib import Path

import magic

from agentic_document_extraction.models import (
    FormatFamily,
    FormatInfo,
    ProcessingCategory,
)
from agentic_document_extraction.utils.exceptions import UnsupportedFormatError
from agentic_document_extraction.utils.logging import get_logger

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "FormatDetector",
    "UnsupportedFormatError",
    "EXTENSION_TO_MIME",
    "MIME_TO_EXTENSION",
    "SUPPORTED_MIME_TYPES",
]


# Mapping of file extensions to MIME types
EXTENSION_TO_MIME: dict[str, str] = {
    # Plain text
    ".txt": "text/plain",
    ".csv": "text/csv",
    # PDF
    ".pdf": "application/pdf",
    # Word documents
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".odt": "application/vnd.oasis.opendocument.text",
    # Presentations
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    # Spreadsheets
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # Images
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".psd": "image/vnd.adobe.photoshop",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Reverse mapping for MIME to extension (using canonical extensions)
MIME_TO_EXTENSION: dict[str, str] = {
    "text/plain": ".txt",
    "text/csv": ".csv",
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/vnd.adobe.photoshop": ".psd",
    "image/tiff": ".tiff",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

# Format family classification by MIME type
MIME_TO_FORMAT_FAMILY: dict[str, FormatFamily] = {
    # Plain text
    "text/plain": FormatFamily.PLAIN_TEXT,
    "text/csv": FormatFamily.SPREADSHEET,
    # PDF
    "application/pdf": FormatFamily.PDF,
    # Word documents
    "application/msword": FormatFamily.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        FormatFamily.DOCUMENT
    ),
    "application/vnd.oasis.opendocument.text": FormatFamily.DOCUMENT,
    # Presentations
    "application/vnd.ms-powerpoint": FormatFamily.PRESENTATION,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        FormatFamily.PRESENTATION
    ),
    "application/vnd.oasis.opendocument.presentation": FormatFamily.PRESENTATION,
    # Spreadsheets
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (
        FormatFamily.SPREADSHEET
    ),
    # Images
    "image/jpeg": FormatFamily.IMAGE,
    "image/png": FormatFamily.IMAGE,
    "image/bmp": FormatFamily.IMAGE,
    "image/vnd.adobe.photoshop": FormatFamily.IMAGE,
    "image/tiff": FormatFamily.IMAGE,
    "image/gif": FormatFamily.IMAGE,
    "image/webp": FormatFamily.IMAGE,
}

# Processing category by MIME type
# Text-based: txt, csv - can be processed directly
# Visual: everything else - requires visual processing pipeline
MIME_TO_PROCESSING_CATEGORY: dict[str, ProcessingCategory] = {
    # Text-based documents
    "text/plain": ProcessingCategory.TEXT_BASED,
    "text/csv": ProcessingCategory.TEXT_BASED,
    # Visual documents
    "application/pdf": ProcessingCategory.VISUAL,
    "application/msword": ProcessingCategory.VISUAL,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        ProcessingCategory.VISUAL
    ),
    "application/vnd.oasis.opendocument.text": ProcessingCategory.VISUAL,
    "application/vnd.ms-powerpoint": ProcessingCategory.VISUAL,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        ProcessingCategory.VISUAL
    ),
    "application/vnd.oasis.opendocument.presentation": ProcessingCategory.VISUAL,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (
        ProcessingCategory.VISUAL
    ),
    "image/jpeg": ProcessingCategory.VISUAL,
    "image/png": ProcessingCategory.VISUAL,
    "image/bmp": ProcessingCategory.VISUAL,
    "image/vnd.adobe.photoshop": ProcessingCategory.VISUAL,
    "image/tiff": ProcessingCategory.VISUAL,
    "image/gif": ProcessingCategory.VISUAL,
    "image/webp": ProcessingCategory.VISUAL,
}

# Set of supported MIME types
SUPPORTED_MIME_TYPES: set[str] = set(MIME_TO_FORMAT_FAMILY.keys())


class FormatDetector:
    """Detects and classifies document formats.

    Uses both file extension and magic bytes (file content signatures) to
    reliably detect document formats. Falls back to extension-based detection
    when content analysis is not possible or yields ambiguous results.
    """

    def __init__(self) -> None:
        """Initialize the format detector with a magic instance."""
        self._magic = magic.Magic(mime=True)

    def detect_from_path(self, file_path: str | Path) -> FormatInfo:
        """Detect format from a file path.

        Args:
            file_path: Path to the file to analyze.

        Returns:
            FormatInfo with detected format information.

        Raises:
            UnsupportedFormatError: If the format is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        original_extension = path.suffix.lower() if path.suffix else None
        content = path.read_bytes()

        return self._detect(
            content=content,
            filename=path.name,
            original_extension=original_extension,
        )

    def detect_from_content(
        self,
        content: bytes,
        filename: str | None = None,
    ) -> FormatInfo:
        """Detect format from file content bytes.

        Args:
            content: File content as bytes.
            filename: Optional filename for extension-based fallback.

        Returns:
            FormatInfo with detected format information.

        Raises:
            UnsupportedFormatError: If the format is not supported.
        """
        original_extension = None
        if filename:
            ext = Path(filename).suffix.lower()
            original_extension = ext if ext else None

        return self._detect(
            content=content,
            filename=filename,
            original_extension=original_extension,
        )

    def _detect(
        self,
        content: bytes,
        filename: str | None,  # noqa: ARG002
        original_extension: str | None,
    ) -> FormatInfo:
        """Internal detection logic.

        Priority:
        1. Try to detect from magic bytes (content analysis)
        2. Fall back to extension if magic detection fails or is ambiguous
        3. Handle edge cases (missing extensions, incorrect extensions)

        Args:
            content: File content as bytes.
            filename: Optional filename.
            original_extension: Original file extension (lowercase, with dot).

        Returns:
            FormatInfo with detected format information.

        Raises:
            UnsupportedFormatError: If the format is not supported.
        """
        # Try magic-based detection first
        detected_mime = self._detect_mime_from_content(content)
        mime_from_extension = self._get_mime_from_extension(original_extension)

        # Determine the final MIME type and whether we detected from content
        detected_from_content = False
        final_mime: str | None = None
        original_ext_differs: str | None = None

        if detected_mime and self._is_supported_mime(detected_mime):
            # Magic detection succeeded with a supported type
            final_mime = detected_mime
            detected_from_content = True

            # Check if extension differs from detected type
            if mime_from_extension and mime_from_extension != detected_mime:
                original_ext_differs = original_extension
                logger.warning(
                    "File extension does not match detected MIME type",
                    extension=original_extension,
                    detected_mime=detected_mime,
                )
        elif mime_from_extension and self._is_supported_mime(mime_from_extension):
            # Fall back to extension-based detection
            final_mime = mime_from_extension
            detected_from_content = False
        elif detected_mime:
            # Magic found something but it's not supported
            raise UnsupportedFormatError(
                f"Unsupported document format: {detected_mime}",
                detected_mime=detected_mime,
            )
        else:
            # Neither magic nor extension yielded a result
            raise UnsupportedFormatError(
                "Unable to detect document format. "
                "Please provide a file with a supported extension or format.",
                detected_mime=None,
            )

        # Build the FormatInfo
        extension = MIME_TO_EXTENSION.get(final_mime, original_extension or "")
        format_family = MIME_TO_FORMAT_FAMILY.get(final_mime, FormatFamily.UNKNOWN)
        processing_category = MIME_TO_PROCESSING_CATEGORY.get(
            final_mime, ProcessingCategory.VISUAL
        )

        return FormatInfo(
            mime_type=final_mime,
            extension=extension,
            format_family=format_family,
            processing_category=processing_category,
            detected_from_content=detected_from_content,
            original_extension=original_ext_differs,
        )

    def _detect_mime_from_content(self, content: bytes) -> str | None:
        """Detect MIME type from file content using magic bytes.

        Args:
            content: File content as bytes.

        Returns:
            Detected MIME type or None if detection fails.
        """
        if not content:
            return None

        try:
            detected = self._magic.from_buffer(content)
            return self._normalize_mime_type(detected)
        except Exception as e:
            logger.warning(
                "Magic detection failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _normalize_mime_type(self, mime_type: str) -> str:
        """Normalize MIME type to canonical form.

        Handles variations and aliases in MIME types.

        Args:
            mime_type: Raw MIME type string.

        Returns:
            Normalized MIME type.
        """
        # Handle common variations
        normalizations: dict[str, str] = {
            # Text variations
            "text/x-csv": "text/csv",
            "application/csv": "text/csv",
            # Image variations
            "image/x-ms-bmp": "image/bmp",
            "image/x-bmp": "image/bmp",
            # TIFF variations
            "image/x-tiff": "image/tiff",
        }

        # MIME types that should pass through unchanged
        # (e.g., ZIP-based formats that need extension-based detection)
        passthrough_types = {"application/zip", "application/x-zip-compressed"}

        if mime_type in passthrough_types:
            return mime_type

        return normalizations.get(mime_type, mime_type)

    def _get_mime_from_extension(self, extension: str | None) -> str | None:
        """Get MIME type from file extension.

        Args:
            extension: File extension (with dot, lowercase).

        Returns:
            MIME type or None if extension is not recognized.
        """
        if not extension:
            return None
        return EXTENSION_TO_MIME.get(extension)

    def _is_supported_mime(self, mime_type: str) -> bool:
        """Check if a MIME type is supported.

        Args:
            mime_type: MIME type to check.

        Returns:
            True if supported, False otherwise.
        """
        return mime_type in SUPPORTED_MIME_TYPES

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions (with dots).
        """
        return sorted(EXTENSION_TO_MIME.keys())

    @staticmethod
    def get_supported_mime_types() -> list[str]:
        """Get list of supported MIME types.

        Returns:
            List of supported MIME types.
        """
        return sorted(SUPPORTED_MIME_TYPES)

    @staticmethod
    def is_text_based(format_info: FormatInfo) -> bool:
        """Check if a format should be processed as text-based.

        Args:
            format_info: FormatInfo to check.

        Returns:
            True if text-based, False if visual.
        """
        return format_info.processing_category == ProcessingCategory.TEXT_BASED

    @staticmethod
    def is_visual(format_info: FormatInfo) -> bool:
        """Check if a format should be processed with the visual pipeline.

        Args:
            format_info: FormatInfo to check.

        Returns:
            True if visual, False if text-based.
        """
        return format_info.processing_category == ProcessingCategory.VISUAL
