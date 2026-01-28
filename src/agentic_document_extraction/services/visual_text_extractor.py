"""Visual text extraction service for PDFs and images.

This module provides functionality to extract text from visual documents
(PDFs and images) using OCR (Optical Character Recognition). It returns
text with bounding box coordinates for each text element, confidence scores,
and supports multi-page documents.

Uses:
- pdfplumber for PDF text extraction with bounding boxes
- PaddleOCR-VL for OCR on images
- pdf2image for converting PDF pages to images when OCR is needed
"""

import inspect
import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agentic_document_extraction.config import settings

logger = logging.getLogger(__name__)


class VisualExtractionError(Exception):
    """Raised when visual text extraction fails."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        page_number: int | None = None,
    ) -> None:
        """Initialize with message and optional context.

        Args:
            message: Error message.
            file_path: Optional path to the file that failed.
            page_number: Optional page number where extraction failed.
        """
        super().__init__(message)
        self.file_path = file_path
        self.page_number = page_number


class ExtractionMethod(str, Enum):
    """Method used for text extraction."""

    PDF_NATIVE = "pdf_native"
    """Text extracted directly from PDF (embedded text)."""

    OCR = "ocr"
    """Text extracted using OCR (Optical Character Recognition)."""

    HYBRID = "hybrid"
    """Combination of native PDF extraction and OCR."""


@dataclass
class BoundingBox:
    """Bounding box coordinates for a text element.

    Coordinates are in pixels from the top-left corner of the page/image.
    """

    x0: float
    """Left edge x-coordinate."""

    y0: float
    """Top edge y-coordinate."""

    x1: float
    """Right edge x-coordinate."""

    y1: float
    """Bottom edge y-coordinate."""

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        """Center point (x, y) of the bounding box."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with bounding box coordinates.
        """
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class TextElement:
    """A text element extracted from a visual document.

    Represents a piece of text with its location and confidence.
    """

    text: str
    """The extracted text content."""

    bbox: BoundingBox
    """Bounding box coordinates."""

    confidence: float
    """Confidence score (0.0-1.0) for the extraction."""

    page_number: int
    """Page number (1-indexed) where this element was found."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with text element information.
        """
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "page_number": self.page_number,
        }


@dataclass
class PageExtractionResult:
    """Result of text extraction from a single page."""

    page_number: int
    """Page number (1-indexed)."""

    text_elements: list[TextElement]
    """List of text elements extracted from the page."""

    full_text: str
    """Full text content of the page in reading order."""

    extraction_method: ExtractionMethod
    """Method used for extraction."""

    page_width: float
    """Width of the page in pixels/points."""

    page_height: float
    """Height of the page in pixels/points."""

    average_confidence: float
    """Average confidence score for all elements on this page."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with page extraction result.
        """
        return {
            "page_number": self.page_number,
            "text_elements": [elem.to_dict() for elem in self.text_elements],
            "full_text": self.full_text,
            "extraction_method": self.extraction_method.value,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "average_confidence": self.average_confidence,
        }


@dataclass
class VisualExtractionResult:
    """Result of visual text extraction from a document."""

    pages: list[PageExtractionResult]
    """Extraction results for each page."""

    full_text: str
    """Full text content of the entire document."""

    total_pages: int
    """Total number of pages in the document."""

    extraction_method: ExtractionMethod
    """Primary extraction method used."""

    average_confidence: float
    """Average confidence score across all pages."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the extraction."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with extraction result information.
        """
        return {
            "pages": [page.to_dict() for page in self.pages],
            "full_text": self.full_text,
            "total_pages": self.total_pages,
            "extraction_method": self.extraction_method.value,
            "average_confidence": self.average_confidence,
            "metadata": self.metadata,
        }

    def get_all_text_elements(self) -> list[TextElement]:
        """Get all text elements from all pages.

        Returns:
            List of all text elements in reading order.
        """
        elements: list[TextElement] = []
        for page in self.pages:
            elements.extend(page.text_elements)
        return elements


@dataclass(frozen=True)
class PaddleOCRConfig:
    """Configuration for PaddleOCR-VL."""

    language: str
    use_gpu: bool
    use_angle_cls: bool
    det_model_dir: str | None
    rec_model_dir: str | None
    cls_model_dir: str | None
    enable_mkldnn: bool
    cpu_threads: int


class VisualTextExtractor:
    """Extracts text from visual documents (PDFs and images).

    Supports PDF files with native text extraction and OCR fallback,
    as well as direct image processing with OCR.
    """

    # Supported image extensions
    IMAGE_EXTENSIONS = {
        ".jpeg",
        ".jpg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".webp",
    }

    # Supported PDF extension
    PDF_EXTENSION = ".pdf"

    # Default DPI for PDF to image conversion
    DEFAULT_PDF_DPI = 300

    # Minimum confidence threshold for native PDF text
    MIN_PDF_TEXT_CONFIDENCE = 0.9

    # PaddleOCR language
    DEFAULT_OCR_LANGUAGE = "en"

    def __init__(
        self,
        ocr_language: str | None = None,
        pdf_dpi: int = DEFAULT_PDF_DPI,
        ocr_config: PaddleOCRConfig | None = None,
    ) -> None:
        """Initialize the visual text extractor.

        Args:
            ocr_language: Language code for PaddleOCR (default: 'en').
            pdf_dpi: DPI for PDF to image conversion (default: 300).
            ocr_config: Optional PaddleOCR configuration override.
        """
        resolved_language = ocr_language or settings.paddleocr_language
        self.pdf_dpi = pdf_dpi
        if ocr_config is not None:
            self.ocr_config = ocr_config
            self.ocr_language = ocr_config.language
        else:
            self.ocr_language = resolved_language
            self.ocr_config = PaddleOCRConfig(
                language=resolved_language,
                use_gpu=settings.paddleocr_use_gpu,
                use_angle_cls=settings.paddleocr_use_angle_cls,
                det_model_dir=settings.paddleocr_det_model_dir,
                rec_model_dir=settings.paddleocr_rec_model_dir,
                cls_model_dir=settings.paddleocr_cls_model_dir,
                enable_mkldnn=settings.paddleocr_enable_mkldnn,
                cpu_threads=settings.paddleocr_cpu_threads,
            )
        self._ocr_engine: Any | None = None

    def extract_from_path(self, file_path: str | Path) -> VisualExtractionResult:
        """Extract text from a file path.

        Args:
            file_path: Path to the file to extract text from.

        Returns:
            VisualExtractionResult with extracted text and metadata.

        Raises:
            VisualExtractionError: If extraction fails.
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
    ) -> VisualExtractionResult:
        """Extract text from file content bytes.

        Args:
            content: File content as bytes.
            extension: File extension (e.g., '.pdf', '.png').
            filename: Optional filename for error messages.

        Returns:
            VisualExtractionResult with extracted text and metadata.

        Raises:
            VisualExtractionError: If extraction fails.
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
    ) -> VisualExtractionResult:
        """Internal extraction logic.

        Args:
            content: File content as bytes.
            extension: File extension (lowercase, with dot).
            source: Source identifier for error messages.

        Returns:
            VisualExtractionResult with extracted text and metadata.

        Raises:
            VisualExtractionError: If extraction fails.
        """
        if extension == self.PDF_EXTENSION:
            return self._extract_pdf(content, source)
        elif extension in self.IMAGE_EXTENSIONS:
            return self._extract_image(content, extension, source)
        else:
            raise VisualExtractionError(
                f"Unsupported file extension for visual extraction: {extension}",
                file_path=source,
            )

    def _extract_pdf(
        self,
        content: bytes,
        source: str | None,
    ) -> VisualExtractionResult:
        """Extract text from a PDF document.

        First attempts native text extraction using pdfplumber.
        Falls back to OCR if native extraction yields insufficient text.

        Args:
            content: PDF file content as bytes.
            source: Source identifier for error messages.

        Returns:
            VisualExtractionResult with extracted text.

        Raises:
            VisualExtractionError: If extraction fails.
        """
        import pdfplumber

        try:
            pages: list[PageExtractionResult] = []
            all_text_parts: list[str] = []

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing PDF with {total_pages} pages")

                for page_num, page in enumerate(pdf.pages, start=1):
                    page_result = self._extract_pdf_page(
                        page, page_num, content, source
                    )
                    pages.append(page_result)
                    if page_result.full_text.strip():
                        all_text_parts.append(page_result.full_text)

            # Determine primary extraction method
            methods = {p.extraction_method for p in pages}
            if len(methods) == 1:
                primary_method = methods.pop()
            elif (
                ExtractionMethod.OCR in methods
                and ExtractionMethod.PDF_NATIVE in methods
            ):
                primary_method = ExtractionMethod.HYBRID
            elif ExtractionMethod.OCR in methods:
                primary_method = ExtractionMethod.OCR
            else:
                primary_method = ExtractionMethod.PDF_NATIVE

            # Calculate average confidence
            total_confidence = sum(p.average_confidence for p in pages)
            avg_confidence = total_confidence / len(pages) if pages else 0.0

            full_text = "\n\n".join(all_text_parts)

            logger.info(
                f"Extracted PDF: {total_pages} pages, method={primary_method.value}, "
                f"avg_confidence={avg_confidence:.2f}"
            )

            return VisualExtractionResult(
                pages=pages,
                full_text=full_text,
                total_pages=total_pages,
                extraction_method=primary_method,
                average_confidence=avg_confidence,
                metadata={
                    "source": source,
                    "pdf_dpi": self.pdf_dpi,
                },
            )

        except Exception as e:
            if isinstance(e, VisualExtractionError):
                raise
            raise VisualExtractionError(
                f"Failed to extract text from PDF: {e}",
                file_path=source,
            ) from e

    def _extract_pdf_page(
        self,
        page: Any,  # pdfplumber.Page
        page_num: int,
        pdf_content: bytes,
        source: str | None,
    ) -> PageExtractionResult:
        """Extract text from a single PDF page.

        Args:
            page: pdfplumber Page object.
            page_num: Page number (1-indexed).
            pdf_content: Full PDF content (for OCR fallback).
            source: Source identifier for error messages.

        Returns:
            PageExtractionResult for the page.
        """
        page_width = float(page.width)
        page_height = float(page.height)

        # Try native text extraction first
        text_elements, native_text = self._extract_pdf_page_native(page, page_num)

        # Check if we got meaningful text
        if native_text.strip() and len(text_elements) > 0:
            avg_confidence = sum(e.confidence for e in text_elements) / len(
                text_elements
            )

            logger.debug(
                f"Page {page_num}: native extraction successful, "
                f"{len(text_elements)} elements, conf={avg_confidence:.2f}"
            )

            return PageExtractionResult(
                page_number=page_num,
                text_elements=text_elements,
                full_text=native_text,
                extraction_method=ExtractionMethod.PDF_NATIVE,
                page_width=page_width,
                page_height=page_height,
                average_confidence=avg_confidence,
            )

        # Fall back to OCR
        logger.debug(f"Page {page_num}: falling back to OCR")
        return self._extract_pdf_page_ocr(
            pdf_content, page_num, page_width, page_height, source
        )

    def _extract_pdf_page_native(
        self,
        page: Any,  # pdfplumber.Page
        page_num: int,
    ) -> tuple[list[TextElement], str]:
        """Extract text natively from a PDF page using pdfplumber.

        Args:
            page: pdfplumber Page object.
            page_num: Page number (1-indexed).

        Returns:
            Tuple of (text_elements, full_text).
        """
        text_elements: list[TextElement] = []

        # Extract words with bounding boxes
        words = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["fontname", "size"],
        )

        for word in words:
            bbox = BoundingBox(
                x0=float(word["x0"]),
                y0=float(word["top"]),
                x1=float(word["x1"]),
                y1=float(word["bottom"]),
            )
            text_elements.append(
                TextElement(
                    text=word["text"],
                    bbox=bbox,
                    confidence=self.MIN_PDF_TEXT_CONFIDENCE,
                    page_number=page_num,
                )
            )

        # Get full text preserving layout
        full_text = page.extract_text(layout=True) or ""

        return text_elements, full_text

    def _extract_pdf_page_ocr(
        self,
        pdf_content: bytes,
        page_num: int,
        page_width: float,
        page_height: float,
        source: str | None,
    ) -> PageExtractionResult:
        """Extract text from a PDF page using OCR.

        Args:
            pdf_content: Full PDF content as bytes.
            page_num: Page number (1-indexed).
            page_width: Width of the page.
            page_height: Height of the page.
            source: Source identifier for error messages.

        Returns:
            PageExtractionResult from OCR.

        Raises:
            VisualExtractionError: If OCR fails.
        """
        from pdf2image import convert_from_bytes

        try:
            # Convert specific page to image
            images = convert_from_bytes(
                pdf_content,
                dpi=self.pdf_dpi,
                first_page=page_num,
                last_page=page_num,
            )

            if not images:
                raise VisualExtractionError(
                    f"Failed to convert PDF page {page_num} to image",
                    file_path=source,
                    page_number=page_num,
                )

            image = images[0]

            # Perform OCR on the image
            text_elements, full_text, avg_confidence = self._perform_ocr(
                image, page_num
            )

            return PageExtractionResult(
                page_number=page_num,
                text_elements=text_elements,
                full_text=full_text,
                extraction_method=ExtractionMethod.OCR,
                page_width=page_width,
                page_height=page_height,
                average_confidence=avg_confidence,
            )

        except Exception as e:
            if isinstance(e, VisualExtractionError):
                raise
            raise VisualExtractionError(
                f"OCR failed for PDF page {page_num}: {e}",
                file_path=source,
                page_number=page_num,
            ) from e

    def _extract_image(
        self,
        content: bytes,
        extension: str,
        source: str | None,
    ) -> VisualExtractionResult:
        """Extract text from an image using OCR.

        Args:
            content: Image file content as bytes.
            extension: File extension.
            source: Source identifier for error messages.

        Returns:
            VisualExtractionResult with extracted text.

        Raises:
            VisualExtractionError: If extraction fails.
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(content))

            # Handle multi-page images (TIFF)
            if extension in {".tiff", ".tif"} and hasattr(image, "n_frames"):
                return self._extract_multipage_image(image, source)

            # Single image extraction
            image_width = float(image.width)
            image_height = float(image.height)

            text_elements, full_text, avg_confidence = self._perform_ocr(image, 1)

            page_result = PageExtractionResult(
                page_number=1,
                text_elements=text_elements,
                full_text=full_text,
                extraction_method=ExtractionMethod.OCR,
                page_width=image_width,
                page_height=image_height,
                average_confidence=avg_confidence,
            )

            logger.info(
                f"Extracted image: {len(text_elements)} elements, "
                f"avg_confidence={avg_confidence:.2f}"
            )

            return VisualExtractionResult(
                pages=[page_result],
                full_text=full_text,
                total_pages=1,
                extraction_method=ExtractionMethod.OCR,
                average_confidence=avg_confidence,
                metadata={
                    "source": source,
                    "image_width": image_width,
                    "image_height": image_height,
                    "image_format": image.format,
                },
            )

        except Exception as e:
            if isinstance(e, VisualExtractionError):
                raise
            raise VisualExtractionError(
                f"Failed to extract text from image: {e}",
                file_path=source,
            ) from e

    def _extract_multipage_image(
        self,
        image: Image.Image,
        source: str | None,
    ) -> VisualExtractionResult:
        """Extract text from a multi-page image (e.g., multi-page TIFF).

        Args:
            image: PIL Image object with multiple frames.
            source: Source identifier for error messages.

        Returns:
            VisualExtractionResult with extracted text from all pages.
        """
        pages: list[PageExtractionResult] = []
        all_text_parts: list[str] = []

        n_frames = getattr(image, "n_frames", 1)
        logger.info(f"Processing multi-page image with {n_frames} pages")

        for frame_num in range(n_frames):
            image.seek(frame_num)
            page_num = frame_num + 1

            # Convert to RGB if necessary
            frame_image = image.convert("RGB") if image.mode != "RGB" else image.copy()

            image_width = float(frame_image.width)
            image_height = float(frame_image.height)

            text_elements, full_text, avg_confidence = self._perform_ocr(
                frame_image, page_num
            )

            page_result = PageExtractionResult(
                page_number=page_num,
                text_elements=text_elements,
                full_text=full_text,
                extraction_method=ExtractionMethod.OCR,
                page_width=image_width,
                page_height=image_height,
                average_confidence=avg_confidence,
            )
            pages.append(page_result)
            if full_text.strip():
                all_text_parts.append(full_text)

        # Calculate overall average confidence
        total_confidence = sum(p.average_confidence for p in pages)
        overall_avg_confidence = total_confidence / len(pages) if pages else 0.0

        return VisualExtractionResult(
            pages=pages,
            full_text="\n\n".join(all_text_parts),
            total_pages=n_frames,
            extraction_method=ExtractionMethod.OCR,
            average_confidence=overall_avg_confidence,
            metadata={
                "source": source,
                "multipage": True,
            },
        )

    def _perform_ocr(
        self,
        image: Image.Image,
        page_num: int,
    ) -> tuple[list[TextElement], str, float]:
        """Perform OCR on an image using PaddleOCR-VL.

        Args:
            image: PIL Image to process.
            page_num: Page number for the text elements.

        Returns:
            Tuple of (text_elements, full_text, average_confidence).
        """
        ocr_engine = self._get_ocr_engine()

        # Convert to RGB if necessary for PaddleOCR
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            ocr_output = ocr_engine.ocr(
                np.array(image),
                cls=self.ocr_config.use_angle_cls,
            )
        except (TypeError, ValueError) as exc:
            if "cls" not in str(exc):
                raise
            ocr_output = ocr_engine.ocr(np.array(image))

        text_elements: list[TextElement] = []
        confidences: list[float] = []

        if not ocr_output:
            return [], "", 0.0

        def _is_line(item: Any) -> bool:
            return (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], (list, tuple))
                and isinstance(item[1], (list, tuple))
                and len(item[1]) == 2
                and isinstance(item[1][0], str)
            )

        def _is_ocr_result(item: Any) -> bool:
            """Check if item is a new-style OCRResult object."""
            return hasattr(item, "get") and "rec_texts" in item

        full_text_parts: list[str] = []

        # Handle new OCRResult format (PaddleOCR v3+)
        if ocr_output and _is_ocr_result(ocr_output[0]):
            page_result = ocr_output[0]
            rec_texts = page_result.get("rec_texts", [])
            rec_scores = page_result.get("rec_scores", [])
            rec_polys = page_result.get("rec_polys", [])

            for i, text in enumerate(rec_texts):
                if not text:
                    continue
                confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                points = rec_polys[i] if i < len(rec_polys) else None

                if points is not None and len(points) >= 4:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    bbox = BoundingBox(
                        x0=float(min(x_coords)),
                        y0=float(min(y_coords)),
                        x1=float(max(x_coords)),
                        y1=float(max(y_coords)),
                    )
                else:
                    bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0)

                confidences.append(float(confidence))
                full_text_parts.append(text)
                text_elements.append(
                    TextElement(
                        text=text,
                        bbox=bbox,
                        confidence=float(confidence),
                        page_number=page_num,
                    )
                )
        else:
            # Handle legacy format: list of [points, (text, confidence)]
            ocr_lines = ocr_output
            if (
                ocr_output
                and not _is_line(ocr_output[0])
                and isinstance(ocr_output[0], list)
            ):
                ocr_lines = ocr_output[0]

            for line in ocr_lines:
                if not _is_line(line):
                    continue
                points, (text, confidence) = line
                if not text:
                    continue

                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                bbox = BoundingBox(
                    x0=float(min(x_coords)),
                    y0=float(min(y_coords)),
                    x1=float(max(x_coords)),
                    y1=float(max(y_coords)),
                )

                confidences.append(float(confidence))
                full_text_parts.append(text)
                text_elements.append(
                    TextElement(
                        text=text,
                        bbox=bbox,
                        confidence=float(confidence),
                        page_number=page_num,
                    )
                )

        full_text = "\n".join(full_text_parts)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text_elements, full_text, avg_confidence

    def _get_ocr_engine(self) -> Any:
        """Initialize and cache the PaddleOCR engine."""
        if self._ocr_engine is None:
            try:
                self._ensure_langchain_docstore_stub()
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise VisualExtractionError(
                    "PaddleOCR is not installed. Install dependencies with `uv sync`."
                ) from exc

            config = self.ocr_config
            init_params = inspect.signature(PaddleOCR.__init__).parameters
            ocr_kwargs: dict[str, Any] = {}

            if "lang" in init_params:
                ocr_kwargs["lang"] = config.language
            if "use_gpu" in init_params:
                ocr_kwargs["use_gpu"] = config.use_gpu
            if "use_angle_cls" in init_params:
                ocr_kwargs["use_angle_cls"] = config.use_angle_cls
            if "use_textline_orientation" in init_params:
                ocr_kwargs["use_textline_orientation"] = config.use_angle_cls
            if "enable_mkldnn" in init_params:
                ocr_kwargs["enable_mkldnn"] = config.enable_mkldnn
            if "cpu_threads" in init_params:
                ocr_kwargs["cpu_threads"] = config.cpu_threads
            if "show_log" in init_params:
                ocr_kwargs["show_log"] = False
            if "text_detection_model_dir" in init_params and config.det_model_dir:
                ocr_kwargs["text_detection_model_dir"] = config.det_model_dir
            if "text_recognition_model_dir" in init_params and config.rec_model_dir:
                ocr_kwargs["text_recognition_model_dir"] = config.rec_model_dir
            if "textline_orientation_model_dir" in init_params and config.cls_model_dir:
                ocr_kwargs["textline_orientation_model_dir"] = config.cls_model_dir

            self._ocr_engine = PaddleOCR(**ocr_kwargs)
        return self._ocr_engine

    @staticmethod
    def _ensure_langchain_docstore_stub() -> None:
        """Provide minimal langchain stubs for PaddleOCR imports."""
        import importlib
        import sys
        import types

        import langchain

        langchain_module: Any = langchain

        def _ensure_module(name: str) -> bool:
            try:
                importlib.import_module(name)
                return True
            except ModuleNotFoundError:
                return False

        if not _ensure_module("langchain.docstore.document"):
            docstore_module: Any = types.ModuleType("langchain.docstore")
            document_module: Any = types.ModuleType("langchain.docstore.document")

            @dataclass
            class Document:
                page_content: str
                metadata: dict[str, Any] | None = None

                def __post_init__(self) -> None:
                    if self.metadata is None:
                        self.metadata = {}

            document_module.Document = Document
            docstore_module.document = document_module
            sys.modules.setdefault("langchain.docstore", docstore_module)
            sys.modules.setdefault("langchain.docstore.document", document_module)
            langchain_module.docstore = docstore_module

        if not _ensure_module("langchain.text_splitter"):
            text_splitter_module: Any = types.ModuleType("langchain.text_splitter")

            class RecursiveCharacterTextSplitter:
                def __init__(self, *_: Any, **__: Any) -> None:
                    pass

                def split_text(self, text: str) -> list[str]:
                    return [text]

            text_splitter_module.RecursiveCharacterTextSplitter = (
                RecursiveCharacterTextSplitter
            )
            sys.modules.setdefault("langchain.text_splitter", text_splitter_module)
            langchain_module.text_splitter = text_splitter_module

    def extract_pdf(self, content: bytes) -> VisualExtractionResult:
        """Public method to extract text from PDF content.

        Args:
            content: PDF file content as bytes.

        Returns:
            VisualExtractionResult with extracted text.
        """
        return self._extract_pdf(content, None)

    def extract_image(
        self,
        content: bytes,
        extension: str = ".png",
    ) -> VisualExtractionResult:
        """Public method to extract text from image content.

        Args:
            content: Image file content as bytes.
            extension: Image file extension (default: '.png').

        Returns:
            VisualExtractionResult with extracted text.
        """
        ext = extension.lower() if not extension.startswith(".") else extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return self._extract_image(content, ext, None)

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions (with dots).
        """
        extensions = [".pdf"]
        extensions.extend(sorted(VisualTextExtractor.IMAGE_EXTENSIONS))
        return extensions

    @staticmethod
    def is_pdf(extension: str) -> bool:
        """Check if extension is PDF.

        Args:
            extension: File extension to check.

        Returns:
            True if PDF, False otherwise.
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return ext == ".pdf"

    @staticmethod
    def is_image(extension: str) -> bool:
        """Check if extension is a supported image format.

        Args:
            extension: File extension to check.

        Returns:
            True if supported image, False otherwise.
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return ext in VisualTextExtractor.IMAGE_EXTENSIONS
