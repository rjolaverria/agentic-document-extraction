"""Tests for the extraction processor service.

This module tests the extraction processor's ability to correctly route
different document types to the appropriate extraction pipelines.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_document_extraction.models import (
    FormatFamily,
    FormatInfo,
    ProcessingCategory,
)
from agentic_document_extraction.services.extraction_processor import (
    process_extraction_job,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_schema() -> dict:
    """Create a simple schema for testing."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Test Schema",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number"},
        },
        "required": ["name"],
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_png_file(temp_dir: Path) -> Path:
    """Create a sample PNG file for testing."""
    img = Image.new("RGB", (100, 100), color="white")
    png_path = temp_dir / "test_image.png"
    img.save(png_path, format="PNG")
    return png_path


@pytest.fixture
def sample_txt_file(temp_dir: Path) -> Path:
    """Create a sample TXT file for testing."""
    txt_path = temp_dir / "test.txt"
    txt_path.write_text("Hello World\nThis is a test document.")
    return txt_path


@pytest.fixture
def sample_schema_file(temp_dir: Path, sample_schema: dict) -> Path:
    """Create a sample schema file for testing."""
    schema_path = temp_dir / "schema.json"
    schema_path.write_text(json.dumps(sample_schema))
    return schema_path


def _create_mock_loop_result():
    """Create a mock loop result usable by both AgenticLoop and ExtractionAgent."""
    mock_loop_result = MagicMock()
    mock_loop_result.iterations_completed = 1
    mock_loop_result.converged = True
    mock_loop_result.total_tokens = 100
    mock_final_result = MagicMock()
    mock_final_result.extracted_data = {"name": "test"}
    mock_final_result.model_used = "gpt-4"
    mock_loop_result.final_result = mock_final_result
    mock_final_verification = MagicMock()
    mock_final_verification.metrics.overall_confidence = 0.9
    mock_final_verification.to_dict.return_value = {"passed": True}
    mock_loop_result.final_verification = mock_final_verification
    return mock_loop_result


def create_mock_agentic_loop():
    """Create a mock agentic loop with all required attributes."""
    mock_agentic_loop = MagicMock()
    mock_agentic_loop.run.return_value = _create_mock_loop_result()
    return mock_agentic_loop


def create_mock_extraction_agent():
    """Create a mock ExtractionAgent with all required attributes."""
    mock_agent = MagicMock()
    mock_agent.extract.return_value = _create_mock_loop_result()
    return mock_agent


def create_mock_markdown_generator():
    """Create a mock markdown generator."""
    mock_markdown_gen = MagicMock()
    mock_markdown_output = MagicMock()
    mock_markdown_output.markdown = "# Test\nExtracted content"
    mock_markdown_gen.generate.return_value = mock_markdown_output
    return mock_markdown_gen


def create_mock_visual_extractor():
    """Create a mock visual text extractor.

    Creates a mock that works with the new unified extract_with_layout() method.
    """
    mock_visual_extractor = MagicMock()

    # Mock the combined extraction result (new pipeline)
    mock_extraction_with_layout = MagicMock()
    mock_extraction_with_layout.full_text = "Extracted text from image"
    mock_extraction_with_layout.total_pages = 1
    mock_extraction_with_layout.average_confidence = 0.85
    mock_extraction_with_layout.get_all_regions.return_value = []

    # Mock the OCR result inside the combined result
    mock_ocr_result = MagicMock()
    mock_ocr_result.extraction_method.value = "ocr"
    mock_ocr_result.full_text = "Extracted text from image"
    mock_ocr_result.total_pages = 1
    mock_ocr_result.average_confidence = 0.85
    mock_extraction_with_layout.ocr_result = mock_ocr_result

    # Mock the layout result
    mock_layout_result = MagicMock()
    mock_layout_result.pages = []
    mock_extraction_with_layout.layout_result = mock_layout_result

    mock_visual_extractor.extract_with_layout.return_value = mock_extraction_with_layout

    # Also mock extract_from_path for backward compatibility tests
    mock_visual_result = MagicMock()
    mock_visual_result.full_text = "Extracted text from image"
    mock_visual_result.extraction_method.value = "ocr"
    mock_visual_result.total_pages = 1
    mock_visual_result.average_confidence = 0.85
    mock_visual_extractor.extract_from_path.return_value = mock_visual_result

    return mock_visual_extractor


def create_mock_text_extractor():
    """Create a mock text extractor."""
    mock_text_extractor = MagicMock()
    mock_text_result = MagicMock()
    mock_text_result.text = "Extracted text content"
    mock_text_extractor.extract_from_path.return_value = mock_text_result
    return mock_text_extractor


class TestExtractionProcessorRouting:
    """Tests for document routing in extraction processor."""

    @patch(
        "agentic_document_extraction.services.extraction_processor.MarkdownGenerator"
    )
    @patch("agentic_document_extraction.services.extraction_processor.ExtractionAgent")
    @patch(
        "agentic_document_extraction.services.extraction_processor.VisualTextExtractor"
    )
    async def test_png_file_routes_to_visual_extractor(
        self,
        mock_visual_extractor_class: MagicMock,
        mock_extraction_agent_class: MagicMock,
        mock_markdown_gen_class: MagicMock,
        sample_png_file: Path,
        sample_schema_file: Path,
    ) -> None:
        """Test that PNG files are routed to the visual text extractor."""
        # Set up mocks
        mock_visual_extractor = create_mock_visual_extractor()
        mock_visual_extractor_class.return_value = mock_visual_extractor
        mock_extraction_agent_class.return_value = create_mock_extraction_agent()
        mock_markdown_gen_class.return_value = create_mock_markdown_generator()

        # Process the job
        result = await process_extraction_job(
            job_id="test-png-routing",
            filename="test_image.png",
            file_path=str(sample_png_file),
            schema_path=str(sample_schema_file),
            progress=None,
        )

        # Verify visual extractor's extract_with_layout was called (new unified pipeline)
        mock_visual_extractor.extract_with_layout.assert_called_once()
        # JsonGenerator adds None for optional fields, so check required field
        assert result["extracted_data"]["name"] == "test"

    @patch(
        "agentic_document_extraction.services.extraction_processor.MarkdownGenerator"
    )
    @patch("agentic_document_extraction.services.extraction_processor.ExtractionAgent")
    @patch("agentic_document_extraction.services.extraction_processor.TextExtractor")
    async def test_txt_file_routes_to_text_extractor(
        self,
        mock_text_extractor_class: MagicMock,
        mock_extraction_agent_class: MagicMock,
        mock_markdown_gen_class: MagicMock,
        sample_txt_file: Path,
        sample_schema_file: Path,
    ) -> None:
        """Test that TXT files are routed to the text extractor."""
        # Set up mocks
        mock_text_extractor = create_mock_text_extractor()
        mock_text_extractor_class.return_value = mock_text_extractor
        mock_extraction_agent_class.return_value = create_mock_extraction_agent()
        mock_markdown_gen_class.return_value = create_mock_markdown_generator()

        # Process the job
        result = await process_extraction_job(
            job_id="test-txt-routing",
            filename="test.txt",
            file_path=str(sample_txt_file),
            schema_path=str(sample_schema_file),
            progress=None,
        )

        # Verify text extractor was called (not visual extractor)
        mock_text_extractor.extract_from_path.assert_called_once()
        # JsonGenerator adds None for optional fields, so check required field
        assert result["extracted_data"]["name"] == "test"


class TestExtractionProcessorVisualFormats:
    """Tests for visual format document processing."""

    @pytest.mark.parametrize(
        "extension,format_family",
        [
            (".png", FormatFamily.IMAGE),
            (".jpg", FormatFamily.IMAGE),
            (".jpeg", FormatFamily.IMAGE),
            (".pdf", FormatFamily.PDF),
            (".bmp", FormatFamily.IMAGE),
            (".gif", FormatFamily.IMAGE),
            (".webp", FormatFamily.IMAGE),
            (".tiff", FormatFamily.IMAGE),
        ],
    )
    @patch(
        "agentic_document_extraction.services.extraction_processor.MarkdownGenerator"
    )
    @patch("agentic_document_extraction.services.extraction_processor.ExtractionAgent")
    @patch(
        "agentic_document_extraction.services.extraction_processor.VisualTextExtractor"
    )
    @patch("agentic_document_extraction.services.extraction_processor.FormatDetector")
    async def test_visual_formats_use_visual_extractor(
        self,
        mock_format_detector_class: MagicMock,
        mock_visual_extractor_class: MagicMock,
        mock_extraction_agent_class: MagicMock,
        mock_markdown_gen_class: MagicMock,
        temp_dir: Path,
        sample_schema: dict,
        extension: str,
        format_family: FormatFamily,
    ) -> None:
        """Test that all visual formats use the visual text extractor."""
        # Create test file
        test_file = temp_dir / f"test{extension}"
        if extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"]:
            img = Image.new("RGB", (100, 100), color="white")
            fmt = extension.lstrip(".").upper().replace("JPG", "JPEG")
            img.save(test_file, format=fmt)
        else:
            # For PDF, create a minimal file
            test_file.write_bytes(b"%PDF-1.4\n%EOF")

        # Create schema file
        schema_file = temp_dir / "schema.json"
        schema_file.write_text(json.dumps(sample_schema))

        # Set up mock format detector to return visual category
        mock_format_detector = MagicMock()
        mock_format_info = FormatInfo(
            extension=extension,
            mime_type=f"image/{extension.lstrip('.')}",
            format_family=format_family,
            processing_category=ProcessingCategory.VISUAL,
            detected_by_magic=True,
        )
        mock_format_detector.detect_from_path.return_value = mock_format_info
        mock_format_detector_class.return_value = mock_format_detector

        # Set up mocks
        mock_visual_extractor = create_mock_visual_extractor()
        mock_visual_extractor_class.return_value = mock_visual_extractor
        mock_extraction_agent_class.return_value = create_mock_extraction_agent()
        mock_markdown_gen_class.return_value = create_mock_markdown_generator()

        # Process the job
        result = await process_extraction_job(
            job_id=f"test-{extension.lstrip('.')}-routing",
            filename=f"test{extension}",
            file_path=str(test_file),
            schema_path=str(schema_file),
            progress=None,
        )

        # Verify visual extractor's extract_with_layout was called (new unified pipeline)
        mock_visual_extractor.extract_with_layout.assert_called_once()
        # JsonGenerator adds None for optional fields, so check required field
        assert result["extracted_data"]["name"] == "test"


class TestExtractionProcessorTextFormats:
    """Tests for text format document processing."""

    @pytest.mark.parametrize(
        "extension",
        [".txt", ".csv"],
    )
    @patch(
        "agentic_document_extraction.services.extraction_processor.MarkdownGenerator"
    )
    @patch("agentic_document_extraction.services.extraction_processor.ExtractionAgent")
    @patch("agentic_document_extraction.services.extraction_processor.TextExtractor")
    @patch("agentic_document_extraction.services.extraction_processor.FormatDetector")
    async def test_text_formats_use_text_extractor(
        self,
        mock_format_detector_class: MagicMock,
        mock_text_extractor_class: MagicMock,
        mock_extraction_agent_class: MagicMock,
        mock_markdown_gen_class: MagicMock,
        temp_dir: Path,
        sample_schema: dict,
        extension: str,
    ) -> None:
        """Test that text formats use the text extractor."""
        # Create test file
        test_file = temp_dir / f"test{extension}"
        if extension == ".csv":
            test_file.write_text("name,value\ntest,123")
        else:
            test_file.write_text("Hello World")

        # Create schema file
        schema_file = temp_dir / "schema.json"
        schema_file.write_text(json.dumps(sample_schema))

        # Set up mock format detector to return text category
        mock_format_detector = MagicMock()
        mock_format_info = FormatInfo(
            extension=extension,
            mime_type=f"text/{extension.lstrip('.')}",
            format_family=FormatFamily.PLAIN_TEXT,
            processing_category=ProcessingCategory.TEXT_BASED,
            detected_by_magic=True,
        )
        mock_format_detector.detect_from_path.return_value = mock_format_info
        mock_format_detector_class.return_value = mock_format_detector

        # Set up mocks
        mock_text_extractor = create_mock_text_extractor()
        mock_text_extractor_class.return_value = mock_text_extractor
        mock_extraction_agent_class.return_value = create_mock_extraction_agent()
        mock_markdown_gen_class.return_value = create_mock_markdown_generator()

        # Process the job
        result = await process_extraction_job(
            job_id=f"test-{extension.lstrip('.')}-routing",
            filename=f"test{extension}",
            file_path=str(test_file),
            schema_path=str(schema_file),
            progress=None,
        )

        # Verify text extractor was called
        mock_text_extractor.extract_from_path.assert_called_once()
        # JsonGenerator adds None for optional fields, so check required field
        assert result["extracted_data"]["name"] == "test"


class TestExtractionProcessorStructuredFormats:
    """Tests for structured spreadsheet processing."""

    @patch(
        "agentic_document_extraction.services.extraction_processor.MarkdownGenerator"
    )
    @patch("agentic_document_extraction.services.extraction_processor.ExtractionAgent")
    @patch("agentic_document_extraction.services.extraction_processor.ExcelExtractor")
    @patch("agentic_document_extraction.services.extraction_processor.FormatDetector")
    async def test_xlsx_routes_to_excel_extractor(
        self,
        mock_format_detector_class: MagicMock,
        mock_excel_extractor_class: MagicMock,
        mock_extraction_agent_class: MagicMock,
        mock_markdown_gen_class: MagicMock,
        temp_dir: Path,
        sample_schema: dict,
    ) -> None:
        """Excel files use native extractor when enabled."""
        test_file = temp_dir / "test.xlsx"
        test_file.write_bytes(b"dummy")  # Content not read by extractor mock

        schema_file = temp_dir / "schema.json"
        schema_file.write_text(json.dumps(sample_schema))

        mock_format_detector = MagicMock()
        mock_format_info = FormatInfo(
            extension=".xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            format_family=FormatFamily.SPREADSHEET,
            processing_category=ProcessingCategory.STRUCTURED,
            detected_from_content=True,
        )
        mock_format_detector.detect_from_path.return_value = mock_format_info
        mock_format_detector_class.return_value = mock_format_detector

        mock_excel_doc = MagicMock()
        mock_excel_doc.sheets = []
        mock_excel_doc.active_sheet = "Sheet1"
        mock_excel_extractor_class.return_value.extract_from_path.return_value = (
            mock_excel_doc
        )

        mock_extraction_agent_class.return_value = create_mock_extraction_agent()
        mock_markdown_gen_class.return_value = create_mock_markdown_generator()

        result = await process_extraction_job(
            job_id="test-xlsx-routing",
            filename="test.xlsx",
            file_path=str(test_file),
            schema_path=str(schema_file),
            progress=None,
        )

        mock_excel_extractor_class.return_value.extract_from_path.assert_called_once()
        call_kwargs = mock_extraction_agent_class.return_value.extract.call_args.kwargs
        assert call_kwargs.get("spreadsheet") is mock_excel_doc
        assert result["extracted_data"]["name"] == "test"


class TestExtractionProcessorToolAgent:
    """Tests for the use_tool_agent=True code path in extraction processor.

    The default setting is use_tool_agent=True, so no app_settings mock needed.
    The new pipeline uses extract_with_layout() which combines OCR and layout detection.
    """

    @patch(
        "agentic_document_extraction.services.extraction_processor.MarkdownGenerator"
    )
    @patch("agentic_document_extraction.services.extraction_processor.ExtractionAgent")
    @patch(
        "agentic_document_extraction.services.extraction_processor.VisualTextExtractor"
    )
    async def test_visual_doc_with_layout_regions_passes_to_agent(
        self,
        mock_visual_extractor_class: MagicMock,
        mock_extraction_agent_class: MagicMock,
        mock_markdown_gen_class: MagicMock,
        sample_png_file: Path,
        sample_schema_file: Path,
    ) -> None:
        """Visual doc with use_tool_agent=True uses unified pipeline and passes regions."""
        # Set up mock visual extractor with layout regions
        mock_visual_extractor = MagicMock()
        mock_extraction_with_layout = MagicMock()
        mock_extraction_with_layout.full_text = "Extracted text"
        mock_extraction_with_layout.total_pages = 1
        mock_extraction_with_layout.average_confidence = 0.9

        # Create mock layout region
        mock_region = MagicMock()
        mock_region.region_type.value = "table"
        mock_extraction_with_layout.get_all_regions.return_value = [mock_region]

        # Mock the OCR result
        mock_ocr_result = MagicMock()
        mock_ocr_result.extraction_method.value = "ocr"
        mock_extraction_with_layout.ocr_result = mock_ocr_result

        # Mock layout result with pages
        mock_layout_page = MagicMock()
        mock_layout_page.page_number = 1
        mock_layout_page.regions = [mock_region]
        mock_layout_result = MagicMock()
        mock_layout_result.pages = [mock_layout_page]
        mock_extraction_with_layout.layout_result = mock_layout_result

        mock_visual_extractor.extract_with_layout.return_value = (
            mock_extraction_with_layout
        )
        mock_visual_extractor_class.return_value = mock_visual_extractor

        mock_extraction_agent_class.return_value = create_mock_extraction_agent()
        mock_markdown_gen_class.return_value = create_mock_markdown_generator()

        result = await process_extraction_job(
            job_id="test-visual-layout",
            filename="test_image.png",
            file_path=str(sample_png_file),
            schema_path=str(sample_schema_file),
            progress=None,
        )

        # Unified extract_with_layout was called
        mock_visual_extractor.extract_with_layout.assert_called_once()
        # ExtractionAgent.extract received layout_regions
        call_kwargs = mock_extraction_agent_class.return_value.extract.call_args
        assert call_kwargs.kwargs.get("layout_regions") is not None
        assert result["extracted_data"]["name"] == "test"

    @patch(
        "agentic_document_extraction.services.extraction_processor.MarkdownGenerator"
    )
    @patch("agentic_document_extraction.services.extraction_processor.ExtractionAgent")
    @patch(
        "agentic_document_extraction.services.extraction_processor.VisualTextExtractor"
    )
    async def test_visual_doc_without_regions_uses_fallback(
        self,
        mock_visual_extractor_class: MagicMock,
        mock_extraction_agent_class: MagicMock,
        mock_markdown_gen_class: MagicMock,
        sample_png_file: Path,
        sample_schema_file: Path,
    ) -> None:
        """When extract_with_layout returns no regions, fallback full-image region is used."""
        mock_visual_extractor = create_mock_visual_extractor()
        mock_visual_extractor_class.return_value = mock_visual_extractor

        mock_extraction_agent_class.return_value = create_mock_extraction_agent()
        mock_markdown_gen_class.return_value = create_mock_markdown_generator()

        result = await process_extraction_job(
            job_id="test-no-regions",
            filename="test_image.png",
            file_path=str(sample_png_file),
            schema_path=str(sample_schema_file),
            progress=None,
        )

        # ExtractionAgent.extract still called with fallback region
        call_kwargs = mock_extraction_agent_class.return_value.extract.call_args
        layout_regions = call_kwargs.kwargs.get("layout_regions")
        # Should have fallback full-image region
        assert layout_regions is not None
        assert len(layout_regions) == 1
        assert layout_regions[0].region_id == "fallback_full_image"
        assert result["extracted_data"]["name"] == "test"

    @patch(
        "agentic_document_extraction.services.extraction_processor.MarkdownGenerator"
    )
    @patch("agentic_document_extraction.services.extraction_processor.ExtractionAgent")
    @patch("agentic_document_extraction.services.extraction_processor.TextExtractor")
    async def test_text_doc_skips_layout_detection(
        self,
        mock_text_extractor_class: MagicMock,
        mock_extraction_agent_class: MagicMock,
        mock_markdown_gen_class: MagicMock,
        sample_txt_file: Path,
        sample_schema_file: Path,
    ) -> None:
        """Text doc with use_tool_agent=True never instantiates LayoutDetector."""
        mock_text_extractor = create_mock_text_extractor()
        mock_text_extractor_class.return_value = mock_text_extractor
        mock_extraction_agent_class.return_value = create_mock_extraction_agent()
        mock_markdown_gen_class.return_value = create_mock_markdown_generator()

        with patch(
            "agentic_document_extraction.services.layout_detector.LayoutDetector"
        ) as mock_ld:
            result = await process_extraction_job(
                job_id="test-text-no-layout",
                filename="test.txt",
                file_path=str(sample_txt_file),
                schema_path=str(sample_schema_file),
                progress=None,
            )
            # LayoutDetector never instantiated for text docs
            mock_ld.assert_not_called()

        assert result["extracted_data"]["name"] == "test"
