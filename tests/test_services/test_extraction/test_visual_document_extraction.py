"""Tests for visual document extraction service."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
    FieldExtraction,
)
from agentic_document_extraction.services.extraction.visual_document_extraction import (
    VisualDocumentExtractionService,
    VisualExtractionError,
)
from agentic_document_extraction.services.schema_validator import (
    FieldInfo,
    SchemaInfo,
)


@pytest.fixture
def sample_schema_info() -> SchemaInfo:
    """Create a sample schema info for testing."""
    return SchemaInfo(
        schema={
            "type": "object",
            "properties": {
                "brand": {"type": "string"},
                "media_type": {"type": "string"},
                "issue_date": {"type": "string"},
            },
            "required": ["brand", "media_type"],
        },
        required_fields=[
            FieldInfo(name="brand", field_type="string", required=True, path="brand"),
            FieldInfo(
                name="media_type", field_type="string", required=True, path="media_type"
            ),
        ],
        optional_fields=[
            FieldInfo(
                name="issue_date",
                field_type="string",
                required=False,
                path="issue_date",
            ),
        ],
        schema_type="object",
    )


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample test image."""
    return Image.new("RGB", (100, 100), color="white")


class TestVisualDocumentExtractionService:
    """Tests for VisualDocumentExtractionService."""

    def test_init_default_settings(self) -> None:
        """Test service initializes with default settings."""
        service = VisualDocumentExtractionService()
        assert service.model is not None
        assert service.temperature is not None

    def test_init_custom_settings(self) -> None:
        """Test service initializes with custom settings."""
        service = VisualDocumentExtractionService(
            api_key="test-key",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=1000,
        )
        assert service.api_key == "test-key"
        assert service.model == "gpt-4o"
        assert service.temperature == 0.5
        assert service.max_tokens == 1000

    def test_llm_property_creates_instance(self) -> None:
        """Test llm property creates ChatOpenAI instance."""
        service = VisualDocumentExtractionService(api_key="test-key")
        llm = service.llm
        assert llm is not None

    def test_llm_property_reuses_instance(self) -> None:
        """Test llm property returns same instance on subsequent calls."""
        service = VisualDocumentExtractionService(api_key="test-key")
        llm1 = service.llm
        llm2 = service.llm
        assert llm1 is llm2

    def test_llm_property_raises_without_api_key(self) -> None:
        """Test llm property raises error if no API key."""
        service = VisualDocumentExtractionService(api_key="")
        with pytest.raises(VisualExtractionError) as exc_info:
            _ = service.llm
        assert "API key not configured" in str(exc_info.value)
        assert exc_info.value.error_type == "configuration_error"

    def test_encode_image_to_base64_rgb(self, sample_image: Image.Image) -> None:
        """Test encoding RGB image to base64."""
        service = VisualDocumentExtractionService(api_key="test-key")
        result = service._encode_image_to_base64(sample_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_image_to_base64_rgba(self) -> None:
        """Test encoding RGBA image converts to RGB."""
        service = VisualDocumentExtractionService(api_key="test-key")
        rgba_image = Image.new("RGBA", (100, 100), color=(255, 255, 255, 128))
        result = service._encode_image_to_base64(rgba_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_load_image_from_pil(self, sample_image: Image.Image) -> None:
        """Test loading image from PIL Image."""
        service = VisualDocumentExtractionService(api_key="test-key")
        result = service._load_image(sample_image)
        assert result is sample_image

    def test_load_image_from_path(self, tmp_path: Path) -> None:
        """Test loading image from file path."""
        image_path = tmp_path / "test.png"
        Image.new("RGB", (100, 100), color="white").save(image_path)

        service = VisualDocumentExtractionService(api_key="test-key")
        result = service._load_image(image_path)
        assert isinstance(result, Image.Image)

    def test_load_image_file_not_found(self) -> None:
        """Test loading nonexistent image raises error."""
        service = VisualDocumentExtractionService(api_key="test-key")
        with pytest.raises(VisualExtractionError) as exc_info:
            service._load_image("/nonexistent/path.png")
        assert "not found" in str(exc_info.value)
        assert exc_info.value.error_type == "file_not_found"

    def test_parse_json_response_valid(self, sample_schema_info: SchemaInfo) -> None:
        """Test parsing valid JSON response."""
        service = VisualDocumentExtractionService(api_key="test-key")
        response = '{"brand": "Test Brand", "media_type": "Email"}'
        result = service._parse_json_response(response, sample_schema_info)
        assert result == {"brand": "Test Brand", "media_type": "Email"}

    def test_parse_json_response_extracts_from_text(
        self, sample_schema_info: SchemaInfo
    ) -> None:
        """Test parsing JSON embedded in text."""
        service = VisualDocumentExtractionService(api_key="test-key")
        response = 'Here is the JSON: {"brand": "Test"} That is all.'
        result = service._parse_json_response(response, sample_schema_info)
        assert result == {"brand": "Test"}

    def test_parse_json_response_invalid(self, sample_schema_info: SchemaInfo) -> None:
        """Test parsing invalid JSON raises error."""
        service = VisualDocumentExtractionService(api_key="test-key")
        with pytest.raises(VisualExtractionError) as exc_info:
            service._parse_json_response("not valid json", sample_schema_info)
        assert exc_info.value.error_type == "parse_error"

    def test_parse_json_response_non_object(
        self, sample_schema_info: SchemaInfo
    ) -> None:
        """Test parsing non-object JSON raises error."""
        service = VisualDocumentExtractionService(api_key="test-key")
        with pytest.raises(VisualExtractionError) as exc_info:
            service._parse_json_response('["array", "items"]', sample_schema_info)
        assert "Expected JSON object" in str(exc_info.value)

    def test_build_field_extractions(self, sample_schema_info: SchemaInfo) -> None:
        """Test building field extractions from data."""
        service = VisualDocumentExtractionService(api_key="test-key")
        data = {
            "brand": "Test Brand",
            "media_type": "Email",
            "issue_date": "2024-01-01",
        }
        extractions = service._build_field_extractions(data, sample_schema_info)

        assert len(extractions) == 3
        assert extractions[0].field_path == "brand"
        assert extractions[0].value == "Test Brand"
        assert extractions[1].field_path == "media_type"
        assert extractions[1].value == "Email"

    def test_get_nested_value(self) -> None:
        """Test getting nested values from dict."""
        service = VisualDocumentExtractionService(api_key="test-key")
        data = {"level1": {"level2": {"value": "found"}}}

        assert service._get_nested_value(data, "level1.level2.value") == "found"
        assert service._get_nested_value(data, "level1.level2") == {"value": "found"}
        assert service._get_nested_value(data, "nonexistent") is None

    @patch.object(VisualDocumentExtractionService, "llm", new_callable=MagicMock)
    def test_extract_success(
        self,
        mock_llm: MagicMock,
        sample_image: Image.Image,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test successful extraction from image."""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "brand": "OLD GOLD",
                "media_type": "DIRECT MAIL",
                "issue_date": "10/4/99",
            }
        )
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        mock_llm.invoke.return_value = mock_response

        service = VisualDocumentExtractionService(api_key="test-key")
        result = service.extract(sample_image, sample_schema_info)

        assert isinstance(result, ExtractionResult)
        assert result.extracted_data["brand"] == "OLD GOLD"
        assert result.extracted_data["media_type"] == "DIRECT MAIL"
        assert result.total_tokens == 150

    @patch.object(VisualDocumentExtractionService, "llm", new_callable=MagicMock)
    def test_extract_with_ocr_text(
        self,
        mock_llm: MagicMock,
        sample_image: Image.Image,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test extraction includes OCR text in prompt."""
        mock_response = MagicMock()
        mock_response.content = '{"brand": "Test", "media_type": "Email"}'
        mock_response.usage_metadata = {}
        mock_llm.invoke.return_value = mock_response

        service = VisualDocumentExtractionService(api_key="test-key")
        service.extract(
            sample_image,
            sample_schema_info,
            ocr_text="Some OCR extracted text",
        )

        # Verify OCR text was included in the prompt
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        # The second message is the HumanMessage with the prompt
        human_msg = messages[1]
        # Content is a list with text and image_url
        text_content = human_msg.content[0]["text"]
        assert "Some OCR extracted text" in text_content

    @patch.object(VisualDocumentExtractionService, "llm", new_callable=MagicMock)
    def test_extract_llm_error(
        self,
        mock_llm: MagicMock,
        sample_image: Image.Image,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test extraction handles LLM errors."""
        mock_llm.invoke.side_effect = Exception("API error")

        service = VisualDocumentExtractionService(api_key="test-key")
        with pytest.raises(VisualExtractionError) as exc_info:
            service.extract(sample_image, sample_schema_info)

        assert "Visual extraction failed" in str(exc_info.value)
        assert exc_info.value.error_type == "vlm_error"


class TestVisualDocumentExtractionServicePrompts:
    """Tests for prompt construction in VisualDocumentExtractionService."""

    def test_system_prompt_contains_form_instructions(self) -> None:
        """Test system prompt includes form-specific instructions."""
        prompt = VisualDocumentExtractionService.EXTRACTION_SYSTEM_PROMPT
        assert "form" in prompt.lower()
        assert "label" in prompt.lower()
        assert "value" in prompt.lower()

    def test_user_prompt_template_has_placeholders(self) -> None:
        """Test user prompt template has required placeholders."""
        template = VisualDocumentExtractionService.EXTRACTION_USER_PROMPT
        assert "{schema}" in template
        assert "{required_fields}" in template
        assert "{optional_fields}" in template
        assert "{ocr_text_section}" in template


class TestExtractionResultCompatibility:
    """Tests to verify ExtractionResult compatibility."""

    def test_extraction_result_has_required_fields(self) -> None:
        """Test ExtractionResult has all fields needed by agentic loop."""
        result = ExtractionResult(
            extracted_data={"test": "data"},
            field_extractions=[],
            model_used="gpt-4o",
            total_tokens=100,
            prompt_tokens=80,
            completion_tokens=20,
            chunks_processed=1,
            is_chunked=False,
        )

        # These are the fields used by the agentic loop
        assert hasattr(result, "extracted_data")
        assert hasattr(result, "field_extractions")
        assert hasattr(result, "model_used")
        assert hasattr(result, "total_tokens")

    def test_field_extraction_dataclass(self) -> None:
        """Test FieldExtraction has expected structure."""
        fe = FieldExtraction(
            field_path="test.path",
            value="test value",
            confidence=0.95,
            source_text="source",
            reasoning="because",
        )

        assert fe.field_path == "test.path"
        assert fe.value == "test value"
        assert fe.confidence == 0.95
