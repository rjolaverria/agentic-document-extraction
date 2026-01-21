"""Tests for the visual document synthesis service."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_document_extraction.services.extraction.region_visual_extraction import (
    DocumentRegionExtractionResult,
    ExtractionStrategy,
    RegionExtractionResult,
)
from agentic_document_extraction.services.extraction.synthesis import (
    RegionSourceReference,
    SynthesisError,
    SynthesisResult,
    SynthesisService,
    SynthesizedContent,
)
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionType,
)
from agentic_document_extraction.services.reading_order_detector import (
    DocumentReadingOrder,
    OrderedRegion,
    PageReadingOrder,
)
from agentic_document_extraction.services.schema_validator import FieldInfo, SchemaInfo


# Helper functions for creating test objects
def create_region(
    region_id: str,
    region_type: RegionType,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_number: int = 1,
    confidence: float = 0.9,
    parent_region_id: str | None = None,
) -> LayoutRegion:
    """Helper to create a layout region for testing."""
    return LayoutRegion(
        region_type=region_type,
        bbox=RegionBoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        confidence=confidence,
        page_number=page_number,
        region_id=region_id,
        parent_region_id=parent_region_id,
    )


def create_ordered_region(
    region: LayoutRegion,
    order_index: int,
    confidence: float = 0.9,
    skip_in_reading: bool = False,
) -> OrderedRegion:
    """Helper to create an ordered region for testing."""
    return OrderedRegion(
        region=region,
        order_index=order_index,
        confidence=confidence,
        skip_in_reading=skip_in_reading,
    )


def create_region_extraction_result(
    region_id: str,
    region_type: RegionType,
    extracted_content: dict,
    confidence: float = 0.9,
) -> RegionExtractionResult:
    """Helper to create a region extraction result for testing."""
    return RegionExtractionResult(
        region_id=region_id,
        region_type=region_type,
        extracted_content=extracted_content,
        confidence=confidence,
        strategy_used=ExtractionStrategy.VLM_ONLY,
    )


def create_sample_schema_info() -> SchemaInfo:
    """Create a sample schema info for testing."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Document title"},
            "date": {"type": "string", "description": "Document date"},
            "summary": {"type": "string", "description": "Document summary"},
            "items": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
        "required": ["title"],
    }

    required_fields = [
        FieldInfo(
            name="title",
            field_type="string",
            required=True,
            description="Document title",
            path="title",
        )
    ]

    optional_fields = [
        FieldInfo(
            name="date",
            field_type="string",
            required=False,
            description="Document date",
            path="date",
        ),
        FieldInfo(
            name="summary",
            field_type="string",
            required=False,
            description="Document summary",
            path="summary",
        ),
        FieldInfo(
            name="items",
            field_type="array",
            required=False,
            description="Items list",
            path="items",
        ),
    ]

    return SchemaInfo(
        schema=schema,
        required_fields=required_fields,
        optional_fields=optional_fields,
        schema_type="object",
    )


@pytest.fixture
def service() -> SynthesisService:
    """Create a SynthesisService instance for testing."""
    return SynthesisService(api_key="test-key")


@pytest.fixture
def sample_schema_info() -> SchemaInfo:
    """Create a sample schema info."""
    return create_sample_schema_info()


@pytest.fixture
def sample_regions() -> list[LayoutRegion]:
    """Create sample layout regions."""
    return [
        create_region("r1", RegionType.TITLE, 100, 50, 900, 100),
        create_region("r2", RegionType.TEXT, 100, 150, 900, 300),
        create_region("r3", RegionType.TABLE, 100, 350, 900, 550),
    ]


@pytest.fixture
def sample_ordered_regions(sample_regions: list[LayoutRegion]) -> list[OrderedRegion]:
    """Create sample ordered regions."""
    return [
        create_ordered_region(sample_regions[0], 0),
        create_ordered_region(sample_regions[1], 1),
        create_ordered_region(sample_regions[2], 2),
    ]


@pytest.fixture
def sample_region_extraction() -> DocumentRegionExtractionResult:
    """Create sample document region extraction result."""
    region_results = [
        create_region_extraction_result(
            "r1",
            RegionType.TITLE,
            {"text": "Annual Report 2024", "text_type": "title"},
            confidence=0.95,
        ),
        create_region_extraction_result(
            "r2",
            RegionType.TEXT,
            {
                "text": "This report summarizes our company's performance...",
                "text_type": "paragraph",
            },
            confidence=0.9,
        ),
        create_region_extraction_result(
            "r3",
            RegionType.TABLE,
            {
                "headers": ["Quarter", "Revenue", "Growth"],
                "rows": [["Q1", "$1M", "10%"], ["Q2", "$1.2M", "20%"]],
            },
            confidence=0.85,
        ),
    ]

    return DocumentRegionExtractionResult(
        region_results=region_results,
        total_regions=3,
        successful_extractions=3,
        failed_extractions=0,
        model_used="gpt-4o",
        total_tokens=500,
        prompt_tokens=300,
        completion_tokens=200,
        processing_time_seconds=2.5,
    )


@pytest.fixture
def sample_reading_order(
    sample_ordered_regions: list[OrderedRegion],
) -> DocumentReadingOrder:
    """Create sample document reading order."""
    page_order = PageReadingOrder(
        page_number=1,
        ordered_regions=sample_ordered_regions,
        overall_confidence=0.9,
        layout_type="single_column",
        processing_time_seconds=0.5,
    )

    return DocumentReadingOrder(
        pages=[page_order],
        total_pages=1,
        total_regions=3,
        model_used="gpt-4o",
        processing_time_seconds=0.5,
    )


@pytest.fixture
def mock_synthesis_response() -> str:
    """Mock synthesis response from LLM."""
    return json.dumps(
        {
            "extracted_data": {
                "title": "Annual Report 2024",
                "date": "2024",
                "summary": "This report summarizes our company's performance...",
                "items": [
                    {"quarter": "Q1", "revenue": "$1M", "growth": "10%"},
                    {"quarter": "Q2", "revenue": "$1.2M", "growth": "20%"},
                ],
            },
            "field_confidence": {
                "title": 0.95,
                "date": 0.7,
                "summary": 0.9,
                "items": 0.85,
            },
            "overall_confidence": 0.88,
            "reasoning": "Combined information from title, text, and table regions.",
        }
    )


class TestSynthesisError:
    """Tests for SynthesisError exception."""

    def test_error_with_message(self) -> None:
        """Test error with just message."""
        error = SynthesisError("Test error")

        assert str(error) == "Test error"
        assert error.error_type == "synthesis_error"
        assert error.details == {}

    def test_error_with_error_type(self) -> None:
        """Test error with custom error type."""
        error = SynthesisError("Test error", error_type="parse_error")

        assert error.error_type == "parse_error"

    def test_error_with_details(self) -> None:
        """Test error with details."""
        error = SynthesisError(
            "Test error",
            error_type="llm_error",
            details={"original_error": "API timeout"},
        )

        assert error.details == {"original_error": "API timeout"}


class TestRegionSourceReference:
    """Tests for RegionSourceReference dataclass."""

    def test_create_source_reference(self) -> None:
        """Test creating a RegionSourceReference."""
        ref = RegionSourceReference(
            region_id="r1",
            region_type="text",
            page_number=1,
            reading_order_position=0,
            bbox={"x0": 100, "y0": 100, "x1": 500, "y1": 300},
            confidence=0.9,
        )

        assert ref.region_id == "r1"
        assert ref.region_type == "text"
        assert ref.page_number == 1
        assert ref.reading_order_position == 0
        assert ref.confidence == 0.9

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        ref = RegionSourceReference(
            region_id="r1",
            region_type="table",
            page_number=2,
            reading_order_position=3,
            bbox={"x0": 50, "y0": 50, "x1": 400, "y1": 200},
            confidence=0.85,
        )

        data = ref.to_dict()

        assert data["region_id"] == "r1"
        assert data["region_type"] == "table"
        assert data["page_number"] == 2
        assert data["reading_order_position"] == 3
        assert data["confidence"] == 0.85


class TestSynthesizedContent:
    """Tests for SynthesizedContent dataclass."""

    def test_create_synthesized_content(self) -> None:
        """Test creating SynthesizedContent."""
        source_ref = RegionSourceReference(
            region_id="r1",
            region_type="text",
            page_number=1,
            reading_order_position=0,
        )

        content = SynthesizedContent(
            content={"text": "Hello world"},
            content_type="text",
            source_ref=source_ref,
        )

        assert content.content == {"text": "Hello world"}
        assert content.content_type == "text"
        assert content.source_ref.region_id == "r1"

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        source_ref = RegionSourceReference(
            region_id="r1",
            region_type="table",
            page_number=1,
            reading_order_position=0,
        )

        content = SynthesizedContent(
            content={"headers": ["A", "B"]},
            content_type="table_data",
            source_ref=source_ref,
        )

        data = content.to_dict()

        assert data["content"] == {"headers": ["A", "B"]}
        assert data["content_type"] == "table_data"
        assert data["source_ref"]["region_id"] == "r1"


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_create_synthesis_result(self) -> None:
        """Test creating a SynthesisResult."""
        result = SynthesisResult(
            extracted_data={"title": "Test Document"},
            model_used="gpt-4o",
            total_tokens=500,
            prompt_tokens=300,
            completion_tokens=200,
            processing_time_seconds=1.5,
            total_regions_processed=3,
            total_pages=1,
            synthesis_confidence=0.9,
        )

        assert result.extracted_data == {"title": "Test Document"}
        assert result.model_used == "gpt-4o"
        assert result.total_tokens == 500
        assert result.synthesis_confidence == 0.9

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        result = SynthesisResult(
            extracted_data={"title": "Test"},
            model_used="gpt-4o",
            total_tokens=100,
            prompt_tokens=60,
            completion_tokens=40,
            processing_time_seconds=0.5,
            total_regions_processed=2,
            total_pages=1,
            synthesis_confidence=0.85,
        )

        data = result.to_dict()

        assert data["extracted_data"] == {"title": "Test"}
        assert data["metadata"]["model_used"] == "gpt-4o"
        assert data["metadata"]["total_tokens"] == 100
        assert data["metadata"]["synthesis_confidence"] == 0.85

    def test_to_extraction_result(self) -> None:
        """Test converting to ExtractionResult."""
        result = SynthesisResult(
            extracted_data={"title": "Test Document", "date": "2024"},
            model_used="gpt-4o",
            total_tokens=500,
            prompt_tokens=300,
            completion_tokens=200,
            processing_time_seconds=1.5,
            total_regions_processed=3,
            total_pages=2,
            synthesis_confidence=0.9,
            raw_response="test response",
        )

        extraction_result = result.to_extraction_result()

        assert isinstance(extraction_result, ExtractionResult)
        assert extraction_result.extracted_data == {
            "title": "Test Document",
            "date": "2024",
        }
        assert extraction_result.model_used == "gpt-4o"
        assert extraction_result.total_tokens == 500
        assert extraction_result.chunks_processed == 2
        assert extraction_result.is_chunked is True
        assert extraction_result.raw_response == "test response"


class TestSynthesisServiceInit:
    """Tests for SynthesisService initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with defaults."""
        with patch(
            "agentic_document_extraction.services.extraction.synthesis.settings"
        ) as mock_settings:
            mock_settings.get_openai_api_key.return_value = "env-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.0
            mock_settings.openai_max_tokens = 4096

            service = SynthesisService()

            assert service.api_key == "env-key"
            assert service.model == "gpt-4o"
            assert service.temperature == 0.0
            assert service.max_tokens == 4096

    def test_custom_initialization(self) -> None:
        """Test initialization with custom values."""
        service = SynthesisService(
            api_key="custom-key",
            model="gpt-4-turbo",
            temperature=0.5,
            max_tokens=2048,
        )

        assert service.api_key == "custom-key"
        assert service.model == "gpt-4-turbo"
        assert service.temperature == 0.5
        assert service.max_tokens == 2048

    def test_llm_property_raises_without_key(self) -> None:
        """Test that accessing llm property without API key raises error."""
        service = SynthesisService(api_key="")

        with pytest.raises(SynthesisError) as exc_info:
            _ = service.llm

        assert "API key not configured" in str(exc_info.value)


class TestSynthesisServiceContentBuilding:
    """Tests for content building functionality."""

    def test_get_content_type_text(self, service: SynthesisService) -> None:
        """Test content type mapping for text regions."""
        assert service._get_content_type(RegionType.TEXT) == "text"
        assert service._get_content_type(RegionType.TITLE) == "title"
        assert service._get_content_type(RegionType.SECTION_HEADER) == "section_header"

    def test_get_content_type_table(self, service: SynthesisService) -> None:
        """Test content type mapping for table regions."""
        assert service._get_content_type(RegionType.TABLE) == "table_data"

    def test_get_content_type_figure(self, service: SynthesisService) -> None:
        """Test content type mapping for figure regions."""
        assert service._get_content_type(RegionType.PICTURE) == "figure_description"

    def test_get_content_type_unknown(self, service: SynthesisService) -> None:
        """Test content type mapping for unknown regions."""
        assert service._get_content_type(RegionType.UNKNOWN) == "other"

    def test_build_ordered_content(
        self,
        service: SynthesisService,
        sample_region_extraction: DocumentRegionExtractionResult,
        sample_reading_order: DocumentReadingOrder,
    ) -> None:
        """Test building ordered content from extractions."""
        contents, refs = service._build_ordered_content(
            sample_region_extraction, sample_reading_order
        )

        assert len(contents) == 3
        assert len(refs) == 3

        # Check order follows reading order
        assert contents[0].content_type == "title"
        assert contents[1].content_type == "text"
        assert contents[2].content_type == "table_data"

        # Check source references
        assert refs[0].region_id == "r1"
        assert refs[0].page_number == 1
        assert refs[0].reading_order_position == 0

    def test_build_ordered_content_skips_reading_order(
        self,
        service: SynthesisService,
        sample_regions: list[LayoutRegion],
        sample_region_extraction: DocumentRegionExtractionResult,
    ) -> None:
        """Test that regions marked as skip_in_reading are skipped."""
        # Create ordered regions with first one skipped
        ordered_regions = [
            create_ordered_region(sample_regions[0], 0, skip_in_reading=True),
            create_ordered_region(sample_regions[1], 1),
            create_ordered_region(sample_regions[2], 2),
        ]

        page_order = PageReadingOrder(
            page_number=1,
            ordered_regions=ordered_regions,
            overall_confidence=0.9,
            layout_type="single_column",
        )

        reading_order = DocumentReadingOrder(
            pages=[page_order],
            total_pages=1,
            total_regions=3,
            model_used="gpt-4o",
        )

        contents, refs = service._build_ordered_content(
            sample_region_extraction, reading_order
        )

        # First region should be skipped
        assert len(contents) == 2
        assert contents[0].content_type == "text"
        assert contents[1].content_type == "table_data"


class TestSynthesisServiceFormatting:
    """Tests for prompt formatting functionality."""

    def test_format_regions_for_prompt(
        self,
        service: SynthesisService,
    ) -> None:
        """Test formatting regions for LLM prompt."""
        source_ref = RegionSourceReference(
            region_id="r1",
            region_type="text",
            page_number=1,
            reading_order_position=0,
            confidence=0.9,
        )

        contents = [
            SynthesizedContent(
                content={"text": "Hello world"},
                content_type="text",
                source_ref=source_ref,
            )
        ]

        formatted = service._format_regions_for_prompt(contents)

        assert "Region 1: TEXT" in formatted
        assert "Page: 1" in formatted
        assert "Region Type: text" in formatted
        assert "Confidence: 0.90" in formatted
        assert "Hello world" in formatted

    def test_format_fields(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test formatting fields for prompt."""
        formatted = service._format_fields(sample_schema_info.required_fields)

        assert "title" in formatted
        assert "string" in formatted
        assert "Document title" in formatted

    def test_format_fields_empty(self, service: SynthesisService) -> None:
        """Test formatting empty fields list."""
        formatted = service._format_fields([])
        assert formatted == "None"


class TestSynthesisServiceResponseParsing:
    """Tests for response parsing functionality."""

    def test_parse_synthesis_response_valid(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
        mock_synthesis_response: str,
    ) -> None:
        """Test parsing valid synthesis response."""
        data, confidence_map, overall_conf, reasoning = (
            service._parse_synthesis_response(
                mock_synthesis_response, sample_schema_info
            )
        )

        assert data["title"] == "Annual Report 2024"
        assert data["date"] == "2024"
        assert "items" in data
        assert confidence_map["title"] == 0.95
        assert overall_conf == 0.88
        assert "combined" in reasoning.lower()

    def test_parse_synthesis_response_with_extra_text(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test parsing response with extra text before JSON."""
        response = 'Here is the extracted data: {"extracted_data": {"title": "Test"}, "overall_confidence": 0.8}'

        data, _, confidence, _ = service._parse_synthesis_response(
            response, sample_schema_info
        )

        assert data["title"] == "Test"
        assert confidence == 0.8

    def test_parse_synthesis_response_clamps_confidence(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test that confidence is clamped to valid range."""
        response = json.dumps(
            {
                "extracted_data": {"title": "Test"},
                "overall_confidence": 1.5,
            }
        )

        _, _, confidence, _ = service._parse_synthesis_response(
            response, sample_schema_info
        )

        assert confidence == 1.0

    def test_parse_synthesis_response_invalid_json(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test parsing invalid JSON response."""
        with pytest.raises(SynthesisError) as exc_info:
            service._parse_synthesis_response(
                "not valid json at all", sample_schema_info
            )

        assert "Failed to parse" in str(exc_info.value)


class TestSynthesisServiceFieldExtractions:
    """Tests for field extraction building."""

    def test_build_field_extractions(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test building field extractions."""
        data = {"title": "Test", "date": "2024", "summary": None}
        confidence_map = {"title": 0.95, "date": 0.8}
        source_refs = [
            RegionSourceReference(
                region_id="r1",
                region_type="text",
                page_number=1,
                reading_order_position=0,
            )
        ]

        extractions = service._build_field_extractions(
            data, sample_schema_info, confidence_map, source_refs
        )

        # Should have extractions for all fields
        assert len(extractions) >= 3

        # Check title extraction
        title_extraction = next(e for e in extractions if e.field_path == "title")
        assert title_extraction.value == "Test"
        assert title_extraction.confidence == 0.95

        # Check date extraction
        date_extraction = next(e for e in extractions if e.field_path == "date")
        assert date_extraction.value == "2024"
        assert date_extraction.confidence == 0.8


class TestSynthesisServiceNestedValues:
    """Tests for nested value handling."""

    def test_get_nested_value_simple(self, service: SynthesisService) -> None:
        """Test getting simple nested values."""
        data = {"name": "Test", "value": 42}

        assert service._get_nested_value(data, "name") == "Test"
        assert service._get_nested_value(data, "value") == 42

    def test_get_nested_value_deep(self, service: SynthesisService) -> None:
        """Test getting deeply nested values."""
        data = {"address": {"city": {"name": "New York", "zip": "10001"}}}

        assert service._get_nested_value(data, "address.city.name") == "New York"
        assert service._get_nested_value(data, "address.city.zip") == "10001"

    def test_get_nested_value_missing(self, service: SynthesisService) -> None:
        """Test getting missing nested values."""
        data = {"name": "Test"}

        assert service._get_nested_value(data, "missing") is None
        assert service._get_nested_value(data, "name.nested") is None

    def test_set_nested_value(self, service: SynthesisService) -> None:
        """Test setting nested values."""
        data: dict = {}

        service._set_nested_value(data, "name", "Test")
        assert data["name"] == "Test"

        service._set_nested_value(data, "address.city", "New York")
        assert data["address"]["city"] == "New York"


class TestSynthesisServicePipelineMetadata:
    """Tests for pipeline metadata building."""

    def test_build_pipeline_metadata(
        self,
        service: SynthesisService,
        sample_region_extraction: DocumentRegionExtractionResult,
        sample_reading_order: DocumentReadingOrder,
    ) -> None:
        """Test building pipeline metadata."""
        metadata = service._build_pipeline_metadata(
            sample_region_extraction, sample_reading_order
        )

        assert "region_extraction" in metadata
        assert "reading_order" in metadata

        # Check region extraction metadata
        region_meta = metadata["region_extraction"]
        assert region_meta["total_regions"] == 3
        assert region_meta["successful_extractions"] == 3
        assert region_meta["failed_extractions"] == 0
        assert "region_types" in region_meta

        # Check reading order metadata
        order_meta = metadata["reading_order"]
        assert order_meta["total_pages"] == 1
        assert order_meta["total_regions"] == 3


class TestSynthesisServiceEmptyResult:
    """Tests for empty result handling."""

    def test_create_empty_result(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
        sample_region_extraction: DocumentRegionExtractionResult,
    ) -> None:
        """Test creating empty result for empty documents."""
        result = service._create_empty_result(
            sample_schema_info, sample_region_extraction
        )

        assert result.extracted_data is not None
        assert result.synthesis_confidence == 0.0
        assert result.total_regions_processed == 0
        assert result.pipeline_metadata["status"] == "empty_document"


class TestSynthesisServiceSynthesize:
    """Tests for main synthesize functionality."""

    def test_synthesize_success(
        self,
        service: SynthesisService,
        sample_region_extraction: DocumentRegionExtractionResult,
        sample_reading_order: DocumentReadingOrder,
        sample_schema_info: SchemaInfo,
        mock_synthesis_response: str,
    ) -> None:
        """Test successful synthesis."""
        mock_response = MagicMock()
        mock_response.content = mock_synthesis_response
        mock_response.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        result = service.synthesize(
            sample_region_extraction, sample_reading_order, sample_schema_info
        )

        assert result.extracted_data["title"] == "Annual Report 2024"
        assert result.synthesis_confidence == 0.88
        assert result.total_regions_processed == 3
        assert result.total_pages == 1
        assert result.model_used == service.model

        # Check tokens include both synthesis and region extraction
        assert result.total_tokens > 300

    def test_synthesize_empty_document(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test synthesis with empty document."""
        # Empty region extraction
        region_extraction = DocumentRegionExtractionResult(
            region_results=[],
            total_regions=0,
            successful_extractions=0,
            failed_extractions=0,
            model_used="gpt-4o",
        )

        # Empty reading order
        page_order = PageReadingOrder(
            page_number=1,
            ordered_regions=[],
            overall_confidence=1.0,
            layout_type="empty",
        )
        reading_order = DocumentReadingOrder(
            pages=[page_order],
            total_pages=1,
            total_regions=0,
            model_used="gpt-4o",
        )

        result = service.synthesize(
            region_extraction, reading_order, sample_schema_info
        )

        assert result.synthesis_confidence == 0.0
        assert result.total_regions_processed == 0
        assert result.pipeline_metadata["status"] == "empty_document"

    def test_synthesize_handles_llm_error(
        self,
        service: SynthesisService,
        sample_region_extraction: DocumentRegionExtractionResult,
        sample_reading_order: DocumentReadingOrder,
        sample_schema_info: SchemaInfo,
    ) -> None:
        """Test error handling when LLM call fails."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API Error")
        service._llm = mock_llm

        with pytest.raises(SynthesisError) as exc_info:
            service.synthesize(
                sample_region_extraction, sample_reading_order, sample_schema_info
            )

        assert "Synthesis failed" in str(exc_info.value)
        assert exc_info.value.error_type == "llm_error"

    def test_synthesize_builds_source_references(
        self,
        service: SynthesisService,
        sample_region_extraction: DocumentRegionExtractionResult,
        sample_reading_order: DocumentReadingOrder,
        sample_schema_info: SchemaInfo,
        mock_synthesis_response: str,
    ) -> None:
        """Test that source references are built correctly."""
        mock_response = MagicMock()
        mock_response.content = mock_synthesis_response
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        result = service.synthesize(
            sample_region_extraction, sample_reading_order, sample_schema_info
        )

        assert len(result.source_references) == 3
        assert result.source_references[0].region_id == "r1"
        assert result.source_references[0].page_number == 1
        assert result.source_references[0].reading_order_position == 0


class TestSynthesisServiceMultiPage:
    """Tests for multi-page document synthesis."""

    def test_synthesize_multi_page_document(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
        mock_synthesis_response: str,
    ) -> None:
        """Test synthesis of multi-page document."""
        # Create regions for two pages
        page1_regions = [
            create_region("p1_r1", RegionType.TITLE, 100, 50, 900, 100, page_number=1),
            create_region("p1_r2", RegionType.TEXT, 100, 150, 900, 300, page_number=1),
        ]
        page2_regions = [
            create_region("p2_r1", RegionType.TABLE, 100, 50, 900, 300, page_number=2),
        ]

        # Create extraction results
        region_results = [
            create_region_extraction_result(
                "p1_r1", RegionType.TITLE, {"text": "Title"}
            ),
            create_region_extraction_result(
                "p1_r2", RegionType.TEXT, {"text": "Content"}
            ),
            create_region_extraction_result(
                "p2_r1", RegionType.TABLE, {"headers": ["A", "B"]}
            ),
        ]

        region_extraction = DocumentRegionExtractionResult(
            region_results=region_results,
            total_regions=3,
            successful_extractions=3,
            failed_extractions=0,
            model_used="gpt-4o",
        )

        # Create reading orders
        page1_order = PageReadingOrder(
            page_number=1,
            ordered_regions=[
                create_ordered_region(page1_regions[0], 0),
                create_ordered_region(page1_regions[1], 1),
            ],
            overall_confidence=0.9,
            layout_type="single_column",
        )
        page2_order = PageReadingOrder(
            page_number=2,
            ordered_regions=[
                create_ordered_region(page2_regions[0], 0),
            ],
            overall_confidence=0.85,
            layout_type="single_column",
        )

        reading_order = DocumentReadingOrder(
            pages=[page1_order, page2_order],
            total_pages=2,
            total_regions=3,
            model_used="gpt-4o",
        )

        mock_response = MagicMock()
        mock_response.content = mock_synthesis_response
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        result = service.synthesize(
            region_extraction, reading_order, sample_schema_info
        )

        assert result.total_pages == 2
        assert result.total_regions_processed == 3

        # Check source references span multiple pages
        pages = {ref.page_number for ref in result.source_references}
        assert pages == {1, 2}


class TestSynthesisServiceSynthesizeFromPageResults:
    """Tests for synthesize_from_page_results convenience method."""

    def test_synthesize_from_page_results(
        self,
        service: SynthesisService,
        sample_schema_info: SchemaInfo,
        mock_synthesis_response: str,
    ) -> None:
        """Test synthesizing from per-page results."""
        # Create per-page extractions
        page1_extraction = DocumentRegionExtractionResult(
            region_results=[
                create_region_extraction_result(
                    "p1_r1", RegionType.TITLE, {"text": "Title"}
                ),
            ],
            total_regions=1,
            successful_extractions=1,
            failed_extractions=0,
            model_used="gpt-4o",
            total_tokens=100,
        )

        page2_extraction = DocumentRegionExtractionResult(
            region_results=[
                create_region_extraction_result(
                    "p2_r1", RegionType.TEXT, {"text": "Content"}
                ),
            ],
            total_regions=1,
            successful_extractions=1,
            failed_extractions=0,
            model_used="gpt-4o",
            total_tokens=150,
        )

        # Create per-page reading orders
        page1_regions = [
            create_region("p1_r1", RegionType.TITLE, 100, 50, 900, 100, page_number=1),
        ]
        page2_regions = [
            create_region("p2_r1", RegionType.TEXT, 100, 50, 900, 200, page_number=2),
        ]

        page1_order = PageReadingOrder(
            page_number=1,
            ordered_regions=[create_ordered_region(page1_regions[0], 0)],
            overall_confidence=0.9,
            layout_type="single_column",
        )
        page2_order = PageReadingOrder(
            page_number=2,
            ordered_regions=[create_ordered_region(page2_regions[0], 0)],
            overall_confidence=0.85,
            layout_type="single_column",
        )

        mock_response = MagicMock()
        mock_response.content = mock_synthesis_response
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        result = service.synthesize_from_page_results(
            page_extractions=[page1_extraction, page2_extraction],
            page_orders=[page1_order, page2_order],
            schema_info=sample_schema_info,
        )

        assert result.total_pages == 2
        assert result.total_regions_processed == 2


class TestSynthesisServiceIntegration:
    """Integration tests for the synthesis service."""

    def test_full_synthesis_pipeline(
        self,
        service: SynthesisService,
        sample_region_extraction: DocumentRegionExtractionResult,
        sample_reading_order: DocumentReadingOrder,
        sample_schema_info: SchemaInfo,
        mock_synthesis_response: str,
    ) -> None:
        """Test complete synthesis pipeline."""
        mock_response = MagicMock()
        mock_response.content = mock_synthesis_response
        mock_response.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        result = service.synthesize(
            sample_region_extraction, sample_reading_order, sample_schema_info
        )

        # Check all components are properly built
        assert result.extracted_data is not None
        assert len(result.field_extractions) > 0
        assert len(result.source_references) == 3
        assert len(result.synthesized_contents) == 3
        assert result.pipeline_metadata is not None

        # Check conversion to ExtractionResult
        extraction_result = result.to_extraction_result()
        assert extraction_result.extracted_data == result.extracted_data
        assert extraction_result.model_used == result.model_used

        # Check dict representation
        data = result.to_dict()
        assert "extracted_data" in data
        assert "field_extractions" in data
        assert "source_references" in data
        assert "metadata" in data

    def test_synthesis_with_complex_schema(
        self,
        service: SynthesisService,
        sample_region_extraction: DocumentRegionExtractionResult,
        sample_reading_order: DocumentReadingOrder,
    ) -> None:
        """Test synthesis with complex nested schema."""
        # Create complex schema
        schema = {
            "type": "object",
            "properties": {
                "document": {
                    "type": "object",
                    "properties": {
                        "header": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "date": {"type": "string"},
                            },
                        },
                        "body": {
                            "type": "object",
                            "properties": {
                                "sections": {"type": "array"},
                            },
                        },
                    },
                },
            },
            "required": ["document"],
        }

        nested_field = FieldInfo(
            name="header",
            field_type="object",
            required=False,
            path="document.header",
            nested_fields=[
                FieldInfo(
                    name="title",
                    field_type="string",
                    required=False,
                    path="document.header.title",
                ),
            ],
        )

        schema_info = SchemaInfo(
            schema=schema,
            required_fields=[
                FieldInfo(
                    name="document",
                    field_type="object",
                    required=True,
                    path="document",
                    nested_fields=[nested_field],
                )
            ],
            optional_fields=[],
            schema_type="object",
        )

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "extracted_data": {
                    "document": {
                        "header": {"title": "Report", "date": "2024"},
                        "body": {"sections": []},
                    }
                },
                "overall_confidence": 0.9,
            }
        )
        mock_response.usage_metadata = {}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        service._llm = mock_llm

        result = service.synthesize(
            sample_region_extraction, sample_reading_order, schema_info
        )

        assert result.extracted_data["document"]["header"]["title"] == "Report"
