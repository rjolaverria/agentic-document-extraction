"""Tests for the tool-based extraction agent."""

from unittest.mock import MagicMock

from PIL import Image

from agentic_document_extraction.agents.extraction_agent import ExtractionAgent
from agentic_document_extraction.services.extraction.text_extraction import (
    ExtractionResult,
)
from agentic_document_extraction.services.extraction.visual_document_extraction import (
    VisualDocumentExtractionService,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionType,
)
from agentic_document_extraction.services.reading_order_detector import OrderedRegion
from agentic_document_extraction.services.schema_validator import FieldInfo, SchemaInfo


def build_schema_info() -> SchemaInfo:
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total": {"type": "number"},
        },
        "required": ["invoice_number"],
    }
    return SchemaInfo(
        schema=schema,
        required_fields=[
            FieldInfo(
                name="invoice_number",
                field_type="string",
                required=True,
                path="invoice_number",
                description="Invoice number",
            )
        ],
        optional_fields=[
            FieldInfo(
                name="total",
                field_type="number",
                required=False,
                path="total",
                description="Invoice total",
            )
        ],
        schema_type="object",
    )


def build_region() -> LayoutRegion:
    return LayoutRegion(
        region_type=RegionType.TABLE,
        bbox=RegionBoundingBox(x0=10, y0=20, x1=110, y1=220),
        confidence=0.9,
        page_number=1,
        region_id="region-1",
        metadata={"text": "Item | Price"},
    )


def test_extraction_agent_builds_tools() -> None:
    agent = ExtractionAgent(api_key="test-key")
    tools = agent._build_tools([build_region()])

    tool_names = {tool.name for tool in tools}
    assert tool_names == {"AnalyzeChart", "AnalyzeTable"}


def test_extraction_agent_prompt_contains_schema_and_regions() -> None:
    agent = ExtractionAgent(api_key="test-key")
    schema_info = build_schema_info()
    region = build_region()
    ordered_region = OrderedRegion(
        region=region, order_index=0, confidence=0.88, reasoning="Top-left"
    )

    prompt = agent._build_system_prompt(
        ordered_text="Invoice #1234\nTotal: $500",
        regions=[ordered_region.to_dict()],
        schema_info=schema_info,
    )

    assert "Invoice #1234" in prompt
    assert "region-1" in prompt
    assert '"invoice_number"' in prompt


def test_visual_document_extraction_uses_tool_agent() -> None:
    schema_info = build_schema_info()
    region = build_region()
    ordered_region = OrderedRegion(region=region, order_index=0, confidence=0.9)

    layout_result = MagicMock()
    layout_result.get_all_regions.return_value = [region]

    reading_order = MagicMock()
    reading_order.pages = [MagicMock(ordered_regions=[ordered_region])]

    tool_agent = MagicMock()
    tool_agent.extract.return_value = ExtractionResult(extracted_data={})

    service = VisualDocumentExtractionService(api_key="test-key")
    service._layout_detector = MagicMock(detect_from_images=lambda _: layout_result)
    service._reading_order_detector = MagicMock(
        detect_reading_order=lambda _: reading_order
    )
    service._tool_agent = tool_agent

    image = Image.new("RGB", (100, 100), color="white")
    result = service.extract(image_source=image, schema_info=schema_info)

    assert result.extracted_data == {}
    tool_agent.extract.assert_called_once_with(
        ordered_text="Item | Price",
        schema_info=schema_info,
        regions=[region],
        ordered_regions=[ordered_region],
    )
