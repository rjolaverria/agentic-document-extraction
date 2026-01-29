"""Tests for the AnalyzeDiagram tool."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException
from PIL import Image, ImageDraw

from agentic_document_extraction.agents.tools.analyze_diagram import (
    AnalyzeDiagram,
    _normalize_connections,
    _normalize_id_list,
    _normalize_nodes,
    analyze_diagram_impl,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionBoundingBox,
    RegionImage,
    RegionType,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
    "ADE_OPENAI_API_KEY", ""
)
SKIP_INTEGRATION = not OPENAI_API_KEY or os.environ.get("ADE_SKIP_REAL_API_TESTS")
INTEGRATION_SKIP_REASON = (
    "OpenAI API key not set (set OPENAI_API_KEY or ADE_OPENAI_API_KEY) "
    "or ADE_SKIP_REAL_API_TESTS is enabled"
)


def create_simple_flowchart_image() -> Image.Image:
    """Create a simple flowchart diagram image."""
    image = Image.new("RGB", (400, 500), color="white")
    draw = ImageDraw.Draw(image)

    # Start node (rounded rect approximated as rectangle)
    draw.rounded_rectangle((150, 20, 250, 60), radius=10, outline="black", width=2)
    draw.text((180, 30), "Start", fill="black")

    # Process node (rectangle)
    draw.rectangle((125, 100, 275, 150), outline="black", width=2)
    draw.text((145, 115), "Process Data", fill="black")

    # Decision node (diamond)
    draw.polygon([(200, 190), (280, 240), (200, 290), (120, 240)], outline="black")
    draw.text((165, 230), "Valid?", fill="black")

    # Yes branch - End node
    draw.rounded_rectangle((250, 330, 350, 370), radius=10, outline="black", width=2)
    draw.text((280, 340), "End", fill="black")

    # No branch - Error node
    draw.rectangle((50, 330, 150, 380), outline="black", width=2)
    draw.text((75, 345), "Error", fill="black")

    # Arrows
    # Start -> Process
    draw.line((200, 60, 200, 100), fill="black", width=2)
    draw.polygon([(195, 95), (200, 100), (205, 95)], fill="black")

    # Process -> Decision
    draw.line((200, 150, 200, 190), fill="black", width=2)
    draw.polygon([(195, 185), (200, 190), (205, 185)], fill="black")

    # Decision -> End (Yes)
    draw.line((240, 240, 300, 240), fill="black", width=2)
    draw.line((300, 240, 300, 330), fill="black", width=2)
    draw.polygon([(295, 325), (300, 330), (305, 325)], fill="black")
    draw.text((270, 220), "Yes", fill="green")

    # Decision -> Error (No)
    draw.line((160, 240, 100, 240), fill="black", width=2)
    draw.line((100, 240, 100, 330), fill="black", width=2)
    draw.polygon([(95, 325), (100, 330), (105, 325)], fill="black")
    draw.text((110, 220), "No", fill="red")

    return image


def create_network_diagram_image() -> Image.Image:
    """Create a simple network topology diagram."""
    image = Image.new("RGB", (500, 400), color="white")
    draw = ImageDraw.Draw(image)

    # Server (rectangle)
    draw.rectangle((200, 20, 300, 80), outline="blue", width=2)
    draw.text((220, 40), "Server", fill="blue")

    # Router (circle)
    draw.ellipse((220, 150, 280, 210), outline="green", width=2)
    draw.text((232, 170), "Router", fill="green")

    # Client 1 (rectangle)
    draw.rectangle((50, 300, 150, 360), outline="black", width=2)
    draw.text((70, 320), "Client A", fill="black")

    # Client 2 (rectangle)
    draw.rectangle((200, 300, 300, 360), outline="black", width=2)
    draw.text((220, 320), "Client B", fill="black")

    # Client 3 (rectangle)
    draw.rectangle((350, 300, 450, 360), outline="black", width=2)
    draw.text((370, 320), "Client C", fill="black")

    # Connections
    # Server <-> Router
    draw.line((250, 80, 250, 150), fill="gray", width=2)
    draw.text((255, 110), "HTTPS", fill="gray")

    # Router <-> Clients
    draw.line((235, 210, 100, 300), fill="gray", width=2)
    draw.line((250, 210, 250, 300), fill="gray", width=2)
    draw.line((265, 210, 400, 300), fill="gray", width=2)

    return image


def create_org_chart_image() -> Image.Image:
    """Create a simple organizational chart."""
    image = Image.new("RGB", (500, 350), color="white")
    draw = ImageDraw.Draw(image)

    # CEO (top)
    draw.rectangle((200, 20, 300, 70), outline="navy", width=2)
    draw.text((230, 35), "CEO", fill="navy")

    # Level 2 - VP Engineering, VP Sales
    draw.rectangle((80, 120, 180, 170), outline="navy", width=2)
    draw.text((93, 135), "VP Eng", fill="navy")

    draw.rectangle((320, 120, 420, 170), outline="navy", width=2)
    draw.text((335, 135), "VP Sales", fill="navy")

    # Level 3 - Team Leads
    draw.rectangle((30, 220, 130, 270), outline="black", width=2)
    draw.text((48, 235), "Team A", fill="black")

    draw.rectangle((140, 220, 240, 270), outline="black", width=2)
    draw.text((158, 235), "Team B", fill="black")

    draw.rectangle((270, 220, 370, 270), outline="black", width=2)
    draw.text((285, 235), "Sales 1", fill="black")

    draw.rectangle((380, 220, 480, 270), outline="black", width=2)
    draw.text((395, 235), "Sales 2", fill="black")

    # Hierarchical lines
    draw.line((250, 70, 250, 90), fill="navy", width=2)
    draw.line((130, 90, 370, 90), fill="navy", width=2)
    draw.line((130, 90, 130, 120), fill="navy", width=2)
    draw.line((370, 90, 370, 120), fill="navy", width=2)

    draw.line((130, 170, 130, 190), fill="black", width=2)
    draw.line((80, 190, 190, 190), fill="black", width=2)
    draw.line((80, 190, 80, 220), fill="black", width=2)
    draw.line((190, 190, 190, 220), fill="black", width=2)

    draw.line((370, 170, 370, 190), fill="black", width=2)
    draw.line((320, 190, 430, 190), fill="black", width=2)
    draw.line((320, 190, 320, 220), fill="black", width=2)
    draw.line((430, 190, 430, 220), fill="black", width=2)

    return image


def create_er_diagram_image() -> Image.Image:
    """Create a simple entity-relationship diagram."""
    image = Image.new("RGB", (500, 300), color="white")
    draw = ImageDraw.Draw(image)

    # User entity
    draw.rectangle((50, 100, 150, 180), outline="black", width=2)
    draw.rectangle((50, 100, 150, 130), fill="lightblue", outline="black", width=2)
    draw.text((80, 105), "User", fill="black")
    draw.text((60, 140), "user_id", fill="black")
    draw.text((60, 160), "name", fill="black")

    # Order entity
    draw.rectangle((350, 100, 450, 180), outline="black", width=2)
    draw.rectangle((350, 100, 450, 130), fill="lightblue", outline="black", width=2)
    draw.text((380, 105), "Order", fill="black")
    draw.text((360, 140), "order_id", fill="black")
    draw.text((360, 160), "total", fill="black")

    # Relationship diamond
    draw.polygon([(250, 115), (290, 140), (250, 165), (210, 140)], outline="black")
    draw.text((230, 130), "has", fill="black")

    # Connection lines
    draw.line((150, 140, 210, 140), fill="black", width=2)
    draw.line((290, 140, 350, 140), fill="black", width=2)
    draw.text((165, 125), "1", fill="black")
    draw.text((320, 125), "N", fill="black")

    return image


def create_regions(
    image: Image.Image,
    *,
    region_id: str = "region-1",
    region_type: RegionType = RegionType.PICTURE,
) -> list[LayoutRegion]:
    """Create layout regions for testing."""
    bbox = RegionBoundingBox(x0=0, y0=0, x1=image.width, y1=image.height)
    return [
        LayoutRegion(
            region_type=region_type,
            bbox=bbox,
            confidence=0.95,
            page_number=1,
            region_id=region_id,
            region_image=RegionImage(image=image),
        )
    ]


class TestAnalyzeDiagramTool:
    """Tests for the AnalyzeDiagram tool function."""

    def test_analyze_diagram_parses_flowchart_response(self) -> None:
        """Test parsing flowchart diagram response."""
        response_payload: dict[str, Any] = {
            "diagram_type": "flowchart",
            "title": "Data Processing Flow",
            "nodes": [
                {
                    "id": "A",
                    "label": "Start",
                    "node_type": "start",
                    "description": None,
                },
                {
                    "id": "B",
                    "label": "Process Data",
                    "node_type": "process",
                    "description": "Main processing step",
                },
                {
                    "id": "C",
                    "label": "Valid?",
                    "node_type": "decision",
                    "description": "Validation check",
                },
                {"id": "D", "label": "End", "node_type": "end", "description": None},
                {
                    "id": "E",
                    "label": "Error",
                    "node_type": "process",
                    "description": "Error handling",
                },
            ],
            "connections": [
                {
                    "from_node": "A",
                    "to_node": "B",
                    "label": None,
                    "connection_type": "directed",
                },
                {
                    "from_node": "B",
                    "to_node": "C",
                    "label": None,
                    "connection_type": "directed",
                },
                {
                    "from_node": "C",
                    "to_node": "D",
                    "label": "Yes",
                    "connection_type": "directed",
                },
                {
                    "from_node": "C",
                    "to_node": "E",
                    "label": "No",
                    "connection_type": "directed",
                },
            ],
            "flow_sequence": ["A", "B", "C", "D"],
            "decision_points": ["C"],
            "description": "A simple data processing flowchart with validation.",
            "notes": None,
        }

        regions = create_regions(create_simple_flowchart_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeDiagram.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["diagram_type"] == "flowchart"
        assert result["title"] == "Data Processing Flow"
        assert len(result["nodes"]) == 5
        assert len(result["connections"]) == 4
        assert result["flow_sequence"] == ["A", "B", "C", "D"]
        assert result["decision_points"] == ["C"]

    def test_analyze_diagram_parses_network_response(self) -> None:
        """Test parsing network diagram response."""
        response_payload: dict[str, Any] = {
            "diagram_type": "network",
            "title": "Network Topology",
            "nodes": [
                {
                    "id": "1",
                    "label": "Server",
                    "node_type": "component",
                    "description": "Main server",
                },
                {
                    "id": "2",
                    "label": "Router",
                    "node_type": "component",
                    "description": "Network router",
                },
                {
                    "id": "3",
                    "label": "Client A",
                    "node_type": "actor",
                    "description": None,
                },
                {
                    "id": "4",
                    "label": "Client B",
                    "node_type": "actor",
                    "description": None,
                },
            ],
            "connections": [
                {
                    "from_node": "1",
                    "to_node": "2",
                    "label": "HTTPS",
                    "connection_type": "bidirectional",
                },
                {
                    "from_node": "2",
                    "to_node": "3",
                    "label": None,
                    "connection_type": "bidirectional",
                },
                {
                    "from_node": "2",
                    "to_node": "4",
                    "label": None,
                    "connection_type": "bidirectional",
                },
            ],
            "flow_sequence": None,
            "decision_points": None,
            "description": "Network topology showing server-router-client connections.",
            "notes": None,
        }

        regions = create_regions(create_network_diagram_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeDiagram.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["diagram_type"] == "network"
        assert len(result["nodes"]) == 4
        assert len(result["connections"]) == 3
        assert result["connections"][0]["connection_type"] == "bidirectional"
        assert result["connections"][0]["label"] == "HTTPS"

    def test_analyze_diagram_parses_org_chart_response(self) -> None:
        """Test parsing organizational chart response."""
        response_payload: dict[str, Any] = {
            "diagram_type": "org_chart",
            "title": "Organization Structure",
            "nodes": [
                {
                    "id": "CEO",
                    "label": "CEO",
                    "node_type": "actor",
                    "description": "Chief Executive",
                },
                {
                    "id": "VPE",
                    "label": "VP Eng",
                    "node_type": "actor",
                    "description": None,
                },
                {
                    "id": "VPS",
                    "label": "VP Sales",
                    "node_type": "actor",
                    "description": None,
                },
            ],
            "connections": [
                {
                    "from_node": "CEO",
                    "to_node": "VPE",
                    "label": None,
                    "connection_type": "hierarchical",
                },
                {
                    "from_node": "CEO",
                    "to_node": "VPS",
                    "label": None,
                    "connection_type": "hierarchical",
                },
            ],
            "flow_sequence": None,
            "decision_points": None,
            "description": "Company organizational hierarchy.",
            "notes": None,
        }

        regions = create_regions(create_org_chart_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeDiagram.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["diagram_type"] == "org_chart"
        assert len(result["nodes"]) == 3
        assert result["connections"][0]["connection_type"] == "hierarchical"

    def test_analyze_diagram_parses_er_diagram_response(self) -> None:
        """Test parsing ER diagram response."""
        response_payload: dict[str, Any] = {
            "diagram_type": "er_diagram",
            "title": "User-Order Relationship",
            "nodes": [
                {
                    "id": "User",
                    "label": "User",
                    "node_type": "entity",
                    "description": "User entity",
                },
                {
                    "id": "Order",
                    "label": "Order",
                    "node_type": "entity",
                    "description": "Order entity",
                },
            ],
            "connections": [
                {
                    "from_node": "User",
                    "to_node": "Order",
                    "label": "has (1:N)",
                    "connection_type": "directed",
                },
            ],
            "flow_sequence": None,
            "decision_points": None,
            "description": "One-to-many relationship between users and orders.",
            "notes": None,
        }

        regions = create_regions(create_er_diagram_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeDiagram.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["diagram_type"] == "er_diagram"
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["node_type"] == "entity"
        assert "1:N" in result["connections"][0]["label"]

    def test_analyze_diagram_handles_wrapped_json(self) -> None:
        """Test handling of wrapped JSON in VLM response."""
        wrapped = (
            "Here is the diagram analysis:\n"
            + json.dumps(
                {
                    "diagram_type": "flowchart",
                    "title": "Test Flow",
                    "nodes": [{"id": "A", "label": "Start", "node_type": "start"}],
                    "connections": [],
                    "description": "Simple test",
                }
            )
            + "\nEnd of analysis."
        )
        regions = create_regions(create_simple_flowchart_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
            return_value=wrapped,
        ):
            result = AnalyzeDiagram.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["diagram_type"] == "flowchart"
        assert result["title"] == "Test Flow"
        assert len(result["nodes"]) == 1

    def test_analyze_diagram_invalid_region_id(self) -> None:
        """Test error handling for invalid region ID."""
        regions = create_regions(create_simple_flowchart_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            AnalyzeDiagram.invoke({"region_id": "nonexistent", "regions": regions})

    def test_analyze_diagram_vlm_failure(self) -> None:
        """Test error handling when VLM call fails."""
        regions = create_regions(create_simple_flowchart_image())
        with (
            patch(
                "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
                side_effect=RuntimeError("VLM service unavailable"),
            ),
            pytest.raises(ToolException, match="AnalyzeDiagram VLM call failed"),
        ):
            AnalyzeDiagram.invoke({"region_id": "region-1", "regions": regions})

    def test_analyze_diagram_adds_note_for_unexpected_region_type(self) -> None:
        """Test that a note is added for unexpected region types."""
        response_payload: dict[str, Any] = {
            "diagram_type": "flowchart",
            "nodes": [{"id": "A", "label": "Start", "node_type": "start"}],
            "connections": [],
            "description": "Test diagram",
        }

        regions = create_regions(
            create_simple_flowchart_image(),
            region_type=RegionType.TEXT,  # Unexpected type for a diagram
        )
        with patch(
            "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeDiagram.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["notes"] is not None
        assert "text" in result["notes"].lower()

    def test_analyze_diagram_normalizes_invalid_diagram_type(self) -> None:
        """Test that invalid diagram types default to 'other'."""
        response_payload: dict[str, Any] = {
            "diagram_type": "invalid_type",
            "nodes": [],
            "connections": [],
            "description": "Test",
        }

        regions = create_regions(create_simple_flowchart_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = AnalyzeDiagram.invoke(
                {"region_id": "region-1", "regions": regions}
            )

        assert result["diagram_type"] == "other"

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_diagram_integration_flowchart(self) -> None:
        """Integration test with a flowchart diagram image."""
        regions = create_regions(create_simple_flowchart_image())
        result = AnalyzeDiagram.invoke({"region_id": "region-1", "regions": regions})

        assert "diagram_type" in result
        assert "nodes" in result
        assert isinstance(result["nodes"], list)

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_diagram_integration_network(self) -> None:
        """Integration test with a network diagram image."""
        regions = create_regions(create_network_diagram_image())
        result = AnalyzeDiagram.invoke({"region_id": "region-1", "regions": regions})

        assert "diagram_type" in result
        assert "nodes" in result
        assert isinstance(result["nodes"], list)

    @pytest.mark.skipif(SKIP_INTEGRATION, reason=INTEGRATION_SKIP_REASON)
    def test_analyze_diagram_integration_org_chart(self) -> None:
        """Integration test with an org chart image."""
        regions = create_regions(create_org_chart_image())
        result = AnalyzeDiagram.invoke({"region_id": "region-1", "regions": regions})

        assert "diagram_type" in result
        assert "nodes" in result


class TestAnalyzeDiagramImpl:
    """Tests for the analyze_diagram_impl helper function."""

    def test_impl_parses_response(self) -> None:
        """Test that impl function parses VLM response correctly."""
        response_payload: dict[str, Any] = {
            "diagram_type": "flowchart",
            "title": "Test Flow",
            "nodes": [
                {"id": "A", "label": "Start", "node_type": "start"},
                {"id": "B", "label": "End", "node_type": "end"},
            ],
            "connections": [
                {"from_node": "A", "to_node": "B", "connection_type": "directed"},
            ],
            "flow_sequence": ["A", "B"],
            "decision_points": None,
            "description": "Simple flow from start to end",
            "notes": "Test notes",
        }
        regions = create_regions(create_simple_flowchart_image())
        with patch(
            "agentic_document_extraction.agents.tools.analyze_diagram.call_vlm_with_image",
            return_value=json.dumps(response_payload),
        ):
            result = analyze_diagram_impl("region-1", regions)

        assert result["diagram_type"] == "flowchart"
        assert result["title"] == "Test Flow"
        assert len(result["nodes"]) == 2
        assert len(result["connections"]) == 1
        assert result["flow_sequence"] == ["A", "B"]
        assert result["notes"] == "Test notes"

    def test_impl_raises_on_missing_region(self) -> None:
        """Test error when region ID not found."""
        regions = create_regions(create_simple_flowchart_image())
        with pytest.raises(ToolException, match="Unknown region_id"):
            analyze_diagram_impl("nonexistent", regions)

    def test_impl_raises_on_missing_image(self) -> None:
        """Test error when region has no image."""
        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.PICTURE,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="no-img",
                region_image=None,
            )
        ]
        with pytest.raises(ToolException, match="Region image not provided"):
            analyze_diagram_impl("no-img", regions)

    def test_impl_raises_on_empty_image_data(self) -> None:
        """Test error when region image has no data."""
        bbox = RegionBoundingBox(x0=0, y0=0, x1=100, y1=100)
        regions = [
            LayoutRegion(
                region_type=RegionType.PICTURE,
                bbox=bbox,
                confidence=0.9,
                page_number=1,
                region_id="empty-img",
                region_image=RegionImage(image=None, base64=None),
            )
        ]
        with pytest.raises(ToolException, match="Region image missing"):
            analyze_diagram_impl("empty-img", regions)


class TestNormalizeNodes:
    """Tests for the _normalize_nodes helper function."""

    def test_normalize_empty_list(self) -> None:
        """Test normalization of empty node list."""
        result = _normalize_nodes([])
        assert result == []

    def test_normalize_invalid_input(self) -> None:
        """Test normalization of invalid input types."""
        assert _normalize_nodes(None) == []
        assert _normalize_nodes("not a list") == []
        assert _normalize_nodes(123) == []

    def test_normalize_filters_invalid_items(self) -> None:
        """Test that non-dict items are filtered out."""
        nodes = [
            {"id": "A", "label": "Valid"},
            "invalid string",
            123,
            None,
        ]
        result = _normalize_nodes(nodes)
        assert len(result) == 1
        assert result[0]["id"] == "A"

    def test_normalize_skips_nodes_without_id(self) -> None:
        """Test that nodes without IDs are skipped."""
        nodes = [
            {"label": "No ID"},  # Missing id
            {"id": "A", "label": "Has ID"},
            {"id": "", "label": "Empty ID"},  # Empty id
        ]
        result = _normalize_nodes(nodes)
        assert len(result) == 1
        assert result[0]["id"] == "A"

    def test_normalize_provides_defaults(self) -> None:
        """Test that missing fields get default values."""
        nodes = [{"id": "A"}]  # Minimal node
        result = _normalize_nodes(nodes)

        assert result[0]["id"] == "A"
        assert result[0]["label"] is None
        assert result[0]["node_type"] == "other"
        assert result[0]["description"] is None

    def test_normalize_invalid_node_type(self) -> None:
        """Test that invalid node types default to 'other'."""
        nodes = [{"id": "A", "node_type": "invalid_type"}]
        result = _normalize_nodes(nodes)
        assert result[0]["node_type"] == "other"

    def test_normalize_valid_node_types(self) -> None:
        """Test that all valid node types are accepted."""
        valid_types = [
            "start",
            "end",
            "process",
            "decision",
            "data",
            "entity",
            "component",
            "actor",
            "other",
        ]
        for node_type in valid_types:
            nodes = [{"id": "A", "node_type": node_type}]
            result = _normalize_nodes(nodes)
            assert result[0]["node_type"] == node_type

    def test_normalize_strips_description_whitespace(self) -> None:
        """Test that description whitespace is stripped."""
        nodes = [{"id": "A", "description": "  Some description  "}]
        result = _normalize_nodes(nodes)
        assert result[0]["description"] == "Some description"

    def test_normalize_empty_description_becomes_none(self) -> None:
        """Test that empty description becomes None."""
        nodes = [{"id": "A", "description": "   "}]
        result = _normalize_nodes(nodes)
        assert result[0]["description"] is None


class TestNormalizeConnections:
    """Tests for the _normalize_connections helper function."""

    def test_normalize_empty_list(self) -> None:
        """Test normalization of empty connection list."""
        result = _normalize_connections([])
        assert result == []

    def test_normalize_invalid_input(self) -> None:
        """Test normalization of invalid input types."""
        assert _normalize_connections(None) == []
        assert _normalize_connections("not a list") == []
        assert _normalize_connections(123) == []

    def test_normalize_filters_invalid_items(self) -> None:
        """Test that non-dict items are filtered out."""
        connections = [
            {"from_node": "A", "to_node": "B"},
            "invalid string",
            123,
            None,
        ]
        result = _normalize_connections(connections)
        assert len(result) == 1
        assert result[0]["from_node"] == "A"

    def test_normalize_skips_incomplete_connections(self) -> None:
        """Test that connections without from_node or to_node are skipped."""
        connections = [
            {"to_node": "B"},  # Missing from_node
            {"from_node": "A"},  # Missing to_node
            {"from_node": "A", "to_node": "B"},  # Complete
            {"from_node": "", "to_node": "B"},  # Empty from_node
        ]
        result = _normalize_connections(connections)
        assert len(result) == 1
        assert result[0]["from_node"] == "A"
        assert result[0]["to_node"] == "B"

    def test_normalize_provides_defaults(self) -> None:
        """Test that missing fields get default values."""
        connections = [{"from_node": "A", "to_node": "B"}]  # Minimal connection
        result = _normalize_connections(connections)

        assert result[0]["from_node"] == "A"
        assert result[0]["to_node"] == "B"
        assert result[0]["label"] is None
        assert result[0]["connection_type"] == "directed"

    def test_normalize_invalid_connection_type(self) -> None:
        """Test that invalid connection types default to 'directed'."""
        connections = [
            {"from_node": "A", "to_node": "B", "connection_type": "invalid_type"}
        ]
        result = _normalize_connections(connections)
        assert result[0]["connection_type"] == "directed"

    def test_normalize_valid_connection_types(self) -> None:
        """Test that all valid connection types are accepted."""
        valid_types = ["directed", "bidirectional", "hierarchical"]
        for conn_type in valid_types:
            connections = [
                {"from_node": "A", "to_node": "B", "connection_type": conn_type}
            ]
            result = _normalize_connections(connections)
            assert result[0]["connection_type"] == conn_type

    def test_normalize_strips_label_whitespace(self) -> None:
        """Test that label whitespace is stripped."""
        connections = [{"from_node": "A", "to_node": "B", "label": "  Yes  "}]
        result = _normalize_connections(connections)
        assert result[0]["label"] == "Yes"

    def test_normalize_empty_label_becomes_none(self) -> None:
        """Test that empty label becomes None."""
        connections = [{"from_node": "A", "to_node": "B", "label": "   "}]
        result = _normalize_connections(connections)
        assert result[0]["label"] is None


class TestNormalizeIdList:
    """Tests for the _normalize_id_list helper function."""

    def test_normalize_valid_list(self) -> None:
        """Test normalization of valid ID list."""
        result = _normalize_id_list(["A", "B", "C"])
        assert result == ["A", "B", "C"]

    def test_normalize_empty_list(self) -> None:
        """Test normalization of empty list returns None."""
        result = _normalize_id_list([])
        assert result is None

    def test_normalize_invalid_input(self) -> None:
        """Test normalization of invalid input types."""
        assert _normalize_id_list(None) is None
        assert _normalize_id_list("not a list") is None
        assert _normalize_id_list(123) is None

    def test_normalize_filters_invalid_items(self) -> None:
        """Test that None and empty string items are filtered."""
        result = _normalize_id_list(["A", None, "", "B", "  ", "C"])
        assert result == ["A", "B", "C"]

    def test_normalize_converts_to_strings(self) -> None:
        """Test that non-string IDs are converted to strings."""
        result = _normalize_id_list([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_normalize_all_invalid_returns_none(self) -> None:
        """Test that list with only invalid items returns None."""
        result = _normalize_id_list([None, "", "  "])
        assert result is None
