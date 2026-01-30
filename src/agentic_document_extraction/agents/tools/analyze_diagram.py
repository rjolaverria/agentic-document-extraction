"""LangChain tool for analyzing diagram regions using a VLM."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import InjectedState

from agentic_document_extraction.agents.tools.vlm_utils import (
    call_vlm_with_image,
    encode_image_to_base64,
    parse_json_response,
)
from agentic_document_extraction.services.layout_detector import (
    LayoutRegion,
    RegionType,
)

logger = logging.getLogger(__name__)


DIAGRAM_ANALYSIS_PROMPT = """You are a Diagram Analysis specialist.
Analyze this diagram image and extract its structural and relational information.

**ANALYSIS STEPS:**

1. **DIAGRAM TYPE** - Identify the type:
   - flowchart: Process flows with decision points
   - network: Network topology, system connections
   - architecture: System/software architecture
   - org_chart: Organizational hierarchy
   - sequence: Sequence/timing diagrams
   - er_diagram: Entity-relationship diagrams
   - state_diagram: State machine diagrams
   - other: Other diagram types

2. **NODES/COMPONENTS** - For each node:
   - Assign a unique ID (A, B, C... or 1, 2, 3... based on reading order)
   - Extract the label/text inside the node
   - Identify the node type:
     * start: Entry point (rounded rect, circle with "Start")
     * end: Exit point (rounded rect, circle with "End")
     * process: Standard process step (rectangle)
     * decision: Decision/branch point (diamond)
     * data: Data input/output (parallelogram)
     * entity: Database entity (rectangle in ER diagrams)
     * component: System component
     * actor: Person/system actor
     * other: Other node types

3. **CONNECTIONS** - For each arrow/line:
   - Source node ID
   - Target node ID
   - Connection label (if any, e.g., "Yes", "No", "HTTP", "sends data")
   - Connection type: "directed" (arrow), "bidirectional" (double arrow), "hierarchical" (parent-child)

4. **FLOW ANALYSIS**:
   - Main flow sequence: List node IDs in order of main process flow
   - Decision points: List node IDs that represent decision/branch points

5. **DESCRIPTION**: Summarize the diagram's purpose and key insights.

**OUTPUT FORMAT** - Return a JSON object:
```json
{
  "diagram_type": "flowchart",
  "title": "Order Processing Flow",
  "nodes": [
    {
      "id": "A",
      "label": "Start",
      "node_type": "start",
      "description": "Entry point of the process"
    },
    {
      "id": "B",
      "label": "Receive Order",
      "node_type": "process",
      "description": null
    },
    {
      "id": "C",
      "label": "In Stock?",
      "node_type": "decision",
      "description": "Check inventory availability"
    }
  ],
  "connections": [
    {
      "from_node": "A",
      "to_node": "B",
      "label": null,
      "connection_type": "directed"
    },
    {
      "from_node": "C",
      "to_node": "D",
      "label": "Yes",
      "connection_type": "directed"
    },
    {
      "from_node": "C",
      "to_node": "E",
      "label": "No",
      "connection_type": "directed"
    }
  ],
  "flow_sequence": ["A", "B", "C", "D", "F"],
  "decision_points": ["C"],
  "description": "This flowchart shows the order processing workflow...",
  "notes": "The diagram has a single decision point for inventory check."
}
```

Analyze the diagram thoroughly and return accurate structural information.
"""

DIAGRAM_DEFAULT_RESPONSE: dict[str, Any] = {
    "diagram_type": "other",
    "title": None,
    "nodes": [],
    "connections": [],
    "flow_sequence": None,
    "decision_points": None,
    "description": "Unable to analyze diagram structure.",
    "notes": None,
}


def analyze_diagram_impl(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Core diagram analysis logic shared by the tool and direct callers.

    Args:
        region_id: ID of the region to analyze.
        regions: List of layout regions with images.

    Returns:
        Parsed diagram analysis result dict with nodes, connections, and flow info.

    Raises:
        ToolException: When the region or its image cannot be found.
    """
    region = next((r for r in regions if r.region_id == region_id), None)
    if region is None:
        raise ToolException(f"Unknown region_id: {region_id}")

    region_image = region.region_image
    if region_image is None:
        raise ToolException(f"Region image not provided for region_id: {region_id}")

    if region_image.base64:
        image_base64 = region_image.base64
    elif region_image.image is not None:
        image_base64 = encode_image_to_base64(region_image.image)
    else:
        raise ToolException(f"Region image missing image/base64 for {region_id}")

    try:
        response_text = call_vlm_with_image(image_base64, DIAGRAM_ANALYSIS_PROMPT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("AnalyzeDiagram VLM call failed: %s", exc)
        raise ToolException("AnalyzeDiagram VLM call failed") from exc

    result = parse_json_response(
        response_text,
        default=dict(DIAGRAM_DEFAULT_RESPONSE),
        tool_name="AnalyzeDiagram",
    )

    # Validate and normalize nodes and connections
    result["nodes"] = _normalize_nodes(result.get("nodes", []))
    result["connections"] = _normalize_connections(result.get("connections", []))
    result["flow_sequence"] = _normalize_id_list(result.get("flow_sequence"))
    result["decision_points"] = _normalize_id_list(result.get("decision_points"))

    # Validate diagram_type
    valid_diagram_types = {
        "flowchart",
        "network",
        "architecture",
        "org_chart",
        "sequence",
        "er_diagram",
        "state_diagram",
        "other",
    }
    if result.get("diagram_type") not in valid_diagram_types:
        result["diagram_type"] = "other"

    # Add note if region type doesn't match expected diagram types
    # Diagrams typically appear as PICTURE regions
    diagram_region_types = {RegionType.PICTURE}
    if region.region_type not in diagram_region_types:
        notes = result.get("notes")
        note = (
            f"Region type is {region.region_type.value}; may not be a diagram region."
        )
        result["notes"] = f"{notes} {note}".strip() if notes else note

    return result


def _normalize_nodes(nodes: Any) -> list[dict[str, Any]]:
    """Normalize and validate diagram nodes from VLM response.

    Args:
        nodes: Raw nodes list from VLM response.

    Returns:
        Normalized list of node dictionaries.
    """
    if not isinstance(nodes, list):
        return []

    valid_node_types = {
        "start",
        "end",
        "process",
        "decision",
        "data",
        "entity",
        "component",
        "actor",
        "other",
    }

    normalized = []
    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_id = node.get("id")
        if not node_id:
            continue  # Skip nodes without IDs

        normalized_node: dict[str, Any] = {
            "id": str(node_id),
            "label": str(node.get("label", "")) or None,
            "node_type": str(node.get("node_type", "other")).lower(),
            "description": node.get("description"),
        }

        # Validate node_type
        if normalized_node["node_type"] not in valid_node_types:
            normalized_node["node_type"] = "other"

        # Clean up description
        if isinstance(normalized_node["description"], str):
            normalized_node["description"] = (
                normalized_node["description"].strip() or None
            )
        elif normalized_node["description"] is not None:
            normalized_node["description"] = None

        normalized.append(normalized_node)

    return normalized


def _normalize_connections(connections: Any) -> list[dict[str, Any]]:
    """Normalize and validate diagram connections from VLM response.

    Args:
        connections: Raw connections list from VLM response.

    Returns:
        Normalized list of connection dictionaries.
    """
    if not isinstance(connections, list):
        return []

    valid_connection_types = {"directed", "bidirectional", "hierarchical"}

    normalized = []
    for conn in connections:
        if not isinstance(conn, dict):
            continue

        from_node = conn.get("from_node")
        to_node = conn.get("to_node")

        # Skip connections without source or target
        if not from_node or not to_node:
            continue

        normalized_conn: dict[str, Any] = {
            "from_node": str(from_node),
            "to_node": str(to_node),
            "label": conn.get("label"),
            "connection_type": str(conn.get("connection_type", "directed")).lower(),
        }

        # Validate connection_type
        if normalized_conn["connection_type"] not in valid_connection_types:
            normalized_conn["connection_type"] = "directed"

        # Clean up label
        if isinstance(normalized_conn["label"], str):
            normalized_conn["label"] = normalized_conn["label"].strip() or None
        elif normalized_conn["label"] is not None:
            normalized_conn["label"] = None

        normalized.append(normalized_conn)

    return normalized


def _normalize_id_list(ids: Any) -> list[str] | None:
    """Normalize a list of node IDs.

    Args:
        ids: Raw list of node IDs.

    Returns:
        Normalized list of string IDs, or None if empty/invalid.
    """
    if not isinstance(ids, list):
        return None

    normalized = [str(id_) for id_ in ids if id_ is not None and str(id_).strip()]
    return normalized if normalized else None


@tool("analyze_diagram")
def AnalyzeDiagram(region_id: str, regions: list[LayoutRegion]) -> dict[str, Any]:
    """Analyze diagram regions to extract nodes, connections, and flow structure."""
    return analyze_diagram_impl(region_id, regions)


@tool("analyze_diagram_agent")
def analyze_diagram(
    region_id: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> dict[str, Any]:
    """Extract structure and relationships from diagrams, flowcharts, and org charts.

    Use this tool when:
    - The region_type is PICTURE and contains a flowchart, process diagram, or org chart
    - You need to extract nodes, connections, flow direction, or hierarchy
    - The OCR text captures labels but not the visual relationships

    Do NOT use when:
    - The image is a data chart or graph (use analyze_chart_agent)
    - The image is a photo or illustration (use analyze_image_agent)

    Args:
        region_id: The ID from the Document Regions table (e.g., "region_5")

    Returns:
        JSON with diagram_type, nodes, connections, flow_sequence, decision_points
    """
    regions: list[LayoutRegion] = state.get("regions", [])
    return analyze_diagram_impl(region_id, regions)
