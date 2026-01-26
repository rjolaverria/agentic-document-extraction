# Task 0046: Implement AnalyzeDiagram Tool for LangChain Agent

## Objective
Create a LangChain tool that allows the extraction agent to analyze flowcharts, process diagrams, network diagrams, and architectural drawings for structural and relational information.

## Context
Diagrams encode information through visual relationships that OCR cannot capture:
- Node connections and flow direction
- Process sequences and decision points
- Hierarchical relationships
- Network topologies
- System architectures

The AnalyzeDiagram tool enables the agent to:
- Identify diagram type
- Extract nodes/components and their labels
- Understand connections and flow direction
- Capture decision logic and branches
- Describe overall structure and purpose

## Acceptance Criteria
- [ ] Create `AnalyzeDiagramTool` class implementing LangChain `BaseTool`
- [ ] Tool accepts region_id parameter
- [ ] Crops image region using bounding boxes from layout detection
- [ ] Sends cropped image to GPT-4V with diagram analysis prompt
- [ ] Returns structured output with:
  - Diagram type (flowchart, network, architecture, org chart, etc.)
  - Nodes/components with labels and types
  - Connections/edges with direction and labels
  - Flow sequence or hierarchy
  - Decision points and branches
  - Overall purpose/description
- [ ] Tool description clearly explains when to use (for diagram regions)
- [ ] Integration with PaddleOCR layout detection results
- [ ] Unit tests for tool functionality
- [ ] Integration tests with sample diagrams
- [ ] Error handling for invalid region IDs or failed VLM calls

## Tool Schema
```python
class AnalyzeDiagramInput(BaseModel):
    region_id: str = Field(description="ID of the diagram region to analyze")

class DiagramNode(BaseModel):
    id: str  # Node identifier (e.g., "A", "1", "Start")
    label: str  # Text label in the node
    node_type: str | None  # "start", "end", "process", "decision", "data", etc.
    description: str | None  # Additional context

class DiagramConnection(BaseModel):
    from_node: str  # Source node ID
    to_node: str  # Target node ID
    label: str | None  # Connection label (e.g., "Yes", "No", "sends data")
    connection_type: str | None  # "directed", "bidirectional", "hierarchical"

class AnalyzeDiagramOutput(BaseModel):
    diagram_type: str  # "flowchart", "network", "architecture", "org_chart", "sequence", "er_diagram"
    title: str | None  # Diagram title if present
    nodes: list[DiagramNode]
    connections: list[DiagramConnection]
    flow_sequence: list[str] | None  # Ordered node IDs showing main flow
    decision_points: list[str] | None  # Node IDs of decision/branch points
    description: str  # Overall diagram purpose and key insights
    notes: str | None  # Additional observations
```

## Dependencies
- PaddleOCR layout detection with region bounding boxes
- GPT-4V API access (already configured)
- Image cropping utilities (shared with other tools)

## Implementation Notes
- Different diagram types require different analysis approaches
- Flowcharts: focus on process flow and decisions
- Network diagrams: focus on topology and connections
- Org charts: focus on hierarchy
- Architecture diagrams: focus on components and interactions
- Use consistent node ID scheme for referencing
- Handle complex diagrams by focusing on main flow first

## VLM Prompt Strategy
```
You are analyzing a diagram. Perform the following analysis:

1. DIAGRAM TYPE:
   Identify the type: flowchart, network diagram, system architecture, 
   organizational chart, sequence diagram, ER diagram, or other.

2. COMPONENTS:
   For each node/component:
   - Assign an ID (A, B, C... or 1, 2, 3...)
   - Extract the label/text
   - Identify the type (start, end, process, decision, entity, etc.)

3. CONNECTIONS:
   For each arrow/line:
   - Source node ID
   - Target node ID
   - Direction (one-way, two-way)
   - Label (if any, e.g., "Yes", "No", "HTTP")

4. FLOW:
   - What is the main flow or sequence?
   - What are the key decision points?

5. PURPOSE:
   - What does this diagram represent?
   - What are the key insights?

Return structured information about nodes, connections, and flow.
```

## Testing Strategy
- Create test fixtures with various diagram types:
  - Simple flowcharts (3-5 nodes)
  - Complex flowcharts with multiple decision points
  - Network topology diagrams
  - System architecture diagrams
  - Organizational charts
  - Sequence diagrams
- Mock VLM responses for unit tests
- Integration tests with real GPT-4V calls
- Test accuracy: node identification, connection detection, flow understanding
- Test edge cases: crossed lines, complex layouts, handdrawn diagrams

## Use Cases
- Process documentation
- Technical specifications
- Patent applications
- Network documentation
- System design documents
- Business process modeling
- Organizational structure documents
- Software architecture documentation
