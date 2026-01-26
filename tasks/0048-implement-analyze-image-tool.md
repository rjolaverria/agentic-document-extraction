# Task 0048: Implement AnalyzeImage Tool for LangChain Agent

## Objective
Create a LangChain tool that allows the extraction agent to analyze embedded images for object detection, counting, condition assessment, and contextual understanding.

## Context
Many documents contain images that provide critical information:
- Product photos in catalogs or receipts
- Damage assessment photos in insurance claims
- Inventory items with visual details
- Equipment condition documentation
- Property photos in real estate
- Medical imaging in health records

The AnalyzeImage tool enables the agent to:
- Identify and describe objects in images
- Count items
- Assess condition (new, damaged, worn, etc.)
- Extract visual attributes (color, size, brand)
- Provide contextual descriptions

## Acceptance Criteria
- [ ] Create `AnalyzeImageTool` class implementing LangChain `BaseTool`
- [ ] Tool accepts region_id parameter and optional query/focus
- [ ] Crops image region using bounding boxes from layout detection
- [ ] Sends cropped image to GPT-4V with image analysis prompt
- [ ] Returns structured output with:
  - Image description
  - Detected objects with counts
  - Object attributes (color, brand, condition, etc.)
  - Condition assessment
  - Relevant extracted text (product names, labels)
- [ ] Tool description clearly explains when to use (for embedded image regions)
- [ ] Integration with PaddleOCR layout detection results
- [ ] Unit tests for tool functionality
- [ ] Integration tests with sample documents containing images
- [ ] Error handling for invalid region IDs or failed VLM calls

## Tool Schema
```python
class AnalyzeImageInput(BaseModel):
    region_id: str = Field(description="ID of the image region to analyze")
    focus: str | None = Field(
        default=None,
        description="Optional: what to focus on (e.g., 'count items', 'assess damage', 'identify products')"
    )

class ImageObject(BaseModel):
    object_type: str  # "product", "item", "damage", "equipment", etc.
    description: str
    count: int | None  # How many of this object
    attributes: dict[str, Any] | None  # color, brand, size, condition, etc.
    confidence: str  # "high", "medium", "low"

class AnalyzeImageOutput(BaseModel):
    description: str  # Overall image description
    objects: list[ImageObject]
    total_items: int | None  # Total count if applicable
    condition_assessment: str | None  # "excellent", "good", "fair", "poor", "damaged"
    extracted_text: list[str] | None  # Any visible text (labels, signs, product names)
    notes: str | None  # Additional observations
```

## Dependencies
- PaddleOCR layout detection with region bounding boxes
- GPT-4V API access (already configured)
- Image cropping utilities (shared with other tools)

## Implementation Notes
- Allow optional focus parameter to guide analysis (e.g., "count items", "assess damage")
- Handle multiple objects in one image
- Extract visible text from labels, signs, packaging
- Provide attribute extraction for common domains (products, equipment, etc.)
- Consider image quality assessment
- May need to distinguish between image region and chart/diagram regions

## VLM Prompt Strategy
```
You are analyzing an embedded image in a document.

IMAGE: [cropped image region]

FOCUS: {focus or "general analysis"}

TASK:
1. Describe what the image shows
2. Identify and count distinct objects/items
3. For each object, extract:
   - Type/category
   - Description
   - Quantity
   - Visible attributes (color, brand, size, condition, model, etc.)
4. Assess overall condition if relevant (new, used, damaged, etc.)
5. Extract any visible text (product names, labels, signs)

Be specific and factual. If counting items, be precise.

Return structured information about the image contents.
```

## Testing Strategy
- Create test fixtures with various image types:
  - Product photos (single and multiple items)
  - Damage assessment photos
  - Equipment/machinery photos
  - Inventory item photos
  - Receipt line item images
  - Property photos
  - Medical images (if applicable)
- Mock VLM responses for unit tests
- Integration tests with real GPT-4V calls
- Test accuracy: object detection, counting, attribute extraction
- Test edge cases: low quality images, occluded objects, complex scenes

## Use Cases
- E-commerce product catalogs
- Insurance claim assessment
- Inventory documentation
- Equipment condition reports
- Property inspection reports
- Receipt line item extraction
- Quality control documentation
- Asset management
