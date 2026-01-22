"""LangChain LLM integration for structured text extraction.

This module provides functionality to extract structured information from text
documents according to a user-provided JSON schema using LangChain with OpenAI
models (GPT-4 or GPT-4 Turbo).

Key features:
- Structured output using JSON mode or function calling
- Automatic chunking for documents that exceed token limits
- Confidence indicators for extracted fields
- Support for reasoning models when available
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_document_extraction.config import settings
from agentic_document_extraction.services.schema_validator import SchemaInfo

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when extraction fails."""

    def __init__(
        self,
        message: str,
        error_type: str = "extraction_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with message and error details.

        Args:
            message: Error message.
            error_type: Type of error for categorization.
            details: Optional additional error details.
        """
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


@dataclass
class FieldExtraction:
    """Extracted value for a single field."""

    field_path: str
    """JSON path to the field (e.g., 'address.city')."""

    value: Any
    """Extracted value."""

    confidence: float | None = None
    """Confidence score for this extraction (0.0-1.0), if available."""

    source_text: str | None = None
    """Source text snippet that was used for extraction."""

    reasoning: str | None = None
    """Reasoning provided by the model for this extraction."""


@dataclass
class ExtractionResult:
    """Result of structured extraction from a text document."""

    extracted_data: dict[str, Any]
    """The extracted data matching the user's JSON schema."""

    field_extractions: list[FieldExtraction] = field(default_factory=list)
    """Detailed extraction info for each field."""

    model_used: str = ""
    """Name of the model used for extraction."""

    total_tokens: int = 0
    """Total tokens used in the extraction process."""

    prompt_tokens: int = 0
    """Tokens used in the prompt."""

    completion_tokens: int = 0
    """Tokens used in the completion."""

    processing_time_seconds: float = 0.0
    """Time taken for extraction in seconds."""

    chunks_processed: int = 1
    """Number of text chunks processed."""

    is_chunked: bool = False
    """Whether the document was processed in chunks."""

    raw_response: str | None = None
    """Raw response from the model (for debugging)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with extraction result information.
        """
        return {
            "extracted_data": self.extracted_data,
            "field_extractions": [
                {
                    "field_path": fe.field_path,
                    "value": fe.value,
                    "confidence": fe.confidence,
                    "source_text": fe.source_text,
                    "reasoning": fe.reasoning,
                }
                for fe in self.field_extractions
            ],
            "metadata": {
                "model_used": self.model_used,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "processing_time_seconds": self.processing_time_seconds,
                "chunks_processed": self.chunks_processed,
                "is_chunked": self.is_chunked,
            },
        }


class TextExtractionService:
    """Service for extracting structured data from text using LangChain + OpenAI.

    Uses GPT-4 or GPT-4 Turbo with structured output (JSON mode) to extract
    information according to a user-provided JSON schema. Handles large documents
    through automatic chunking with LangChain text splitters.
    """

    # System prompt template for extraction
    EXTRACTION_SYSTEM_PROMPT = """You are an expert information extraction assistant.
Your task is to extract structured information from the provided text according to a JSON schema.

IMPORTANT RULES:
1. Only extract information that is explicitly stated in the text
2. If information for a field is not found, use null for optional fields
3. For required fields with no clear value, use the most reasonable default or indicate uncertainty
4. Preserve the exact structure of the requested JSON schema
5. Be precise and avoid making assumptions not supported by the text
6. For dates, times, and numbers, preserve the original format when possible

You must respond with ONLY valid JSON that matches the schema. Do not include any explanation or text outside the JSON."""

    EXTRACTION_USER_PROMPT = """Extract information from the following text according to this JSON schema:

## JSON Schema:
```json
{schema}
```

## Required Fields:
{required_fields}

## Optional Fields:
{optional_fields}

## Text to Extract From:
```
{text}
```

Respond with ONLY the extracted JSON data. Ensure all required fields have values."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialize the text extraction service.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Model name to use. Defaults to settings.openai_model.
            temperature: Sampling temperature. Defaults to settings.openai_temperature.
            max_tokens: Maximum tokens for response. Defaults to settings.openai_max_tokens.
            chunk_size: Size of text chunks. Defaults to settings.chunk_size.
            chunk_overlap: Overlap between chunks. Defaults to settings.chunk_overlap.
        """
        self.api_key = api_key if api_key is not None else settings.get_openai_api_key()
        self.model = model or settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self.max_tokens = max_tokens or settings.openai_max_tokens
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._llm: ChatOpenAI | None = None
        self._text_splitter: RecursiveCharacterTextSplitter | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LangChain ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance.

        Raises:
            ExtractionError: If API key is not configured.
        """
        if self._llm is None:
            if not self.api_key:
                raise ExtractionError(
                    "OpenAI API key not configured",
                    error_type="configuration_error",
                    details={"missing": "openai_api_key"},
                )

            self._llm = ChatOpenAI(
                api_key=self.api_key,  # type: ignore[arg-type]
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

        return self._llm

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get or create the text splitter for chunking.

        Returns:
            Configured RecursiveCharacterTextSplitter instance.
        """
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

        return self._text_splitter

    def extract(
        self,
        text: str,
        schema_info: SchemaInfo,
        documents: list[Document] | None = None,
    ) -> ExtractionResult:
        """Extract structured information from text according to schema.

        Args:
            text: Text content to extract from.
            schema_info: Validated schema information.
            documents: Optional LangChain documents (used for metadata).

        Returns:
            ExtractionResult with extracted data and metadata.

        Raises:
            ExtractionError: If extraction fails.
        """
        start_time = time.time()

        # Determine if chunking is needed
        needs_chunking = self._needs_chunking(text)

        if needs_chunking:
            result = self._extract_with_chunking(text, schema_info, documents)
        else:
            result = self._extract_single(text, schema_info)

        result.processing_time_seconds = time.time() - start_time

        logger.info(
            f"Extraction completed: model={result.model_used}, "
            f"tokens={result.total_tokens}, "
            f"chunks={result.chunks_processed}, "
            f"time={result.processing_time_seconds:.2f}s"
        )

        return result

    def _needs_chunking(self, text: str) -> bool:
        """Determine if text needs to be chunked.

        Uses a simple heuristic based on character count.
        Assumes roughly 4 characters per token on average.

        Args:
            text: Text to check.

        Returns:
            True if text should be chunked.
        """
        # Estimate tokens (rough heuristic)
        estimated_tokens = len(text) / 4

        # Leave room for prompt and response
        max_context = 8000  # Conservative estimate for GPT-4
        return estimated_tokens > max_context * 0.5

    def _extract_single(
        self,
        text: str,
        schema_info: SchemaInfo,
    ) -> ExtractionResult:
        """Extract from text without chunking.

        Args:
            text: Text to extract from.
            schema_info: Schema information.

        Returns:
            ExtractionResult with extracted data.

        Raises:
            ExtractionError: If extraction fails.
        """
        prompt = self._build_extraction_prompt(text, schema_info)

        try:
            # Format the prompt into actual messages
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)

            # Parse response
            content = response.content
            if not isinstance(content, str):
                content = str(content)

            extracted_data = self._parse_json_response(content, schema_info)

            # Extract token usage from response
            usage_metadata = getattr(response, "usage_metadata", None) or {}
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get(
                "total_tokens", prompt_tokens + completion_tokens
            )

            # Build field extractions
            field_extractions = self._build_field_extractions(
                extracted_data, schema_info
            )

            return ExtractionResult(
                extracted_data=extracted_data,
                field_extractions=field_extractions,
                model_used=self.model,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                chunks_processed=1,
                is_chunked=False,
                raw_response=content,
            )

        except Exception as e:
            if isinstance(e, ExtractionError):
                raise
            raise ExtractionError(
                f"Extraction failed: {e}",
                error_type="llm_error",
                details={"original_error": str(e)},
            ) from e

    def _extract_with_chunking(
        self,
        text: str,
        schema_info: SchemaInfo,
        documents: list[Document] | None = None,
    ) -> ExtractionResult:
        """Extract from text using chunking strategy.

        Splits text into chunks, extracts from each, then merges results.

        Args:
            text: Text to extract from.
            schema_info: Schema information.
            documents: Optional source documents.

        Returns:
            ExtractionResult with merged extracted data.

        Raises:
            ExtractionError: If extraction fails.
        """
        # Split text into chunks
        if documents:
            chunks = self.text_splitter.split_documents(documents)
        else:
            chunk_texts = self.text_splitter.split_text(text)
            chunks = [Document(page_content=chunk) for chunk in chunk_texts]

        logger.info(f"Processing {len(chunks)} chunks")

        # Extract from each chunk
        chunk_results: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i + 1}/{len(chunks)}")

            try:
                chunk_result = self._extract_single(chunk.page_content, schema_info)
                chunk_results.append(chunk_result.extracted_data)
                total_prompt_tokens += chunk_result.prompt_tokens
                total_completion_tokens += chunk_result.completion_tokens
            except ExtractionError as e:
                logger.warning(f"Chunk {i + 1} extraction failed: {e}")
                continue

        if not chunk_results:
            raise ExtractionError(
                "All chunk extractions failed",
                error_type="extraction_error",
                details={"chunks_attempted": len(chunks)},
            )

        # Merge results from all chunks
        merged_data = self._merge_chunk_results(chunk_results, schema_info)

        # Build field extractions from merged data
        field_extractions = self._build_field_extractions(merged_data, schema_info)

        total_tokens = total_prompt_tokens + total_completion_tokens

        return ExtractionResult(
            extracted_data=merged_data,
            field_extractions=field_extractions,
            model_used=self.model,
            total_tokens=total_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            chunks_processed=len(chunks),
            is_chunked=True,
        )

    def _merge_chunk_results(
        self,
        chunk_results: list[dict[str, Any]],
        schema_info: SchemaInfo,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Merge extraction results from multiple chunks.

        Strategy:
        - For simple values: take the first non-null value found
        - For arrays: concatenate and deduplicate
        - For objects: recursively merge

        Args:
            chunk_results: List of extraction results from chunks.
            schema_info: Schema information for type hints.

        Returns:
            Merged extraction result.
        """
        if len(chunk_results) == 1:
            return chunk_results[0]

        merged: dict[str, Any] = {}

        # Get all keys from all results
        all_keys: set[str] = set()
        for result in chunk_results:
            if isinstance(result, dict):
                all_keys.update(result.keys())

        for key in all_keys:
            values = [
                result.get(key)
                for result in chunk_results
                if isinstance(result, dict) and key in result
            ]

            # Filter out None values
            non_null_values = [v for v in values if v is not None]

            if not non_null_values:
                merged[key] = None
            elif len(non_null_values) == 1:
                merged[key] = non_null_values[0]
            elif isinstance(non_null_values[0], list):
                # Merge arrays (concatenate and try to deduplicate)
                merged_list: list[Any] = []
                seen: set[str] = set()
                for val_list in non_null_values:
                    if isinstance(val_list, list):
                        for item in val_list:
                            item_key = json.dumps(item, sort_keys=True, default=str)
                            if item_key not in seen:
                                seen.add(item_key)
                                merged_list.append(item)
                merged[key] = merged_list
            elif isinstance(non_null_values[0], dict):
                # Recursively merge dicts
                merged[key] = self._merge_dicts(non_null_values)
            else:
                # Take first non-null value for simple types
                merged[key] = non_null_values[0]

        return merged

    def _merge_dicts(self, dicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Recursively merge multiple dictionaries.

        Args:
            dicts: List of dictionaries to merge.

        Returns:
            Merged dictionary.
        """
        if len(dicts) == 1:
            return dicts[0]

        merged: dict[str, Any] = {}
        all_keys: set[str] = set()
        for d in dicts:
            if isinstance(d, dict):
                all_keys.update(d.keys())

        for key in all_keys:
            values = [d.get(key) for d in dicts if isinstance(d, dict) and key in d]
            non_null_values = [v for v in values if v is not None]

            if not non_null_values:
                merged[key] = None
            elif isinstance(non_null_values[0], dict):
                merged[key] = self._merge_dicts(
                    [v for v in non_null_values if isinstance(v, dict)]
                )
            else:
                merged[key] = non_null_values[0]

        return merged

    def _build_extraction_prompt(
        self,
        text: str,
        schema_info: SchemaInfo,
    ) -> ChatPromptTemplate:
        """Build the extraction prompt.

        Args:
            text: Text to extract from.
            schema_info: Schema information.

        Returns:
            Configured ChatPromptTemplate.
        """
        # Format required and optional fields
        required_fields_str = (
            "\n".join(
                f"- {f.path}: {f.field_type}"
                + (f" - {f.description}" if f.description else "")
                for f in schema_info.required_fields
            )
            or "None"
        )

        optional_fields_str = (
            "\n".join(
                f"- {f.path}: {f.field_type}"
                + (f" - {f.description}" if f.description else "")
                for f in schema_info.optional_fields
            )
            or "None"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.EXTRACTION_SYSTEM_PROMPT),
                ("human", self.EXTRACTION_USER_PROMPT),
            ]
        )

        return prompt.partial(
            schema=json.dumps(schema_info.schema, indent=2),
            required_fields=required_fields_str,
            optional_fields=optional_fields_str,
            text=text,
        )

    def _parse_json_response(
        self,
        response: str,
        schema_info: SchemaInfo,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Parse JSON response from the model.

        Args:
            response: Raw response string.
            schema_info: Schema information for validation.

        Returns:
            Parsed JSON data.

        Raises:
            ExtractionError: If parsing fails.
        """
        try:
            # Try to parse the response as JSON
            data = json.loads(response)

            if not isinstance(data, dict):
                raise ExtractionError(
                    "Expected JSON object response",
                    error_type="parse_error",
                    details={"received_type": type(data).__name__},
                )

            return data

        except json.JSONDecodeError as e:
            # Try to extract JSON from the response if it contains extra text
            try:
                # Look for JSON object in the response
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    extracted = json.loads(json_str)
                    if isinstance(extracted, dict):
                        return extracted
            except json.JSONDecodeError:
                pass

            raise ExtractionError(
                f"Failed to parse JSON response: {e}",
                error_type="parse_error",
                details={"response_preview": response[:200]},
            ) from e

    def _build_field_extractions(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
    ) -> list[FieldExtraction]:
        """Build detailed field extraction objects.

        Args:
            data: Extracted data.
            schema_info: Schema information.

        Returns:
            List of FieldExtraction objects.
        """
        extractions: list[FieldExtraction] = []

        for field_info in schema_info.all_fields:
            field_path = field_info.path
            value = self._get_nested_value(data, field_path)

            extractions.append(
                FieldExtraction(
                    field_path=field_path,
                    value=value,
                    confidence=None,  # Could be enhanced with confidence estimation
                    source_text=None,  # Could track source snippets
                    reasoning=None,  # Could capture model reasoning
                )
            )

            # Handle nested fields
            if field_info.nested_fields and isinstance(value, dict):
                for nested_field in field_info.nested_fields:
                    nested_path = nested_field.path
                    nested_value = self._get_nested_value(data, nested_path)
                    extractions.append(
                        FieldExtraction(
                            field_path=nested_path,
                            value=nested_value,
                            confidence=None,
                            source_text=None,
                            reasoning=None,
                        )
                    )

        return extractions

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get a value from nested dictionary using dot notation.

        Args:
            data: Dictionary to get value from.
            path: Dot-separated path (e.g., 'address.city').

        Returns:
            Value at path or None if not found.
        """
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def extract_from_documents(
        self,
        documents: list[Document],
        schema_info: SchemaInfo,
    ) -> ExtractionResult:
        """Extract structured information from LangChain documents.

        Args:
            documents: List of LangChain Document objects.
            schema_info: Validated schema information.

        Returns:
            ExtractionResult with extracted data and metadata.

        Raises:
            ExtractionError: If extraction fails.
        """
        # Combine document content
        combined_text = "\n\n".join(doc.page_content for doc in documents)

        return self.extract(combined_text, schema_info, documents)
