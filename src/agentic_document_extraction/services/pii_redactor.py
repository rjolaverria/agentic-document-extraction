"""PII redaction utilities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

from agentic_document_extraction.config import settings
from agentic_document_extraction.services.schema_validator import FieldInfo, SchemaInfo
from agentic_document_extraction.utils.logging import get_logger

logger = get_logger(__name__)

PIIPolicy = Literal["allow", "mask", "hash", "drop"]


_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ein": re.compile(r"\b\d{2}-\d{7}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "dob": re.compile(r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b"),
    "address": re.compile(
        r"\b\d{1,5}\s+[A-Za-z0-9.\-']+\s+(?:Ave|Avenue|St|Street|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b",
        re.IGNORECASE,
    ),
}


@dataclass
class RedactionMetrics:
    """Metrics for redaction actions."""

    masked: int = 0
    hashed: int = 0
    dropped: int = 0
    detections: int = 0
    fields_processed: int = 0
    text_spans_redacted: int = 0

    def merge(self, other: RedactionMetrics) -> RedactionMetrics:
        """Merge another metrics object into this one."""
        self.masked += other.masked
        self.hashed += other.hashed
        self.dropped += other.dropped
        self.detections += other.detections
        self.fields_processed += other.fields_processed
        self.text_spans_redacted += other.text_spans_redacted
        return self

    def to_dict(self) -> dict[str, int]:
        """Return metrics as a dictionary."""
        return {
            "masked": self.masked,
            "hashed": self.hashed,
            "dropped": self.dropped,
            "detections": self.detections,
            "fields_processed": self.fields_processed,
            "text_spans_redacted": self.text_spans_redacted,
        }


class PIIRedactor:
    """Detect and redact PII values using regex patterns and optional LLM fallback."""

    def __init__(
        self,
        mask_char: str | None = None,
        mask_keep_last: int | None = None,
        hash_salt: str | None = None,
        llm_detector: Callable[[str], Iterable[str]] | None = None,
    ) -> None:
        """Initialize the redactor.

        Args:
            mask_char: Character to use for masking.
            mask_keep_last: Number of trailing characters to keep.
            hash_salt: Optional salt when hashing values.
            llm_detector: Optional callable to detect PII spans in free text.
        """
        self.mask_char = mask_char or settings.pii_mask_char
        self.mask_keep_last = (
            settings.pii_mask_keep_last if mask_keep_last is None else mask_keep_last
        )
        self.hash_salt = hash_salt or settings.pii_hash_salt or ""
        self.llm_detector = llm_detector

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def redact_outputs(
        self,
        data: dict[str, Any],
        markdown: str | None,
        schema_info: SchemaInfo,
        redact_output: bool,
    ) -> tuple[dict[str, Any], str | None, RedactionMetrics]:
        """Redact structured data and markdown summary."""
        metrics = RedactionMetrics()
        if not redact_output:
            return data, markdown, metrics

        redacted_data, data_metrics = self.redact_data(
            data=data,
            schema_info=schema_info,
        )
        metrics.merge(data_metrics)

        redacted_markdown = (
            self.redact_text(markdown, metrics) if markdown is not None else None
        )

        logger.info(
            "PII redaction applied",
            **metrics.to_dict(),
        )
        return redacted_data, redacted_markdown, metrics

    def redact_data(
        self,
        data: dict[str, Any],
        schema_info: SchemaInfo,
    ) -> tuple[dict[str, Any], RedactionMetrics]:
        """Apply field-level PII policies to structured data."""
        metrics = RedactionMetrics()
        output = self._deep_copy(data)

        for field_info in schema_info.all_fields:
            policy = self._resolve_policy(field_info)
            if policy == "allow":
                continue
            metrics.fields_processed += 1
            output, field_metrics = self._apply_policy_to_path(
                output,
                field_info.path,
                policy,
            )
            metrics.merge(field_metrics)

        # Optionally apply a default policy to fields not in schema
        default_policy = settings.effective_default_pii_policy()
        if default_policy in {"allow", "mask", "hash", "drop"}:
            extra_keys = set(output.keys()) - {f.name for f in schema_info.all_fields}
            for key in extra_keys:
                output, extra_metrics = self._apply_policy_to_path(
                    output, key, cast(PIIPolicy, default_policy)
                )
                metrics.merge(extra_metrics)

        # Sweep remaining strings for incidental PII patterns
        output = self._redact_string_values(output, metrics)

        return output, metrics

    def redact_text(
        self,
        text: str,
        metrics: RedactionMetrics | None = None,
    ) -> str:
        """Redact PII spans in a plain text blob."""
        if metrics is None:
            metrics = RedactionMetrics()

        redacted = text
        for pattern in _PATTERNS.values():
            redacted, count = pattern.subn(self._mask_match, redacted)
            metrics.text_spans_redacted += count
            metrics.detections += count

        if self.llm_detector:
            try:
                spans = list(self.llm_detector(text))
                for span in spans:
                    if span and span in redacted:
                        redacted = redacted.replace(span, self._mask_string(span))
                        metrics.text_spans_redacted += 1
                        metrics.detections += 1
            except Exception:  # pragma: no cover - defensive
                logger.warning("LLM fallback detector failed", exc_info=True)

        return redacted

    def _mask_match(self, match: re.Match[str]) -> str:
        """Mask regex match helper for substitutions."""
        keep_last = max(self.mask_keep_last, 4)
        return self._mask_string(match.group(0), keep_last=keep_last)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _resolve_policy(self, field_info: FieldInfo) -> PIIPolicy:
        policy = field_info.pii_policy or settings.effective_default_pii_policy()
        if policy in ("allow", "mask", "hash", "drop"):
            return cast(PIIPolicy, policy)
        return "allow"

    def _apply_policy_to_path(
        self,
        data: dict[str, Any],
        path: str,
        policy: PIIPolicy,
    ) -> tuple[dict[str, Any], RedactionMetrics]:
        metrics = RedactionMetrics()
        parts = path.split(".")
        updated = self._redact_recursive(data, parts, policy, metrics)
        return updated, metrics

    def _redact_recursive(
        self,
        current: Any,
        parts: list[str],
        policy: PIIPolicy,
        metrics: RedactionMetrics,
    ) -> Any:
        if not parts:
            new_value, action = self._apply_policy(current, policy)
            if action == "masked":
                metrics.masked += 1
            elif action == "hashed":
                metrics.hashed += 1
            elif action == "dropped":
                metrics.dropped += 1
            return new_value

        part = parts[0]
        is_array = part.endswith("[]")
        key = part[:-2] if is_array else part

        if isinstance(current, dict) and key in current:
            if is_array:
                array_val = current.get(key)
                if isinstance(array_val, list):
                    current[key] = [
                        self._redact_recursive(item, parts[1:], policy, metrics)
                        for item in array_val
                    ]
            else:
                current[key] = self._redact_recursive(
                    current.get(key), parts[1:], policy, metrics
                )

            # Drop logic for dicts (handled at leaf)
            if parts == [part] and policy == "drop":
                current.pop(key, None)
        elif isinstance(current, list):
            current = [
                self._redact_recursive(item, parts, policy, metrics) for item in current
            ]

        return current

    def _apply_policy(self, value: Any, policy: PIIPolicy) -> tuple[Any, str | None]:
        if policy == "allow":
            return value, None
        if policy == "drop":
            return None, "dropped"
        if policy == "mask":
            return self._mask_value(value), "masked"
        if policy == "hash":
            return self._hash_value(value), "hashed"
        return value, None

    def _redact_string_values(
        self, value: Any, metrics: RedactionMetrics
    ) -> Any:
        """Apply regex-based text redaction to all string leaves."""
        if isinstance(value, dict):
            return {
                k: self._redact_string_values(v, metrics) for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._redact_string_values(v, metrics) for v in value]
        if isinstance(value, str):
            return self.redact_text(value, metrics)
        return value

    def _mask_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._mask_string(value)
        if isinstance(value, (int, float)):
            return self._mask_string(str(value))
        if isinstance(value, list):
            return [self._mask_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._mask_value(v) for k, v in value.items()}
        return value

    def _hash_value(self, value: Any) -> str:
        digest = hashlib.sha256()
        digest.update(self.hash_salt.encode("utf-8"))
        digest.update(str(value).encode("utf-8"))
        return digest.hexdigest()

    def _mask_string(self, value: str, keep_last: int | None = None) -> str:
        if not value:
            return value
        keep = self.mask_keep_last if keep_last is None else keep_last
        if len(value) <= keep:
            return self.mask_char * len(value)
        trailing = value[-keep:] if keep else ""
        masked_len = len(value) - len(trailing)
        return f"{self.mask_char * masked_len}{trailing}"

    @staticmethod
    def _deep_copy(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: PIIRedactor._deep_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [PIIRedactor._deep_copy(v) for v in value]
        return value
