"""Tests for PIIRedactor service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agentic_document_extraction.services.pii_redactor import PIIRedactor
from agentic_document_extraction.services.retention import purge_expired_artifacts
from agentic_document_extraction.services.schema_validator import SchemaValidator


def _make_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "ssn": {"type": "string", "pii": "mask"},
            "email": {"type": "string", "pii": "hash"},
            "note": {"type": "string"},
        },
        "required": ["ssn", "email"],
    }


def test_redact_data_applies_policies() -> None:
    validator = SchemaValidator()
    schema_info = validator.validate(_make_schema())
    redactor = PIIRedactor(mask_keep_last=4)

    data = {
        "ssn": "123-45-6789",
        "email": "user@example.com",
        "note": "keep this",
    }

    redacted, metrics = redactor.redact_data(data, schema_info)

    assert redacted["ssn"].endswith("6789")
    assert redacted["ssn"].startswith("***")
    assert redacted["email"] != "user@example.com"
    assert len(redacted["email"]) == 64  # sha256 hex digest length
    assert redacted["note"] == "keep this"
    assert metrics.masked >= 1
    assert metrics.hashed >= 1


def test_redact_text_overlapping_patterns() -> None:
    redactor = PIIRedactor(mask_keep_last=2, mask_char="#")
    text = "SSN 123-45-6789 and phone 555-123-4567"
    redacted = redactor.redact_text(text)
    assert "6789" in redacted
    assert "123-45" not in redacted
    assert "555-123-4567" not in redacted


def test_purge_expired_artifacts_removes_old_files(tmp_path) -> None:
    old_file = tmp_path / "old.txt"
    old_file.write_text("pii")
    past = datetime.now(UTC) - timedelta(days=10)
    # set mtime in the past
    old_ts = past.timestamp()
    old_file.touch()
    old_file.stat()
    import os

    os.utime(old_file, (old_ts, old_ts))

    recent_file = tmp_path / "recent.txt"
    recent_file.write_text("ok")

    result = purge_expired_artifacts(paths=[str(tmp_path)], retention_days=7)

    assert not old_file.exists()
    assert recent_file.exists()
    assert result.removed == 1
