"""Retention and purge utilities for stored artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agentic_document_extraction.config import settings
from agentic_document_extraction.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PurgeResult:
    """Result of a retention purge run."""

    scanned: int = 0
    removed: int = 0
    skipped: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "removed": self.removed,
            "skipped": self.skipped,
            "errors": self.errors,
        }


def purge_expired_artifacts(
    paths: Iterable[str] | None = None,
    retention_days: int | None = None,
    now: datetime | None = None,
) -> PurgeResult:
    """Remove files older than the retention window.

    Args:
        paths: Iterable of base paths to scan. Defaults to temp_upload_dir.
        retention_days: Retention window in days. Defaults to settings.retention_days.
        now: Optional reference time (useful for testing).

    Returns:
        PurgeResult with counts.
    """
    result = PurgeResult()
    if retention_days is None:
        retention_days = settings.retention_days
    cutoff = (now or datetime.now(UTC)) - timedelta(days=retention_days)

    scan_paths = list(paths) if paths else [settings.temp_upload_dir]
    for base in scan_paths:
        base_path = Path(base)
        if not base_path.exists():
            continue

        for item in base_path.rglob("*"):
            if not item.is_file():
                continue
            result.scanned += 1
            try:
                if datetime.fromtimestamp(item.stat().st_mtime, UTC) < cutoff:
                    item.unlink()
                    result.removed += 1
                else:
                    result.skipped += 1
            except Exception:  # pragma: no cover - defensive logging
                result.errors += 1
                logger.warning("Failed to remove expired artifact", path=str(item))

    logger.info(
        "Retention purge complete",
        **result.to_dict(),
        retention_days=retention_days,
        paths=len(scan_paths),
    )
    return result
