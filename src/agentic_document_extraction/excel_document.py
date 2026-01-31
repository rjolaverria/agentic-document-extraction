"""Dataclasses representing a parsed Excel workbook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExcelCell:
    """Represents a single Excel cell with type metadata."""

    value: Any
    data_type: str
    formula: str | None = None
    is_merged: bool = False


@dataclass
class ExcelSheet:
    """Represents a single worksheet within a workbook."""

    name: str
    rows: list[list[ExcelCell]]
    headers: list[str] | None
    row_count: int
    column_count: int


@dataclass
class ExcelDocument:
    """Represents a parsed Excel workbook with sheet metadata."""

    sheets: list[ExcelSheet]
    active_sheet: str
    metadata: dict[str, Any]
