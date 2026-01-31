"""LangChain tool for inspecting Excel spreadsheets natively."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import InjectedState

from agentic_document_extraction.excel_document import (
    ExcelCell,
    ExcelDocument,
    ExcelSheet,
)

logger = logging.getLogger(__name__)


def _get_sheet(spreadsheet: ExcelDocument, sheet_name: str | None = None) -> ExcelSheet:
    if sheet_name:
        for sheet in spreadsheet.sheets:
            if sheet.name == sheet_name:
                return sheet
        raise ToolException(f"Sheet '{sheet_name}' not found")
    return spreadsheet.sheets[0]


def _parse_row_range(row_range: str | None, max_rows: int) -> tuple[int, int]:
    if not row_range:
        return 1, max_rows
    try:
        start_str, end_str = row_range.split(":")
        start = int(start_str)
        end = int(end_str)
        if start < 1 or end < start:
            raise ValueError
        return start, min(end, max_rows)
    except Exception as exc:
        raise ToolException(
            "row_range must be in 'start:end' format with positive integers"
        ) from exc


def _cell_to_json(cell: ExcelCell) -> Any:
    if cell.value is None:
        return None
    if cell.data_type == "date":
        return str(cell.value)
    return cell.value


def _extract_rows(
    sheet: ExcelSheet,
    start_row: int,
    end_row: int,
    columns: list[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(sheet.rows, start=1):
        if row_idx < start_row or row_idx > end_row:
            continue
        row_dict: dict[str, Any] = {}
        for col_idx, cell in enumerate(row, start=1):
            col_letter = chr(ord("A") + col_idx - 1)
            if columns and col_letter not in columns:
                continue
            row_dict[col_letter] = _cell_to_json(cell)
        if row_dict:
            rows.append(row_dict)
    return rows


@tool("analyze_spreadsheet")
def analyze_spreadsheet(
    sheet_name: Annotated[
        str | None, "Sheet name to analyze (None for active sheet)"
    ] = None,
    row_range: Annotated[str | None, "Row range like '1:100' or '5:50'"] = None,
    columns: Annotated[
        list[str] | None, "Column letters to include, e.g. ['A','B']"
    ] = None,
    state: Annotated[dict[str, Any] | None, InjectedState] = None,
) -> dict[str, Any]:
    """Inspect native Excel data with controlled ranges and columns."""
    spreadsheet: ExcelDocument | None = None
    if state is not None:
        spreadsheet = state.get("spreadsheet")
    if spreadsheet is None:
        raise ToolException("No spreadsheet data available in state")

    sheet = _get_sheet(spreadsheet, sheet_name)
    start, end = _parse_row_range(row_range, sheet.row_count)

    rows = _extract_rows(sheet, start, end, columns)

    return {
        "sheet": sheet.name,
        "headers": columns if columns else None,
        "row_range": f"{start}:{end}",
        "rows": rows,
        "notes": "Values are native Excel cell contents; formulas preserved where available.",
    }
