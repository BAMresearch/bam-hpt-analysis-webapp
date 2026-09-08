"""Header / layout detection for HPT data-collection templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import pandas as pd


@dataclass
class HeaderLayout:
    header_row: int  # 0-based Excel row index of name row
    units_row: Optional[int]
    avg_row: Optional[int]
    data_start_row: int
    matched_markers: list[str]


def _cell_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def detect_header_layout(
    raw: pd.DataFrame,
    marker_any_of: Sequence[str],
    *,
    units_row_offset: int = 1,
    avg_row_offset: int = 2,
    data_row_offset: int = 3,
    fallback_header_row_0based: int = 3,
    max_scan_rows: int = 15,
) -> HeaderLayout:
    """
    Scan the first rows of an unheadered sheet for marker column names.

    HPT REV0 typically: names ~row 4 (1-based) → index 3; units/avg/data follow.
    """
    markers = [m.strip() for m in marker_any_of if m and str(m).strip()]
    scan_n = min(max_scan_rows, len(raw))

    for r in range(scan_n):
        row_vals = [_cell_str(v) for v in raw.iloc[r].tolist()]
        matched = [m for m in markers if any(m == v or m in v for v in row_vals if v)]
        # Prefer a row that looks like a header strip (several non-empty cells + Date/Time)
        nonempty = sum(1 for v in row_vals if v)
        if matched and nonempty >= 4:
            header_row = r
            units_row = header_row + units_row_offset if units_row_offset else None
            avg_row = header_row + avg_row_offset if avg_row_offset else None
            data_start = header_row + data_row_offset
            return HeaderLayout(
                header_row=header_row,
                units_row=units_row if units_row is not None and units_row < len(raw) else None,
                avg_row=avg_row if avg_row is not None and avg_row < len(raw) else None,
                data_start_row=min(data_start, len(raw)),
                matched_markers=matched,
            )

    # Fallback for known REV0 layout
    hr = min(fallback_header_row_0based, max(0, len(raw) - 1))
    return HeaderLayout(
        header_row=hr,
        units_row=hr + units_row_offset if hr + units_row_offset < len(raw) else None,
        avg_row=hr + avg_row_offset if hr + avg_row_offset < len(raw) else None,
        data_start_row=min(hr + data_row_offset, len(raw)),
        matched_markers=[],
    )


def load_sheet_raw(path: str, sheet_name: str | int) -> pd.DataFrame:
    """Read a sheet with no header so row indices match Excel 0-based data rows."""
    return pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")


def frame_from_layout(raw: pd.DataFrame, layout: HeaderLayout) -> pd.DataFrame:
    """Build a typed DataFrame using detected header row; drop units/avg banner rows."""
    headers = [_cell_str(v) for v in raw.iloc[layout.header_row].tolist()]
    # Ensure unique column names for pandas
    seen: dict[str, int] = {}
    unique_headers: list[str] = []
    for h in headers:
        key = h if h else "Unnamed"
        n = seen.get(key, 0)
        seen[key] = n + 1
        unique_headers.append(key if n == 0 else f"{key}.{n}")

    skip = {layout.header_row}
    if layout.units_row is not None:
        skip.add(layout.units_row)
    if layout.avg_row is not None:
        skip.add(layout.avg_row)

    body = raw.drop(index=list(skip)).reset_index(drop=True)
    # Keep only rows from original data_start onward after drop is awkward;
    # instead slice raw from data_start then assign columns.
    body = raw.iloc[layout.data_start_row :].copy()
    body.columns = unique_headers[: body.shape[1]]
    body = body.reset_index(drop=True)
    # Drop fully empty rows
    body = body.dropna(how="all")
    return body


def list_data_sheets(
    path: str,
    expected: Optional[Iterable[str]] = None,
) -> list[str]:
    """Return sheet names present in workbook (optionally filtered to expected set)."""
    xl = pd.ExcelFile(path, engine="openpyxl")
    names = list(xl.sheet_names)
    if expected is None:
        return names
    want = set(expected)
    return [n for n in names if n in want] or names
