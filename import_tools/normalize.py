"""Normalize HPT template sheets into the Plotdaten / WebApp column contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .detect import detect_header_layout, frame_from_layout, load_sheet_raw

# Reuse BAM elapsed-time helper when available
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

try:
    from hp_tools.utils import get_elapsed_time
except ImportError:  # pragma: no cover
    get_elapsed_time = None  # type: ignore


MAPS_DIR = Path(__file__).resolve().parent / "maps"


def load_map(map_id: str = "hpt_aw_rev1") -> dict[str, Any]:
    """Load ``maps/{map_id}.json``. Default is AIR-TO-WATER REV1."""
    path = MAPS_DIR / f"{map_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in MAPS_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"Import map not found: {path}. Available: {available}"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_maps() -> list[str]:
    """Return available map ids (JSON stems in ``maps/``)."""
    return sorted(p.stem for p in MAPS_DIR.glob("*.json"))


def _norm_key(name: str) -> str:
    return " ".join(str(name).split()).strip().lower()


def _resolve_source_column(df: pd.DataFrame, source: str, aliases: Optional[list[str]]) -> Optional[str]:
    """Match map source name (or alias) to an actual DataFrame column (whitespace-tolerant)."""
    candidates = [source] + list(aliases or [])
    by_norm = {_norm_key(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_key(cand)
        if key in by_norm:
            return by_norm[key]
        # substring soft match only for long unique markers
        if len(key) >= 12:
            for nk, original in by_norm.items():
                if key in nk or nk in key:
                    return original
    return None


def _add_time_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time_elapsed" in out.columns:
        return out
    if "Time" not in out.columns or out.empty:
        return out

    # Prefer Date+Time combined (handles midnight / multi-day runs)
    if "Date" in out.columns:
        try:
            dates = pd.to_datetime(out["Date"], errors="coerce")
            times = out["Time"]
            # Time may be datetime.time, Timestamp, or string
            combined = []
            for d, t in zip(dates, times):
                if pd.isna(d) or t is None or (isinstance(t, float) and pd.isna(t)):
                    combined.append(pd.NaT)
                    continue
                if isinstance(t, pd.Timestamp):
                    tt = t
                    if d is not pd.NaT:
                        combined.append(
                            pd.Timestamp(
                                year=d.year, month=d.month, day=d.day,
                                hour=tt.hour, minute=tt.minute, second=tt.second,
                                microsecond=tt.microsecond,
                            )
                        )
                    else:
                        combined.append(tt)
                elif hasattr(t, "hour"):  # datetime.time
                    combined.append(
                        pd.Timestamp(
                            year=int(d.year), month=int(d.month), day=int(d.day),
                            hour=t.hour, minute=t.minute, second=getattr(t, "second", 0),
                            microsecond=getattr(t, "microsecond", 0),
                        )
                    )
                else:
                    combined.append(pd.to_datetime(f"{d.date()} {t}", errors="coerce"))
            series = pd.Series(combined, index=out.index)
            if series.notna().any():
                t0 = series.dropna().iloc[0]
                out["time_elapsed"] = (series - t0).dt.total_seconds()
                return out
        except Exception:
            pass

    time_col = out["Time"]
    # import_v2: if Time.1 exists and is numeric seconds, prefer it
    if "Time.1" in out.columns:
        try:
            out["time_elapsed"] = pd.to_numeric(out["Time.1"], errors="coerce").round(2)
            return out
        except Exception:
            pass

    zerotime = time_col.iloc[0]
    col_index = list(out.columns).index("Time") + 1  # itertuples: index 0 is Index
    if get_elapsed_time is not None:
        try:
            out["time_elapsed"] = get_elapsed_time(out, zerotime, col_index)
            return out
        except Exception:
            pass

    # Fallback: try datetime subtraction or numeric
    series = pd.to_datetime(time_col, errors="coerce")
    if series.notna().any():
        out["time_elapsed"] = (series - series.iloc[0]).dt.total_seconds()
        return out
    numeric = pd.to_numeric(time_col, errors="coerce")
    out["time_elapsed"] = numeric - numeric.iloc[0]
    return out


def _maybe_convert_volume_flow(series: pd.Series, unit_hint: Optional[str] = None) -> tuple[pd.Series, list[str]]:
    """Prefer m³/h; if values look like m³/s (typical max << 1), convert ×3600."""
    notes: list[str] = []
    s = pd.to_numeric(series, errors="coerce")
    finite = s.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return s, notes
    if unit_hint and "m3/s" in unit_hint.replace("³", "3").replace(" ", "").lower():
        notes.append("volume flow: unit row indicates m³/s → converted ×3600 to m³/h")
        return s * 3600.0, notes
    # Magnitude heuristic: heating HP flows in m³/h are typically > 0.2; in m³/s often < 0.05
    med = float(finite.median())
    if 0 < med < 0.05:
        notes.append(
            f"volume flow: median={med:.5g} looks like m³/s → converted ×3600 to m³/h (decision 3.4 legacy)"
        )
        return s * 3600.0, notes
    return s, notes


def normalize_dataframe(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    *,
    units_row: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply rename/scale/fill rules from an import map.

    Returns (normalized_df, report) where report has renames, warnings, missing sources.
    """
    report: dict[str, Any] = {
        "renames": {},
        "missing_sources": [],
        "warnings": [],
        "fills": [],
        "map_id": mapping.get("map_id"),
        "map_version": mapping.get("version"),
    }
    out = pd.DataFrame(index=df.index)
    used_sources: set[str] = set()

    # Targets whose all-empty version would read as a measurement (mass flow, corrected Q/P,
    # virtual and real BUH). Maps without the key keep the old "drop every empty optional".
    skip_empty_targets: Optional[set[str]] = None
    if "skip_empty_targets" in mapping:
        skip_empty_targets = set(mapping.get("skip_empty_targets") or [])

    # First pass: explicit column map (order in JSON matters for Virtual Back-up vs BUH)
    for colmap in mapping.get("columns", []):
        source = colmap["source"]
        target = colmap["target"]
        aliases = colmap.get("aliases")
        actual = _resolve_source_column(df, source, aliases)
        if actual is None:
            if colmap.get("required"):
                report["missing_sources"].append(source)
            continue

        series = df[actual]
        used_sources.add(actual)

        # Unit / scale
        if colmap.get("scale") is not None:
            series = pd.to_numeric(series, errors="coerce") * float(colmap["scale"])
        if target == "volume flow" or "volume flow" in _norm_key(source):
            unit_hint = None
            if units_row is not None and actual in getattr(units_row, "index", []):
                unit_hint = str(units_row[actual])
            series, vnotes = _maybe_convert_volume_flow(series, unit_hint)
            report["warnings"].extend(vnotes)

        # Do not create optional columns that are entirely empty (e.g. mass flow_sink all-NaN)
        if not colmap.get("required", False) and series.isna().all():
            if skip_empty_targets is None or target in skip_empty_targets:
                report["fills"].append(f"skipped empty optional {actual}->{target}")
                continue
            # Monitoring extra: keep the column so the template field stays visible.
            report["fills"].append(f"kept empty optional {actual}->{target} (all-NaN)")

        # e.g. Backup heater → BUH only if Virtual Back-up did not already fill
        if colmap.get("fill_if_target_missing") and target in out.columns:
            if out[target].notna().any():
                report["fills"].append(f"skipped {source}->{target} (already present)")
                continue

        out[target] = series
        report["renames"][actual] = target

    # Passthrough unmapped (optional extras)
    if mapping.get("passthrough_unmapped", True):
        for c in df.columns:
            if c in used_sources:
                continue
            if c in out.columns:
                continue
            # Avoid colliding with targets already taken
            out[c] = df[c]

    # A test uses either the virtual or the real back-up, not both. Keep both series if the
    # lab filled both — merging or dropping one would falsify the power balance — but say so:
    # only the virtual column drives the WebApp BUH corrections.
    if "Electrical power input BUH" in out.columns and "Real BUH power input" in out.columns:
        virtual = pd.to_numeric(out["Electrical power input BUH"], errors="coerce")
        real = pd.to_numeric(out["Real BUH power input"], errors="coerce")
        if virtual.notna().any() and real.notna().any():
            report["warnings"].append(
                "Both Virtual Back-up and Backup heater Input carry data — kept as two "
                "columns; WebApp BUH corrections still use Virtual Back-up only. Check "
                "with the lab which back-up was actually active."
            )

    out = _add_time_elapsed(out)

    # Fill-if-missing rules
    for rule in mapping.get("fill_if_missing", []):
        target = rule["target"]
        if target in out.columns and out[target].notna().any():
            continue
        rname = rule.get("rule")
        src = rule.get("from")

        if rname == "copy_uncorr_until_lab_or_bam_rule" and src and src in out.columns:
            out[target] = out[src]
            flag_col = rule.get("flag_column", "corr_fill_rule")
            out[flag_col] = rule.get("flag_value", "copied_uncorr_v0")
            report["fills"].append(f"{target} <- {src} ({rule.get('flag_value')})")
        elif rname == "volume_flow_m3h_times_density_or_997" and "volume flow" in out.columns:
            density_col = next((c for c in ("density_sink", "density") if c in out.columns), None)
            density = out[density_col] if density_col else 997.0
            out[target] = pd.to_numeric(out["volume flow"], errors="coerce") / 3600.0 * pd.to_numeric(
                density, errors="coerce"
            ).fillna(997.0)
            report["fills"].append(f"{target} <- volume flow/3600*density")
        elif rname == "zero_if_absent":
            out[target] = 0.0
            report["fills"].append(f"{target} <- 0")

    return out, report


def normalize_sheet(
    path: str,
    sheet_name: str,
    mapping: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect layout, normalize one sheet, return (df, report)."""
    mapping = mapping or load_map()
    raw = load_sheet_raw(path, sheet_name)
    hd = mapping.get("header_detection", {})
    layout = detect_header_layout(
        raw,
        hd.get("marker_any_of", ["Date", "Time"]),
        units_row_offset=hd.get("units_row_offset", 1),
        avg_row_offset=hd.get("avg_row_offset", 2),
        data_row_offset=hd.get("data_row_offset", 3),
        fallback_header_row_0based=hd.get("fallback_header_row_0based", 3),
    )
    df = frame_from_layout(raw, layout)
    units_row = None
    if layout.units_row is not None:
        units_vals = raw.iloc[layout.units_row]
        # align by position to df columns
        units_row = pd.Series(units_vals.to_numpy()[: len(df.columns)], index=df.columns)

    out, report = normalize_dataframe(df, mapping, units_row=units_row)
    report["sheet"] = sheet_name
    report["layout"] = {
        "header_row": layout.header_row,
        "units_row": layout.units_row,
        "avg_row": layout.avg_row,
        "data_start_row": layout.data_start_row,
        "matched_markers": layout.matched_markers,
    }
    return out, report


def write_plotdaten(
    df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False, engine="openpyxl")
    return out_path
