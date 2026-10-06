"""Validate Plotdaten frames against the WebApp required-column contract."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd


def validate_plotdaten(
    df: pd.DataFrame,
    required_columns: Optional[Sequence[str]] = None,
    *,
    mapping: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Check presence of WebApp required columns (and basic sanity).

    Does not fail hard — returns a structured report for the notebook UI.
    """
    if required_columns is None and mapping is not None:
        required_columns = mapping.get("webapp_required_columns", [])
    required_columns = list(required_columns or [])

    present = [c for c in required_columns if c in df.columns]
    missing = [c for c in required_columns if c not in df.columns]

    emptyish = []
    for c in present:
        s = df[c]
        if s.isna().all():
            emptyish.append(c)

    warnings: list[str] = []
    # Warning only: flows are converted solely from a declared m³/s unit row (normalize).
    for flow_col in ("volume flow", "volume flow_source"):
        if flow_col not in df.columns:
            continue
        vf = pd.to_numeric(df[flow_col], errors="coerce").dropna()
        if not vf.empty and float(vf.median()) < 0.05:
            warnings.append(
                f"{flow_col} median still < 0.05 after normalize — check m³/h vs m³/s"
            )
    if "Pressure difference" in df.columns:
        dp = pd.to_numeric(df["Pressure difference"], errors="coerce").dropna()
        if not dp.empty and float(dp.abs().median()) > 50:
            warnings.append(
                "Pressure difference median > 50 — values may still be in kPa (expected bar)"
            )

    ok = len(missing) == 0
    return {
        "ok": ok,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "required_present": present,
        "required_missing": missing,
        "required_all_null": emptyish,
        "warnings": warnings,
        "columns": list(df.columns),
    }
