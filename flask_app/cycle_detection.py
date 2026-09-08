"""
Cycle detection for BAM load-based testing (Phase 1).
Detects cycle boundaries and compressor ON/OFF states from time series power data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Any

# Default detection parameters (can be overridden by config)
DEFAULT_DETECTION_THRESHOLDS = {
    "cycle": {
        "min_duration": 300,
        "max_duration": 3600,
        "startup_threshold_multiplier": 0.5,
        "power_increase_ratio": 1.2,
    },
    "compressor": {
        "power_change_threshold": 0.5,
        "smoothing_window": 5,
    },
}


def _get_time_col(df: pd.DataFrame) -> pd.Series:
    """Return time column (time_elapsed or index)."""
    if "time_elapsed" in df.columns:
        return df["time_elapsed"]
    return pd.Series(df.index, index=df.index)


def detect_cycle_boundaries(
    df: pd.DataFrame,
    min_cycle_duration: float = 300,
    max_cycle_duration: float = 3600,
    startup_threshold_multiplier: float = 0.5,
    power_increase_ratio: float = 1.2,
    power_column: str = "Electric Power Input (corrected)",
) -> List[Tuple[float, float]]:
    """
    Detect cycle boundaries from power time series (compressor startup events).

    A cycle is assumed to start at an abrupt power increase (compressor ON).
    Cycle ends when the next cycle starts, or at end of data.

    Args:
        df: DataFrame with time_elapsed and power column.
        min_cycle_duration: Minimum cycle duration in seconds (filter noise).
        max_cycle_duration: Maximum cycle duration in seconds (filter invalid).
        startup_threshold_multiplier: Threshold = power.std() * this (e.g. 0.5).
        power_increase_ratio: Minimum relative increase for startup (e.g. 1.2 = 20%).
        power_column: Column name for electric power (kW).

    Returns:
        List of (cycle_start_time, cycle_end_time) in same units as time_elapsed.
    """
    if power_column not in df.columns:
        return []

    power = pd.Series(df[power_column].values, index=df.index).astype(float)
    time_col = _get_time_col(df)
    power = power.dropna()
    if len(power) < 2:
        return []

    # Align time to power index
    time_vals = time_col.reindex(power.index).ffill().bfill()

    power_diff = power.diff()
    std_power = power.std()
    if std_power == 0 or np.isnan(std_power):
        return []
    startup_threshold = float(std_power * startup_threshold_multiplier)

    potential_starts: List[float] = []
    for i in range(1, len(power)):
        idx = power.index[i]
        prev_idx = power.index[i - 1]
        if power_diff.loc[idx] > startup_threshold:
            prev_val = power.loc[prev_idx]
            if prev_val != 0 and prev_val == prev_val:
                if power.loc[idx] >= prev_val * power_increase_ratio:
                    t = time_vals.loc[idx]
                    if isinstance(t, (int, float)) and not np.isnan(t):
                        potential_starts.append(float(t))

    # Build cycles: from each start to the next start (or end of data)
    cycles: List[Tuple[float, float]] = []
    for i in range(len(potential_starts)):
        cycle_start = potential_starts[i]
        if i + 1 < len(potential_starts):
            cycle_end = potential_starts[i + 1]
        else:
            # Last cycle: end at last time in data
            cycle_end = float(time_vals.iloc[-1]) if len(time_vals) else cycle_start
        duration = cycle_end - cycle_start
        if min_cycle_duration <= duration <= max_cycle_duration:
            cycles.append((cycle_start, cycle_end))
    return cycles


def detect_compressor_states(
    df: pd.DataFrame,
    cycle_start: float,
    cycle_end: float,
    power_change_threshold: float = 0.5,
    smoothing_window: int = 5,
    power_column: str = "Electric Power Input (corrected)",
) -> pd.DataFrame:
    """
    Label compressor ON/OFF state within a cycle using power rate of change.

    Args:
        df: Full DataFrame with time_elapsed and power column.
        cycle_start: Cycle start time (seconds).
        cycle_end: Cycle end time (seconds).
        power_change_threshold: Min rate (kW/min) to count as ON or OFF transition.
        smoothing_window: Rolling window size for smoothing power derivative.
        power_column: Column name for electric power (kW).

    Returns:
        DataFrame slice for the cycle with added column 'compressor_state' (True=ON, False=OFF).
    """
    time_series = _get_time_col(df)
    time_col = time_series.name if hasattr(time_series, "name") and time_series.name else "time_elapsed"
    mask = (time_series >= cycle_start) & (time_series <= cycle_end)
    cycle_df = df.loc[mask].copy()
    if cycle_df.empty or power_column not in cycle_df.columns:
        cycle_df["compressor_state"] = False
        return cycle_df

    power = cycle_df[power_column].astype(float)
    time_vals = _get_time_col(cycle_df)
    dt = time_vals.diff()
    dt = dt.replace(0, np.nan).ffill().fillna(1.0 / 60.0)
    power_rate = power.diff() / dt.values * 60.0
    power_rate_smooth = power_rate.rolling(window=min(smoothing_window, len(power_rate)), center=True).mean()

    state = pd.Series(False, index=cycle_df.index)
    current = False
    for i, idx in enumerate(cycle_df.index):
        r = power_rate_smooth.loc[idx]
        if pd.isna(r):
            state.loc[idx] = current
            continue
        if r > power_change_threshold:
            current = True
        elif r < -power_change_threshold:
            current = False
        state.loc[idx] = current

    cycle_df["compressor_state"] = state
    return cycle_df


def get_detection_config(config: Optional[dict] = None) -> dict:
    """Return detection thresholds from app config or defaults."""
    if config and isinstance(config.get("detection_thresholds"), dict):
        det = config["detection_thresholds"]
        return {
            "cycle": {**DEFAULT_DETECTION_THRESHOLDS["cycle"], **det.get("cycle", {})},
            "compressor": {**DEFAULT_DETECTION_THRESHOLDS["compressor"], **det.get("compressor", {})},
        }
    return DEFAULT_DETECTION_THRESHOLDS.copy()
