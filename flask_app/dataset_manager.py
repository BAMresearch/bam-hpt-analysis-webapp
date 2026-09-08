"""Dataset (SQLite database) management for first-run and switching (B1.6).

Safety rules:
- Never delete, truncate, DROP, or overwrite an existing .db file.
- Creating a database only succeeds when the target path does not already exist.
- Schema setup uses CREATE TABLE IF NOT EXISTS / ADD COLUMN only.
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_APP_DIR, ".."))
# Managed folder for lab databases (gitignored *.db). Existing BAM DB lives here too.
_DATASETS_DIR = os.path.normpath(os.path.join(_REPO_ROOT, "notebooks"))

# Minimal results table for a brand-new empty database (never used to replace an existing file).
RESULTS_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS results (
    file_name TEXT,
    data_set TEXT,
    start_time REAL,
    end_time REAL,
    avg_t_db REAL,
    avg_t_wb REAL,
    avg_t_supply REAL,
    avg_t_return_emu REAL,
    avg_t_return_calc REAL,
    avg_t_b_calc REAL,
    avg_t_h_calc REAL,
    avg_q_hb REAL,
    avg_q_ba REAL,
    avg_volume_flow REAL,
    avg_mass_flow REAL,
    avg_heating_capacity_uncorr REAL,
    avg_power_input_uncorr REAL,
    cop_uncorr REAL,
    avg_heating_capacity_corr REAL,
    avg_power_input_corr REAL,
    cop_corr REAL,
    avg_heating_capacity_bam_corr REAL,
    avg_power_input_bam_corr REAL,
    cop_bam_corr REAL,
    avg_t_mean_log REAL,
    t_mean_from_avgs REAL,
    avg_dt_ln REAL,
    avg_dp REAL,
    avg_adj_dp REAL,
    avg_P_hyd REAL,
    avg_eff_pump REAL,
    avg_qcorr_bam REAL,
    avg_pcorr_bam REAL,
    avg_power_buh REAL,
    avg_power_buh_from_ts REAL,
    avg_t_sup_buh REAL,
    has_time_series INTEGER,
    time_series_data TEXT,
    Lab_ID TEXT,
    HP_ID TEXT,
    Test_label TEXT,
    test_cond TEXT,
    climate TEXT,
    application TEXT,
    flow_config TEXT,
    COP_dataset TEXT,
    scatter_dataset TEXT,
    Notes TEXT,
    profile_id TEXT,
    condition_set_id TEXT,
    cycle_type TEXT,
    DTret REAL,
    avg_lab_air_db REAL,
    avg_lab_atm_pressure REAL,
    avg_cp_out_sink REAL,
    avg_cp_in_sink REAL,
    avg_density_sink REAL,
    avg_compressor_frequency REAL,
    avg_fan_speed REAL,
    avg_voltage REAL,
    avg_current REAL,
    avg_frequency REAL,
    avg_cos_phi REAL,
    avg_real_buh_power REAL,
    avg_t_in_cond REAL,
    avg_t_out_cond REAL,
    avg_t_in_evap REAL,
    avg_t_out_evap REAL
)
"""

# Optional monitoring means from the HPT air-to-water REV1 template (v0.1.0).
# Extras only: nullable, never part of the core insert_query, written after insert by rowid.
# avg_real_buh_power is the REAL backup-heater meter and is unrelated to avg_power_buh
# (which stays the virtual back-up / "Electrical power input BUH" path).
OPTIONAL_MONITORING_MEAN_COLUMNS = {
    "avg_lab_air_db": "REAL",
    "avg_lab_atm_pressure": "REAL",
    "avg_cp_out_sink": "REAL",
    "avg_cp_in_sink": "REAL",
    "avg_density_sink": "REAL",
    "avg_compressor_frequency": "REAL",
    "avg_fan_speed": "REAL",
    "avg_voltage": "REAL",
    "avg_current": "REAL",
    "avg_frequency": "REAL",
    "avg_cos_phi": "REAL",
    "avg_real_buh_power": "REAL",
    "avg_t_in_cond": "REAL",
    "avg_t_out_cond": "REAL",
    "avg_t_in_evap": "REAL",
    "avg_t_out_evap": "REAL",
}

# Columns required by config insert_query / mean-value page that older empty DBs omitted.
# ADD COLUMN only — never used to rebuild an existing table.
ANALYSIS_MEAN_COLUMNS = {
    "avg_t_b_calc": "REAL",
    "avg_t_h_calc": "REAL",
    "avg_q_hb": "REAL",
    "avg_q_ba": "REAL",
    "avg_eff_pump": "REAL",
    "avg_qcorr_bam": "REAL",
    "avg_pcorr_bam": "REAL",
    "avg_power_buh": "REAL",
    "avg_power_buh_from_ts": "REAL",
    "avg_t_sup_buh": "REAL",
    "has_time_series": "INTEGER",
    "time_series_data": "TEXT",
    **OPTIONAL_MONITORING_MEAN_COLUMNS,
}


def datasets_dir() -> str:
    os.makedirs(_DATASETS_DIR, exist_ok=True)
    return _DATASETS_DIR


def sanitize_db_name(name: str) -> str:
    """Return a safe database file stem (no path, no extension)."""
    raw = (name or "").strip()
    if not raw:
        raise ValueError("Please enter a database name.")
    raw = os.path.basename(raw)
    if raw.lower().endswith(".db"):
        raw = raw[:-3]
    raw = raw.strip()
    if not raw:
        raise ValueError("Please enter a database name.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\- ]{0,80}", raw):
        raise ValueError(
            "Use letters, numbers, spaces, underscore or hyphen only (max ~80 characters)."
        )
    return raw.replace(" ", "_")


def path_for_new_name(name: str) -> str:
    stem = sanitize_db_name(name)
    return os.path.normpath(os.path.join(datasets_dir(), f"{stem}.db"))


def resolve_db_path(path: str) -> str:
    path = (path or "").strip().strip('"')
    if not path:
        raise ValueError("No database path provided.")
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(_APP_DIR, path))
    return os.path.normpath(path)


def is_sqlite_file(path: str) -> bool:
    """True if path exists and looks like a SQLite database (header check)."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as fh:
            return fh.read(16).startswith(b"SQLite format 3")
    except OSError:
        return False


def database_has_results_table(path: str) -> bool:
    if not is_sqlite_file(path):
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='results' LIMIT 1"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False


def list_managed_databases(active_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List .db files in the managed notebooks/ folder (read-only listing)."""
    active_norm = os.path.normpath(active_path) if active_path else None
    items: List[Dict[str, Any]] = []
    root = datasets_dir()
    try:
        names = sorted(os.listdir(root), key=str.lower)
    except OSError:
        names = []
    for name in names:
        if not name.lower().endswith(".db"):
            continue
        path = os.path.normpath(os.path.join(root, name))
        if not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
            size = st.st_size
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except OSError:
            size, mtime = 0, ""
        items.append(
            {
                "name": name,
                "path": path,
                "size_bytes": size,
                "modified": mtime,
                "has_results": database_has_results_table(path),
                "is_active": active_norm == path if active_norm else False,
            }
        )
    return items


def create_new_database(name: str) -> str:
    """Create a new empty .db with a results table. Refuses if the file already exists."""
    path = path_for_new_name(name)
    if os.path.exists(path):
        raise FileExistsError(
            f"A database named '{os.path.basename(path)}' already exists. "
            "Choose another name or open the existing file. Existing files are never overwritten."
        )
    # Create parent dir; create file only via sqlite (new file).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(RESULTS_CREATE_SQL)
        conn.commit()
    finally:
        conn.close()
    # Double-check we did not somehow replace a non-empty pre-existing file
    if not os.path.isfile(path):
        raise RuntimeError("Database file was not created.")
    return path


def is_under_datasets_dir(abs_path: str) -> bool:
    """True if path is inside the managed notebooks/ folder."""
    try:
        root = datasets_dir()
        norm = os.path.normpath(abs_path)
        return os.path.commonpath([norm, root]) == root
    except ValueError:
        return False


def path_to_config_value(abs_path: str) -> str:
    """Store a stable path in config.json.

    - Under the WebApp tree (e.g. notebooks/): path relative to flask_app/
    - Anywhere else on the PC: absolute path (so Browse/open outside notebooks stays reliable)
    """
    abs_path = os.path.normpath(abs_path)
    try:
        rel = os.path.relpath(abs_path, _APP_DIR).replace("\\", "/")
        # Keep relative only when it stays inside the repo (flask_app/.. = WebApp root)
        repo_rel = os.path.relpath(abs_path, _REPO_ROOT).replace("\\", "/")
        if not repo_rel.startswith(".."):
            return rel
    except ValueError:
        pass
    return abs_path


def browse_open_database_dialog() -> Optional[str]:
    """Native OS file dialog (local app). Returns absolute path or None if cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError(
            "File browser is not available in this environment. "
            "Paste the full path to a .db file instead."
        ) from e

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    initial = datasets_dir()
    path = filedialog.askopenfilename(
        title="Open existing analysis database",
        initialdir=initial if os.path.isdir(initial) else _REPO_ROOT,
        filetypes=[
            ("SQLite database", "*.db"),
            ("SQLite database", "*.sqlite"),
            ("SQLite database", "*.sqlite3"),
            ("All files", "*.*"),
        ],
    )
    try:
        root.destroy()
    except Exception:
        pass
    if not path:
        return None
    return os.path.normpath(path)


def file_fingerprint(path: str) -> Tuple[int, int]:
    """(size, mtime_ns) for safety checks — read-only."""
    st = os.stat(path)
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    return int(st.st_size), int(mtime_ns)
