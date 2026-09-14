import sqlite3
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify, abort
from werkzeug.exceptions import HTTPException
import json
import os
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import uuid
from datetime import datetime
from collections import OrderedDict

# Cache for scatter plot images (key=uuid, value=base64 payload) so browser loads by URL and avoids huge JSON
SCATTER_PLOT_CACHE = {}
SCATTER_PLOT_CACHE_MAX = 500

app = Flask(__name__)
# Signs the session cookie and flash messages. Set FLASK_SECRET_KEY in the
# environment for anything reachable beyond localhost. The fallback is a fixed
# development value, not a secret, so local sessions survive a restart.
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or 'dev-insecure-key-set-FLASK_SECRET_KEY'
if not os.environ.get('FLASK_SECRET_KEY'):
    print('WARNING: FLASK_SECRET_KEY is not set; using the insecure development key.')


@app.context_processor
def inject_dataset_context():
    """Expose open database label and default condition set to all templates (B1.5 / B1.6)."""
    try:
        db_path = get_database_path()
        db_label = os.path.basename(db_path) if db_path else ''
        db_ready = database_is_ready()
    except Exception:
        db_label, db_path, db_ready = '', '', False
    try:
        cond_id = unit_config.get_default_condition_set_id()
    except Exception:
        cond_id = None
    return {
        'open_database_label': db_label if db_ready else (db_label or 'none'),
        'open_database_path': db_path,
        'database_is_ready': db_ready,
        'default_condition_set_id': cond_id,
    }


@app.before_request
def _require_active_database():
    """First-run / missing DB: send users to Datasets (except dataset APIs themselves)."""
    ep = request.endpoint or ''
    if ep in (
        'datasets_page',
        'api_datasets_list',
        'api_datasets_create',
        'api_datasets_open',
        'api_datasets_browse',
        'static',
    ):
        return None
    if ep.startswith('api_datasets'):
        return None
    try:
        if database_is_ready():
            return None
    except Exception:
        pass
    if request.path.startswith('/api/'):
        # A fetch() would follow the redirect and read the Datasets *page*, so the
        # caller would only see "Invalid JSON from server".
        return jsonify({'success': False,
                        'message': 'No database is open. Open or create one on the Datasets page.'}), 409
    return redirect(url_for('datasets_page'))


def _api_error_response(exc, status):
    """JSON body for a failed /api/ request, with the cause the caller can act on."""
    if status in (404, 405):
        msg = (f'No such API endpoint: {request.method} {request.path} (HTTP {status}). '
               'If this feature was just added, restart the app and reload the page.')
    elif isinstance(exc, HTTPException):
        msg = f'{exc.name} (HTTP {status}).'
    else:
        msg = f'{type(exc).__name__}: {exc}'
    return jsonify({'success': False, 'message': msg, 'status': status}), status


@app.errorhandler(HTTPException)
def _handle_http_exception(exc):
    """API paths answer JSON even for 404 / 405 / 400; pages keep the HTML page."""
    if not request.path.startswith('/api/'):
        return exc.get_response()
    return _api_error_response(exc, exc.code or 500)


@app.errorhandler(Exception)
def _handle_unexpected_exception(exc):
    """Never let an /api/ route reply with a Flask HTML traceback page.

    The browser reads every API reply with ``resp.json()``, so an uncaught raise
    used to surface as the undiagnosable "Invalid JSON from server" — including a
    raise from a helper called *before* a handler's own try block.
    """
    import traceback
    traceback.print_exc()
    if not request.path.startswith('/api/'):
        raise exc
    return _api_error_response(exc, 500)

# Custom template filters
@app.template_filter('show_time')
def show_time(value):
    """Format time values as integers (no decimal places)"""
    if value is None:
        return '-'
    try:
        return f"{int(float(value))}"
    except (ValueError, TypeError):
        return '-'

@app.template_filter('show_float')
def show_float(value):
    """Format float values with 3 decimal places"""
    if value is None:
        return '-'
    try:
        return f"{float(value):.3f}"
    except (ValueError, TypeError):
        return '-'

@app.template_filter('abs')
def template_abs(value):
    """Absolute value for template (signed deviation coloring)"""
    if value is None:
        return None
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return value

# Define app directory for path resolution
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configuration. Prefer local config.json (gitignored). Clones use the template.
_CONFIG_JSON = os.path.join(_APP_DIR, 'config.json')
_CONFIG_TEMPLATE = os.path.join(_APP_DIR, 'config_template.json')
_config_path = _CONFIG_JSON if os.path.exists(_CONFIG_JSON) else _CONFIG_TEMPLATE
with open(_config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Resolve data_dir against this script, not the process cwd (run_app.ps1 cds to flask_app,
# but a clone must still work if Python is started from the repo root).
_data_dir = (config.get('data_dir') or '').strip()
if _data_dir and not os.path.isabs(_data_dir):
    config['data_dir'] = os.path.normpath(os.path.join(_APP_DIR, _data_dir))

# Unit / condition configuration (B1): JSON under flask_app/config/
import unit_config  # noqa: E402
import summary_export  # noqa: E402  # T4.2 C2 summary CSV/XLS
import dataset_manager  # noqa: E402  # B1.6 dataset create/open/switch

# Optional monitoring means (HPT air-to-water REV1): Plotdaten column → results column.
# Extras only — they are never added to required_columns / average_columns, so BAM Plotdaten
# without them still insert. 'Real BUH power input' is the real backup-heater meter and is
# never the WebApp BUH; that stays 'Electrical power input BUH' → avg_power_buh.
OPTIONAL_MEAN_DB_COLUMNS = {
    'Lab air temperature (DB)': 'avg_lab_air_db',
    'Lab atmospheric pressure': 'avg_lab_atm_pressure',
    'cp_out_sink': 'avg_cp_out_sink',
    'cp_in_sink': 'avg_cp_in_sink',
    'density_sink': 'avg_density_sink',
    'Compressor Frequency': 'avg_compressor_frequency',
    'Fan Speed': 'avg_fan_speed',
    'Voltage': 'avg_voltage',
    'Current': 'avg_current',
    'Frequency': 'avg_frequency',
    'cos_phi': 'avg_cos_phi',
    'Real BUH power input': 'avg_real_buh_power',
    'T_in_cond': 'avg_t_in_cond',
    'T_out_cond': 'avg_t_out_cond',
    'T_in_evap': 'avg_t_in_evap',
    'T_out_evap': 'avg_t_out_evap',
}


def optional_average_columns():
    """Plotdaten columns averaged as extras.

    Read with a default so a pre-0.1.0 local config.json (no 'optional_average_columns')
    keeps working; unknown names in the config are ignored rather than raising.
    """
    configured = config.get('optional_average_columns', list(OPTIONAL_MEAN_DB_COLUMNS))
    return [c for c in configured if c in OPTIONAL_MEAN_DB_COLUMNS]


def optional_mean_db_columns():
    """results columns written by the optional-means UPDATE, in contract order."""
    return [OPTIONAL_MEAN_DB_COLUMNS[c] for c in optional_average_columns()]


# --- Derived mass flow (one rule for BAM and HPT) ----------------------------
# BAM Plotdaten may omit 'mass flow'; HPT Plotdaten always ship the column but the
# lab may leave it empty. Both mean "not reported", so both are derived the same way.
# Constants live here on purpose: a pre-0.1.0 flask_app/config.json must keep working.
WATER_DENSITY_DEFAULT = 997.0   # kg/m³, water around 25 °C
DENSITY_MIN_PLAUSIBLE = 900.0   # outside this band the value is not liquid water
DENSITY_MAX_PLAUSIBLE = 1100.0  # catches ρ reported in g/cm³ (0.997) instead of kg/m³
# Sink side only, in this order. Water-to-water will add a source density later; that one
# must never be used for the sink mass flow.
DENSITY_COLUMNS = ('density_sink', 'density')
# The "nothing was derived" note. Still reported, so the analyst always knows which path
# ran, but flashed as information rather than as a fallback warning.
MASS_FLOW_MEASURED_NOTICE = "Mass flow: measured values from the file used (no derivation)."


def _numeric_series(df, column):
    """Numeric view of a column (±inf as NaN), or None when the column is absent."""
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _density_for_mass_flow(df):
    """(density [kg/m³] per row, wording for the notice, warnings) for derived mass flow."""
    warnings = []
    seen_unusable = False
    for column in DENSITY_COLUMNS:
        series = _numeric_series(df, column)
        if series is None:
            continue
        if not series.notna().any():
            seen_unusable = True
            continue
        implausible = series.notna() & (
            (series < DENSITY_MIN_PLAUSIBLE) | (series > DENSITY_MAX_PLAUSIBLE)
        )
        if implausible.any():
            warnings.append(
                f"Density column '{column}' has {int(implausible.sum())} value(s) outside "
                f"{DENSITY_MIN_PLAUSIBLE:.0f}–{DENSITY_MAX_PLAUSIBLE:.0f} kg/m³ "
                f"(e.g. {float(series[implausible].iloc[0]):g} — g/cm³ instead of kg/m³?); "
                f"{WATER_DENSITY_DEFAULT:.0f} kg/m³ used for those rows."
            )
            series = series.mask(implausible)
        if not series.notna().any():
            seen_unusable = True
            continue
        wording = f"measured {column} (mean {series.mean():.1f} kg/m³)"
        gaps = int(series.isna().sum())
        if gaps:
            wording += (
                f" with {WATER_DENSITY_DEFAULT:.0f} kg/m³ on {gaps} row(s) without density"
            )
        return series.fillna(WATER_DENSITY_DEFAULT), wording, warnings
    reason = "no usable density value in the file" if seen_unusable else "no density column in the file"
    return (
        pd.Series(WATER_DENSITY_DEFAULT, index=df.index, dtype=float),
        f"{WATER_DENSITY_DEFAULT:.0f} kg/m³ ({reason})",
        warnings,
    )


def derive_mass_flow_if_needed(df):
    """Fill missing mass flow from volume flow and density. Returns (df, notices).

    Order, identical for BAM and HPT:
      1. measured 'mass flow' wins and is never overwritten — zeros are measurements;
      2. rows without a measured value get volume flow [m³/h] / 3600 × ρ [kg/m³];
      3. ρ from 'density_sink', else 'density', else 997 kg/m³ — per row, so a partly
         filled density column is used where it is finite.
    An absent column and an all-empty column are the same case. The frame is changed in
    place for analysis only; nothing is written back to the Excel file. notices always
    names the path that ran, so derivation is never silent.
    """
    notices = []
    if df is None:
        return df, notices

    measured = _numeric_series(df, 'mass flow')
    gaps = measured.isna() if measured is not None else None
    if measured is not None and not gaps.any():
        notices.append(MASS_FLOW_MEASURED_NOTICE)
        return df, notices

    volume = _numeric_series(df, 'volume flow')
    if volume is None:
        state = "is not in the file" if measured is None else "is empty in the file"
        notices.append(f"Mass flow {state} and there is no volume flow to derive it from.")
        return df, notices

    density, density_wording, warnings = _density_for_mass_flow(df)
    derived = volume / 3600.0 * density
    if measured is None:
        gaps = pd.Series(True, index=df.index)
        df['mass flow'] = derived
    else:
        df['mass flow'] = measured.where(~gaps, derived)
    filled = int((gaps & derived.notna()).sum())

    if not filled:
        state = "was not in the file" if measured is None else "is empty in the file"
        notices.append(
            f"Mass flow {state} and volume flow has no values either — mass flow stays empty."
        )
    elif measured is None or not measured.notna().any():
        state = "was not in the file" if measured is None else "was empty in the file"
        notices.append(
            f"Mass flow {state} — derived from volume flow × {density_wording}. "
            "Prefer measured mass flow when available."
        )
        df.attrs['mass_flow_derived'] = True
    else:
        notices.append(
            f"Mass flow: {filled} of {len(gaps)} rows were empty and derived from "
            f"volume flow × {density_wording}; the measured values are unchanged."
        )
        df.attrs['mass_flow_derived'] = True
    notices.extend(warnings)
    return df, notices


# --- Means column defaults (display only; does not change insert / COP / BUH math) ---
MEANS_STANDARD_COLUMNS = ('file_name', 'data_set', 'start_time', 'end_time')
MEANS_BLOCK_A_COLUMNS = (
    'profile_id', 'HP_ID', 'Lab_ID', 'test_cond', 'climate', 'application',
    'flow_config', 'COP_dataset', 'Indicator', 'Notes',
)
MEANS_BLOCK_B_COLUMNS = (
    'avg_heating_capacity_uncorr', 'avg_power_input_uncorr', 'cop_uncorr',
    'avg_heating_capacity_corr', 'avg_power_input_corr', 'cop_corr',
    'avg_heating_capacity_bam_corr', 'avg_power_input_bam_corr', 'cop_bam_corr',
    'QCorrwBUH', 'PCorrwBUH', 'COPCorrwBUH', 'Ts_buh',
    'avg_power_buh',
)
MEANS_BLOCK_C_COLUMNS = (
    'avg_t_db', 'avg_t_wb', 'avg_t_supply', 'avg_t_return_emu', 'avg_t_return_calc',
    'avg_volume_flow', 'avg_mass_flow', 'avg_dp',
    'avg_t_b_calc', 'avg_t_h_calc', 'avg_q_hb', 'avg_q_ba',
)
MEANS_DEFAULT_VISIBLE_COLUMNS = frozenset(
    MEANS_STANDARD_COLUMNS + MEANS_BLOCK_A_COLUMNS + MEANS_BLOCK_B_COLUMNS + MEANS_BLOCK_C_COLUMNS
)
MEANS_DEFAULT_VISIBLE_COLUMNS_L = frozenset(c.lower() for c in MEANS_DEFAULT_VISIBLE_COLUMNS)
MEANS_HIDDEN_PREFIXES = ('dev_', 'dtreturn_cache_', 'gutegrad_', 'cop_carnot_', 'pump_correction_')
MEANS_HIDDEN_EXPLICIT = frozenset({
    'QUncorrwBUH', 'PUncorrwBUH', 'COPUncorrwBUH',
    'avg_t_sup_buh',
    'avg_adj_dp', 'avg_P_hyd', 'avg_eff_pump', 'avg_qcorr_bam', 'avg_pcorr_bam',
    'avg_power_buh_from_ts', 'powerbuh',
    'scatter_dataset', 'Test_label', 'DPfix',
    'avg_t_mean_log', 't_mean_from_avgs', 'avg_dt_ln',
    'cycle_type', 'cycle_start_marker',
    'pelec_transition_time', 'pelec_transition_source', 'pelec_detect_failed',
})
MEANS_HIDDEN_EXPLICIT_L = frozenset(c.lower() for c in MEANS_HIDDEN_EXPLICIT)
_IMPORT_NOTICE_FILE_RE = re.compile(r'^(.+\.(?:xlsx|xls|csv)):\s*(.*)$', re.IGNORECASE)
_IMPORT_NOTICE_DENSITY_RE = re.compile(
    r'volume flow\s*[×x]\s*(.+?)(?:\.\s*Prefer|;|$)', re.IGNORECASE
)


def _results_column_map(column_names):
    """Lowercase → actual SQLite column name (results names are not consistently cased)."""
    return {str(c).lower(): c for c in column_names if c}


def resolve_results_column(name, column_names, name_map=None):
    """Match a JSON / UI column name to the live results table, ignoring case."""
    if not name:
        return None
    if name in column_names:
        return name
    mapping = name_map if name_map is not None else _results_column_map(column_names)
    return mapping.get(str(name).lower())


def is_means_column_default_visible(col):
    """True when a Means column should show if it is missing from column_visibility.json."""
    if not col:
        return False
    cl = col.lower()
    if cl in {c.lower() for c in dataset_manager.OPTIONAL_MONITORING_MEAN_COLUMNS}:
        return False
    if col.startswith(MEANS_HIDDEN_PREFIXES):
        return False
    if cl in MEANS_HIDDEN_EXPLICIT_L:
        return False
    if cl in MEANS_DEFAULT_VISIBLE_COLUMNS_L:
        return True
    return True


def means_column_is_visible(col, column_visibility, deviation_columns=None):
    """Resolve Means visibility: JSON override, else the committed default."""
    if column_visibility:
        cl = str(col).lower()
        for key, val in column_visibility.items():
            if str(key).lower() == cl:
                return bool(val)
    if deviation_columns and col in deviation_columns:
        return False
    return is_means_column_default_visible(col)


def _tag_notice_with_file(file_name, notice):
    """Prefix a raw notice with the Plotdaten file so collapse can group by file."""
    text = (notice or '').strip()
    if not text or not file_name:
        return text
    prefix = f"{file_name}: "
    if text.startswith(prefix) or text.lower().startswith(f"{str(file_name).lower()}:"):
        return text
    return prefix + text


def _split_notice_file(text):
    raw = (text or '').strip()
    match = _IMPORT_NOTICE_FILE_RE.match(raw)
    if match:
        return match.group(1), match.group(2)
    return None, raw


def _classify_import_notice(text):
    file_name, body = _split_notice_file(text)
    body_l = (body or '').lower()
    if body == MASS_FLOW_MEASURED_NOTICE or body_l.startswith('mass flow: measured'):
        return file_name, 'mass_flow_measured', body
    if 'density column' in body_l and 'outside' in body_l:
        return file_name, 'density_implausible', body
    if ('virtual' in body_l and 'real' in body_l) or 'both virtual' in body_l:
        return file_name, 'both_buh', body
    if 'lab-corrected' in body_l:
        return file_name, 'lab_corr_missing', body
    if 'mass flow' in body_l and 'derived' in body_l:
        return file_name, 'mass_flow_derived', body
    if 'mass flow' in body_l:
        return file_name, 'other', body
    return file_name, 'other', body


def _density_wording_from_notice(body):
    match = _IMPORT_NOTICE_DENSITY_RE.search(body or '')
    if match:
        return match.group(1).strip().rstrip('.')
    return 'density'


def collapse_import_notices(raw_notices):
    """Collapse per-cycle raw notices into one line per (file, kind) for flash / modal.

    Raw strings from calculate_and_insert are kept until this helper runs. Lab-corr
    Q/P kW figures are dropped from the displayed line; N is the cycle count.
    """
    grouped = OrderedDict()
    file_order = []
    for raw in raw_notices or []:
        if not raw:
            continue
        file_name, kind, body = _classify_import_notice(raw)
        if file_name and file_name not in file_order:
            file_order.append(file_name)
        key = (file_name, kind)
        grouped.setdefault(key, []).append(body)

    # Generics without a filename join the sole file in this batch, if there is one.
    if len(file_order) == 1:
        only = file_order[0]
        remapped = OrderedDict()
        for (file_name, kind), bodies in grouped.items():
            use_file = file_name or only
            remapped.setdefault((use_file, kind), []).extend(bodies)
        grouped = remapped

    collapsed = []
    for (file_name, kind), bodies in grouped.items():
        label = file_name or 'file'
        if kind == 'lab_corr_missing':
            n_cycles = sum(
                1 for b in bodies
                if 'q=' in b.lower() or 'stored' in b.lower()
            )
            if n_cycles == 0:
                n_cycles = len(bodies)
            unit = 'cycle' if n_cycles == 1 else 'cycles'
            collapsed.append({
                'category': 'warning',
                'message': (
                    f"{label}: lab-corrected Q/P missing on {n_cycles} {unit} — "
                    "stored BAM pump-corrected averages."
                ),
            })
        elif kind == 'mass_flow_derived':
            wording = _density_wording_from_notice(bodies[0])
            collapsed.append({
                'category': 'warning',
                'message': (
                    f"{label}: mass flow not reported — derived from volume flow × {wording}."
                ),
            })
        elif kind == 'mass_flow_measured':
            collapsed.append({
                'category': 'info',
                'message': f"{label}: {MASS_FLOW_MEASURED_NOTICE}",
            })
        elif kind == 'density_implausible':
            collapsed.append({
                'category': 'warning',
                'message': f"{label}: {bodies[0]}",
            })
        elif kind == 'both_buh':
            collapsed.append({
                'category': 'warning',
                'message': f"{label}: {bodies[0]}",
            })
        else:
            seen = set()
            for body in bodies:
                if body in seen:
                    continue
                seen.add(body)
                collapsed.append({
                    'category': 'warning',
                    'message': f"{label}: {body}" if file_name and not body.startswith(str(file_name)) else body,
                })
    return collapsed


def collapsed_notice_messages(raw_notices):
    """Display strings only (cycle-extract status / JSON)."""
    return [item['message'] for item in collapse_import_notices(raw_notices)]


# Function to extract letter from filename
def extract_letter(file_name):
    return file_name.split('_')[2][0] if len(file_name.split('_')) > 2 else None

# Function to determine BUH cap value (profile first; legacy cap_values with warning; default with warning)
def determine_cap(file_name, hp_id=None, profile_id=None):
    cap, status = unit_config.resolve_buh_cap(
        file_name=file_name,
        hp_id=hp_id,
        profile_id=profile_id,
        fallback_cap_values=config.get('cap_values') or {},
        default_cap=config.get('default_cap'),
    )
    print(f"The cap value for {file_name} is {cap} ({status})")
    return cap if cap is not None else config.get('default_cap', 6)

def _normalize_hp_id(value):
    """Collapse sub-unit hp ids onto their base heat pump number.

    e.g. '1a' -> '1', '1b' -> '1', '1' -> '1', '2' -> '2'. Values that are not a
    number with an optional single trailing letter are returned unchanged.
    Used so sub-unit ids group under their base heat pump on the dTreturn Insights page.
    """
    return unit_config.normalize_hp_id(value)


def _list_analysis_units(conn):
    """Unique units in the open results table (profile first, then hp_id, then file)."""
    cols = {r[1] for r in conn.execute('PRAGMA table_info(results)')}
    select = ['file_name']
    if 'profile_id' in cols:
        select.append('profile_id')
    if 'HP_ID' in cols:
        select.append('HP_ID')
    elif 'hp_id' in cols:
        select.append('hp_id')
    rows = conn.execute(f"SELECT {', '.join(select)} FROM results").fetchall()
    return unit_config.list_units_from_entries([dict(r) for r in rows])


def get_database_path():
    """Absolute path to the SQLite results database (configurable; B1.5 / B1.6)."""
    configured = (config.get('database_path') or '').strip()
    if configured:
        path = configured if os.path.isabs(configured) else os.path.normpath(os.path.join(_APP_DIR, configured))
    else:
        path = os.path.normpath(os.path.join(_APP_DIR, '..', 'notebooks', 'data_analysis.db'))
    return os.path.normpath(path)


def get_database_label():
    """Short label for UI: which dataset file is open."""
    return os.path.basename(get_database_path())


def database_is_ready() -> bool:
    """True when the configured DB file exists and has a results table (no auto-create)."""
    path = get_database_path()
    return dataset_manager.database_has_results_table(path)


def get_db_connection():
    """Open the active database. Does not create a missing file (avoids empty accidental DBs)."""
    path = get_database_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Database file not found: {path}. Open or create a database on the Datasets page."
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Expose hp-id normalization to SQL (e.g. WHERE norm_hp(HP_ID)=?).
    try:
        conn.create_function("norm_hp", 1, _normalize_hp_id)
    except Exception:
        pass
    return conn


def _clear_runtime_caches_after_db_switch() -> None:
    """Drop in-memory plot caches so pages cannot show the previous dataset."""
    try:
        SCATTER_PLOT_CACHE.clear()
    except Exception:
        pass


def save_database_path_to_config(abs_path: str) -> str:
    """Persist database_path in local config.json. Never writes config_template.json."""
    global config
    abs_path = os.path.normpath(abs_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Database file not found: {abs_path}")
    rel = dataset_manager.path_to_config_value(abs_path)
    config['database_path'] = rel
    # Always the gitignored local file — clones must not dirty the committed template.
    out_path = _CONFIG_JSON
    existing = {}
    seed_path = out_path if os.path.isfile(out_path) else _CONFIG_TEMPLATE
    if os.path.isfile(seed_path):
        with open(seed_path, 'r', encoding='utf-8') as fh:
            existing = json.load(fh)
    existing['database_path'] = rel
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(existing, fh, indent=4)
        fh.write('\n')
    return abs_path


def ensure_results_table_exists() -> None:
    """CREATE TABLE IF NOT EXISTS results on the *active* DB only (never DROP)."""
    conn = get_db_connection()
    try:
        conn.execute(dataset_manager.RESULTS_CREATE_SQL)
        conn.commit()
    finally:
        conn.close()


def prepare_active_database() -> None:
    """Ensure auxiliary schema on the active DB using IF NOT EXISTS / ADD COLUMN only."""
    ensure_results_table_exists()
    init_recalc_history_table()
    init_hp_design_table()
    ensure_cycle_periods_table()
    ensure_dtreturn_exclusions_table()
    try:
        ensure_cycle_extraction_suggestions_table()
    except Exception:
        pass
    init_deviation_columns()
    try:
        ensure_results_columns(dataset_manager.ANALYSIS_MEAN_COLUMNS)
    except Exception as e:
        print(f"Analysis mean columns ensure skipped: {e}")
    try:
        init_display_order_column()
    except Exception:
        pass
    try:
        init_scatter_dataset_column()
    except Exception:
        pass
    try:
        backfill_profile_and_condition_ids(only_null=True)
    except Exception as e:
        print(f"Profile/condition id backfill skipped: {e}")


def activate_database(abs_path: str, *, prepare: bool = True) -> str:
    """Point the app at an existing DB file. Does not modify another DB's contents."""
    abs_path = dataset_manager.resolve_db_path(abs_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Database file not found: {abs_path}")
    if not dataset_manager.is_sqlite_file(abs_path):
        raise ValueError(f"Not a SQLite database file: {abs_path}")
    # Fingerprint before schema ensure — must still match after (size may grow only via ADD COLUMN)
    before = dataset_manager.file_fingerprint(abs_path)
    save_database_path_to_config(abs_path)
    _clear_runtime_caches_after_db_switch()
    if prepare:
        prepare_active_database()
    after = dataset_manager.file_fingerprint(abs_path)
    # Allow size growth from ADD COLUMN / new empty aux tables; never allow shrinkage.
    if after[0] < before[0]:
        raise RuntimeError(
            f"Refusing to continue: database file shrank unexpectedly ({before[0]} → {after[0]} bytes). "
            f"Path: {abs_path}"
        )
    return abs_path


def session_note_inserted_rowids(rowids) -> None:
    """Remember new results rowids so the home page can open the metadata review modal."""
    if not rowids:
        return
    clean = []
    for x in rowids:
        try:
            clean.append(int(x))
        except (TypeError, ValueError):
            continue
    if not clean:
        return
    prev = list(session.get('last_inserted_rowids') or [])
    seen = set(prev)
    for r in clean:
        if r not in seen:
            prev.append(r)
            seen.add(r)
    session['last_inserted_rowids'] = prev


def _profile_picker_items():
    """Compact profile list for Cycle Extract / review-new-entries pickers."""
    items = []
    for p in unit_config.list_profiles():
        pid = p.get('profile_id')
        if not pid:
            continue
        items.append({
            'profile_id': pid,
            'display_name': p.get('display_name') or pid,
            'filename_patterns': list(p.get('filename_patterns') or []),
            'default_climate': p.get('default_climate'),
            'default_application': p.get('default_application'),
            'flow_mode': p.get('flow_mode'),
            'unit_type': p.get('unit_type'),
            'hp_id': unit_config.hp_id_from_profile(pid),
        })
    return items


def _stamp_profile_on_rowids(rowids, profile_id):
    """Write profile_id, default condition set, and hp_id onto new results rows."""
    if not rowids:
        return
    pid = (str(profile_id).strip() if profile_id not in (None, '') else '') or None
    ensure_results_columns({'profile_id': 'TEXT', 'condition_set_id': 'TEXT', 'hp_id': 'INTEGER'})
    default_cs = unit_config.get_default_condition_set_id()
    derived_hp_id = unit_config.hp_id_from_profile(pid) if pid else None
    conn = get_db_connection()
    try:
        for rid in rowids:
            conn.execute(
                "UPDATE results SET profile_id = ?, "
                "condition_set_id = COALESCE(NULLIF(condition_set_id, ''), ?), "
                "hp_id = CASE WHEN hp_id IS NULL OR TRIM(CAST(hp_id AS TEXT)) = '' THEN ? ELSE hp_id END "
                "WHERE rowid = ?",
                (pid, default_cs, derived_hp_id, int(rid)),
            )
        conn.commit()
    finally:
        conn.close()


def resolve_selected_rows(conn, file_names, data_sets, row_ids=None):
    """
    Map UI selection to database rows in order.

    The same Excel file and data_set can appear more than once with different
    start/end times. Lookup by (file_name, data_set) alone is ambiguous
    (SQLite returns an arbitrary match). When the form posts row_ids from the
    table row's SQLite rowid, use that first.

    ``rowid`` is selected explicitly so callers that identify an entry by it
    (Guideline windows) get it back even for a row matched by file/data_set.
    """
    if row_ids is None:
        row_ids = []
    if len(file_names) != len(data_sets):
        return []
    out = []
    for i, file_name in enumerate(file_names):
        data_set = data_sets[i]
        rec = None
        if i < len(row_ids) and row_ids[i] not in (None, '', 'undefined'):
            try:
                rid = int(row_ids[i])
                rec = conn.execute('SELECT rowid, * FROM results WHERE rowid = ?', (rid,)).fetchone()
            except (ValueError, TypeError):
                rec = None
        if rec is None:
            rec = conn.execute(
                'SELECT rowid, * FROM results WHERE file_name = ? AND data_set = ?',
                (file_name, data_set)
            ).fetchone()
        if rec:
            out.append(dict(rec))
    return out


def ensure_results_columns(columns: dict[str, str]) -> None:
    """Ensure `results` table contains the requested columns."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(results)")
        existing = {row[1] for row in cur.fetchall()}
        for col_name, col_type in columns.items():
            if col_name in existing:
                continue
            try:
                cur.execute(f'ALTER TABLE results ADD COLUMN "{col_name}" {col_type}')
                existing.add(col_name)
                print(f'Added results column: {col_name} ({col_type})')
            except Exception as e:
                print(f'Could not add column {col_name}: {e}')
        conn.commit()
    finally:
        conn.close()


def infer_metadata_from_filename(file_name: str) -> dict:
    """
    Infer metadata fields from filename + defaults.
    Example: <unit>_<lab>_B_VarFl.xlsx
    """
    base = os.path.basename(file_name or "")
    lower = base.lower()

    hp_id = None
    lab_id = None
    test_label = None

    # Don't use word-boundaries here because filenames often have underscores (word chars),
    # e.g. "hp<n>_lab<nn>_..." would fail \b...\b.
    m_hp = re.search(r'hp\s*0*([0-9]+)', lower, re.IGNORECASE)
    if m_hp:
        try:
            hp_id = int(m_hp.group(1))
        except Exception:
            hp_id = None

    m_lab = re.search(r'lab\s*0*([0-9]+)', lower, re.IGNORECASE)
    if m_lab:
        try:
            lab_id = int(m_lab.group(1))
        except Exception:
            lab_id = None

    # Prefer the token between underscores (e.g. _B_)
    parts = re.split(r'[_\-\s]+', os.path.splitext(base)[0])
    for p in parts:
        if re.fullmatch(r'[A-Ga-g]', p or ""):
            test_label = p.upper()
            break

    test_cond = test_label

    climate = None
    application = None
    for p in parts:
        pu = (p or "").upper()
        if pu in ("LT", "MT", "HT"):
            application = pu
        elif pu in ("AVERAGE", "WARMER", "COLDER"):
            climate = pu.capitalize()

    profile, _ = unit_config.resolve_profile(file_name=base, hp_id=hp_id)
    if profile:
        climate = climate or profile.get("default_climate")
        application = application or profile.get("default_application")

    # Flow from filename tokens only (BAM: _VarFl_ / _StdFl_ / _off_).
    # Do not substring-match: 'average' contains 'var'. If no token, leave empty
    # so the unit profile's flow_mode decides (HPT_RRT1 = variable, RRT2 = fixed).
    flow_config = None
    for t in (p.lower() for p in parts if p):
        if t == 'off' or t.startswith('off'):
            flow_config = 'off'
            break
        if t in ('var', 'variable') or t.startswith('varfl'):
            flow_config = 'var'
            break
        if t in ('std', 'standard') or t.startswith('stdfl'):
            flow_config = 'std'
            break
        if t in ('fixed', 'fix') or t.startswith('fixfl'):
            flow_config = 'fixed'
            break

    return {
        # inferred
        'hp_id': hp_id,
        'lab_id': lab_id,
        'test_label': test_label,
        'test_cond': test_cond,
        'climate': climate,
        'application': application,
        'flow_config': flow_config,
        # defaults
        'cop_dataset': 'YES',
        'scatter_dataset': 'YES',
        'indicator': 'Pelec',
        'notes': 'drop',
        'dpfix': None,
        # derived-calculation enabler
        'dev_test_condition': test_cond,
    }


def _insert_column_names_from_config():
    """Column names populated by config['insert_query'] (do not overwrite after insert)."""
    q = config.get('insert_query', '')
    m = re.search(r'INSERT\s+INTO\s+\w+\s*\(([^)]+)\)', q, re.IGNORECASE)
    if not m:
        return frozenset()
    return frozenset(
        p.strip().strip('"').strip("'") for p in m.group(1).split(',') if p.strip()
    )


def preserve_row_metadata_after_key_change(old_row, new_file_name, new_data_set, new_start_time, new_end_time):
    """
    After delete+insert (e.g. edit changed file or dataset), copy user/metadata columns
    from the old row onto the new row. Insert already filled file, dataset, times, and averages.
    """
    if not old_row:
        return
    insert_cols = _insert_column_names_from_config()
    if not insert_cols:
        print('WARNING: could not parse insert_query; skipping metadata preservation')
        return
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            'SELECT rowid FROM results WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?',
            (new_file_name, new_data_set, new_start_time, new_end_time),
        )
        found = cur.fetchone()
        if not found:
            print('preserve_row_metadata_after_key_change: new row not found')
            return
        rowid = found['rowid']

        cur.execute('PRAGMA table_info(results)')
        pragma_cols = [r[1] for r in cur.fetchall()]

        # Optional monitoring means belong to the new window and were just recomputed,
        # so they are not user metadata to carry over.
        skip = (
            set(insert_cols)
            | {'rowid', 'time_series_data', 'has_time_series'}
            | set(dataset_manager.OPTIONAL_MONITORING_MEAN_COLUMNS)
        )
        for col in pragma_cols:
            if col in skip or col.startswith('dev_'):
                continue
            if col not in old_row:
                continue
            val = old_row[col]
            if val is None:
                continue
            qcol = col.replace('"', '""')
            cur.execute(f'UPDATE results SET "{qcol}" = ? WHERE rowid = ?', (val, rowid))
        conn.commit()
    except Exception as e:
        print(f'preserve_row_metadata_after_key_change: {e}')
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


# Indoor air used for logarithmic mean water temperature (ecodesign / EN 442).
T_AIR_INDOOR_C = 20.0
_LEAVING_WATER_COLS = ('Ts Buh', 'Ts_buh', 'T_supply_afterBUH', 'T_supply')


def _leaving_and_return_series(df: pd.DataFrame):
    """Leaving water prefers post-BUH supply when present; return is T_return_emu."""
    t_r = pd.to_numeric(df['T_return_emu'], errors='coerce') if 'T_return_emu' in df.columns else None
    t_l = None
    leaving_col = None
    for col in _LEAVING_WATER_COLS:
        if col in df.columns:
            t_l = pd.to_numeric(df[col], errors='coerce')
            leaving_col = col
            break
    return t_l, t_r, leaving_col


def logarithmic_mean_water_temperature(t_l, t_r, t_air: float = T_AIR_INDOOR_C) -> np.ndarray:
    """Absolute logarithmic mean water temperature (°C). NaN where LMTD vs t_air is undefined."""
    t_l = np.asarray(t_l, dtype=float)
    t_r = np.asarray(t_r, dtype=float)
    d_l = t_l - t_air
    d_r = t_r - t_air
    spread = t_l - t_r
    out = np.full(np.shape(t_l), np.nan, dtype=float)
    finite = np.isfinite(t_l) & np.isfinite(t_r)
    equal = finite & (np.abs(spread) < 1e-9) & (d_l > 0)
    ok = finite & (d_l > 0) & (d_r > 0) & (np.abs(spread) >= 1e-9)
    out[equal] = t_l[equal]
    with np.errstate(divide='ignore', invalid='ignore'):
        out[ok] = t_air + spread[ok] / np.log(d_l[ok] / d_r[ok])
    return out


def t_mean_from_average_pair(t_l_bar, t_r_bar, t_air: float = T_AIR_INDOOR_C):
    """LMTD from period-average leaving/return (stationary rating check)."""
    try:
        if t_l_bar is None or t_r_bar is None:
            return None
        v = logarithmic_mean_water_temperature([float(t_l_bar)], [float(t_r_bar)], t_air)[0]
        return None if not np.isfinite(v) else float(v)
    except (TypeError, ValueError):
        return None


def add_t_mean_log_column(df: pd.DataFrame, t_air: float = T_AIR_INDOOR_C) -> pd.DataFrame:
    """Add T_mean_log and dT_ln time-series columns (does nothing if temps are missing)."""
    t_l, t_r, _ = _leaving_and_return_series(df)
    if t_l is None or t_r is None:
        return df
    t_mean = logarithmic_mean_water_temperature(t_l.to_numpy(), t_r.to_numpy(), t_air)
    df['T_mean_log'] = t_mean
    df['dT_ln'] = df['T_mean_log'] - t_air
    return df


def compute_t_mean_period_stats(df_filtered: pd.DataFrame, avg_t_sup_buh=None, avg_values=None):
    """Return (avg_t_mean_log, t_mean_from_avgs, avg_dt_ln) for a cycle window."""
    add_t_mean_log_column(df_filtered)
    avg_t_mean_log = None
    avg_dt_ln = None
    if 'T_mean_log' in df_filtered.columns:
        s = pd.to_numeric(df_filtered['T_mean_log'], errors='coerce')
        if s.notna().any():
            avg_t_mean_log = float(s.mean())
            avg_dt_ln = avg_t_mean_log - T_AIR_INDOOR_C
    t_l, t_r, _ = _leaving_and_return_series(df_filtered)
    t_l_bar = float(t_l.mean()) if t_l is not None and t_l.notna().any() else None
    t_r_bar = float(t_r.mean()) if t_r is not None and t_r.notna().any() else None
    if t_l_bar is None and avg_t_sup_buh is not None:
        t_l_bar = float(avg_t_sup_buh)
    if t_l_bar is None and avg_values and avg_values.get('T_supply') is not None:
        t_l_bar = float(avg_values['T_supply'])
    if t_r_bar is None and avg_values and avg_values.get('T_return_emu') is not None:
        t_r_bar = float(avg_values['T_return_emu'])
    t_from_avgs = t_mean_from_average_pair(t_l_bar, t_r_bar)
    return avg_t_mean_log, t_from_avgs, avg_dt_ln


def _t_mean_stats_from_df(df_full, stf: float, etf: float):
    """Full-cycle T_mean stats from an Excel sheet window."""
    if df_full is None or 'time_elapsed' not in getattr(df_full, 'columns', []):
        return None, None, None
    df = df_full[(df_full['time_elapsed'] >= stf) & (df_full['time_elapsed'] <= etf)].copy()
    if df.empty:
        return None, None, None
    return compute_t_mean_period_stats(df)


def _write_results_t_mean(conn, rid, df_full, stf: float, etf: float):
    """Store avg_t_mean_log / t_mean_from_avgs / avg_dt_ln on the results row."""
    avg_t_mean_log, t_mean_from_avgs, avg_dt_ln = _t_mean_stats_from_df(df_full, stf, etf)
    conn.execute(
        "UPDATE results SET avg_t_mean_log=?, t_mean_from_avgs=?, avg_dt_ln=? WHERE rowid=?",
        (avg_t_mean_log, t_mean_from_avgs, avg_dt_ln, rid),
    )
    return avg_t_mean_log, t_mean_from_avgs, avg_dt_ln


# --- Single-column recomputation helpers ---
def compute_single_column_value(df_filtered: pd.DataFrame, column_name: str) -> float | None:
    """Compute a single result column value from filtered time-series.
    Supports simple means and COP variants using raw/corrected columns if present.
    Returns None if not computable.
    """
    try:
        if column_name == 'avg_t_db' and 'T_outdoor (DB)' in df_filtered.columns:
            return float(df_filtered['T_outdoor (DB)'].mean())
        if column_name == 'avg_t_wb' and 'T_outdoor (WB)' in df_filtered.columns:
            return float(df_filtered['T_outdoor (WB)'].mean())
        if column_name == 'avg_t_supply':
            # Prefer Ts Buh (supply temperature with BUH) over T_supply for deviation analysis
            if 'Ts Buh' in df_filtered.columns:
                return float(df_filtered['Ts Buh'].mean())
            elif 'T_supply' in df_filtered.columns:
                return float(df_filtered['T_supply'].mean())
            return None
        if column_name == 'avg_ts_buh':
            # Explicit BUH supply temp for deviation checks; same as avg_t_supply (Ts Buh when present else T_supply)
            if 'Ts Buh' in df_filtered.columns:
                return float(df_filtered['Ts Buh'].mean())
            if 'Ts_buh' in df_filtered.columns:
                return float(df_filtered['Ts_buh'].mean())
            if 'T_supply' in df_filtered.columns:
                return float(df_filtered['T_supply'].mean())
            return None
        if column_name == 'avg_t_return_emu' and 'T_return_emu' in df_filtered.columns:
            return float(df_filtered['T_return_emu'].mean())
        if column_name == 'avg_t_return_calc' and 'T_return_calc' in df_filtered.columns:
            return float(df_filtered['T_return_calc'].mean())
        if column_name in ('avg_t_mean_log', 'avg_dt_ln', 't_mean_from_avgs'):
            add_t_mean_log_column(df_filtered)
            if column_name == 'avg_t_mean_log':
                if 'T_mean_log' not in df_filtered.columns:
                    return None
                s = pd.to_numeric(df_filtered['T_mean_log'], errors='coerce').dropna()
                return float(s.mean()) if len(s) else None
            if column_name == 'avg_dt_ln':
                t_mean = compute_single_column_value(df_filtered, 'avg_t_mean_log')
                return None if t_mean is None else float(t_mean) - T_AIR_INDOOR_C
            t_l, t_r, _ = _leaving_and_return_series(df_filtered)
            if t_l is None or t_r is None:
                return None
            return t_mean_from_average_pair(t_l.mean(), t_r.mean())
        if column_name == 'avg_t_b_calc' and 'T_B_calc' in df_filtered.columns:
            return float(df_filtered['T_B_calc'].mean())
        if column_name == 'avg_t_h_calc' and 'T_H_calc' in df_filtered.columns:
            return float(df_filtered['T_H_calc'].mean())
        if column_name == 'avg_q_hb' and 'Q_HB' in df_filtered.columns:
            return float(df_filtered['Q_HB'].mean())
        if column_name == 'avg_q_ba' and 'Q_BA' in df_filtered.columns:
            return float(df_filtered['Q_BA'].mean())
        if column_name == 'avg_volume_flow' and 'volume flow' in df_filtered.columns:
            return float(df_filtered['volume flow'].mean())
        if column_name == 'avg_mass_flow':
            # Same fallback as import: measured wins, empty or absent is derived
            df_mass, _ = derive_mass_flow_if_needed(df_filtered.copy())
            series = _numeric_series(df_mass, 'mass flow')
            if series is None or not series.notna().any():
                return None
            return float(series.mean())
        if column_name == 'avg_dp' and 'Pressure difference' in df_filtered.columns:
            return float(df_filtered['Pressure difference'].mean())
        # Heating/Power means and COPs
        if column_name == 'avg_heating_capacity_uncorr' and 'Heating Capacity (without corr)' in df_filtered.columns:
            return float(df_filtered['Heating Capacity (without corr)'].mean())
        if column_name == 'avg_power_input_uncorr' and 'Electric Power Input (without correction)' in df_filtered.columns:
            return float(df_filtered['Electric Power Input (without correction)'].mean())
        if column_name == 'cop_uncorr' and all(c in df_filtered.columns for c in ['Heating Capacity (without corr)', 'Electric Power Input (without correction)']):
            p = df_filtered['Electric Power Input (without correction)'].mean()
            return float(df_filtered['Heating Capacity (without corr)'].mean() / p) if p else None
        if column_name == 'avg_heating_capacity_corr' and 'Heating Capacity (corrected)' in df_filtered.columns:
            return float(df_filtered['Heating Capacity (corrected)'].mean())
        if column_name == 'avg_heating_capacity_corr_wbuh':
            # BUH-corrected heating capacity for deviation checks; prefer QCorrwBUH when present
            if 'QCorrwBUH' in df_filtered.columns:
                return float(df_filtered['QCorrwBUH'].mean())
            if 'Heating Capacity (corrected)' in df_filtered.columns:
                return float(df_filtered['Heating Capacity (corrected)'].mean())
            return None
        if column_name == 'avg_q_corr_wbuh':
            return compute_single_column_value(df_filtered, 'avg_heating_capacity_corr_wbuh')
        if column_name == 'avg_p_corr_wbuh':
            if 'PCorrwBUH' in df_filtered.columns:
                return float(df_filtered['PCorrwBUH'].mean())
            if 'Electric Power Input (corrected)' in df_filtered.columns:
                return float(df_filtered['Electric Power Input (corrected)'].mean())
            return None
        if column_name == 'avg_cop_corr_wbuh':
            if 'COPCorrwBUH' in df_filtered.columns:
                return float(df_filtered['COPCorrwBUH'].mean())
            q = compute_single_column_value(df_filtered, 'avg_q_corr_wbuh')
            p = compute_single_column_value(df_filtered, 'avg_p_corr_wbuh')
            if q is not None and p not in (None, 0):
                return float(q) / float(p)
            return None
        if column_name == 'avg_power_input_corr' and 'Electric Power Input (corrected)' in df_filtered.columns:
            return float(df_filtered['Electric Power Input (corrected)'].mean())
        if column_name == 'cop_corr' and all(c in df_filtered.columns for c in ['Heating Capacity (corrected)', 'Electric Power Input (corrected)']):
            p = df_filtered['Electric Power Input (corrected)'].mean()
            return float(df_filtered['Heating Capacity (corrected)'].mean() / p) if p else None
        # Not implemented yet
        return None
    except Exception:
        return None


# --- Recalculation history support ---
def init_recalc_history_table():
    """Ensure the recalculation history table exists."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recalc_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                data_set REAL NOT NULL,
                scope TEXT DEFAULT 'all',
                status TEXT NOT NULL,
                message TEXT,
                duration_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def init_display_order_column():
    """Add display_order column if it doesn't exist and initialize it for existing rows.
    NOTE: This function does NOT normalize display_order - normalization should only happen
    when explicitly needed (e.g., before a move operation), not on every page load."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        column_just_added = False
        if 'display_order' not in columns:
            # Add the column
            cursor.execute("ALTER TABLE results ADD COLUMN display_order INTEGER")
            conn.commit()
            print("Added display_order column")
            column_just_added = True
        
        # Ensure all rows have display_order set (use rowid if NULL)
        cursor.execute("UPDATE results SET display_order = rowid WHERE display_order IS NULL")
        rows_updated = cursor.rowcount
        if rows_updated > 0:
            conn.commit()
            print(f"Initialized display_order for {rows_updated} rows that had NULL values")
        
        # Only normalize if we just added the column (first time setup)
        if column_just_added:
            # Normalize display_order to be sequential (1, 2, 3, ...) based on current order
            print("Column just added - normalizing display_order for first time")
            normalize_display_order(conn, cursor)
            conn.commit()
        
    except Exception as e:
        print(f"Error initializing display_order column: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

def normalize_display_order(conn=None, cursor=None):
    """Normalize display_order to be sequential (1, 2, 3, ...) based on current order."""
    close_conn = False
    try:
        if conn is None:
            conn = get_db_connection()
            close_conn = True
        if cursor is None:
            cursor = conn.cursor()
        
        # Get all entries ordered by current display_order
        cursor.execute("SELECT rowid FROM results ORDER BY display_order ASC, rowid ASC")
        rows = cursor.fetchall()
        
        # Assign sequential numbers starting from 1
        for index, row in enumerate(rows, start=1):
            cursor.execute("UPDATE results SET display_order = ? WHERE rowid = ?", (index, row['rowid']))
        
        if close_conn:
            conn.commit()
            conn.close()
        print(f"Normalized display_order for {len(rows)} entries")
    except Exception as e:
        print(f"Error normalizing display_order: {e}")
        import traceback
        traceback.print_exc()
        if close_conn and 'conn' in locals():
            conn.close()

def log_recalc_event(file_name: str, data_set, status: str, scope: str = 'all', message: str = None, duration_ms: int = None) -> None:
    """Insert a row into recalc_history."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO recalc_history (file_name, data_set, scope, status, message, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_name, data_set, scope, status, message, duration_ms)
        )
        conn.commit()
    finally:
        conn.close()

@app.route('/recalc_history')
def recalc_history():
    """Simple UI to list recalculation history with optional filters."""
    try:
        file_filter = request.args.get('file_name', '').strip()
        status_filter = request.args.get('status', '').strip()
        scope_filter = request.args.get('scope', '').strip()

        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT file_name, data_set, scope, status, message, duration_ms, created_at FROM recalc_history"
        clauses = []
        params = []
        if file_filter:
            clauses.append("file_name = ?")
            params.append(file_filter)
        if status_filter:
            clauses.append("status = ?")
            params.append(status_filter)
        if scope_filter:
            clauses.append("scope = ?")
            params.append(scope_filter)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT 500"

        rows = cursor.execute(query, params).fetchall()
        results = [dict(r) for r in rows]
        conn.close()

        return render_template('recalc_history.html', history=results, file_filter=file_filter, status_filter=status_filter, scope_filter=scope_filter)
    except Exception as e:
        flash(f'Error loading recalculation history: {str(e)}', 'danger')
        return redirect(url_for('index'))

def load_time_series_data(file_name, start_time, end_time):
    """Load time series data from Excel file for plotting"""
    try:
        file_path = os.path.join(config['data_dir'], file_name)
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_excel(file_path)
        if 'time_elapsed' not in df.columns:
            return None
        
        # Filter by time range
        df_filtered = df[(df['time_elapsed'] >= start_time) & (df['time_elapsed'] <= end_time)].copy()
        
        if df_filtered.empty:
            return None
            
        return df_filtered
    except Exception as e:
        print(f"Error loading time series data for {file_name}: {e}")
        return None

def import_time_series(file_name, data_set, start_time, end_time):
    """Import time series data for a single record"""
    try:
        # Load time series data
        df = load_time_series_data(file_name, start_time, end_time)
        if df is None:
            return False
        
        # Select only essential columns for plotting
        essential_columns = [
            'time_elapsed', 'T_outdoor (DB)', 'T_outdoor (WB)', 
            'T_supply', 'Ts Buh', 'T_return_emu', 'Heating Capacity (corrected)',
            'Electric Power Input (corrected)', 'Pressure difference', 'volume flow'
        ]
        
        # Keep only columns that exist
        available_columns = [col for col in essential_columns if col in df.columns]
        df_plot = df[available_columns].copy()
        
        # Convert to JSON
        time_series_json = df_plot.to_json(orient='records', date_format='iso')
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE results SET has_time_series = 1, time_series_data = ? WHERE file_name = ? AND data_set = ?",
            (time_series_json, file_name, data_set)
        )
        conn.commit()
        conn.close()
        
        print(f"Imported time series for {file_name}, dataset {data_set}")
        return True
        
    except Exception as e:
        print(f"Error importing time series for {file_name}: {e}")
        return False

def calculate_derived_quantities(df):
    """Calculate derived quantities exactly like in the notebook"""
    df_calc = df.copy()
    print(f"Calculating derived quantities. Available columns: {list(df_calc.columns)}")
    
    # Calculate dT_HP (Heat pump temperature difference) - exactly like notebook
    if 'T_supply' in df_calc.columns and 'T_return_emu' in df_calc.columns:
        df_calc['dT_HP'] = df_calc['T_supply'] - df_calc['T_return_emu']
        print(f"Calculated dT_HP: {df_calc['dT_HP'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for dT_HP calculation")
    add_t_mean_log_column(df_calc)
    
    # Calculate dTreturn (Return temperature difference) - exactly like notebook
    if 'T_return_emu' in df_calc.columns and 'T_return_calc' in df_calc.columns:
        df_calc['dTreturn'] = df_calc['T_return_emu'] - df_calc['T_return_calc']
        print(f"Calculated dTreturn: {df_calc['dTreturn'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for dTreturn calculation")
    
    # Calculate abs_pressure_difference - exactly like notebook
    if 'Pressure difference' in df_calc.columns:
        # Check if there is any change of sign in the 'Pressure difference' column
        if np.any(np.diff(np.sign(df_calc['Pressure difference']))):
            # Apply the custom logic if there is a change of sign
            df_calc['abs_pressure_difference'] = df_calc['Pressure difference'].apply(lambda x: abs(x) if x < 0 else -x)
            print(f"Applied custom pressure difference logic")
        else:
            # Apply the absolute value function if there is no change of sign
            df_calc['abs_pressure_difference'] = df_calc['Pressure difference'].abs()
            print(f"Applied absolute pressure difference")
        
        # Convert from Pa to bar
        df_calc['abs_pressure_difference'] = df_calc['abs_pressure_difference'] / 100000
        print(f"Calculated abs_pressure_difference: {df_calc['abs_pressure_difference'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for abs_pressure_difference calculation")
    
    # Calculate normalised_Electric Power Input (corrected) - exactly like notebook
    if 'Electric Power Input (corrected)' in df_calc.columns:
        # Use the mean as reference for normalization (like in notebook)
        power_mean = df_calc['Electric Power Input (corrected)'].mean()
        if power_mean != 0:
            df_calc['normalised_Electric Power Input (corrected)'] = df_calc['Electric Power Input (corrected)'] / power_mean
        else:
            df_calc['normalised_Electric Power Input (corrected)'] = 0
        print(f"Calculated normalised_Electric Power Input (corrected): {df_calc['normalised_Electric Power Input (corrected)'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for normalised_Electric Power Input (corrected) calculation")
    
    # Volume flow is already in m³/h, no conversion needed
    # Just copy it if it exists
    if 'volume flow' in df_calc.columns:
        df_calc['volume flow'] = df_calc['volume flow']  # Already in m³/h
        print(f"Volume flow available: {df_calc['volume flow'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    
    # Mass flow: measured wins, missing or empty is derived from volume flow × density
    df_calc, mass_flow_notices = derive_mass_flow_if_needed(df_calc)
    for note in mass_flow_notices:
        print(note)
    
    # Calculate T_return_calc, T_B_calc, T_H_calc, Q_HB, Q_BA (from your existing calculations)
    if 'T_supply' in df_calc.columns and 'Heating Capacity (corrected)' in df_calc.columns and 'mass flow' in df_calc.columns:
        df_calc['T_return_calc'] = df_calc['T_supply'] - (df_calc['Heating Capacity (corrected)'] / (df_calc['mass flow'] * 4.183))
    
    if 'T_return_emu' in df_calc.columns and 'Q_BA' in df_calc.columns and 'mass flow' in df_calc.columns:
        df_calc['T_B_calc'] = df_calc['T_return_emu'] - (df_calc['Q_BA'] / (df_calc['mass flow'] * 4.183))
    
    if 'T_supply' in df_calc.columns and 'Q_HB' in df_calc.columns and 'mass flow' in df_calc.columns:
        df_calc['T_H_calc'] = df_calc['T_supply'] + (df_calc['Q_HB'] / (df_calc['mass flow'] * 4.183))
    
    print(f"Final columns after calculation: {list(df_calc.columns)}")
    return df_calc

def create_notebook_style_plot(selected_records):
    """Create the exact same plot as overlayplots-combine-axlabel-norm2.ipynb"""
    try:
        # Set up matplotlib exactly like the notebook
        plt.rcParams['text.usetex'] = False
        plt.style.use('default')
        
        # Define the exact columns from the notebook
        columns_to_plot = [
            'T_outdoor (DB)',
            'T_outdoor (WB)', 
            'T_supply',
            'T_return_emu',
            'Heating Capacity (corrected)',
            'Heating Capacity (without corr)',
            'Electric Power Input (corrected)',
            'Electric Power Input (without correction)',
            'Pressure difference',
            'volume flow'
        ]
        
        # Custom y labels exactly like the notebook
        custom_y_labels = {
            'T_outdoor (DB)': 'T_outdoor (DB) [°C]',
            'T_outdoor (WB)': 'T_outdoor (WB) [°C]',
            'T_supply': 'T_supply [°C]',
            'T_return_emu': 'T_return_emu [°C]',
            'Heating Capacity (corrected)': 'Heating Capacity (corrected) [kW]',
            'Heating Capacity (without corr)': 'Heating Capacity (without corr) [kW]',
            'Electric Power Input (corrected)': 'Electric Power Input (corrected) [kW]',
            'Electric Power Input (without correction)': 'Electric Power Input (without correction) [kW]',
            'Pressure difference': 'Pressure difference [Pa]',
            'volume flow': 'Volume flow [L/h]'
        }
        
        # Load data for each selected record
        data_dict = {}
        for record in selected_records:
            file_name = record['file_name']
            start_time = record['start_time']
            end_time = record['end_time']
            data_set = record['data_set']
            
            df = load_time_series_data(file_name, start_time, end_time)
            if df is not None:
                print(f"Loaded data for {file_name}, shape: {df.shape}")
                print(f"Available columns: {list(df.columns)}")
                
                # Calculate derived quantities (like in the notebook)
                df_with_derived = calculate_derived_quantities(df)
                
                # Normalize time to start from 0 (like in the notebook)
                df_normalized = df_with_derived.copy()
                df_normalized['time_elapsed'] = df_normalized['time_elapsed'] - df_normalized['time_elapsed'].iloc[0]
                
                key = f"{file_name}_dataset_{data_set}"
                data_dict[key] = df_normalized
        
        if not data_dict:
            return None
        
        # Create figure with subplots exactly like the notebook
        n_plots = len(columns_to_plot)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Font settings like the notebook
        axis_font = {'fontname': 'Arial', 'size': 12}
        legend_font = {'size': 10}
        tick_font = {'size': 10}
        
        # Plot each column exactly like the notebook
        for i, column in enumerate(columns_to_plot):
            ax = axes[i]
            
            for key, df in data_dict.items():
                if column in df.columns:
                    # Create legend label exactly like the notebook
                    file_name = key.split('_dataset_')[0]
                    data_set = key.split('_dataset_')[1]
                    legend_label = f"{file_name} (set {data_set})"
                    
                    # Plot with the same styling
                    ax.plot(df['time_elapsed'], df[column], label=legend_label, linewidth=1.5)
            
            # Set labels and formatting exactly like the notebook
            ax.set_xlabel('Time [s]', **axis_font)
            ax.set_ylabel(custom_y_labels.get(column, column), **axis_font)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=legend_font)
            ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()
        
    except Exception as e:
        print(f"Error creating notebook style plot: {e}")
        return None

def create_notebook_style_plot_with_columns(selected_records, plot_columns):
    """Create notebook-style plot with selected columns and normalized time"""
    try:
        print(f"=== NOTEBOOK STYLE PLOT WITH COLUMNS CALLED ===")
        print(f"Selected records: {len(selected_records)}")
        print(f"Plot columns: {plot_columns}")
        
        # Set up matplotlib exactly like the notebook
        plt.rcParams['text.usetex'] = False
        plt.style.use('default')
        
        # If no columns specified, use all columns from the notebook (including derived quantities)
        if not plot_columns:
            columns_to_plot = [
                'T_outdoor (DB)',
                'T_outdoor (WB)', 
                'T_supply',
                'T_return_emu',
                'T_mean_log',
                'dT_HP',
                'dTreturn',
                'T_return_calc',
                'T_B_calc',
                'T_H_calc',
                'Heating Capacity (corrected)',
                'Heating Capacity (without corr)',
                'Electric Power Input (corrected)',
                'Electric Power Input (without correction)',
                'normalised_Electric Power Input (corrected)',
                'abs_pressure_difference',
                'Q_HB',
                'Q_BA',
                'mass flow',
                'volume flow'
            ]
        else:
            columns_to_plot = plot_columns
        
        # Custom y labels exactly like the notebook
        custom_y_labels = {
            'T_outdoor (DB)': r'$T_{\mathrm{db}}$ (°C)',
            'T_outdoor (WB)': r'$T_{\mathrm{wb}}$ (°C)',
            'T_supply': r'$T_{\mathrm{sup}}$ (°C)',
            'T_return_emu': r'$T_{\mathrm{ret}}$ (°C)',
            'T_mean_log': r'$T_{\mathrm{mean,log}}$ (°C)',
            'dT_ln': r'$\Delta T_{\ln}$ (K)',
            'dT_HP': r'$\Delta T_{\mathrm{HP}}$ ($^\circ$C)',
            'dTreturn': r'$\Delta T_{\mathrm{ret}}$ ($^\circ$C)',
            'T_return_calc': 'T_return_calc [°C]',
            'T_B_calc': 'T_B_calc [°C]',
            'T_H_calc': 'T_H_calc [°C]',
            'Heating Capacity (corrected)': 'Heating Capacity (w/ corr) (kW)',
            'Heating Capacity (without corr)': 'Heating Capacity (w/o corr) (kW)',
            'Electric Power Input (corrected)': 'Power Input (w/ corr) (kW)',
            'Electric Power Input (without correction)': 'Power Input (w/o corr) (kW)',
            'normalised_Electric Power Input (corrected)': 'Normalised Power (-)',
            'abs_pressure_difference': r'$\Delta P_{\mathrm{HP}}$ (bar)',
            'Q_HB': 'Q_HB [kW]',
            'Q_BA': 'Q_BA [kW]',
            'mass flow': 'Mass flow [kg/s]',
            'volume flow': r'Volume Flow (m³/h)'
        }
        
        # Load data for each selected record
        data_dict = {}
        for record in selected_records:
            file_name = record['file_name']
            start_time = record['start_time']
            end_time = record['end_time']
            data_set = record['data_set']
            
            df = load_time_series_data(file_name, start_time, end_time)
            if df is not None:
                print(f"Loaded data for {file_name}, shape: {df.shape}")
                print(f"Available columns: {list(df.columns)}")
                
                # Calculate derived quantities (like in the notebook)
                df_with_derived = calculate_derived_quantities(df)
                
                # Normalize time to start from 0 (like in the notebook)
                df_normalized = df_with_derived.copy()
                df_normalized['time_elapsed'] = df_normalized['time_elapsed'] - df_normalized['time_elapsed'].iloc[0]
                
                key = f"{file_name}_dataset_{data_set}"
                data_dict[key] = df_normalized
        
        if not data_dict:
            return None
        
        # Create figure with subplots exactly like the notebook
        n_plots = len(columns_to_plot)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Font settings like the notebook
        axis_font = {'fontname': 'Arial', 'size': 12}
        legend_font = {'size': 10}
        tick_font = {'size': 10}
        
        # Plot each column exactly like the notebook
        for i, column in enumerate(columns_to_plot):
            ax = axes[i]
            
            for key, df in data_dict.items():
                if column in df.columns:
                    # Create legend label exactly like the notebook
                    file_name = key.split('_dataset_')[0]
                    data_set = key.split('_dataset_')[1]
                    legend_label = f"{file_name} (set {data_set})"
                    
                    # Plot with the same styling
                    ax.plot(df['time_elapsed'], df[column], label=legend_label, linewidth=1.5)
            
            # Set labels and formatting exactly like the notebook
            ax.set_xlabel('Time [s]', **axis_font)
            ax.set_ylabel(custom_y_labels.get(column, column), **axis_font)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=legend_font)
            ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()
        
    except Exception as e:
        print(f"Error creating notebook style plot with columns: {e}")
        return None

def create_plot_with_original_time(selected_records, plot_columns, layout=None, legend_style='auto', custom_legend_labels=''):
    """Create plots with original timestamps (not normalized)"""
    try:
        # Set up matplotlib
        plt.rcParams['text.usetex'] = False
        plt.style.use('default')
        
        # Define available columns for plotting
        available_columns = {
            'T_outdoor (DB)': 'T_outdoor (DB)',
            'T_outdoor (WB)': 'T_outdoor (WB)', 
            'T_supply': 'T_supply',
            'T_return_emu': 'T_return_emu',
            'Heating Capacity (corrected)': 'Heating Capacity (corrected)',
            'Heating Capacity (without corr)': 'Heating Capacity (without corr)',
            'Electric Power Input (corrected)': 'Electric Power Input (corrected)',
            'Pressure difference': 'Pressure difference',
            'volume flow': 'volume flow',
            'mass flow': 'mass flow'
        }
        
        # Filter to only available columns
        plot_columns = [col for col in plot_columns if col in available_columns]
        
        if not plot_columns:
            return None
            
        # Create figure
        n_plots = len(plot_columns)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Load data for each selected record (with original timestamps)
        data_dict = {}
        for record in selected_records:
            file_name = record['file_name']
            start_time = record['start_time']
            end_time = record['end_time']
            data_set = record['data_set']
            
            df = load_time_series_data(file_name, start_time, end_time)
            if df is not None:
                key = f"{file_name}_dataset_{data_set}"
                data_dict[key] = df
        
        if not data_dict:
            return None
        
        # Generate legend labels once
        legend_labels = generate_legend_labels(selected_records, legend_style, custom_legend_labels)
        print(f"Generated legend labels: {legend_labels}")
        
        # Plot each column
        for i, column in enumerate(plot_columns):
            ax = axes[i]
            
            for j, (key, df) in enumerate(data_dict.items()):
                if column in df.columns:
                    # Use the generated legend label if available, otherwise use key
                    if j < len(legend_labels) and legend_labels[j] is not None:
                        label = legend_labels[j]
                    else:
                        label = key
                    ax.plot(df['time_elapsed'], df[column], label=label, linewidth=1.5)
            
            ax.set_xlabel('Time [s]', fontsize=12)
            ax.set_ylabel(column, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Only show legend if legend_style is not 'none'
            if legend_style != 'none':
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()
        
    except Exception as e:
        print(f"Error creating plot with original time: {e}")
        return None

def create_plot(selected_records, plot_columns, plot_type='overlay'):
    """Create time series plots for selected records"""
    try:
        # Set up matplotlib
        plt.rcParams['text.usetex'] = False
        plt.style.use('default')
        
        # Define available columns for plotting
        available_columns = {
            'T_outdoor (DB)': 'T_outdoor (DB)',
            'T_outdoor (WB)': 'T_outdoor (WB)', 
            'T_supply': 'T_supply',
            'T_return_emu': 'T_return_emu',
            'Heating Capacity (corrected)': 'Heating Capacity (corrected)',
            'Heating Capacity (without corr)': 'Heating Capacity (without corr)',
            'Electric Power Input (corrected)': 'Electric Power Input (corrected)',
            'Electric Power Input (without correction)': 'Electric Power Input (without correction)',
            'Pressure difference': 'Pressure difference',
            'volume flow': 'volume flow',
            'mass flow': 'mass flow'
        }
        
        # Filter to only available columns
        plot_columns = [col for col in plot_columns if col in available_columns]
        
        if not plot_columns:
            return None
            
        # Create figure
        n_plots = len(plot_columns)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Load data for each selected record
        data_dict = {}
        for record in selected_records:
            file_name = record['file_name']
            start_time = record['start_time']
            end_time = record['end_time']
            data_set = record['data_set']
            
            df = load_time_series_data(file_name, start_time, end_time)
            if df is not None:
                key = f"{file_name}_dataset_{data_set}"
                data_dict[key] = df
        
        if not data_dict:
            return None
        
        # Plot each column
        for i, column in enumerate(plot_columns):
            ax = axes[i]
            
            for key, df in data_dict.items():
                if column in df.columns:
                    # Create legend label
                    file_name = key.split('_dataset_')[0]
                    data_set = key.split('_dataset_')[1]
                    legend_label = f"{file_name} (set {data_set})"
                    
                    ax.plot(df['time_elapsed'], df[column], label=legend_label, linewidth=1.5)
            
            ax.set_xlabel('Time [s]', fontsize=12)
            ax.set_ylabel(column, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

def process_file(file_name, data_set, start_time, end_time):
    """Process a single Excel file and return filtered DataFrame"""
    file_path = os.path.join(config['data_dir'], file_name)
    print(f"Looking for file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None
    
    try:
        # Coerce to float so pandas comparison works (e.g. when called from AJAX with string numbers)
        start_time = float(start_time)
        end_time = float(end_time)
    except (TypeError, ValueError):
        print(f"ERROR: Invalid start_time or end_time: {start_time!r}, {end_time!r}")
        return None
    
    try:
        print(f"Reading Excel file: {file_name}")
        df = pd.read_excel(file_path)
        print(f"File loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Handle missing columns (HPT Plotdaten may omit mass flow / lab-corrected Q/P)
        missing_columns = [col for col in config['required_columns'] if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in {file_name}: {missing_columns}")

        notices = []
        lab_corr_missing = (
            'Heating Capacity (corrected)' not in df.columns
            or 'Electric Power Input (corrected)' not in df.columns
        )

        df, mass_flow_notices = derive_mass_flow_if_needed(df)
        mass_flow_derived = bool(df.attrs.get('mass_flow_derived'))
        for note in mass_flow_notices:
            print(note)

        # Do NOT invent lab-corrected Q/P here. If missing, calculate_and_insert
        # will fill the DB "corr" fields from BAM-corrected values and notify the user.
        if lab_corr_missing:
            if 'Heating Capacity (without corr)' not in df.columns:
                print(f"ERROR: Missing heating capacity (without corr) in {file_name}")
                return None
            if 'Electric Power Input (without correction)' not in df.columns:
                print(f"ERROR: Missing electric power (without correction) in {file_name}")
                return None
            msg = (
                "Lab-corrected heating capacity / electric power are missing — "
                "database 'corrected' fields will use BAM pump-corrected values instead."
            )
            notices.append(msg)
            print(msg)

        if 'Electrical power input BUH' not in df.columns:
            df['Electrical power input BUH'] = 0
            df['has_buh'] = False  # Add a flag
            print("Column 'Electrical power input BUH' absent — has_buh=False (no power_BUH_from_Ts)")
        else:
            buh = pd.to_numeric(df['Electrical power input BUH'], errors='coerce')
            # HPT may ship an all-zero BUH placeholder; BAM uses absent column OR NaNs to fill.
            # All explicit zeros (no NaNs) → treat like column absent (skip power_BUH_from_Ts).
            # Any NaN → keep BAM behaviour (from_Ts can fill gaps).
            # Any non-zero → real/virtual BUH present.
            if buh.notna().all() and len(buh) > 0 and (buh == 0).all():
                df['has_buh'] = False
                print(
                    "Column 'Electrical power input BUH' is all zeros — treating as no BUH "
                    "(skip power_BUH_from_Ts); same as missing column in BAM RRT"
                )
            else:
                df['has_buh'] = True

        print(f"Filtering data for time range: {start_time} to {end_time}")
        df_filtered = df[(df['time_elapsed'] >= start_time) & (df['time_elapsed'] <= end_time)].copy()
        print(f"Filtered data shape: {df_filtered.shape}")

        # If first sheet has no data in range, try other sheets (multi-sheet workbooks)
        if df_filtered.empty and 'time_elapsed' in df.columns:
            print(f"No data in first sheet for time range {start_time} to {end_time}; available: {df['time_elapsed'].min()} to {df['time_elapsed'].max()}")
        if df_filtered.empty:
            try:
                all_sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, sheet_df in all_sheets.items():
                    if 'time_elapsed' not in sheet_df.columns:
                        continue
                    sheet_filtered = sheet_df[(sheet_df['time_elapsed'] >= start_time) & (sheet_df['time_elapsed'] <= end_time)].copy()
                    if not sheet_filtered.empty:
                        df_filtered = sheet_filtered
                        # Re-run on this sheet: the first sheet's mass-flow verdict does not apply
                        df_filtered, mass_flow_notices = derive_mass_flow_if_needed(df_filtered)
                        mass_flow_derived = bool(df_filtered.attrs.get('mass_flow_derived'))
                        for note in mass_flow_notices:
                            print(note)
                        lab_corr_missing = (
                            'Heating Capacity (corrected)' not in df_filtered.columns
                            or 'Electric Power Input (corrected)' not in df_filtered.columns
                        )
                        if 'Electrical power input BUH' not in df_filtered.columns:
                            df_filtered['Electrical power input BUH'] = 0
                            df_filtered['has_buh'] = False
                        else:
                            buh = pd.to_numeric(df_filtered['Electrical power input BUH'], errors='coerce')
                            if buh.notna().all() and len(buh) > 0 and (buh == 0).all():
                                df_filtered['has_buh'] = False
                            else:
                                df_filtered['has_buh'] = True
                        print(f"Found data in sheet: {sheet_name}")
                        break
            except Exception as ex:
                print(f"Could not try other sheets: {ex}")

        if df_filtered.empty:
            print(f"No data found for {file_name} in time range {start_time} to {end_time}")
            if 'time_elapsed' in df.columns:
                print(f"Available time range in first sheet: {df['time_elapsed'].min()} to {df['time_elapsed'].max()}")
            return None
            
        # Add T_supply_set based on file name
        df_filtered['T_supply_set'] = config['t_supply_set_mapping'].get(extract_letter(file_name), None)

        # Re-evaluate has_buh on the filtered window (cycle may be all-zero even if file is not)
        if 'Electrical power input BUH' in df_filtered.columns:
            buh_win = pd.to_numeric(df_filtered['Electrical power input BUH'], errors='coerce')
            if buh_win.notna().all() and len(buh_win) > 0 and (buh_win == 0).all():
                df_filtered['has_buh'] = False
            elif 'has_buh' not in df_filtered.columns:
                df_filtered['has_buh'] = True

        # Carry processing notes into calculate_and_insert / UI. The mass-flow note is
        # added last so it reflects the sheet the window actually came from.
        for note in mass_flow_notices:
            if note not in notices:
                notices.append(note)
        df_filtered.attrs['import_notices'] = notices
        df_filtered.attrs['mass_flow_notices'] = list(mass_flow_notices)
        df_filtered.attrs['mass_flow_derived'] = mass_flow_derived
        df_filtered.attrs['lab_corr_missing'] = lab_corr_missing
      
        return df_filtered
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def compute_optional_means(df_window):
    """Arithmetic mean over the cycle window for each optional monitoring column.

    None (→ SQL NULL, shown as n/a) when the column is absent or has no finite value in
    the window. A missing series must never be stored as 0.0.
    """
    means = {}
    for plot_col in optional_average_columns():
        db_col = OPTIONAL_MEAN_DB_COLUMNS[plot_col]
        if plot_col not in df_window.columns:
            means[db_col] = None
            continue
        series = pd.to_numeric(df_window[plot_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        means[db_col] = float(series.mean()) if series.notna().any() else None
    return means


def write_optional_means(cursor, df_window, rowid):
    """Write the optional means by rowid, after the core insert/update.

    Deliberately separate from config['insert_query'] and its UPDATE list: those stay the
    BAM contract. Recalculation refreshes these values because it runs the same path.
    """
    means = compute_optional_means(df_window)
    if not means:
        return
    cols = list(means)
    sql = 'UPDATE results SET ' + ', '.join(f'"{c}" = ?' for c in cols) + ' WHERE rowid = ?'
    values = [means[c] for c in cols] + [rowid]
    try:
        cursor.execute(sql, values)
    except sqlite3.OperationalError:
        ensure_results_columns({c: 'REAL' for c in cols})
        cursor.execute(sql, values)


ANALYSIS_LAB_CORR_COLUMNS = ('Heating Capacity (corrected)', 'Electric Power Input (corrected)')

# Sheet columns the analysis-time block can do without: the lab-corrected pair is
# filled from the BAM pump correction, and a file with no BUH gets a zero column.
ANALYSIS_SUPPLIED_COLUMNS = ANALYSIS_LAB_CORR_COLUMNS + ('Electrical power input BUH',)

# Plain period averages: sheet column → results column.
ANALYSIS_AVERAGE_DB_COLUMNS = {
    'T_outdoor (DB)': 'avg_t_db',
    'T_outdoor (WB)': 'avg_t_wb',
    'T_supply': 'avg_t_supply',
    'T_return_emu': 'avg_t_return_emu',
    'volume flow': 'avg_volume_flow',
    'mass flow': 'avg_mass_flow',
    'Heating Capacity (without corr)': 'avg_heating_capacity_uncorr',
    'Electric Power Input (without correction)': 'avg_power_input_uncorr',
    'Heating Capacity (corrected)': 'avg_heating_capacity_corr',
    'Electric Power Input (corrected)': 'avg_power_input_corr',
    'Pressure difference': 'avg_dp',
    'T_return_calc': 'avg_t_return_calc',
    'T_B_calc': 'avg_t_b_calc',
    'T_H_calc': 'avg_t_h_calc',
    'Q_HB': 'avg_q_hb',
    'Q_BA': 'avg_q_ba',
}

# The values config['insert_query'] expects between (file_name, data_set) and
# (start_time, end_time).
ANALYSIS_MEAN_INSERT_KEYS = (
    'avg_t_db', 'avg_t_wb', 'avg_t_supply', 'avg_t_return_emu',
    'avg_volume_flow', 'avg_mass_flow',
    'avg_heating_capacity_uncorr', 'avg_power_input_uncorr', 'cop_uncorr',
    'avg_heating_capacity_corr', 'avg_power_input_corr', 'cop_corr',
    'avg_dp', 'avg_adj_dp', 'avg_P_hyd',
    'avg_t_return_calc', 'avg_t_b_calc', 'avg_t_h_calc', 'avg_q_hb', 'avg_q_ba',
    'avg_eff_pump', 'avg_qcorr_bam', 'avg_pcorr_bam',
    'avg_power_buh', 'avg_power_buh_from_ts', 'avg_t_sup_buh',
    'avg_heating_capacity_bam_corr', 'avg_power_input_bam_corr', 'cop_bam_corr',
)


def analysis_window_means(df_window, file_name, lab_corr_missing=None, notices=None) -> dict:
    """Analysis-time means of one time window, keyed by results column name.

    This is the block Apply runs on a cycle before it stores it: the period
    averages, the BAM pump correction, BUH power and supply temperature after
    BUH, and the "corrected" averages — the lab-corrected series when the sheet
    carries them, the BAM-corrected ones when it does not.

    Nothing is written here. `calculate_and_insert` stores the result for a
    parent cycle; Guideline Windows asks for the same quantities on an
    evaluation window, so both read one formula. `df_window` gains the
    intermediate series (P_hyd, power_BUH, ...) as it always has inside Apply.
    A required column that the sheet does not carry raises KeyError.
    """
    if lab_corr_missing is None:
        lab_corr_missing = any(c not in df_window.columns for c in ANALYSIS_LAB_CORR_COLUMNS)

    # Averages for columns that exist; lab-corrected may be filled from BAM below
    avg_values = {}
    for col in config['average_columns']:
        if col in df_window.columns:
            avg_values[col] = df_window[col].mean()
        elif col in ANALYSIS_LAB_CORR_COLUMNS:
            avg_values[col] = None  # placeholder until BAM fallback
        else:
            raise KeyError(col)

    # Uncorrected COP (always from without-corr)
    cop_uncorr = (
        avg_values['Heating Capacity (without corr)']
        / avg_values['Electric Power Input (without correction)']
    )

    # Calculate pressure difference
    ignore_percentage = 0.01
    ignore_rows = int(len(df_window) * ignore_percentage)
    check_data = df_window.iloc[ignore_rows:-ignore_rows]['Pressure difference'].dropna()
    if len(check_data) > 0 and np.any(np.diff(np.sign(check_data))):
        df_window['adjusted_pressure_difference'] = df_window['Pressure difference'].apply(lambda x: abs(x) if x < 0 else -x)
        print(f"The value of pressure difference for the integrated circulator is assumed to be negative by default.")
    else:
        df_window['adjusted_pressure_difference'] = df_window['Pressure difference'].abs()
    avg_adj_dp = df_window['adjusted_pressure_difference'].mean()

    # Calculate hydraulic power
    df_window['P_hyd'] = ((df_window['volume flow'] / 3600) * df_window['adjusted_pressure_difference'] * 100000).abs()
    avg_P_hyd = df_window['P_hyd'].mean()

    # Handle BUH power calculations
    if 'has_buh' in df_window.columns and not df_window['has_buh'].iloc[0]:
        # If the column does not exist at all, handle as before
        df_window['power_BUH'] = 0
        df_window['power_BUH_from_Ts'] = 0
        df_window['T_supply_afterBUH'] = df_window['T_supply']
        print("Column 'Electrical power input BUH' does not exist in the DataFrame. Assuming value as 0.")
    else:
        # Calculate power_BUH_from_Ts as before
        df_window['power_BUH_from_Ts_uncapped'] = (df_window['mass flow']) * 4.183 * (df_window['T_supply_set']-df_window['T_supply'])
        cap_value = determine_cap(file_name)
        df_window['power_BUH_from_Ts'] = df_window['power_BUH_from_Ts_uncapped'].apply(lambda x: max(min(x, cap_value), 0))

        # Fill missing values in 'Electrical power input BUH' with calculated values
        num_missing = df_window['Electrical power input BUH'].isna().sum()
        df_window['power_BUH'] = df_window['Electrical power input BUH'].copy()
        df_window.loc[df_window['power_BUH'].isna(), 'power_BUH'] = df_window['power_BUH_from_Ts']
        if num_missing > 0:
            print(f"Filled {num_missing} missing values in 'Electrical power input BUH' with calculated 'power_BUH_from_Ts'.")

        # Set T_supply_afterBUH accordingly
        df_window['T_supply_afterBUH'] = df_window['T_supply'] + df_window['power_BUH']/(df_window['mass flow'] * 4.183)

    # Calculate pump efficiency and BAM corrections
    df_window['efficiency'] = ((df_window['P_hyd']*0.35844) / ((1.7*df_window['P_hyd']) + (17*(1 - np.exp(-0.3*df_window['P_hyd']))))) * (0.49 / 0.23)
    df_window['powercorrection_BAM'] = (df_window['P_hyd']/df_window['efficiency'])/1000
    df_window['heatingcorrection_BAM'] = (df_window['P_hyd']*(1-df_window['efficiency'])/df_window['efficiency'])/1000

    # Calculate BAM corrected values
    df_window['Heating Capacity (BAM corrected)'] = np.where(df_window['adjusted_pressure_difference'] > 0,
        df_window['Heating Capacity (without corr)'] - df_window['heatingcorrection_BAM'],
        df_window['Heating Capacity (without corr)'] + df_window['heatingcorrection_BAM'])
    df_window['Electric Power Input (BAM corrected)'] = np.where(df_window['adjusted_pressure_difference'] > 0,
        df_window['Electric Power Input (without correction)'] - df_window['powercorrection_BAM'],
        df_window['Electric Power Input (without correction)'] + df_window['powercorrection_BAM'])

    # Add the new calculated values to averages
    avg_eff_pump = df_window['efficiency'].mean()
    avg_qcorr_bam = df_window['heatingcorrection_BAM'].mean()
    avg_pcorr_bam = df_window['powercorrection_BAM'].mean()
    avg_power_buh = df_window['power_BUH'].mean()
    avg_power_buh_from_ts = df_window['power_BUH_from_Ts'].mean()
    avg_t_sup_buh = df_window['T_supply_afterBUH'].mean()
    avg_t_mean_log, t_mean_from_avgs, avg_dt_ln = compute_t_mean_period_stats(
        df_window, avg_t_sup_buh=avg_t_sup_buh, avg_values=avg_values
    )
    avg_heating_capacity_bam_corr = df_window['Heating Capacity (BAM corrected)'].mean()
    avg_power_input_bam_corr = df_window['Electric Power Input (BAM corrected)'].mean()

    # Calculate BAM corrected COP
    cop_bam_corr = avg_heating_capacity_bam_corr / avg_power_input_bam_corr

    # If lab-corrected Q/P missing: use BAM-corrected for the DB "corr" fields
    if lab_corr_missing:
        avg_values['Heating Capacity (corrected)'] = avg_heating_capacity_bam_corr
        avg_values['Electric Power Input (corrected)'] = avg_power_input_bam_corr
        cop_corr = cop_bam_corr
        note = (
            f"{file_name}: lab-corrected Q/P missing — stored 'corrected' averages "
            f"from BAM pump correction (Q={avg_heating_capacity_bam_corr:.4g} kW, "
            f"P={avg_power_input_bam_corr:.4g} kW)."
        )
        if notices is not None:
            notices.append(note)
        print(note)
    else:
        cop_corr = (
            avg_values['Heating Capacity (corrected)']
            / avg_values['Electric Power Input (corrected)']
        )

    means = {db_col: avg_values[sheet_col]
             for sheet_col, db_col in ANALYSIS_AVERAGE_DB_COLUMNS.items()}
    means.update({
        'cop_uncorr': cop_uncorr,
        'cop_corr': cop_corr,
        'avg_adj_dp': avg_adj_dp,
        'avg_P_hyd': avg_P_hyd,
        'avg_eff_pump': avg_eff_pump,
        'avg_qcorr_bam': avg_qcorr_bam,
        'avg_pcorr_bam': avg_pcorr_bam,
        'avg_power_buh': avg_power_buh,
        'avg_power_buh_from_ts': avg_power_buh_from_ts,
        'avg_t_sup_buh': avg_t_sup_buh,
        'avg_heating_capacity_bam_corr': avg_heating_capacity_bam_corr,
        'avg_power_input_bam_corr': avg_power_input_bam_corr,
        'cop_bam_corr': cop_bam_corr,
        'avg_t_mean_log': avg_t_mean_log,
        't_mean_from_avgs': t_mean_from_avgs,
        'avg_dt_ln': avg_dt_ln,
    })
    return means


def analysis_wbuh_fields(means: dict) -> dict:
    """The BUH-corrected derived fields of column_metadata.json, from a means dict.

    `calculate_derived_columns_for_entry` evaluates these formulas against a
    stored results row; a window that is never stored needs the same numbers
    without a row, and the deviation getters prefer exactly these keys.
    COP is the ratio of the two means, never the mean of a ratio.
    """
    def num(key):
        try:
            val = float(means.get(key))
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) else None

    out = {k: None for k in ('powerbuh', 'QCorrwBUH', 'PCorrwBUH', 'COPCorrwBUH', 'Ts_buh')}
    buh, buh_from_ts = num('avg_power_buh'), num('avg_power_buh_from_ts')
    powerbuh = buh if (buh is not None and buh > 0) else buh_from_ts
    out['powerbuh'] = powerbuh
    q_corr, p_corr = num('avg_heating_capacity_corr'), num('avg_power_input_corr')
    if powerbuh is not None:
        if q_corr is not None:
            out['QCorrwBUH'] = powerbuh + q_corr
        if p_corr is not None:
            out['PCorrwBUH'] = powerbuh + p_corr
    if out['QCorrwBUH'] is not None and out['PCorrwBUH']:
        out['COPCorrwBUH'] = out['QCorrwBUH'] / out['PCorrwBUH']
    t_sup_buh, t_supply = num('avg_t_sup_buh'), num('avg_t_supply')
    candidates = [t for t in (t_sup_buh, t_supply) if t is not None]
    if candidates:
        out['Ts_buh'] = max(candidates)
    return out


def calculate_and_insert(cursor, df_filtered, file_name, data_set, start_time, end_time, update_existing=False, old_start_time=None, old_end_time=None, notices_out=None):

    try:
        rowid = None
        notices = list(df_filtered.attrs.get('import_notices') or [])
        lab_corr_missing = bool(df_filtered.attrs.get('lab_corr_missing')) or (
            'Heating Capacity (corrected)' not in df_filtered.columns
            or 'Electric Power Input (corrected)' not in df_filtered.columns
        )
        # Frames that did not come through process_file (or lost their attrs) get the
        # same fallback here, so the DB never sees a silently missing mass flow.
        mass_flow_notices = list(df_filtered.attrs.get('mass_flow_notices') or [])
        if not mass_flow_notices:
            df_filtered, mass_flow_notices = derive_mass_flow_if_needed(df_filtered)
        for note in mass_flow_notices:
            if note not in notices:
                notices.append(note)

        means = analysis_window_means(
            df_filtered, file_name, lab_corr_missing=lab_corr_missing, notices=notices
        )
        avg_t_mean_log = means['avg_t_mean_log']
        t_mean_from_avgs = means['t_mean_from_avgs']
        avg_dt_ln = means['avg_dt_ln']

        # Create insert values in the correct order matching the table schema
        insert_values = (
            [file_name, data_set]
            + [means[key] for key in ANALYSIS_MEAN_INSERT_KEYS]
            + [start_time, end_time]
        )

        if update_existing:
            # Use UPDATE instead of INSERT to avoid duplicates
            # If old_start_time and old_end_time are provided, use them in WHERE clause (for editing)
            # Otherwise, use current start_time and end_time (for normal recalculation)
            where_start_time = old_start_time if old_start_time is not None else start_time
            where_end_time = old_end_time if old_end_time is not None else end_time
            
            update_query = """
                UPDATE results SET
                    avg_t_db = ?, avg_t_wb = ?, avg_t_supply = ?, avg_t_return_emu = ?,
                    avg_volume_flow = ?, avg_mass_flow = ?,
                    avg_heating_capacity_uncorr = ?, avg_power_input_uncorr = ?, cop_uncorr = ?,
                    avg_heating_capacity_corr = ?, avg_power_input_corr = ?, cop_corr = ?,
                    avg_dp = ?, avg_adj_dp = ?, avg_P_hyd = ?,
                    avg_t_return_calc = ?, avg_t_b_calc = ?, avg_t_h_calc = ?, avg_q_hb = ?, avg_q_ba = ?,
                    avg_eff_pump = ?, avg_qcorr_bam = ?, avg_pcorr_bam = ?,
                    avg_power_buh = ?, avg_power_buh_from_ts = ?, avg_t_sup_buh = ?,
                    avg_heating_capacity_bam_corr = ?, avg_power_input_bam_corr = ?, cop_bam_corr = ?,
                    start_time = ?, end_time = ?
                WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?
            """
            # Reorder values: all calculated values first, then WHERE clause values
            update_values = insert_values[2:] + [file_name, data_set, where_start_time, where_end_time]
            print(f"Updating values for {file_name}, dataset {data_set}: {update_values[:5]}...")
            cursor.execute(update_query, update_values)
            
            # Get rowid for the updated entry (use new times after update)
            cursor.execute("""
                SELECT rowid FROM results 
                WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?
            """, (file_name, data_set, start_time, end_time))
            result = cursor.fetchone()
            if result:
                rowid = result['rowid']
                # Automatically calculate derived columns
                calculate_derived_columns_for_entry(cursor, rowid)
        else:
            # Insert the entry first
            print(f"Inserting values for {file_name}: {insert_values[:5]}...")
            cursor.execute(config['insert_query'], insert_values)
            
            # Get the rowid of the newly inserted entry
            cursor.execute("""
                SELECT rowid FROM results 
                WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?
            """, (file_name, data_set, start_time, end_time))
            result = cursor.fetchone()
            rowid = result['rowid'] if result else None
            
            # For new entries, set display_order to the maximum + 1 (append to end)
            # User can reorder manually later using the move buttons
            cursor.execute("SELECT COALESCE(MAX(display_order), 0) + 1 as next_order FROM results")
            next_order = cursor.fetchone()['next_order']
            
            # Update display_order for the newly inserted entry
            cursor.execute("""
                UPDATE results 
                SET display_order = ? 
                WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?
            """, (next_order, file_name, data_set, start_time, end_time))
            
            # Automatically calculate derived columns for the new entry
            if rowid:
                calculate_derived_columns_for_entry(cursor, rowid)

        if rowid:
            try:
                cursor.execute(
                    """UPDATE results SET avg_t_mean_log = ?, t_mean_from_avgs = ?, avg_dt_ln = ?
                       WHERE rowid = ?""",
                    (avg_t_mean_log, t_mean_from_avgs, avg_dt_ln, rowid)
                )
            except sqlite3.OperationalError:
                ensure_results_columns({
                    'avg_t_mean_log': 'REAL',
                    't_mean_from_avgs': 'REAL',
                    'avg_dt_ln': 'REAL',
                })
                cursor.execute(
                    """UPDATE results SET avg_t_mean_log = ?, t_mean_from_avgs = ?, avg_dt_ln = ?
                       WHERE rowid = ?""",
                    (avg_t_mean_log, t_mean_from_avgs, avg_dt_ln, rowid)
                )
            write_optional_means(cursor, df_filtered, rowid)

        # De-duplicate notices while preserving order
        seen = set()
        unique_notices = []
        for n in notices:
            if n not in seen:
                seen.add(n)
                unique_notices.append(n)
        if notices_out is not None:
            notices_out.extend(unique_notices)

        return True

    except Exception as e:
        print(f"Error calculating/inserting data for {file_name}: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/test')
def test():
    return "Flask app is running with updated code!"

@app.route('/test_recalculate_column', methods=['GET'])
def test_recalculate_column():
    """Test endpoint to verify the recalculate_column route is accessible"""
    return jsonify({
        'success': True,
        'message': 'recalculate_column route is accessible',
        'route': '/recalculate_column',
        'method': 'POST'
    })


@app.route('/')
def index():
    print("=== INDEX ROUTE CALLED ===")
    # Collapse per-cycle raw notices here (flash / modal time), not in calculate_and_insert.
    pending_notices = session.pop('data_import_notices', None) or []
    import_notices = collapse_import_notices(pending_notices)

    # Ensure display_order column exists
    init_display_order_column()
    # Empty campaign DBs created before ANALYSIS_MEAN_COLUMNS were in CREATE TABLE
    # otherwise 500 on this page (ordered_columns listed avg_t_b_calc etc. that PRAGMA lacked).
    try:
        ensure_results_columns(dataset_manager.ANALYSIS_MEAN_COLUMNS)
    except Exception as e:
        print(f"Analysis mean columns ensure skipped: {e}")
    
    conn = get_db_connection()
    try:
        # Order by display_order to maintain custom order
        # Explicitly include rowid in the SELECT to ensure it's available
        results = conn.execute('SELECT *, rowid FROM results ORDER BY display_order ASC, rowid ASC').fetchall()
        print(f"Retrieved {len(results)} rows from the database.")
        
        # Debug: Print raw results before conversion
        print("Raw results from database:")
        for i, row in enumerate(results):
            row_dict = dict(row)
            print(f"Row {i}: rowid={row_dict.get('rowid', 'NOT FOUND')}, keys={list(row_dict.keys())[:5]}...")
        
        # Convert sqlite3.Row objects to dictionaries
        results = [dict(row) for row in results]
        
        # Ensure rowid is available in each result
        for i, row in enumerate(results):
            if 'rowid' not in row:
                # Try to get it from the original query result
                # SQLite rowid should be available, but if not, we'll need to add it
                print(f"WARNING: rowid not found in row {i}")
        print(f"Converted results: {results}")
        
        # Debug: Print column names
        if results:
            print(f"Column names: {list(results[0].keys())}")
        else:
            print("No results found in database")
        
        # Check if we have last selected records to restore
        last_selected = session.get('last_selected_records', [])
        print(f"Last selected records: {len(last_selected)}")
        print(f"Last selected data: {last_selected}")
        
        # Get column information for dynamic display
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(results)")
        columns_info = cursor.fetchall()
        
        # Define standard columns that should always be shown in a specific order
        standard_columns = ['file_name', 'data_set', 'start_time', 'end_time']
        calculated_columns = [
            'avg_heating_capacity_uncorr', 'avg_power_input_uncorr', 'cop_uncorr',
            'avg_heating_capacity_corr', 'avg_power_input_corr', 'cop_corr',
            'avg_heating_capacity_bam_corr', 'avg_power_input_bam_corr', 'cop_bam_corr',
            'avg_t_db', 'avg_t_wb', 'avg_t_supply', 'avg_t_return_emu', 'avg_t_return_calc',
            'avg_t_mean_log', 't_mean_from_avgs', 'avg_dt_ln',
            'avg_t_b_calc', 'avg_t_h_calc', 'avg_q_hb', 'avg_q_ba', 'avg_volume_flow',
            'avg_mass_flow', 'avg_dp', 'avg_adj_dp', 'avg_P_hyd', 'avg_eff_pump',
            'avg_qcorr_bam', 'avg_pcorr_bam', 'avg_power_buh', 'avg_power_buh_from_ts', 'avg_t_sup_buh'
        ] + list(dataset_manager.OPTIONAL_MONITORING_MEAN_COLUMNS)
        
        # Get all column names
        all_column_names = [col[1] for col in columns_info]
        
        # Identify custom columns (not in standard or calculated lists)
        # Exclude deviation columns (dev_*) - these are only shown on the deviations page
        system_columns = {'rowid', 'has_time_series', 'time_series_data', 'display_order'}
        deviation_columns = {col for col in all_column_names if col.startswith('dev_')}
        custom_columns = [col for col in all_column_names 
                        if col not in standard_columns 
                        and col not in calculated_columns 
                        and col not in system_columns
                        and col not in deviation_columns]
        
        # Create ordered column list: standard -> calculated -> custom
        # Only columns that exist on this database (a new empty DB does not yet have
        # every BAM mean column; listing them made index.html 500 on columns_info[col]).
        default_ordered_columns = (
            [c for c in standard_columns if c in all_column_names]
            + [c for c in calculated_columns if c in all_column_names]
            + custom_columns
        )
        
        # Try to load saved column order
        import json
        import os
        config_file = os.path.join(os.path.dirname(__file__), 'column_order.json')
        saved_order = []
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    saved_order = data.get('column_order', [])
            except Exception as e:
                print(f"Error loading column order: {e}")
        
        # Load column visibility settings
        column_visibility = load_column_visibility()
        
        # Use saved order if available, otherwise use default
        # BUT always ensure standard columns (file_name, data_set, start_time, end_time) come first
        # Filter columns based on visibility settings (default: hide deviation columns)
        if saved_order:
            name_map = _results_column_map(all_column_names)
            saved_ordered = []
            seen = set()
            for col in saved_order:
                resolved = resolve_results_column(col, all_column_names, name_map)
                if not resolved or resolved in standard_columns or resolved in seen:
                    continue
                saved_ordered.append(resolved)
                seen.add(resolved)
            
            # Start with standard columns, then add saved order, then any missing columns
            ordered_columns = [col for col in standard_columns if col in all_column_names]
            ordered_columns.extend(saved_ordered)
            
            # Add any missing columns to the end
            missing = [col for col in default_ordered_columns if col not in ordered_columns]
            ordered_columns.extend(missing)
        else:
            ordered_columns = default_ordered_columns
        ordered_columns = [c for c in ordered_columns if c in all_column_names]
        
        # Filter columns based on visibility settings.
        # Hidden set / monitoring extras / dev_* / dtreturn_cache_* default False
        # when missing from JSON so a clone does not get a 40-column wall.
        visible_columns = []
        for col in ordered_columns:
            if means_column_is_visible(col, column_visibility, deviation_columns):
                visible_columns.append(col)

        monitoring_mean_columns = [
            c for c in dataset_manager.OPTIONAL_MONITORING_MEAN_COLUMNS
            if c in all_column_names
        ]
        show_monitoring_columns = bool(monitoring_mean_columns) and all(
            means_column_is_visible(c, column_visibility, deviation_columns)
            for c in monitoring_mean_columns
        )
        
        ordered_columns = visible_columns
        
        # Detect COP_Dataset and scatter_dataset column names for toggle buttons
        cop_dataset_column = None
        scatter_dataset_column = None
        if results:
            first_entry = results[0]
            all_keys = list(first_entry.keys())
            for key in all_keys:
                key_lower = key.lower()
                if not cop_dataset_column and (key_lower == 'cop_dataset' or key_lower == 'cop dataset' or 
                    'cop' in key_lower and 'dataset' in key_lower):
                    cop_dataset_column = key
                if not scatter_dataset_column and (key_lower == 'scatter_dataset' or key_lower == 'scatter dataset'):
                    scatter_dataset_column = key
        
        return render_template('index.html', 
                             results=results, 
                             last_selected=last_selected,
                             ordered_columns=ordered_columns,
                             columns_info={col[1]: {'type': col[2]} for col in columns_info},
                             cop_dataset_column=cop_dataset_column,
                             scatter_dataset_column=scatter_dataset_column,
                             optional_mean_columns=list(dataset_manager.OPTIONAL_MONITORING_MEAN_COLUMNS),
                             show_monitoring_columns=show_monitoring_columns,
                             import_notices=import_notices,
                             last_inserted_rowids=session.get('last_inserted_rowids', []))
    finally:
        conn.close()

def process_new_data(file_name, data_set, start_time, end_time, update_existing=False, old_start_time=None, old_end_time=None, return_rowid: bool = False, notices_out=None):
    """Process new data and update database if needed.

    If notices_out is a list, user-facing processing notes (mass-flow derivation,
    BAM-corr fallback, …) are appended for the caller / UI.
    """
    print(f"=== PROCESS_NEW_DATA CALLED ===")
    print(f"Processing file: {file_name}, dataset: {data_set}, time range: {start_time}-{end_time}, update_existing={update_existing}")
    if old_start_time is not None and old_end_time is not None:
        print(f"Old time range: {old_start_time}-{old_end_time} (for WHERE clause)")
    
    try:
        df_filtered = process_file(file_name, data_set, start_time, end_time)
        print(f"process_file returned: {df_filtered is not None}")
        
        if df_filtered is not None:
            print(f"DataFrame shape: {df_filtered.shape}")
            conn = get_db_connection()
            try:
                print("Calling calculate_and_insert...")
                cur = conn.cursor()
                local_notices = []
                calculate_and_insert(
                    cur, df_filtered, file_name, data_set, start_time, end_time,
                    update_existing=update_existing, old_start_time=old_start_time,
                    old_end_time=old_end_time, notices_out=local_notices,
                )
                tagged = [_tag_notice_with_file(file_name, n) for n in local_notices]
                if notices_out is not None:
                    notices_out.extend(tagged)
                # Persist raw (tagged) notices for the mean-values page; collapse at display.
                if tagged:
                    prev = session.get('data_import_notices') or []
                    merged = list(prev)
                    for n in tagged:
                        if n not in merged:
                            merged.append(n)
                    session['data_import_notices'] = merged[-20:]
                conn.commit()
                action = "updated" if update_existing else "inserted"
                print(f"Successfully {action} data for {file_name}, dataset {data_set}")
                if return_rowid and not update_existing:
                    cur.execute(
                        "SELECT rowid FROM results WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?",
                        (file_name, data_set, int(float(start_time)), int(float(end_time))),
                    )
                    r = cur.fetchone()
                    return int(r["rowid"]) if r else None
                return True  # Success
            except Exception as e:
                print(f"Error in calculate_and_insert: {e}")
                import traceback
                traceback.print_exc()
                return None if return_rowid else False  # Error
            finally:
                conn.close()
        else:
            print(f"Failed to process file {file_name} - file not found or no data in time range")
            return None if return_rowid else False  # File not found or no data
    except Exception as e:
        print(f"Error in process_new_data: {e}")
        import traceback
        traceback.print_exc()
        return None if return_rowid else False


@app.route('/add_bulk_data', methods=['POST'])
def add_bulk_data():
    print("=== ADD_BULK_DATA ROUTE CALLED ===")
    print(f"Form data: {dict(request.form)}")
    try:
        bulk_data = request.form['bulk_data'].strip()
        if not bulk_data:
            flash('Please enter some data first.', 'error')
            return redirect(url_for('index'))
        
        entries_added = 0
        entries_skipped = 0
        errors = []
        inserted_rowids = []
        
        # Split by lines and process each row
        lines = bulk_data.split('\n')
        print(f"Processing {len(lines)} lines")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            print(f"Processing line {i+1}: {line}")
            
            try:
                # Try to split by tab first, then by comma, then by multiple spaces
                columns = line.split('\t')
                if len(columns) < 4:
                    columns = line.split(',')
                if len(columns) < 4:
                    # Split by multiple spaces
                    import re
                    columns = re.split(r'\s+', line)
                
                print(f"Columns: {columns}")
                
                if len(columns) < 4:
                    errors.append(f"Line {i+1}: Not enough columns - '{line}'")
                    continue
                
                file_name = columns[0].strip()
                data_set = float(columns[1].strip())  # Convert to float to match existing database format
                start_time = float(columns[2].strip())
                end_time = float(columns[3].strip())
                
                print(f"Parsed: file_name={file_name}, data_set={data_set}, start_time={start_time}, end_time={end_time}")
                
                # Check if entry already exists (check all four columns: file_name, data_set, start_time, end_time)
                conn = get_db_connection()
                existing = conn.execute(
                    'SELECT * FROM results WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?',
                    (file_name, data_set, start_time, end_time)
                ).fetchone()
                conn.close()
                
                if existing:
                    print(f"Entry already exists (all 4 columns match): {file_name}, {data_set}, {start_time}, {end_time}")
                    entries_skipped += 1
                    continue
                
                print(f"Processing new data for {file_name}, {data_set}")
                rowid = process_new_data(file_name, data_set, start_time, end_time, return_rowid=True)
                if rowid:
                    entries_added += 1
                    inserted_rowids.append(int(rowid))
                    print(f"Successfully processed entry {entries_added}")
                else:
                    errors.append(f"Line {i+1}: File not found or no data in time range - '{line}'")
                
            except ValueError as e:
                print(f"ValueError on line {i+1}: {e}")
                errors.append(f"Line {i+1}: Invalid numeric values - '{line}'")
            except Exception as e:
                print(f"Exception on line {i+1}: {e}")
                errors.append(f"Line {i+1}: {str(e)} - '{line}'")
        
        print(f"Final results: added={entries_added}, skipped={entries_skipped}, errors={len(errors)}")
        
        # Flash appropriate message
        if entries_added > 0:
            flash(f'✅ Successfully added {entries_added} entries!', 'success')
            session['last_inserted_rowids'] = inserted_rowids
        if entries_skipped > 0:
            flash(f'⚠️ Skipped {entries_skipped} entries (already exist)', 'warning')
        if errors:
            error_msg = f'Errors in {len(errors)} lines: ' + '; '.join(errors[:3])
            if len(errors) > 3:
                error_msg += f' ... and {len(errors) - 3} more'
            flash('❌ ' + error_msg, 'danger')
    
    except Exception as e:
        print(f"Exception in add_bulk_data: {e}")
        flash(f'❌ Error adding data: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

@app.route('/delete_data', methods=['POST'])
def delete_data():
    try:
        file_name = request.form['file_name']
        data_set = request.form['data_set']
        
        conn = get_db_connection()
        conn.execute('DELETE FROM results WHERE file_name = ? AND data_set = ?',
                    (file_name, data_set))
        conn.commit()
        conn.close()
        
        flash('Data deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting data: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/edit_entry', methods=['POST'])
def edit_entry():
    """Edit entry file_name, data_set, start_time and end_time, then recalculate all values"""
    try:
        # Get new values
        new_file_name = request.form['file_name'].strip()
        new_data_set = float(request.form['data_set'])
        new_start_time = int(float(request.form['start_time']))
        new_end_time = int(float(request.form['end_time']))
        
        # Get old values (for finding the record to update)
        old_file_name = request.form['old_file_name'].strip()
        old_data_set = float(request.form['old_data_set'])
        old_start_time = int(float(request.form['old_start_time']))
        old_end_time = int(float(request.form['old_end_time']))
        
        # Validate inputs
        if not new_file_name:
            flash('Error: File name cannot be empty', 'danger')
            return redirect(url_for('index'))
        
        if new_start_time >= new_end_time:
            flash('Error: Start time must be less than end time', 'danger')
            return redirect(url_for('index'))
        
        # Get the original record
        conn = get_db_connection()
        record = conn.execute(
            'SELECT * FROM results WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?',
            (old_file_name, old_data_set, old_start_time, old_end_time)
        ).fetchone()
        conn.close()
        
        if not record:
            flash('Record not found', 'danger')
            return redirect(url_for('index'))
        
        # Check if new file_name/data_set combination already exists (and it's not the same entry)
        conn = get_db_connection()
        existing = conn.execute(
            'SELECT * FROM results WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?',
            (new_file_name, new_data_set, new_start_time, new_end_time)
        ).fetchone()
        conn.close()
        
        if existing and (existing['file_name'] != old_file_name or 
                        existing['data_set'] != old_data_set or 
                        existing['start_time'] != old_start_time or 
                        existing['end_time'] != old_end_time):
            flash(f'Error: An entry with file_name="{new_file_name}", data_set={new_data_set}, '
                  f'start_time={new_start_time}, end_time={new_end_time} already exists', 'danger')
            return redirect(url_for('index'))
        
        # Validate source file exists
        file_path = os.path.join(config['data_dir'], new_file_name)
        if not os.path.exists(file_path):
            flash(f'Source file not found for {new_file_name}', 'danger')
            return redirect(url_for('index'))
        
        # Check if new time range has data
        try:
            df_test = process_file(new_file_name, new_data_set, new_start_time, new_end_time)
            if df_test is None or len(df_test) == 0:
                flash(f'No data found in the specified time range ({new_start_time} - {new_end_time}) for {new_file_name}', 'danger')
                return redirect(url_for('index'))
        except Exception as e:
            flash(f'Error validating time range: {str(e)}', 'danger')
            return redirect(url_for('index'))
        
        # If file_name or data_set changed, we need to delete the old entry and create a new one
        # Otherwise, we can just update the existing entry
        file_or_dataset_changed = (new_file_name != old_file_name or abs(new_data_set - old_data_set) > 0.001)
        # Snapshot before delete so we can re-apply notes / custom columns onto the new row.
        old_row_snapshot = dict(record) if file_or_dataset_changed else None

        if file_or_dataset_changed:
            # Delete the old entry first
            conn = get_db_connection()
            conn.execute(
                'DELETE FROM results WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?',
                (old_file_name, old_data_set, old_start_time, old_end_time)
            )
            conn.commit()
            conn.close()
            
            # Create new entry with new values
            start_ts = datetime.now()
            success = process_new_data(new_file_name, new_data_set, new_start_time, new_end_time, update_existing=False)
            duration_ms = int((datetime.now() - start_ts).total_seconds() * 1000)
            if success and old_row_snapshot:
                preserve_row_metadata_after_key_change(
                    old_row_snapshot,
                    new_file_name,
                    new_data_set,
                    new_start_time,
                    new_end_time,
                )
        else:
            # Just update times and recalculate
            start_ts = datetime.now()
            success = process_new_data(new_file_name, new_data_set, new_start_time, new_end_time, 
                                     update_existing=True, old_start_time=old_start_time, old_end_time=old_end_time)
            duration_ms = int((datetime.now() - start_ts).total_seconds() * 1000)
        
        if success:
            # Build change message
            changes = []
            if new_file_name != old_file_name:
                changes.append(f'file_name: {old_file_name} → {new_file_name}')
            if abs(new_data_set - old_data_set) > 0.001:
                changes.append(f'data_set: {old_data_set} → {new_data_set}')
            if new_start_time != old_start_time or new_end_time != old_end_time:
                changes.append(f'time_range: {old_start_time}-{old_end_time} → {new_start_time}-{new_end_time}')
            
            change_msg = ', '.join(changes) if changes else 'no changes'
            
            # Log the edit event
            log_recalc_event(
                new_file_name, new_data_set, 
                status='success', 
                scope='all', 
                message=f'Edited entry: {change_msg}',
                duration_ms=duration_ms
            )
            extra = ''
            if file_or_dataset_changed:
                extra = ' Custom fields (notes, lab id, etc.) were kept; time-series flags were reset for the new file/dataset.'
            flash(
                f'Successfully updated entry. Changes: {change_msg}. All values have been recalculated.{extra}',
                'success'
            )
        else:
            log_recalc_event(
                new_file_name, new_data_set, 
                status='failed', 
                scope='all', 
                message=f'Failed to edit entry',
                duration_ms=duration_ms
            )
            flash(f'Failed to update entry', 'danger')
        
    except ValueError as e:
        flash(f'Invalid input: {str(e)}', 'danger')
    except Exception as e:
        log_recalc_event(
            request.form.get('file_name', ''), 
            request.form.get('data_set', ''), 
            status='error', 
            scope='all', 
            message=f'Error editing entry: {str(e)}'
        )
        flash(f'Error editing entry: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()
    
    return redirect(url_for('index'))

@app.route('/bulk_delete_data', methods=['POST'])
def bulk_delete_data():
    try:
        file_names = request.form.getlist('file_names')
        data_sets = request.form.getlist('data_sets')
        row_ids = request.form.getlist('row_ids')
        
        if len(file_names) != len(data_sets):
            flash('❌ Error: Mismatched file names and data sets', 'danger')
            return redirect(url_for('index'))
        
        conn = get_db_connection()
        deleted_count = 0
        
        for i, (file_name, data_set) in enumerate(zip(file_names, data_sets)):
            if i < len(row_ids) and row_ids[i] not in (None, '', 'undefined'):
                try:
                    rid = int(row_ids[i])
                    cur = conn.execute('DELETE FROM results WHERE rowid = ?', (rid,))
                    deleted_count += cur.rowcount
                    continue
                except (ValueError, TypeError):
                    pass
            conn.execute(
                'DELETE FROM results WHERE file_name = ? AND data_set = ?',
                (file_name, data_set)
            )
            deleted_count += 1
        
        conn.commit()
        conn.close()
        
        flash(f'✅ Successfully deleted {deleted_count} record(s)!', 'success')
    except Exception as e:
        flash(f'❌ Error deleting data: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

@app.route('/recalculate_entry', methods=['POST'])
def recalculate_entry():
    """Recalculate a single entry"""
    try:
        file_name = request.form['file_name']
        data_set = request.form['data_set']
        # columns to recalc (optional, multiple)
        recalc_columns = request.form.getlist('recalc_columns')
        
        # Get the original record
        conn = get_db_connection()
        record = conn.execute(
            'SELECT * FROM results WHERE file_name = ? AND data_set = ?',
            (file_name, data_set)
        ).fetchone()
        conn.close()
        
        if not record:
            flash('Record not found', 'danger')
            return redirect(url_for('index'))
        
        # Validate source file exists before recalculation
        file_path = os.path.join(config['data_dir'], file_name)
        if not os.path.exists(file_path):
            msg = 'Source file not found'
            if recalc_columns:
                msg += '; columns=' + ','.join(recalc_columns)
            log_recalc_event(file_name, data_set, status='failed', scope='columns' if recalc_columns else 'all', message=msg)
            flash(f'Source file not found for {file_name}', 'danger')
            return redirect(url_for('index'))

        # Recalculate using the same process as new data; measure duration
        start_ts = datetime.now()
        # Note: selective recompute not enforced yet; record requested columns in history
        success = process_new_data(file_name, data_set, record['start_time'], record['end_time'], update_existing=True)
        duration_ms = int((datetime.now() - start_ts).total_seconds() * 1000)
        
        if success:
            msg = None
            if recalc_columns:
                msg = 'columns=' + ','.join(recalc_columns)
            log_recalc_event(file_name, data_set, status='success', scope='columns' if recalc_columns else 'all', message=msg, duration_ms=duration_ms)
            flash(f'Successfully recalculated {file_name}, dataset {data_set}', 'success')
        else:
            msg = 'process_new_data returned False'
            if recalc_columns:
                msg += '; columns=' + ','.join(recalc_columns)
            log_recalc_event(file_name, data_set, status='failed', scope='columns' if recalc_columns else 'all', duration_ms=duration_ms, message=msg)
            flash(f'Failed to recalculate {file_name}, dataset {data_set}', 'danger')
        
    except Exception as e:
        cols = request.form.getlist('recalc_columns')
        log_recalc_event(request.form.get('file_name', ''), request.form.get('data_set', ''), status='error', scope='columns' if cols else 'all', message=str(e))
        flash(f'Error recalculating entry: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

@app.route('/bulk_recalculate', methods=['POST'])
def bulk_recalculate():
    """Recalculate multiple selected entries"""
    try:
        file_names = request.form.getlist('file_names')
        data_sets = request.form.getlist('data_sets')
        recalc_columns = request.form.getlist('recalc_columns')
        
        if len(file_names) != len(data_sets):
            flash('Error: Mismatched file names and data sets', 'danger')
            return redirect(url_for('index'))
        
        if not file_names:
            flash('No records selected for recalculation', 'danger')
            return redirect(url_for('index'))
        
        row_ids = request.form.getlist('row_ids')
        # Get records from database
        conn = get_db_connection()
        records_to_recalculate = resolve_selected_rows(conn, file_names, data_sets, row_ids)
        
        conn.close()
        
        if not records_to_recalculate:
            flash('No valid records found for recalculation', 'warning')
            return redirect(url_for('index'))
        
        # Recalculate each record
        recalculated_count = 0
        failed_count = 0
        errors = []
        
        for record in records_to_recalculate:
            # Validate source file exists
            file_path = os.path.join(config['data_dir'], record['file_name'])
            if not os.path.exists(file_path):
                msg = 'Source file not found'
                if recalc_columns:
                    msg += '; columns=' + ','.join(recalc_columns)
                log_recalc_event(record['file_name'], record['data_set'], status='failed', scope='columns' if recalc_columns else 'all', message=msg)
                failed_count += 1
                errors.append(f"{record['file_name']} dataset {record['data_set']} (missing file)")
                continue

            start_ts = datetime.now()
            success = process_new_data(
                record['file_name'], 
                record['data_set'], 
                record['start_time'], 
                record['end_time'],
                update_existing=True  # This prevents duplicate entries
            )
            duration_ms = int((datetime.now() - start_ts).total_seconds() * 1000)
            
            if success:
                msg = None
                if recalc_columns:
                    msg = 'columns=' + ','.join(recalc_columns)
                log_recalc_event(record['file_name'], record['data_set'], status='success', scope='columns' if recalc_columns else 'all', message=msg, duration_ms=duration_ms)
                recalculated_count += 1
            else:
                msg = 'process_new_data returned False'
                if recalc_columns:
                    msg += '; columns=' + ','.join(recalc_columns)
                log_recalc_event(record['file_name'], record['data_set'], status='failed', scope='columns' if recalc_columns else 'all', duration_ms=duration_ms, message=msg)
                failed_count += 1
                errors.append(f"{record['file_name']} dataset {record['data_set']}")
        
        # Flash results
        if recalculated_count > 0:
            flash(f'Successfully recalculated {recalculated_count} record(s)!', 'success')
        
        if failed_count > 0:
            error_msg = f'Failed to recalculate {failed_count} record(s): ' + ', '.join(errors[:3])
            if len(errors) > 3:
                error_msg += f' ... and {len(errors) - 3} more'
            flash(error_msg, 'danger')
        
    except Exception as e:
        flash(f'Error during bulk recalculation: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

# REMOVED: Old /recalculate_column route that loaded Excel files
# This route has been replaced by /recalculate_column -> recalculate_single_column
# which only recalculates derived columns using stored formulas without loading Excel files

@app.route('/cleanup_duplicates', methods=['GET', 'POST'])
def cleanup_duplicates():
    """Remove duplicate entries from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total count of all entries in database
        cursor.execute("SELECT COUNT(*) as total FROM results")
        total_entries = cursor.fetchone()['total']
        
        if request.method == 'GET':
            # Preview mode: show what duplicates exist
            # Find duplicates based on (file_name, data_set, start_time, end_time)
            cursor.execute("""
                SELECT file_name, data_set, start_time, end_time, COUNT(*) as count
                FROM results
                GROUP BY file_name, data_set, start_time, end_time
                HAVING COUNT(*) > 1
                ORDER BY count DESC, file_name, data_set
            """)
            duplicate_groups = cursor.fetchall()
            
            # Get total unique entries (after removing duplicates) - count distinct combinations
            cursor.execute("""
                SELECT COUNT(*) as unique_count
                FROM (
                    SELECT DISTINCT file_name, data_set, start_time, end_time
                    FROM results
                )
            """)
            unique_entries = cursor.fetchone()['unique_count']
            
            # Get total duplicate count
            total_duplicates = 0
            duplicate_details = []
            for group in duplicate_groups:
                count = group['count']
                duplicates_to_remove = count - 1  # Keep one, remove the rest
                total_duplicates += duplicates_to_remove
                
                # Get the actual duplicate rows
                cursor.execute("""
                    SELECT rowid, file_name, data_set, start_time, end_time
                    FROM results
                    WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?
                    ORDER BY rowid ASC
                """, (group['file_name'], group['data_set'], group['start_time'], group['end_time']))
                rows = cursor.fetchall()
                
                # Keep the first (original) rowid, mark others for deletion
                rows_to_keep = [rows[0]['rowid']]
                rows_to_delete = [r['rowid'] for r in rows[1:]]
                
                duplicate_details.append({
                    'file_name': group['file_name'],
                    'data_set': group['data_set'],
                    'start_time': group['start_time'],
                    'end_time': group['end_time'],
                    'total_count': count,
                    'to_remove': duplicates_to_remove,
                    'keep_rowid': rows_to_keep[0],
                    'delete_rowids': rows_to_delete
                })
            
            conn.close()
            
            # Calculate expected entries after cleanup
            expected_after_cleanup = total_entries - total_duplicates
            
            return render_template('cleanup_duplicates.html', 
                                 duplicate_groups=duplicate_groups,
                                 duplicate_details=duplicate_details,
                                 total_duplicates=total_duplicates,
                                 total_groups=len(duplicate_groups),
                                 total_entries=total_entries,
                                 unique_entries=unique_entries,
                                 expected_after_cleanup=expected_after_cleanup)
        
        else:  # POST - actually remove duplicates
            # CRITICAL: Verify we're checking all four columns
            # Find all duplicates - MUST use all four columns: file_name, data_set, start_time, end_time
            cursor.execute("""
                SELECT file_name, data_set, start_time, end_time, COUNT(*) as count
                FROM results
                GROUP BY file_name, data_set, start_time, end_time
                HAVING COUNT(*) > 1
            """)
            duplicate_groups = cursor.fetchall()
            
            # Log what we're about to delete for safety
            print(f"=== CLEANUP DUPLICATES: Found {len(duplicate_groups)} duplicate groups ===")
            for group in duplicate_groups[:10]:  # Log first 10
                print(f"  Group: {group['file_name']}, dataset={group['data_set']}, start={group['start_time']}, end={group['end_time']}, count={group['count']}")
            
            total_removed = 0
            errors = []
            deleted_details = []  # Track what we delete for potential recovery
            
            for group in duplicate_groups:
                try:
                    # CRITICAL: Must check all four columns in WHERE clause
                    # Get all rows for this duplicate group, ordered by rowid (keep the first/original)
                    cursor.execute("""
                        SELECT rowid, file_name, data_set, start_time, end_time
                        FROM results
                        WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?
                        ORDER BY rowid ASC
                    """, (group['file_name'], group['data_set'], group['start_time'], group['end_time']))
                    rows = cursor.fetchall()
                    
                    if len(rows) > 1:
                        # Verify all rows have matching values for all four columns
                        for row in rows:
                            if (row['file_name'] != group['file_name'] or 
                                row['data_set'] != group['data_set'] or 
                                row['start_time'] != group['start_time'] or 
                                row['end_time'] != group['end_time']):
                                errors.append(f"ERROR: Row {row['rowid']} doesn't match group criteria!")
                                continue
                        
                        # Keep the first row (lowest rowid = original), delete the rest
                        rowids_to_delete = [r['rowid'] for r in rows[1:]]
                        
                        # Log what we're deleting
                        deleted_details.append({
                            'file_name': group['file_name'],
                            'data_set': group['data_set'],
                            'start_time': group['start_time'],
                            'end_time': group['end_time'],
                            'kept_rowid': rows[0]['rowid'],
                            'deleted_rowids': rowids_to_delete
                        })
                        
                        for rowid in rowids_to_delete:
                            cursor.execute("DELETE FROM results WHERE rowid = ?", (rowid,))
                            total_removed += 1
                            print(f"  Deleted rowid {rowid} (file={group['file_name']}, dataset={group['data_set']}, start={group['start_time']}, end={group['end_time']})")
                
                except Exception as e:
                    errors.append(f"Error removing duplicates for {group['file_name']}, dataset {group['data_set']}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            if total_removed > 0:
                conn.commit()
                flash(f'Successfully removed {total_removed} duplicate entries!', 'success')
            else:
                flash('No duplicates found to remove.', 'info')
            
            if errors:
                flash(f'Some errors occurred: {"; ".join(errors[:5])}', 'warning')
            
            conn.close()
            return redirect(url_for('index'))
            
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        flash(f'Error cleaning up duplicates: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))

@app.route('/move_entry', methods=['POST'])
def move_entry():
    """Move an entry up or down in the display order"""
    try:
        data = request.get_json()
        print(f"Received move_entry request: {data}")
        
        file_name = data.get('file_name')
        data_set = data.get('data_set')
        direction = data.get('direction')  # 'up' or 'down'
        target_position = data.get('target_position')  # Optional: specific position
        rowid = data.get('rowid')  # Use rowid if provided for exact identification
        current_visual_position = data.get('current_visual_position')  # Visual position from frontend
        
        # Convert data_set to int if it's a string or float
        if isinstance(data_set, str):
            try:
                data_set = int(float(data_set))  # Handle "1.0" -> 1
            except (ValueError, TypeError):
                return jsonify({'success': False, 'message': f'Invalid data_set value: {data_set}'})
        elif isinstance(data_set, float):
            data_set = int(data_set)  # Handle 1.0 -> 1
        
        if not file_name or data_set is None:
            return jsonify({'success': False, 'message': 'File name and data set are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure display_order column exists and is initialized
        init_display_order_column()
        
        # Initialize current_entry
        current_entry = None
        
        # Get current entry - prefer rowid if provided (most reliable)
        if rowid:
            print(f"Using rowid={rowid} to find entry (most reliable method)")
            cursor.execute("""
                SELECT rowid, display_order, data_set, file_name 
                FROM results 
                WHERE rowid = ?
            """, (rowid,))
            current_entry = cursor.fetchone()
            if current_entry:
                print(f"Found entry by rowid: file_name={current_entry['file_name']}, data_set={current_entry['data_set']}, display_order={current_entry['display_order']}")
        else:
            print(f"rowid not provided, using file_name and data_set to find entry")
        
        # If not found by rowid, try file_name and data_set
        if not current_entry:
            # First, try exact match
            cursor.execute("""
                SELECT rowid, display_order, data_set 
                FROM results 
                WHERE file_name = ? AND data_set = ?
            """, (file_name, data_set))
            current_entry = cursor.fetchone()
        
        if not current_entry:
            # Try with string conversion (in case data_set is stored as text)
            cursor.execute("""
                SELECT rowid, display_order, data_set 
                FROM results 
                WHERE file_name = ? AND CAST(data_set AS TEXT) = ?
            """, (file_name, str(data_set)))
            current_entry = cursor.fetchone()
        
        if not current_entry:
            # Try with numeric conversion (in case data_set is stored as number but we're comparing as string)
            cursor.execute("""
                SELECT rowid, display_order, data_set 
                FROM results 
                WHERE file_name = ? AND CAST(data_set AS INTEGER) = ?
            """, (file_name, int(data_set)))
            current_entry = cursor.fetchone()
        
        if not current_entry:
            # Try with trimmed file_name (in case of whitespace)
            cursor.execute("""
                SELECT rowid, display_order, data_set 
                FROM results 
                WHERE TRIM(file_name) = ? AND CAST(data_set AS INTEGER) = ?
            """, (file_name.strip(), int(data_set)))
            current_entry = cursor.fetchone()
        
        if not current_entry:
            # Debug: Show what entries exist for this file_name
            cursor.execute("""
                SELECT rowid, file_name, data_set, display_order 
                FROM results 
                WHERE file_name LIKE ? OR TRIM(file_name) = ?
                LIMIT 10
            """, (f'%{file_name}%', file_name.strip()))
            similar_entries = cursor.fetchall()
            conn.close()
            print(f"Entry not found: file_name='{file_name}' (len={len(file_name)}), data_set={data_set} (type: {type(data_set)})")
            print(f"Similar entries for {file_name}:")
            for entry in similar_entries:
                print(f"  - rowid={entry['rowid']}, file_name='{entry['file_name']}' (len={len(entry['file_name'])}), data_set={entry['data_set']} (type: {type(entry['data_set'])})")
            return jsonify({'success': False, 'message': f'Entry not found: {file_name}, dataset {data_set}. Check Flask console for similar entries.'})
        
        current_order = current_entry['display_order']
        current_rowid = current_entry['rowid']
        
        if current_order is None:
            # Initialize with rowid if NULL
            cursor.execute("UPDATE results SET display_order = ? WHERE rowid = ?", (current_rowid, current_rowid))
            current_order = current_rowid
            conn.commit()
            print(f"Initialized display_order for entry: {current_order}")
        
        # Normalize display_order FIRST so that display_order = visual position
        # This ensures consistency - display_order 1 = visual position 1, etc.
        print(f"Normalizing display_order so it matches visual positions...")
        normalize_display_order(conn, cursor)
        
        # After normalization, re-fetch the current entry's position
        cursor.execute("SELECT display_order FROM results WHERE rowid = ?", (current_rowid,))
        normalized_entry = cursor.fetchone()
        if normalized_entry:
            current_order = normalized_entry['display_order']
            print(f"After normalization: display_order={current_order}")
        
        print(f"Moving entry: file={file_name}, dataset={data_set}, current_order={current_order}, direction={direction}")
        
        if target_position is not None:
            # Move to specific position
            new_order = target_position
            # Shift other entries
            if new_order < current_order:
                cursor.execute("""
                    UPDATE results 
                    SET display_order = display_order + 1 
                    WHERE display_order >= ? AND display_order < ? AND rowid != ?
                """, (new_order, current_order, current_rowid))
            else:
                cursor.execute("""
                    UPDATE results 
                    SET display_order = display_order - 1 
                    WHERE display_order > ? AND display_order <= ? AND rowid != ?
                """, (current_order, new_order, current_rowid))
        else:
            # Move up or down by one
            if direction == 'up':
                # Find entry above
                cursor.execute("""
                    SELECT rowid, display_order 
                    FROM results 
                    WHERE display_order < ? 
                    ORDER BY display_order DESC 
                    LIMIT 1
                """, (current_order,))
                above_entry = cursor.fetchone()
                
                if above_entry:
                    # Swap orders
                    cursor.execute("UPDATE results SET display_order = ? WHERE rowid = ?", (above_entry['display_order'], current_rowid))
                    cursor.execute("UPDATE results SET display_order = ? WHERE rowid = ?", (current_order, above_entry['rowid']))
                else:
                    conn.close()
                    return jsonify({'success': False, 'message': 'Entry is already at the top'})
                    
            elif direction == 'down':
                # Find entry below
                cursor.execute("""
                    SELECT rowid, display_order 
                    FROM results 
                    WHERE display_order > ? 
                    ORDER BY display_order ASC 
                    LIMIT 1
                """, (current_order,))
                below_entry = cursor.fetchone()
                
                if below_entry:
                    # Swap orders
                    cursor.execute("UPDATE results SET display_order = ? WHERE rowid = ?", (below_entry['display_order'], current_rowid))
                    cursor.execute("UPDATE results SET display_order = ? WHERE rowid = ?", (current_order, below_entry['rowid']))
                else:
                    conn.close()
                    return jsonify({'success': False, 'message': 'Entry is already at the bottom'})
            else:
                conn.close()
                return jsonify({'success': False, 'message': 'Invalid direction'})
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'Entry moved {direction}'})
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return jsonify({'success': False, 'message': f'Error moving entry: {str(e)}'})

@app.route('/set_entry_position', methods=['POST'])
def set_entry_position():
    """Set a specific entry to a specific position"""
    try:
        data = request.get_json()
        print(f"Received request data: {data}")
        
        file_name = data.get('file_name')
        data_set = data.get('data_set')
        new_position = data.get('position')
        rowid = data.get('rowid')  # Use rowid if provided for exact identification
        current_visual_position = data.get('current_visual_position')  # Visual position from frontend
        
        # Convert data_set to int if it's a string or float
        if isinstance(data_set, str):
            try:
                data_set = int(float(data_set))  # Handle "1.0" -> 1
            except (ValueError, TypeError):
                return jsonify({'success': False, 'message': f'Invalid data_set value: {data_set}'})
        elif isinstance(data_set, float):
            data_set = int(data_set)  # Handle 1.0 -> 1
        
        print(f"Parsed values: file_name={file_name}, data_set={data_set} (type: {type(data_set)}), position={new_position}")
        
        if not file_name or data_set is None or new_position is None:
            return jsonify({'success': False, 'message': 'File name, data set, and position are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure display_order column exists and is initialized
        init_display_order_column()
        
        # Initialize current_entry
        current_entry = None
        
        # Get current entry - prefer rowid if provided (most reliable)
        if rowid:
            print(f"Using rowid={rowid} to find entry (most reliable method)")
            cursor.execute("""
                SELECT rowid, display_order, data_set, file_name 
                FROM results 
                WHERE rowid = ?
            """, (rowid,))
            current_entry = cursor.fetchone()
            if current_entry:
                print(f"Found entry by rowid: file_name={current_entry['file_name']}, data_set={current_entry['data_set']}, display_order={current_entry['display_order']}")
        else:
            print(f"rowid not provided, using file_name and data_set to find entry")
        
        # If not found by rowid, try file_name and data_set
        if not current_entry:
            # First, try exact match
            cursor.execute("""
                SELECT rowid, display_order, data_set 
                FROM results 
                WHERE file_name = ? AND data_set = ?
            """, (file_name, data_set))
            current_entry = cursor.fetchone()
        
        if not current_entry:
            # Try with string conversion (in case data_set is stored as text)
            cursor.execute("""
                SELECT rowid, display_order, data_set 
                FROM results 
                WHERE file_name = ? AND CAST(data_set AS TEXT) = ?
            """, (file_name, str(data_set)))
            current_entry = cursor.fetchone()
        
        if not current_entry:
            # Try with numeric conversion (in case data_set is stored as number but we're comparing as string)
            cursor.execute("""
                SELECT rowid, display_order, data_set 
                FROM results 
                WHERE file_name = ? AND CAST(data_set AS INTEGER) = ?
            """, (file_name, int(data_set)))
            current_entry = cursor.fetchone()
        
        if not current_entry:
            # Try with trimmed file_name (in case of whitespace)
            cursor.execute("""
                SELECT rowid, display_order, data_set 
                FROM results 
                WHERE TRIM(file_name) = ? AND CAST(data_set AS INTEGER) = ?
            """, (file_name.strip(), int(data_set)))
            current_entry = cursor.fetchone()
        
        if not current_entry:
            # Debug: Show what entries exist for this file_name
            cursor.execute("""
                SELECT rowid, file_name, data_set, display_order 
                FROM results 
                WHERE file_name LIKE ? OR TRIM(file_name) = ?
                LIMIT 10
            """, (f'%{file_name}%', file_name.strip()))
            similar_entries = cursor.fetchall()
            conn.close()
            print(f"Entry not found: file_name='{file_name}' (len={len(file_name)}), data_set={data_set} (type: {type(data_set)})")
            print(f"Similar entries for {file_name}:")
            for entry in similar_entries:
                print(f"  - rowid={entry['rowid']}, file_name='{entry['file_name']}' (len={len(entry['file_name'])}), data_set={entry['data_set']} (type: {type(entry['data_set'])})")
            return jsonify({'success': False, 'message': f'Entry not found: {file_name}, dataset {data_set}. Check Flask console for similar entries.'})
        
        current_order = current_entry['display_order']
        current_rowid = current_entry['rowid']
        
        if current_order is None:
            # Initialize with rowid if NULL
            cursor.execute("UPDATE results SET display_order = ? WHERE rowid = ?", (current_rowid, current_rowid))
            current_order = current_rowid
            conn.commit()
            print(f"Initialized display_order for entry: {current_order}")
        
        new_order = int(new_position)
        
        # Get the visual position (rank) - what position the user sees in the table
        # This is based on ORDER BY display_order ASC, rowid ASC
        cursor.execute("""
            SELECT COUNT(*) + 1 as visual_position
            FROM results 
            WHERE display_order < ? OR (display_order = ? AND rowid < ?)
        """, (current_order, current_order, current_rowid))
        visual_position_result = cursor.fetchone()
        visual_position = visual_position_result['visual_position'] if visual_position_result else current_order
        
        # Validate that we found the correct entry
        if current_visual_position and visual_position != int(current_visual_position):
            print(f"WARNING: Visual position mismatch!")
            print(f"  - Frontend says visual position: {current_visual_position}")
            print(f"  - Backend calculated visual position: {visual_position}")
            print(f"  - This might indicate we found the wrong entry!")
            # But continue anyway - the rowid should be correct
        
        print(f"Setting position: file={file_name}, dataset={data_set}")
        print(f"  - Database display_order: {current_order}")
        print(f"  - Visual position (what user sees): {visual_position}")
        if current_visual_position:
            print(f"  - Frontend reported visual position: {current_visual_position}")
        print(f"  - Target position (user wants): {new_order}")
        print(f"  - rowid: {current_rowid}")
        
        # Get total count to validate position
        cursor.execute("SELECT COUNT(*) as total FROM results")
        total_count = cursor.fetchone()['total']
        
        if new_order < 1 or new_order > total_count:
            conn.close()
            return jsonify({'success': False, 'message': f'Position must be between 1 and {total_count}'})
        
        # Check if already at target position using VISUAL position (what user sees)
        if new_order == visual_position:
            conn.close()
            print(f"Entry is already at visual position {new_order} (display_order={current_order})")
            return jsonify({
                'success': True, 
                'message': f'Entry is already at position {new_order} (visual_position={visual_position}, display_order={current_order})'
            })
        
        # Normalize display_order FIRST so that display_order = visual position
        # This ensures consistency - display_order 1 = visual position 1, etc.
        print(f"Normalizing display_order so it matches visual positions...")
        normalize_display_order(conn, cursor)
        
        # After normalization, display_order should equal visual position
        # Re-fetch the current entry's position
        cursor.execute("SELECT display_order FROM results WHERE rowid = ?", (current_rowid,))
        normalized_entry = cursor.fetchone()
        if normalized_entry:
            current_order = normalized_entry['display_order']
            print(f"After normalization: display_order={current_order}, target={new_order}")
            
            # Check again if already at target (after normalization)
            if new_order == current_order:
                conn.close()
                print(f"Entry is already at position {new_order} after normalization")
                # Include debug info in the message
                return jsonify({
                    'success': True, 
                    'message': f'Entry is already at position {new_order} (display_order={current_order} after normalization). Check Flask console for details.'
                })
        
        # Now shift entries - current_order and new_order are both display_order values
        # and they match visual positions after normalization
        if new_order < current_order:
            # Moving up: shift entries from new_position to current_position-1 down by 1
            cursor.execute("""
                UPDATE results 
                SET display_order = display_order + 1 
                WHERE display_order >= ? AND display_order < ? AND rowid != ?
            """, (new_order, current_order, current_rowid))
            rows_shifted = cursor.rowcount
            print(f"Moving UP: Shifted {rows_shifted} entries from position {new_order} to {current_order-1} down by 1")
        elif new_order > current_order:
            # Moving down: shift entries from current_position+1 to new_position up by 1
            cursor.execute("""
                UPDATE results 
                SET display_order = display_order - 1 
                WHERE display_order > ? AND display_order <= ? AND rowid != ?
            """, (current_order, new_order, current_rowid))
            rows_shifted = cursor.rowcount
            print(f"Moving DOWN: Shifted {rows_shifted} entries from position {current_order+1} to {new_order} up by 1")
        else:
            # Already at the target position
            conn.close()
            return jsonify({'success': True, 'message': 'Entry is already at the target position'})
        
        # Set new position
        cursor.execute("UPDATE results SET display_order = ? WHERE rowid = ?", (new_order, current_rowid))
        rows_updated = cursor.rowcount
        
        print(f"Updated {rows_updated} row(s) - setting entry (rowid={current_rowid}) to position {new_order}")
        
        # Verify the update worked
        cursor.execute("SELECT display_order FROM results WHERE rowid = ?", (current_rowid,))
        verify_entry = cursor.fetchone()
        if verify_entry:
            print(f"Verification: Entry now has display_order = {verify_entry['display_order']}")
        
        # Commit the transaction
        conn.commit()
        print(f"Transaction committed successfully")
        
        conn.close()
        
        return jsonify({'success': True, 'message': f'Entry moved to position {new_position}'})
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return jsonify({'success': False, 'message': f'Error setting position: {str(e)}'})

@app.route('/diagnose_database', methods=['GET'])
def diagnose_database():
    """Diagnostic tool to analyze database state and check for potential issues"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as total FROM results")
        total_entries = cursor.fetchone()['total']
        
        # Check for entries with same file_name and data_set but different times
        cursor.execute("""
            SELECT file_name, data_set, COUNT(*) as total_rows
            FROM (
                SELECT DISTINCT file_name, data_set, start_time, end_time
                FROM results
            )
            GROUP BY file_name, data_set
            HAVING COUNT(*) > 1
            ORDER BY total_rows DESC
        """)
        same_file_dataset = cursor.fetchall()
        
        # Check for actual duplicates (all four columns match)
        cursor.execute("""
            SELECT file_name, data_set, start_time, end_time, COUNT(*) as count
            FROM results
            GROUP BY file_name, data_set, start_time, end_time
            HAVING COUNT(*) > 1
        """)
        true_duplicates = cursor.fetchall()
        
        # Get unique combinations
        cursor.execute("""
            SELECT COUNT(*) as unique_count
            FROM (
                SELECT DISTINCT file_name, data_set, start_time, end_time
                FROM results
            )
        """)
        unique_combinations = cursor.fetchone()['unique_count']
        
        # Check for entries that might have been incorrectly grouped
        cursor.execute("""
            SELECT file_name, data_set, COUNT(*) as count
            FROM results
            GROUP BY file_name, data_set
            HAVING COUNT(*) > 1
            ORDER BY count DESC
        """)
        same_file_dataset_groups = cursor.fetchall()
        
        conn.close()
        
        return render_template('diagnose_database.html',
                             total_entries=total_entries,
                             unique_combinations=unique_combinations,
                             same_file_dataset=same_file_dataset,
                             true_duplicates=true_duplicates,
                             same_file_dataset_groups=same_file_dataset_groups)
    except Exception as e:
        flash(f'Error diagnosing database: {str(e)}', 'danger')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))


@app.route('/import_time_series', methods=['POST'])
def import_time_series_route():
    """Import time series data for selected records"""
    try:
        file_name = request.form['file_name']
        data_set = request.form['data_set']
        
        # Get record details
        conn = get_db_connection()
        record = conn.execute(
            'SELECT * FROM results WHERE file_name = ? AND data_set = ?',
            (file_name, data_set)
        ).fetchone()
        conn.close()
        
        if not record:
            flash('Record not found', 'danger')
            return redirect(url_for('index'))
        
        # Import time series
        success = import_time_series(file_name, data_set, record['start_time'], record['end_time'])
        
        if success:
            flash(f'Time series imported for {file_name}, dataset {data_set}', 'success')
        else:
            flash(f'Failed to import time series for {file_name}', 'danger')
        
    except Exception as e:
        flash(f'Error importing time series: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

@app.route('/bulk_import_time_series', methods=['POST'])
def bulk_import_time_series():
    """Import time series data for multiple selected records"""
    try:
        file_names = request.form.getlist('file_names')
        data_sets = request.form.getlist('data_sets')
        
        if len(file_names) != len(data_sets):
            flash('Error: Mismatched file names and data sets', 'danger')
            return redirect(url_for('index'))
        
        if not file_names:
            flash('No records selected for import', 'danger')
            return redirect(url_for('index'))
        
        row_ids = request.form.getlist('row_ids')
        conn = get_db_connection()
        resolved = resolve_selected_rows(conn, file_names, data_sets, row_ids)
        records_to_import = [r for r in resolved if not r.get('has_time_series')]
        conn.close()
        
        if not records_to_import:
            flash('No records found that need time series import', 'warning')
            return redirect(url_for('index'))
        
        # Import time series for each record
        imported_count = 0
        failed_count = 0
        errors = []
        
        for record in records_to_import:
            success = import_time_series(
                record['file_name'], 
                record['data_set'], 
                record['start_time'], 
                record['end_time']
            )
            
            if success:
                imported_count += 1
            else:
                failed_count += 1
                errors.append(f"{record['file_name']} dataset {record['data_set']}")
        
        # Flash results
        if imported_count > 0:
            flash(f'Successfully imported time series for {imported_count} record(s)!', 'success')
        
        if failed_count > 0:
            error_msg = f'Failed to import {failed_count} record(s): ' + ', '.join(errors[:3])
            if len(errors) > 3:
                error_msg += f' ... and {len(errors) - 3} more'
            flash(error_msg, 'danger')
        
    except Exception as e:
        flash(f'Error during bulk import: {str(e)}', 'danger')
    
    return redirect(url_for('index'))

@app.route('/plot_selected', methods=['POST'])
def plot_selected():
    """Create plots for selected records with original time"""
    try:
        # Get selected records from form
        file_names = request.form.getlist('file_names')
        data_sets = request.form.getlist('data_sets')
        plot_columns = request.form.getlist('plot_columns')
        matrix_layout = request.form.get('matrix_layout')
        legend_style = request.form.get('legend_style', 'auto')
        custom_legend_labels = request.form.get('custom_legend_labels', '')
        # Per-variable y-axis limits (default auto)
        y_axis_limits = {}
        for col in plot_columns:
            y_min_s = request.form.get('y_min_' + col, '').strip() or None
            y_max_s = request.form.get('y_max_' + col, '').strip() or None
            y_min, y_max = None, None
            try:
                if y_min_s:
                    y_min = float(y_min_s)
                if y_max_s:
                    y_max = float(y_max_s)
            except ValueError:
                pass
            y_axis_limits[col] = (y_min, y_max)
        
        if not file_names or not data_sets:
            flash('Please select at least one record to plot', 'danger')
            return redirect(url_for('index'))
        
        # Validate custom legend labels if provided
        if legend_style == 'custom' and custom_legend_labels:
            custom_labels = [line.strip() for line in custom_legend_labels.split('\n') if line.strip()]
            if len(custom_labels) != len(file_names):
                flash(f'Custom legend labels count ({len(custom_labels)}) does not match selected records ({len(file_names)}). Please provide exactly {len(file_names)} labels.', 'warning')
                return redirect(url_for('index'))
        
        # Parse matrix layout if provided
        layout = None
        if matrix_layout:
            if matrix_layout == '1':
                layout = (1, 1)  # 1 column = single column layout
            elif matrix_layout == '2':
                layout = (1, 2)  # 2 columns = side-by-side layout
            elif 'x' in matrix_layout:
                try:
                    rows, cols = map(int, matrix_layout.split('x'))
                    layout = (rows, cols)
                except ValueError:
                    pass
        
        row_ids = request.form.getlist('row_ids')
        conn = get_db_connection()
        selected_records = resolve_selected_rows(conn, file_names, data_sets, row_ids)
        conn.close()
        
        if not selected_records:
            flash('No valid records found for plotting', 'danger')
            return redirect(url_for('index'))
        
        # Create plot with original time
        plot_data = create_plot_with_original_time(selected_records, plot_columns, matrix_layout=layout, plot_type='overlay', legend_style=legend_style, custom_legend_labels=custom_legend_labels, y_axis_limits=y_axis_limits)
        
        if plot_data is None:
            flash('Error creating plot - check if Excel files exist and contain required columns', 'danger')
            return redirect(url_for('index'))
        
        # Convert to base64 for display
        plot_base64 = base64.b64encode(plot_data).decode('utf-8')
        
        # Store plot data in a temporary file instead of session
        import tempfile
        import os
        
        # Create a temporary file to store plot data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.write(plot_data)
        temp_file.close()
        
        # Store only the file path in session (much smaller)
        session['current_plot_file'] = temp_file.name
        session['current_plot_columns'] = plot_columns
        session['current_selected_count'] = len(selected_records)
        
        # Store selected records for restoration (only minimal data)
        minimal_records = []
        for record in selected_records:
            minimal_records.append({
                'file_name': record['file_name'],
                'data_set': record['data_set']
            })
        session['last_selected_records'] = minimal_records
        
        return render_template('plot_result.html', 
                             plot_data=plot_base64,
                             selected_count=len(selected_records),
                             plot_columns=plot_columns)
        
    except Exception as e:
        flash(f'Error creating plot: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/plot_notebook_style', methods=['POST'])
def plot_notebook_style():
    """Create notebook-style overlay plots for selected records"""
    try:
        # Get selected records from form
        file_names = request.form.getlist('file_names')
        data_sets = request.form.getlist('data_sets')
        plot_columns = request.form.getlist('plot_columns')
        matrix_layout = request.form.get('matrix_layout')
        legend_style = request.form.get('legend_style', 'auto')
        custom_legend_labels = request.form.get('custom_legend_labels', '')
        # Per-variable y-axis limits (default auto)
        y_axis_limits = {}
        for col in plot_columns:
            y_min_s = request.form.get('y_min_' + col, '').strip() or None
            y_max_s = request.form.get('y_max_' + col, '').strip() or None
            y_min, y_max = None, None
            try:
                if y_min_s:
                    y_min = float(y_min_s)
                if y_max_s:
                    y_max = float(y_max_s)
            except ValueError:
                pass
            y_axis_limits[col] = (y_min, y_max)
        
        if not file_names or not data_sets:
            flash('Please select at least one record to plot', 'danger')
            return redirect(url_for('index'))
        
        # Validate custom legend labels if provided
        if legend_style == 'custom' and custom_legend_labels:
            custom_labels = [line.strip() for line in custom_legend_labels.split('\n') if line.strip()]
            if len(custom_labels) != len(file_names):
                flash(f'Custom legend labels count ({len(custom_labels)}) does not match selected records ({len(file_names)}). Please provide exactly {len(file_names)} labels.', 'warning')
                return redirect(url_for('index'))
        
        # Parse matrix layout if provided
        layout = None
        if matrix_layout:
            if matrix_layout == '1':
                layout = (1, 1)  # 1 column = single column layout
            elif matrix_layout == '2':
                layout = (1, 2)  # 2 columns = side-by-side layout
            elif 'x' in matrix_layout:
                try:
                    rows, cols = map(int, matrix_layout.split('x'))
                    layout = (rows, cols)
                except ValueError:
                    pass
        
        row_ids = request.form.getlist('row_ids')
        conn = get_db_connection()
        selected_records = resolve_selected_rows(conn, file_names, data_sets, row_ids)
        conn.close()
        
        if not selected_records:
            flash('No valid records found for plotting', 'danger')
            return redirect(url_for('index'))
        
        # Create notebook-style plot with selected columns
        plot_data = create_notebook_style_plot_with_columns(selected_records, plot_columns, layout, legend_style, custom_legend_labels, y_axis_limits=y_axis_limits)
        
        if plot_data is None:
            flash('Error creating plot - check if Excel files exist and contain required columns', 'danger')
            return redirect(url_for('index'))
        
        # Convert to base64 for display
        plot_base64 = base64.b64encode(plot_data).decode('utf-8')
        
        # Store plot data in a temporary file instead of session
        import tempfile
        import os
        
        # Create a temporary file to store plot data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.write(plot_data)
        temp_file.close()
        
        # Store only the file path in session (much smaller)
        session['current_plot_file'] = temp_file.name
        session['current_plot_columns'] = plot_columns if plot_columns else ['All variables (notebook style)']
        session['current_selected_count'] = len(selected_records)
        
        # Store selected records for restoration (only minimal data)
        minimal_records = []
        for record in selected_records:
            minimal_records.append({
                'file_name': record['file_name'],
                'data_set': record['data_set']
            })
        session['last_selected_records'] = minimal_records
        
        return render_template('plot_result.html', 
                             plot_data=plot_base64,
                             selected_count=len(selected_records),
                             plot_columns=plot_columns if plot_columns else ['All variables (notebook style)'])
        
    except Exception as e:
        flash(f'Error creating notebook style plot: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/download_plot')
def download_plot():
    """Download the current plot as PNG file"""
    try:
        plot_file = session.get('current_plot_file')
        if not plot_file or not os.path.exists(plot_file):
            flash('No plot data available for download', 'danger')
            return redirect(url_for('index'))
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"time_series_plot_{timestamp}.png"
        
        return send_file(plot_file, 
                        as_attachment=True, 
                        download_name=filename,
                        mimetype='image/png')
        
    except Exception as e:
        flash(f'Error downloading plot: {str(e)}', 'danger')
        return redirect(url_for('index'))

def load_time_series_data(file_name, start_time, end_time, data_set=None, skip_derived=False):
    """Load time series data from Excel file or database.
    data_set: optional; used only for Excel sheet selection (data_set N → sheet index N-1).
    DB lookup always uses file_name + start_time + end_time to avoid TEXT/REAL type mismatches.
    skip_derived: if True, skip the expensive calculate_derived_quantities step."""
    try:
        # Coerce to float for consistent filtering (callers may pass int or string)
        try:
            start_time = float(start_time)
            end_time = float(end_time)
        except (TypeError, ValueError):
            return None
        # Check database first — always query by file_name + start_time + end_time
        # (data_set column is TEXT in SQLite; passing a number would cause a type mismatch)
        conn = get_db_connection()
        record = conn.execute(
            'SELECT has_time_series, time_series_data FROM results WHERE file_name = ? AND start_time = ? AND end_time = ?',
            (file_name, start_time, end_time)
        ).fetchone()
        # If not found (possibly float vs stored int), also try matching by data_set as string
        if not record and data_set is not None:
            record = conn.execute(
                'SELECT has_time_series, time_series_data FROM results WHERE file_name = ? AND CAST(data_set AS TEXT) = CAST(? AS TEXT)',
                (file_name, data_set)
            ).fetchone()
        conn.close()
        
        if record and record['has_time_series'] and record['time_series_data']:
            # Load from database
            print(f"Loading time series data from database for {file_name}")
            data = json.loads(record['time_series_data'])
            df = pd.DataFrame(data)
            print(f"Loaded from database. Shape: {df.shape}")
            if skip_derived:
                return df
            df_with_derived = calculate_derived_quantities(df)
            return df_with_derived
        else:
            # Load from Excel file
            print(f"Loading time series data from Excel file for {file_name}")
            file_path = os.path.join(config['data_dir'], file_name)
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            df_filtered = None
            # If data_set provided, try that sheet first (0-indexed: data_set 1 -> sheet 0)
            if data_set is not None:
                try:
                    sheet_idx = int(float(data_set)) - 1
                    if sheet_idx >= 0:
                        df = pd.read_excel(file_path, sheet_name=sheet_idx)
                        if 'time_elapsed' in df.columns:
                            df_filtered = df[(df['time_elapsed'] >= start_time) & (df['time_elapsed'] <= end_time)].copy()
                except (ValueError, IndexError, KeyError):
                    pass
            if df_filtered is None or df_filtered.empty:
                df = pd.read_excel(file_path)
                df_filtered = df[(df['time_elapsed'] >= start_time) & (df['time_elapsed'] <= end_time)].copy()
            
            if df_filtered.empty:
                print(f"No data found for {file_name} in time range {start_time} to {end_time}")
                return None
            
            print(f"Loaded from Excel. Shape: {df_filtered.shape}")
            if skip_derived:
                return df_filtered
            df_with_derived = calculate_derived_quantities(df_filtered)
            return df_with_derived
            
    except Exception as e:
        print(f"Error loading time series data: {e}")
        return None

def generate_legend_labels(selected_records, legend_style, custom_legend_labels):
    """Generate legend labels based on the selected style"""
    try:
        if legend_style == 'none':
            return [None] * len(selected_records)
        
        if legend_style == 'custom' and custom_legend_labels:
            # Parse custom labels from textarea (one per line)
            labels = [line.strip() for line in custom_legend_labels.split('\n') if line.strip()]
            # Pad with auto-generated labels if not enough provided
            while len(labels) < len(selected_records):
                labels.append(f"Lab{len(labels) + 1}")
            return labels[:len(selected_records)]
        
        if legend_style == 'filename':
            labels = []
            for record in selected_records:
                try:
                    # Convert data_set to integer, handling both string and numeric values
                    data_set_num = int(float(record['data_set']))
                    label = f"{record['file_name'].replace('.xlsx', '')}_slice_{data_set_num}"
                    labels.append(label)
                except (ValueError, TypeError):
                    # If conversion fails, use the original data_set value
                    label = f"{record['file_name'].replace('.xlsx', '')}_slice_{record['data_set']}"
                    labels.append(label)
            return labels
        
        # Default: auto (Lab1, Lab2, Lab3...)
        return [f"Lab{i+1}" for i in range(len(selected_records))]
    except Exception as e:
        print(f"Error in generate_legend_labels: {e}")
        # Fallback to simple labels
        return [f"Series_{i+1}" for i in range(len(selected_records))]

def calculate_derived_quantities(df):
    """Calculate derived quantities exactly like in the notebook"""
    df_calc = df.copy()
    print(f"Calculating derived quantities. Available columns: {list(df_calc.columns)}")
    
    # Calculate dT_HP (Heat pump temperature difference) - exactly like notebook
    if 'T_supply' in df_calc.columns and 'T_return_emu' in df_calc.columns:
        df_calc['dT_HP'] = df_calc['T_supply'] - df_calc['T_return_emu']
        print(f"Calculated dT_HP: {df_calc['dT_HP'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for dT_HP calculation")
    add_t_mean_log_column(df_calc)
    
    # Calculate dTreturn (Return temperature difference) - exactly like notebook
    if 'T_return_emu' in df_calc.columns and 'T_return_calc' in df_calc.columns:
        df_calc['dTreturn'] = df_calc['T_return_emu'] - df_calc['T_return_calc']
        print(f"Calculated dTreturn: {df_calc['dTreturn'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for dTreturn calculation")
    
    # Calculate abs_pressure_difference - exactly like notebook
    if 'Pressure difference' in df_calc.columns:
        # Check if there is any change of sign in the 'Pressure difference' column
        if np.any(np.diff(np.sign(df_calc['Pressure difference']))):
            # Apply the custom logic if there is a change of sign
            df_calc['abs_pressure_difference'] = df_calc['Pressure difference'].apply(lambda x: abs(x) if x < 0 else -x)
            print(f"Applied custom pressure difference logic")
        else:
            # Apply the absolute value function if there is no change of sign
            df_calc['abs_pressure_difference'] = df_calc['Pressure difference'].abs()
            print(f"Applied absolute pressure difference")
        
        # Convert from Pa to bar
        df_calc['abs_pressure_difference'] = df_calc['abs_pressure_difference'] / 100000
        print(f"Calculated abs_pressure_difference: {df_calc['abs_pressure_difference'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for abs_pressure_difference calculation")
    
    # Calculate normalised_Electric Power Input (corrected) - exactly like notebook
    if 'Electric Power Input (corrected)' in df_calc.columns:
        # Use the mean as reference for normalization (like in notebook)
        power_mean = df_calc['Electric Power Input (corrected)'].mean()
        if power_mean != 0:
            df_calc['normalised_Electric Power Input (corrected)'] = df_calc['Electric Power Input (corrected)'] / power_mean
        else:
            df_calc['normalised_Electric Power Input (corrected)'] = 0
        print(f"Calculated normalised_Electric Power Input (corrected): {df_calc['normalised_Electric Power Input (corrected)'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for normalised_Electric Power Input (corrected) calculation")
    
    # Volume flow is already in m³/h, no conversion needed
    # Just copy it if it exists
    if 'volume flow' in df_calc.columns:
        df_calc['volume flow'] = df_calc['volume flow']  # Already in m³/h
        print(f"Volume flow available: {df_calc['volume flow'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    
    # Mass flow: measured wins, missing or empty is derived from volume flow × density
    df_calc, mass_flow_notices = derive_mass_flow_if_needed(df_calc)
    for note in mass_flow_notices:
        print(note)
    
    # Use existing T_return_calc, T_B_calc, T_H_calc, Q_HB, Q_BA if available, otherwise calculate
    if 'T_return_calc' not in df_calc.columns and 'T_supply' in df_calc.columns and 'Heating Capacity (corrected)' in df_calc.columns and 'mass flow' in df_calc.columns:
        df_calc['T_return_calc'] = df_calc['T_supply'] - (df_calc['Heating Capacity (corrected)'] / (df_calc['mass flow'] * 4.183))
        print(f"Calculated T_return_calc: {df_calc['T_return_calc'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    elif 'T_return_calc' in df_calc.columns:
        print(f"Using existing T_return_calc: {df_calc['T_return_calc'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for T_return_calc calculation")
    
    if 'T_B_calc' not in df_calc.columns and 'T_return_emu' in df_calc.columns and 'Q_BA' in df_calc.columns and 'mass flow' in df_calc.columns:
        df_calc['T_B_calc'] = df_calc['T_return_emu'] - (df_calc['Q_BA'] / (df_calc['mass flow'] * 4.183))
        print(f"Calculated T_B_calc: {df_calc['T_B_calc'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    elif 'T_B_calc' in df_calc.columns:
        print(f"Using existing T_B_calc: {df_calc['T_B_calc'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for T_B_calc calculation")
    
    if 'T_H_calc' not in df_calc.columns and 'T_supply' in df_calc.columns and 'Q_HB' in df_calc.columns and 'mass flow' in df_calc.columns:
        df_calc['T_H_calc'] = df_calc['T_supply'] + (df_calc['Q_HB'] / (df_calc['mass flow'] * 4.183))
        print(f"Calculated T_H_calc: {df_calc['T_H_calc'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    elif 'T_H_calc' in df_calc.columns:
        print(f"Using existing T_H_calc: {df_calc['T_H_calc'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Missing columns for T_H_calc calculation")
    
    # Add debug prints for existing variables
    if 'Q_HB' in df_calc.columns:
        print(f"Q_HB available: {df_calc['Q_HB'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Q_HB not found in data")
    
    if 'Q_BA' in df_calc.columns:
        print(f"Q_BA available: {df_calc['Q_BA'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Q_BA not found in data")
    
    if 'Heating Capacity (without corr)' in df_calc.columns:
        print(f"Heating Capacity (without corr) available: {df_calc['Heating Capacity (without corr)'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Heating Capacity (without corr) not found in data")
    
    if 'Electric Power Input (without correction)' in df_calc.columns:
        print(f"Electric Power Input (without correction) available: {df_calc['Electric Power Input (without correction)'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Electric Power Input (without correction) not found in data")
    
    if 'Electrical power input BUH' in df_calc.columns:
        print(f"Electrical power input BUH available: {df_calc['Electrical power input BUH'].iloc[0] if len(df_calc) > 0 else 'N/A'}")
    else:
        print("Electrical power input BUH not found in data")
    
    # Instantaneous COP (without corr): Heating Capacity (without corr) / Electric Power Input (without correction)
    if 'Heating Capacity (without corr)' in df_calc.columns and 'Electric Power Input (without correction)' in df_calc.columns:
        p_uncorr = df_calc['Electric Power Input (without correction)']
        df_calc['COP (without corr)'] = np.where(p_uncorr != 0, df_calc['Heating Capacity (without corr)'] / p_uncorr, np.nan)
        print(f"Calculated COP (without corr)")
    
    # Instantaneous COP (corrected): Heating Capacity (corrected) / Electric Power Input (corrected)
    if 'Heating Capacity (corrected)' in df_calc.columns and 'Electric Power Input (corrected)' in df_calc.columns:
        p_corr = df_calc['Electric Power Input (corrected)']
        df_calc['COP (corrected)'] = np.where(p_corr != 0, df_calc['Heating Capacity (corrected)'] / p_corr, np.nan)
        print(f"Calculated COP (corrected)")
    
    # Instantaneous COP (with BUH): (Heating Capacity (corrected) + Electrical power input BUH) / (Electric Power Input (corrected) + Electrical power input BUH)
    if 'Heating Capacity (corrected)' in df_calc.columns and 'Electric Power Input (corrected)' in df_calc.columns:
        q_buh = df_calc['Electrical power input BUH'].fillna(0) if 'Electrical power input BUH' in df_calc.columns else pd.Series(0, index=df_calc.index)
        denom = df_calc['Electric Power Input (corrected)'] + q_buh
        df_calc['COP (with BUH)'] = np.where(denom != 0, (df_calc['Heating Capacity (corrected)'] + q_buh) / denom, np.nan)
        print(f"Calculated COP (with BUH)")
    
    print(f"Final columns after calculation: {list(df_calc.columns)}")
    return df_calc

def create_notebook_style_plot_with_columns(selected_records, plot_columns, matrix_layout=None, legend_style='auto', custom_legend_labels='', y_axis_limits=None):
    """Create notebook-style plot with selected columns and normalized time. y_axis_limits: dict mapping column name -> (y_min, y_max), None means auto."""
    y_axis_limits = y_axis_limits or {}
    try:
        print(f"=== NOTEBOOK STYLE PLOT WITH COLUMNS CALLED ===")
        print(f"Selected records: {len(selected_records)}")
        print(f"Plot columns: {plot_columns}")
        print(f"Matrix layout: {matrix_layout}")
        print(f"Legend style: {legend_style}")
        print(f"Custom legend labels: {custom_legend_labels}")
        
        # Set up matplotlib exactly like the notebook
        plt.rcParams['text.usetex'] = False
        plt.style.use('default')
        
        # If no columns specified, use all columns from the notebook (including derived quantities)
        if not plot_columns:
            columns_to_plot = [
                'T_outdoor (DB)',
                'T_outdoor (WB)', 
                'T_supply',
                'T_return_emu',
                'T_mean_log',
                'dT_HP', # Updated name
                'dTreturn', # Updated name
                'T_return_calc',
                'T_B_calc',
                'T_H_calc',
                'Heating Capacity (corrected)',
                'Heating Capacity (without corr)',
                'Electric Power Input (corrected)',
                'Electric Power Input (without correction)',
                'normalised_Electric Power Input (corrected)', # Updated name
                'abs_pressure_difference', # Updated name
                'Q_HB',
                'Q_BA',
                'mass flow',
                'volume flow',
                'COP (without corr)',
                'COP (corrected)',
                'COP (with BUH)'
            ]
        else:
            columns_to_plot = plot_columns
        
        # Custom y labels exactly like the notebook
        custom_y_labels = {
            'T_outdoor (DB)': r'$T_{\mathrm{db}}$ (°C)',
            'T_outdoor (WB)': r'$T_{\mathrm{wb}}$ (°C)',
            'T_supply': r'$T_{\mathrm{sup}}$ (°C)',
            'T_return_emu': r'$T_{\mathrm{ret}}$ (°C)',
            'T_mean_log': r'$T_{\mathrm{mean,log}}$ (°C)',
            'dT_ln': r'$\Delta T_{\ln}$ (K)',
            'dT_HP': r'$\Delta T_{\mathrm{HP}}$ ($^\circ$C)', # Updated label
            'dTreturn': r'$\Delta T_{\mathrm{ret}}$ ($^\circ$C)', # Updated label
            'T_return_calc': r'$T_{\mathrm{ret,calc}}$ (°C)',
            'T_B_calc': r'$T_{\mathrm{B,calc}}$ (°C)',
            'T_H_calc': r'$T_{\mathrm{H,calc}}$ (°C)',
            'Heating Capacity (corrected)': 'Heating Capacity (w/ corr) (kW)',
            'Heating Capacity (without corr)': 'Heating Capacity (w/o corr) (kW)',
            'Electric Power Input (corrected)': 'Power Input (w/ corr) (kW)',
            'Electric Power Input (without correction)': 'Power Input (w/o corr) (kW)',
            'normalised_Electric Power Input (corrected)': 'Normalised Power (-)', # Updated label
            'abs_pressure_difference': r'$\Delta P_{\mathrm{HP}}$ (bar)', # Updated label
            'Q_HB': 'Q_HB (kW)',
            'Q_BA': 'Q_BA (kW)',
            'mass flow': 'Mass flow (kg/s)',
            'volume flow': r'Volume Flow (m³/h)',
            'Pressure difference': 'Pressure difference (bar)',
            'Electrical power input BUH': 'Electrical power input BUH (kW)',
            'COP (without corr)': 'COP (w/o corr) (-)',
            'COP (corrected)': 'COP (corrected) (-)',
            'COP (with BUH)': 'COP (with BUH) (-)'
        }
        
        # Load data for each selected record
        data_dict = {}
        for record in selected_records:
            file_name = record['file_name']
            start_time = record['start_time']
            end_time = record['end_time']
            data_set = record['data_set']
            
            df = load_time_series_data(file_name, start_time, end_time)
            if df is not None:
                # Data already has derived quantities calculated in load_time_series_data
                print(f"Available columns for notebook plotting: {list(df.columns)}")
                # Normalize time to start from 0 (like in the notebook)
                df_normalized = df.copy()
                df_normalized['time_elapsed'] = df_normalized['time_elapsed'] - df_normalized['time_elapsed'].iloc[0]
                
                key = f"{file_name}_dataset_{data_set}"
                data_dict[key] = df_normalized
        
        if not data_dict:
            return None
        
        # Determine layout based on matrix_layout parameter
        num_columns = len(columns_to_plot)
        if matrix_layout == (1, 1):  # 1 column layout
            rows, cols = num_columns, 1
        elif matrix_layout == (1, 2):  # 2 column layout
            rows = (num_columns + 1) // 2  # Ceiling division
            cols = 2
        else:
            # Default to 2 columns if no layout specified
            rows = (num_columns + 1) // 2
            cols = 2
        
        # Create figure with subplots - exactly like notebook
        # Calculate figure size based on number of subplots to maintain good aspect ratio
        if rows == 1 and cols == 1:
            figsize = (8, 6)
        elif rows == 1:
            figsize = (16, 6)  # Wide for horizontal layout
        elif cols == 1:
            figsize = (8, 6 * rows)  # Tall for vertical layout
        else:
            # For 2+ columns, use notebook-style sizing
            figsize = (20, 5 * rows)  # Similar to notebook but scaled by rows
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Define font properties exactly like notebook
        title_font = {'fontname': 'Arial', 'size': '20', 'weight': 'bold'}
        axis_font = {'fontname': 'Arial', 'size': '20', 'weight': 'bold'}
        tick_font = {'fontname': 'Arial', 'size': '18'}
        legend_font = {'size': '16'}
        
        # Generate legend labels once
        legend_labels = generate_legend_labels(selected_records, legend_style, custom_legend_labels)
        print(f"Generated legend labels: {legend_labels}")
        
        # Plot each column
        for i, column in enumerate(columns_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot data for each record
            column_found = False
            
            for j, (key, df) in enumerate(data_dict.items()):
                if column in df.columns:
                    # Use the generated legend label if available, otherwise use key
                    if j < len(legend_labels) and legend_labels[j] is not None:
                        label = legend_labels[j]
                    else:
                        label = key
                    ax.plot(df['time_elapsed'], df[column], label=label)
                    column_found = True
                else:
                    print(f"Column '{column}' not found in {key}")
            
            if not column_found:
                print(f"Column '{column}' not found in any dataset")
            
            # Set labels and formatting exactly like notebook
            ax.set_ylabel(custom_y_labels.get(column, column), **axis_font)
            ax.set_xlabel('Time [s]', **axis_font)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
            
            # Smart legend placement (only if not 'none')
            if legend_style != 'none':
                ax.legend(prop=legend_font, loc='best', framealpha=0.9)
            
            # Optional y-axis range (per variable, default auto)
            if y_axis_limits:
                lim = y_axis_limits.get(column, (None, None))
                y_min, y_max = lim[0], lim[1]
                if y_min is not None or y_max is not None:
                    ylo, yhi = ax.get_ylim()
                    ax.set_ylim(y_min if y_min is not None else ylo, y_max if y_max is not None else yhi)
        
        # Hide unused subplots
        for i in range(len(columns_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        # Save plot to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        plt.close()
        
        return plot_data
        
    except Exception as e:
        print(f"Error in create_notebook_style_plot_with_columns: {str(e)}")
        return None

def create_plot_with_original_time(selected_records, plot_columns, matrix_layout=None, plot_type='overlay', legend_style='auto', custom_legend_labels='', y_axis_limits=None):
    """Create plot with original timestamps. y_axis_limits: dict mapping column name -> (y_min, y_max), None means auto."""
    y_axis_limits = y_axis_limits or {}
    try:
        print(f"=== CREATE PLOT WITH ORIGINAL TIME CALLED ===")
        print(f"Selected records: {len(selected_records)}")
        print(f"Plot columns: {plot_columns}")
        print(f"Matrix layout: {matrix_layout}")
        
        # Set up matplotlib
        plt.rcParams['text.usetex'] = False
        plt.style.use('default')
        
        # If no columns specified, use a default set (including COP)
        if not plot_columns:
            columns_to_plot = [
                'T_outdoor (DB)',
                'T_outdoor (WB)', 
                'T_supply',
                'T_return_emu',
                'Heating Capacity (corrected)',
                'Electric Power Input (corrected)',
                'volume flow',
                'COP (without corr)',
                'COP (corrected)',
                'COP (with BUH)'
            ]
        else:
            columns_to_plot = plot_columns
        
        # Custom y labels - using exact column names from database
        custom_y_labels = {
            'T_outdoor (DB)': r'$T_{\mathrm{db}}$ (°C)',
            'T_outdoor (WB)': r'$T_{\mathrm{wb}}$ (°C)',
            'T_supply': r'$T_{\mathrm{sup}}$ (°C)',
            'T_return_emu': r'$T_{\mathrm{ret}}$ (°C)',
            'T_mean_log': r'$T_{\mathrm{mean,log}}$ (°C)',
            'dT_ln': r'$\Delta T_{\ln}$ (K)',
            'T_return_calc': r'$T_{\mathrm{ret,calc}}$ (°C)',
            'T_B_calc': r'$T_{\mathrm{B,calc}}$ (°C)',
            'T_H_calc': r'$T_{\mathrm{H,calc}}$ (°C)',
            'dT_HP': r'$\Delta T_{\mathrm{HP}}$ (°C)',
            'dTreturn': r'$\Delta T_{\mathrm{ret}}$ (°C)',
            'Heating Capacity (corrected)': 'Heating Capacity (w/ corr) (kW)',
            'Heating Capacity (without corr)': 'Heating Capacity (w/o corr) (kW)',
            'Electric Power Input (corrected)': 'Power Input (w/ corr) (kW)',
            'Electric Power Input (without correction)': 'Power Input (w/o corr) (kW)',
            'normalised_Electric Power Input (corrected)': 'Normalised Power (-)',
            'abs_pressure_difference': r'$\Delta P_{\mathrm{HP}}$ (bar)',
            'Q_HB': 'Q_HB (kW)',
            'Q_BA': 'Q_BA (kW)',
            'mass flow': 'Mass flow (kg/s)',
            'volume flow': r'Volume Flow (m³/h)',
            'Pressure difference': 'Pressure difference (bar)',
            'Electrical power input BUH': 'Electrical power input BUH (kW)',
            'COP (without corr)': 'COP (w/o corr) (-)',
            'COP (corrected)': 'COP (corrected) (-)',
            'COP (with BUH)': 'COP (with BUH) (-)'
        }
        
        # Load data for each selected record
        data_dict = {}
        for record in selected_records:
            file_name = record['file_name']
            start_time = record['start_time']
            end_time = record['end_time']
            data_set = record['data_set']
            
            df = load_time_series_data(file_name, start_time, end_time)
            if df is not None:
                # Data already has derived quantities calculated in load_time_series_data
                print(f"Available columns for plotting: {list(df.columns)}")
                key = f"{file_name}_dataset_{data_set}"
                data_dict[key] = df
        
        if not data_dict:
            return None
        
        # Create figure with subplots
        # Determine layout based on matrix_layout parameter or auto-calculate
        num_columns = len(columns_to_plot)
        if matrix_layout:
            if matrix_layout == (1, 1):  # 1 column layout
                rows, cols = num_columns, 1
            elif matrix_layout == (1, 2):  # 2 column layout
                rows = (num_columns + 1) // 2  # Ceiling division
                cols = 2
            else:
                rows, cols = matrix_layout
        else:
            # Auto-calculate layout
            if num_columns <= 2:
                rows, cols = 1, 2
            elif num_columns <= 4:
                rows, cols = 2, 2
            elif num_columns <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = 3, 3
        
        # Calculate figure size based on number of subplots to maintain good aspect ratio
        if rows == 1 and cols == 1:
            figsize = (8, 6)
        elif rows == 1:
            figsize = (16, 6)  # Wide for horizontal layout
        elif cols == 1:
            figsize = (8, 6 * rows)  # Tall for vertical layout
        else:
            # For 2+ columns, use notebook-style sizing
            figsize = (20, 5 * rows)  # Similar to notebook but scaled by rows
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Define font properties exactly like notebook
        title_font = {'fontname': 'Arial', 'size': '18', 'weight': 'bold'}
        axis_font = {'fontname': 'Arial', 'size': '20', 'weight': 'bold'}
        tick_font = {'fontname': 'Arial', 'size': '16'}
        legend_font = {'size': '12'}
        
        # Generate legend labels once
        legend_labels = generate_legend_labels(selected_records, legend_style, custom_legend_labels)
        print(f"Generated legend labels: {legend_labels}")
        
        # Plot each column
        for i, column in enumerate(columns_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot data for each record
            column_found = False
            
            for j, (key, df) in enumerate(data_dict.items()):
                if column in df.columns:
                    # Use the generated legend label if available, otherwise use key
                    if j < len(legend_labels) and legend_labels[j] is not None:
                        label = legend_labels[j]
                    else:
                        label = key
                    ax.plot(df['time_elapsed'], df[column], label=label)
                    column_found = True
                else:
                    print(f"Column '{column}' not found in {key}")
            
            if not column_found:
                print(f"Column '{column}' not found in any dataset")
            
            # Set labels and formatting exactly like notebook
            ax.set_ylabel(custom_y_labels.get(column, column), **axis_font)
            ax.set_xlabel('Time [s]', **axis_font)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
            
            # Smart legend placement (only if not 'none')
            if legend_style != 'none':
                ax.legend(prop=legend_font, loc='best', framealpha=0.9)
            
            # Optional y-axis range (per variable, default auto)
            if y_axis_limits:
                lim = y_axis_limits.get(column, (None, None))
                y_min, y_max = lim[0], lim[1]
                if y_min is not None or y_max is not None:
                    ylo, yhi = ax.get_ylim()
                    ax.set_ylim(y_min if y_min is not None else ylo, y_max if y_max is not None else yhi)
        
        # Hide unused subplots
        for i in range(len(columns_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        # Save plot to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        plt.close()
        
        return plot_data
        
    except Exception as e:
        print(f"Error in create_plot_with_original_time: {str(e)}")
        return None

def import_time_series(file_name, data_set, start_time, end_time):
    """Import time series data and store in database"""
    try:
        # Load data from Excel file
        file_path = os.path.join(config['data_dir'], file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
        
        df = pd.read_excel(file_path)
        df_filtered = df[(df['time_elapsed'] >= start_time) & (df['time_elapsed'] <= end_time)].copy()
        
        if df_filtered.empty:
            print(f"No data found for {file_name} in time range {start_time} to {end_time}")
            return False
        
        # Select only the essential columns for plotting (including all the ones you mentioned)
        essential_columns = [
            'time_elapsed',
            'T_outdoor (DB)',
            'T_outdoor (WB)', 
            'T_supply',
            'T_return_emu',
            'T_return_calc',
            'T_B_calc',
            'T_H_calc',
            'Q_HB',
            'Q_BA',
            'Heating Capacity (corrected)',
            'Heating Capacity (without corr)',
            'Electric Power Input (corrected)',
            'Electric Power Input (without correction)',
            'Pressure difference',
            'volume flow',
            'mass flow',
            'Electrical power input BUH'
        ]
        
        # Filter to only include columns that exist in the data
        available_columns = [col for col in essential_columns if col in df_filtered.columns]
        df_selected = df_filtered[available_columns].copy()
        
        print(f"Importing {len(available_columns)} columns: {available_columns}")
        
        # Convert to JSON for storage
        time_series_json = df_selected.to_json(orient='records')
        
        # Update database
        conn = get_db_connection()
        conn.execute(
            'UPDATE results SET has_time_series = ?, time_series_data = ? WHERE file_name = ? AND data_set = ?',
            (True, time_series_json, file_name, data_set)
        )
        conn.commit()
        conn.close()
        
        print(f"Time series data imported for {file_name}, dataset {data_set}")
        return True
        
    except Exception as e:
        print(f"Error importing time series: {e}")
        return False

@app.route('/cleanup_plot')
def cleanup_plot():
    """Clean up temporary plot file and clear selections"""
    try:
        plot_file = session.get('current_plot_file')
        if plot_file and os.path.exists(plot_file):
            os.remove(plot_file)
            session.pop('current_plot_file', None)
        
        # Clear other plot-related session data
        session.pop('current_plot_columns', None)
        session.pop('current_selected_count', None)
        
        # Clear selected records
        session.pop('last_selected_records', None)
        
        flash('Plot files cleaned up and selections cleared', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error cleaning up: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/view_timeseries/<file_name>/<data_set>')
def view_timeseries(file_name, data_set):
    """View time series details for a specific entry"""
    try:
        conn = get_db_connection()
        record = conn.execute(
            'SELECT * FROM results WHERE file_name = ? AND data_set = ?',
            (file_name, data_set)
        ).fetchone()
        conn.close()
        
        if not record:
            flash('Record not found', 'danger')
            return redirect(url_for('index'))
        
        # Parse time series data if available
        time_series_data = None
        time_series_columns = []
        if record['has_time_series'] and record['time_series_data']:
            try:
                time_series_data = json.loads(record['time_series_data'])
                if time_series_data:
                    time_series_columns = list(time_series_data[0].keys()) if time_series_data else []
            except:
                time_series_data = None
                time_series_columns = []
        
        # Check what's available in the Excel file
        excel_columns = []
        excel_file_path = os.path.join(config['data_dir'], file_name)
        if os.path.exists(excel_file_path):
            try:
                df = pd.read_excel(excel_file_path)
                excel_columns = list(df.columns)
            except:
                excel_columns = []
        
        return render_template('timeseries_details.html',
                             record=dict(record),
                             time_series_data=time_series_data,
                             time_series_columns=time_series_columns,
                             excel_columns=excel_columns,
                             excel_file_exists=os.path.exists(excel_file_path))
        
    except Exception as e:
        flash(f'Error viewing time series: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/update_timeseries/<file_name>/<data_set>')
def update_timeseries(file_name, data_set):
    """Update/re-import time series data for a specific entry"""
    try:
        conn = get_db_connection()
        record = conn.execute(
            'SELECT * FROM results WHERE file_name = ? AND data_set = ?',
            (file_name, data_set)
        ).fetchone()
        conn.close()
        
        if not record:
            flash('Record not found', 'danger')
            return redirect(url_for('index'))
        
        # Re-import time series data
        success = import_time_series(file_name, data_set, record['start_time'], record['end_time'])
        
        if success:
            flash(f'Time series data updated for {file_name}, dataset {data_set}', 'success')
        else:
            flash(f'Failed to update time series data for {file_name}', 'danger')
        
        return redirect(url_for('view_timeseries', file_name=file_name, data_set=data_set))
        
    except Exception as e:
        flash(f'Error updating time series: {str(e)}', 'danger')
        return redirect(url_for('index'))

# ==================== COLUMN MANAGEMENT ROUTES ====================

def get_column_metadata_file():
    """Get the path to the column metadata file"""
    return os.path.join(os.path.dirname(__file__), 'column_metadata.json')

def load_column_metadata():
    """Load column metadata (formulas, analysis types, etc.)"""
    import json
    metadata_file = get_column_metadata_file()
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading column metadata: {e}")
            return {}
    return {}

def get_column_visibility_file():
    """Get the path to the column visibility file"""
    return os.path.join(os.path.dirname(__file__), 'column_visibility.json')

def load_column_visibility():
    """Load column visibility settings"""
    import json
    visibility_file = get_column_visibility_file()
    if os.path.exists(visibility_file):
        try:
            with open(visibility_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading column visibility: {e}")
            return {}
    return {}

def save_column_visibility(visibility_settings):
    """Save column visibility settings"""
    import json
    visibility_file = get_column_visibility_file()
    try:
        with open(visibility_file, 'w') as f:
            json.dump(visibility_settings, f, indent=2)
    except Exception as e:
        print(f"Error saving column visibility: {e}")

def save_column_metadata(column_name, category, metadata):
    """Save column metadata (formula, analysis type, etc.)"""
    import json
    all_metadata = load_column_metadata()
    all_metadata[column_name] = {
        'category': category,
        **metadata
    }
    metadata_file = get_column_metadata_file()
    try:
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving column metadata: {e}")

def calculate_derived_columns_for_entry(cursor, rowid):
    """Automatically calculate derived columns for a specific entry based on stored formulas"""
    try:
        # Load column metadata
        column_metadata = load_column_metadata()
        
        # Get all derived column formulas
        derived_columns = {}
        for col_name, meta in column_metadata.items():
            if meta.get('category') == 'derived' and meta.get('formula'):
                derived_columns[col_name] = meta['formula']
        
        if not derived_columns:
            return  # No derived columns to calculate
        
        # Get column names from the database once (for efficiency)
        cursor.execute("PRAGMA table_info(results)")
        db_columns = [col[1] for col in cursor.fetchall()]
        
        # Create a mapping of normalized column names to actual column names
        column_name_map = {}
        for db_col in db_columns:
            normalized = db_col.lower().replace(' ', '_')
            if normalized not in column_name_map:
                column_name_map[normalized] = db_col
        
        # Define calculation order to handle dependencies (order matters within each pass).
        # gutegrad_uncorr/corr need COP*BUH and Ts_buh; those need powerbuh first.
        # gutegrad_bam_corr only needs insert columns (cop_bam_corr, avg_t_supply, avg_t_db).
        # cop_carnot_* need gutegrad_* plus dev_tsup_setpoint / dev_db_setpoint (see setpoint guard below).
        calculation_order = [
            'powerbuh',
            'Ts_buh',
            'QUncorrwBUH',
            'PUncorrwBUH',
            'COPUncorrwBUH',
            'QCorrwBUH',
            'PCorrwBUH',
            'COPCorrwBUH',
            'gutegrad_uncorr',
            'gutegrad_corr',
            'gutegrad_bam_corr',
            'pump_correction_difference',
            'pump_correction_check',
            'cop_carnot_uncorr',
            'cop_carnot_corr',
            'cop_carnot_bamm_corr',
        ]
        
        # Add any columns not in the predefined order
        for col_name in derived_columns.keys():
            if col_name not in calculation_order:
                calculation_order.append(col_name)
        
        # Perform multiple passes to handle dependencies
        max_passes = 5  # Prevent infinite loops
        calculated = set()
        last_calculated_count = 0
        
        for pass_num in range(max_passes):
            if len(calculated) == len(derived_columns):
                break  # All columns calculated
            
            # Get the entry data fresh for each pass (to get updated derived column values)
            # Note: SQLite should see uncommitted changes within the same transaction
            cursor.execute("SELECT rowid, * FROM results WHERE rowid = ?", (rowid,))
            entry = cursor.fetchone()
            
            if not entry:
                return
            
            entry_dict = dict(entry)
            
            # Create evaluation context with entry values
            eval_context = {}
            # Setpoint columns that should always be numeric
            setpoint_columns = ['dev_db_setpoint', 'dev_wb_setpoint', 'dev_tsup_setpoint']
            
            # Get test condition for calculating setpoints if needed
            test_condition = entry_dict.get('dev_test_condition')
            file_name = entry_dict.get('file_name')
            
            for key, value in entry_dict.items():
                if key != 'rowid' and key != 'time_series_data':
                    # Use the key as-is, but also add normalized versions
                    try:
                        if value is not None:
                            # For setpoint columns, always convert to float
                            if key in setpoint_columns:
                                eval_context[key] = float(value)
                                normalized = key.lower().replace(' ', '_')
                                if normalized not in eval_context:
                                    eval_context[normalized] = float(value)
                            else:
                                eval_context[key] = float(value)
                                # Also add normalized key for formula matching
                                normalized = key.lower().replace(' ', '_')
                                if normalized not in eval_context:
                                    eval_context[normalized] = float(value)
                        else:
                            # If value is None, check if it's a setpoint column that we can calculate
                            if key == 'dev_tsup_setpoint' and test_condition:
                                # Calculate tsup_setpoint on-the-fly if missing
                                calculated_value = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                eval_context[key] = calculated_value
                                normalized = key.lower().replace(' ', '_')
                                if normalized not in eval_context:
                                    eval_context[normalized] = calculated_value
                            elif key == 'dev_db_setpoint' and test_condition:
                                # Calculate db_setpoint on-the-fly if missing
                                calculated_value = unit_config.get_tdb_setpoint_for(test_condition)
                                if calculated_value is not None:
                                    eval_context[key] = float(calculated_value)
                                    normalized = key.lower().replace(' ', '_')
                                    if normalized not in eval_context:
                                        eval_context[normalized] = float(calculated_value)
                                else:
                                    eval_context[key] = None
                                    normalized = key.lower().replace(' ', '_')
                                    if normalized not in eval_context:
                                        eval_context[normalized] = None
                            elif key == 'dev_wb_setpoint' and test_condition:
                                # Calculate wb_setpoint on-the-fly if missing (db_setpoint - 1)
                                db_setpoint = unit_config.get_tdb_setpoint_for(test_condition)
                                if db_setpoint is not None:
                                    calculated_value = db_setpoint - 1
                                    eval_context[key] = float(calculated_value)
                                    normalized = key.lower().replace(' ', '_')
                                    if normalized not in eval_context:
                                        eval_context[normalized] = float(calculated_value)
                                else:
                                    eval_context[key] = None
                                    normalized = key.lower().replace(' ', '_')
                                    if normalized not in eval_context:
                                        eval_context[normalized] = None
                            else:
                                eval_context[key] = None
                                normalized = key.lower().replace(' ', '_')
                                if normalized not in eval_context:
                                    eval_context[normalized] = None
                    except (ValueError, TypeError):
                        # If conversion fails, try to handle setpoint columns specially
                        if key in setpoint_columns:
                            # For setpoint columns, try harder to convert or calculate
                            try:
                                if isinstance(value, str):
                                    # Remove any non-numeric characters except decimal point and minus
                                    cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                                    if cleaned:
                                        eval_context[key] = float(cleaned)
                                    else:
                                        # Try calculating if we have test_condition
                                        if key == 'dev_tsup_setpoint' and test_condition:
                                            eval_context[key] = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                        else:
                                            eval_context[key] = None
                                else:
                                    # Try calculating if we have test_condition
                                    if key == 'dev_tsup_setpoint' and test_condition:
                                        eval_context[key] = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                    else:
                                        eval_context[key] = None
                            except:
                                eval_context[key] = None
                        else:
                            eval_context[key] = value
                        normalized = key.lower().replace(' ', '_')
                        if normalized not in eval_context:
                            eval_context[normalized] = eval_context[key]
            
            # Add common mathematical functions to the evaluation context
            import math
            safe_builtins = {
                'abs': abs,
                'min': min,
                'max': max,
                'round': round,
                'pow': pow,
                'sum': sum,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'isinstance': isinstance,
                'type': type,
                # Math module functions
                'sqrt': math.sqrt,
                'exp': math.exp,
                'log': math.log,
                'log10': math.log10,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'ceil': math.ceil,
                'floor': math.floor,
            }
            eval_context.update(safe_builtins)
            
            # Calculate columns in order
            for column_name in calculation_order:
                if column_name in calculated or column_name not in derived_columns:
                    continue
                
                try:
                    formula = derived_columns[column_name]
                    
                    # Find the actual column name (handle case variations and spaces)
                    column_name_lower = column_name.lower().replace(' ', '_')
                    actual_column_name = column_name_map.get(column_name_lower)
                    
                    if actual_column_name is None:
                        # Try direct match as fallback
                        if column_name in db_columns:
                            actual_column_name = column_name
                        else:
                            # Column doesn't exist - try to create it
                            # Determine column type from metadata or default to REAL
                            col_type = 'REAL'
                            if column_name in column_metadata:
                                # Check if there's a type hint in metadata
                                meta = column_metadata[column_name]
                                if 'type' in meta:
                                    col_type = meta['type']
                                # Special case: if formula returns string, use TEXT
                                elif 'pump_correction_check' in column_name.lower():
                                    col_type = 'TEXT'
                            
                            try:
                                cursor.execute(f'ALTER TABLE results ADD COLUMN "{column_name}" {col_type}')
                                actual_column_name = column_name
                                # Update column name map
                                column_name_map[column_name_lower] = column_name
                                db_columns.append(column_name)
                                print(f"Created column '{column_name}' as {col_type}")
                            except Exception as create_error:
                                print(f"Warning: Could not create column '{column_name}': {create_error}")
                                calculated.add(column_name)  # Mark as calculated to avoid infinite loop
                                continue
                    
                    # Evaluate formula
                    try:
                        # Check if any required values are None before evaluating
                        # This prevents errors like "unsupported operand type(s) for +: 'NoneType' and 'float'"
                        # Extract variable names from formula (simple check for common patterns)
                        import re
                        # Check if any setpoint columns used in formula are None
                        setpoint_vars = ['dev_db_setpoint', 'dev_wb_setpoint', 'dev_tsup_setpoint']
                        missing_vars = []
                        for var in setpoint_vars:
                            if var in formula:
                                value = eval_context.get(var)
                                if value is None:
                                    missing_vars.append(var)
                        
                        if missing_vars:
                            # Log which entry and which variables are missing
                            file_name = entry_dict.get('file_name', 'unknown')
                            data_set = entry_dict.get('data_set', 'unknown')
                            print(f"Warning: Entry {rowid} ({file_name}, dataset {data_set}) has None values for: {', '.join(missing_vars)}. Formula: {formula}")
                            # Set result to None if required values are missing
                            result = None
                        else:
                            result = eval(formula, {"__builtins__": {}}, eval_context)
                        
                        # Handle division by zero and None values
                        if result is not None:
                            # For string results (like pump_correction_check)
                            if isinstance(result, str):
                                # Store as string (SQLite will handle type conversion)
                                cursor.execute(f"""
                                    UPDATE results 
                                    SET "{actual_column_name}" = ? 
                                    WHERE rowid = ?
                                """, (result, rowid))
                            else:
                                # Check for NaN or Inf
                                if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
                                    cursor.execute(f"""
                                        UPDATE results 
                                        SET "{actual_column_name}" = NULL 
                                        WHERE rowid = ?
                                    """, (rowid,))
                                else:
                                    # Convert to appropriate numeric type
                                    try:
                                        if isinstance(result, (int, float)):
                                            cursor.execute(f"""
                                                UPDATE results 
                                                SET "{actual_column_name}" = ? 
                                                WHERE rowid = ?
                                            """, (result, rowid))
                                        else:
                                            # Try to convert to float
                                            cursor.execute(f"""
                                                UPDATE results 
                                                SET "{actual_column_name}" = ? 
                                                WHERE rowid = ?
                                            """, (float(result), rowid))
                                    except (ValueError, TypeError):
                                        # If conversion fails, store as string
                                        cursor.execute(f"""
                                            UPDATE results 
                                            SET "{actual_column_name}" = ? 
                                            WHERE rowid = ?
                                        """, (str(result), rowid))
                        else:
                            cursor.execute(f"""
                                UPDATE results 
                                SET "{actual_column_name}" = NULL 
                                WHERE rowid = ?
                            """, (rowid,))
                        
                        # Update eval_context with the calculated value for dependent columns
                        eval_context[column_name] = result
                        eval_context[column_name_lower] = result
                        
                        calculated.add(column_name)
                        if pass_num == 0:  # Only print on first pass to avoid spam
                            print(f"Calculated {column_name} for entry {rowid}: {result}")
                        
                    except ZeroDivisionError:
                        print(f"Division by zero when calculating {column_name} for entry {rowid}")
                        cursor.execute(f"""
                            UPDATE results 
                            SET "{actual_column_name}" = NULL 
                            WHERE rowid = ?
                        """, (rowid,))
                        calculated.add(column_name)  # Mark as calculated to avoid infinite loop
                    except Exception as e:
                        print(f"Error evaluating formula for {column_name} (entry {rowid}): {e}")
                        import traceback
                        traceback.print_exc()
                        calculated.add(column_name)  # Mark as calculated to avoid infinite loop
                        
                except Exception as e:
                    print(f"Error calculating {column_name} for entry {rowid}: {e}")
                    import traceback
                    traceback.print_exc()
                    calculated.add(column_name)  # Mark as calculated to avoid infinite loop
            
            # Check if we made progress in this pass
            if len(calculated) == last_calculated_count:
                # No progress made, might be stuck
                remaining = set(derived_columns.keys()) - calculated
                if remaining:
                    print(f"Warning: Could not calculate some columns for entry {rowid} after {pass_num + 1} passes. Remaining: {remaining}")
                break
            last_calculated_count = len(calculated)
        
        if len(calculated) < len(derived_columns):
            remaining = set(derived_columns.keys()) - calculated
            print(f"Warning: Not all derived columns calculated for entry {rowid}. Calculated: {len(calculated)}/{len(derived_columns)}. Remaining: {remaining}")
                
    except Exception as e:
        print(f"Error in calculate_derived_columns_for_entry: {e}")
        import traceback
        traceback.print_exc()

def recalculate_all_derived_columns():
    """Recalculate all derived columns for all entries in the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all rowids
        cursor.execute("SELECT rowid FROM results")
        rowids = cursor.fetchall()
        
        count = 0
        errors = []
        for (rowid,) in rowids:
            try:
                calculate_derived_columns_for_entry(cursor, rowid)
                count += 1
            except Exception as e:
                errors.append(f"Entry {rowid}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        return count, errors
    except Exception as e:
        print(f"Error in recalculate_all_derived_columns: {e}")
        import traceback
        traceback.print_exc()
        return 0, [str(e)]

@app.route('/recalculate_all_derived', methods=['POST'])
def recalculate_all_derived():
    """Route to recalculate all derived columns for all entries"""
    try:
        count, errors = recalculate_all_derived_columns()
        if errors:
            flash(f'Recalculated {count} entries. {len(errors)} errors occurred.', 'warning')
            if len(errors) <= 5:
                for error in errors:
                    flash(error, 'warning')
        else:
            flash(f'Successfully recalculated derived columns for {count} entries!', 'success')
    except Exception as e:
        flash(f'Error recalculating derived columns: {str(e)}', 'danger')
    
    return redirect(url_for('index'))


@app.route('/manage_columns')
def manage_columns():
    """Display column management interface"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all columns in the results table
        cursor.execute("PRAGMA table_info(results)")
        columns = cursor.fetchall()
        
        # Get existing custom columns (exclude system columns and deviation columns)
        system_columns = {'rowid', 'file_name', 'data_set', 'start_time', 'end_time', 
                         'has_time_series', 'time_series_data', 'display_order'}
        # Exclude deviation columns (dev_*) - these are only shown on deviations page
        all_column_names = [col[1] for col in columns]
        deviation_columns = {col for col in all_column_names if col.startswith('dev_')}
        
        # Load column metadata to identify user-created columns
        column_metadata = load_column_metadata()
        user_created_columns = set(column_metadata.keys())
        
        # User-created derived columns (formulas). Built-in calculated fields such as
        # avg_t_mean_log live in metadata for documentation but are not custom.
        custom_columns = [col for col in columns 
                         if col[1] in user_created_columns 
                         and column_metadata.get(col[1], {}).get('category') == 'derived'
                         and col[1] not in system_columns 
                         and col[1] not in deviation_columns]
        
        # Get TEXT columns and REAL/INTEGER columns separately for dropdowns
        # Include setpoint columns (dev_db_setpoint, dev_wb_setpoint, dev_tsup_setpoint) in numeric_columns
        # as they are REAL type and should be usable in formulas
        setpoint_columns = {'dev_db_setpoint', 'dev_wb_setpoint', 'dev_tsup_setpoint'}
        text_columns = [col[1] for col in columns if col[2] == 'TEXT' and col[1] not in system_columns and col[1] not in deviation_columns]
        numeric_columns = [col[1] for col in columns if col[2] in ('REAL', 'INTEGER') and col[1] not in system_columns and (col[1] not in deviation_columns or col[1] in setpoint_columns)]
        
        # Get all columns for ordering (excluding only system columns that shouldn't be displayed)
        # Include deviation columns here so they can be toggled visible/hidden
        displayable_columns = [col for col in columns if col[1] not in {'rowid', 'has_time_series', 'time_series_data', 'display_order'}]
        
        # Load column visibility settings
        column_visibility = load_column_visibility()
        
        # Get all results for bulk text input
        results = conn.execute('SELECT *, rowid FROM results ORDER BY display_order ASC, rowid ASC').fetchall()
        results = [dict(row) for row in results]
        
        conn.close()
        
        return render_template('manage_columns.html', 
                            custom_columns=custom_columns,
                            all_column_names=all_column_names,
                            text_columns=text_columns,
                            numeric_columns=numeric_columns,
                            displayable_columns=displayable_columns,
                            column_metadata=column_metadata,
                            column_visibility=column_visibility,
                            deviation_columns=list(deviation_columns),
                            monitoring_mean_columns=list(dataset_manager.OPTIONAL_MONITORING_MEAN_COLUMNS),
                            default_hidden_columns=sorted(MEANS_HIDDEN_EXPLICIT),
                            results=results)
    except Exception as e:
        flash(f'Error loading column management: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/api/columns')
def api_columns():
    """API endpoint to get column information"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(results)")
        columns = cursor.fetchall()
        
        system_columns = {'rowid', 'file_name', 'data_set', 'start_time', 'end_time', 
                         'has_time_series', 'time_series_data', 'display_order'}
        
        text_columns = [col[1] for col in columns if col[2] == 'TEXT' and col[1] not in system_columns]
        numeric_columns = [col[1] for col in columns if col[2] in ('REAL', 'INTEGER') and col[1] not in system_columns]
        all_columns = [{'name': col[1], 'type': col[2]} for col in columns if col[1] not in system_columns]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'text_columns': text_columns,
            'numeric_columns': numeric_columns,
            'all_columns': all_columns
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/export_entries_format')
def api_export_entries_format():
    """Export database entries in the format needed for text column input"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all entries ordered by display_order
        cursor.execute("""
            SELECT file_name, data_set, start_time, end_time 
            FROM results 
            WHERE start_time IS NOT NULL AND end_time IS NOT NULL
            ORDER BY display_order ASC, rowid ASC
        """)
        entries = cursor.fetchall()
        
        if not entries:
            conn.close()
            return jsonify({'success': False, 'message': 'No entries found in database'})
        
        # Format as CSV
        lines = ['File Name, Data Set, Start Time, End Time, Value']
        for entry in entries:
            file_name = entry['file_name']
            data_set = entry['data_set']
            start_time = int(entry['start_time']) if entry['start_time'] is not None else ''
            end_time = int(entry['end_time']) if entry['end_time'] is not None else ''
            lines.append(f"{file_name}, {data_set}, {start_time}, {end_time}, ")
        
        export_data = '\n'.join(lines)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'export_data': export_data,
            'entry_count': len(entries)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/add_column', methods=['POST'])
def add_column():
    """Add a new column to the database"""
    try:
        data = request.get_json()
        column_name = data.get('column_name', '').strip()
        column_type = data.get('column_type', 'TEXT')  # TEXT, REAL, INTEGER
        column_category = data.get('column_category')  # 'text', 'derived', 'timeseries'
        
        if not column_name:
            return jsonify({'success': False, 'message': 'Column name is required'})
        
        # Validate column name (SQLite identifier rules)
        if not column_name.replace('_', '').isalnum():
            return jsonify({'success': False, 'message': 'Column name can only contain letters, numbers, and underscores'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(results)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        if column_name in existing_columns:
            conn.close()
            return jsonify({'success': False, 'message': f'Column "{column_name}" already exists'})
        
        # Add column to database
        try:
            cursor.execute(f"ALTER TABLE results ADD COLUMN {column_name} {column_type}")
            conn.commit()
            print(f"Added column {column_name} of type {column_type}")
        except sqlite3.OperationalError as e:
            conn.close()
            return jsonify({'success': False, 'message': f'Error adding column: {str(e)}'})
        
        # Store column metadata (category, formula if applicable)
        # We'll use a simple JSON file or add a metadata table later if needed
        # For now, just return success
        
        conn.close()
        return jsonify({'success': True, 'message': f'Column "{column_name}" added successfully'})
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        print(f"Error adding column: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/update_text_column', methods=['POST'])
def update_text_column():
    """Update text column values from pasted data"""
    try:
        data = request.get_json()
        column_name = data.get('column_name')
        pasted_data = data.get('pasted_data', '')  # Tab or comma separated
        
        if not column_name or not pasted_data:
            return jsonify({'success': False, 'message': 'Column name and data are required'})
        
        # Parse pasted data (expecting: File Name, Data Set, Start Time, End Time, Value)
        lines = [line.strip() for line in pasted_data.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:
            return jsonify({'success': False, 'message': 'Please provide at least a header row and one data row'})
        
        # Skip header row if it looks like headers
        header = lines[0].lower()
        if 'file' in header and 'name' in header:
            lines = lines[1:]
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(results)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        if column_name not in existing_columns:
            conn.close()
            return jsonify({'success': False, 'message': f'Column "{column_name}" does not exist'})
        
        updated_count = 0
        errors = []
        
        for line_num, line in enumerate(lines, start=1):
            # Try tab-separated first, then comma-separated
            if '\t' in line:
                parts = [p.strip() for p in line.split('\t')]
            else:
                parts = [p.strip() for p in line.split(',')]
            
            if len(parts) < 5:
                errors.append(f"Line {line_num}: Expected 5 columns (File Name, Data Set, Start Time, End Time, Value), got {len(parts)}")
                continue
            
            file_name = parts[0]
            try:
                data_set = int(float(parts[1]))  # Handle "1.0" -> 1
            except (ValueError, TypeError):
                errors.append(f"Line {line_num}: Invalid data_set value: {parts[1]}")
                continue
            
            try:
                start_time = int(float(parts[2]))
            except (ValueError, TypeError):
                errors.append(f"Line {line_num}: Invalid start_time value: {parts[2]}")
                continue
            
            try:
                end_time = int(float(parts[3]))
            except (ValueError, TypeError):
                errors.append(f"Line {line_num}: Invalid end_time value: {parts[3]}")
                continue
            
            value = parts[4]  # Text value
            
            # Update the database - try multiple matching strategies
            try:
                # Strategy 1: Exact match
                cursor.execute(f"""
                    UPDATE results 
                    SET {column_name} = ? 
                    WHERE file_name = ? AND data_set = ? AND start_time = ? AND end_time = ?
                """, (value, file_name, data_set, start_time, end_time))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    continue
                
                # Strategy 2: Try with string conversion for data_set (in case it's stored as text)
                cursor.execute(f"""
                    UPDATE results 
                    SET {column_name} = ? 
                    WHERE file_name = ? AND CAST(data_set AS TEXT) = ? AND start_time = ? AND end_time = ?
                """, (value, file_name, str(data_set), start_time, end_time))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    continue
                
                # Strategy 3: Try with integer conversion for data_set (in case it's stored as number but we're comparing as string)
                cursor.execute(f"""
                    UPDATE results 
                    SET {column_name} = ? 
                    WHERE file_name = ? AND CAST(data_set AS INTEGER) = ? AND start_time = ? AND end_time = ?
                """, (value, file_name, int(data_set), start_time, end_time))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    continue
                
                # Strategy 4: Try with trimmed file_name (in case of whitespace)
                cursor.execute(f"""
                    UPDATE results 
                    SET {column_name} = ? 
                    WHERE TRIM(file_name) = ? AND CAST(data_set AS INTEGER) = ? AND start_time = ? AND end_time = ?
                """, (value, file_name.strip(), int(data_set), start_time, end_time))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    continue
                
                # Strategy 5: Try with flexible time matching (within 1 second tolerance)
                cursor.execute(f"""
                    UPDATE results 
                    SET {column_name} = ? 
                    WHERE file_name = ? AND CAST(data_set AS INTEGER) = ? 
                    AND ABS(start_time - ?) <= 1 AND ABS(end_time - ?) <= 1
                """, (value, file_name, int(data_set), start_time, end_time))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    continue
                
                # If all strategies fail, try to find similar entries for debugging
                cursor.execute("""
                    SELECT file_name, data_set, start_time, end_time 
                    FROM results 
                    WHERE file_name LIKE ? OR TRIM(file_name) = ?
                    LIMIT 5
                """, (f'%{file_name}%', file_name.strip()))
                similar_entries = cursor.fetchall()
                
                if similar_entries:
                    similar_str = ', '.join([f"{e['file_name']}, ds={e['data_set']}, t={e['start_time']}-{e['end_time']}" 
                                            for e in similar_entries])
                    errors.append(f"Line {line_num}: No exact match for {file_name}, dataset {data_set}, time {start_time}-{end_time}. Similar entries: {similar_str}")
                else:
                    errors.append(f"Line {line_num}: No matching entry found for {file_name}, dataset {data_set}, time {start_time}-{end_time}")
                    
            except Exception as e:
                errors.append(f"Line {line_num}: Error updating: {str(e)}")
                import traceback
                traceback.print_exc()
        
        conn.commit()
        conn.close()
        
        message = f'Updated {updated_count} entries'
        if errors:
            message += f'. {len(errors)} errors occurred.'
        
        return jsonify({
            'success': True, 
            'message': message,
            'updated_count': updated_count,
            'errors': errors[:10]  # Limit errors shown
        })
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        print(f"Error updating text column: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/api/new_entries_preview', methods=['GET'])
def api_new_entries_preview():
    """Return inferred/default metadata for the most recently inserted rowids."""
    try:
        rowids = session.get('last_inserted_rowids', [])
        if not rowids:
            return jsonify({'success': True, 'entries': []})

        # Ensure expected columns exist (safe no-op if already there)
        ensure_results_columns({
            'hp_id': 'INTEGER',
            'lab_id': 'INTEGER',
            'test_label': 'TEXT',
            'test_cond': 'TEXT',
            'climate': 'TEXT',
            'application': 'TEXT',
            'flow_config': 'TEXT',
            'cop_dataset': 'TEXT',
            'scatter_dataset': 'TEXT',
            'indicator': 'TEXT',
            'notes': 'TEXT',
            'dpfix': 'REAL',
            'dev_test_condition': 'TEXT',
        })

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            placeholders = ",".join(["?"] * len(rowids))
            cur.execute(
                f"SELECT rowid, * FROM results WHERE rowid IN ({placeholders}) ORDER BY rowid ASC",
                [int(r) for r in rowids],
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        entries = []
        for r in rows:
            d = dict(r)
            inferred = infer_metadata_from_filename(d.get('file_name'))

            # Prefer already-saved DB values; otherwise use inferred/defaults
            merged = {}
            for key, default_val in inferred.items():
                db_val = d.get(key)
                merged[key] = db_val if db_val is not None and db_val != '' else default_val

            stored_pid = d.get('profile_id')
            profile, reason = unit_config.resolve_profile(
                file_name=d.get('file_name'),
                hp_id=merged.get('hp_id') or d.get('hp_id') or d.get('HP_ID'),
                profile_id=stored_pid if stored_pid not in (None, '') else None,
            )
            merged['profile_id'] = (profile or {}).get('profile_id') or stored_pid or None
            merged['profile_resolve_reason'] = reason

            entries.append({
                'rowid': int(d['rowid']),
                'file_name': d.get('file_name'),
                'data_set': d.get('data_set'),
                'start_time': d.get('start_time'),
                'end_time': d.get('end_time'),
                'values': merged,
            })

        return jsonify({
            'success': True,
            'entries': entries,
            'profiles': _profile_picker_items(),
        })
    except Exception as e:
        print(f"Error in api_new_entries_preview: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/apply_new_entries_metadata', methods=['POST'])
def api_apply_new_entries_metadata():
    """Apply user-confirmed metadata for rowids, then compute derived columns."""
    try:
        payload = request.get_json() or {}
        entries = payload.get('entries') or []
        if not entries:
            return jsonify({'success': False, 'message': 'No entries provided'})

        ensure_results_columns({
            'hp_id': 'INTEGER',
            'lab_id': 'INTEGER',
            'test_label': 'TEXT',
            'test_cond': 'TEXT',
            'climate': 'TEXT',
            'application': 'TEXT',
            'flow_config': 'TEXT',
            'cop_dataset': 'TEXT',
            'scatter_dataset': 'TEXT',
            'indicator': 'TEXT',
            'notes': 'TEXT',
            'dpfix': 'REAL',
            'dev_test_condition': 'TEXT',
            'profile_id': 'TEXT',
            'condition_set_id': 'TEXT',
        })

        updatable = {
            'hp_id', 'lab_id', 'test_label', 'test_cond', 'climate', 'application', 'flow_config',
            'cop_dataset', 'scatter_dataset', 'indicator', 'notes', 'dpfix',
            'dev_test_condition', 'profile_id', 'condition_set_id',
        }

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            updated = 0
            derived = 0
            dev_calculated = 0
            for entry in entries:
                rowid = int(entry.get('rowid'))
                values = entry.get('values') or {}

                # If user set test_cond, keep dev_test_condition in sync unless explicitly provided
                if 'test_cond' in values and 'dev_test_condition' not in values:
                    values['dev_test_condition'] = values.get('test_cond')
                # Review modal shows COP dataset only; keep scatter in lockstep unless sent separately
                if 'cop_dataset' in values and 'scatter_dataset' not in values:
                    values['scatter_dataset'] = values.get('cop_dataset')
                if values.get('profile_id') in ('', None):
                    values['profile_id'] = None
                else:
                    if not values.get('condition_set_id'):
                        values['condition_set_id'] = unit_config.get_default_condition_set_id()
                    if values.get('hp_id') is None:
                        derived = unit_config.hp_id_from_profile(values['profile_id'])
                        if derived is not None:
                            values['hp_id'] = derived

                set_parts = []
                set_vals = []
                for k, v in values.items():
                    if k not in updatable:
                        continue
                    set_parts.append(f"\"{k}\" = ?")
                    set_vals.append(v)
                if set_parts:
                    set_vals.append(rowid)
                    cur.execute(f"UPDATE results SET {', '.join(set_parts)} WHERE rowid = ?", set_vals)
                    updated += 1

                try:
                    calculate_derived_columns_for_entry(cur, rowid)
                    derived += 1
                except Exception as calc_err:
                    print(f"Derived calc failed for rowid={rowid}: {calc_err}")

                # Also calculate permissible deviation statistics right away (fills dev_* columns)
                try:
                    init_deviation_columns()
                    cur.execute("SELECT * FROM results WHERE rowid = ?", (rowid,))
                    row = cur.fetchone()
                    if row:
                        stats = calculate_entry_deviations(
                            row["file_name"],
                            row["data_set"],
                            row["start_time"],
                            row["end_time"],
                            warning_list=None,
                            hp_id=row["hp_id"] if "hp_id" in row.keys() else None,
                            entry=dict(row),
                        )
                        if stats:
                            from datetime import datetime as _dt
                            cur.execute('''
                                UPDATE results SET
                                    dev_test_condition = ?,
                                    dev_db_setpoint = ?,
                                    dev_wb_setpoint = ?,
                                    dev_tsup_setpoint = ?,
                                    dev_total_points = ?,
                                    dev_db_outside_count = ?,
                                    dev_wb_outside_count = ?,
                                    dev_dtreturn_outside_count = ?,
                                    dev_db_percentage = ?,
                                    dev_wb_percentage = ?,
                                    dev_dtreturn_percentage = ?,
                                    dev_max_db_deviation = ?,
                                    dev_max_wb_deviation = ?,
                                    dev_max_dtreturn_deviation = ?,
                                    dev_tsup_outside_count = ?,
                                    dev_tsup_percentage = ?,
                                    dev_max_tsup_deviation = ?,
                                    dev_max_db_pos = ?,
                                    dev_max_db_neg = ?,
                                    dev_max_wb_pos = ?,
                                    dev_max_wb_neg = ?,
                                    dev_max_tsup_pos = ?,
                                    dev_max_tsup_neg = ?,
                                    dev_max_dtreturn_pos = ?,
                                    dev_max_dtreturn_neg = ?,
                                    dev_max_flow_pos_pct = ?,
                                    dev_max_flow_neg_pct = ?,
                                    dev_avg_timestep = ?,
                                    dev_wb_calculated_from_rh = ?,
                                    dev_flow_setpoint = ?,
                                    dev_flow_set_type = ?,
                                    dev_mean_flow_dev_pct = ?,
                                    dev_flow_outside_count = ?,
                                    dev_flow_percentage = ?,
                                    dev_last_calculated = ?
                                WHERE rowid = ?
                            ''', (
                                stats.get('test_condition'),
                                float(stats['db_setpoint']) if stats.get('db_setpoint') is not None else None,
                                float(stats['wb_setpoint']) if stats.get('wb_setpoint') is not None else None,
                                float(stats['tsup_setpoint']) if stats.get('tsup_setpoint') is not None else None,
                                stats.get('total_points'),
                                stats.get('db_outside_band_count'),
                                stats.get('wb_outside_band_count'),
                                stats.get('dtreturn_outside_band_count'),
                                float(stats['db_percentage']) if stats.get('db_percentage') is not None else None,
                                float(stats['wb_percentage']) if stats.get('wb_percentage') is not None else None,
                                float(stats.get('dtreturn_percentage')) if stats.get('dtreturn_percentage') is not None else None,
                                float(stats['max_db_deviation']) if stats.get('max_db_deviation') is not None else None,
                                float(stats['max_wb_deviation']) if stats.get('max_wb_deviation') is not None else None,
                                float(stats.get('max_dtreturn_deviation')) if stats.get('max_dtreturn_deviation') is not None else None,
                                stats.get('tsup_outside_band_count'),
                                float(stats.get('tsup_percentage')) if stats.get('tsup_percentage') is not None else None,
                                float(stats.get('max_tsup_deviation')) if stats.get('max_tsup_deviation') is not None else None,
                                *_signed_extreme_params(stats),
                                float(stats['avg_timestep_size']) if stats.get('avg_timestep_size') is not None else None,
                                1 if stats.get('wb_calculated_from_rh', False) else 0,
                                float(stats['flow_setpoint']) if stats.get('flow_setpoint') is not None else None,
                                stats.get('flow_set_type'),
                                float(stats['mean_flow_dev_pct']) if stats.get('mean_flow_dev_pct') is not None else None,
                                stats.get('flow_outside_count'),
                                float(stats.get('flow_percentage')) if stats.get('flow_percentage') is not None else None,
                                _dt.now().isoformat(),
                                rowid,
                            ))
                            dev_calculated += 1
                except Exception as dev_err:
                    print(f"Deviation calc failed for rowid={rowid}: {dev_err}")

            conn.commit()
        finally:
            conn.close()

        session['last_inserted_rowids'] = []
        return jsonify({'success': True, 'message': f'Updated {updated} entries, recalculated derived columns for {derived}, and calculated deviation stats for {dev_calculated}.', 'updated': updated, 'derived': derived, 'dev_calculated': dev_calculated})
    except Exception as e:
        print(f"Error in api_apply_new_entries_metadata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/bulk_update_text_columns', methods=['POST'])
def bulk_update_text_columns():
    """Bulk update multiple text columns for multiple entries at once"""
    try:
        data = request.get_json()
        updates = data.get('updates', [])
        
        if not updates:
            return jsonify({'success': False, 'message': 'No updates provided'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Group updates by column for efficiency
        updates_by_column = {}
        for update in updates:
            column = update.get('column')
            if column not in updates_by_column:
                updates_by_column[column] = []
            updates_by_column[column].append(update)
        
        updated_count = 0
        errors = []
        
        # Update each column
        for column_name, column_updates in updates_by_column.items():
            # Verify column exists and is TEXT type
            cursor.execute("PRAGMA table_info(results)")
            columns_info = cursor.fetchall()
            column_exists = False
            for col_info in columns_info:
                if col_info[1] == column_name and col_info[2] == 'TEXT':
                    column_exists = True
                    break
            
            if not column_exists:
                errors.append(f"Column '{column_name}' does not exist or is not TEXT type")
                continue
            
            # Update each entry for this column
            for update in column_updates:
                rowid = update.get('rowid')
                value = update.get('value', '')
                file_name = update.get('file_name', '')
                data_set = update.get('data_set', '')
                
                try:
                    if rowid:
                        # Use rowid for direct update (most reliable)
                        cursor.execute(f"""
                            UPDATE results 
                            SET "{column_name}" = ? 
                            WHERE rowid = ?
                        """, (value, rowid))
                        
                        if cursor.rowcount > 0:
                            updated_count += 1
                        else:
                            errors.append(f"Row {rowid}: No entry found")
                    elif file_name and data_set:
                        # Fallback to file_name and data_set
                        cursor.execute(f"""
                            UPDATE results 
                            SET "{column_name}" = ? 
                            WHERE file_name = ? AND data_set = ?
                        """, (value, file_name, data_set))
                        
                        if cursor.rowcount > 0:
                            updated_count += 1
                        else:
                            errors.append(f"{file_name}, dataset {data_set}: No entry found")
                    else:
                        errors.append(f"Missing rowid or file_name/data_set for update")
                        
                except Exception as e:
                    errors.append(f"Error updating {column_name} for row {rowid}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        message = f'Successfully updated {updated_count} values'
        if errors:
            message += f'. {len(errors)} errors occurred.'
        
        return jsonify({
            'success': True,
            'message': message,
            'updated_count': updated_count,
            'errors': errors[:20]  # Limit errors shown
        })
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        print(f"Error in bulk_update_text_columns: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/calculate_derived_column', methods=['POST'])
def calculate_derived_column():
    """Calculate derived column values from existing columns"""
    try:
        data = request.get_json()
        column_name = data.get('column_name')
        formula = data.get('formula', '').strip()  # Python expression using column names
        
        if not column_name or not formula:
            return jsonify({'success': False, 'message': 'Column name and formula are required'})
        
        # Store the formula in metadata
        save_column_metadata(column_name, 'derived', {'formula': formula})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(results)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        if column_name not in existing_columns:
            conn.close()
            return jsonify({'success': False, 'message': f'Column "{column_name}" does not exist'})
        
        # Get all entries
        cursor.execute("SELECT rowid, * FROM results")
        entries = cursor.fetchall()
        
        updated_count = 0
        errors = []
        
        for entry in entries:
            entry_dict = dict(entry)
            
            # Evaluate formula with entry data
            try:
                # Create a safe evaluation context with only entry values
                eval_context = {}
                # Setpoint columns that should always be numeric
                setpoint_columns = ['dev_db_setpoint', 'dev_wb_setpoint', 'dev_tsup_setpoint']
                # Get test condition for calculating setpoints if needed
                test_condition = entry_dict.get('dev_test_condition')
                file_name = entry_dict.get('file_name')
                
                for key, value in entry_dict.items():
                    if key != 'rowid' and key != 'time_series_data':
                        # Convert to float if possible, otherwise keep as string
                        try:
                            if value is not None:
                                # For setpoint columns, always convert to float
                                if key in setpoint_columns:
                                    eval_context[key] = float(value)
                                else:
                                    eval_context[key] = float(value)
                            else:
                                # If value is None, check if it's a setpoint column that we can calculate
                                if key == 'dev_tsup_setpoint' and test_condition:
                                    calculated_value = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                    eval_context[key] = calculated_value
                                elif key == 'dev_db_setpoint' and test_condition:
                                    calculated_value = unit_config.get_tdb_setpoint_for(test_condition)
                                    if calculated_value is not None:
                                        eval_context[key] = float(calculated_value)
                                    else:
                                        eval_context[key] = None
                                elif key == 'dev_wb_setpoint' and test_condition:
                                    db_setpoint = unit_config.get_tdb_setpoint_for(test_condition)
                                    if db_setpoint is not None:
                                        calculated_value = db_setpoint - 1
                                        eval_context[key] = float(calculated_value)
                                    else:
                                        eval_context[key] = None
                                else:
                                    eval_context[key] = None
                        except (ValueError, TypeError):
                            # If conversion fails, try harder for setpoint columns
                            if key in setpoint_columns:
                                try:
                                    if isinstance(value, str):
                                        # Remove any non-numeric characters except decimal point and minus
                                        cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                                        if cleaned:
                                            eval_context[key] = float(cleaned)
                                        else:
                                            # Try calculating if we have test_condition
                                            if key == 'dev_tsup_setpoint' and test_condition:
                                                eval_context[key] = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                            elif key == 'dev_db_setpoint' and test_condition:
                                                calculated_value = unit_config.get_tdb_setpoint_for(test_condition)
                                                if calculated_value is not None:
                                                    eval_context[key] = float(calculated_value)
                                                else:
                                                    eval_context[key] = None
                                            elif key == 'dev_wb_setpoint' and test_condition:
                                                db_setpoint = unit_config.get_tdb_setpoint_for(test_condition)
                                                if db_setpoint is not None:
                                                    calculated_value = db_setpoint - 1
                                                    eval_context[key] = float(calculated_value)
                                                else:
                                                    eval_context[key] = None
                                            else:
                                                eval_context[key] = None
                                    else:
                                        # Try calculating if we have test_condition
                                        if key == 'dev_tsup_setpoint' and test_condition:
                                            eval_context[key] = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                        elif key == 'dev_db_setpoint' and test_condition:
                                            calculated_value = unit_config.get_tdb_setpoint_for(test_condition)
                                            if calculated_value is not None:
                                                eval_context[key] = float(calculated_value)
                                            else:
                                                eval_context[key] = None
                                        elif key == 'dev_wb_setpoint' and test_condition:
                                            db_setpoint = unit_config.get_tdb_setpoint_for(test_condition)
                                            if db_setpoint is not None:
                                                calculated_value = db_setpoint - 1
                                                eval_context[key] = float(calculated_value)
                                            else:
                                                eval_context[key] = None
                                        else:
                                            eval_context[key] = None
                                except Exception as calc_error:
                                    eval_context[key] = None
                            else:
                                eval_context[key] = value
                
                # Add common mathematical functions to the evaluation context
                import math
                safe_builtins = {
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'round': round,
                    'pow': pow,
                    'sum': sum,
                    'len': len,
                    'int': int,
                    'float': float,
                    'str': str,
                    'isinstance': isinstance,
                    'type': type,
                    # Math module functions
                    'sqrt': math.sqrt,
                    'exp': math.exp,
                    'log': math.log,
                    'log10': math.log10,
                    'sin': math.sin,
                    'cos': math.cos,
                    'tan': math.tan,
                    'asin': math.asin,
                    'acos': math.acos,
                    'atan': math.atan,
                    'ceil': math.ceil,
                    'floor': math.floor,
                }
                eval_context.update(safe_builtins)
                
                # Check if any required values are None before evaluating
                # This prevents errors like "unsupported operand type(s) for +: 'NoneType' and 'float'"
                # But only fail if we can't calculate them (no test_condition), not if we calculated them on-the-fly
                import re
                setpoint_vars = ['dev_db_setpoint', 'dev_wb_setpoint', 'dev_tsup_setpoint']
                missing_vars = []
                for var in setpoint_vars:
                    if var in formula:
                        value = eval_context.get(var)
                        if value is None:
                            # Only consider it missing if we can't calculate it (no test_condition)
                            # If we have test_condition, we should have been able to calculate it
                            if not test_condition:
                                missing_vars.append(var)
                            else:
                                # We have test_condition but still got None - this shouldn't happen, but log it
                                file_name_val = entry_dict.get('file_name', 'unknown')
                                data_set_val = entry_dict.get('data_set', 'unknown')
                                rowid_val = entry_dict.get('rowid', 'unknown')
                                print(f"Warning: Entry rowid={rowid_val} ({file_name_val}, dataset {data_set_val}) has test_condition '{test_condition}' but {var} could not be calculated. Formula: {formula}")
                                missing_vars.append(var)
                
                if missing_vars:
                    # Log which entry and which variables are missing
                    file_name_val = entry_dict.get('file_name', 'unknown')
                    data_set_val = entry_dict.get('data_set', 'unknown')
                    rowid_val = entry_dict.get('rowid', 'unknown')
                    print(f"Warning: Entry rowid={rowid_val} ({file_name_val}, dataset {data_set_val}) has None values for: {', '.join(missing_vars)}. Formula: {formula}")
                    # Set result to None if required values are missing
                    result = None
                else:
                    # Evaluate formula (using eval with restricted context)
                    # WARNING: This uses eval() which can be dangerous. In production, use a safer expression evaluator
                    result = eval(formula, {"__builtins__": {}}, eval_context)
                
                # Update database
                if result is not None:
                    cursor.execute(f"""
                        UPDATE results 
                        SET {column_name} = ? 
                        WHERE rowid = ?
                    """, (result, entry_dict['rowid']))
                    updated_count += 1
                else:
                    # Set to NULL if result is None
                    cursor.execute(f"""
                        UPDATE results 
                        SET {column_name} = NULL 
                        WHERE rowid = ?
                    """, (entry_dict['rowid'],))
                    
            except Exception as e:
                errors.append(f"Entry {entry_dict.get('file_name', 'unknown')}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        message = f'Calculated {updated_count} values'
        if errors:
            message += f'. {len(errors)} errors occurred.'
        
        return jsonify({
            'success': True,
            'message': message,
            'updated_count': updated_count,
            'errors': errors[:10]
        })
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        print(f"Error calculating derived column: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/recalculate_column', methods=['POST'])
def recalculate_single_column():
    """Recalculate a single column - for derived columns, use stored formula; for others, recalculate from source"""
    import time
    import sys
    import traceback
    
    # Force output to be flushed immediately so we can see it in real-time
    print(f"\n{'='*60}", flush=True)
    print(f"=== RECALCULATE COLUMN ROUTE CALLED ===", flush=True)
    print(f"Route handler: recalculate_single_column", flush=True)
    print(f"Request method: {request.method}", flush=True)
    print(f"Request path: {request.path}", flush=True)
    print(f"{'='*60}", flush=True)
    sys.stdout.flush()
    
    start_time = time.time()
    conn = None
    
    try:
        print("About to parse JSON data...", flush=True)
        sys.stdout.flush()
        data = request.get_json()
        if not data:
            print("ERROR: No JSON data received", flush=True)
            return jsonify({'success': False, 'message': 'No data received'})
        
        column_name = data.get('column')
        entry_ids = data.get('entry_ids') or []
        try:
            entry_ids = [int(x) for x in entry_ids if x is not None and str(x).strip() != '']
        except Exception:
            entry_ids = []
        print(f"Received request for column: {column_name}", flush=True)
        
        if not column_name:
            print("ERROR: Column name is missing", flush=True)
            return jsonify({'success': False, 'message': 'Column name is required'})
        
        print(f"=== RECALCULATE COLUMN: {column_name} ===", flush=True)
        print(f"Starting at {time.strftime('%H:%M:%S')}", flush=True)
        
        # Load column metadata to check if this is a derived column
        column_metadata = load_column_metadata()
        
        if column_name in column_metadata and column_metadata[column_name].get('category') == 'derived':
            # This is a derived column - recalculate using stored formula
            formula = column_metadata[column_name].get('formula')
            if not formula:
                return jsonify({'success': False, 'message': f'No formula found for derived column "{column_name}"'})
            
            print(f"Formula for {column_name}: {formula}", flush=True)
            
            # Use the existing calculate_derived_column logic but only for this column
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get entries (all, or only selected rowids)
            if entry_ids:
                placeholders = ",".join(["?"] * len(entry_ids))
                cursor.execute(f"SELECT rowid, * FROM results WHERE rowid IN ({placeholders})", entry_ids)
            else:
                cursor.execute("SELECT rowid, * FROM results")
            entries = cursor.fetchall()
            scope_msg = f"{len(entry_ids)} selected entries" if entry_ids else "all entries"
            print(f"Found {len(entries)} entries to process ({scope_msg})", flush=True)
            sys.stdout.flush()
            
            # Verify column exists ONCE before the loop (performance optimization)
            cursor.execute("PRAGMA table_info(results)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            if column_name not in existing_columns:
                conn.close()
                print(f"ERROR: Column '{column_name}' does not exist in database", flush=True)
                return jsonify({'success': False, 'message': f'Column "{column_name}" does not exist in database'})
            
            updated_count = 0
            errors = []
            
            # Setpoint columns that should always be numeric
            setpoint_columns = ['dev_db_setpoint', 'dev_wb_setpoint', 'dev_tsup_setpoint']
            
            entry_start_time = time.time()
            for idx, entry in enumerate(entries):
                entry_dict = dict(entry)
                
                try:
                    # Create evaluation context
                    eval_context = {}
                    # Get test condition for calculating setpoints if needed
                    test_condition = entry_dict.get('dev_test_condition')
                    file_name = entry_dict.get('file_name')
                    
                    for key, value in entry_dict.items():
                        if key != 'rowid' and key != 'time_series_data':
                            try:
                                if value is not None:
                                    if key in setpoint_columns:
                                        eval_context[key] = float(value)
                                    else:
                                        eval_context[key] = float(value)
                                else:
                                    # If value is None, check if it's a setpoint column that we can calculate
                                    if key == 'dev_tsup_setpoint' and test_condition:
                                        calculated_value = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                        eval_context[key] = calculated_value
                                    elif key == 'dev_db_setpoint' and test_condition:
                                        calculated_value = unit_config.get_tdb_setpoint_for(test_condition)
                                        if calculated_value is not None:
                                            eval_context[key] = float(calculated_value)
                                        else:
                                            eval_context[key] = None
                                    elif key == 'dev_wb_setpoint' and test_condition:
                                        db_setpoint = unit_config.get_tdb_setpoint_for(test_condition)
                                        if db_setpoint is not None:
                                            calculated_value = db_setpoint - 1
                                            eval_context[key] = float(calculated_value)
                                        else:
                                            eval_context[key] = None
                                    else:
                                        eval_context[key] = None
                            except (ValueError, TypeError):
                                if key in setpoint_columns:
                                    try:
                                        if isinstance(value, str):
                                            cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                                            if cleaned:
                                                eval_context[key] = float(cleaned)
                                            else:
                                                # Try calculating if we have test_condition
                                                if key == 'dev_tsup_setpoint' and test_condition:
                                                    eval_context[key] = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                                else:
                                                    eval_context[key] = None
                                        else:
                                            # Try calculating if we have test_condition
                                            if key == 'dev_tsup_setpoint' and test_condition:
                                                eval_context[key] = _tsup_setpoint_from_entry(entry_dict, test_condition, file_name)
                                            else:
                                                eval_context[key] = None
                                    except Exception as calc_error:
                                        eval_context[key] = None
                                else:
                                    eval_context[key] = value
                    
                    # Add math functions
                    import math
                    safe_builtins = {
                        'abs': abs, 'min': min, 'max': max, 'round': round, 'pow': pow,
                        'sum': sum, 'len': len, 'int': int, 'float': float, 'str': str,
                        'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log, 'log10': math.log10,
                        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
                        'ceil': math.ceil, 'floor': math.floor,
                    }
                    eval_context.update(safe_builtins)
                    
                    # Check if any required values are None before evaluating
                    # This prevents errors like "unsupported operand type(s) for +: 'NoneType' and 'float'"
                    import re
                    setpoint_vars = ['dev_db_setpoint', 'dev_wb_setpoint', 'dev_tsup_setpoint']
                    missing_vars = []
                    for var in setpoint_vars:
                        if var in formula:
                            value = eval_context.get(var)
                            if value is None:
                                missing_vars.append(var)
                    
                    if missing_vars:
                        # Log which entry and which variables are missing
                        file_name = entry_dict.get('file_name', 'unknown')
                        data_set = entry_dict.get('data_set', 'unknown')
                        rowid_val = entry_dict.get('rowid', 'unknown')
                        print(f"Warning: Entry rowid={rowid_val} ({file_name}, dataset {data_set}) has None values for: {', '.join(missing_vars)}. Formula: {formula}")
                        # Set result to None if required values are missing
                        result = None
                    else:
                        # Evaluate formula
                        result = eval(formula, {"__builtins__": {}}, eval_context)
                    
                    # Update database - ONLY update the requested column
                    # Column existence was already verified before the loop
                    if result is not None:
                        if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
                            cursor.execute(f'UPDATE results SET "{column_name}" = NULL WHERE rowid = ?', (entry_dict['rowid'],))
                        else:
                            cursor.execute(f'UPDATE results SET "{column_name}" = ? WHERE rowid = ?', (result, entry_dict['rowid']))
                            updated_count += 1
                    else:
                        cursor.execute(f'UPDATE results SET "{column_name}" = NULL WHERE rowid = ?', (entry_dict['rowid'],))
                
                except Exception as e:
                    errors.append(f"Entry {entry_dict.get('file_name', 'unknown')}: {str(e)}")
                    import traceback
                    print(f"Error processing entry {entry_dict.get('rowid')}: {e}")
                
                # Log progress every 10 entries and commit periodically to avoid long transactions
                if (idx + 1) % 10 == 0:
                    elapsed = time.time() - entry_start_time
                    print(f"Processed {idx + 1}/{len(entries)} entries in {elapsed:.2f}s", flush=True)
                    sys.stdout.flush()
                    # Commit every 10 entries to avoid long transactions and potential timeouts
                    try:
                        conn.commit()
                    except Exception as commit_error:
                        print(f"Warning: Error committing at entry {idx + 1}: {commit_error}", flush=True)
            
            # Final commit
            conn.commit()
            conn.close()
            
            total_time = time.time() - start_time
            print(f"{'='*60}", flush=True)
            print(f"=== COMPLETED: {column_name} ===", flush=True)
            print(f"Updated {updated_count} entries in {total_time:.2f} seconds", flush=True)
            print(f"Finished at {time.strftime('%H:%M:%S')}", flush=True)
            print(f"{'='*60}\n", flush=True)
            sys.stdout.flush()
            
            message = f'Recalculated {column_name} for {updated_count} entries in {total_time:.1f}s'
            if errors:
                message += f'. {len(errors)} errors occurred.'
            
            return jsonify({
                'success': True,
                'message': message,
                'updated_count': updated_count,
                'errors': errors[:10]
            })
        else:
            # Code-computed from the Excel time series (not a stored formula on other result columns).
            if column_name in ('avg_t_mean_log', 't_mean_from_avgs', 'avg_dt_ln'):
                conn = get_db_connection()
                cursor = conn.cursor()
                if entry_ids:
                    placeholders = ",".join(["?"] * len(entry_ids))
                    cursor.execute(
                        f"SELECT rowid, file_name, data_set, start_time, end_time FROM results WHERE rowid IN ({placeholders})",
                        entry_ids,
                    )
                else:
                    cursor.execute("SELECT rowid, file_name, data_set, start_time, end_time FROM results")
                entries = cursor.fetchall()
                updated_count = 0
                errors = []
                from collections import OrderedDict
                _excel_cache = OrderedDict()
                MAX_CACHE = 40

                def _cached_read(fname, dset):
                    key = (fname, dset)
                    if key in _excel_cache:
                        _excel_cache.move_to_end(key)
                        return _excel_cache[key]
                    df = _read_excel_sheet(fname, dset)
                    _excel_cache[key] = df
                    if len(_excel_cache) > MAX_CACHE:
                        _excel_cache.popitem(last=False)
                    return df

                for entry in entries:
                    e = dict(entry)
                    try:
                        stf, etf = float(e['start_time']), float(e['end_time'])
                    except (TypeError, ValueError):
                        errors.append(f"Entry {e.get('rowid')}: invalid start/end time")
                        continue
                    df_full = _cached_read(e.get('file_name'), e.get('data_set'))
                    try:
                        avg_t_mean_log, t_mean_from_avgs, avg_dt_ln = _write_results_t_mean(
                            conn, int(e['rowid']), df_full, stf, etf
                        )
                        if (column_name == 'avg_t_mean_log' and avg_t_mean_log is not None) or (
                            column_name == 't_mean_from_avgs' and t_mean_from_avgs is not None
                        ) or (column_name == 'avg_dt_ln' and avg_dt_ln is not None):
                            updated_count += 1
                    except Exception as e_inner:
                        errors.append(f"Entry {e.get('file_name', e.get('rowid'))}: {e_inner}")
                conn.commit()
                conn.close()
                total_time = time.time() - start_time
                message = f'Recalculated {column_name} from Excel for {updated_count} entries in {total_time:.1f}s'
                if errors:
                    message += f'. {len(errors)} errors occurred.'
                return jsonify({
                    'success': True,
                    'message': message,
                    'updated_count': updated_count,
                    'errors': errors[:10],
                })
            return jsonify({
                'success': False,
                'message': f'Column "{column_name}" is not a derived column. Use the "Manage Columns" page to recalculate it.'
            })
            
    except Exception as e:
        if conn is not None:
            try:
                conn.close()
            except:
                pass
        print(f"{'='*60}", flush=True)
        print(f"ERROR recalculating column: {e}", flush=True)
        print(f"Error type: {type(e).__name__}", flush=True)
        traceback.print_exc()
        print(f"{'='*60}\n", flush=True)
        sys.stdout.flush()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/remove_column', methods=['POST'])
def remove_column():
    """Remove a column from the database"""
    try:
        data = request.get_json()
        column_name = data.get('column_name', '').strip()
        
        if not column_name:
            return jsonify({'success': False, 'message': 'Column name is required'})
        
        # Protect system columns from deletion
        system_columns = {'rowid', 'file_name', 'data_set', 'start_time', 'end_time', 
                         'has_time_series', 'time_series_data', 'display_order'}
        
        if column_name in system_columns:
            return jsonify({'success': False, 'message': f'Cannot remove system column: {column_name}'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(results)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        if column_name not in existing_columns:
            conn.close()
            return jsonify({'success': False, 'message': f'Column "{column_name}" does not exist'})
        
        # SQLite 3.35.0+ supports DROP COLUMN
        # For older versions, we'd need to recreate the table, but let's try the modern approach first
        try:
            cursor.execute(f"ALTER TABLE results DROP COLUMN {column_name}")
            conn.commit()
            print(f"Removed column {column_name}")
        except sqlite3.OperationalError as e:
            error_msg = str(e)
            if "DROP COLUMN" in error_msg or "not supported" in error_msg.lower():
                # SQLite version doesn't support DROP COLUMN - need to recreate table
                conn.close()
                return jsonify({
                    'success': False, 
                    'message': f'Your SQLite version does not support DROP COLUMN. Column removal requires SQLite 3.35.0+. Error: {error_msg}'
                })
            else:
                conn.close()
                return jsonify({'success': False, 'message': f'Error removing column: {error_msg}'})
        
        # Remove column metadata if it exists
        all_metadata = load_column_metadata()
        if column_name in all_metadata:
            del all_metadata[column_name]
            metadata_file = get_column_metadata_file()
            try:
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(all_metadata, f, indent=2)
            except Exception as e:
                print(f"Error removing column metadata: {e}")
        
        conn.close()
        return jsonify({'success': True, 'message': f'Column "{column_name}" removed successfully'})
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        print(f"Error removing column: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/save_column_order', methods=['POST'])
def save_column_order():
    """Save the preferred column display order and visibility settings"""
    try:
        data = request.get_json()
        column_order = data.get('column_order', [])
        column_visibility = data.get('column_visibility', {})
        
        if not column_order:
            return jsonify({'success': False, 'message': 'Column order is required'})
        
        # Filter out deviation columns (dev_*) from saved order
        # (they can still be in visibility settings)
        column_order = [col for col in column_order if not col.startswith('dev_')]
        
        # Save column order to a JSON file for persistence
        import json
        config_file = os.path.join(os.path.dirname(__file__), 'column_order.json')
        try:
            with open(config_file, 'w') as f:
                json.dump({'column_order': column_order}, f, indent=2)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error saving column order: {str(e)}'})
        
        # Save column visibility settings
        if column_visibility:
            save_column_visibility(column_visibility)
        
        return jsonify({'success': True, 'message': 'Column order and visibility saved successfully'})
        
    except Exception as e:
        print(f"Error saving column order: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/toggle_monitoring_columns', methods=['POST'])
def toggle_monitoring_columns():
    """Show or hide the 16 optional monitoring extras on Means. Same JSON as Manage Columns."""
    try:
        data = request.get_json() or {}
        show = bool(data.get('show'))
        visibility = load_column_visibility()
        for col in dataset_manager.OPTIONAL_MONITORING_MEAN_COLUMNS:
            visibility[col] = show
        save_column_visibility(visibility)
        return jsonify({'success': True, 'show': show})
    except Exception as e:
        print(f"Error toggling monitoring columns: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/save_column_formula', methods=['POST'])
def save_column_formula():
    """Save a formula for an existing column (for columns created before metadata saving was added)"""
    try:
        data = request.get_json()
        column_name = data.get('column_name')
        formula = data.get('formula', '').strip()
        
        if not column_name or not formula:
            return jsonify({'success': False, 'message': 'Column name and formula are required'})
        
        # Save the formula in metadata
        save_column_metadata(column_name, 'derived', {'formula': formula})
        
        return jsonify({'success': True, 'message': f'Formula saved for column "{column_name}"'})
        
    except Exception as e:
        print(f"Error saving column formula: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_column_order', methods=['GET'])
def get_column_order():
    """Get the saved column display order"""
    try:
        import json
        import os
        config_file = os.path.join(os.path.dirname(__file__), 'column_order.json')
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data = json.load(f)
                return jsonify({'success': True, 'column_order': data.get('column_order', [])})
        else:
            return jsonify({'success': True, 'column_order': []})
        
    except Exception as e:
        print(f"Error getting column order: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/calculate_timeseries_column', methods=['POST'])
def calculate_timeseries_column():
    """Calculate time series analysis column values"""
    try:
        data = request.get_json()
        column_name = data.get('column_name')
        analysis_type = data.get('analysis_type')  # 'count_violations', 'statistical', 'custom'
        parameters = data.get('parameters', {})  # Analysis-specific parameters
        
        if not column_name:
            return jsonify({'success': False, 'message': 'Column name is required'})
        
        # Store the analysis configuration in metadata
        save_column_metadata(column_name, 'timeseries', {
            'analysis_type': analysis_type,
            'parameters': parameters
        })
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(results)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        if column_name not in existing_columns:
            conn.close()
            return jsonify({'success': False, 'message': f'Column "{column_name}" does not exist'})
        
        # Get all entries with time series data
        cursor.execute("""
            SELECT rowid, file_name, data_set, start_time, end_time, has_time_series 
            FROM results 
            WHERE has_time_series = 1 AND start_time IS NOT NULL AND end_time IS NOT NULL
        """)
        entries = cursor.fetchall()
        
        updated_count = 0
        errors = []
        
        for entry in entries:
            entry_dict = dict(entry)
            file_name = entry_dict['file_name']
            data_set = entry_dict['data_set']
            start_time = entry_dict['start_time']
            end_time = entry_dict['end_time']
            
            try:
                # Load time series data
                df = load_time_series_data(file_name, start_time, end_time)
                
                if df is None or df.empty:
                    errors.append(f"{file_name}, dataset {data_set}: No time series data available")
                    continue
                
                # Perform analysis based on type
                result = None
                
                if analysis_type == 'count_violations':
                    # Count violations beyond permissible limits
                    param_name = parameters.get('parameter_name')
                    upper_limit = parameters.get('upper_limit')
                    lower_limit = parameters.get('lower_limit')
                    
                    if param_name and param_name in df.columns:
                        violations = 0
                        if upper_limit is not None:
                            violations += (df[param_name] > float(upper_limit)).sum()
                        if lower_limit is not None:
                            violations += (df[param_name] < float(lower_limit)).sum()
                        result = int(violations)
                    else:
                        errors.append(f"{file_name}, dataset {data_set}: Parameter '{param_name}' not found in time series")
                        continue
                
                elif analysis_type == 'statistical':
                    # Statistical analysis (mean, std, min, max, etc.)
                    param_name = parameters.get('parameter_name')
                    stat_type = parameters.get('stat_type', 'mean')  # mean, std, min, max, median
                    
                    if param_name and param_name in df.columns:
                        if stat_type == 'mean':
                            result = float(df[param_name].mean())
                        elif stat_type == 'std':
                            result = float(df[param_name].std())
                        elif stat_type == 'min':
                            result = float(df[param_name].min())
                        elif stat_type == 'max':
                            result = float(df[param_name].max())
                        elif stat_type == 'median':
                            result = float(df[param_name].median())
                    else:
                        errors.append(f"{file_name}, dataset {data_set}: Parameter '{param_name}' not found in time series")
                        continue
                
                elif analysis_type == 'custom':
                    # Custom formula evaluation on time series
                    formula = parameters.get('formula', '')
                    if formula:
                        # Evaluate formula on dataframe
                        # WARNING: Uses eval() - should be replaced with safer evaluator in production
                        try:
                            result = float(eval(formula, {"__builtins__": {}}, {"df": df}))
                        except Exception as e:
                            errors.append(f"{file_name}, dataset {data_set}: Formula error: {str(e)}")
                            continue
                
                # Update database
                if result is not None:
                    cursor.execute(f"""
                        UPDATE results 
                        SET {column_name} = ? 
                        WHERE rowid = ?
                    """, (result, entry_dict['rowid']))
                    updated_count += 1
                else:
                    cursor.execute(f"""
                        UPDATE results 
                        SET {column_name} = NULL 
                        WHERE rowid = ?
                    """, (entry_dict['rowid'],))
                    
            except Exception as e:
                errors.append(f"{file_name}, dataset {data_set}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        message = f'Calculated {updated_count} values'
        if errors:
            message += f'. {len(errors)} errors occurred.'
        
        return jsonify({
            'success': True,
            'message': message,
            'updated_count': updated_count,
            'errors': errors[:10]
        })
        
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        print(f"Error calculating time series column: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# ==================== Permissible Deviation Analysis ====================

def calculate_deviation_bands(setpoints, db_band, wb_band):
    """Calculate permissible deviation bands for each test condition"""
    deviation_bands = {}
    for condition, db_setpoint in setpoints.items():
        deviation_bands[condition] = {
            'DB': (db_setpoint - db_band, db_setpoint + db_band),
            'WB': (db_setpoint - wb_band - 1, db_setpoint + wb_band - 1)  # WB setpoint is 1°C lower
        }
    return deviation_bands

def _reload_setpoint_maps():
    """Load outdoor DB and supply setpoints from config/condition_sets.json (B1.1)."""
    global DEVIATION_SETPOINTS, TSUP_SETPOINTS
    DEVIATION_SETPOINTS = unit_config.get_tdb_setpoints()
    TSUP_SETPOINTS = unit_config.get_tsup_setpoints()
    # Keep config.json alias in sync so older call sites using t_supply_set_mapping agree
    config['t_supply_set_mapping'] = dict(TSUP_SETPOINTS)


_reload_setpoint_maps()


def get_tsup_setpoint(test_condition, file_name=None, condition_set_id=None,
                      climate=None, application=None):
    """Get supply temperature setpoint based on test condition (from condition_sets.json).

    Variations like "A real BUH" / "C70min" use the same setpoint as the base letter.
    climate / application select the Table 3 slice; omitted values default to Average / MT.
    """
    value = unit_config.get_tsup_setpoint_for(
        test_condition,
        condition_set_id=condition_set_id,
        file_name=file_name,
        climate=climate,
        application=application,
    )
    if value is not None:
        return value
    # Last resort: in-memory Average/MT map, only when the caller did not pick another slice
    clim = unit_config.normalize_climate(climate)
    app = unit_config.normalize_application(application)
    using_defaults = clim in (None, unit_config.get_default_climate(condition_set_id)) and app in (
        None, unit_config.get_default_application(condition_set_id)
    )
    if using_defaults:
        letter = unit_config.condition_letter(test_condition)
        if letter:
            return TSUP_SETPOINTS.get(letter)
    return None


def _entry_climate_application(entry_dict):
    """Resolve climate × application from the results row, then the unit profile."""
    entry_dict = entry_dict or {}
    profile, _ = unit_config.resolve_profile(
        file_name=entry_dict.get('file_name'),
        hp_id=entry_dict.get('hp_id'),
        profile_id=entry_dict.get('profile_id'),
    )
    return unit_config.resolve_climate_application(
        climate=entry_dict.get('climate'),
        application=entry_dict.get('application'),
        profile=profile,
        condition_set_id=entry_dict.get('condition_set_id'),
    )


def _entry_is_variable_flow(entry_dict, flow_config=None):
    """True when the test was run with variable water flow.

    Profile flow_mode wins when it is 'variable' or 'fixed'. Per-entry
    flow_config (var / std / fixed) is used only for 'per_test' units.
    Needed because the ecodesign Table 3 mean temperature may only be held
    by varying the flow (or raising Tsup when flow is variable).
    """
    entry_dict = entry_dict or {}
    profile, _ = unit_config.resolve_profile(
        file_name=entry_dict.get('file_name'),
        hp_id=entry_dict.get('hp_id'),
        profile_id=entry_dict.get('profile_id'),
    )
    mode = str((profile or {}).get('flow_mode') or '').strip().lower()
    if mode == 'variable':
        return True
    if mode == 'fixed':
        return False
    fc = flow_config if flow_config is not None else entry_dict.get('flow_config')
    fc = str(fc or '').strip().lower()
    return fc in ('var', 'variable')


# Air-source capabilities: what every BAM/HPT unit was assumed to have before
# unit_types.json existed. Used when a profile or the config file is missing so
# an unconfigured database keeps its current DB/WB checks.
_DEFAULT_UNIT_CAPABILITIES = {
    'source_medium': 'air',
    'has_wetbulb': True,
    'source_circuit_measured': False,
    'has_gas_input': False,
}
_ALL_CHECK_IDS = ('db', 'wb', 'tsup', 'dtreturn', 'flow', 'tmean', 'gas_input')


def _entry_unit_capabilities(entry_dict):
    """Capability flags (unit_types.json) for the unit this results row belongs to."""
    entry_dict = entry_dict or {}
    profile, _ = unit_config.resolve_profile(
        file_name=entry_dict.get('file_name'),
        hp_id=entry_dict.get('hp_id'),
        profile_id=entry_dict.get('profile_id'),
    )
    unit_type = str((profile or {}).get('unit_type') or '').strip()
    caps = (unit_config.load_unit_types() or {}).get(unit_type)
    if not isinstance(caps, dict):
        caps = {}
    return {**_DEFAULT_UNIT_CAPABILITIES, **caps}


def _entry_applicable_checks(entry_dict, flow_config=None):
    """Check ids from checks.json that apply to this entry (CF-02 / CF-03).

    A check applies when every key of its applies_when matches the unit's
    capabilities or the resolved flow mode. Inapplicable checks must end up
    empty (NULL / n/a), never scored as zero violations.
    """
    context = _entry_unit_capabilities(entry_dict)
    context['entry_flow_mode'] = (
        'variable' if _entry_is_variable_flow(entry_dict, flow_config) else 'fixed'
    )
    checks = (unit_config.load_checks() or {}).get('checks') or []
    if not checks:
        return set(_ALL_CHECK_IDS)
    applicable = set()
    for check in checks:
        if not isinstance(check, dict):
            continue
        check_id = str(check.get('id') or '').strip()
        if not check_id:
            continue
        applies_when = check.get('applies_when') or {}
        if all(context.get(key) == value for key, value in applies_when.items()):
            applicable.add(check_id)
    return applicable


def _tmean_setpoint_from_entry(entry_dict, test_condition=None):
    """Ecodesign Table 3 mean water temperature for this entry's letter × climate × application."""
    entry_dict = entry_dict or {}
    climate, application = _entry_climate_application(entry_dict)
    return unit_config.get_tmean_setpoint_for(
        test_condition if test_condition is not None else (
            entry_dict.get('dev_test_condition') or entry_dict.get('test_cond')
        ),
        condition_set_id=entry_dict.get('condition_set_id'),
        climate=climate,
        application=application,
    )


def _tsup_setpoint_from_entry(entry_dict, test_condition=None, file_name=None):
    entry_dict = entry_dict or {}
    climate, application = _entry_climate_application(entry_dict)
    return get_tsup_setpoint(
        test_condition if test_condition is not None else (
            entry_dict.get('dev_test_condition') or entry_dict.get('test_cond')
        ),
        file_name=file_name if file_name is not None else entry_dict.get('file_name'),
        condition_set_id=entry_dict.get('condition_set_id'),
        climate=climate,
        application=application,
    )


# Permissible deviation configuration file path
def get_permissible_deviations_file():
    """Get the path to the permissible deviations configuration JSON file"""
    try:
        app_dir = _APP_DIR
    except NameError:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, 'permissible_deviations.json')

def load_permissible_deviations():
    """Load permissible deviations configuration from JSON file.
    Includes: transient bands (DB/WB/Tsup for time series), mean limits (cycle mean),
    and violation coloring thresholds (DB %, WB %, violation counts).
    """
    config_file = get_permissible_deviations_file()
    default_config = {
        # Transient operation: bands for time series plots
        'DB': {
            'value': 1.0,
            'unit': 'K',
            'description': 'Dry-bulb temperature deviation band (transient)',
            'standard_reference': 'SN EN 14511-3:2022 Table 6, Interval H'
        },
        'WB': {
            'value': 1.0,
            'unit': 'K',
            'description': 'Wet-bulb temperature deviation band (transient)',
            'standard_reference': 'SN EN 14511-3:2022 Table 6, Interval H'
        },
        'Tsup': {
            'value': 0.5,
            'unit': 'K',
            'description': 'Supply temperature deviation band (transient)',
            'standard_reference': 'SN EN 14511-3:2022 Table 6, Interval H - Arithmetical mean'
        },
        'dTreturn': {
            'lower': -2.0,
            'upper': 2.0,
            'unit': 'K',
            'description': 'Return temperature difference band (transient): dTreturn = T_return_emu − T_return_calc',
            'standard_reference': ''
        },
        # Mean water temperature vs ecodesign Table 3 — documentation only; band is mean_tmean_k.
        'Tmean': {
            'unit': 'K',
            'scope': 'cycle_mean',
            'applies_to': 'variable_flow',
            'band_key': 'mean_tmean_k',
            'description': (
                'Mean water temperature: cycle mean of T_mean_log against the ecodesign Table 3 mean. '
                'Variable-flow tests only — fixed-flow tests run above the Table 3 mean at part load by '
                'physics, so they are judged on Tsup instead. No transient band is defined.'
            ),
            'standard_reference': (
                'BAM lab proposal — the ecodesign draft defines no tolerance for the mean temperature. '
                '0.5 K chosen as the analogue of SN EN 14511-3:2022 Table 6 Interval H (mean supply temperature).'
            ),
        },
        # Mean value permissible deviations (cycle mean) — used to color Mean DB/WB/Tsup Dev columns
        'mean_db_k': 0.6,
        'mean_wb_k': 0.4,
        'mean_tsup_k': 0.5,
        'mean_tmean_k': 0.5,
        'mean_q_band_pct': 5.0,   # scatter plot Q (heating capacity) band ±%
        # Violation coloring: red above _red, yellow above _yellow (percent or count)
        'db_pct_red': 5.0,
        'db_pct_yellow': 1.0,
        'wb_pct_red': 5.0,
        'wb_pct_yellow': 1.0,
        'db_violations_red': 0,   # red if violations > this
        'wb_violations_red': 0,
        'tsup_pct_red': 5.0,
        'tsup_pct_yellow': 1.0,
        'tsup_violations_red': 0,
        'dtreturn_pct_red': 5.0,
        'dtreturn_pct_yellow': 1.0,
        'dtreturn_violations_red': 0,
        # Mean Q dev (%) column coloring
        'mean_q_pct_red': 10.0,
        'mean_q_pct_yellow': 5.0,
        # Flow permissible deviations (EN 14511): mean ±1%, instantaneous ±2.5%
        'mean_flow_pct': 1.0,
        'flow_instantaneous_pct': 2.5,
        'flow_violations_red': 0,
        'flow_pct_red': 5.0,
        'flow_pct_yellow': 1.0,
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key in default_config:
                    if key not in loaded_config:
                        loaded_config[key] = default_config[key]
                    elif key in ('DB', 'WB', 'Tsup', 'dTreturn', 'Tmean') and isinstance(default_config[key], dict):
                        for sub in default_config[key]:
                            if sub not in loaded_config[key]:
                                loaded_config[key][sub] = default_config[key][sub]
                return loaded_config
        except Exception as e:
            print(f"Error loading permissible deviations config: {e}")
            return default_config
    else:
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default permissible deviations config at {config_file}")
        except Exception as e:
            print(f"Error creating default permissible deviations config: {e}")
        return default_config

def save_permissible_deviations(config):
    """Save permissible deviations configuration to JSON file"""
    config_file = get_permissible_deviations_file()
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving permissible deviations config: {e}")
        return False

# Load permissible deviations configuration
PERMISSIBLE_DEVIATIONS = load_permissible_deviations()

# Get deviation band values (for backward compatibility)
DB_BAND = PERMISSIBLE_DEVIATIONS.get('DB', {}).get('value', 1.0)
WB_BAND = PERMISSIBLE_DEVIATIONS.get('WB', {}).get('value', 1.0)
TSUP_BAND = PERMISSIBLE_DEVIATIONS.get('Tsup', {}).get('value', 0.5)

# Calculate deviation bands
DEVIATION_BANDS = calculate_deviation_bands(DEVIATION_SETPOINTS, DB_BAND, WB_BAND)

# --- Design parameters (Pdesign, UA, Qset) — stored in database ---
TSET_INDOOR_HEATING = 16   # °C set temperature for heating indoor condition
TDESIGN_MT = -10           # Application design temperature for MT (E condition dev_db_setpoint)

def init_hp_design_table():
    """Create hp_design table if it doesn't exist.
    Columns: hp_id (TEXT PK), pdesign_kw (REAL), flow_type (TEXT: 'mass'|'volume'|NULL), flow_set_value (REAL|NULL).
    For fixed-flow units: flow_type and flow_set_value define the set point (kg/s or m³/h).
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hp_design (
                hp_id TEXT PRIMARY KEY,
                pdesign_kw REAL NOT NULL,
                flow_type TEXT,
                flow_set_value REAL
            )
        """)
        conn.commit()
        # Add flow columns if table existed without them
        cursor.execute("PRAGMA table_info(hp_design)")
        cols = [row[1] for row in cursor.fetchall()]
        if 'flow_type' not in cols:
            cursor.execute("ALTER TABLE hp_design ADD COLUMN flow_type TEXT")
            conn.commit()
        if 'flow_set_value' not in cols:
            cursor.execute("ALTER TABLE hp_design ADD COLUMN flow_set_value REAL")
            conn.commit()
    finally:
        conn.close()

def normalize_hp_id(hp_id):
    """Return canonical HP id for legacy hp_design lookups: '1a' -> '1', '10' -> '10'."""
    if hp_id is None:
        return None
    s = _normalize_hp_id(hp_id)
    return s or None

def get_pdesign_for_hp(hp_id, conn=None, file_name=None, profile_id=None):
    """Return Pdesign (kW). Prefer unit profile; fall back to hp_design table."""
    pdesign, reason = unit_config.resolve_pdesign(file_name=file_name, hp_id=hp_id, profile_id=profile_id)
    if pdesign is not None:
        return pdesign
    if hp_id is None:
        return None
    s = str(hp_id).strip()
    norm = normalize_hp_id(hp_id)
    close = conn is None
    if close:
        conn = get_db_connection()
    try:
        cursor = conn.cursor()
        for key in (norm, s):
            if not key:
                continue
            cursor.execute("SELECT pdesign_kw FROM hp_design WHERE hp_id = ?", (key,))
            row = cursor.fetchone()
            if row is not None:
                return float(row[0])
        return None
    finally:
        if close:
            conn.close()

def get_flow_set_for_hp(hp_id, conn=None, file_name=None, profile_id=None):
    """Return (flow_type, flow_set_value). Prefer unit profile; fall back to hp_design table."""
    ft, fv, reason = unit_config.resolve_flow_set(file_name=file_name, hp_id=hp_id, profile_id=profile_id)
    if ft and fv is not None:
        return ft, fv
    if hp_id is None:
        return None, None
    norm = normalize_hp_id(hp_id)
    s = str(hp_id).strip()
    keys_to_try = [norm, s]
    if s and len(s) > 2 and s.upper().startswith('HP'):
        keys_to_try.append(s[2:].strip())
    keys_to_try = [k for k in keys_to_try if k]
    close = conn is None
    if close:
        conn = get_db_connection()
    try:
        cursor = conn.cursor()
        for key in keys_to_try:
            cursor.execute("SELECT flow_type, flow_set_value FROM hp_design WHERE hp_id = ?", (key,))
            row = cursor.fetchone()
            if row is not None and row[0] and row[1] is not None:
                ft = (row[0] or '').strip().lower()
                if ft in ('mass', 'volume'):
                    return ft, float(row[1])
                return None, None
        return None, None
    finally:
        if close:
            conn.close()

def get_all_hp_designs(conn=None):
    """Return list of {hp_id, pdesign_kw, flow_type, flow_set_value} from hp_design table."""
    close = conn is None
    if close:
        conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT hp_id, pdesign_kw, flow_type, flow_set_value FROM hp_design ORDER BY hp_id")
        rows = cursor.fetchall()
        out = []
        for row in rows:
            item = {'hp_id': row[0], 'pdesign_kw': float(row[1])}
            # Coerce to str: SQLite may return TEXT as str; avoid .strip() on int
            item['flow_type'] = (str(row[2]).strip() or None) if row[2] is not None else None
            item['flow_set_value'] = float(row[3]) if row[3] is not None else None
            out.append(item)
        return out
    finally:
        if close:
            conn.close()

FLOW_MODES = ('per_test', 'fixed', 'variable')


def _unit_design_row_from_profile(profile, entry_count=0, fixed_entry_count=0):
    """One Design parameters row for a unit profile."""
    pid = profile.get('profile_id')
    source = profile.get('_source') or 'local'
    flow_mode = str(profile.get('flow_mode') or 'per_test').strip().lower()
    if flow_mode not in FLOW_MODES:
        flow_mode = 'per_test'
    flow_type = (str(profile.get('flow_type')).strip().lower() or None) if profile.get('flow_type') else None
    flow_set_value = profile.get('flow_set_value')
    return {
        'row_id': f'profile:{pid}',
        'kind': 'profile',
        'profile_id': pid,
        'unit_label': pid,
        'display_name': profile.get('display_name') or '',
        'hp_id': None,
        'pdesign_kw': profile.get('pdesign_kw'),
        'flow_type': flow_type,
        'flow_set_value': flow_set_value,
        'flow_mode': flow_mode,
        'source_label': f'profile {pid} ({source})',
        'destination': source,
        'entry_count': entry_count,
        'fixed_entry_count': fixed_entry_count,
        # flow_type without a value never reaches the flow check: resolve_flow_set needs both.
        'flow_incomplete': bool(flow_type) and flow_set_value is None,
        'flow_expected': flow_mode == 'fixed' or (flow_mode == 'per_test' and fixed_entry_count > 0),
    }


def get_unit_design_rows(results=None, hp_key=None, flow_key=None, conn=None):
    """Rows for the Design parameters panel, keyed by unit rather than by hp_id.

    Profiles come first because they are what the calculations read. Legacy hp_design
    rows appear only when no profile claims them, and unit ids found in the data that
    resolve to nothing at all are listed so they can be configured.
    """
    profiles = {p.get('profile_id'): p for p in unit_config.list_profiles() if p.get('profile_id')}
    counts = {pid: [0, 0] for pid in profiles}
    legacy_counts = {}
    unresolved_counts = {}
    resolve_cache = {}

    for entry in results or []:
        hp_id = entry.get(hp_key) if hp_key else None
        file_name = entry.get('file_name')
        is_fixed = bool(flow_key) and str(entry.get(flow_key) or '').strip().lower() == 'fixed'
        pid = str(entry.get('profile_id') or '').strip()
        if pid not in profiles:
            cache_key = (file_name, str(hp_id))
            if cache_key not in resolve_cache:
                profile, _reason = unit_config.resolve_profile(file_name=file_name, hp_id=hp_id)
                resolve_cache[cache_key] = (profile or {}).get('profile_id')
            pid = resolve_cache[cache_key] or ''
        if pid in profiles:
            bucket = counts[pid]
        else:
            legacy_key = normalize_hp_id(hp_id)
            target = legacy_counts if legacy_key else unresolved_counts
            key = legacy_key or (file_name or 'unknown')
            bucket = target.setdefault(key, [0, 0])
        bucket[0] += 1
        if is_fixed:
            bucket[1] += 1

    rows = [
        _unit_design_row_from_profile(profiles[pid], counts[pid][0], counts[pid][1])
        for pid in profiles
    ]

    # Legacy table rows are only in charge for units that no profile claims.
    for item in get_all_hp_designs(conn):
        hp_id = str(item.get('hp_id') or '').strip()
        if not hp_id:
            continue
        profile, _reason = unit_config.resolve_profile(hp_id=hp_id)
        if profile is not None:
            continue
        total, fixed = legacy_counts.pop(hp_id, [0, 0])
        rows.append({
            'row_id': f'legacy:{hp_id}',
            'kind': 'legacy',
            'profile_id': None,
            'unit_label': f'HP{hp_id}',
            'display_name': '',
            'hp_id': hp_id,
            'pdesign_kw': item.get('pdesign_kw'),
            'flow_type': item.get('flow_type'),
            'flow_set_value': item.get('flow_set_value'),
            'flow_mode': 'per_test',
            'source_label': 'legacy hp_design row (no profile)',
            'destination': 'local',
            'entry_count': total,
            'fixed_entry_count': fixed,
            'flow_incomplete': bool(item.get('flow_type')) and item.get('flow_set_value') is None,
            'flow_expected': fixed > 0,
        })

    # Units seen in the data that resolve to neither a profile nor a legacy row.
    for hp_id, (total, fixed) in sorted(legacy_counts.items()):
        rows.append({
            'row_id': f'unconfigured:{hp_id}',
            'kind': 'unconfigured',
            'profile_id': None,
            'unit_label': f'HP{hp_id}',
            'display_name': '',
            'hp_id': hp_id,
            'pdesign_kw': None,
            'flow_type': None,
            'flow_set_value': None,
            'flow_mode': 'per_test',
            'source_label': 'not configured',
            'destination': 'local',
            'entry_count': total,
            'fixed_entry_count': fixed,
            'flow_incomplete': False,
            'flow_expected': fixed > 0,
        })

    kind_rank = {'profile': 0, 'legacy': 1, 'unconfigured': 2}
    rows.sort(key=lambda r: (kind_rank.get(r['kind'], 9), -r['entry_count'], str(r['unit_label'])))
    unresolved_entry_count = sum(v[0] for v in unresolved_counts.values())
    return rows, unresolved_entry_count


def compute_ua_and_qset(pdesign_kw, db_setpoint):
    """UA = Pdesign / (Tset_indoor - Tdesign_MT), Qset = UA * (Tset_indoor - db_setpoint)."""
    if pdesign_kw is None or db_setpoint is None:
        return None, None
    try:
        tset = TSET_INDOOR_HEATING
        tdesign = TDESIGN_MT
        delta_design = tset - tdesign  # 16 - (-10) = 26
        if abs(delta_design) < 1e-6:
            return None, None
        ua = pdesign_kw / delta_design
        delta_part = tset - float(db_setpoint)
        qset = ua * delta_part
        return ua, qset
    except (TypeError, ValueError):
        return None, None

def get_mean_heating_capacity(entry):
    """Get mean heating capacity (kW) from entry. Prefer BAM corr, then corr, then uncorr."""
    for col in ('avg_heating_capacity_bam_corr', 'avg_heating_capacity_corr', 'avg_heating_capacity_uncorr'):
        val = entry.get(col)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None

def get_mean_supply_for_deviations(entry):
    """Mean supply temperature for deviation checks. Prefer Ts_buh (BUH) so BUH and non-BUH cycles are comparable.
    Tries: avg_ts_buh, avg_t_sup_buh, Ts Buh, Ts_buh, avg_t_supply."""
    for key in ('avg_ts_buh', 'avg_t_sup_buh', 'Ts Buh', 'Ts_buh', 'avg_t_supply'):
        val = entry.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None

def get_tsup_series_column(df):
    """Supply temperature time-series column for deviation checks.
    Prefer 'Ts Buh' so BUH and non-BUH cycles are treated the same way as the mean Tsup column."""
    if df is None:
        return None
    for col in ('Ts Buh', 'Ts_buh', 'T_supply'):
        if col in df.columns:
            return col
    return None


def _signed_extremes(series):
    """Upper and lower extreme of a signed deviation series.

    Returns (max, min) of the series, i.e. the largest excursion above and below the
    setpoint. Both can share a sign when the series never crosses the setpoint.
    """
    if series is None:
        return None, None
    s = series.dropna()
    if len(s) == 0:
        return None, None
    return float(s.max()), float(s.min())


def _note_missing_quantity(file_name, data_set, quantity, warning_list=None):
    """Record that a quantity had no valid data points, so it is reported and not read as zero."""
    message = (f"{file_name} (dataset {data_set}): no valid {quantity} data in the selected window — "
               f"statistics reported as not available instead of 0")
    print(f"WARNING: {message}")
    if warning_list is not None:
        warning_list.append(message)


def _has_valid_points(series):
    """True when the series holds at least one non-NaN value.

    Violation counts compare with < / >, which are False for NaN, so an all-NaN
    series would otherwise be reported as zero violations — indistinguishable from
    a compliant one. Callers report the statistic as unavailable instead.
    """
    if series is None:
        return False
    try:
        return bool(series.notna().any())
    except Exception:
        return False


SIGNED_EXTREME_KEYS = (
    'max_db_pos', 'max_db_neg',
    'max_wb_pos', 'max_wb_neg',
    'max_tsup_pos', 'max_tsup_neg',
    'max_dtreturn_pos', 'max_dtreturn_neg',
    'max_flow_pos_pct', 'max_flow_neg_pct',
)


def _signed_extreme_params(stats):
    """Values for the dev_max_*_pos / dev_max_*_neg columns, in SIGNED_EXTREME_KEYS order."""
    out = []
    for key in SIGNED_EXTREME_KEYS:
        val = stats.get(key)
        out.append(float(val) if val is not None else None)
    return out


def _get_tsup_band():
    """Instantaneous supply temperature band half-width (K) from the deviations config."""
    try:
        return float(load_permissible_deviations().get('Tsup', {}).get('value', 0.5))
    except (TypeError, ValueError):
        return 0.5


def get_mean_q_for_deviations(entry):
    """Mean heating capacity (kW) for deviation checks. Prefer QCorrwBUH (BUH-corrected) so BUH and non-BUH are comparable.
    Tries: avg_heating_capacity_corr_wbuh, QCorrwBUH, then get_mean_heating_capacity."""
    for key in ('avg_heating_capacity_corr_wbuh', 'QCorrwBUH'):
        val = entry.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return get_mean_heating_capacity(entry)


def get_mean_p_for_deviations(entry):
    """Mean electrical power (kW) for deviation checks. Prefer PCorrwBUH (BUH-corrected)."""
    for key in ('PCorrwBUH', 'avg_power_input_corr', 'avg_power_input_bam_corr'):
        val = entry.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def get_mean_cop_for_deviations(entry):
    """Mean COP for deviation checks. Prefer COPCorrwBUH (BUH-corrected)."""
    for key in ('COPCorrwBUH', 'cop_corr', 'cop_bam_corr'):
        val = entry.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def _full_cycle_stats_from_entry(entry: dict) -> dict:
    """Authoritative full-cycle mean stats already stored on the results row."""
    dur = None
    try:
        st, et = entry.get('start_time'), entry.get('end_time')
        if st is not None and et is not None:
            dur = max(0.0, float(et) - float(st))
    except (TypeError, ValueError):
        dur = None
    return {
        'duration_s': dur,
        'avg_t_db': entry.get('avg_t_db'),
        'avg_t_wb': entry.get('avg_t_wb'),
        'avg_ts_buh': get_mean_supply_for_deviations(entry),
        'avg_q_corr_wbuh': get_mean_q_for_deviations(entry),
        'avg_p_corr_wbuh': get_mean_p_for_deviations(entry),
        'avg_cop_corr_wbuh': get_mean_cop_for_deviations(entry),
        'avg_mass_flow': entry.get('avg_mass_flow'),
        'avg_volume_flow': entry.get('avg_volume_flow'),
        'dtreturn_min': entry.get('dtreturn_cache_min'),
        'dtreturn_max': entry.get('dtreturn_cache_max'),
        'dtreturn_avg': entry.get('dtreturn_cache_avg'),
        'dtreturn_n_valid': entry.get('dtreturn_cache_n_valid'),
        'avg_t_mean_log': entry.get('avg_t_mean_log'),
        't_mean_from_avgs': entry.get('t_mean_from_avgs'),
    }


def _apply_full_cycle_slot(out: dict, prefix: str, entry: dict) -> None:
    """Populate combined cycle columns (D+H / Off+On) from results, not sub-period averages."""
    full = _full_cycle_stats_from_entry(entry)
    for mk, val in full.items():
        out[f'{prefix}_{mk}'] = val

def init_scatter_dataset_column():
    """Initialize scatter_dataset column if it doesn't exist (starts empty - all entries deselected)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'scatter_dataset' not in columns:
            # Add the column (all entries will be NULL by default - deselected)
            cursor.execute("ALTER TABLE results ADD COLUMN scatter_dataset TEXT")
            conn.commit()
            print("Added scatter_dataset column (initialized as empty - all entries deselected)")
        
    except Exception as e:
        print(f"Error initializing scatter_dataset column: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

def init_deviation_columns():
    """Initialize columns for storing deviation statistics"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Check if columns exist and their types
        cursor.execute("PRAGMA table_info(results)")
        columns_info = cursor.fetchall()
        columns = [col[1] for col in columns_info]
        
        # Verify column types for setpoint columns
        setpoint_columns = ['dev_db_setpoint', 'dev_wb_setpoint', 'dev_tsup_setpoint']
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]
            if col_name in setpoint_columns:
                if col_type != 'REAL':
                    print(f"WARNING: Column {col_name} has type {col_type}, expected REAL")
        
        deviation_columns = {
            'dev_test_condition': 'TEXT',
            'dev_db_setpoint': 'REAL',
            'dev_wb_setpoint': 'REAL',
            'dev_tsup_setpoint': 'REAL',
            'dev_total_points': 'INTEGER',
            'dev_db_outside_count': 'INTEGER',
            'dev_wb_outside_count': 'INTEGER',
            'dev_db_percentage': 'REAL',
            'dev_wb_percentage': 'REAL',
            'dev_max_db_deviation': 'REAL',
            'dev_max_wb_deviation': 'REAL',
            # Supply temperature (instantaneous band around dev_tsup_setpoint)
            'dev_tsup_outside_count': 'INTEGER',
            'dev_tsup_percentage': 'REAL',
            'dev_max_tsup_deviation': 'REAL',
            # Signed extremes per quantity: largest excursion above / below setpoint
            'dev_max_db_pos': 'REAL',
            'dev_max_db_neg': 'REAL',
            'dev_max_wb_pos': 'REAL',
            'dev_max_wb_neg': 'REAL',
            'dev_max_tsup_pos': 'REAL',
            'dev_max_tsup_neg': 'REAL',
            'dev_max_dtreturn_pos': 'REAL',
            'dev_max_dtreturn_neg': 'REAL',
            'dev_max_flow_pos_pct': 'REAL',
            'dev_max_flow_neg_pct': 'REAL',
            'dev_avg_timestep': 'REAL',
            'dev_wb_calculated_from_rh': 'INTEGER',  # 1 if calculated, 0 if measured
            'dev_last_calculated': 'TEXT',  # Timestamp
            # Flow (fixed-flow set point and deviations)
            'dev_flow_setpoint': 'REAL',
            'dev_flow_set_type': 'TEXT',  # 'mass'|'volume'
            'dev_mean_flow_dev_pct': 'REAL',
            'dev_flow_outside_count': 'INTEGER',
            'dev_flow_percentage': 'REAL',
            # Return temperature difference dTreturn = T_return_emu - T_return_calc
            'dev_dtreturn_outside_count': 'INTEGER',
            'dev_dtreturn_percentage': 'REAL',
            'dev_max_dtreturn_deviation': 'REAL',
            # Cached dTreturn point-level sample for the insights page
            'dtreturn_cache_n_valid': 'INTEGER',
            'dtreturn_cache_n_nan': 'INTEGER',
            'dtreturn_cache_sample': 'TEXT',
            # Per-entry extremeness metrics for traceability/outlier triage
            'dtreturn_cache_max_abs': 'REAL',
            'dtreturn_cache_max_signed': 'REAL',
            # Signed min (typically negative) / max (typically positive) dTreturn
            # over the full cycle, for the Group entries table.
            'dtreturn_cache_min': 'REAL',
            'dtreturn_cache_max': 'REAL',
            'dtreturn_cache_avg': 'REAL',
            'dtreturn_cache_p99_abs': 'REAL',
            # Cycle classification and Pelec transition detection
            'cycle_type': 'TEXT',
            'cycle_start_marker': 'TEXT',
            'pelec_transition_time': 'REAL',
            # 'auto' | 'manual' — manual transitions are preserved on Rebuild all.
            'pelec_transition_source': 'TEXT',
            # 1 if the pipeline already attempted detection/period generation for
            # this (classified, non-'other') entry and it could not produce a
            # usable transition/period split. Prevents endless re-queuing of
            # inherently undetectable or degenerate cycles on every "Update cache".
            # "Rebuild all" (force_all) ignores this flag and retries.
            'pelec_detect_failed': 'INTEGER',
            # B1.3: which unit profile / condition set was used (nullable)
            'profile_id': 'TEXT',
            'condition_set_id': 'TEXT',
            'climate': 'TEXT',
            'application': 'TEXT',
            # Logarithmic mean water temperature vs 20 °C indoor air (ecodesign)
            'avg_t_mean_log': 'REAL',
            't_mean_from_avgs': 'REAL',
            'avg_dt_ln': 'REAL',
        }
        
        for col_name, col_type in deviation_columns.items():
            if col_name not in columns:
                try:
                    cursor.execute(f"ALTER TABLE results ADD COLUMN {col_name} {col_type}")
                    conn.commit()
                    print(f"Added column {col_name} to results table as {col_type}")
                except sqlite3.OperationalError as e:
                    print(f"Could not add column {col_name}: {e}")

        # Idempotent: preserve existing cycle splits by marking all transitions manual.
        try:
            cursor.execute(
                """
                UPDATE results SET pelec_transition_source='manual'
                WHERE pelec_transition_time IS NOT NULL
                  AND cycle_type IN ('defrost_cycle', 'on_off_cycle')
                  AND COALESCE(pelec_transition_source, '') != 'manual'
                """
            )
            if cursor.rowcount:
                conn.commit()
                print(f"Marked {cursor.rowcount} existing transitions as manual")
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()


def ensure_dtreturn_exclusions_table() -> None:
    """Create table storing manual exclusions for dTreturn insights."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dtreturn_insights_exclusions (
                entry_rowid INTEGER PRIMARY KEY,
                reason TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def ensure_cycle_periods_table() -> None:
    """Create the cycle_periods table for D/H sub-period analysis."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cycle_periods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_rowid INTEGER NOT NULL,
                period_type TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                detection_method TEXT,
                dtreturn_n_valid INTEGER,
                dtreturn_n_nan INTEGER,
                dtreturn_sample TEXT,
                dtreturn_max_abs REAL,
                dtreturn_max_signed REAL,
                dtreturn_min REAL,
                dtreturn_max REAL,
                dtreturn_avg REAL,
                dtreturn_p99_abs REAL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (entry_rowid) REFERENCES results(rowid)
            )
            """
        )
        conn.commit()
        # Forward-compatible: add buffer_s if missing (for sensitivity studies)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(cycle_periods)").fetchall()]
        if 'buffer_s' not in cols:
            try:
                conn.execute("ALTER TABLE cycle_periods ADD COLUMN buffer_s INTEGER")
                conn.commit()
            except Exception:
                # best-effort; older SQLite builds can occasionally reject ALTERs mid-run
                pass
        # Forward-compatible: signed min/avg/max of dTreturn per period (for the
        # defrost/heating extremes and mean shown in the period table).
        for _new_col in ('dtreturn_min', 'dtreturn_max', 'dtreturn_avg'):
            if _new_col not in cols:
                try:
                    conn.execute(f"ALTER TABLE cycle_periods ADD COLUMN {_new_col} REAL")
                    conn.commit()
                except Exception:
                    pass
        # Per-period mean operating-point statistics (cached from Excel trace).
        for _new_col in (
            'avg_t_db', 'avg_t_wb', 'avg_ts_buh', 'avg_q_corr_wbuh', 'avg_p_corr_wbuh',
            'avg_cop_corr_wbuh', 'avg_volume_flow', 'avg_mass_flow', 'period_stats_schema',
            'avg_t_mean_log', 't_mean_from_avgs',
        ):
            if _new_col not in cols:
                try:
                    col_type = 'INTEGER' if _new_col == 'period_stats_schema' else 'REAL'
                    conn.execute(f"ALTER TABLE cycle_periods ADD COLUMN {_new_col} {col_type}")
                    conn.commit()
                except Exception:
                    pass
        cols = [r[1] for r in conn.execute("PRAGMA table_info(cycle_periods)").fetchall()]
    finally:
        conn.close()


def ensure_cycle_extraction_suggestions_table() -> None:
    """Create table storing auto-suggested cycles for user review/approval."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cycle_extraction_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                data_set REAL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                marker_type TEXT NOT NULL, -- 'drop' (phase 2 default)
                confidence REAL,
                notes_auto TEXT,
                pelec_drop_mag REAL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cycle_sugg_file_ds ON cycle_extraction_suggestions(file_name, data_set)"
        )
        conn.commit()
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# D/H period analysis: constants, Pelec detection, classification, API
# ---------------------------------------------------------------------------

SETTLING_BUFFER_S = 600  # post-transition buffer per EN 14511 (seconds)
BUFFER_OPTIONS = [600, 900, 1200]  # used for sensitivity; cached in parallel

DEFROST_TEST_CONDS = {'A', 'B', 'E', 'F',
                      'E real BUH', 'E virtual BUH',
                      'A real BUH', 'A virtual BUH'}
ON_OFF_TEST_CONDS = {'C', 'D', 'C70min'}

PELEC_COL = 'Electric Power Input (without correction)'

# --- Guideline clock lengths (D/S buffer, equilibrium, evaluation) ----------
# Lengths come from config/cycle_periods.json so that 10 / 60 / 70 min never
# end up hardcoded in logic or column names. Draft values; they will change.
CYCLE_PERIODS_CONFIG_DEFAULTS = {
    'buffer_min': 10.0,
    'eq_min': 60.0,
    'eval_min': 70.0,
    'defrost_indicator_columns': [],
}
_cycle_periods_cfg_cache = None
_cycle_periods_cfg_mtime = -1.0


def load_cycle_periods_config() -> dict:
    """Guideline clock lengths from config/cycle_periods.json (re-read on change)."""
    global _cycle_periods_cfg_cache, _cycle_periods_cfg_mtime
    path = os.path.join(unit_config.config_dir(), 'cycle_periods.json')
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if _cycle_periods_cfg_cache is not None and mtime == _cycle_periods_cfg_mtime:
        return _cycle_periods_cfg_cache

    cfg = dict(CYCLE_PERIODS_CONFIG_DEFAULTS)
    if mtime is not None:
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                raw = json.load(fh) or {}
            cfg.update({k: v for k, v in raw.items() if k in CYCLE_PERIODS_CONFIG_DEFAULTS})
        except Exception as exc:
            print(f"[cycle_periods] cannot read {path}: {exc}")
    for key in ('buffer_min', 'eq_min', 'eval_min'):
        try:
            cfg[key] = max(0.0, float(cfg[key]))
        except (TypeError, ValueError):
            cfg[key] = float(CYCLE_PERIODS_CONFIG_DEFAULTS[key])
    if not isinstance(cfg.get('defrost_indicator_columns'), list):
        cfg['defrost_indicator_columns'] = []

    _cycle_periods_cfg_cache = cfg
    _cycle_periods_cfg_mtime = mtime
    return cfg


def _guideline_lengths() -> dict:
    """Config lengths in both minutes (for the UI) and seconds (for the clocks)."""
    cfg = load_cycle_periods_config()
    return {
        'buffer_min': cfg['buffer_min'],
        'eq_min': cfg['eq_min'],
        'eval_min': cfg['eval_min'],
        'buffer_s': cfg['buffer_min'] * 60.0,
        'eq_s': cfg['eq_min'] * 60.0,
        'eval_s': cfg['eval_min'] * 60.0,
    }


def _median_time_step_seconds(t_arr: np.ndarray) -> float:
    """Median positive Δt from ``time_elapsed`` samples (robust grid period)."""
    if t_arr.size < 2:
        return 1.0
    dv = np.diff(np.asarray(t_arr, dtype=float))
    dv = dv[(dv > 0) & np.isfinite(dv)]
    if dv.size == 0:
        return 1.0
    return float(np.median(dv))


def _last_time_on_axis_before(t_arr: np.ndarray, boundary: float, min_t: float):
    """Last timestamp in ``t_arr`` with ``boundary`` excluded and ``min_t`` inclusive.

    For uniform 10 s grids, if the next cycle starts at ``boundary``, this is typically
    ``boundary - 10`` — not ``boundary - 1``.
    """
    if t_arr.size == 0:
        return None
    mask = np.isfinite(t_arr) & (t_arr >= min_t) & (t_arr < boundary)
    below = np.where(mask)[0]
    if below.size == 0:
        return None
    return float(t_arr[int(below[-1])])


def _boundary_times_one_step_earlier(t_arr: np.ndarray, boundaries, dt_hint: float):
    """Move each boundary one sampling grid backwards on the measured ``time_elapsed`` axis.

    Causal median smoothing can shift detected cliffs forward by roughly one Δt versus the raw plateau
    last seen in plotted series; lab viewers label the **last raw high sample** before the drop.

    Returns a list parallel to ``boundaries`` (same length); does not merge duplicates — caller keeps
    magnitudes aligned by index.
    """
    if t_arr.size == 0 or not boundaries:
        return list(boundaries)
    lo = float(np.nanmin(t_arr[np.isfinite(t_arr)]))
    dt_use = float(dt_hint) if np.isfinite(float(dt_hint)) and float(dt_hint) > 0 else 1.0
    out = []
    for tau in boundaries:
        tgt = float(tau) - dt_use
        mask = np.isfinite(t_arr) & (t_arr <= tgt + 1e-6)
        if not np.any(mask):
            out.append(float(tau))
            continue
        snapped = float(t_arr[int(np.where(mask)[0][-1])])
        if snapped + 1e-9 < lo:
            out.append(float(tau))
            continue
        out.append(snapped)
    return out


def _merge_consecutive_cycle_starts_if_gap_never_truly_off(
    t_arr: np.ndarray,
    p_raw: np.ndarray,
    starts: list[float],
    mags: list[float],
    *,
    p_lo_ref: float,
    span: float,
) -> tuple[list[float], list[float]]:
    """Drop interior cycle-start boundaries when the raw-power gap never reaches true off (kW).

    VarFl shallow dips (high→~2 kW→high) still pass slope/land rules and create 2–5 min “cycles”.
    If the minimum *Pelec* between consecutive starts stays above the file’s idle/off corridor,
    the later start is removed and its drop magnitude is discarded for suggestions.
    """
    if len(starts) < 2 or len(starts) != len(mags):
        return list(starts), list(mags)
    frac_off_band = 0.28
    idle_hard = float(p_lo_ref + frac_off_band * float(span))
    # Must reach at or below this to treat the gap as a real compressor-off / cycle break.
    true_off_max_kw = float(min(0.55, max(0.25, idle_hard + 0.10 * float(span))))

    out_s = [float(starts[0])]
    out_m = [float(mags[0])]
    for k in range(1, len(starts)):
        t0 = out_s[-1]
        t1 = float(starts[k])
        if not (np.isfinite(t0) and np.isfinite(t1)):
            continue
        if t1 <= t0 + 1e-9:
            continue
        mask = np.isfinite(t_arr) & np.isfinite(p_raw) & (t_arr >= t0) & (t_arr <= t1)
        if not np.any(mask):
            out_s.append(t1)
            out_m.append(float(mags[k]))
            continue
        pmin = float(np.nanmin(p_raw[mask]))
        if pmin > true_off_max_kw:
            continue
        out_s.append(t1)
        out_m.append(float(mags[k]))

    if len(out_s) < 2 and len(starts) >= 2:
        return [float(x) for x in starts], [float(x) for x in mags]
    return out_s, out_m


def _shoulder_time_before_long_off_run(
    i0: int,
    t_arr: np.ndarray,
    p_raw: np.ndarray,
    *,
    off_kw: float = 0.15,
    shallow_hi: float = 0.85,
    staged_drop_min_kw: float = 0.10,
) -> float:
    """Return stored *shoulder* time for an off run whose first sample is index ``i0`` (``P[i0] <= off_kw``).

    Normally this is ``t[i0-1]`` (last timestep still clearly above deep off).

    When the next long off is reached via a **two-step** descent
    ``high → shallow staging (> off_kw but <= shallow_hi) → deep off``,
    Lab viewers place the cycle break **one grid step earlier** than ``t[i0-1]`` —
    before the shallow dip (a ``t[i0-2] → t[i0-1] → t[i0]`` descent breaks at ``t[i0-2]``).

    Tiny one-grid “wiggles” (``P[i-2] → P[i-1]`` drop ≪ ``staged_drop_min_kw``) are **not** staging —
    keep ``t[i0-1]`` (on a dense 1 Hz grid a single-second dip must not pull the break earlier).
    """
    if i0 <= 0:
        return float(t_arr[0])
    pr0 = float(p_raw[i0])
    if pr0 > float(off_kw):
        return float(t_arr[i0])
    p_prev = float(p_raw[i0 - 1])
    staged = float(max(0.0, float(p_raw[i0 - 2]) - p_prev))
    if (
        i0 >= 2
        and p_prev > float(off_kw)
        and p_prev <= float(shallow_hi)
        and float(p_raw[i0 - 2]) > p_prev
        and staged >= float(staged_drop_min_kw)
    ):
        return float(t_arr[i0 - 2])
    return float(t_arr[i0 - 1])


def _detect_cycle_boundaries_long_off_runs(
    df_full: pd.DataFrame,
    *,
    min_off_sec: float = 88.0,
    off_kw: float = 0.15,
    shallow_hi: float = 0.85,
    staged_drop_min_kw: float = 0.10,
    leading_prefix_on_median_kw: float = 2.65,
) -> list[tuple[float, float, float, float]]:
    """Cycles from long *true off* runs on raw uncorrected Pelec (coarse VarFl-style exports).

    Consecutive compressor *logical* cycles are separated only where ``P`` stays at or below
    ``off_kw`` for at least ``min_off_sec`` (glitch single-step offs are ignored). Boundaries never
    come from partial mid-load dips that never reach deep off (cf. slope-based false splits at ~7589 s).

    If the file **starts while the unit is already at full load** (no prior true-off in the series),
    the first shoulder–shoulder segment is an incomplete “tail”: drop the first boundary when the
    median power before the first long off is clearly in the on band (see ``leading_prefix_on_median_kw``).

    Returns ``(start, end, confidence, pelec_drop_mag)`` matching :func:`_detect_cycle_boundaries_drop_to_drop`.
    """
    if df_full is None or "time_elapsed" not in df_full.columns:
        return []
    pelec_col = _find_uncorrected_pelec_column(df_full)
    if not pelec_col:
        return []

    df = df_full[["time_elapsed", pelec_col]].copy()
    df.loc[:, "time_elapsed"] = pd.to_numeric(df["time_elapsed"], errors="coerce")
    df.loc[:, pelec_col] = pd.to_numeric(df[pelec_col], errors="coerce")
    df = df.dropna(subset=["time_elapsed", pelec_col])
    if df.empty or len(df) < 20:
        return []

    df = df.sort_values("time_elapsed", kind="mergesort")
    t_arr = df["time_elapsed"].to_numpy(dtype=float)
    p_raw = df[pelec_col].to_numpy(dtype=float)

    dt_grid = float(_median_time_step_seconds(t_arr))
    if not np.isfinite(dt_grid) or dt_grid <= 0:
        return []

    n = len(p_raw)
    i = 0
    run_starts: list[int] = []
    while i < n:
        if float(p_raw[i]) > float(off_kw):
            i += 1
            continue
        j = i
        while j < n and float(p_raw[j]) <= float(off_kw):
            j += 1
        run_len_sec = float(t_arr[j - 1] - t_arr[i]) + float(dt_grid)
        if run_len_sec + 1e-6 >= float(min_off_sec):
            run_starts.append(int(i))
        i = j

    b_times: list[float] = []
    for i0 in run_starts:
        b_times.append(
            _shoulder_time_before_long_off_run(
                i0,
                t_arr,
                p_raw,
                off_kw=off_kw,
                shallow_hi=shallow_hi,
                staged_drop_min_kw=staged_drop_min_kw,
            )
        )

    boundaries = sorted({float(x) for x in b_times if np.isfinite(x)})
    if len(boundaries) < 2:
        return []

    # Drop leading boundary when logging began mid–on-cycle: everything before the first *long* off has
    # median P clearly at or above ``leading_prefix_on_median_kw`` (FixFl / mid-test start), not a lower cold-start ramp.
    if len(boundaries) >= 3 and run_starts:
        i0f = int(run_starts[0])
        if i0f > 0:
            med_pre = float(np.nanmedian(p_raw[0:i0f]))
            if med_pre >= float(leading_prefix_on_median_kw):
                boundaries = boundaries[1:]

    if len(boundaries) < 2:
        return []

    pk_ref = float(np.nanpercentile(p_raw, 92))
    if not np.isfinite(pk_ref) or pk_ref <= 1e-9:
        pk_ref = float(np.nanmax(p_raw))

    out: list[tuple[float, float, float, float]] = []
    for k in range(len(boundaries) - 1):
        st = float(boundaries[k])
        nxt = float(boundaries[k + 1])
        et = _last_time_on_axis_before(t_arr, nxt, st)
        if et is None:
            step = dt_grid if (nxt - st) >= dt_grid else max(1.0, 0.5 * (nxt - st))
            et = float(nxt) - float(step)
        if et <= st:
            continue
        i_hi = int(np.searchsorted(t_arr, st + 1e-6, side="right"))
        lo = max(0, i_hi - max(5, int(max(3.0, 80.0 / max(dt_grid, 1e-9)))))
        plateau = float(np.nanmax(p_raw[lo : max(lo + 1, i_hi)])) if i_hi > lo else float(p_raw[min(i_hi, n - 1)])
        mag = float(max(1e-6, plateau - float(off_kw)))
        conf = float(min(1.0, max(0.0, mag / max(1e-6, 0.5 * pk_ref))))
        out.append((st, et, conf, mag))
    return out


def _find_uncorrected_pelec_column(df: pd.DataFrame):
    """Resolve the uncorrected electric-power column (exact name varies by export)."""
    if df is None or df.empty:
        return None
    if PELEC_COL in df.columns:
        return PELEC_COL
    low = {str(c).strip().lower(): c for c in df.columns}
    if PELEC_COL.lower() in low:
        return low[PELEC_COL.lower()]
    for key in (
        'electric power input (without correction)',
        'electric power input (w/o correction)',
        'electric power input w/o correction',
    ):
        if key in low:
            return low[key]
    for c in df.columns:
        s = str(c).strip().lower()
        # Short export names (e.g. "Pel_el (uncorr)")
        if 'pel' in s and 'uncorr' in s:
            return c
        if 'electric' in s and 'power' in s and 'without' in s and 'correction' in s:
            return c
    return None


def _detect_cycle_boundaries_drop_to_drop(df_full: pd.DataFrame,
                                         min_gap_s: float = 600.0,
                                         high_frac: float = 0.60,
                                         low_frac: float = 0.20,
                                         smooth_window: int = 3):
    """Drop→drop markers on uncorrected Pelec (**cliffs = steep negative slope**, shoulder timestamp).

    0. **Segmentation preference:** on raw uncorrected Pelec, detect long stretches at or below ~0.15 kW
       (true off). **Coarse** grids (median Δt ≥ ~4 s) use a short minimum off duration (~88 s). **Dense**
       1 Hz logs use a higher floor derived from **Min gap** (typically ~250 s) so brief defrost-length offs
       do not create extra cycles, while ~4 min inter-pulse gaps still break segments. Shoulder times use a
       two-step ``high → shallow → deep off`` rule only when the shallow step drops by at least ~0.10 kW; keep the shoulder at ``t[i0-1]``.

    1. **Fallback (same function name):** causal median on ``pw`` then forward slope ``(P[i]-P[i-1])/Δt``.

    2. Candidate step ``i-1→i`` requires sharp negative slope & minimum drop magnitude (single step or
       horizon fall). The **left row** must qualify as either a **full on‑plateau** (classic gate) or —
       on **coarse** ``Δt`` grids (≥ 4 s in practice) — the **standby pre‑off band** just below that:
       last low hold before a row that already hits the hard idle cluster.

       **Plus** a **near‑zero jump** path: if the very next row already sits at true off (``P[i]`` ≤ ~0.3 kW)
       but ``P[i‑1]`` is only a **low hold** sitting *below* both gates, the shoulder is still accepted.
    3. **Landing rule**:

       • **Hard idle**: ``trough_ahead ≤`` idle spine + band (same as plotted low cluster).

       • **Idle‑soft (coarse grids only, ``Δt ≥`` ~4 s)** after **heavy on** samples (≳ 0.93×Q₉₀):
       ``trough_ahead ≤ p_lo + 0.82·span`` admits long ramp‑downs that plateau near ~3 kW before the global idle band without firing spurious splits on dense 1 Hz files.

       • **Staged plunge**: sizeable cumulative drop with ``trough`` well below both immediate pre‑row level and nominal on (~58 % of ``Q₈₈``). Effective magnitude ``max(one-step ΔP, P₀−trough_ahead)``.
    4. **Cluster key** ``min_gap_s``: candidates whose shoulders fall within ``min_gap_s`` of the **cluster
       anchor** collapse to one — physically one event split across timestamps. **Moderate** shoulder
       gaps (roughly 90–420 s, lower bound scales weakly with ``min_gap_s``) start a new cluster to
       separate two real shutdown marks without cutting long steady-on stretches that have no candidates
       for minutes. Choosing **pure argmin‑slope** rewards the **last** row of a long staircase ⇒ late
       markers (1160 s). We therefore keep slopes within **≥ 92 %** of the steepest magnitude and pick
       the **earliest shoulder** inside that tie‑band.

    5. **Stored ``start_time``** = **`t[i-1]`** (last timestep still on plateau before plunge). Earlier
       experiment storing ``t[i]`` slid boundaries to idle rows and confused plots/tables vs lab labelling.

    6. After clustering, boundary times are nudged **one median Δt earlier** on the file’s time axis
       so Starts/Ends line up with the **raw** plateau edge (median smoothing introduces ~1-sample lag).

    7. **Why many pulses lump into one block:** overly strict idle gating ⇒ **few** boundaries; shading
       then spans gaps. Loose staged rule + finer clustering fixes density; **`min_gap_s` still merges**
       pulses separated by **less than** that many seconds --- lower it if bursts sit closer together.

    8. **Shallow-dip merge:** consecutive starts are merged when **uncorrected** power between them never
       dips into the file’s **true-off** kW band (idle spine + margin). That removes false splits from
       standby plateaus that are not near-zero off events. **Min gap does not set minimum cycle length.**

    ``high_frac`` / ``low_frac``: unused (**API shim**).

    Ends: ``last time_elapsed < next_start``.
    """
    if df_full is None or 'time_elapsed' not in df_full.columns:
        return []
    pelec_col = _find_uncorrected_pelec_column(df_full)
    if not pelec_col:
        return []

    df = df_full[['time_elapsed', pelec_col]].copy()
    df.loc[:, 'time_elapsed'] = pd.to_numeric(df['time_elapsed'], errors='coerce')
    df.loc[:, pelec_col] = pd.to_numeric(df[pelec_col], errors='coerce')
    df = df.dropna(subset=['time_elapsed', pelec_col])
    if df.empty or len(df) < 20:
        return []

    df = df.sort_values('time_elapsed', kind='mergesort')

    t = df['time_elapsed'].to_numpy(dtype=float)
    p_raw = df[pelec_col].to_numpy(dtype=float)

    glo_peak = float(np.nanmax(p_raw))
    if not np.isfinite(glo_peak) or glo_peak <= 1e-9:
        return []

    dt_grid = float(_median_time_step_seconds(t))
    if not np.isfinite(dt_grid) or dt_grid <= 0:
        return []
    # Prefer physical long true-off gaps on raw Pelec (VarFl): 10 s exports use a short min_off;
    # 1 Hz logs need a higher floor so mid-test defrost-length offs (~2 min) do not become extra cycle
    # boundaries, while normal ~4.3 min inter-pulse gaps remain. Tie loosely to Min gap (seconds).
    if dt_grid >= 4.0:
        min_off_sec_use = 88.0
    else:
        min_off_sec_use = float(
            max(200.0, min(252.0, 0.42 * float(min_gap_s) + 12.0 * float(dt_grid)))
        )
    _lo = _detect_cycle_boundaries_long_off_runs(df_full, min_off_sec=min_off_sec_use)
    if len(_lo) >= 1:
        return _lo

    w_smooth = int(smooth_window) if smooth_window is not None else 3
    w_smooth = max(1, w_smooth)
    # Cap median window ~20% series length — keeps pulse edges from over-smoothing.
    w_smooth = min(w_smooth, max(3, len(p_raw) // 5))
    if w_smooth <= 1:
        pw = p_raw
    else:
        pw = pd.Series(p_raw).rolling(w_smooth, center=False, min_periods=1).median().to_numpy(dtype=float)

    p_lo_ref = float(np.nanpercentile(pw, 4))
    p_hi_ref = float(np.nanpercentile(pw, 88))
    if not np.isfinite(p_lo_ref):
        p_lo_ref = float(np.nanmin(pw))
    if not np.isfinite(p_hi_ref):
        p_hi_ref = glo_peak
    span = float(max(1e-6, p_hi_ref - p_lo_ref))
    frac_off_band = 0.28     # classic idle slack above Q₄‑based idle spine
    frac_on_band = 0.42      # left side sufficiently “on” vs idle percentile band

    # Representative on-load level for ΔP thresholds (don't key off lone spikes > Q₉₆).
    pk_scale = float(np.nanpercentile(pw, 90))
    if not np.isfinite(pk_scale) or pk_scale <= 1e-9:
        pk_scale = float(max(p_hi_ref, np.nanmedian(pw)))
    pk_scale = float(max(pk_scale, 1e-6))

    # Must step down materially; relative gate + percentile-based absolute scale (fixes 1 Hz ramps).
    min_drop_abs_kw = float(max(0.02, 0.03 * pk_scale))
    min_drop_rel = 0.05

    n = len(pw)
    # Wall-clock trough lookahead scales with Δt_grid: coarse (e.g. 10 s) logs need ≥ several minutes before
    # smoothed power reaches the idle/off band — a fixed „42 s / dt“ collapses to ~12 samples (≈2 min at 10 s)
    # and falsely suppresses shutdowns everywhere except the trace tail.
    if float(dt_grid) <= 2.5:
        horizon_sec = float(max(160.0, min(560.0, 220.0 + 52.0 * float(dt_grid))))
    else:
        horizon_sec = float(min(920.0, max(340.0, 36.5 * float(dt_grid))))
    fwd_horizon = int(
        max(12, min(max(3, n - 2), int(np.ceil(horizon_sec / max(dt_grid, 1e-9)))))
    )

    cand = []  # (t_shoulder=t[i-1], slope<0, drop_mag, p0, p_tail)
    for i in range(1, n):
        dt_step = float(t[i] - t[i - 1])
        if not np.isfinite(dt_step) or dt_step <= 1e-9:
            continue
        p0 = float(pw[i - 1])
        p1 = float(pw[i])
        if not (np.isfinite(p0) and np.isfinite(p1)):
            continue
        slope = (p1 - p0) / dt_step  # negative when falling
        if slope >= 0.0:
            continue

        fut = pw[i : min(n, i + fwd_horizon + 1)]
        trough_ahead = float(np.nanmin(fut)) if fut.size else float(p1)
        horizon_drop = float(p0 - trough_ahead)
        drop_inst = float(p0 - p1)
        substantial = float(max(drop_inst, horizon_drop))

        if substantial < max(min_drop_abs_kw, min_drop_rel * abs(p0)):
            continue

        idle_hard = float(p_lo_ref + frac_off_band * span)
        coarse_grid = float(dt_grid) >= 4.0  # omit permissive paths on dense 1 Hz traces

        # “On‑plateau” excludes the low standby plateau some VarFl rigs hold right before an off dip.
        on_plateau = float(p0) >= float(p_lo_ref + frac_on_band * span)
        standby_pre_off = (
            coarse_grid
            and (not on_plateau)
            and float(p0) >= float(p_lo_ref + 0.17 * span)
            and float(p0) < float(p_lo_ref + frac_on_band * span)
            and float(drop_inst) >= float(max(min_drop_abs_kw, 0.06 * abs(p0)))
            and float(p1) <= idle_hard
        )
        # A brief low hold immediately before a near-zero row misses both on_plateau and
        # standby_pre_off (a VarFl final pulse: the shoulder is the row before the near-zero sample).
        crisp_nearzero_off = (
            coarse_grid
            and (not on_plateau)
            and float(p0) < float(p_lo_ref + frac_on_band * span)
            and float(p0) >= float(p_lo_ref + 0.08 * span)
            and float(p1) <= 0.30
            and float(drop_inst) >= float(max(min_drop_abs_kw, 0.08 * abs(p0)))
        )

        if not (on_plateau or standby_pre_off or crisp_nearzero_off):
            continue

        heavy_on = float(p0) >= float(
            max(1e-6, pk_scale * 0.925, float(p_hi_ref) * 0.95, p_lo_ref + 0.55 * span))

        idle_soft_hi = float(p_lo_ref + 0.82 * span)  # long coast‑down tails on coarse grids (~3 kW)

        legacy_idle = float(trough_ahead) <= idle_hard or (
            coarse_grid and heavy_on and float(trough_ahead) <= idle_soft_hi
        )

        staged_plunge = (
            substantial >= float(max(min_drop_abs_kw, 0.09 * pk_scale))
            and float(trough_ahead) <= float(max(1e-6, 0.58 * abs(p0)))
            and float(trough_ahead) <= float(max(1e-6, 0.58 * abs(p_hi_ref)))
        )

        if not (legacy_idle or staged_plunge):
            continue

        cand.append((float(t[i - 1]), float(slope), substantial, float(p0), float(p1)))

    if len(cand) < 1:
        return []

    cand.sort(key=lambda z: z[0])

    # Cluster neighbouring shoulders (< min_gap) from one electrical event split across samples.
    # Also split when consecutive shoulders sit in a *moderate* gap band: long on-plateaus can emit no
    # candidates for many minutes (700–3000 s); widening clusters across that silence would pick a far
    # later cliff, but that silence is not two shutdowns — thresholds below (~100–420 s) catch pairs like
    # shoulder pairs only a few minutes apart without fragmenting steady-load segments.
    split_gap_lo = float(min(220.0, max(90.0, 0.04 * float(min_gap_s))))
    split_gap_hi = 420.0

    def _squash(cluster):
        if not cluster:
            return None
        # Negative slopes → “largest magnitude” is most-negative value.
        steep = min(z[1] for z in cluster)
        thresh = steep * 0.92  # ±8% slack in slope magnitude
        pool = [z for z in cluster if z[1] <= thresh]
        earliest = min(pool, key=lambda z: z[0])
        slope_hit = earliest[1]
        dm = earliest[2]
        return (earliest[0], slope_hit, dm, earliest[3])

    clustered = []
    cluster = [cand[0]]
    anchor_t = cand[0][0]
    mg = float(min_gap_s)

    for k in range(1, len(cand)):
        gap_prev = float(cand[k][0] - cluster[-1][0])
        gap_anchor = float(cand[k][0] - anchor_t)
        split_due_gap = split_gap_lo <= gap_prev <= split_gap_hi
        if gap_anchor <= mg and not split_due_gap:
            cluster.append(cand[k])
        else:
            clustered.append(_squash(cluster))
            cluster = [cand[k]]
            anchor_t = cand[k][0]
    clustered.append(_squash(cluster))
    clustered = [c for c in clustered if c is not None]

    pk_ref = float(np.nanpercentile(pw, 92))
    if not np.isfinite(pk_ref) or pk_ref <= 1e-9:
        pk_ref = glo_peak

    dedup_times = _boundary_times_one_step_earlier(t, [float(c[0]) for c in clustered], dt_grid)
    dedup_mags = [float(c[2]) for c in clustered]

    dedup_times, dedup_mags = _merge_consecutive_cycle_starts_if_gap_never_truly_off(
        t, p_raw, dedup_times, dedup_mags, p_lo_ref=p_lo_ref, span=span
    )

    if len(dedup_times) < 2:
        return []

    suggestions = []
    for i in range(len(dedup_times) - 1):
        st = dedup_times[i]
        nxt = dedup_times[i + 1]
        et = _last_time_on_axis_before(t, nxt, st)
        if et is None:
            step = dt_grid if (nxt - st) >= dt_grid else max(1.0, 0.5 * (nxt - st))
            et = float(nxt) - float(step)
        if et <= st:
            continue
        mag = dedup_mags[i]
        conf = float(min(1.0, max(0.0, mag / max(1e-6, 0.5 * pk_ref))))
        suggestions.append((st, et, conf, mag))
    return suggestions


def _detect_pelec_rampup(pelec_series, time_series):
    """Detect compressor ramp-up onset after a defrost / off period.

    Definition (matches manual labelling): the ramp-up onset is the *last time
    the smoothed power sits at its lowest level* before the sustained climb that
    follows.  After this point the power rises (possibly intermittently / in
    steps) and never returns to that low level within the cycle.  This is the
    deepest dip, not the point where the power first crosses some high
    threshold (which lands too late on slow / staged ramps).

    Returns the ``time_elapsed`` value at the onset, or None.
    Used for 'drop' (defrost_start) and 'off_start' cycles.
    """
    if pelec_series is None or len(pelec_series) < 10:
        return None

    p = pelec_series.values.astype(float)
    t = time_series.values.astype(float)

    # Smoothed power (rolling median, ~30s window ≈ 5 points at ~6s intervals)
    window = min(5, len(p) // 3)
    if window < 2:
        window = 2
    p_series = pd.Series(p)
    p_smooth = p_series.rolling(window, center=True, min_periods=1).median().values

    finite = p_smooth[np.isfinite(p_smooth)]
    if finite.size == 0:
        return None
    # Robust peak: a high percentile of the sustained (heating) level, so a brief
    # inrush / BUH spike cannot inflate the threshold and push the onset late.
    peak_power = float(np.nanpercentile(finite, 90))
    if not np.isfinite(peak_power) or peak_power <= 0:
        peak_power = float(np.nanmax(finite))
    base_power = float(np.nanmin(finite))
    span = peak_power - base_power
    if not np.isfinite(peak_power) or peak_power <= 0 or span <= 0:
        return None

    n = len(p_smooth)

    # Bound the search to the part before power becomes sustainably "on", so a
    # later defrost / shutdown near the cycle end cannot be mistaken for the dip.
    high_thr = base_power + 0.5 * span
    sustain_n = max(3, min(12, n // 20))
    above = p_smooth >= high_thr
    run = 0
    first_sustained = None
    for i in range(n):
        run = (run + 1) if above[i] else 0
        if run >= sustain_n:
            first_sustained = i - sustain_n + 1
            break
    search_end = first_sustained if first_sustained is not None else n
    if search_end < 1:
        search_end = n

    # "Low level" tolerance band around the global minimum.  Generous enough that
    # several dips at a similar low level count as the same level (so the *last*
    # of them wins), but tight enough to ignore intermediate plateaus.
    low_band = base_power + max(0.10 * span, 0.05 * peak_power)

    onset = None
    for i in range(search_end - 1, -1, -1):
        if np.isfinite(p_smooth[i]) and p_smooth[i] <= low_band:
            onset = i
            break

    if onset is None:
        # Fallback: global argmin within the search region.
        region = p_smooth[:search_end]
        if region.size == 0 or not np.any(np.isfinite(region)):
            return None
        onset = int(np.nanargmin(region))

    return float(t[onset])


def _detect_pelec_drop(pelec_series, time_series):
    """Detect the last significant power drop (compressor off / defrost onset).

    Returns the time_elapsed value at the drop onset, or None.
    Used for 'ramp' cycles to find when the next defrost starts near cycle end.
    """
    if pelec_series is None or len(pelec_series) < 10:
        return None

    p = pelec_series.values.astype(float)
    t = time_series.values.astype(float)

    window = min(5, len(p) // 3)
    if window < 2:
        window = 2
    p_series = pd.Series(p)
    p_smooth = p_series.rolling(window, center=True, min_periods=1).median().values

    finite = p_smooth[np.isfinite(p_smooth)]
    if finite.size == 0:
        return None
    # Robust peak (90th pct) so a brief spike doesn't distort the high/low bands.
    peak_power = float(np.nanpercentile(finite, 90))
    if not np.isfinite(peak_power) or peak_power <= 0:
        peak_power = float(np.nanmax(finite))
    if np.isnan(peak_power) or peak_power <= 0:
        return None

    # Goal: for "ramp" (defrost_end) cycles, find the onset of the next defrost/off
    # period near the cycle end. Return the *shoulder* — the last point still on the
    # high plateau, where power starts to drop suddenly — NOT the trough at the bottom
    # of the drop (the trough lands the marker too late; manual labelling uses the
    # start of the sudden drop).
    high_thr = 0.60 * peak_power
    low_thr = 0.20 * peak_power

    n = len(p_smooth)
    start_idx = int(max(0, n * 0.55))  # focus on later half
    best_shoulder = None
    best_drop = 0.0
    lookahead = max(3, min(15, n // 40))  # allow multi-step drops

    for i in range(start_idx, n - 1):
        if not np.isfinite(p_smooth[i]):
            continue
        if p_smooth[i] < high_thr:
            continue
        j2 = min(n, i + 1 + lookahead)
        seg = p_smooth[i + 1:j2]
        if seg.size == 0:
            continue
        jmin_rel = int(np.nanargmin(seg))
        jmin = i + 1 + jmin_rel
        if not np.isfinite(p_smooth[jmin]):
            continue
        if p_smooth[jmin] > low_thr:
            continue
        drop = p_smooth[i] - p_smooth[jmin]
        if drop > best_drop:
            best_drop = drop
            # Shoulder = last "on" point before the descent toward the trough.
            # Walk forward from i while power stays high; the first point that
            # falls below the high threshold marks the start of the sudden drop.
            shoulder = i
            for k in range(i, jmin):
                if np.isfinite(p_smooth[k]) and p_smooth[k] >= high_thr:
                    shoulder = k
                else:
                    break
            best_shoulder = shoulder

    if best_shoulder is not None:
        return float(t[best_shoulder])

    # fallback to old threshold-crossing logic (shoulder before first crossing)
    threshold = 0.35 * peak_power
    above_region_end = None
    for i in range(n - 1, -1, -1):
        if p_smooth[i] >= threshold:
            above_region_end = i
            break
    if above_region_end is None:
        return None
    for j in range(above_region_end, n):
        if p_smooth[j] < threshold:
            return float(t[max(0, j - 1)])
    return None


def _classify_entry(notes, test_cond):
    """Classify a single entry based on Notes and test_cond.

    Returns (cycle_type, cycle_start_marker) or (None, None) if unclassifiable.
    """
    tc = (test_cond or '').strip()
    notes_lower = (notes or '').lower().strip()

    is_defrost_cond = tc in DEFROST_TEST_CONDS
    is_onoff_cond = tc in ON_OFF_TEST_CONDS

    has_drop = 'drop' in notes_lower
    has_ramp = ('ramp' in notes_lower) and not has_drop

    if is_defrost_cond:
        if has_drop:
            return ('defrost_cycle', 'defrost_start')
        if has_ramp:
            return ('defrost_cycle', 'defrost_end')
        return ('other', 'other')

    if is_onoff_cond:
        if has_drop:
            return ('on_off_cycle', 'off_start')
        if has_ramp:
            return ('on_off_cycle', 'on_start')
        return ('other', 'other')

    return ('other', 'other')


def _generate_periods_for_entry(entry_rowid, cycle_start_marker,
                                cycle_start, cycle_end,
                                pelec_transition_time,
                                buffer_s: int = None):
    """Generate cycle_periods rows from classification + detected transition.

    Returns a list of dicts ready for INSERT into cycle_periods.
    """
    periods = []
    if pelec_transition_time is None:
        return periods
    if buffer_s is None:
        buffer_s = SETTLING_BUFFER_S

    if cycle_start_marker == 'defrost_start':
        d_end = pelec_transition_time + buffer_s
        if d_end >= cycle_end:
            # Entire cycle is defrost
            periods.append({
                'entry_rowid': entry_rowid,
                'period_type': 'defrost',
                'start_time': cycle_start,
                'end_time': cycle_end,
                'detection_method': 'auto_pelec',
            })
        else:
            periods.append({
                'entry_rowid': entry_rowid,
                'period_type': 'defrost',
                'start_time': cycle_start,
                'end_time': d_end,
                'detection_method': 'auto_pelec',
            })
            periods.append({
                'entry_rowid': entry_rowid,
                'period_type': 'heating',
                'start_time': d_end,
                'end_time': cycle_end,
                'detection_method': 'auto_pelec',
            })

    elif cycle_start_marker == 'defrost_end':
        d1_end = cycle_start + buffer_s
        drop_time = pelec_transition_time

        if d1_end >= drop_time:
            # No heating period; D1 merges into D2
            periods.append({
                'entry_rowid': entry_rowid,
                'period_type': 'defrost',
                'start_time': cycle_start,
                'end_time': cycle_end,
                'detection_method': 'auto_pelec',
            })
        else:
            periods.append({
                'entry_rowid': entry_rowid,
                'period_type': 'defrost',
                'start_time': cycle_start,
                'end_time': d1_end,
                'detection_method': 'auto_pelec',
            })
            periods.append({
                'entry_rowid': entry_rowid,
                'period_type': 'heating',
                'start_time': d1_end,
                'end_time': drop_time,
                'detection_method': 'auto_pelec',
            })
            periods.append({
                'entry_rowid': entry_rowid,
                'period_type': 'defrost',
                'start_time': drop_time,
                'end_time': cycle_end,
                'detection_method': 'auto_pelec',
            })

    elif cycle_start_marker == 'off_start':
        # On-off cycle starts in off period; transition is compressor on (ramp-up)
        trans = pelec_transition_time
        if trans <= cycle_start or trans >= cycle_end:
            return periods
        periods.append({
            'entry_rowid': entry_rowid,
            'period_type': 'off',
            'start_time': cycle_start,
            'end_time': trans,
            'detection_method': 'auto_pelec',
        })
        periods.append({
            'entry_rowid': entry_rowid,
            'period_type': 'on',
            'start_time': trans,
            'end_time': cycle_end,
            'detection_method': 'auto_pelec',
        })

    elif cycle_start_marker == 'on_start':
        # On-off cycle starts in on period; transition is compressor off (drop)
        trans = pelec_transition_time
        if trans <= cycle_start or trans >= cycle_end:
            return periods
        periods.append({
            'entry_rowid': entry_rowid,
            'period_type': 'on',
            'start_time': cycle_start,
            'end_time': trans,
            'detection_method': 'auto_pelec',
        })
        periods.append({
            'entry_rowid': entry_rowid,
            'period_type': 'off',
            'start_time': trans,
            'end_time': cycle_end,
            'detection_method': 'auto_pelec',
        })

    return periods


# ---------------------------------------------------------------------------
# Guideline clocks (Phase 1): D or S + H, plus equilibrium / evaluation inside H
# ---------------------------------------------------------------------------

GUIDELINE_DETECTION_METHOD = 'guideline'

# A detected drop this close to the first D/S end or to the parent end is not a
# second span: a window cut defrost start → next defrost start ends *at* the next
# drop, and that defrost belongs to the next parent row (§2.1.1).
GUIDELINE_MIN_SPAN_S = 60.0

# The only guideline kinds (§2.2.1). 'other' is never one of them: the old
# pipeline's cycle_type='other' means "could not classify", i.e. unknown.
GUIDELINE_KINDS = ('defrost', 'on_off', 'continuous')

# Stored cycle_type → kind. 'other' is deliberately absent.
GUIDELINE_KIND_FROM_CYCLE_TYPE = {'defrost_cycle': 'defrost', 'on_off_cycle': 'on_off'}

# Kinds whose H carries the equilibrium / evaluation clocks.
GUIDELINE_EQ_EVAL_KINDS = ('defrost', 'continuous')


def _normalise_guideline_kind(value):
    """One of ``GUIDELINE_KINDS`` for an analyst/UI value, else None.

    Never returns a kind for 'other' or for a test-condition letter — kind is not
    taken from E/A/B/C/D or from the dataset name (§2.2.1).
    """
    v = str(value or '').strip().lower().replace('-', '_').replace(' ', '_')
    if v in ('on_off', 'onoff'):
        return 'on_off'
    if v == 'defrost':
        return 'defrost'
    if v == 'continuous':
        return 'continuous'
    return None


def _resolve_guideline_kind(row_kind, stored_cycle_type, indicator_flag, indicator_col,
                            saved_types, default_kind):
    """(kind, source) in the fill order of §2.2.1.

    Analyst choice on the row, stored ``cycle_type``, defrost-indicator column,
    clocks already saved for the entry, then the **file-level default** — which
    therefore never overwrites a kind already set on a row. Anything left over is
    ``unknown``: in particular ``cycle_type='other'`` means "could not classify",
    not ``continuous``. Kind is never taken from a test-condition letter or from
    the dataset name.
    """
    kind = _normalise_guideline_kind(row_kind)
    if kind:
        return kind, 'analyst'
    kind = GUIDELINE_KIND_FROM_CYCLE_TYPE.get(stored_cycle_type)
    if kind:
        return kind, 'stored cycle_type'
    if indicator_flag is True:
        return 'defrost', f'indicator column "{indicator_col}"'
    if indicator_flag is False:
        return 'on_off', f'indicator column "{indicator_col}"'
    saved_types = set(saved_types or ())
    if 'defrost' in saved_types:
        return 'defrost', 'saved periods'
    if saved_types & {'off', 'on'}:
        return 'on_off', 'saved periods'
    if 'heating' in saved_types:
        return 'continuous', 'saved periods'
    default_kind = _normalise_guideline_kind(default_kind)
    if default_kind:
        return default_kind, 'file-level default'
    if stored_cycle_type == 'other':
        return 'unknown', "stored cycle_type 'other' — unclassified, choose one"
    return 'unknown', 'not determined — choose one'


def _guideline_period_labels(kind: str):
    """(pre, main) cycle_periods types for a kind, reusing the existing names."""
    if kind == 'on_off':
        return 'off', 'on'
    if kind == 'continuous':
        return None, 'heating'   # no D, no S — the parent window is H
    return 'defrost', 'heating'


def _guideline_buffer_s() -> int:
    return int(round(_guideline_lengths()['buffer_s']))


def _cycle_opens_low(pelec_values) -> bool:
    """True when the parent window opens near idle power (cycle begins in D or S).

    Notes-independent stand-in for the drop/ramp keyword: compare the power level
    at the very start of the window with the idle and on levels of that same
    window. Returns None when the trace is too short or too flat to judge.
    """
    if pelec_values is None or len(pelec_values) < 10:
        return None
    p = np.asarray(pelec_values, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < 10:
        return None
    base = float(np.nanpercentile(p, 5))
    peak = float(np.nanpercentile(p, 90))
    span = peak - base
    if not np.isfinite(span) or span <= 0:
        return None
    head_n = max(3, min(int(p.size // 20), 30))
    head = float(np.nanmedian(p[:head_n]))
    return bool(head < base + 0.4 * span)


def _defrost_indicator_state(df_slice, column_names):
    """(is_defrost, column) from a defrost-indicator column, if the sheet has one."""
    if df_slice is None or df_slice.empty or not column_names:
        return None, None
    lower = {str(c).strip().lower(): c for c in df_slice.columns}
    for want in column_names:
        col = lower.get(str(want).strip().lower())
        if col is None:
            continue
        vals = pd.to_numeric(df_slice[col], errors='coerce').dropna()
        if vals.empty:
            continue
        return bool((vals > 0).any()), str(col)
    return None, None


def _begins_in_from_marker(marker: str):
    """'ds' if the parent window opens inside D/S, 'h' if it opens inside H."""
    if marker in ('defrost_start', 'off_start'):
        return 'ds'
    if marker in ('defrost_end', 'on_start'):
        return 'h'
    return None


def _marker_from_begins_in(begins_in: str, kind: str) -> str:
    """cycle_start_marker equivalent, kept so stored markers stay readable."""
    if kind == 'on_off':
        return 'off_start' if begins_in == 'ds' else 'on_start'
    return 'defrost_start' if begins_in == 'ds' else 'defrost_end'


def _guideline_second_ds_start(override, next_drop, ds_end, cycle_end):
    """Start of the second D/S span (§2.1.1), or None when the cycle has one.

    The analyst comes first: ``ds2_start`` moves or adds the span, ``no_second_ds``
    removes it. Otherwise the detected drop counts only when it leaves both a
    usable H before it and a usable span after it.
    """
    if override.get('no_second_ds'):
        return None
    forced = override.get('ds2_start') is not None
    cand = override.get('ds2_start') if forced else next_drop
    if cand is None:
        return None
    try:
        cand = float(cand)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cand):
        return None
    cand = max(ds_end, min(cycle_end, cand))
    if forced:
        return cand if ds_end < cand < cycle_end else None
    if cand <= ds_end + GUIDELINE_MIN_SPAN_S or cand >= cycle_end - GUIDELINE_MIN_SPAN_S:
        return None
    return cand


def _build_guideline_clocks(cycle_start, cycle_end, kind, begins_in, trans,
                            lengths, override=None, next_drop=None):
    """Guideline intervals for one parent cycle: D or S, H, equilibrium, evaluation.

    ``begins_in`` is 'ds' (window opens inside defrost/standby, the transition is
    the power rise) or 'h' (window opens inside heating, the transition is the
    drop that ends it). ``next_drop`` is a further drop detected inside the same
    window; on a window that opens in D/S it turns D into two spans (§2.1.1).
    Both are ignored for ``continuous``, which has no D/S and needs no transition:
    the parent window is H and eq/eval run from its start.

    D/S and H follow the same buffer rule as ``_generate_periods_for_entry``,
    with the two additions the spec asks for: the buffer also applies to standby
    on on-off cycles, and the analyst may move the D/S end directly. Equilibrium
    and evaluation are clocks *inside* H, on defrost and continuous cycles only:
    the first ``eq_min`` of H, then the next ``eval_min`` — never the last N
    minutes of the cycle.

    Returns ``{'periods': [...], 'h_start', 'h_end', 'notes': [...]}``.
    """
    override = override or {}
    cycle_start = float(cycle_start)
    cycle_end = float(cycle_end)
    buffer_s = float(lengths['buffer_s'])
    eq_s = float(lengths['eq_s'])
    eval_s = float(lengths['eval_s'])
    pre_type, main_type = _guideline_period_labels(kind)

    notes = []
    spans = []
    h_start = h_end = None

    def _clamp(v):
        return max(cycle_start, min(cycle_end, float(v)))

    ds_end_override = override.get('ds_end')
    if kind == 'continuous':
        # No defrost, no standby: the parent window *is* H, so no power
        # transition is required and eq/eval start at the parent start (§2.2.1).
        h_start, h_end = cycle_start, cycle_end
        spans.append((main_type, h_start, h_end))
    elif trans is None and ds_end_override is None:
        notes.append('No interior power transition available — D/S and H cannot be proposed.')
        return {'periods': [], 'h_start': None, 'h_end': None, 'notes': notes}
    elif begins_in == 'ds':
        # Cycle opens inside D/S; the transition is the power rise (t_buf).
        ds_end = _clamp(ds_end_override if ds_end_override is not None else float(trans) + buffer_s)
        spans.append((pre_type, cycle_start, ds_end))
        if ds_end >= cycle_end:
            notes.append('Buffer end reaches the cycle end — no H interval left.')
        else:
            # A drop after that first span means the window was cut defrost end →
            # next defrost end, so D has a second span and H is only the gap
            # between them (§2.1.1). Without such a drop this stays one span.
            drop2 = _guideline_second_ds_start(override, next_drop, ds_end, cycle_end)
            h_start, h_end = ds_end, (drop2 if drop2 is not None else cycle_end)
            spans.append((main_type, h_start, h_end))
            if drop2 is not None:
                spans.append((pre_type, drop2, cycle_end))
                notes.append(
                    f'Drop at {drop2:.0f} s inside the window — '
                    f'{"D" if pre_type == "defrost" else "S"} is two spans, H is the gap between them.'
                )
    else:
        # Cycle opens inside H; the transition is the drop that ends it. The
        # first buffer_min still belongs to the previous D/S.
        ds1_end = _clamp(ds_end_override if ds_end_override is not None else cycle_start + buffer_s)
        drop = _clamp(trans) if trans is not None else cycle_end
        if ds1_end >= drop:
            spans.append((pre_type, cycle_start, cycle_end))
            notes.append('Buffer end is at or after the power drop — the whole cycle is D/S.')
        else:
            spans.append((pre_type, cycle_start, ds1_end))
            h_start, h_end = ds1_end, drop
            spans.append((main_type, h_start, h_end))
            if drop < cycle_end:
                spans.append((pre_type, drop, cycle_end))

    eq_win = None
    eval_win = None
    eq_locked = bool(override.get('eq_locked', True))
    eval_locked = bool(override.get('eval_locked', True))

    if kind not in GUIDELINE_EQ_EVAL_KINDS:
        if kind == 'on_off' and h_start is not None:
            notes.append('Equilibrium / evaluation do not apply to on–off cycles.')
    elif h_start is not None:
        h_len = h_end - h_start
        if (not eq_locked) and override.get('eq_start') is not None and override.get('eq_end') is not None:
            eq_win = (_clamp(override['eq_start']), _clamp(override['eq_end']))
        elif eq_s <= 0:
            notes.append('eq_min is 0 — no equilibrium window.')
        elif h_len + 1e-6 >= eq_s:
            eq_win = (h_start, h_start + eq_s)
        else:
            notes.append(
                f'H is {h_len / 60.0:.0f} min, shorter than eq_min '
                f'({eq_s / 60.0:.0f} min) — no equilibrium, no evaluation.'
            )

        if (not eval_locked) and override.get('eval_start') is not None and override.get('eval_end') is not None:
            eval_win = (_clamp(override['eval_start']), _clamp(override['eval_end']))
        elif eq_win is None:
            pass  # reason already reported above
        elif eval_s <= 0:
            notes.append('eval_min is 0 — no evaluation window.')
        elif eq_win[1] + eval_s <= h_end + 1e-6:
            eval_win = (eq_win[1], eq_win[1] + eval_s)
        else:
            notes.append(
                f'H is {h_len / 60.0:.0f} min, shorter than eq_min + eval_min '
                f'({(eq_s + eval_s) / 60.0:.0f} min) — evaluation omitted (interrupted / transient).'
            )

    # Order matters: D/S and H first, so readers that look for a defrost→heating
    # boundary in start-time order still find it.
    if eq_win:
        spans.append(('equilibrium', eq_win[0], eq_win[1]))
    if eval_win:
        spans.append(('evaluation', eval_win[0], eval_win[1]))

    periods = [
        {'period_type': t, 'start_time': float(s), 'end_time': float(e)}
        for (t, s, e) in spans if float(e) > float(s)
    ]
    return {'periods': periods, 'h_start': h_start, 'h_end': h_end, 'notes': notes}


@app.route('/api/cycle_periods/auto_classify', methods=['POST'])
def api_cycle_periods_auto_classify():
    """Classify entries and detect Pelec transitions, then generate cycle_periods."""
    # Deprecated: prefer /api/pipeline/run which also caches multi-buffer periods + full-cycle cache
    ensure_cycle_periods_table()
    init_deviation_columns()

    payload = request.get_json() or {}
    force_all = payload.get('force_all', False)
    try:
        buffer_s = int(payload.get('buffer_s', SETTLING_BUFFER_S))
    except Exception:
        buffer_s = SETTLING_BUFFER_S
    buffer_s = max(60, min(3600, buffer_s))

    conn = get_db_connection()
    try:
        if force_all:
            rows = conn.execute(
                "SELECT rowid, file_name, data_set, start_time, end_time, "
                "Notes, Indicator, test_cond "
                "FROM results ORDER BY file_name, data_set, rowid"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT rowid, file_name, data_set, start_time, end_time, "
                "Notes, Indicator, test_cond "
                "FROM results WHERE cycle_type IS NULL "
                "ORDER BY file_name, data_set, rowid"
            ).fetchall()
        rows = [dict(r) for r in rows]
    finally:
        conn.close()

    total = len(rows)
    if total == 0:
        return jsonify({'success': True, 'classified': 0, 'detected': 0,
                        'failed_detection': 0, 'skipped': 0, 'total': 0,
                        'message': 'All entries already classified.'})

    from collections import OrderedDict
    import time as _time
    t0 = _time.time()

    MAX_CACHE = 40
    _excel_cache = OrderedDict()

    def _cached_read(fname, dset):
        key = (fname, dset)
        if key in _excel_cache:
            _excel_cache.move_to_end(key)
            return _excel_cache[key]
        df = _read_excel_sheet(fname, dset)
        _excel_cache[key] = df
        if len(_excel_cache) > MAX_CACHE:
            _excel_cache.popitem(last=False)
        return df

    classified = 0
    detected = 0
    failed_detection = 0
    skipped = 0

    conn = get_db_connection()
    try:
        for idx, r in enumerate(rows):
            rid = r['rowid']
            notes = r.get('Notes') or r.get('notes') or ''
            test_cond = r.get('test_cond') or ''
            fn = r.get('file_name')
            ds = r.get('data_set')
            st_val = r.get('start_time')
            et_val = r.get('end_time')

            cycle_type, start_marker = _classify_entry(notes, test_cond)

            if cycle_type is None or cycle_type == 'other':
                conn.execute(
                    "UPDATE results SET cycle_type=?, cycle_start_marker=? WHERE rowid=?",
                    (cycle_type or 'other', start_marker or 'other', rid))
                skipped += 1
                continue

            classified += 1
            conn.execute(
                "UPDATE results SET cycle_type=?, cycle_start_marker=? WHERE rowid=?",
                (cycle_type, start_marker, rid))

            # Only attempt Pelec detection for defrost_cycle entries
            if cycle_type != 'defrost_cycle':
                continue

            try:
                stf, etf = float(st_val), float(et_val)
            except (TypeError, ValueError):
                failed_detection += 1
                continue

            df_full = _cached_read(fn, ds)
            if df_full is None or 'time_elapsed' not in df_full.columns:
                failed_detection += 1
                continue
            if PELEC_COL not in df_full.columns:
                failed_detection += 1
                continue

            df_slice = df_full[
                (df_full['time_elapsed'] >= stf) & (df_full['time_elapsed'] <= etf)
            ].copy()
            if df_slice.empty or len(df_slice) < 10:
                failed_detection += 1
                continue

            pelec = df_slice[PELEC_COL].dropna()
            t_elapsed = df_slice.loc[pelec.index, 'time_elapsed']

            if len(pelec) < 10:
                failed_detection += 1
                continue

            trans_time = None
            if start_marker == 'defrost_start':
                trans_time = _detect_pelec_rampup(pelec, t_elapsed)
            elif start_marker == 'defrost_end':
                trans_time = _detect_pelec_drop(pelec, t_elapsed)

            if trans_time is None:
                failed_detection += 1
                conn.execute(
                    "UPDATE results SET pelec_transition_time=NULL WHERE rowid=?",
                    (rid,))
                continue

            detected += 1
            conn.execute(
                "UPDATE results SET pelec_transition_time=? WHERE rowid=?",
                (trans_time, rid))

            # Remove old auto-generated periods for this entry
            conn.execute(
                "DELETE FROM cycle_periods WHERE entry_rowid=? AND detection_method='auto_pelec'",
                (rid,))

            periods = _generate_periods_for_entry(rid, start_marker, stf, etf, trans_time, buffer_s=buffer_s)
            for p in periods:
                conn.execute(
                    "INSERT INTO cycle_periods "
                    "(entry_rowid, period_type, start_time, end_time, detection_method, buffer_s) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (p['entry_rowid'], p['period_type'],
                     p['start_time'], p['end_time'], p['detection_method'], buffer_s))

            if (idx + 1) % 50 == 0:
                conn.commit()
                elapsed = _time.time() - t0
                print(f"[auto_classify] {idx+1}/{total}  "
                      f"classified={classified} detected={detected} "
                      f"failed={failed_detection} ({elapsed:.1f}s)", flush=True)

        conn.commit()
    finally:
        conn.close()

    elapsed = _time.time() - t0
    msg = (f"Classified {classified}, detected {detected} Pelec transitions, "
           f"{failed_detection} detection failures, {skipped} skipped "
           f"in {elapsed:.1f}s")
    print(f"[auto_classify] DONE: {msg}", flush=True)
    return jsonify({
        'success': True,
        'classified': classified,
        'detected': detected,
        'failed_detection': failed_detection,
        'skipped': skipped,
        'total': total,
        'message': msg,
        'deprecated': True,
        'use_instead': '/api/pipeline/run',
    })


@app.route('/api/cycle_periods/regenerate', methods=['POST'])
def api_cycle_periods_regenerate():
    """Regenerate period boundaries for entries that have pelec_transition_time set.

    Useful after manually editing pelec_transition_time or changing SETTLING_BUFFER_S.
    """
    ensure_cycle_periods_table()
    init_deviation_columns()

    payload = request.get_json() or {}
    entry_rowids = payload.get('entry_rowids', None)
    try:
        buffer_s = int(payload.get('buffer_s', SETTLING_BUFFER_S))
    except Exception:
        buffer_s = SETTLING_BUFFER_S
    buffer_s = max(60, min(3600, buffer_s))

    conn = get_db_connection()
    try:
        if entry_rowids:
            placeholders = ','.join('?' * len(entry_rowids))
            rows = conn.execute(
                f"SELECT rowid, start_time, end_time, cycle_start_marker, pelec_transition_time "
                f"FROM results WHERE rowid IN ({placeholders}) "
                f"AND pelec_transition_time IS NOT NULL",
                entry_rowids
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT rowid, start_time, end_time, cycle_start_marker, pelec_transition_time "
                "FROM results WHERE pelec_transition_time IS NOT NULL"
            ).fetchall()
        rows = [dict(r) for r in rows]

        regenerated = 0
        for r in rows:
            rid = r['rowid']
            conn.execute(
                "DELETE FROM cycle_periods WHERE entry_rowid=? AND detection_method='auto_pelec'",
                (rid,))
            periods = _generate_periods_for_entry(
                rid, r['cycle_start_marker'],
                float(r['start_time']), float(r['end_time']),
                float(r['pelec_transition_time']),
                buffer_s=buffer_s)
            for p in periods:
                conn.execute(
                    "INSERT INTO cycle_periods "
                    "(entry_rowid, period_type, start_time, end_time, detection_method, buffer_s) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (p['entry_rowid'], p['period_type'],
                     p['start_time'], p['end_time'], p['detection_method'], buffer_s))
            regenerated += 1
        conn.commit()
    finally:
        conn.close()

    return jsonify({'success': True, 'regenerated': regenerated, 'deprecated': True, 'use_instead': '/api/pipeline/run'})


@app.route('/api/cycle_periods/cache_refresh', methods=['POST'])
def api_cycle_periods_cache_refresh():
    """Compute and store per-period dTreturn caches for cycle_periods rows that lack them."""
    # Deprecated: prefer /api/pipeline/run which caches all buffers in one pass
    ensure_cycle_periods_table()
    init_deviation_columns()

    payload = request.get_json() or {}
    force_all = payload.get('force_all', False)

    conn = get_db_connection()
    try:
        if force_all:
            period_rows = conn.execute(
                "SELECT cp.id, cp.entry_rowid, cp.start_time, cp.end_time, "
                "r.file_name, r.data_set "
                "FROM cycle_periods cp JOIN results r ON cp.entry_rowid = r.rowid "
                "ORDER BY r.file_name, r.data_set, cp.id"
            ).fetchall()
        else:
            period_rows = conn.execute(
                "SELECT cp.id, cp.entry_rowid, cp.start_time, cp.end_time, "
                "r.file_name, r.data_set "
                "FROM cycle_periods cp JOIN results r ON cp.entry_rowid = r.rowid "
                "WHERE cp.dtreturn_n_valid IS NULL "
                "   OR (cp.dtreturn_n_valid > 0 AND cp.dtreturn_min IS NULL) "
                "   OR (cp.dtreturn_n_valid > 0 AND cp.avg_t_db IS NULL) "
                "ORDER BY r.file_name, r.data_set, cp.id"
            ).fetchall()
        period_rows = [dict(r) for r in period_rows]
    finally:
        conn.close()

    total = len(period_rows)
    if total == 0:
        return jsonify({'success': True, 'processed': 0, 'total': 0,
                        'message': 'All periods already cached.'})

    import time as _time
    import random
    from collections import OrderedDict
    t0 = _time.time()

    MAX_CACHE = 40
    _excel_cache = OrderedDict()

    def _cached_read(fname, dset):
        key = (fname, dset)
        if key in _excel_cache:
            _excel_cache.move_to_end(key)
            return _excel_cache[key]
        df = _read_excel_sheet(fname, dset)
        _excel_cache[key] = df
        if len(_excel_cache) > MAX_CACHE:
            _excel_cache.popitem(last=False)
        return df

    processed = 0
    skipped = 0
    max_sample = 5000

    conn = get_db_connection()
    try:
        for idx, pr in enumerate(period_rows):
            pid = pr['id']
            fn = pr['file_name']
            ds = pr['data_set']
            p_st = pr['start_time']
            p_et = pr['end_time']

            if not fn or p_st is None or p_et is None:
                skipped += 1
                conn.execute(
                    "UPDATE cycle_periods SET dtreturn_n_valid=0, dtreturn_n_nan=0, "
                    "dtreturn_sample='[]' WHERE id=?", (pid,))
                continue

            df_full = _cached_read(fn, ds)
            if df_full is None or 'time_elapsed' not in df_full.columns:
                skipped += 1
                conn.execute(
                    "UPDATE cycle_periods SET dtreturn_n_valid=0, dtreturn_n_nan=0, "
                    "dtreturn_sample='[]' WHERE id=?", (pid,))
                continue

            try:
                stf, etf = float(p_st), float(p_et)
            except (TypeError, ValueError):
                skipped += 1
                continue

            _refresh_cycle_period_row_stats(conn, pid, df_full, stf, etf)
            processed += 1

            if (idx + 1) % 50 == 0:
                conn.commit()
                elapsed = _time.time() - t0
                print(f"[period_cache] {idx+1}/{total} processed={processed} "
                      f"({elapsed:.1f}s)", flush=True)

        conn.commit()
    finally:
        conn.close()

    elapsed = _time.time() - t0
    msg = f"Cached {processed} periods, skipped {skipped} in {elapsed:.1f}s"
    print(f"[period_cache] DONE: {msg}", flush=True)
    return jsonify({
        'success': True,
        'processed': processed,
        'skipped': skipped,
        'total': total,
        'message': msg,
        'deprecated': True,
        'use_instead': '/api/pipeline/run',
    })


# ---------------------------------------------------------------------------
# Phase 2: automated cycle extraction (drop→drop) with review/approval
# ---------------------------------------------------------------------------

@app.route('/cycle_extract', methods=['GET'])
def cycle_extract():
    """Review/approve auto-suggested cycles for insertion into results."""
    ensure_cycle_extraction_suggestions_table()
    # list available excel files from data directory
    try:
        files = sorted([f for f in os.listdir(config['data_dir']) if f.lower().endswith('.xlsx')])
    except Exception:
        files = []

    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT id, file_name, data_set, start_time, end_time, confidence, notes_auto, pelec_drop_mag, created_at "
            "FROM cycle_extraction_suggestions ORDER BY created_at DESC, file_name ASC, data_set ASC, start_time ASC "
            "LIMIT 500"
        ).fetchall()
        suggestions = [dict(r) for r in rows]
    finally:
        conn.close()

    return render_template(
        'cycle_extract.html',
        excel_files=files,
        suggestions=suggestions,
        profiles=_profile_picker_items(),
    )


@app.route('/api/cycle_extract/suggest', methods=['POST'])
def api_cycle_extract_suggest():
    """Generate and store cycle suggestions for a given file/dataset."""
    ensure_cycle_extraction_suggestions_table()
    try:
        payload = request.get_json() or {}
        file_name = (payload.get('file_name') or '').strip()
        data_set = payload.get('data_set', None)
        try:
            data_set = float(data_set) if data_set is not None and str(data_set).strip() != '' else None
        except Exception:
            data_set = None
        try:
            min_gap_s = float(payload.get('min_gap_s', 900.0))
        except Exception:
            min_gap_s = 900.0
        min_gap_s = max(60.0, min(7200.0, min_gap_s))

        if not file_name:
            return jsonify({'success': False, 'message': 'file_name required'})

        df_full = _read_excel_sheet(file_name, data_set)
        if df_full is None:
            return jsonify({'success': False, 'message': f'Cannot read {file_name}'})

        sugg = _detect_cycle_boundaries_drop_to_drop(df_full, min_gap_s=min_gap_s)
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM cycle_extraction_suggestions WHERE file_name=? AND (data_set IS ? OR data_set=?)",
                         (file_name, data_set, data_set if data_set is not None else None))
            for (st, et, conf, mag) in sugg:
                conn.execute(
                    "INSERT INTO cycle_extraction_suggestions(file_name, data_set, start_time, end_time, marker_type, confidence, notes_auto, pelec_drop_mag) "
                    "VALUES (?, ?, ?, ?, 'drop', ?, ?, ?)",
                    (file_name, data_set, float(st), float(et), float(conf), 'auto drop→drop', float(mag))
                )
            conn.commit()
        finally:
            conn.close()

        return jsonify({'success': True, 'count': len(sugg)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_extract/load_data', methods=['POST'])
def api_cycle_extract_load_data():
    """Return all numeric time-series columns from the Excel sheet as JSON for Plotly.

    For files up to ~20 000 rows the full data is sent.
    Larger files are downsampled evenly.  Result is LRU-cached so reloads are instant.
    """
    try:
        payload = request.get_json() or {}
        file_name = (payload.get('file_name') or '').strip()
        data_set = payload.get('data_set', None)
        try:
            data_set = float(data_set) if data_set is not None and str(data_set).strip() != '' else None
        except Exception:
            data_set = None
        max_points = int(payload.get('max_points', 20000))

        if not file_name:
            return jsonify({'success': False, 'message': 'file_name required'})

        # Read full sheet (LRU cached, so subsequent loads are instant)
        df_full = _read_excel_sheet(file_name, data_set)
        if df_full is None or 'time_elapsed' not in df_full.columns:
            return jsonify({'success': False, 'message': f'Cannot read {file_name}'})

        # Coerce time axis to numeric (some files may read as object)
        df_work = df_full.copy()
        df_work['time_elapsed'] = pd.to_numeric(df_work['time_elapsed'], errors='coerce')
        df_work = df_work.dropna(subset=['time_elapsed'])

        # Numeric columns: robust (includes int/float; avoids empty col_list surprises)
        numeric_cols = []
        for c in df_work.columns:
            if c == 'time_elapsed':
                continue
            if str(c).strip().lower().startswith('unnamed'):
                continue
            if pd.api.types.is_numeric_dtype(df_work[c]):
                numeric_cols.append(c)
            else:
                coerced = pd.to_numeric(df_work[c], errors='coerce')
                if coerced.notna().sum() >= max(30, int(0.05 * len(df_work))):
                    df_work[c] = coerced
                    numeric_cols.append(c)

        # Derived dTreturn
        has_dtreturn = 'T_return_emu' in df_work.columns and 'T_return_calc' in df_work.columns
        if has_dtreturn:
            te = pd.to_numeric(df_work['T_return_emu'], errors='coerce')
            tc = pd.to_numeric(df_work['T_return_calc'], errors='coerce')
            df_work['dTreturn'] = te - tc
            if 'dTreturn' not in numeric_cols:
                numeric_cols.append('dTreturn')

        # Downsample if needed
        n = len(df_work)
        if n > max_points:
            step = max(1, n // max_points)
            df_out = df_work.iloc[::step]
        else:
            df_out = df_work

        import math

        def _safe_key(name: str) -> str:
            return (
                str(name)
                .replace(' ', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('/', '_')
                .replace('.', '_')
            )

        def _safe_list(series):
            return [
                None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                for v in series.tolist()
            ]

        data = {'time': _safe_list(df_out['time_elapsed'])}
        col_list = []
        for c in numeric_cols:
            sk = _safe_key(c)
            data[sk] = _safe_list(df_out[c])
            col_list.append({'key': sk, 'label': c})

        # Stable short aliases (works even if front-end column metadata is missing / cached old)
        if PELEC_COL in df_out.columns:
            data['pelec'] = _safe_list(df_out[PELEC_COL])
        if has_dtreturn:
            data['dtreturn'] = _safe_list(df_out['dTreturn'])
        hc_key = None
        for cand in ('Heating Capacity (without corr)',):
            if cand in df_out.columns:
                hc_key = _safe_key(cand)
                data['heating_uncorr'] = _safe_list(df_out[cand])
                break

        aliases = [{'key': 'pelec', 'label': PELEC_COL}]
        if has_dtreturn:
            aliases.append({'key': 'dtreturn', 'label': 'dTreturn'})
        if hc_key:
            aliases.append({'key': 'heating_uncorr', 'label': 'Heating Capacity (without corr)'})

        seen_keys = {c['key'] for c in col_list}
        for a in aliases:
            if a['key'] not in seen_keys:
                col_list.append(a)
                seen_keys.add(a['key'])

        return jsonify({
            'success': True,
            'data': data,
            'columns': col_list,
            'total_points': int(n),
            'returned_points': int(len(df_out)),
            'has_dtreturn': has_dtreturn,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_extract/existing_entries', methods=['POST'])
def api_cycle_extract_existing_entries():
    """Return existing DB entries for a given file (optionally filtered by data_set)."""
    try:
        payload = request.get_json() or {}
        file_name = (payload.get('file_name') or '').strip()
        data_set = payload.get('data_set', None)
        try:
            data_set = float(data_set) if data_set is not None and str(data_set).strip() != '' else None
        except Exception:
            data_set = None

        if not file_name:
            return jsonify({'success': True, 'entries': []})

        cols = (
            "SELECT rowid, file_name, data_set, start_time, end_time, "
            "HP_ID AS hp_id, dev_test_condition, indicator AS indicator_notes, notes, "
            "cycle_type, cycle_start_marker, pelec_transition_time, pelec_transition_source "
            "FROM results "
        )
        conn = get_db_connection()
        try:
            if data_set is not None:
                rows = conn.execute(
                    cols + "WHERE file_name=? AND data_set=? ORDER BY start_time ASC",
                    (file_name, data_set)
                ).fetchall()
            else:
                rows = conn.execute(
                    cols + "WHERE file_name=? ORDER BY data_set ASC, start_time ASC",
                    (file_name,)
                ).fetchall()
        finally:
            conn.close()
        return jsonify({'success': True, 'entries': [dict(r) for r in rows]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


def _guideline_pelec_for_window(df_full, stf, etf):
    """(slice, power, time) for one parent window, or (slice, None, None)."""
    if df_full is None or 'time_elapsed' not in df_full.columns:
        return None, None, None
    t_all = pd.to_numeric(df_full['time_elapsed'], errors='coerce')
    df_slice = df_full[(t_all >= stf) & (t_all <= etf)]
    if df_slice.empty:
        return None, None, None
    pcol = _find_uncorrected_pelec_column(df_slice)
    if pcol is None:
        return df_slice, None, None
    pelec = pd.to_numeric(df_slice[pcol], errors='coerce').dropna()
    if len(pelec) < 10:
        return df_slice, None, None
    return df_slice, pelec, pd.to_numeric(df_slice.loc[pelec.index, 'time_elapsed'], errors='coerce')


def _detect_next_drop_after(pelec, p_time, after_t):
    """Power drop later in the same parent window — start of a second D/S span.

    Only the trace after the first D/S end is searched, so the rise that opened
    the window cannot be mistaken for the next defrost.
    """
    if pelec is None or p_time is None or after_t is None:
        return None
    tail = p_time > float(after_t)
    if int(tail.sum()) < 10:
        return None
    return _detect_pelec_drop(pelec[tail], p_time[tail])


def _guideline_proposals_for_file(file_name, want_rowids=None, overrides=None, default_kind=None):
    """Derive the guideline clocks for the DB entries of one file.

    Shared by ``propose_periods`` (read-only) and ``save_all_periods``, so both
    see exactly the same clocks. ``overrides[rowid]`` replays the analyst's edits
    and ``default_kind`` is the file-level "treat unknown rows as" choice, which
    fills only rows whose kind is still unknown (§2.2.1).

    Returns ``{'entries': [...], 'lengths': {...}, 'buffer_s': int}``. Only this
    file's entries are read — no full-database walk.
    """
    overrides = overrides if isinstance(overrides, dict) else {}
    default_kind = _normalise_guideline_kind(default_kind)

    cfg = load_cycle_periods_config()
    lengths = _guideline_lengths()
    buffer_s = _guideline_buffer_s()

    conn = get_db_connection()
    try:
        entries = [dict(r) for r in conn.execute(
            "SELECT rowid, data_set, start_time, end_time, cycle_type, cycle_start_marker, "
            "pelec_transition_time, pelec_transition_source "
            "FROM results WHERE file_name=? ORDER BY data_set ASC, start_time ASC",
            (file_name,)
        ).fetchall()]
        if isinstance(want_rowids, list) and want_rowids:
            keep = {int(x) for x in want_rowids}
            entries = [e for e in entries if int(e['rowid']) in keep]
        saved_map = {}
        if entries:
            ph = ",".join(["?"] * len(entries))
            rows = conn.execute(
                f"SELECT entry_rowid, period_type, start_time, end_time FROM cycle_periods "
                f"WHERE entry_rowid IN ({ph}) AND buffer_s=? AND detection_method=? "
                f"ORDER BY start_time",
                [int(e['rowid']) for e in entries] + [buffer_s, GUIDELINE_DETECTION_METHOD]
            ).fetchall()
            for p in rows:
                saved_map.setdefault(int(p['entry_rowid']), []).append(dict(p))
    finally:
        conn.close()

    sheets = {}   # one Excel read per data_set (sheet), LRU-cached underneath
    out = []
    for e in entries:
        rid = int(e['rowid'])
        if e.get('start_time') is None or e.get('end_time') is None:
            continue
        stf, etf = float(e['start_time']), float(e['end_time'])
        # NaN would be written to the reply as the literal `NaN`, which no browser
        # can parse — such a row has no usable window anyway.
        if not (np.isfinite(stf) and np.isfinite(etf)):
            continue
        ov = overrides.get(str(rid)) or overrides.get(rid) or {}
        saved = saved_map.get(rid, [])

        ds_key = e.get('data_set')
        if ds_key not in sheets:
            sheets[ds_key] = _read_excel_sheet(file_name, ds_key)
        df_slice, pelec, p_time = _guideline_pelec_for_window(sheets[ds_key], stf, etf)

        # --- kind (§2.2.1): analyst, stored cycle_type, indicator column,
        # already-saved clocks, then the file-level default. ---
        flag, ind_col = (None, None)
        if _normalise_guideline_kind(ov.get('kind')) is None \
                and e.get('cycle_type') not in GUIDELINE_KIND_FROM_CYCLE_TYPE:
            flag, ind_col = _defrost_indicator_state(df_slice, cfg.get('defrost_indicator_columns'))
        kind, kind_source = _resolve_guideline_kind(
            ov.get('kind'), e.get('cycle_type'), flag, ind_col,
            {p['period_type'] for p in saved}, default_kind,
        )

        # --- where the window opens: D/S or H (continuous has neither) ---
        begins_override = str(ov.get('begins_in') or '').strip().lower()
        if kind == 'continuous':
            begins_in, begins_source = None, 'not used — the parent window is H'
        elif begins_override in ('ds', 'h'):
            begins_in, begins_source = begins_override, 'analyst'
        elif _begins_in_from_marker(e.get('cycle_start_marker')):
            begins_in = _begins_in_from_marker(e.get('cycle_start_marker'))
            begins_source = 'stored cycle_start_marker'
        else:
            opens_low = _cycle_opens_low(pelec.values) if pelec is not None else None
            if opens_low is None:
                begins_in, begins_source = 'ds', 'default (power level unreadable)'
            else:
                begins_in = 'ds' if opens_low else 'h'
                begins_source = 'power level at cycle start'

        # --- interior power event (drop or rise). Not a kind classifier. ---
        try:
            trans_override = float(ov['transition_time'])
        except (KeyError, TypeError, ValueError):
            trans_override = None
        if kind == 'continuous':
            trans, trans_source = None, 'not required — no D/S on a continuous cycle'
        elif trans_override is not None:
            trans, trans_source = trans_override, 'analyst'
        elif e.get('pelec_transition_time') is not None:
            trans = float(e['pelec_transition_time'])
            trans_source = e.get('pelec_transition_source') or 'auto'
        elif pelec is not None:
            detector = _detect_pelec_rampup if begins_in == 'ds' else _detect_pelec_drop
            trans = detector(pelec, p_time)
            trans_source = 'detected now' if trans is not None else 'detection failed'
        else:
            trans, trans_source = None, 'no power data for this window'
        if trans is not None and not np.isfinite(trans):
            trans, trans_source = None, 'transition time is not a finite number'

        def _num(key, _ov=ov):
            try:
                return float(_ov[key])
            except (KeyError, TypeError, ValueError):
                return None

        # --- second D/S span: a further drop inside the same window (§2.1.1) ---
        next_drop, next_drop_source = None, None
        if kind != 'continuous' and begins_in == 'ds':
            if _num('ds2_start') is not None:
                next_drop_source = 'analyst'
            elif ov.get('no_second_ds'):
                next_drop_source = 'removed by the analyst'
            else:
                ds_end_guess = _num('ds_end')
                if ds_end_guess is None and trans is not None:
                    ds_end_guess = float(trans) + lengths['buffer_s']
                if ds_end_guess is not None and pelec is not None:
                    next_drop = _detect_next_drop_after(pelec, p_time, min(ds_end_guess, etf))
                next_drop_source = 'detected now' if next_drop is not None else 'no later drop found'

        clocks = _build_guideline_clocks(
            stf, etf, kind, begins_in, trans, lengths,
            override={
                'ds_end': _num('ds_end'),
                'ds2_start': _num('ds2_start'),
                'no_second_ds': bool(ov.get('no_second_ds')),
                'eq_locked': ov.get('eq_locked', True),
                'eval_locked': ov.get('eval_locked', True),
                'eq_start': _num('eq_start'),
                'eq_end': _num('eq_end'),
                'eval_start': _num('eval_start'),
                'eval_end': _num('eval_end'),
            },
            next_drop=next_drop,
        )
        notes = list(clocks['notes'])
        if kind == 'unknown':
            notes.append('Kind unknown — pick defrost, on–off or continuous, '
                         'or set the file-level default, before saving.')

        out.append({
            'rowid': rid,
            'data_set': e.get('data_set'),
            'start_time': stf,
            'end_time': etf,
            'kind': kind,
            'kind_source': kind_source,
            'begins_in': begins_in,
            'begins_in_source': begins_source,
            'transition_time': trans,
            'transition_source': trans_source,
            'second_drop': next_drop,
            'second_drop_source': next_drop_source,
            'periods': clocks['periods'],
            'h_start': clocks['h_start'],
            'h_end': clocks['h_end'],
            'notes': notes,
            'saved_periods': saved,
            'has_saved': bool(saved),
        })

    return {'entries': out, 'lengths': lengths, 'buffer_s': buffer_s}


@app.route('/api/cycle_extract/propose_periods', methods=['POST'])
def api_cycle_extract_propose_periods():
    """Guideline clocks for the existing DB entries of the currently open file.

    Read-only proposal. ``overrides[rowid]`` lets the analyst move the
    transition, the D/S end, the kind, or (once unlocked) the eq/eval times and
    re-derive the rest with the same server-side rule, so the clock logic lives
    in one place. ``default_kind`` is the file-level "treat unknown rows as"
    choice. Nothing is written here — clocks are stored only on save (§2.2.2).
    """
    try:
        ensure_cycle_periods_table()
        payload = request.get_json() or {}
        file_name = (payload.get('file_name') or '').strip()
        if not file_name:
            return jsonify({'success': False, 'message': 'file_name required'})
        result = _guideline_proposals_for_file(
            file_name,
            want_rowids=payload.get('rowids'),
            overrides=payload.get('overrides'),
            default_kind=payload.get('default_kind'),
        )
        return jsonify({
            'success': True,
            'entries': result['entries'],
            'lengths': result['lengths'],
            'buffer_s': result['buffer_s'],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


GUIDELINE_PERIOD_TYPES = ('defrost', 'heating', 'off', 'on', 'equilibrium', 'evaluation')


def _clean_guideline_periods(periods_in, cycle_start=None, cycle_end=None):
    """Keep the known period types with a positive length, clamped to the parent."""
    out = []
    for p in periods_in or []:
        ptype = str(p.get('period_type') or '').strip()
        if ptype not in GUIDELINE_PERIOD_TYPES:
            continue
        try:
            s, en = float(p['start_time']), float(p['end_time'])
        except (TypeError, ValueError, KeyError):
            continue
        if cycle_start is not None and cycle_end is not None:
            s = max(cycle_start, min(cycle_end, s))
            en = max(cycle_start, min(cycle_end, en))
        if en > s:
            out.append({'period_type': ptype, 'start_time': s, 'end_time': en})
    return out


def _persist_guideline_clocks(conn, rid, kind, begins_in, trans, periods, buffer_s, df_full):
    """Write one entry's clocks. No second results row — the parent stays whole.

    Marks ``pelec_transition_source='manual'`` (the analyst confirmed this split,
    and 'manual' is what keeps Rebuild all from overwriting it) and replaces the
    entry's ``cycle_periods`` rows for the configured buffer. A ``continuous``
    cycle has no interior transition and no D/S, so it stores neither a
    transition time nor a start marker.
    """
    if kind == 'continuous':
        trans, marker = None, None
    else:
        marker = _marker_from_begins_in(begins_in, kind)
    conn.execute(
        "UPDATE results SET pelec_transition_time=?, pelec_transition_source='manual', "
        "pelec_detect_failed=0, cycle_start_marker=COALESCE(cycle_start_marker, ?) WHERE rowid=?",
        (trans, marker, rid)
    )
    # Replace only this buffer's rows, so the other buffer_s variants and
    # the Period Statistics pivot keep exactly one D/S and one H row.
    conn.execute(
        "DELETE FROM cycle_periods WHERE entry_rowid=? AND buffer_s=? "
        "AND detection_method IN ('auto_pelec', 'manual', ?)",
        (rid, buffer_s, GUIDELINE_DETECTION_METHOD)
    )
    for p in periods:
        _insert_cycle_period_row(conn, rid, p, df_full, GUIDELINE_DETECTION_METHOD, buffer_s)
    return len(periods)


@app.route('/api/cycle_extract/save_periods', methods=['POST'])
def api_cycle_extract_save_periods():
    """Persist the guideline clocks of one entry (Edit + Save on a single row)."""
    try:
        ensure_cycle_periods_table()
        init_deviation_columns()
        payload = request.get_json() or {}
        rid = int(payload['rowid'])
        kind = _normalise_guideline_kind(payload.get('kind'))
        if kind is None:
            return jsonify({'success': False,
                            'message': 'Choose defrost, on–off or continuous before saving.'})
        begins_in = str(payload.get('begins_in') or 'ds').strip().lower()
        if begins_in not in ('ds', 'h'):
            begins_in = 'ds'
        trans = payload.get('transition_time')
        trans = float(trans) if trans is not None and str(trans).strip() != '' else None

        if not _clean_guideline_periods(payload.get('periods')):
            return jsonify({'success': False, 'message': 'Nothing to save — no usable periods.'})

        buffer_s = _guideline_buffer_s()
        conn = get_db_connection()
        try:
            entry = conn.execute(
                "SELECT file_name, data_set, start_time, end_time FROM results WHERE rowid=?",
                (rid,)
            ).fetchone()
            if not entry:
                return jsonify({'success': False, 'message': f'Entry rowid {rid} not found'})
            periods = _clean_guideline_periods(
                payload.get('periods'), float(entry['start_time']), float(entry['end_time'])
            )
            df_full = _read_excel_sheet(entry['file_name'], entry['data_set'])
            n = _persist_guideline_clocks(conn, rid, kind, begins_in, trans, periods, buffer_s, df_full)
            conn.commit()
        finally:
            conn.close()

        return jsonify({'success': True, 'saved': n, 'buffer_s': buffer_s})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_extract/save_all_periods', methods=['POST'])
def api_cycle_extract_save_all_periods():
    """Save the current proposals for a whole file, or for the ticked rows (§2.2.2).

    The proposals are re-derived here with the same helper the read-only propose
    call uses, so "Save all" stores exactly what the analyst is looking at while
    the clock rules stay in one place. A row whose kind is still unknown (and for
    which no file-level default is set) is skipped and reported; the rest of the
    batch still saves. Per-row Edit + Save stays for corrections.

    ``rowids`` limits the batch to the rows the analyst ticked. Null or empty
    means the **whole open file**, not "nothing" (§Phase 1.2b). The reply is
    always JSON, so a failure never reaches the browser as an HTML error page.
    """
    try:
        ensure_cycle_periods_table()
        init_deviation_columns()
        payload = request.get_json() or {}
        file_name = (payload.get('file_name') or '').strip()
        if not file_name:
            return jsonify({'success': False, 'message': 'file_name required'})
        result = _guideline_proposals_for_file(
            file_name,
            want_rowids=payload.get('rowids'),
            overrides=payload.get('overrides'),
            default_kind=payload.get('default_kind'),
        )
        buffer_s = result['buffer_s']

        saved_rowids, skipped, total_periods = [], [], 0
        conn = get_db_connection()
        try:
            sheets = {}
            for pr in result['entries']:
                rid = int(pr['rowid'])
                if pr['kind'] not in GUIDELINE_KINDS:
                    skipped.append({'rowid': rid, 'reason': 'kind still unknown — '
                                                            'set it on the row or pick a file-level default'})
                    continue
                periods = _clean_guideline_periods(
                    pr['periods'], float(pr['start_time']), float(pr['end_time'])
                )
                if not periods:
                    skipped.append({'rowid': rid,
                                    'reason': (pr['notes'][0] if pr['notes'] else 'no usable periods')})
                    continue
                ds_key = pr.get('data_set')
                if ds_key not in sheets:
                    sheets[ds_key] = _read_excel_sheet(file_name, ds_key)
                total_periods += _persist_guideline_clocks(
                    conn, rid, pr['kind'], pr['begins_in'], pr['transition_time'],
                    periods, buffer_s, sheets[ds_key]
                )
                saved_rowids.append(rid)
            conn.commit()
        finally:
            conn.close()

        return jsonify({
            'success': True,
            'saved_rowids': saved_rowids,
            'saved_entries': len(saved_rowids),
            'saved_periods': total_periods,
            'skipped': skipped,
            'buffer_s': buffer_s,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# ---------------------------------------------------------------------------
# Guideline windows page (Phase 2): read the saved clocks back, never re-propose
# ---------------------------------------------------------------------------

GUIDELINE_WINDOW_CSV_COLUMNS = [
    ('rowid', 'rowid'), ('file_name', 'file'), ('data_set', 'data_set'),
    ('test_cond', 'test_cond'), ('profile_id', 'profile'), ('hp_id', 'hp_id'),
    ('start_time', 'parent_start_s'), ('end_time', 'parent_end_s'), ('kind', 'kind'),
    ('d1_start', 'D1_start_s'), ('d1_end', 'D1_end_s'),
    ('d2_start', 'D2_start_s'), ('d2_end', 'D2_end_s'),
    ('s1_start', 'S1_start_s'), ('s1_end', 'S1_end_s'),
    ('s2_start', 'S2_start_s'), ('s2_end', 'S2_end_s'),
    ('h_start', 'H_start_s'), ('h_end', 'H_end_s'),
    ('eq_start', 'eq_start_s'), ('eq_end', 'eq_end_s'),
    ('eval_start', 'eval_start_s'), ('eval_end', 'eval_end_s'),
    ('parent_tsup', 'parent_Tsup_C'), ('parent_q', 'parent_Q_kW'),
    ('parent_p', 'parent_P_kW'), ('parent_cop', 'parent_COP'),
    ('eval_tsup', 'eval_Tsup_C'), ('eval_q', 'eval_Q_kW'),
    ('eval_p', 'eval_P_kW'), ('eval_cop', 'eval_COP'),
    ('note', 'note'),
]


def _guideline_num(value):
    """A float the browser can parse, or None. NaN is missing, never 0."""
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _guideline_kind_from_stored(period_types) -> str:
    """Kind of a parent row, inferred from the guideline periods stored on it.

    Save writes no kind column, so the stored period types are the record. Never
    ``other``, and never read from the test-condition letter or the dataset name.
    A row with no guideline periods is ``unknown``, not ``continuous``.
    """
    types = set(period_types or ())
    if 'defrost' in types:
        return 'defrost'
    if types & {'off', 'on'}:
        return 'on_off'
    if 'heating' in types:
        return 'continuous'
    return 'unknown'


def _prepare_sheet_for_analysis(df_full, file_name):
    """(prepared sheet, lab-corr flag, reason) for the analysis-time means path.

    ``process_file``'s preparation — derived mass flow, the BUH flag, the
    ``T_supply_set`` of the filename letter — applied once to a sheet that is
    already in memory, so several windows of one file share it. The frame is
    copied because the Excel cache hands out a shared dataframe.

    The sheet is returned only when it carries everything the analysis-time
    block needs; otherwise it comes back as ``None`` with a reason that names
    the missing columns, and the caller falls back instead of failing the table.
    """
    if df_full is None or 'time_elapsed' not in getattr(df_full, 'columns', []):
        return None, False, 'sheet could not be read'

    df = df_full.copy()
    df, _mass_flow_notices = derive_mass_flow_if_needed(df)
    lab_corr_missing = any(c not in df.columns for c in ANALYSIS_LAB_CORR_COLUMNS)

    if 'Electrical power input BUH' not in df.columns:
        df['Electrical power input BUH'] = 0
        df['has_buh'] = False
    else:
        buh = pd.to_numeric(df['Electrical power input BUH'], errors='coerce')
        df['has_buh'] = not (buh.notna().all() and len(buh) > 0 and (buh == 0).all())
    df['T_supply_set'] = config['t_supply_set_mapping'].get(extract_letter(file_name), None)

    missing = [c for c in config['average_columns']
               if c not in df.columns and c not in ANALYSIS_SUPPLIED_COLUMNS]
    if missing:
        return None, lab_corr_missing, f"sheet has no {', '.join(missing)}"
    return df, lab_corr_missing, ''


def _analysis_means_for_window(df_sheet, file_name, stf: float, etf: float, lab_corr_missing: bool):
    """The parent's analysis-time means, computed on one window. None if not possible.

    The evaluation means must be the same quantities as the entry's, so the
    window goes through ``analysis_window_means`` (the block Apply runs) plus the
    BUH-corrected derived fields — not through the raw sheet columns, which BAM
    Plotdaten does not carry for Q and P.
    """
    window = df_sheet[(df_sheet['time_elapsed'] >= stf) & (df_sheet['time_elapsed'] <= etf)].copy()
    if window.empty:
        return None
    # A window can be all-zero BUH even when the file is not, as process_file checks.
    if 'Electrical power input BUH' in window.columns:
        buh = pd.to_numeric(window['Electrical power input BUH'], errors='coerce')
        if buh.notna().all() and len(buh) > 0 and (buh == 0).all():
            window['has_buh'] = False
    try:
        means = analysis_window_means(window, file_name, lab_corr_missing=lab_corr_missing)
    except Exception as e:
        print(f"[guideline windows] {file_name} {stf}–{etf} s: analysis means failed: {e}")
        return None
    means.update(analysis_wbuh_fields(means))
    return means


def _guideline_window_rows(rowids=None) -> dict:
    """One row per parent entry: stored clocks, parent means, evaluation means.

    Read-only. Only **saved** guideline periods are read (``detection_method``
    ``guideline`` at the configured buffer) — nothing is proposed here, so clocks
    the analyst has not saved on Cycle Extract stay invisible and the row still
    appears with its identity and parent means.

    ``rowids`` None means every parent row in the open database. That is an
    explicit request from the analyst, so the list is not capped.
    """
    lengths = _guideline_lengths()
    buffer_s = _guideline_buffer_s()

    want = None
    if rowids is not None:
        seen = []
        for raw in rowids:
            try:
                seen.append(int(raw))
            except (TypeError, ValueError):
                continue
        want = sorted(set(seen))

    conn = get_db_connection()
    try:
        result_cols = {r[1] for r in conn.execute('PRAGMA table_info(results)').fetchall()}
        # Mean Values order, so the analyst reads the same sequence on both pages.
        order_by = 'display_order ASC, rowid ASC' if 'display_order' in result_cols else 'rowid ASC'
        if want is None:
            entries = [dict(r) for r in conn.execute(
                f'SELECT rowid, * FROM results ORDER BY {order_by}').fetchall()]
        elif want:
            ph = ','.join(['?'] * len(want))
            entries = [dict(r) for r in conn.execute(
                f'SELECT rowid, * FROM results WHERE rowid IN ({ph}) ORDER BY {order_by}',
                want).fetchall()]
        else:
            entries = []

        stored = {}
        if entries:
            ids = [int(e['rowid']) for e in entries]
            ph = ','.join(['?'] * len(ids))
            for p in conn.execute(
                f'SELECT * FROM cycle_periods WHERE entry_rowid IN ({ph}) '
                f'AND detection_method=? AND buffer_s=? ORDER BY start_time ASC, id ASC',
                ids + [GUIDELINE_DETECTION_METHOD, buffer_s]
            ).fetchall():
                stored.setdefault(int(p['entry_rowid']), []).append(dict(p))
    finally:
        conn.close()

    rows, need_sheet = [], []
    for entry in entries:
        rid = int(entry['rowid'])
        periods = stored.get(rid, [])
        kind = _guideline_kind_from_stored(p['period_type'] for p in periods)

        ds_type = {'defrost': 'defrost', 'on_off': 'off'}.get(kind)
        ds_spans = [p for p in periods if p['period_type'] == ds_type] if ds_type else []
        h_type = 'on' if kind == 'on_off' else 'heating'
        h = next((p for p in periods if p['period_type'] == h_type), None)
        eq = next((p for p in periods if p['period_type'] == 'equilibrium'), None)
        # Periods come back ordered by start_time, so this is the earliest evaluation.
        ev = next((p for p in periods if p['period_type'] == 'evaluation'), None)

        row = {
            'rowid': rid,
            'file_name': entry.get('file_name'),
            'data_set': entry.get('data_set'),
            'test_cond': (entry.get('dev_test_condition') or entry.get('test_cond') or ''),
            'profile_id': entry.get('profile_id') or '',
            'hp_id': entry.get('HP_ID') if entry.get('HP_ID') is not None else entry.get('hp_id'),
            'start_time': _guideline_num(entry.get('start_time')),
            'end_time': _guideline_num(entry.get('end_time')),
            'kind': kind,
            'h_start': _guideline_num(h['start_time']) if h else None,
            'h_end': _guideline_num(h['end_time']) if h else None,
            'eq_start': _guideline_num(eq['start_time']) if eq else None,
            'eq_end': _guideline_num(eq['end_time']) if eq else None,
            'eval_start': _guideline_num(ev['start_time']) if ev else None,
            'eval_end': _guideline_num(ev['end_time']) if ev else None,
            'parent_tsup': _guideline_num(get_mean_supply_for_deviations(entry)),
            'parent_q': _guideline_num(get_mean_q_for_deviations(entry)),
            'parent_p': _guideline_num(get_mean_p_for_deviations(entry)),
            'parent_cop': _guideline_num(get_mean_cop_for_deviations(entry)),
            'eval_tsup': None, 'eval_q': None, 'eval_p': None, 'eval_cop': None,
            'note': '',
        }
        # D1/D2 for a defrost cycle, S1/S2 for an on-off one; the other pair stays empty.
        prefix = 'd' if kind == 'defrost' else ('s' if kind == 'on_off' else None)
        for slot in (1, 2):
            for edge in ('start', 'end'):
                row[f'd{slot}_{edge}'] = None
                row[f's{slot}_{edge}'] = None
        if prefix:
            for i, span in enumerate(ds_spans[:2], start=1):
                row[f'{prefix}{i}_start'] = _guideline_num(span['start_time'])
                row[f'{prefix}{i}_end'] = _guideline_num(span['end_time'])

        if kind == 'unknown':
            row['note'] = 'No guideline clocks saved — save them on Cycle Extract first.'
        elif kind == 'on_off':
            row['note'] = 'On–off cycle — no equilibrium/evaluation window by rule.'
        elif ev is None:
            h_len = None
            if row['h_start'] is not None and row['h_end'] is not None:
                h_len = row['h_end'] - row['h_start']
            if h_len is not None and h_len < lengths['eq_s'] + lengths['eval_s']:
                row['note'] = (f"H is shorter than eq {lengths['eq_min']:g} min + "
                               f"eval {lengths['eval_min']:g} min — no evaluation window stored.")
            else:
                row['note'] = 'No evaluation window stored for this entry.'
        else:
            need_sheet.append((row, ev))

        rows.append(row)

    # The evaluation means are recomputed from the sheet whenever it can be read:
    # the cache Save wrote holds Tsup for a BAM defrost row but no Q/P/COP, and a
    # partial cache must not pass as done. Requested rows only, one read per
    # sheet. A missing sheet leaves the cells empty with a reason and the cached
    # Tsup if there is one; it never fails the table.
    by_sheet = {}
    for row, ev in need_sheet:
        by_sheet.setdefault((row['file_name'], row['data_set']), []).append((row, ev))
    for (file_name, data_set), items in by_sheet.items():
        df_full = _read_excel_sheet(file_name, data_set)
        if df_full is None:
            for row, ev in items:
                row['eval_tsup'] = _guideline_num(ev.get('avg_ts_buh'))
                row['note'] = 'Evaluation window stored, but its sheet could not be read.'
            continue
        df_sheet, lab_corr_missing, prep_reason = _prepare_sheet_for_analysis(df_full, file_name)
        for row, ev in items:
            stf, etf = float(ev['start_time']), float(ev['end_time'])
            means = None
            if df_sheet is not None:
                means = _analysis_means_for_window(df_sheet, file_name, stf, etf, lab_corr_missing)
            if means is not None:
                row['eval_tsup'] = _guideline_num(get_mean_supply_for_deviations(means))
                row['eval_q'] = _guideline_num(get_mean_q_for_deviations(means))
                row['eval_p'] = _guideline_num(get_mean_p_for_deviations(means))
                row['eval_cop'] = _guideline_num(get_mean_cop_for_deviations(means))
            else:
                # Last resort: the stored series themselves. A sheet that already
                # carries QCorrwBUH / lab-corrected Q and P still answers here.
                stats = _period_stats_from_df(df_full, stf, etf)
                row['eval_tsup'] = _guideline_num(stats.get('avg_ts_buh'))
                row['eval_q'] = _guideline_num(stats.get('avg_q_corr_wbuh'))
                row['eval_p'] = _guideline_num(stats.get('avg_p_corr_wbuh'))
                row['eval_cop'] = _guideline_num(stats.get('avg_cop_corr_wbuh'))
                if prep_reason and any(row[k] is None for k in ('eval_q', 'eval_p', 'eval_cop')):
                    row['note'] = ("Evaluation Q/P/COP need the entry's own quantities: "
                                   f'{prep_reason}.')
            if all(row[k] is None for k in ('eval_tsup', 'eval_q', 'eval_p', 'eval_cop')):
                row['eval_tsup'] = _guideline_num(ev.get('avg_ts_buh'))
                row['note'] = 'Evaluation window stored, but the sheet holds no values in it.'

    with_clocks = sum(1 for r in rows if r['kind'] != 'unknown')
    with_eval = sum(1 for r in rows if r['eval_start'] is not None)
    message = (f'{len(rows)} entries — {with_clocks} with saved guideline clocks, '
               f'{with_eval} with an evaluation window (buffer {buffer_s} s).')
    return {'rows': rows, 'buffer_s': buffer_s, 'lengths': lengths, 'message': message}


def _guideline_windows_selector(payload) -> list | None:
    """``rowids`` from a request body: a list of ints, or None meaning all parents."""
    rowids = payload.get('rowids', None) if isinstance(payload, dict) else None
    if rowids is None:
        return None
    if not isinstance(rowids, list):
        raise ValueError('rowids must be a list of integers or null')
    return rowids


@app.route('/guideline_windows', methods=['GET'])
def guideline_windows():
    """Empty shell. Nothing is read until the analyst asks for rows (§2.5)."""
    return render_template('guideline_windows.html', preselected_rowids=[])


@app.route('/guideline_windows', methods=['POST'])
def guideline_windows_from_selection():
    """Open the page for the rows ticked on Mean Values (same pattern as Plot)."""
    conn = get_db_connection()
    try:
        records = resolve_selected_rows(
            conn,
            request.form.getlist('file_names'),
            request.form.getlist('data_sets'),
            request.form.getlist('row_ids'),
        )
    finally:
        conn.close()
    rowids = [int(r['rowid']) for r in records if r.get('rowid') is not None]
    if not rowids:
        flash('Select at least one row before opening Guideline windows.', 'warning')
        return redirect(url_for('index'))
    return render_template('guideline_windows.html', preselected_rowids=rowids)


@app.route('/api/guideline_windows', methods=['POST'])
def api_guideline_windows():
    """Table rows for the selected entries, or for the whole open database.

    ``{"rowids": [1, 2]}`` limits the table to those parents; ``{"rowids": null}``
    is the explicit "load all entries in the open database". The reply is always
    JSON, so a failure never reaches the browser as an HTML page (§Phase 1.2b).
    """
    try:
        ensure_cycle_periods_table()
        payload = request.get_json(silent=True) or {}
        result = _guideline_window_rows(_guideline_windows_selector(payload))
        return jsonify({
            'success': True,
            'rows': result['rows'],
            'message': result['message'],
            'buffer_s': result['buffer_s'],
            'lengths': result['lengths'],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'rows': [], 'message': str(e)})


@app.route('/api/guideline_windows/export', methods=['POST'])
def api_guideline_windows_export():
    """CSV of the same table. Nothing is added to the t42_summary_v1 export."""
    try:
        ensure_cycle_periods_table()
        payload = request.get_json(silent=True)
        if payload is None:
            raw = (request.form.get('rowids') or '').strip()
            payload = {'rowids': json.loads(raw) if raw else None}
        result = _guideline_window_rows(_guideline_windows_selector(payload))

        import csv
        from io import StringIO
        buf = StringIO()
        writer = csv.writer(buf, lineterminator='\n')
        writer.writerow([header for _key, header in GUIDELINE_WINDOW_CSV_COLUMNS])
        for row in result['rows']:
            writer.writerow(['' if row.get(key) is None else row.get(key)
                             for key, _header in GUIDELINE_WINDOW_CSV_COLUMNS])
        payload_bytes = buf.getvalue().encode('utf-8-sig')
        return send_file(
            BytesIO(payload_bytes),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"guideline_windows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_extract/list', methods=['POST'])
def api_cycle_extract_list():
    """List suggestions for a given file/dataset (or latest if not provided)."""
    ensure_cycle_extraction_suggestions_table()
    try:
        payload = request.get_json() or {}
        file_name = (payload.get('file_name') or '').strip()
        data_set = payload.get('data_set', None)
        try:
            data_set = float(data_set) if data_set is not None and str(data_set).strip() != '' else None
        except Exception:
            data_set = None

        conn = get_db_connection()
        try:
            if file_name:
                rows = conn.execute(
                    "SELECT id, file_name, data_set, start_time, end_time, confidence, notes_auto, pelec_drop_mag, created_at "
                    "FROM cycle_extraction_suggestions "
                    "WHERE file_name=? AND (data_set IS ? OR data_set=?) "
                    "ORDER BY start_time ASC",
                    (file_name, data_set, data_set if data_set is not None else None)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, file_name, data_set, start_time, end_time, confidence, notes_auto, pelec_drop_mag, created_at "
                    "FROM cycle_extraction_suggestions "
                    "ORDER BY created_at DESC, file_name ASC, data_set ASC, start_time ASC LIMIT 500"
                ).fetchall()
        finally:
            conn.close()
        return jsonify({'success': True, 'suggestions': [dict(r) for r in rows]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_extract/apply', methods=['POST'])
def api_cycle_extract_apply():
    """Apply selected suggestions by inserting rows into results via process_new_data()."""
    ensure_cycle_extraction_suggestions_table()
    try:
        payload = request.get_json() or {}
        ids = payload.get('suggestion_ids') or []
        edits_raw = payload.get('edits') or {}
        batch_profile_id = payload.get('profile_id')
        edits = {str(k): v for k, v in edits_raw.items()} if isinstance(edits_raw, dict) else {}
        if not isinstance(ids, list) or not ids:
            return jsonify({'success': False, 'message': 'suggestion_ids required'})

        conn = get_db_connection()
        try:
            placeholders = ",".join(["?"] * len(ids))
            rows = conn.execute(
                f"SELECT id, file_name, data_set, start_time, end_time FROM cycle_extraction_suggestions WHERE id IN ({placeholders})",
                ids
            ).fetchall()
            rows = [dict(r) for r in rows]
        finally:
            conn.close()

        added = 0
        skipped = 0
        errors = []
        inserted_rowids = []
        skipped_duplicate_details = []
        notices = []

        try:
            req_ids = []
            for x in ids:
                try:
                    req_ids.append(int(x))
                except (TypeError, ValueError):
                    pass
            found_ids = {int(r['id']) for r in rows}
            missing = [i for i in req_ids if i not in found_ids]
            if missing:
                errors.append(
                    'Suggestion id(s) not in database (try Suggest cycles again): ' + ', '.join(str(m) for m in missing[:15])
                    + ('…' if len(missing) > 15 else '')
                )
        except Exception:
            pass

        for r in rows:
            sid = str(r['id'])
            fn = r['file_name']
            ds = r.get('data_set')
            st = r['start_time']
            et = r['end_time']
            if sid in edits:
                e = edits.get(sid) or {}
                if 'start_time' in e:
                    st = float(e['start_time'])
                if 'end_time' in e:
                    et = float(e['end_time'])
                if 'data_set' in e and e['data_set'] is not None and str(e['data_set']).strip() != '':
                    ds = float(e['data_set'])

            # Must match process_new_data(): NULL/ missing sheet index → 1.0
            ds_eff = float(ds) if ds is not None else 1.0
            st_eff = float(st)
            et_eff = float(et)

            # skip duplicates (same key as a row already in results)
            conn2 = get_db_connection()
            try:
                existing = conn2.execute(
                    "SELECT rowid FROM results WHERE file_name=? AND data_set=? AND start_time=? AND end_time=?",
                    (fn, ds_eff, st_eff, et_eff)
                ).fetchone()
            finally:
                conn2.close()
            if existing:
                skipped += 1
                skipped_duplicate_details.append({
                    'suggestion_id': int(r['id']),
                    'existing_rowid': int(existing['rowid']),
                    'file_name': fn,
                    'data_set': ds_eff,
                    'start_time': st_eff,
                    'end_time': et_eff,
                })
                continue

            try:
                cycle_notices = []
                rowid = process_new_data(fn, ds_eff, st_eff, et_eff, return_rowid=True, notices_out=cycle_notices)
                for n in cycle_notices:
                    if n not in notices:
                        notices.append(n)
                if rowid:
                    added += 1
                    inserted_rowids.append(int(rowid))
                else:
                    errors.append(f"{fn} ds={ds_eff} {st_eff}-{et_eff}: no data")
            except Exception as e:
                errors.append(f"{fn} ds={ds_eff} {st_eff}-{et_eff}: {e}")

        if inserted_rowids:
            session_note_inserted_rowids(inserted_rowids)
            _stamp_profile_on_rowids(inserted_rowids, batch_profile_id)

        return jsonify({
            'success': True,
            'added': added,
            'skipped': skipped,
            'skipped_duplicate_details': skipped_duplicate_details[:50],
            'errors': errors[:20],
            'inserted_rowids': inserted_rowids,
            'notices': collapsed_notice_messages(notices[:20]),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_extract/add_manual', methods=['POST'])
def api_cycle_extract_add_manual():
    """Insert one or more manually defined cycles into the results table."""
    try:
        payload = request.get_json() or {}
        entries = payload.get('entries') or []
        batch_profile_id = payload.get('profile_id')
        if not entries:
            return jsonify({'success': False, 'message': 'No entries provided'})

        added = 0
        skipped = 0
        errors_list = []
        inserted_rowids = []
        notices = []

        for e in entries:
            fn = (e.get('file_name') or '').strip()
            ds = e.get('data_set', 1)
            st = e.get('start_time')
            et = e.get('end_time')
            if not fn or st is None or et is None:
                errors_list.append(f'Missing fields: {e}')
                continue
            try:
                ds = float(ds) if ds is not None else 1.0
                st = float(st)
                et = float(et)
            except (TypeError, ValueError) as ex:
                errors_list.append(str(ex))
                continue

            conn = get_db_connection()
            try:
                existing = conn.execute(
                    "SELECT rowid FROM results WHERE file_name=? AND data_set=? AND start_time=? AND end_time=?",
                    (fn, ds, st, et)
                ).fetchone()
            finally:
                conn.close()
            if existing:
                skipped += 1
                continue

            try:
                cycle_notices = []
                rowid = process_new_data(fn, ds, st, et, return_rowid=True, notices_out=cycle_notices)
                for n in cycle_notices:
                    if n not in notices:
                        notices.append(n)
                if rowid:
                    added += 1
                    inserted_rowids.append(int(rowid))
                else:
                    errors_list.append(f'{fn} ds={ds} {st}-{et}: no data')
            except Exception as ex:
                errors_list.append(f'{fn} ds={ds} {st}-{et}: {ex}')

        if inserted_rowids:
            session_note_inserted_rowids(inserted_rowids)
            _stamp_profile_on_rowids(inserted_rowids, batch_profile_id)

        return jsonify({
            'success': True, 'added': added, 'skipped': skipped,
            'errors': errors_list[:20], 'inserted_rowids': inserted_rowids,
            'notices': collapsed_notice_messages(notices[:20]),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_extract/plot', methods=['POST'])
def api_cycle_extract_plot():
    """Plot Pelec + dTreturn for a suggestion (or arbitrary file/ds/start/end)."""
    try:
        payload = request.get_json() or {}
        file_name = (payload.get('file_name') or '').strip()
        data_set = payload.get('data_set', None)
        try:
            data_set = float(data_set) if data_set is not None and str(data_set).strip() != '' else None
        except Exception:
            data_set = None
        st = float(payload.get('start_time'))
        et = float(payload.get('end_time'))
        pad_s = float(payload.get('pad_s', 0.0) or 0.0)

        df_full = _read_excel_sheet(file_name, data_set)
        if df_full is None or 'time_elapsed' not in df_full.columns:
            return jsonify({'success': False, 'message': f'Cannot read {file_name}'})
        if PELEC_COL not in df_full.columns:
            return jsonify({'success': False, 'message': f'Column {PELEC_COL} not found'})

        stp = st - pad_s
        etp = et + pad_s
        df = df_full[(df_full['time_elapsed'] >= stp) & (df_full['time_elapsed'] <= etp)].copy()
        if df.empty:
            return jsonify({'success': False, 'message': 'No data in time range'})

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax1.plot(df['time_elapsed'], df[PELEC_COL], color='#343a40', linewidth=0.9)
        ax1.axvline(st, color='green', linestyle='--', linewidth=1.2, label='start')
        ax1.axvline(et, color='red', linestyle='--', linewidth=1.2, label='end')
        ax1.set_ylabel('Electric Power (kW)')
        ax1.grid(True, alpha=0.2)
        ax1.set_title(f'{file_name} ds={data_set} [{st:.0f} – {et:.0f}]')
        ax1.legend(loc='best', fontsize=8)

        if 'T_return_emu' in df.columns and 'T_return_calc' in df.columns:
            dt = df['T_return_emu'] - df['T_return_calc']
            ax2.plot(df['time_elapsed'], dt, color='#0b7285', linewidth=0.9)
            ax2.axhline(0, color='#666', linewidth=0.8, alpha=0.6)
        ax2.axvline(st, color='green', linestyle='--', linewidth=1.2)
        ax2.axvline(et, color='red', linestyle='--', linewidth=1.2)
        ax2.set_ylabel('dTreturn (K)')
        ax2.set_xlabel('time_elapsed (s)')
        ax2.grid(True, alpha=0.2)

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=130)
        plt.close(fig)
        buf.seek(0)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({'success': True, 'image': img_b64})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


# ---------------------------------------------------------------------------
# Unified pipeline (simplified UI): one pass -> classify/detect/periods(3 buffers)/cache
# ---------------------------------------------------------------------------

@app.route('/api/pipeline/status', methods=['GET'])
def api_pipeline_status():
    """Return lightweight status counters for the unified cache pipeline."""
    init_deviation_columns()
    ensure_cycle_periods_table()
    try:
        conn = get_db_connection()
        try:
            total_entries = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]

            # pending if full-cycle cache missing OR classification missing OR transition missing
            # OR any of the multi-buffer period caches missing for classified non-'other' entries.
            placeholders = ",".join(["?"] * len(BUFFER_OPTIONS))
            stale_period_sql = _period_stats_stale_sql('cp')
            pending_entries = conn.execute(
                f"""
                SELECT COUNT(*) FROM results r
                WHERE r.dtreturn_cache_n_valid IS NULL
                   OR (r.dtreturn_cache_n_valid > 0 AND (r.dtreturn_cache_max_abs IS NULL OR r.dtreturn_cache_p99_abs IS NULL OR r.dtreturn_cache_avg IS NULL))
                   OR r.avg_t_mean_log IS NULL
                   OR r.cycle_type IS NULL
                   OR (r.cycle_type IS NOT NULL AND r.cycle_type != 'other'
                        AND r.pelec_transition_time IS NULL
                        AND COALESCE(r.pelec_detect_failed, 0) != 1
                        AND COALESCE(r.pelec_transition_source, 'auto') != 'manual')
                   OR (
                        r.cycle_type IS NOT NULL AND r.cycle_type != 'other'
                        AND COALESCE(r.pelec_detect_failed, 0) != 1
                        AND COALESCE(r.pelec_transition_source, 'auto') != 'manual' AND
                        (SELECT COUNT(DISTINCT cp.buffer_s)
                           FROM cycle_periods cp
                          WHERE cp.entry_rowid = r.rowid
                            AND cp.dtreturn_n_valid IS NOT NULL
                            AND (cp.dtreturn_min IS NOT NULL OR cp.dtreturn_n_valid = 0)
                            AND cp.buffer_s IN ({placeholders})
                        ) < {len(BUFFER_OPTIONS)}
                      )
                   OR EXISTS (
                        SELECT 1 FROM cycle_periods cp
                         WHERE cp.entry_rowid = r.rowid AND {stale_period_sql}
                      )
                """,
                BUFFER_OPTIONS
            ).fetchone()[0]

            total_periods = conn.execute("SELECT COUNT(*) FROM cycle_periods").fetchone()[0]
            uncached_periods = conn.execute("SELECT COUNT(*) FROM cycle_periods WHERE dtreturn_n_valid IS NULL").fetchone()[0]
        finally:
            conn.close()

        return jsonify({
            'success': True,
            'total_entries': int(total_entries),
            'pending_entries': int(pending_entries),
            'total_periods': int(total_periods),
            'uncached_periods': int(uncached_periods),
            'buffers': BUFFER_OPTIONS,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


def _dtreturn_stats_from_df(df_full, stf: float, etf: float, max_sample: int = 5000, rnd_seed: int = 42):
    """Compute dTreturn cache stats from an already-loaded sheet dataframe.

    Returns (n_valid, n_nan, sample, max_abs, max_signed, p99_abs, dt_min, dt_max, dt_avg)
    where dt_min/dt_max are the signed minimum (typically negative) and maximum
    (typically positive) dTreturn over the window, and dt_avg is the arithmetic mean.
    """
    if df_full is None or 'time_elapsed' not in df_full.columns:
        return 0, 0, [], None, None, None, None, None, None
    df = df_full[(df_full['time_elapsed'] >= stf) & (df_full['time_elapsed'] <= etf)]
    if df.empty or 'T_return_emu' not in df.columns or 'T_return_calc' not in df.columns:
        return 0, 0, [], None, None, None, None, None, None

    dt = df['T_return_emu'] - df['T_return_calc']
    valid = dt.dropna()
    n_valid = int(len(valid))
    n_nan = int(len(dt) - n_valid)

    import random
    rnd = random.Random(rnd_seed)
    sample = []
    for i, v in enumerate(valid.values):
        fv = round(float(v), 6)
        if len(sample) < max_sample:
            sample.append(fv)
        else:
            j = rnd.randint(0, i)
            if j < max_sample:
                sample[j] = fv

    max_abs = max_signed = p99_abs = dt_min = dt_max = dt_avg = None
    if n_valid > 0:
        arr = valid.to_numpy(dtype=float)
        abs_arr = np.abs(arr)
        max_idx = int(abs_arr.argmax())
        max_signed = float(arr[max_idx])
        max_abs = float(abs_arr[max_idx])
        dt_min = float(np.nanmin(arr))
        dt_max = float(np.nanmax(arr))
        dt_avg = float(np.nanmean(arr))
        if n_valid >= 2:
            p99_abs = float(np.percentile(abs_arr, 99))
        else:
            p99_abs = max_abs

    return n_valid, n_nan, sample, max_abs, max_signed, p99_abs, dt_min, dt_max, dt_avg


PERIOD_MEAN_METRICS = (
    'avg_t_db', 'avg_t_wb', 'avg_ts_buh', 'avg_q_corr_wbuh', 'avg_p_corr_wbuh',
    'avg_cop_corr_wbuh', 'avg_volume_flow', 'avg_mass_flow',
    'avg_t_mean_log', 't_mean_from_avgs',
)
# Bump when period mean-stat columns or computation change (triggers backfill).
PERIOD_STATS_SCHEMA = 3


def _period_stats_stale_sql(alias: str = 'cp') -> str:
    """SQL predicate: dTreturn cache complete but mean-stat schema outdated."""
    return (
        f"{alias}.dtreturn_n_valid IS NOT NULL "
        f"AND ({alias}.dtreturn_min IS NOT NULL OR {alias}.dtreturn_n_valid = 0) "
        f"AND COALESCE({alias}.period_stats_schema, 0) < {PERIOD_STATS_SCHEMA}"
    )


def _load_period_metrics_stale_entry_ids(conn) -> set:
    """Entry rowids whose cycle_periods rows need mean-stat recomputation."""
    rows = conn.execute(
        f"SELECT DISTINCT entry_rowid FROM cycle_periods WHERE {_period_stats_stale_sql('cycle_periods')}"
    ).fetchall()
    return {int(r[0]) for r in rows}


def _period_stats_from_df(df_full, stf: float, etf: float) -> dict:
    """Mean operating-point values over a period time window."""
    out = {k: None for k in PERIOD_MEAN_METRICS}
    if df_full is None or 'time_elapsed' not in df_full.columns:
        return out
    df = df_full[(df_full['time_elapsed'] >= stf) & (df_full['time_elapsed'] <= etf)]
    if df.empty:
        return out
    for k in PERIOD_MEAN_METRICS:
        try:
            out[k] = compute_single_column_value(df, k)
        except Exception:
            out[k] = None
    return out


def _period_cache_values(df_full, stf: float, etf: float):
    """Combined dTreturn + operating-point caches for one period window."""
    n_valid, n_nan, sample, max_abs, max_signed, p99_abs, dt_min, dt_max, dt_avg = _dtreturn_stats_from_df(
        df_full, stf, etf
    )
    pst = _period_stats_from_df(df_full, stf, etf)
    return n_valid, n_nan, sample, max_abs, max_signed, p99_abs, dt_min, dt_max, dt_avg, pst


def _insert_cycle_period_row(conn, rid, p, df_full, detection_method: str, buffer_s: int) -> None:
    """Insert one cycle_periods row with dTreturn and mean-value caches."""
    stf, etf = float(p['start_time']), float(p['end_time'])
    n_valid, n_nan, sample, max_abs, max_signed, p99_abs, dt_min, dt_max, dt_avg, pst = _period_cache_values(
        df_full, stf, etf
    )
    conn.execute(
        "INSERT INTO cycle_periods "
        "(entry_rowid, period_type, start_time, end_time, detection_method, buffer_s, "
        " dtreturn_n_valid, dtreturn_n_nan, dtreturn_sample, dtreturn_max_abs, dtreturn_max_signed, "
        " dtreturn_min, dtreturn_max, dtreturn_avg, dtreturn_p99_abs, "
        " avg_t_db, avg_t_wb, avg_ts_buh, avg_q_corr_wbuh, avg_p_corr_wbuh, avg_cop_corr_wbuh, "
        " avg_volume_flow, avg_mass_flow, avg_t_mean_log, t_mean_from_avgs, period_stats_schema) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            rid, p['period_type'], stf, etf, detection_method, int(buffer_s),
            n_valid, n_nan, json.dumps(sample), max_abs, max_signed, dt_min, dt_max, dt_avg, p99_abs,
            pst['avg_t_db'], pst['avg_t_wb'], pst['avg_ts_buh'], pst['avg_q_corr_wbuh'],
            pst['avg_p_corr_wbuh'], pst['avg_cop_corr_wbuh'], pst['avg_volume_flow'], pst['avg_mass_flow'],
            pst.get('avg_t_mean_log'), pst.get('t_mean_from_avgs'),
            PERIOD_STATS_SCHEMA,
        ),
    )


def _refresh_cycle_period_row_stats(conn, period_id: int, df_full, stf: float, etf: float) -> None:
    """Recompute cached stats for an existing cycle_periods row."""
    n_valid, n_nan, sample, max_abs, max_signed, p99_abs, dt_min, dt_max, dt_avg, pst = _period_cache_values(
        df_full, stf, etf
    )
    conn.execute(
        "UPDATE cycle_periods SET dtreturn_n_valid=?, dtreturn_n_nan=?, dtreturn_sample=?, "
        "dtreturn_max_abs=?, dtreturn_max_signed=?, dtreturn_min=?, dtreturn_max=?, dtreturn_avg=?, dtreturn_p99_abs=?, "
        "avg_t_db=?, avg_t_wb=?, avg_ts_buh=?, avg_q_corr_wbuh=?, avg_p_corr_wbuh=?, avg_cop_corr_wbuh=?, "
        "avg_volume_flow=?, avg_mass_flow=?, avg_t_mean_log=?, t_mean_from_avgs=?, period_stats_schema=? WHERE id=?",
        (
            n_valid, n_nan, json.dumps(sample), max_abs, max_signed, dt_min, dt_max, dt_avg, p99_abs,
            pst['avg_t_db'], pst['avg_t_wb'], pst['avg_ts_buh'], pst['avg_q_corr_wbuh'],
            pst['avg_p_corr_wbuh'], pst['avg_cop_corr_wbuh'], pst['avg_volume_flow'], pst['avg_mass_flow'],
            pst.get('avg_t_mean_log'), pst.get('t_mean_from_avgs'),
            PERIOD_STATS_SCHEMA,
            period_id,
        ),
    )


def _effective_period_buffer_s(cycle_type, buffer_s: int) -> int:
    """On-off periods are always stored with buffer_s=0."""
    if cycle_type == 'on_off_cycle':
        return 0
    if buffer_s != 0 and buffer_s not in BUFFER_OPTIONS:
        return SETTLING_BUFFER_S
    return buffer_s


def _period_duration_s(p) -> float:
    if not p or p.get('start_time') is None or p.get('end_time') is None:
        return 0.0
    return max(0.0, float(p['end_time']) - float(p['start_time']))


def _weighted_period_metric(periods: list, metric_key: str):
    """Duration-weighted mean of a metric across multiple periods."""
    total_w = 0.0
    total_v = 0.0
    for p in periods:
        dur = _period_duration_s(p)
        val = p.get(metric_key)
        if val is None or dur <= 0:
            continue
        total_v += float(val) * dur
        total_w += dur
    return (total_v / total_w) if total_w > 0 else None


def _pivot_period_stats(periods: list, cycle_type: str, start_marker: str) -> dict:
    """Pivot ordered period rows into labelled slots for the statistics table."""
    periods = sorted(periods, key=lambda p: float(p.get('start_time') or 0))
    out = {}
    metric_keys = list(PERIOD_MEAN_METRICS) + [
        'dtreturn_avg', 'dtreturn_min', 'dtreturn_max', 'dtreturn_n_valid', 'duration_s',
    ]

    def _slot(prefix, p):
        if not p:
            return
        dur = _period_duration_s(p)
        for mk in metric_keys:
            out[f'{prefix}_{mk}'] = p.get(mk) if mk != 'duration_s' else dur

    def _slot_weighted(prefix, plist):
        if not plist:
            return
        total_dur = sum(_period_duration_s(p) for p in plist)
        out[f'{prefix}_duration_s'] = total_dur if total_dur > 0 else None
        for mk in PERIOD_MEAN_METRICS + ('dtreturn_avg', 'dtreturn_min', 'dtreturn_max'):
            if mk == 'dtreturn_min':
                vals = [p.get(mk) for p in plist if p.get(mk) is not None]
                out[f'{prefix}_{mk}'] = min(vals) if vals else None
            elif mk == 'dtreturn_max':
                vals = [p.get(mk) for p in plist if p.get(mk) is not None]
                out[f'{prefix}_{mk}'] = max(vals) if vals else None
            else:
                out[f'{prefix}_{mk}'] = _weighted_period_metric(plist, mk)
        nv = sum(int(p.get('dtreturn_n_valid') or 0) for p in plist)
        out[f'{prefix}_dtreturn_n_valid'] = nv if nv > 0 else None

    if cycle_type == 'on_off_cycle':
        off_p = next((p for p in periods if p.get('period_type') == 'off'), None)
        on_p = next((p for p in periods if p.get('period_type') == 'on'), None)
        _slot('Off', off_p)
        _slot('On', on_p)
        return out

    defs = [p for p in periods if p.get('period_type') == 'defrost']
    heat = next((p for p in periods if p.get('period_type') == 'heating'), None)
    if len(defs) >= 2:
        _slot_weighted('D', defs)
        _slot('D1', defs[0])
        _slot('H', heat)
        _slot('D2', defs[-1])
    elif len(defs) == 1:
        _slot('D', defs[0])
        _slot('H', heat)
    else:
        for i, dp in enumerate(defs, start=1):
            _slot(f'D{i}', dp)
        _slot('H', heat)
    return out


@app.route('/api/pipeline/run', methods=['POST'])
def api_pipeline_run():
    """Unified pipeline: classify + detect + generate periods (for all buffers) + cache dTreturn (full + per-period)."""
    init_deviation_columns()
    ensure_cycle_periods_table()
    try:
        payload = request.get_json() or {}
        force_all = bool(payload.get('force_all', False))
        try:
            limit_entries = payload.get('limit_entries', None)
            limit_entries = int(limit_entries) if limit_entries is not None else None
        except Exception:
            limit_entries = None
        if limit_entries is not None:
            limit_entries = max(1, min(500, limit_entries))

        # Excel LRU cache (shared across all entries processed in this run)
        from collections import OrderedDict
        import time as _time
        t0 = _time.time()
        MAX_CACHE = 40
        _excel_cache = OrderedDict()

        def _cached_read(fname, dset):
            key = (fname, dset)
            if key in _excel_cache:
                _excel_cache.move_to_end(key)
                return _excel_cache[key]
            df = _read_excel_sheet(fname, dset)
            _excel_cache[key] = df
            if len(_excel_cache) > MAX_CACHE:
                _excel_cache.popitem(last=False)
            return df

        conn = get_db_connection()
        try:
            rows = conn.execute(
                "SELECT rowid, file_name, data_set, start_time, end_time, Notes, test_cond, "
                "cycle_type, cycle_start_marker, pelec_transition_time, pelec_transition_source, "
                "pelec_detect_failed, "
                "dtreturn_cache_n_valid, dtreturn_cache_max_abs, dtreturn_cache_p99_abs, dtreturn_cache_avg, "
                "avg_t_mean_log "
                "FROM results ORDER BY file_name, data_set, rowid"
            ).fetchall()
            rows = [dict(r) for r in rows]
        finally:
            conn.close()

        total_results_rows = len(rows)

        # One grouped query avoids O(n) SELECT COUNT(...) per row when checking BUFFER_OPTIONS coverage.
        placeholders_buf = ",".join(["?"] * len(BUFFER_OPTIONS))
        period_buf_count = {}
        conn_bc = get_db_connection()
        try:
            cur_bc = conn_bc.execute(
                f"SELECT entry_rowid, COUNT(DISTINCT buffer_s) FROM cycle_periods "
                f"WHERE dtreturn_n_valid IS NOT NULL "
                f"  AND (dtreturn_min IS NOT NULL OR dtreturn_n_valid = 0) "
                f"  AND buffer_s IN ({placeholders_buf}) "
                f"GROUP BY entry_rowid",
                list(BUFFER_OPTIONS),
            )
            for erid, c in cur_bc.fetchall():
                period_buf_count[int(erid)] = int(c or 0)
        finally:
            conn_bc.close()

        period_zero_cached = set()
        conn_z = get_db_connection()
        try:
            for (erid,) in conn_z.execute(
                "SELECT DISTINCT entry_rowid FROM cycle_periods "
                "WHERE dtreturn_n_valid IS NOT NULL "
                "  AND (dtreturn_min IS NOT NULL OR dtreturn_n_valid = 0) "
                "AND buffer_s = 0"
            ).fetchall():
                period_zero_cached.add(int(erid))
        finally:
            conn_z.close()

        conn_stale = get_db_connection()
        try:
            period_metrics_stale = _load_period_metrics_stale_entry_ids(conn_stale)
        finally:
            conn_stale.close()

        def row_needs_pipeline(r: dict) -> bool:
            if force_all:
                return True
            rid_loc = int(r['rowid'])
            nv_full = r.get('dtreturn_cache_n_valid')
            # A row whose full-cycle read produced 0 valid dTreturn points has been
            # fully attempted; max_abs/p99 are legitimately NULL (no data), so it
            # must NOT count as pending or it re-queues forever.
            needs_full = (nv_full is None) or (
                nv_full > 0 and (r.get('dtreturn_cache_max_abs') is None or r.get('dtreturn_cache_p99_abs') is None or r.get('dtreturn_cache_avg') is None)
            ) or r.get('avg_t_mean_log') is None
            # Classification is cheap (string parsing only): recompute from the
            # current Notes/test_cond and re-queue if the stored value is stale
            # (e.g. metadata filled in after an earlier run left it as 'other').
            fresh_ct, fresh_sm = _classify_entry(r.get('Notes') or '', r.get('test_cond') or '')
            classification_stale = (fresh_ct != r.get('cycle_type')) or (fresh_sm != r.get('cycle_start_marker'))
            if classification_stale:
                return True
            needs_class = (r.get('cycle_type') is None) or (r.get('cycle_start_marker') is None)
            # Entries already attempted and found undetectable/degenerate are not
            # re-queued by "Update cache" (only "Rebuild all" retries them).
            detect_failed = (r.get('pelec_detect_failed') == 1)
            is_manual_trans = (r.get('pelec_transition_source') == 'manual')
            needs_trans = (
                r.get('cycle_type') is not None
                and r.get('cycle_type') != 'other'
                and r.get('pelec_transition_time') is None
                and not detect_failed
                and not is_manual_trans
            )
            needs_periods = False
            sm = r.get('cycle_start_marker')
            if (
                not detect_failed
                and not is_manual_trans
                and r.get('cycle_type') is not None
                and r.get('cycle_type') != 'other'
                and r.get('pelec_transition_time') is not None
            ):
                if sm in ('defrost_start', 'defrost_end'):
                    cnt_pb = period_buf_count.get(rid_loc, 0)
                    needs_periods = int(cnt_pb) < len(BUFFER_OPTIONS)
                elif sm in ('off_start', 'on_start'):
                    needs_periods = rid_loc not in period_zero_cached
            needs_period_metrics = rid_loc in period_metrics_stale
            return needs_full or needs_class or needs_trans or needs_periods or needs_period_metrics

        if not force_all:
            rows = [r for r in rows if row_needs_pipeline(r)]

        queued_entries = len(rows)
        print(
            f"[pipeline] queued={queued_entries}/{total_results_rows} "
            f"(force_all={force_all}); starting run…",
            flush=True,
        )

        processed_entries = 0
        skipped_entries = 0
        updated_full_cache = 0
        updated_period_cache = 0
        detected = 0
        failed_detection = 0

        conn = get_db_connection()
        try:
            for idx, r in enumerate(rows):
                rid = int(r['rowid'])
                fn = r.get('file_name')
                ds = r.get('data_set')
                st = r.get('start_time')
                et = r.get('end_time')
                if not fn or st is None or et is None:
                    skipped_entries += 1
                    continue

                try:
                    stf, etf = float(st), float(et)
                except Exception:
                    skipped_entries += 1
                    continue

                df_full = _cached_read(fn, ds)
                if df_full is None or 'time_elapsed' not in df_full.columns:
                    skipped_entries += 1
                    # mark full cache as empty to avoid repeated retries
                    conn.execute(
                        "UPDATE results SET dtreturn_cache_n_valid=0, dtreturn_cache_n_nan=0, dtreturn_cache_sample='[]' "
                        "WHERE rowid=?", (rid,)
                    )
                    continue

                # 1) full-cycle dtreturn cache + T_mean (same Excel window as the mean-values page)
                n_valid, n_nan, sample, max_abs, max_signed, p99_abs, dt_min, dt_max, dt_avg = _dtreturn_stats_from_df(df_full, stf, etf)
                conn.execute(
                    "UPDATE results SET dtreturn_cache_n_valid=?, dtreturn_cache_n_nan=?, dtreturn_cache_sample=?, "
                    "dtreturn_cache_max_abs=?, dtreturn_cache_max_signed=?, dtreturn_cache_min=?, dtreturn_cache_max=?, dtreturn_cache_avg=?, dtreturn_cache_p99_abs=? "
                    "WHERE rowid=?",
                    (n_valid, n_nan, json.dumps(sample), max_abs, max_signed, dt_min, dt_max, dt_avg, p99_abs, rid)
                )
                _write_results_t_mean(conn, rid, df_full, stf, etf)
                updated_full_cache += 1

                # 2) classification + transition detection
                notes = r.get('Notes') or ''
                test_cond = r.get('test_cond') or ''
                cycle_type, start_marker = _classify_entry(notes, test_cond)
                conn.execute(
                    "UPDATE results SET cycle_type=?, cycle_start_marker=? WHERE rowid=?",
                    (cycle_type, start_marker, rid)
                )

                is_manual_trans = (r.get('pelec_transition_source') == 'manual')
                trans_time = None
                if is_manual_trans:
                    trans_time = r.get('pelec_transition_time')
                elif cycle_type is not None and cycle_type != 'other' and start_marker in ('defrost_start', 'defrost_end', 'off_start', 'on_start'):
                    df_slice = df_full[(df_full['time_elapsed'] >= stf) & (df_full['time_elapsed'] <= etf)].copy()
                    if (not df_slice.empty) and (PELEC_COL in df_slice.columns):
                        pelec = df_slice[PELEC_COL].dropna()
                        t_elapsed = df_slice.loc[pelec.index, 'time_elapsed']
                        if len(pelec) >= 10:
                            if start_marker == 'defrost_start':
                                trans_time = _detect_pelec_rampup(pelec, t_elapsed)
                            elif start_marker == 'defrost_end':
                                trans_time = _detect_pelec_drop(pelec, t_elapsed)
                            elif start_marker == 'off_start':
                                trans_time = _detect_pelec_rampup(pelec, t_elapsed)
                            elif start_marker == 'on_start':
                                trans_time = _detect_pelec_drop(pelec, t_elapsed)

                # Whether this entry is one we attempt to detect a transition for.
                is_detectable_cycle = (
                    cycle_type is not None and cycle_type != 'other'
                    and start_marker in ('defrost_start', 'defrost_end', 'off_start', 'on_start')
                )
                if is_manual_trans:
                    pass  # preserve manual transition and source
                elif trans_time is None and is_detectable_cycle:
                    failed_detection += 1
                    conn.execute(
                        "UPDATE results SET pelec_transition_time=NULL, pelec_transition_source='auto' WHERE rowid=?",
                        (rid,)
                    )
                else:
                    if trans_time is not None:
                        detected += 1
                        conn.execute(
                            "UPDATE results SET pelec_transition_time=?, pelec_transition_source='auto' WHERE rowid=?",
                            (trans_time, rid)
                        )
                    else:
                        trans_time = r.get('pelec_transition_time')

                # 3) multi-buffer period rows + per-period caches
                periods_inserted = 0
                if is_manual_trans and trans_time is not None:
                    existing = conn.execute(
                        "SELECT id, start_time, end_time FROM cycle_periods WHERE entry_rowid=?",
                        (rid,)
                    ).fetchall()
                    for pr in existing:
                        _refresh_cycle_period_row_stats(
                            conn, int(pr['id']), df_full, float(pr['start_time']), float(pr['end_time'])
                        )
                        updated_period_cache += 1
                    periods_inserted = len(existing)
                elif trans_time is not None and start_marker in ('defrost_start', 'defrost_end'):
                    conn.execute(
                        "DELETE FROM cycle_periods WHERE entry_rowid=? AND detection_method='auto_pelec'",
                        (rid,)
                    )
                    for buf_s in BUFFER_OPTIONS:
                        periods = _generate_periods_for_entry(rid, start_marker, stf, etf, float(trans_time), buffer_s=int(buf_s))
                        for p in periods:
                            _insert_cycle_period_row(conn, rid, p, df_full, 'auto_pelec', int(buf_s))
                            updated_period_cache += 1
                            periods_inserted += 1
                elif trans_time is not None and start_marker in ('off_start', 'on_start'):
                    conn.execute(
                        "DELETE FROM cycle_periods WHERE entry_rowid=? AND detection_method='auto_pelec'",
                        (rid,)
                    )
                    periods = _generate_periods_for_entry(rid, start_marker, stf, etf, float(trans_time), buffer_s=0)
                    for p in periods:
                        _insert_cycle_period_row(conn, rid, p, df_full, 'auto_pelec', 0)
                        updated_period_cache += 1
                        periods_inserted += 1

                # Record a terminal "could not process" flag so this entry is not
                # re-queued by every incremental "Update cache". Manual entries are
                # never marked failed here.
                if is_manual_trans:
                    detect_failed_flag = 0
                elif is_detectable_cycle:
                    detect_failed_flag = 1 if (trans_time is None or periods_inserted == 0) else 0
                else:
                    detect_failed_flag = 0
                conn.execute("UPDATE results SET pelec_detect_failed=? WHERE rowid=?", (detect_failed_flag, rid))

                processed_entries += 1
                if limit_entries is not None and processed_entries >= limit_entries:
                    conn.commit()
                    break
                if processed_entries > 0 and processed_entries % 50 == 0:
                    conn.commit()
                    elapsed = _time.time() - t0
                    print(f"[pipeline] processed {processed_entries}/{queued_entries} in queue "
                          f"(results rows in DB={total_results_rows}) "
                          f"full_cache_updates={updated_full_cache} period_rows={updated_period_cache} "
                          f"detected={detected} failed={failed_detection} "
                          f"({elapsed:.1f}s, {len(_excel_cache)} Excel sheets in LRU)",
                          flush=True)

            conn.commit()
        finally:
            conn.close()

        elapsed = _time.time() - t0
        print(
            f"[pipeline] done processed={processed_entries} queued_was={queued_entries} "
            f"db_rows={total_results_rows} force_all={force_all} ({elapsed:.1f}s)",
            flush=True,
        )
        return jsonify({
            'success': True,
            'force_all': force_all,
            'limit_entries': limit_entries,
            'processed_entries': processed_entries,
            'skipped_entries': skipped_entries,
            'updated_full_cache': updated_full_cache,
            'updated_period_rows': updated_period_cache,
            'detected': detected,
            'failed_detection': failed_detection,
            'elapsed_s': round(elapsed, 1),
            'buffers': BUFFER_OPTIONS,
            'total_results_rows': total_results_rows,
            'pipeline_queue_len': queued_entries,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/diagnose_setpoint_values', methods=['GET'])
def diagnose_setpoint_values():
    """Diagnostic route to check which entries have NULL setpoint values"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check column types
        cursor.execute("PRAGMA table_info(results)")
        columns_info = cursor.fetchall()
        column_types = {col[1]: col[2] for col in columns_info}
        
        # Find entries with NULL setpoint values
        cursor.execute("""
            SELECT rowid, file_name, data_set, dev_test_condition,
                   dev_db_setpoint, dev_wb_setpoint, dev_tsup_setpoint
            FROM results
            WHERE dev_db_setpoint IS NULL 
               OR dev_wb_setpoint IS NULL 
               OR dev_tsup_setpoint IS NULL
            ORDER BY rowid
            LIMIT 50
        """)
        null_entries = cursor.fetchall()
        
        # Count total entries
        cursor.execute("SELECT COUNT(*) FROM results")
        total_count = cursor.fetchone()[0]
        
        # Count entries with NULL setpoints
        cursor.execute("""
            SELECT COUNT(*) FROM results
            WHERE dev_db_setpoint IS NULL 
               OR dev_wb_setpoint IS NULL 
               OR dev_tsup_setpoint IS NULL
        """)
        null_count = cursor.fetchone()[0]
        
        conn.close()
        
        result = {
            'column_types': {
                'dev_db_setpoint': column_types.get('dev_db_setpoint', 'NOT FOUND'),
                'dev_wb_setpoint': column_types.get('dev_wb_setpoint', 'NOT FOUND'),
                'dev_tsup_setpoint': column_types.get('dev_tsup_setpoint', 'NOT FOUND')
            },
            'total_entries': total_count,
            'entries_with_null_setpoints': null_count,
            'sample_null_entries': [
                {
                    'rowid': row[0],
                    'file_name': row[1],
                    'data_set': row[2],
                    'test_condition': row[3],
                    'dev_db_setpoint': row[4],
                    'dev_wb_setpoint': row[5],
                    'dev_tsup_setpoint': row[6]
                }
                for row in null_entries
            ]
        }
        
        return jsonify({
            'success': True,
            'diagnosis': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

def calculate_entry_deviations(file_name, data_set, start_time, end_time, warning_list=None, hp_id=None, entry=None):
    """Calculate deviation statistics for a single entry
    
    Args:
        file_name: Name of the file
        data_set: Dataset number
        start_time: Start time
        end_time: End time
        warning_list: Optional list to append warning messages to
        hp_id: Optional HP id for flow set lookup (fixed-flow: mean flow dev %, flow violations)
        entry: Optional results-row dict (climate, application, test_cond, profile_id)
    
    Returns:
        Dictionary with statistics or None if calculation failed
    """
    try:
        entry = dict(entry) if entry else {}
        if not entry:
            try:
                conn_meta = get_db_connection()
                try:
                    row_meta = conn_meta.execute(
                        "SELECT * FROM results WHERE file_name=? AND CAST(data_set AS TEXT)=CAST(? AS TEXT) "
                        "AND start_time=? AND end_time=?",
                        (file_name, data_set, start_time, end_time),
                    ).fetchone()
                    if row_meta:
                        merged = dict(row_meta)
                        merged.update({k: v for k, v in entry.items() if v is not None and v != ''})
                        entry = merged
                finally:
                    conn_meta.close()
            except Exception:
                pass
        # Extract test condition from stored metadata, else filename
        # Try multiple strategies to find the test condition letter (E, A, F, B, C, D, G)
        test_condition = unit_config.condition_letter(
            entry.get('dev_test_condition') or entry.get('test_cond')
        )
        
        # Strategy 1: Standard format <unit>_<lab>_E -> 'E' (third part, first character)
        parts = file_name.split('_')
        if test_condition is None and len(parts) >= 3 and len(parts[2]) > 0:
            potential = parts[2][0]
            if potential in DEVIATION_BANDS:
                test_condition = potential
        
        # Strategy 2: Look for test condition after space (e.g., "14825 A.xlsx" or "14825 A DPfix1.xlsx")
        if test_condition is None:
            # Remove file extension and look for space followed by a valid test condition
            name_without_ext = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
            # Look for pattern: number/space/letter or space/letter
            import re
            # Match space followed by one of the valid test conditions
            match = re.search(r'\s+([A-G])', name_without_ext, re.IGNORECASE)
            if match:
                potential = match.group(1).upper()
                if potential in DEVIATION_BANDS:
                    test_condition = potential
        
        # Strategy 3: Look for test condition anywhere in the filename (as a standalone letter)
        if test_condition is None:
            # Find any occurrence of valid test condition letters
            for letter in DEVIATION_BANDS.keys():
                # Look for the letter as a standalone character (not part of a word)
                pattern = r'(?:^|[\s_\-])' + letter + r'(?:[\s_\-]|\.|$)'
                if re.search(pattern, file_name, re.IGNORECASE):
                    test_condition = letter
                    break
        
        if test_condition is None:
            warning = f"Cannot calculate deviations for {file_name}: Could not extract test condition from filename. Expected one of: {list(DEVIATION_BANDS.keys())}. File name format should contain the test condition letter (E, A, F, B, C, D, G)."
            print(f"WARNING: {warning}")
            if warning_list is not None:
                warning_list.append(warning)
            return None
        
        # Load time series data (pass data_set for correct database lookup)
        df = load_time_series_data(file_name, start_time, end_time, data_set=data_set)
        if df is None or df.empty:
            warning = f"Cannot calculate deviations for {file_name}: Time series data is None or empty"
            print(f"WARNING: {warning}")
            if warning_list is not None:
                warning_list.append(warning)
            return None
        
        # Which checks this unit type is scored on (checks.json × unit_types.json).
        entry_for_checks = dict(entry or {})
        entry_for_checks.setdefault('file_name', file_name)
        if hp_id is not None:
            entry_for_checks.setdefault('hp_id', hp_id)
        applicable_checks = _entry_applicable_checks(entry_for_checks)

        # Outdoor air columns. A water-source unit has no outdoor air at all, and an
        # air-source file may simply not carry the series: either way DB/WB stay empty
        # and the water-side checks (Tsup, dTreturn, flow) are still calculated.
        db_applicable = 'db' in applicable_checks and 'T_outdoor (DB)' in df.columns
        wb_applicable = 'wb' in applicable_checks and 'T_outdoor (WB)' in df.columns
        if 'db' in applicable_checks and 'T_outdoor (DB)' not in df.columns:
            available_cols = [col for col in df.columns if 'outdoor' in col.lower() or 'db' in col.lower() or 'wb' in col.lower()]
            warning = f"{file_name}: 'T_outdoor (DB)' column not found, dry-bulb and wet-bulb checks left empty. Available temperature columns: {available_cols}"
            print(f"WARNING: {warning}")
            if warning_list is not None:
                warning_list.append(warning)
        
        # Filter data for the time range
        cycle_data = df[(df['time_elapsed'] >= start_time) & (df['time_elapsed'] <= end_time)].copy()
        if cycle_data.empty:
            warning = f"Cannot calculate deviations for {file_name}: No data in time range [{start_time}, {end_time}]. DataFrame has {len(df)} rows total."
            print(f"WARNING: {warning}")
            if warning_list is not None:
                warning_list.append(warning)
            return None
        
        # Check drybulb temperatures outside permissible deviation band
        climate, application = _entry_climate_application({
            **entry,
            'file_name': file_name,
            'hp_id': hp_id if hp_id is not None else entry.get('hp_id'),
        })
        db_setpoint = unit_config.get_tdb_setpoint_for(
            test_condition, climate=climate, application=application,
            condition_set_id=entry.get('condition_set_id'),
        )
        if db_setpoint is None:
            db_setpoint = unit_config.get_tdb_setpoint_for(test_condition)
        db_bands = DEVIATION_BANDS.get(test_condition)
        if db_setpoint is not None and not db_bands:
            db_bands = calculate_deviation_bands({test_condition: db_setpoint}, DB_BAND, WB_BAND).get(test_condition)
        if db_setpoint is None or not db_bands:
            warning = (
                f"{file_name}: no outdoor setpoint for condition {test_condition} "
                f"(climate={climate}, application={application})."
            )
            print(f"WARNING: {warning}")
            if warning_list is not None:
                warning_list.append(warning)
            if db_applicable:
                return None
            db_bands = None

        total_points = len(cycle_data)

        # Dry-bulb: air-source units only, and only when the series is present.
        db_count = None
        db_percentage = None
        max_db_deviation = None
        max_db_pos = None
        max_db_neg = None
        if db_applicable and db_bands:
            if not _has_valid_points(cycle_data['T_outdoor (DB)']):
                _note_missing_quantity(file_name, data_set, 'dry-bulb temperature', warning_list)
            else:
                db_outside_band = cycle_data[
                    (cycle_data['T_outdoor (DB)'] < db_bands['DB'][0]) |
                    (cycle_data['T_outdoor (DB)'] > db_bands['DB'][1])
                ]
                db_count = len(db_outside_band)
                db_percentage = (db_count / total_points * 100) if total_points > 0 else None
            # Signed: positive = above setpoint, negative = below (point furthest from setpoint)
            db_diff = (cycle_data['T_outdoor (DB)'] - db_setpoint).dropna()
            max_db_deviation = float(db_diff.loc[db_diff.abs().idxmax()]) if len(db_diff) > 0 else None
            max_db_pos, max_db_neg = _signed_extremes(db_diff)

        # Wet-bulb: air-source units with a wet-bulb sensor.
        # Always treat "T_outdoor (WB)" as wetbulb temperature (no detection/conversion)
        wb_setpoint = None
        wb_count = None
        wb_percentage = None
        max_wb_deviation = None
        max_wb_pos = None
        max_wb_neg = None
        if wb_applicable and db_bands:
            wb_setpoint = db_setpoint - 1
            if _has_valid_points(cycle_data['T_outdoor (WB)']):
                wb_outside_band = cycle_data[
                    (cycle_data['T_outdoor (WB)'] < db_bands['WB'][0]) |
                    (cycle_data['T_outdoor (WB)'] > db_bands['WB'][1])
                ]
                wb_count = len(wb_outside_band)
                wb_percentage = (wb_count / total_points * 100) if total_points > 0 else None
                wb_diff = cycle_data['T_outdoor (WB)'] - wb_setpoint
                max_wb_deviation = float(wb_diff.loc[wb_diff.abs().idxmax()]) if len(cycle_data) > 0 else None
                max_wb_pos, max_wb_neg = _signed_extremes(wb_diff)

        # Calculate average timestep size
        if len(cycle_data) > 1:
            avg_timestep = (cycle_data['time_elapsed'].iloc[-1] - cycle_data['time_elapsed'].iloc[0]) / (len(cycle_data) - 1)
        else:
            avg_timestep = 0
        
        # Calculate tsup_setpoint based on test condition × climate × application
        tsup_setpoint = get_tsup_setpoint(
            test_condition, file_name,
            condition_set_id=entry.get('condition_set_id'),
            climate=climate, application=application,
        )

        # Supply temperature violations: symmetric band around tsup_setpoint.
        # Prefer 'Ts Buh' (same column choice as the Tsup deviation plot and the mean Tsup column).
        tsup_count = None
        tsup_percentage = None
        max_tsup_deviation = None
        max_tsup_pos = None
        max_tsup_neg = None
        tsup_band = _get_tsup_band()
        tsup_column = get_tsup_series_column(cycle_data)
        if tsup_column and tsup_setpoint is not None:
            tsup_series = cycle_data[tsup_column]
            if not _has_valid_points(tsup_series):
                _note_missing_quantity(file_name, data_set, 'supply temperature', warning_list)
            else:
                tsup_outside = tsup_series[
                    (tsup_series < tsup_setpoint - tsup_band) | (tsup_series > tsup_setpoint + tsup_band)
                ]
                tsup_count = int(len(tsup_outside))
                tsup_percentage = (tsup_count / total_points * 100) if total_points > 0 else None
            tsup_diff = (tsup_series - tsup_setpoint).dropna()
            if len(tsup_diff) > 0:
                max_tsup_deviation = float(tsup_diff.loc[tsup_diff.abs().idxmax()])
            max_tsup_pos, max_tsup_neg = _signed_extremes(tsup_diff)

        # dTreturn violations: T_return_emu - T_return_calc (asymmetric bounds from config)
        dtreturn_count = None
        dtreturn_percentage = None
        max_dtreturn_deviation = None
        max_dtreturn_pos = None
        max_dtreturn_neg = None
        dt_cfg = load_permissible_deviations().get('dTreturn', {'lower': -2.0, 'upper': 2.0})
        try:
            dt_lower = float(dt_cfg.get('lower', -2.0))
            dt_upper = float(dt_cfg.get('upper', 2.0))
        except Exception:
            dt_lower, dt_upper = -2.0, 2.0
        if 'T_return_emu' in cycle_data.columns and 'T_return_calc' in cycle_data.columns:
            dtreturn = cycle_data['T_return_emu'] - cycle_data['T_return_calc']
            if not _has_valid_points(dtreturn):
                _note_missing_quantity(file_name, data_set, 'return temperature difference (dTreturn)', warning_list)
            else:
                dtreturn_outside = dtreturn[(dtreturn < dt_lower) | (dtreturn > dt_upper)]
                dtreturn_count = int(len(dtreturn_outside))
                dtreturn_percentage = (dtreturn_count / total_points * 100) if total_points > 0 else None
            dtreturn_nonan = dtreturn.dropna()
            if len(dtreturn_nonan) > 0:
                max_dtreturn_deviation = float(dtreturn_nonan.loc[dtreturn_nonan.abs().idxmax()])
            else:
                max_dtreturn_deviation = None
            max_dtreturn_pos, max_dtreturn_neg = _signed_extremes(dtreturn)
        
        # Flow (fixed-flow only). Variable-flow tests must not be scored against a unit flow set.
        flow_setpoint = None
        flow_set_type = None
        mean_flow_dev_pct = None
        flow_outside_count = None
        flow_percentage = None
        max_flow_pos_pct = None
        max_flow_neg_pct = None
        flow_instantaneous_pct = load_permissible_deviations().get('flow_instantaneous_pct', 2.5)
        entry_for_flow = dict(entry or {})
        entry_for_flow.setdefault('file_name', file_name)
        if hp_id is not None:
            entry_for_flow.setdefault('hp_id', hp_id)
        if not _entry_is_variable_flow(entry_for_flow):
            flow_type, flow_set_value = get_flow_set_for_hp(
                hp_id, file_name=file_name, profile_id=entry_for_flow.get('profile_id')
            ) if (hp_id or file_name) else (None, None)
            if flow_type and flow_set_value is not None and flow_set_value > 0:
                flow_col = 'mass flow' if flow_type == 'mass' else 'volume flow'
                if flow_col in cycle_data.columns:
                    flow_setpoint = flow_set_value
                    flow_set_type = flow_type
                    if not _has_valid_points(cycle_data[flow_col]):
                        _note_missing_quantity(file_name, data_set, f'flow ({flow_col})', warning_list)
                    else:
                        mean_flow = float(cycle_data[flow_col].mean())
                        mean_flow_dev_pct = round(100.0 * (mean_flow - flow_set_value) / flow_set_value, 2) if flow_set_value else None
                        band_low = flow_set_value * (1 - flow_instantaneous_pct / 100.0)
                        band_high = flow_set_value * (1 + flow_instantaneous_pct / 100.0)
                        flow_outside = cycle_data[(cycle_data[flow_col] < band_low) | (cycle_data[flow_col] > band_high)]
                        flow_outside_count = len(flow_outside)
                        flow_percentage = round(100.0 * flow_outside_count / total_points, 2) if total_points > 0 else None
                        flow_dev_pct = 100.0 * (cycle_data[flow_col] - flow_set_value) / flow_set_value
                        max_flow_pos_pct, max_flow_neg_pct = _signed_extremes(flow_dev_pct)
        
        return {
            'test_condition': test_condition,
            'climate': climate,
            'application': application,
            'db_setpoint': db_setpoint,
            'wb_setpoint': wb_setpoint,
            'tsup_setpoint': tsup_setpoint,
            'total_points': total_points,
            'db_outside_band_count': db_count,
            'wb_outside_band_count': wb_count,  # Can be None if no WB data
            'dtreturn_outside_band_count': dtreturn_count,  # Can be None if missing columns
            'dtreturn_percentage': dtreturn_percentage,      # Can be None if missing columns
            'db_percentage': db_percentage,
            'wb_percentage': wb_percentage,  # Can be None if no WB data
            'avg_timestep_size': avg_timestep,
            'max_db_deviation': max_db_deviation,
            'max_wb_deviation': max_wb_deviation,  # Can be None if no WB data
            'tsup_outside_band_count': tsup_count,  # Can be None if no supply temp column
            'tsup_percentage': tsup_percentage,     # Can be None if no supply temp column
            'max_tsup_deviation': max_tsup_deviation,  # Can be None if no supply temp column
            'max_dtreturn_deviation': max_dtreturn_deviation,  # Can be None if missing columns
            # Signed extremes: largest excursion above (pos) and below (neg) the setpoint
            'max_db_pos': max_db_pos,
            'max_db_neg': max_db_neg,
            'max_wb_pos': max_wb_pos,
            'max_wb_neg': max_wb_neg,
            'max_tsup_pos': max_tsup_pos,
            'max_tsup_neg': max_tsup_neg,
            'max_dtreturn_pos': max_dtreturn_pos,
            'max_dtreturn_neg': max_dtreturn_neg,
            'max_flow_pos_pct': max_flow_pos_pct,
            'max_flow_neg_pct': max_flow_neg_pct,
            'wb_calculated_from_rh': False,  # Always False - no longer calculating from RH
            'flow_setpoint': flow_setpoint,
            'flow_set_type': flow_set_type,
            'mean_flow_dev_pct': mean_flow_dev_pct,
            'flow_outside_count': flow_outside_count,
            'flow_percentage': flow_percentage,
            'cycle_data': cycle_data  # Keep for plotting
        }
    except Exception as e:
        print(f"Error calculating deviations for {file_name}, dataset {data_set}: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_entry_deviation_metrics(file_name, data_set, start_time, end_time, metrics, warning_list=None, hp_id=None, entry=None):
    """
    Calculate a subset of deviation metrics for a single entry.
    This is designed to power per-column update buttons on the deviations page.
    It avoids requiring unrelated columns (e.g. dtreturn does not require DB/WB).
    """
    metrics = set(metrics or [])
    if not metrics:
        return None

    # Load time series data first (required for all supported metrics)
    df = load_time_series_data(file_name, start_time, end_time, data_set=data_set)
    if df is None or df.empty:
        warning = f"Cannot calculate metrics for {file_name}: Time series data is None or empty"
        if warning_list is not None:
            warning_list.append(warning)
        return None

    cycle_data = df[(df['time_elapsed'] >= start_time) & (df['time_elapsed'] <= end_time)].copy()
    if cycle_data.empty:
        warning = f"Cannot calculate metrics for {file_name}: No data in time range [{start_time}, {end_time}]"
        if warning_list is not None:
            warning_list.append(warning)
        return None

    out = {'cycle_data': cycle_data}
    total_points = len(cycle_data)

    # Which checks this unit type is scored on (checks.json × unit_types.json).
    # Inapplicable ones are simply not written to `out`, so the caller stores NULL.
    entry_for_checks = dict(entry or {})
    entry_for_checks.setdefault('file_name', file_name)
    if hp_id is not None:
        entry_for_checks.setdefault('hp_id', hp_id)
    applicable_checks = _entry_applicable_checks(entry_for_checks)

    # Metrics that require test_condition / DB setpoint bands
    needs_db_bands = 'db' in applicable_checks and any(m in metrics for m in ('db_outside', 'db_pct', 'max_db', 'db_extremes'))
    needs_wb_bands = 'wb' in applicable_checks and any(m in metrics for m in ('wb_outside', 'wb_pct', 'max_wb', 'wb_extremes'))
    needs_tsup_bands = any(m in metrics for m in ('tsup_outside', 'tsup_pct', 'max_tsup', 'tsup_extremes'))

    test_condition = None
    db_setpoint = None
    wb_setpoint = None
    if needs_db_bands or needs_wb_bands or needs_tsup_bands:
        # Reuse existing filename parsing logic from calculate_entry_deviations (simplified)
        parts = file_name.split('_')
        if len(parts) >= 3 and len(parts[2]) > 0 and parts[2][0] in DEVIATION_BANDS:
            test_condition = parts[2][0]
        if not test_condition:
            for letter in DEVIATION_BANDS.keys():
                pattern = r'(?:^|[\s_\-])' + letter + r'(?:[\s_\-]|\.|$)'
                if re.search(pattern, file_name, re.IGNORECASE):
                    test_condition = letter
                    break
        if not test_condition:
            warning = f"Cannot extract test condition from filename for {file_name}"
            if warning_list is not None:
                warning_list.append(warning)
        else:
            db_setpoint = unit_config.get_tdb_setpoint_for(test_condition)
            wb_setpoint = (db_setpoint - 1) if db_setpoint is not None else None
            out['test_condition'] = test_condition
            out['db_setpoint'] = db_setpoint
            out['wb_setpoint'] = wb_setpoint
            out['tsup_setpoint'] = _tsup_setpoint_from_entry(
                dict(entry or {}, file_name=file_name, hp_id=hp_id),
                test_condition, file_name,
            )

    # Total points / timestep
    if 'total_points' in metrics:
        out['total_points'] = total_points
    if 'avg_timestep' in metrics:
        if len(cycle_data) > 1:
            out['avg_timestep_size'] = (cycle_data['time_elapsed'].iloc[-1] - cycle_data['time_elapsed'].iloc[0]) / (len(cycle_data) - 1)
        else:
            out['avg_timestep_size'] = 0.0

    # DB metrics
    if needs_db_bands and test_condition and db_setpoint is not None:
        if 'T_outdoor (DB)' in cycle_data.columns:
            db_has_data = _has_valid_points(cycle_data['T_outdoor (DB)'])
            if not db_has_data:
                _note_missing_quantity(file_name, data_set, 'dry-bulb temperature', warning_list)
            db_band = DEVIATION_BANDS[test_condition]['DB']
            db_outside = cycle_data[(cycle_data['T_outdoor (DB)'] < db_band[0]) | (cycle_data['T_outdoor (DB)'] > db_band[1])]
            db_count = int(len(db_outside)) if db_has_data else None
            if 'db_outside' in metrics:
                out['db_outside_band_count'] = db_count
            if 'db_pct' in metrics:
                out['db_percentage'] = (db_count / total_points * 100) if (db_count is not None and total_points > 0) else None
            if 'max_db' in metrics:
                db_diff = (cycle_data['T_outdoor (DB)'] - db_setpoint).dropna()
                out['max_db_deviation'] = float(db_diff.loc[db_diff.abs().idxmax()]) if len(db_diff) > 0 else None
            if 'db_extremes' in metrics:
                out['max_db_pos'], out['max_db_neg'] = _signed_extremes(cycle_data['T_outdoor (DB)'] - db_setpoint)

    # WB metrics
    if needs_wb_bands and test_condition and wb_setpoint is not None:
        if 'T_outdoor (WB)' in cycle_data.columns:
            wb_has_data = _has_valid_points(cycle_data['T_outdoor (WB)'])
            if not wb_has_data:
                _note_missing_quantity(file_name, data_set, 'wet-bulb temperature', warning_list)
            wb_band = DEVIATION_BANDS[test_condition]['WB']
            wb_outside = cycle_data[(cycle_data['T_outdoor (WB)'] < wb_band[0]) | (cycle_data['T_outdoor (WB)'] > wb_band[1])]
            wb_count = int(len(wb_outside)) if wb_has_data else None
            if 'wb_outside' in metrics:
                out['wb_outside_band_count'] = wb_count
            if 'wb_pct' in metrics:
                out['wb_percentage'] = (wb_count / total_points * 100) if (wb_count is not None and total_points > 0) else None
            if 'max_wb' in metrics:
                wb_diff = (cycle_data['T_outdoor (WB)'] - wb_setpoint).dropna()
                out['max_wb_deviation'] = float(wb_diff.loc[wb_diff.abs().idxmax()]) if len(wb_diff) > 0 else None
            if 'wb_extremes' in metrics:
                out['max_wb_pos'], out['max_wb_neg'] = _signed_extremes(cycle_data['T_outdoor (WB)'] - wb_setpoint)

    # Tsup metrics (symmetric band around the supply temperature setpoint)
    if needs_tsup_bands and out.get('tsup_setpoint') is not None:
        tsup_column = get_tsup_series_column(cycle_data)
        if tsup_column:
            tsup_setpoint = out['tsup_setpoint']
            tsup_band = _get_tsup_band()
            tsup_series = cycle_data[tsup_column]
            tsup_has_data = _has_valid_points(tsup_series)
            if not tsup_has_data:
                _note_missing_quantity(file_name, data_set, 'supply temperature', warning_list)
            tsup_count = int(len(tsup_series[
                (tsup_series < tsup_setpoint - tsup_band) | (tsup_series > tsup_setpoint + tsup_band)
            ])) if tsup_has_data else None
            if 'tsup_outside' in metrics:
                out['tsup_outside_band_count'] = tsup_count
            if 'tsup_pct' in metrics:
                out['tsup_percentage'] = (tsup_count / total_points * 100) if (tsup_count is not None and total_points > 0) else None
            if 'max_tsup' in metrics:
                tsup_diff = (tsup_series - tsup_setpoint).dropna()
                out['max_tsup_deviation'] = float(tsup_diff.loc[tsup_diff.abs().idxmax()]) if len(tsup_diff) > 0 else None
            if 'tsup_extremes' in metrics:
                out['max_tsup_pos'], out['max_tsup_neg'] = _signed_extremes(tsup_series - tsup_setpoint)

    # dTreturn metrics (asymmetric band)
    if any(m in metrics for m in ('dtreturn_outside', 'dtreturn_pct', 'max_dtreturn', 'dtreturn_extremes')):
        if 'T_return_emu' in cycle_data.columns and 'T_return_calc' in cycle_data.columns:
            dt_cfg = load_permissible_deviations().get('dTreturn', {'lower': -2.0, 'upper': 2.0})
            try:
                dt_lower = float(dt_cfg.get('lower', -2.0))
                dt_upper = float(dt_cfg.get('upper', 2.0))
            except Exception:
                dt_lower, dt_upper = -2.0, 2.0
            dtreturn = cycle_data['T_return_emu'] - cycle_data['T_return_calc']
            dt_has_data = _has_valid_points(dtreturn)
            if not dt_has_data:
                _note_missing_quantity(file_name, data_set, 'return temperature difference (dTreturn)', warning_list)
            dt_outside = dtreturn[(dtreturn < dt_lower) | (dtreturn > dt_upper)]
            dt_count = int(len(dt_outside)) if dt_has_data else None
            if 'dtreturn_outside' in metrics:
                out['dtreturn_outside_band_count'] = dt_count
            if 'dtreturn_pct' in metrics:
                out['dtreturn_percentage'] = (dt_count / total_points * 100) if (dt_count is not None and total_points > 0) else None
            if 'max_dtreturn' in metrics:
                dt_nonan = dtreturn.dropna()
                out['max_dtreturn_deviation'] = float(dt_nonan.loc[dt_nonan.abs().idxmax()]) if len(dt_nonan) > 0 else None
            if 'dtreturn_extremes' in metrics:
                out['max_dtreturn_pos'], out['max_dtreturn_neg'] = _signed_extremes(dtreturn)

    # Flow metrics (fixed-flow set only)
    if any(m in metrics for m in ('flow_outside', 'flow_pct', 'mean_flow_dev', 'flow_extremes')):
        entry_for_flow = dict(entry or {})
        entry_for_flow.setdefault('file_name', file_name)
        if hp_id is not None:
            entry_for_flow.setdefault('hp_id', hp_id)
        if not _entry_is_variable_flow(entry_for_flow):
            flow_instantaneous_pct = load_permissible_deviations().get('flow_instantaneous_pct', 2.5)
            flow_type, flow_set_value = get_flow_set_for_hp(
                hp_id, file_name=file_name, profile_id=entry_for_flow.get('profile_id')
            ) if (hp_id or file_name) else (None, None)
            if flow_type and flow_set_value is not None and flow_set_value > 0:
                flow_col = 'mass flow' if flow_type == 'mass' else 'volume flow'
                if flow_col in cycle_data.columns:
                    flow_has_data = _has_valid_points(cycle_data[flow_col])
                    if not flow_has_data:
                        _note_missing_quantity(file_name, data_set, f'flow ({flow_col})', warning_list)
                    mean_flow = float(cycle_data[flow_col].mean()) if flow_has_data else None
                    band_low = flow_set_value * (1 - flow_instantaneous_pct / 100.0)
                    band_high = flow_set_value * (1 + flow_instantaneous_pct / 100.0)
                    flow_outside = cycle_data[(cycle_data[flow_col] < band_low) | (cycle_data[flow_col] > band_high)]
                    flow_outside_count = int(len(flow_outside)) if flow_has_data else None
                    if 'mean_flow_dev' in metrics:
                        out['mean_flow_dev_pct'] = (round(100.0 * (mean_flow - flow_set_value) / flow_set_value, 2)
                                                    if (flow_has_data and flow_set_value) else None)
                    if 'flow_outside' in metrics:
                        out['flow_outside_count'] = flow_outside_count
                    if 'flow_pct' in metrics:
                        out['flow_percentage'] = (round(100.0 * flow_outside_count / total_points, 2)
                                                  if (flow_outside_count is not None and total_points > 0) else None)
                    if 'flow_extremes' in metrics:
                        out['max_flow_pos_pct'], out['max_flow_neg_pct'] = _signed_extremes(
                            100.0 * (cycle_data[flow_col] - flow_set_value) / flow_set_value)

    return out

def safe_int(value, default=None):
    """Safely convert a value to int, handling bytes, None, and other types"""
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            return int(value.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=None):
    """Safely convert a value to float, handling bytes, None, and other types"""
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            return float(value.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

@app.route('/deviations')
def deviations():
    """Display permissible deviation analysis page - loads from database"""
    # Reload permissible deviations config to ensure latest values
    global PERMISSIBLE_DEVIATIONS, DB_BAND, WB_BAND, TSUP_BAND, DEVIATION_BANDS
    PERMISSIBLE_DEVIATIONS = load_permissible_deviations()
    DB_BAND = PERMISSIBLE_DEVIATIONS.get('DB', {}).get('value', 1.0)
    WB_BAND = PERMISSIBLE_DEVIATIONS.get('WB', {}).get('value', 1.0)
    TSUP_BAND = PERMISSIBLE_DEVIATIONS.get('Tsup', {}).get('value', 0.5)
    DEVIATION_BANDS = calculate_deviation_bands(DEVIATION_SETPOINTS, DB_BAND, WB_BAND)
    
    # Initialize deviation columns if needed
    init_deviation_columns()
    
    # Load permissible deviations for template
    deviations_config = load_permissible_deviations()
    
    conn = get_db_connection()
    try:
        # Get ALL entries from database (not just those with statistics)
        results = conn.execute('''
            SELECT *, rowid FROM results 
            ORDER BY display_order ASC, rowid ASC
        ''').fetchall()
        results = [dict(row) for row in results]
        
        # Find the actual column names for our custom columns
        column_name_map = {}
        if results:
            first_entry = results[0]
            all_keys = list(first_entry.keys())
            
            # Try exact matches first (case-insensitive and case-sensitive)
            for key in all_keys:
                key_lower = key.lower()
                # Exact matches for COP - try various case combinations
                if (key_lower == 'cop_dataset' or key_lower == 'cop dataset' or key_lower == 'cop-dataset' or
                    key == 'COP_dataset' or key == 'COP Dataset' or key == 'COP-Dataset' or
                    key == 'cop_dataset' or key == 'Cop_Dataset'):
                    if 'COP' not in column_name_map:
                        column_name_map['COP'] = key
                # Exact matches for Dpfix - try various case combinations
                if (key_lower == 'dpfix' or key_lower == 'dp_fix' or key_lower == 'dp-fix' or
                    key == 'DPfix' or key == 'DPFix' or key == 'DP_Fix' or key == 'DP-Fix' or
                    key == 'dpfix' or key == 'Dpfix'):
                    if 'Dpfix' not in column_name_map:
                        column_name_map['Dpfix'] = key
            
            # Then try flexible matching - check original key and normalized versions
            for key in all_keys:
                key_lower = key.lower()
                key_normalized = key_lower.replace('_', ' ').replace('-', ' ')
                
                # More flexible matching for COP - check for COP and Dataset separately
                # Also check the original key (case-sensitive) for "COP_dataset"
                if 'COP' not in column_name_map:
                    # Check if it contains both 'cop' and 'dataset' (any case, any separator)
                    has_cop = 'cop' in key_lower or 'COP' in key or 'Cop' in key
                    has_dataset = 'dataset' in key_lower or 'Dataset' in key or 'DATASET' in key
                    if has_cop and has_dataset:
                        column_name_map['COP'] = key
                
                # Check for Dpfix - check original key for "DPfix" or "DPFix"
                if 'Dpfix' not in column_name_map:
                    # Check various patterns
                    if ('dpfix' in key_lower or 'dp fix' in key_normalized or 'd.p.fix' in key_lower or 
                        'dp_fix' in key_lower or 'dp-fix' in key_lower or 
                        (key_lower.startswith('dp') and 'fix' in key_lower) or
                        'DPfix' in key or 'DPFix' in key or 'DP_Fix' in key):
                        column_name_map['Dpfix'] = key
                
                if 'flow' in key_lower and 'config' in key_lower and 'Flow' not in column_name_map:
                    column_name_map['Flow'] = key
                
                if 'lab' in key_lower and 'id' in key_lower and 'Lab' not in column_name_map:
                    column_name_map['Lab'] = key
                
                if 'hp' in key_lower and 'id' in key_lower and 'HP' not in column_name_map:
                    column_name_map['HP'] = key
                
                # COPCorrwBUH (COP corrected with BUH) — exclude uncorr (CopUncorrwBUH)
                if 'COPCorrwBUH' not in column_name_map and 'uncorr' not in key_lower and (
                    'copcorrwbuh' in key_lower or
                    ('cop' in key_lower and 'corr' in key_lower and 'buh' in key_lower) or
                    key == 'COPCorrwBUH' or key == 'cop_corrw_buh'):
                    column_name_map['COPCorrwBUH'] = key
                
                # Cop Carnot Corr — exclude uncorr (Cop Carnot Uncorr)
                if 'CopCarnotCorr' not in column_name_map and 'uncorr' not in key_lower and (
                    ('carnot' in key_lower and 'corr' in key_lower) or
                    key_lower == 'cop carnot corr' or key_lower == 'cop_carnot_corr' or
                    key == 'Cop Carnot Corr' or key == 'cop_carnot_corr'):
                    column_name_map['CopCarnotCorr'] = key
        
        init_hp_design_table()
        hp_key = column_name_map.get('HP')
        flow_key = column_name_map.get('Flow')
        # Design parameters panel: one row per unit, resolved the same way the
        # calculations resolve it, so the numbers shown are the effective ones.
        unit_design_rows, unresolved_entry_count = get_unit_design_rows(
            results=results, hp_key=hp_key, flow_key=flow_key, conn=conn
        )
        units_without_pdesign = [
            row['unit_label'] for row in unit_design_rows
            if row['entry_count'] > 0 and row['pdesign_kw'] is None
        ]
        
        # Convert database columns to the format expected by template
        deviation_stats = []
        for entry in results:
            # Calculate cycle duration
            start_time = entry.get('start_time')
            end_time = entry.get('end_time')
            cycle_duration = (end_time - start_time) if (start_time is not None and end_time is not None) else None
            
            # Check if statistics exist (dev_test_condition alone can be set for other purposes).
            has_stats = entry.get('dev_total_points') is not None and entry.get('dev_last_calculated') is not None
            
            if has_stats:
                # Entry has calculated statistics - convert all numeric values safely
                test_condition = entry.get('dev_test_condition')
                tsup_setpoint_db = safe_float(entry.get('dev_tsup_setpoint'))
                # If tsup_setpoint is missing (NULL) but we have test_condition, calculate it on-the-fly
                if tsup_setpoint_db is None and test_condition:
                    tsup_setpoint_db = _tsup_setpoint_from_entry(entry, test_condition, entry.get('file_name'))
                
                stats = {
                    'test_condition': test_condition,
                    'db_setpoint': safe_float(entry.get('dev_db_setpoint')),
                    'wb_setpoint': safe_float(entry.get('dev_wb_setpoint')),
                    'tsup_setpoint': tsup_setpoint_db,
                    'total_points': safe_int(entry.get('dev_total_points'), 0),
                    'db_outside_band_count': safe_int(entry.get('dev_db_outside_count')),  # None when no valid DB data
                    'wb_outside_band_count': safe_int(entry.get('dev_wb_outside_count')),  # Can be None
                    'tsup_outside_band_count': safe_int(entry.get('dev_tsup_outside_count')),  # Can be None
                    'dtreturn_outside_band_count': safe_int(entry.get('dev_dtreturn_outside_count')),  # Can be None
                    'db_percentage': safe_float(entry.get('dev_db_percentage')),  # None when no valid DB data
                    'wb_percentage': safe_float(entry.get('dev_wb_percentage')),  # Can be None
                    'tsup_percentage': safe_float(entry.get('dev_tsup_percentage')),  # Can be None
                    'dtreturn_percentage': safe_float(entry.get('dev_dtreturn_percentage')),  # Can be None
                    'max_db_deviation': safe_float(entry.get('dev_max_db_deviation')),
                    'max_wb_deviation': safe_float(entry.get('dev_max_wb_deviation')),  # Can be None
                    'max_tsup_deviation': safe_float(entry.get('dev_max_tsup_deviation')),  # Can be None
                    'max_dtreturn_deviation': safe_float(entry.get('dev_max_dtreturn_deviation')),  # Can be None
                    **{k: safe_float(entry.get(f'dev_{k}')) for k in SIGNED_EXTREME_KEYS},
                    'avg_timestep_size': safe_float(entry.get('dev_avg_timestep'), 0.0),
                    'wb_calculated_from_rh': bool(safe_int(entry.get('dev_wb_calculated_from_rh'), 0)),
                    'last_calculated': entry.get('dev_last_calculated'),
                    'flow_setpoint': safe_float(entry.get('dev_flow_setpoint')),
                    'flow_set_type': entry.get('dev_flow_set_type'),
                    'mean_flow_dev_pct': safe_float(entry.get('dev_mean_flow_dev_pct')),
                    'flow_outside_count': safe_int(entry.get('dev_flow_outside_count')),
                    'flow_percentage': safe_float(entry.get('dev_flow_percentage')),
                    'has_statistics': True
                }
            else:
                # Entry doesn't have statistics yet
                stats = {
                    'test_condition': None,
                    'db_setpoint': None,
                    'wb_setpoint': None,
                    'tsup_setpoint': None,
                    'total_points': None,
                    'db_outside_band_count': None,
                    'wb_outside_band_count': None,
                    'tsup_outside_band_count': None,
                    'dtreturn_outside_band_count': None,
                    'db_percentage': None,
                    'wb_percentage': None,
                    'tsup_percentage': None,
                    'dtreturn_percentage': None,
                    'max_db_deviation': None,
                    'max_wb_deviation': None,
                    'max_tsup_deviation': None,
                    'max_dtreturn_deviation': None,
                    **{k: None for k in SIGNED_EXTREME_KEYS},
                    'avg_timestep_size': None,
                    'wb_calculated_from_rh': False,
                    'last_calculated': None,
                    'flow_setpoint': None,
                    'flow_set_type': None,
                    'mean_flow_dev_pct': None,
                    'flow_outside_count': None,
                    'flow_percentage': None,
                    'has_statistics': False,
                    'qset_kw': None,
                    'mean_q_dev_kw': None,
                    'mean_q_dev_pct': None
                }
            
            stats['cycle_duration'] = cycle_duration
            
            # Display string for flow setpoint in setpoint cell (m_set 0.5 kg/s or v_set 1.2 m³/h)
            if stats.get('flow_setpoint') is not None and stats.get('flow_set_type'):
                u = 'kg/s' if stats['flow_set_type'] == 'mass' else 'm³/h'
                pre = 'm_set' if stats['flow_set_type'] == 'mass' else 'v_set'
                stats['flow_setpoint_display'] = f"{pre} {stats['flow_setpoint']:.3f} {u}"
            else:
                stats['flow_setpoint_display'] = None
            
            # Calculate mean DB, WB, and Tsup deviations from setpoints
            # Use Ts_buh (BUH) and QCorrwBUH for fairness: same as avg_t_supply / avg_heating_capacity_corr for non-BUH
            avg_t_db = entry.get('avg_t_db')
            avg_t_wb = entry.get('avg_t_wb')
            avg_t_supply = get_mean_supply_for_deviations(entry)
            
            # Mean DB deviation (K) — signed: positive = above setpoint, negative = below
            if avg_t_db is not None and stats.get('db_setpoint') is not None:
                stats['mean_db_deviation'] = round(avg_t_db - stats['db_setpoint'], 3)
            else:
                stats['mean_db_deviation'] = None
            
            # Mean WB deviation (K) — signed: positive = above setpoint, negative = below
            if avg_t_wb is not None and stats.get('wb_setpoint') is not None:
                stats['mean_wb_deviation'] = round(avg_t_wb - stats['wb_setpoint'], 3)
            else:
                stats['mean_wb_deviation'] = None
            
            # Mean Tsup deviation (K) — signed; uses Ts_buh when available (BUH cycles)
            if avg_t_supply is not None and stats.get('tsup_setpoint') is not None:
                stats['mean_tsup_deviation'] = round(avg_t_supply - stats['tsup_setpoint'], 3)
            else:
                stats['mean_tsup_deviation'] = None

            # Mean water temperature vs ecodesign Table 3 (variable flow only).
            # Setpoint is derived at render time from letter × climate × application so it
            # follows metadata edits without a recalculation.
            stats['tmean_is_variable_flow'] = _entry_is_variable_flow(
                entry, flow_config=entry.get(flow_key) if flow_key else None
            )
            stats['tmean_setpoint'] = _tmean_setpoint_from_entry(entry, stats.get('test_condition'))
            avg_t_mean_log = safe_float(entry.get('avg_t_mean_log'))
            if (stats['tmean_is_variable_flow'] and avg_t_mean_log is not None
                    and stats['tmean_setpoint'] is not None):
                stats['mean_tmean_deviation'] = round(avg_t_mean_log - stats['tmean_setpoint'], 3)
            else:
                stats['mean_tmean_deviation'] = None

            # Flow checks apply only to fixed-flow tests (PD-04). Hide stored % on
            # variable-flow rows so old databases do not need a recalculation.
            if stats['tmean_is_variable_flow']:
                stats['flow_setpoint'] = None
                stats['flow_set_type'] = None
                stats['flow_setpoint_display'] = None
                stats['mean_flow_dev_pct'] = None
                stats['flow_outside_count'] = None
                stats['flow_percentage'] = None
                stats['max_flow_pos_pct'] = None
                stats['max_flow_neg_pct'] = None
            
            # Qset and Mean Q deviation — signed: positive = above Qset, negative = below; uses QCorrwBUH when available
            hp_id = entry.get(hp_key) if hp_key else None
            pdesign = get_pdesign_for_hp(
                hp_id, conn, file_name=entry.get('file_name'), profile_id=entry.get('profile_id')
            )
            db_set = stats.get('db_setpoint')
            ua, qset = compute_ua_and_qset(pdesign, db_set)
            stats['qset_kw'] = round(qset, 3) if qset is not None else None
            mean_q = get_mean_q_for_deviations(entry)
            if mean_q is not None and qset is not None and qset != 0:
                stats['mean_q_dev_kw'] = round(mean_q - qset, 3)
                stats['mean_q_dev_pct'] = round(100.0 * (mean_q - qset) / qset, 2)
            else:
                stats['mean_q_dev_kw'] = None
                stats['mean_q_dev_pct'] = None
            
            # Combine entry info with deviation stats
            # Start with a copy of all entry data to preserve all columns
            combined = dict(entry)  # Make a full copy of entry first
            # Then add/update with stats (stats will override any matching keys, but that's OK)
            combined.update(stats)
            # Ensure all original entry keys are still present (in case stats had None values)
            for key, value in entry.items():
                if key not in combined or combined[key] is None:
                    combined[key] = value
            deviation_stats.append(combined)
        
        # Count entries with and without statistics
        total_entries = len(results)
        calculated_entries = sum(1 for e in deviation_stats if e.get('has_statistics'))
        
        return render_template('deviations.html', 
                             entries=deviation_stats,
                             deviation_bands=DEVIATION_BANDS,
                             setpoints=DEVIATION_SETPOINTS,
                             total_entries=total_entries,
                             calculated_entries=calculated_entries,
                             column_name_map=column_name_map,
                             deviations=deviations_config,
                             unit_design_rows=unit_design_rows,
                             units_without_pdesign=units_without_pdesign,
                             unresolved_entry_count=unresolved_entry_count,
                             flow_modes=FLOW_MODES)
    finally:
        conn.close()


@app.route('/dtreturn_insights')
def dtreturn_insights():
    """Explore point-level dTreturn distribution across many entries."""
    init_deviation_columns()
    ensure_cycle_periods_table()
    conn = get_db_connection()
    try:
        units = _list_analysis_units(conn)
        hp_ids = [u['label'] for u in units]

        # Distinct test conditions: prefer dev_test_condition, fall back to test_cond
        # (later sheets of a file often have test_cond set but dev_test_condition NULL).
        tc_rows = conn.execute(
            "SELECT DISTINCT COALESCE(NULLIF(TRIM(CAST(dev_test_condition AS TEXT)), ''), "
            "                        NULLIF(TRIM(CAST(test_cond AS TEXT)), '')) AS tc "
            "FROM results"
        ).fetchall()
        test_conditions = sorted(str(r['tc']) for r in tc_rows if r['tc'] is not None and str(r['tc']).strip() != '')
        # Include a synthetic "unknown" bucket to help users include entries lacking condition metadata
        if 'unknown' not in [t.lower() for t in test_conditions]:
            test_conditions.append('unknown')

        # Distinct COP_dataset values for filtering
        cop_rows = conn.execute(
            "SELECT DISTINCT COP_dataset AS v FROM results WHERE COP_dataset IS NOT NULL AND TRIM(CAST(COP_dataset AS TEXT)) != '' ORDER BY COP_dataset"
        ).fetchall()
        cop_dataset_values = [str(r['v']) for r in cop_rows if r['v'] is not None]

        return render_template('dtreturn_insights.html', units=units, hp_ids=hp_ids, test_conditions=test_conditions, cop_dataset_values=cop_dataset_values)
    finally:
        conn.close()


PERIOD_STAT_DISPLAY_METRICS = (
    ('duration_s', 'dur (s)'),
    ('avg_t_db', 'T_db'),
    ('avg_t_wb', 'T_wb'),
    ('avg_ts_buh', 'Ts Buh'),
    ('avg_t_mean_log', 'T_mean'),
    ('t_mean_from_avgs', 'T_mean avgs'),
    ('avg_q_corr_wbuh', 'QCorrwBUH'),
    ('avg_p_corr_wbuh', 'PCorrwBUH'),
    ('avg_cop_corr_wbuh', 'COPCorrwBUH'),
    ('avg_mass_flow', 'm_flow'),
    ('avg_volume_flow', 'v_flow'),
    ('dtreturn_min', 'dT min'),
    ('dtreturn_avg', 'dT avg'),
    ('dtreturn_max', 'dT max'),
)


def _period_stat_prefixes_in_rows(rows: list, cycle_kind: str) -> list:
    known = {'Off', 'On', 'Off+On', 'D', 'D1', 'D2', 'H', 'D+H'}
    found = set()
    for row in rows:
        for k in row.keys():
            if '_' not in k:
                continue
            pref = k.split('_', 1)[0]
            if pref in known or (pref.startswith('D') and pref[1:].isdigit()):
                found.add(pref)
    if cycle_kind == 'on_off':
        return [p for p in ('Off', 'On', 'Off+On') if p in found]
    if cycle_kind == 'defrost':
        order = ['D', 'D1', 'H', 'D2', 'D+H']
        extra = sorted(p for p in found if p not in order and (p.startswith('D') or p == 'H'))
        return [p for p in order if p in found] + extra
    # Unified view: all cycle families on one page
    order = ['Off', 'On', 'Off+On', 'D', 'D1', 'H', 'D2', 'D+H']
    extra = sorted(p for p in found if p not in order)
    return [p for p in order if p in found] + extra


def _period_stat_columns(cycle_kind: str, rows: list) -> list:
    prefixes = _period_stat_prefixes_in_rows(rows, cycle_kind)
    cols = []
    for pref in prefixes:
        for mk, mlbl in PERIOD_STAT_DISPLAY_METRICS:
            cols.append({'key': f'{pref}_{mk}', 'label': f'{pref} {mlbl}', 'group': pref})
    return cols


def _load_period_statistics_rows_for_type(cycle_type: str, buffer_s: int) -> list:
    """Load pivoted period-stat rows for one cycle_type."""
    if cycle_type == 'on_off_cycle':
        buf = 0
    else:
        buf = buffer_s if buffer_s in BUFFER_OPTIONS else SETTLING_BUFFER_S

    conn = get_db_connection()
    try:
        entries = conn.execute(
            "SELECT rowid, file_name, data_set, HP_ID, test_cond, dev_test_condition, "
            "cycle_type, cycle_start_marker, pelec_transition_time, pelec_transition_source, "
            "start_time, end_time, display_order, flow_config, COP_dataset, "
            "avg_t_db, avg_t_wb, avg_t_sup_buh, avg_t_supply, Ts_buh, "
            "avg_t_mean_log, t_mean_from_avgs, avg_dt_ln, "
            "QCorrwBUH, PCorrwBUH, COPCorrwBUH, "
            "avg_heating_capacity_corr, avg_power_input_corr, cop_corr, "
            "avg_mass_flow, avg_volume_flow, "
            "dtreturn_cache_min, dtreturn_cache_max, dtreturn_cache_avg, dtreturn_cache_n_valid "
            "FROM results WHERE cycle_type=? "
            "ORDER BY display_order ASC, rowid ASC",
            (cycle_type,)
        ).fetchall()
        entries = [dict(e) for e in entries]
        entry_idents = unit_config.disambiguate_labels(
            [unit_config.unit_key_and_label(e) for e in entries]
        )
        out = []
        for i, e in enumerate(entries):
            rid = int(e['rowid'])
            periods = conn.execute(
                "SELECT period_type, start_time, end_time, "
                "dtreturn_n_valid, dtreturn_min, dtreturn_max, dtreturn_avg, "
                "avg_t_db, avg_t_wb, avg_ts_buh, avg_q_corr_wbuh, avg_p_corr_wbuh, avg_cop_corr_wbuh, "
                "avg_volume_flow, avg_mass_flow, avg_t_mean_log, t_mean_from_avgs "
                "FROM cycle_periods WHERE entry_rowid=? AND buffer_s=? ORDER BY start_time",
                (rid, buf)
            ).fetchall()
            periods = [dict(p) for p in periods]
            pivot = _pivot_period_stats(periods, e['cycle_type'], e.get('cycle_start_marker') or '')
            ct = e.get('cycle_type') or ''
            if ct == 'defrost_cycle':
                _apply_full_cycle_slot(pivot, 'D+H', e)
            elif ct == 'on_off_cycle':
                _apply_full_cycle_slot(pivot, 'Off+On', e)
            ident = entry_idents[i]
            tc = e.get('dev_test_condition')
            if tc is None or str(tc).strip() == '':
                tc = e.get('test_cond')
            row = {
                'rowid': rid,
                'file_name': e.get('file_name'),
                'data_set': e.get('data_set'),
                'hp_id': ident['label'],
                'unit_key': ident['unit_key'],
                'test_condition': tc,
                'cycle_type': ct,
                'cycle_family': 'on-off' if ct == 'on_off_cycle' else ('defrost' if ct == 'defrost_cycle' else ct),
                'cycle_start_marker': e.get('cycle_start_marker'),
                'pelec_transition_time': e.get('pelec_transition_time'),
                'transition_source': e.get('pelec_transition_source') or 'auto',
                'flow_config': e.get('flow_config'),
                'cop_dataset': e.get('COP_dataset'),
                't_start': e.get('start_time'),
                't_end': e.get('end_time'),
                'n_periods': len(periods),
                'display_order': e.get('display_order') or rid,
            }
            row.update(pivot)
            out.append(row)
        return out
    finally:
        conn.close()


def _load_period_statistics_rows(cycle_kind: str, buffer_s: int) -> list:
    """One table row per cycle entry with pivoted sub-period mean columns."""
    ensure_cycle_periods_table()
    init_deviation_columns()
    if cycle_kind == 'all':
        rows = _load_period_statistics_rows_for_type('defrost_cycle', buffer_s)
        rows.extend(_load_period_statistics_rows_for_type('on_off_cycle', 0))
        rows.sort(key=lambda r: (r.get('display_order') or 0, r['rowid']))
        return rows
    if cycle_kind == 'on_off':
        return _load_period_statistics_rows_for_type('on_off_cycle', 0)
    return _load_period_statistics_rows_for_type('defrost_cycle', buffer_s)


@app.route('/period_statistics')
def period_statistics():
    """Flat table of sub-cycle mean operating-point statistics per entry."""
    init_deviation_columns()
    ensure_cycle_periods_table()
    cycle_kind = (request.args.get('kind') or 'all').strip().lower()
    if cycle_kind not in ('defrost', 'on_off', 'all'):
        cycle_kind = 'all'
    try:
        buffer_s = int(request.args.get('buffer_s', SETTLING_BUFFER_S))
    except Exception:
        buffer_s = SETTLING_BUFFER_S
    if buffer_s not in BUFFER_OPTIONS:
        buffer_s = SETTLING_BUFFER_S

    rows = _load_period_statistics_rows(cycle_kind, buffer_s)
    stat_columns = _period_stat_columns(cycle_kind, rows)
    stat_groups = _period_stat_prefixes_in_rows(rows, cycle_kind)

    conn = get_db_connection()
    try:
        units = _list_analysis_units(conn)
        hp_ids = [u['label'] for u in units]
        tc_rows = conn.execute(
            "SELECT DISTINCT COALESCE(NULLIF(TRIM(CAST(dev_test_condition AS TEXT)), ''), "
            "                        NULLIF(TRIM(CAST(test_cond AS TEXT)), '')) AS tc FROM results"
        ).fetchall()
        test_conditions = sorted(str(r['tc']) for r in tc_rows if r['tc'] is not None and str(r['tc']).strip() != '')
    finally:
        conn.close()

    return render_template(
        'period_statistics.html',
        cycle_kind=cycle_kind,
        buffer_s=buffer_s,
        buffer_options=BUFFER_OPTIONS,
        rows=rows,
        stat_columns=stat_columns,
        stat_groups=stat_groups,
        hp_ids=hp_ids,
        test_conditions=test_conditions,
    )


def _reservoir_add(reservoir, seen_count, value, k, rnd):
    """Reservoir sampling for streaming values. Returns updated seen_count."""
    seen_count += 1
    if len(reservoir) < k:
        reservoir.append(float(value))
        return seen_count
    j = rnd.randint(1, seen_count)
    if j <= k:
        reservoir[j - 1] = float(value)
    return seen_count


def _dtreturn_group_key(hp_id, test_condition):
    hp = str(hp_id) if hp_id is not None and str(hp_id).strip() != '' else ''
    tc = str(test_condition) if test_condition is not None and str(test_condition).strip() != '' else 'unknown'
    return (hp, tc)


_CYCLE_EXTRACT_EXCEL_CACHE_MAX = 8
_CYCLE_EXTRACT_EXCEL_CACHE = OrderedDict()  # key -> (mtime, df)
# Bump when Excel read / cycle-extract payload changes so LRU cannot serve stale narrowed frames.
_CYCLE_EXCEL_CACHE_VER = 2


def _read_excel_sheet(file_name, data_set=None, usecols=None):
    """Read a single Excel sheet (optionally limited to usecols) with a small LRU cache.

    Falls back to sheet 0 if requested sheet doesn't exist.
    """
    file_path = os.path.join(config['data_dir'], file_name)
    if not os.path.exists(file_path):
        return None
    try:
        sheet_idx = 0
        if data_set is not None:
            try:
                si = int(float(data_set)) - 1
                if si >= 0:
                    sheet_idx = si
            except (ValueError, TypeError):
                pass

        # Normalize usecols for caching (None or sorted tuple of strings)
        usecols_key = None
        if usecols is not None:
            if isinstance(usecols, (list, tuple, set)):
                usecols_key = tuple(sorted([str(c) for c in usecols]))
            else:
                usecols_key = str(usecols)

        mtime = os.path.getmtime(file_path)
        cache_key = (_CYCLE_EXCEL_CACHE_VER, file_path, int(sheet_idx), usecols_key)
        if cache_key in _CYCLE_EXTRACT_EXCEL_CACHE:
            cached_mtime, df_cached = _CYCLE_EXTRACT_EXCEL_CACHE[cache_key]
            if cached_mtime == mtime:
                _CYCLE_EXTRACT_EXCEL_CACHE.move_to_end(cache_key)
                return df_cached
            # file changed on disk -> invalidate
            _CYCLE_EXTRACT_EXCEL_CACHE.pop(cache_key, None)

        read_kwargs = {'sheet_name': sheet_idx, 'engine': 'openpyxl'}
        if usecols is not None:
            read_kwargs['usecols'] = usecols
        try:
            df = pd.read_excel(file_path, **read_kwargs)
        except Exception:
            # fallback sheet 0
            read_kwargs['sheet_name'] = 0
            df = pd.read_excel(file_path, **read_kwargs)

        _CYCLE_EXTRACT_EXCEL_CACHE[cache_key] = (mtime, df)
        if len(_CYCLE_EXTRACT_EXCEL_CACHE) > _CYCLE_EXTRACT_EXCEL_CACHE_MAX:
            _CYCLE_EXTRACT_EXCEL_CACHE.popitem(last=False)
        return df
    except Exception as e:
        print(f"[_read_excel_sheet] Cannot read {file_name}: {e}")
        return None


def _compute_dtreturn_sample_for_entry(file_name, data_set, start_time, end_time, max_sample=5000):
    """Compute dTreturn values for one entry and return (n_valid, n_nan, sample_list).
    sample_list is a reservoir-sampled list of up to max_sample float values."""
    import random
    try:
        st, et = float(start_time), float(end_time)
    except (TypeError, ValueError):
        return None
    df_full = _read_excel_sheet(file_name, data_set)
    if df_full is None or 'time_elapsed' not in df_full.columns:
        return None
    df = df_full[(df_full['time_elapsed'] >= st) & (df_full['time_elapsed'] <= et)]
    if df.empty:
        return None
    if 'T_return_emu' not in df.columns or 'T_return_calc' not in df.columns:
        return None
    dt = df['T_return_emu'] - df['T_return_calc']
    valid = dt.dropna()
    n_valid = int(len(valid))
    n_nan = int(len(dt) - n_valid)
    # Reservoir sample
    rnd = random.Random(42)
    sample = []
    for i, v in enumerate(valid.values):
        fv = float(v)
        if len(sample) < max_sample:
            sample.append(fv)
        else:
            j = rnd.randint(0, i)
            if j < max_sample:
                sample[j] = fv
    return (n_valid, n_nan, sample)


@app.route('/api/dtreturn_cache_refresh', methods=['POST'])
def api_dtreturn_cache_refresh():
    """Compute and store dTreturn sample cache for entries that don't have one yet (or all if forced)."""
    # Deprecated: prefer /api/pipeline/run which also caches multi-buffer periods
    init_deviation_columns()
    ensure_dtreturn_exclusions_table()
    try:
        payload = request.get_json() or {}
        force_all = payload.get('force_all', False)
        conn = get_db_connection()
        try:
            if force_all:
                rows = conn.execute(
                    "SELECT rowid, file_name, data_set, start_time, end_time FROM results ORDER BY file_name, data_set, rowid"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT rowid, file_name, data_set, start_time, end_time FROM results "
                    "WHERE dtreturn_cache_n_valid IS NULL "
                    "   OR dtreturn_cache_max_abs IS NULL "
                    "   OR dtreturn_cache_p99_abs IS NULL "
                    "ORDER BY file_name, data_set, rowid"
                ).fetchall()
            rows = [dict(r) for r in rows]
        finally:
            conn.close()

        total = len(rows)
        if total == 0:
            return jsonify({'success': True, 'processed': 0, 'total': 0, 'message': 'All entries already cached.'})

        import time as _time
        from collections import OrderedDict
        t0 = _time.time()
        MAX_CACHE = 40
        _excel_cache = OrderedDict()
        processed = 0
        skipped = 0

        def _cached_read(fname, dset):
            key = (fname, dset)
            if key in _excel_cache:
                _excel_cache.move_to_end(key)
                return _excel_cache[key]
            df = _read_excel_sheet(fname, dset)
            _excel_cache[key] = df
            if len(_excel_cache) > MAX_CACHE:
                _excel_cache.popitem(last=False)
            return df

        conn = get_db_connection()
        try:
            for idx, r in enumerate(rows):
                fn = r.get('file_name')
                ds = r.get('data_set')
                st = r.get('start_time')
                et = r.get('end_time')
                rid = r['rowid']
                if not fn or st is None or et is None:
                    skipped += 1
                    continue

                df_full = _cached_read(fn, ds)

                if df_full is None or 'time_elapsed' not in df_full.columns:
                    skipped += 1
                    conn.execute("UPDATE results SET dtreturn_cache_n_valid=0, dtreturn_cache_n_nan=0, dtreturn_cache_sample='[]' WHERE rowid=?", (rid,))
                    continue

                try:
                    stf, etf = float(st), float(et)
                except (TypeError, ValueError):
                    skipped += 1
                    continue

                df = df_full[(df_full['time_elapsed'] >= stf) & (df_full['time_elapsed'] <= etf)]
                if df.empty or 'T_return_emu' not in df.columns or 'T_return_calc' not in df.columns:
                    conn.execute("UPDATE results SET dtreturn_cache_n_valid=0, dtreturn_cache_n_nan=0, dtreturn_cache_sample='[]' WHERE rowid=?", (rid,))
                    skipped += 1
                    continue

                dt = df['T_return_emu'] - df['T_return_calc']
                valid = dt.dropna()
                n_valid = int(len(valid))
                n_nan = int(len(dt) - n_valid)

                import random
                rnd = random.Random(42)
                max_sample = 5000
                sample = []
                for i, v in enumerate(valid.values):
                    fv = float(v)
                    if len(sample) < max_sample:
                        sample.append(round(fv, 6))
                    else:
                        j = rnd.randint(0, i)
                        if j < max_sample:
                            sample[j] = round(fv, 6)

                # Per-entry extremeness metrics (for drill-down/outlier hunting)
                max_abs = None
                max_signed = None
                p99_abs = None
                if n_valid > 0:
                    try:
                        import numpy as np
                        arr = valid.to_numpy(dtype=float)
                        abs_arr = np.abs(arr)
                        max_idx = int(abs_arr.argmax())
                        max_signed = float(arr[max_idx])
                        max_abs = float(abs_arr[max_idx])
                        p99_abs = float(np.percentile(abs_arr, 99.0))
                    except Exception:
                        max_abs = None
                        max_signed = None
                        p99_abs = None

                conn.execute(
                    "UPDATE results SET dtreturn_cache_n_valid=?, dtreturn_cache_n_nan=?, dtreturn_cache_sample=?, "
                    "dtreturn_cache_max_abs=?, dtreturn_cache_max_signed=?, dtreturn_cache_p99_abs=? "
                    "WHERE rowid=?",
                    (n_valid, n_nan, json.dumps(sample), max_abs, max_signed, p99_abs, rid)
                )
                processed += 1

                if (idx + 1) % 20 == 0 or (idx + 1) == total:
                    conn.commit()
                    elapsed = _time.time() - t0
                    print(f"[dtreturn cache] {idx+1}/{total} ({elapsed:.1f}s, {len(_excel_cache)} files cached)", flush=True)

            conn.commit()
        finally:
            conn.close()

        elapsed = _time.time() - t0
        return jsonify({
            'success': True,
            'processed': processed,
            'skipped': skipped,
            'total': total,
            'elapsed_s': round(elapsed, 1),
            'deprecated': True,
            'use_instead': '/api/pipeline/run',
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/dtreturn_insights', methods=['POST'])
def api_dtreturn_insights():
    """Aggregate precomputed dTreturn caches — instant, no Excel I/O.

    When period_filter is 'all' (default), uses per-entry caches in results.
    When period_filter is 'defrost' or 'heating', uses per-period caches in cycle_periods.
    """
    init_deviation_columns()
    ensure_dtreturn_exclusions_table()
    ensure_cycle_periods_table()
    try:
        import numpy as np
        import random
        from io import BytesIO
        import base64
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        payload = request.get_json() or {}
        unit_keys_filter = payload.get('unit_keys') or payload.get('hp_ids') or []
        tc_filter = payload.get('test_conditions') or []
        period_filter = (payload.get('period_filter') or 'all').strip().lower()
        try:
            buffer_s = int(payload.get('buffer_s', SETTLING_BUFFER_S))
        except Exception:
            buffer_s = SETTLING_BUFFER_S
        if buffer_s != 0 and buffer_s not in BUFFER_OPTIONS:
            buffer_s = SETTLING_BUFFER_S
        try:
            coverage = float(payload.get('coverage', 95.0))
        except Exception:
            coverage = 95.0
        coverage = max(50.0, min(99.9, coverage))
        alpha = (100.0 - coverage) / 2.0

        unit_keys_filter = [str(x) for x in unit_keys_filter if x is not None and str(x).strip() != '']
        tc_filter_norm = [str(x).lower() for x in tc_filter if x is not None and str(x).strip() != '']
        cop_dataset_filter = payload.get('cop_dataset') or []
        cop_dataset_filter = [str(x).upper() for x in cop_dataset_filter if x is not None and str(x).strip() != '']

        use_period_cache = period_filter in ('defrost', 'heating', 'off', 'on')

        if period_filter in ('off', 'on'):
            buffer_s = 0

        conn = get_db_connection()
        try:
            excluded_rows = conn.execute("SELECT entry_rowid FROM dtreturn_insights_exclusions").fetchall()
            excluded_set = {int(r[0]) for r in excluded_rows if r and r[0] is not None}

            if use_period_cache:
                rows = conn.execute(
                    "SELECT r.rowid, r.file_name, r.profile_id, r.HP_ID AS hp_id, r.dev_test_condition, r.test_cond, r.COP_dataset, "
                    "cp.dtreturn_n_valid, cp.dtreturn_n_nan, cp.dtreturn_sample, "
                    "cp.dtreturn_max_abs, cp.period_type, cp.buffer_s "
                    "FROM cycle_periods cp "
                    "JOIN results r ON cp.entry_rowid = r.rowid "
                    "WHERE cp.period_type = ? AND cp.buffer_s = ? "
                    "ORDER BY r.display_order ASC, r.rowid ASC, cp.id ASC",
                    (period_filter, buffer_s)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT rowid, file_name, profile_id, HP_ID AS hp_id, dev_test_condition, test_cond, COP_dataset, "
                    "dtreturn_cache_n_valid, dtreturn_cache_n_nan, dtreturn_cache_sample, "
                    "dtreturn_cache_max_abs "
                    "FROM results ORDER BY display_order ASC, rowid ASC"
                ).fetchall()
            rows = [dict(r) for r in rows]

            # Period status counts
            total_classified = conn.execute(
                "SELECT COUNT(*) FROM results WHERE cycle_type IS NOT NULL AND cycle_type != 'other'"
            ).fetchone()[0]
            total_periods = conn.execute(
                "SELECT COUNT(*) FROM cycle_periods"
            ).fetchone()[0]
            uncached_periods = conn.execute(
                "SELECT COUNT(*) FROM cycle_periods WHERE dtreturn_n_valid IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()

        rnd = random.Random(1337)
        max_pool = 200000

        row_idents = unit_config.disambiguate_labels(
            [unit_config.unit_key_and_label(r) for r in rows]
        )

        overall_sample = []
        overall_valid = 0
        overall_nan = 0
        overall_entries = 0
        uncached = 0
        needs_metric_refresh = 0
        excluded = 0
        groups = {}

        for i, r in enumerate(rows):
            rid = int(r.get('rowid')) if r.get('rowid') is not None else None
            if rid is not None and rid in excluded_set:
                excluded += 1
                continue
            ident = row_idents[i]
            unit_key = ident['unit_key']
            unit_label = ident['label']
            tc = r.get('dev_test_condition')
            if tc is None or str(tc).strip() == '':
                tc = r.get('test_cond')
            tc_norm = str(tc).strip().lower() if tc is not None and str(tc).strip() != '' else 'unknown'

            if unit_keys_filter and unit_key not in unit_keys_filter:
                continue
            if tc_filter_norm and tc_norm not in tc_filter_norm:
                continue
            if cop_dataset_filter:
                entry_cop = str(r.get('COP_dataset') or '').upper().strip()
                if entry_cop not in cop_dataset_filter:
                    continue

            nv = r.get('dtreturn_n_valid') if use_period_cache else r.get('dtreturn_cache_n_valid')
            if nv is None:
                uncached += 1
                continue
            if not use_period_cache and r.get('dtreturn_cache_max_abs') is None and nv > 0:
                needs_metric_refresh += 1

            n_nan_key = 'dtreturn_n_nan' if use_period_cache else 'dtreturn_cache_n_nan'
            sample_key = 'dtreturn_sample' if use_period_cache else 'dtreturn_cache_sample'
            n_nan = r.get(n_nan_key) or 0
            sample_json = r.get(sample_key) or '[]'
            try:
                sample = json.loads(sample_json)
            except Exception:
                sample = []

            if nv == 0:
                continue

            overall_entries += 1
            overall_valid += nv
            overall_nan += n_nan

            for v in sample:
                if len(overall_sample) < max_pool:
                    overall_sample.append(v)
                else:
                    j = rnd.randint(0, overall_valid)
                    if j < max_pool:
                        overall_sample[j] = v

            key = _dtreturn_group_key(unit_key, tc_norm)
            if key not in groups:
                groups[key] = {
                    'hp_id': key[0],
                    'unit_label': unit_label,
                    'test_condition': key[1],
                    'entries': 0,
                    'valid': 0,
                    'nan': 0,
                    'sample': [],
                }
            g = groups[key]
            g['entries'] += 1
            g['valid'] += nv
            g['nan'] += n_nan
            for v in sample:
                if len(g['sample']) < max_pool:
                    g['sample'].append(v)
                else:
                    j = rnd.randint(0, g['valid'])
                    if j < max_pool:
                        g['sample'][j] = v

        def summarize_sample(sample_list, entries, valid_pts, nan_pts):
            arr = np.array(sample_list, dtype=float) if sample_list else np.array([], dtype=float)
            if arr.size == 0:
                return {'entries_processed': entries, 'valid_points': valid_pts, 'nan_points': nan_pts,
                        'median': None, 'p95_abs': None, 'recommended_lower': None, 'recommended_upper': None}
            qs = np.percentile(arr, [alpha, 50.0, 100.0 - alpha])
            return {
                'entries_processed': int(entries), 'valid_points': int(valid_pts), 'nan_points': int(nan_pts),
                'median': float(qs[1]), 'p95_abs': float(np.percentile(np.abs(arr), 95.0)),
                'recommended_lower': float(qs[0]), 'recommended_upper': float(qs[2]),
            }

        overall_summary = summarize_sample(overall_sample, overall_entries, overall_valid, overall_nan)

        group_summaries = []
        for key, g in groups.items():
            s = summarize_sample(g['sample'], g['entries'], g['valid'], g['nan'])
            s['hp_id'] = g['hp_id']
            s['unit_label'] = g.get('unit_label') or g['hp_id']
            s['test_condition'] = g['test_condition']
            group_summaries.append(s)
        group_summaries.sort(key=lambda x: (str(x.get('hp_id') or ''), str(x.get('test_condition') or '')))

        period_label = {'defrost': 'Defrost (D)', 'heating': 'Heating (H)'}.get(period_filter, 'All (full cycle)')

        images = {}
        if overall_sample:
            sample_arr = np.array(overall_sample, dtype=float)

            plt.figure(figsize=(10, 4.5))
            plt.hist(sample_arr, bins=80, color='#6f42c1', alpha=0.75, edgecolor='white')
            if overall_summary['recommended_lower'] is not None:
                plt.axvline(overall_summary['recommended_lower'], color='black', linestyle='--', linewidth=1,
                            label=f"lower ({overall_summary['recommended_lower']:.2f})")
                plt.axvline(overall_summary['recommended_upper'], color='black', linestyle='--', linewidth=1,
                            label=f"upper ({overall_summary['recommended_upper']:.2f})")
            plt.title(f"dTreturn distribution [{period_label}] ({overall_valid:,} pts, {overall_entries} entries)")
            plt.xlabel("dTreturn (K)")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.25)
            plt.legend(loc='best', fontsize=9)
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=120)
            plt.close()
            buf.seek(0)
            images['histogram'] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

            xs = np.sort(sample_arr)
            ys = np.arange(1, xs.size + 1) / xs.size
            plt.figure(figsize=(10, 4.5))
            plt.plot(xs, ys, color='#343a40', linewidth=1.5)
            if overall_summary['recommended_lower'] is not None:
                plt.axvline(overall_summary['recommended_lower'], color='#6f42c1', linestyle='--', linewidth=1)
                plt.axvline(overall_summary['recommended_upper'], color='#6f42c1', linestyle='--', linewidth=1)
            plt.title(f"ECDF of dTreturn [{period_label}] (sample)")
            plt.xlabel("dTreturn (K)")
            plt.ylabel(u"P(dTreturn \u2264 x)")
            plt.grid(True, alpha=0.25)
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=120)
            plt.close()
            buf.seek(0)
            images['ecdf'] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'overall': overall_summary,
            'groups': group_summaries,
            'images': images,
            'meta': {
                'coverage': coverage,
                'uncached_entries': uncached,
                'needs_metric_refresh': needs_metric_refresh,
                'excluded_entries': excluded,
                'period_filter': period_filter,
                'buffer_s': buffer_s,
                'total_classified': total_classified,
                'total_periods': total_periods,
                'uncached_periods': uncached_periods,
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/dtreturn_group_entries', methods=['POST'])
def api_dtreturn_group_entries():
    """List entries contributing to a (hp_id, test_condition) group, ranked by extremeness."""
    init_deviation_columns()
    ensure_dtreturn_exclusions_table()
    try:
        payload = request.get_json() or {}
        unit_key_want = payload.get('unit_key') or payload.get('hp_id', None)
        test_condition = payload.get('test_condition', None)
        period_filter = (payload.get('period_filter') or 'all').strip().lower()
        try:
            buffer_s = int(payload.get('buffer_s', SETTLING_BUFFER_S))
        except Exception:
            buffer_s = SETTLING_BUFFER_S
        if buffer_s != 0 and buffer_s not in BUFFER_OPTIONS:
            buffer_s = SETTLING_BUFFER_S
        cop_vals = payload.get('cop_dataset') or []
        if isinstance(cop_vals, str):
            cop_vals = [cop_vals]
        cop_vals = [str(v).strip() for v in cop_vals if str(v).strip() != '']
        try:
            limit = int(payload.get('limit', 200))
        except Exception:
            limit = 200
        limit = max(10, min(2000, limit))

        unit_key_s = None if unit_key_want is None or str(unit_key_want).strip() == '' else str(unit_key_want).strip()
        tc_norm = str(test_condition).strip().lower() if test_condition is not None and str(test_condition).strip() != '' else 'unknown'

        conn = get_db_connection()
        try:
            excluded_rows = conn.execute("SELECT entry_rowid, reason FROM dtreturn_insights_exclusions").fetchall()
            excluded_reason = {int(r[0]): (r[1] or '') for r in excluded_rows if r and r[0] is not None}

            params = []
            if period_filter in ('defrost', 'heating', 'off', 'on'):
                # Period-mode: rank by per-period extremeness and include only entries that contribute to stats
                q = (
                    "SELECT r.rowid, r.file_name, r.data_set, r.start_time, r.end_time, r.profile_id, r.HP_ID AS hp_id, r.dev_test_condition, r.test_cond, "
                    "MAX(cp.dtreturn_n_valid) AS dtreturn_cache_n_valid, "
                    "MAX(cp.dtreturn_max_abs) AS dtreturn_cache_max_abs, "
                    "MAX(cp.dtreturn_max_signed) AS dtreturn_cache_max_signed, "
                    "MIN(cp.dtreturn_min) AS dtreturn_cache_min, "
                    "MAX(cp.dtreturn_max) AS dtreturn_cache_max, "
                    "MAX(cp.dtreturn_p99_abs) AS dtreturn_cache_p99_abs, "
                    "r.cycle_type, r.cycle_start_marker, r.pelec_transition_time "
                    "FROM cycle_periods cp "
                    "JOIN results r ON cp.entry_rowid = r.rowid "
                    "WHERE cp.period_type=? AND cp.buffer_s=? AND cp.dtreturn_n_valid IS NOT NULL "
                )
                params.append(period_filter)
                params.append(0 if period_filter in ('off', 'on') else buffer_s)
            else:
                # Full-cycle mode (results table cache)
                q = (
                    "SELECT rowid, file_name, data_set, start_time, end_time, profile_id, HP_ID AS hp_id, dev_test_condition, test_cond, "
                    "dtreturn_cache_n_valid, dtreturn_cache_max_abs, dtreturn_cache_max_signed, "
                    "dtreturn_cache_min, dtreturn_cache_max, dtreturn_cache_p99_abs, "
                    "cycle_type, cycle_start_marker, pelec_transition_time "
                    "FROM results "
                    "WHERE dtreturn_cache_n_valid IS NOT NULL "
                )

            pfx = "r." if period_filter in ('defrost', 'heating', 'off', 'on') else ""

            # Coalesce dev_test_condition -> test_cond, then normalize to 'unknown'.
            tc_expr = (
                f"COALESCE(NULLIF(TRIM(CAST({pfx}dev_test_condition AS TEXT)), ''), "
                f"NULLIF(TRIM(CAST({pfx}test_cond AS TEXT)), ''))"
            )
            q += f" AND (CASE WHEN {tc_expr} IS NULL THEN 'unknown' ELSE lower({tc_expr}) END)=? "
            params.append(tc_norm)

            if cop_vals:
                # Filter by COP_dataset values (e.g. 'yes')
                placeholders = ",".join(["?"] * len(cop_vals))
                q += f" AND lower(TRIM(CAST({('r.COP_dataset' if period_filter in ('defrost', 'heating', 'off', 'on') else 'COP_dataset')} AS TEXT))) IN ({placeholders}) "
                params.extend([v.lower() for v in cop_vals])

            if period_filter in ('defrost', 'heating', 'off', 'on'):
                q += " GROUP BY r.rowid "

            rowid_order = "r.rowid" if period_filter in ('defrost', 'heating', 'off', 'on') else "rowid"
            q += f" ORDER BY COALESCE(dtreturn_cache_max_abs, -1) DESC, COALESCE(dtreturn_cache_p99_abs, -1) DESC, {rowid_order} ASC"

            rows = conn.execute(q, params).fetchall()
            rows = [dict(r) for r in rows]
            row_idents = unit_config.disambiguate_labels(
                [unit_config.unit_key_and_label(r) for r in rows]
            )
            out = []
            for i, r in enumerate(rows):
                ident = row_idents[i]
                if unit_key_s is not None and ident['unit_key'] != unit_key_s:
                    continue
                rid = int(r['rowid'])
                out.append({
                    'rowid': rid,
                    'file_name': r['file_name'],
                    'data_set': r['data_set'],
                    'start_time': r['start_time'],
                    'end_time': r['end_time'],
                    'hp_id': ident['label'],
                    'unit_key': ident['unit_key'],
                    'test_condition': (r['dev_test_condition'] if (r['dev_test_condition'] is not None and str(r['dev_test_condition']).strip() != '') else r['test_cond']),
                    'n_valid': r['dtreturn_cache_n_valid'],
                    'max_abs': r['dtreturn_cache_max_abs'],
                    'max_signed': r['dtreturn_cache_max_signed'],
                    'dt_min': r['dtreturn_cache_min'],
                    'dt_max': r['dtreturn_cache_max'],
                    'p99_abs': r['dtreturn_cache_p99_abs'],
                    'cycle_type': r['cycle_type'],
                    'cycle_start_marker': r['cycle_start_marker'],
                    'pelec_transition_time': r['pelec_transition_time'],
                    'excluded': rid in excluded_reason,
                    'exclude_reason': excluded_reason.get(rid, ''),
                    'focus_url': url_for('index', focus_rowid=rid),
                })
                if len(out) >= limit:
                    break
        finally:
            conn.close()

        return jsonify({'success': True, 'entries': out})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/dtreturn_exclusion_set', methods=['POST'])
def api_dtreturn_exclusion_set():
    """Exclude/include an entry from dTreturn insights."""
    ensure_dtreturn_exclusions_table()
    try:
        payload = request.get_json() or {}
        rid = payload.get('rowid')
        exclude = bool(payload.get('exclude', True))
        reason = (payload.get('reason') or '').strip()
        rid_i = int(rid)
        conn = get_db_connection()
        try:
            if exclude:
                conn.execute(
                    "INSERT INTO dtreturn_insights_exclusions(entry_rowid, reason) VALUES(?, ?) "
                    "ON CONFLICT(entry_rowid) DO UPDATE SET reason=excluded.reason",
                    (rid_i, reason)
                )
            else:
                conn.execute("DELETE FROM dtreturn_insights_exclusions WHERE entry_rowid=?", (rid_i,))
            conn.commit()
        finally:
            conn.close()
        return jsonify({'success': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/cycle_periods/update_transition', methods=['POST'])
def api_update_pelec_transition():
    """Manually update pelec_transition_time for an entry and recompute all caches for all buffers."""
    ensure_cycle_periods_table()
    init_deviation_columns()
    try:
        payload = request.get_json() or {}
        rid = int(payload['rowid'])
        new_time = payload.get('pelec_transition_time')
        if new_time is not None:
            new_time = float(new_time)

        conn = get_db_connection()
        try:
            # fetch entry context
            entry = conn.execute(
                "SELECT file_name, data_set, start_time, end_time, cycle_start_marker "
                "FROM results WHERE rowid=?",
                (rid,)
            ).fetchone()
            if not entry:
                return jsonify({'success': False, 'message': 'Entry not found'})

            fn = entry['file_name']
            ds = entry['data_set']
            stf, etf = float(entry['start_time']), float(entry['end_time'])
            start_marker = entry['cycle_start_marker']

            # update transition time (mark as manual so Rebuild all preserves it)
            conn.execute(
                "UPDATE results SET pelec_transition_time=?, pelec_transition_source='manual', "
                "pelec_detect_failed=0 WHERE rowid=?",
                (new_time, rid)
            )

            # Load Excel once and refresh full-cycle cache
            df_full = _read_excel_sheet(fn, ds)
            n_valid, n_nan, sample, max_abs, max_signed, p99_abs, dt_min, dt_max, dt_avg = _dtreturn_stats_from_df(df_full, stf, etf)
            conn.execute(
                "UPDATE results SET dtreturn_cache_n_valid=?, dtreturn_cache_n_nan=?, dtreturn_cache_sample=?, "
                "dtreturn_cache_max_abs=?, dtreturn_cache_max_signed=?, dtreturn_cache_min=?, dtreturn_cache_max=?, dtreturn_cache_avg=?, dtreturn_cache_p99_abs=? "
                "WHERE rowid=?",
                (n_valid, n_nan, json.dumps(sample), max_abs, max_signed, dt_min, dt_max, dt_avg, p99_abs, rid)
            )
            _write_results_t_mean(conn, rid, df_full, stf, etf)

            # Remove old periods (auto + manual) and rebuild for all buffer options
            conn.execute(
                "DELETE FROM cycle_periods WHERE entry_rowid=? AND detection_method IN ('auto_pelec', 'manual')",
                (rid,)
            )

            if new_time is not None and start_marker in ('defrost_start', 'defrost_end'):
                for buf_s in BUFFER_OPTIONS:
                    periods = _generate_periods_for_entry(
                        rid, start_marker, stf, etf,
                        float(new_time),
                        buffer_s=int(buf_s)
                    )
                    for p in periods:
                        _insert_cycle_period_row(conn, rid, p, df_full, 'manual', int(buf_s))
            elif new_time is not None and start_marker in ('off_start', 'on_start'):
                periods = _generate_periods_for_entry(
                    rid, start_marker, stf, etf,
                    float(new_time),
                    buffer_s=0
                )
                for p in periods:
                    _insert_cycle_period_row(conn, rid, p, df_full, 'manual', 0)
            conn.commit()
        finally:
            conn.close()
        return jsonify({'success': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_periods/entry_info', methods=['POST'])
def api_cycle_periods_entry_info():
    """Get cycle classification and period info for a specific entry."""
    ensure_cycle_periods_table()
    try:
        payload = request.get_json() or {}
        rid = int(payload['rowid'])
        try:
            buffer_s = int(payload.get('buffer_s', SETTLING_BUFFER_S))
        except Exception:
            buffer_s = SETTLING_BUFFER_S
        conn = get_db_connection()
        try:
            entry = conn.execute(
                "SELECT cycle_type, cycle_start_marker, pelec_transition_time, pelec_transition_source, "
                "start_time, end_time, Notes, test_cond "
                "FROM results WHERE rowid=?",
                (rid,)
            ).fetchone()
            if not entry:
                return jsonify({'success': False, 'message': 'Entry not found'})

            buffer_s = _effective_period_buffer_s(entry['cycle_type'], buffer_s)

            periods = conn.execute(
                "SELECT id, period_type, start_time, end_time, detection_method, buffer_s, "
                "dtreturn_n_valid, dtreturn_max_abs, dtreturn_max_signed, "
                "dtreturn_min, dtreturn_max, dtreturn_p99_abs "
                "FROM cycle_periods WHERE entry_rowid=? AND buffer_s=? ORDER BY start_time",
                (rid, buffer_s)
            ).fetchall()

            return jsonify({
                'success': True,
                'classification': {
                    'cycle_type': entry['cycle_type'],
                    'cycle_start_marker': entry['cycle_start_marker'],
                    'pelec_transition_time': entry['pelec_transition_time'],
                    'pelec_transition_source': entry['pelec_transition_source'],
                    'start_time': entry['start_time'],
                    'end_time': entry['end_time'],
                    'notes': entry['Notes'],
                    'test_cond': entry['test_cond'],
                },
                'periods': [dict(p) for p in periods],
                'settling_buffer_s': SETTLING_BUFFER_S,
                'buffer_s': buffer_s,
            })
        finally:
            conn.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/cycle_periods/pelec_plot', methods=['POST'])
def api_cycle_periods_pelec_plot():
    """Generate a Pelec time series plot with period boundaries for visual verification."""
    ensure_cycle_periods_table()
    try:
        payload = request.get_json() or {}
        rid = int(payload['rowid'])
        try:
            buffer_s = int(payload.get('buffer_s', SETTLING_BUFFER_S))
        except Exception:
            buffer_s = SETTLING_BUFFER_S
        conn = get_db_connection()
        try:
            entry = conn.execute(
                "SELECT file_name, data_set, start_time, end_time, cycle_type, "
                "pelec_transition_time, cycle_start_marker "
                "FROM results WHERE rowid=?", (rid,)
            ).fetchone()
            if not entry:
                return jsonify({'success': False, 'message': 'Entry not found'})

            buffer_s = _effective_period_buffer_s(entry['cycle_type'], buffer_s)

            periods = conn.execute(
                "SELECT period_type, start_time, end_time, buffer_s "
                "FROM cycle_periods WHERE entry_rowid=? AND buffer_s=? ORDER BY start_time",
                (rid, buffer_s)
            ).fetchall()
        finally:
            conn.close()

        fn = entry['file_name']
        ds = entry['data_set']
        st = float(entry['start_time'])
        et = float(entry['end_time'])
        trans = entry['pelec_transition_time']

        df_full = _read_excel_sheet(fn, ds)
        if df_full is None or 'time_elapsed' not in df_full.columns:
            return jsonify({'success': False, 'message': f'Cannot read {fn}'})
        if PELEC_COL not in df_full.columns:
            return jsonify({'success': False, 'message': f'Column {PELEC_COL} not found'})

        df = df_full[(df_full['time_elapsed'] >= st) & (df_full['time_elapsed'] <= et)].copy()
        if df.empty:
            return jsonify({'success': False, 'message': 'No data in time range'})

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True,
            gridspec_kw={'height_ratios': [1.0, 1.0]}
        )

        period_colors = {'defrost': '#ffcccc', 'heating': '#ccffcc', 'off': '#cce5ff', 'on': '#fff3cc',
                         'equilibrium': '#e2d9f3', 'evaluation': '#d3f0ee'}
        for p in periods:
            color = period_colors.get(p['period_type'], '#f0f0f0')
            ax1.axvspan(p['start_time'], p['end_time'], alpha=0.35, color=color, label=p['period_type'])
            ax2.axvspan(p['start_time'], p['end_time'], alpha=0.35, color=color, label=p['period_type'])

        # Top: Pelec
        ax1.plot(df['time_elapsed'], df[PELEC_COL], color='#343a40', linewidth=0.8, label='Pelec')
        ax1.set_ylabel('Electric Power (kW)')
        ax1.grid(True, alpha=0.2)
        ax1.set_title(f'rowid {rid}: {fn}  ds={ds}  [{st:.0f} – {et:.0f}]  buffer={buffer_s}s')

        # Bottom: dTreturn = T_return_emu - T_return_calc
        t_emu_col = 'T_return_emu'
        t_calc_col = 'T_return_calc'
        if (t_emu_col in df.columns) and (t_calc_col in df.columns):
            dt = df[t_emu_col] - df[t_calc_col]
            ax2.plot(df['time_elapsed'], dt, color='#0b7285', linewidth=0.9, label='dTreturn (emu - calc)')
            ax2.axhline(0.0, color='#666', linewidth=0.8, alpha=0.6)
            ax2.set_ylabel('dTreturn (K)')
        else:
            missing = [c for c in (t_emu_col, t_calc_col) if c not in df.columns]
            ax2.text(
                0.5, 0.5,
                f"Cannot compute dTreturn (missing column(s): {', '.join(missing)})",
                transform=ax2.transAxes,
                ha='center', va='center', fontsize=10
            )
            ax2.set_ylabel('dTreturn (K)')
        ax2.grid(True, alpha=0.2)
        ax2.set_xlabel('time_elapsed (s)')

        # Draw transition and true D/H boundary lines derived from periods
        if trans is not None:
            trans_t = float(trans)
            for ax in (ax1, ax2):
                ax.axvline(trans_t, color='red', linestyle='--', linewidth=1.2,
                           label=f'Pelec transition ({trans_t:.0f}s)')

        # Find the first defrost->heating boundary (this is where the buffer ends)
        dh_boundary = None
        for i in range(len(periods) - 1):
            if periods[i]['period_type'] == 'defrost' and periods[i + 1]['period_type'] == 'heating':
                dh_boundary = float(periods[i]['end_time'])
                # sqlite3.Row doesn't support .get(); access via [] safely
                buf = None
                try:
                    buf = periods[i]['buffer_s']
                except Exception:
                    buf = None
                if buf is None:
                    try:
                        buf = periods[i + 1]['buffer_s']
                    except Exception:
                        buf = None
                break
        if dh_boundary is not None:
            for ax in (ax1, ax2):
                ax.axvline(dh_boundary, color='blue', linestyle=':', linewidth=1.2,
                           label=f'buffer end ({dh_boundary:.0f}s)')

        # Legend (dedup) on top axis only
        handles, labels = ax1.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
        if unique:
            ax1.legend(*zip(*unique), loc='best', fontsize=8)

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=130)
        plt.close(fig)
        buf.seek(0)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({'success': True, 'image': img_b64})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/calculate_deviations', methods=['POST'])
def calculate_deviations():
    """Calculate deviation statistics for selected entries or entries missing statistics"""
    init_deviation_columns()
    
    data = request.get_json()
    entry_ids = data.get('entry_ids', None)  # If provided, calculate only for these
    skip_existing = data.get('skip_existing', True)  # Skip entries that already have statistics
    
    conn = get_db_connection()
    try:
        if entry_ids:
            # Calculate for specific selected entries
            placeholders = ','.join(['?'] * len(entry_ids))
            query = f'SELECT *, rowid FROM results WHERE rowid IN ({placeholders}) ORDER BY display_order ASC'
            results = conn.execute(query, entry_ids).fetchall()
        else:
            # Calculate for all entries (or only those missing statistics)
            if skip_existing:
                # Only get entries that don't have statistics yet OR are missing
                # checks that apply to every unit (Tsup / dTreturn). Do not require
                # dry-bulb max: water-to-water rows leave that NULL on purpose.
                results = conn.execute('''
                    SELECT *, rowid FROM results 
                    WHERE dev_test_condition IS NULL
                       OR dev_total_points IS NULL
                       OR dev_last_calculated IS NULL
                       OR dev_dtreturn_outside_count IS NULL
                       OR dev_dtreturn_percentage IS NULL
                       OR dev_tsup_percentage IS NULL
                    ORDER BY display_order ASC, rowid ASC
                ''').fetchall()
            else:
                # Calculate for all entries (will recalculate existing ones)
                results = conn.execute(
                    'SELECT *, rowid FROM results ORDER BY display_order ASC, rowid ASC'
                ).fetchall()
        
        results = [dict(row) for row in results]
        
        # Detect HP column for flow set lookup
        hp_key = None
        if results:
            for key in results[0].keys():
                if key and 'hp' in key.lower() and 'id' in key.lower():
                    hp_key = key
                    break
        
        calculated = 0
        skipped = 0
        errors = 0
        warnings = []  # Collect warnings for failed entries
        from datetime import datetime
        
        for entry in results:
            file_name = entry.get('file_name')
            data_set = entry.get('data_set')
            start_time = entry.get('start_time')
            end_time = entry.get('end_time')
            rowid = entry.get('rowid')
            hp_id = entry.get(hp_key) if hp_key else None
            
            if not all([file_name, data_set is not None, start_time is not None, end_time is not None]):
                continue
            
            # Skip if entry already has statistics and we're skipping existing
            has_complete_stats = (
                entry.get('dev_test_condition') is not None
                and entry.get('dev_total_points') is not None
                and entry.get('dev_last_calculated') is not None
                and entry.get('dev_dtreturn_outside_count') is not None
                and entry.get('dev_dtreturn_percentage') is not None
                and entry.get('dev_tsup_percentage') is not None
            )
            if skip_existing and has_complete_stats and not entry_ids:
                skipped += 1
                continue
            
            stats = calculate_entry_deviations(file_name, data_set, start_time, end_time, warning_list=warnings, hp_id=hp_id, entry=entry)
            if stats:
                # Store in database - ensure all fields are properly handled
                profile, _preason = unit_config.resolve_profile(
                    file_name=file_name, hp_id=hp_id, profile_id=entry.get('profile_id')
                )
                resolved_profile_id = (profile or {}).get('profile_id') or entry.get('profile_id')
                resolved_condition_set_id = (
                    entry.get('condition_set_id') or unit_config.get_default_condition_set_id()
                )
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        UPDATE results SET
                            dev_test_condition = ?,
                            dev_db_setpoint = ?,
                            dev_wb_setpoint = ?,
                            dev_tsup_setpoint = ?,
                            dev_total_points = ?,
                            dev_db_outside_count = ?,
                            dev_wb_outside_count = ?,
                            dev_dtreturn_outside_count = ?,
                            dev_db_percentage = ?,
                            dev_wb_percentage = ?,
                            dev_dtreturn_percentage = ?,
                            dev_max_db_deviation = ?,
                            dev_max_wb_deviation = ?,
                            dev_max_dtreturn_deviation = ?,
                            dev_tsup_outside_count = ?,
                            dev_tsup_percentage = ?,
                            dev_max_tsup_deviation = ?,
                            dev_max_db_pos = ?,
                            dev_max_db_neg = ?,
                            dev_max_wb_pos = ?,
                            dev_max_wb_neg = ?,
                            dev_max_tsup_pos = ?,
                            dev_max_tsup_neg = ?,
                            dev_max_dtreturn_pos = ?,
                            dev_max_dtreturn_neg = ?,
                            dev_max_flow_pos_pct = ?,
                            dev_max_flow_neg_pct = ?,
                            dev_avg_timestep = ?,
                            dev_wb_calculated_from_rh = ?,
                            dev_flow_setpoint = ?,
                            dev_flow_set_type = ?,
                            dev_mean_flow_dev_pct = ?,
                            dev_flow_outside_count = ?,
                            dev_flow_percentage = ?,
                            profile_id = COALESCE(?, profile_id),
                            condition_set_id = COALESCE(?, condition_set_id),
                            dev_last_calculated = ?
                        WHERE rowid = ?
                    ''', (
                        stats['test_condition'],
                        float(stats['db_setpoint']) if stats['db_setpoint'] is not None else None,
                        float(stats['wb_setpoint']) if stats['wb_setpoint'] is not None else None,
                        float(stats['tsup_setpoint']) if stats['tsup_setpoint'] is not None else None,
                        stats['total_points'],
                        stats['db_outside_band_count'],
                        stats['wb_outside_band_count'],  # Can be None
                        stats.get('dtreturn_outside_band_count'),
                        float(stats['db_percentage']) if stats['db_percentage'] is not None else None,
                        float(stats['wb_percentage']) if stats['wb_percentage'] is not None else None,
                        float(stats.get('dtreturn_percentage')) if stats.get('dtreturn_percentage') is not None else None,
                        float(stats['max_db_deviation']) if stats['max_db_deviation'] is not None else None,
                        float(stats['max_wb_deviation']) if stats['max_wb_deviation'] is not None else None,
                        float(stats.get('max_dtreturn_deviation')) if stats.get('max_dtreturn_deviation') is not None else None,
                        stats.get('tsup_outside_band_count'),
                        float(stats.get('tsup_percentage')) if stats.get('tsup_percentage') is not None else None,
                        float(stats.get('max_tsup_deviation')) if stats.get('max_tsup_deviation') is not None else None,
                        *_signed_extreme_params(stats),
                        float(stats['avg_timestep_size']) if stats['avg_timestep_size'] is not None else None,
                        1 if stats.get('wb_calculated_from_rh', False) else 0,
                        float(stats['flow_setpoint']) if stats.get('flow_setpoint') is not None else None,
                        stats.get('flow_set_type'),
                        float(stats['mean_flow_dev_pct']) if stats.get('mean_flow_dev_pct') is not None else None,
                        stats.get('flow_outside_count'),
                        float(stats['flow_percentage']) if stats.get('flow_percentage') is not None else None,
                        resolved_profile_id,
                        resolved_condition_set_id,
                        datetime.now().isoformat(),
                        rowid
                    ))
                    calculated += 1
                except Exception as e:
                    print(f"ERROR: Error saving deviation stats for {file_name}, dataset {data_set}, rowid {rowid}: {e}")
                    import traceback
                    traceback.print_exc()
                    errors += 1
            else:
                # Entry couldn't be calculated (no valid test condition, missing data, etc.)
                # The calculate_entry_deviations function already prints detailed warnings
                errors += 1
                print(f"ERROR: Could not calculate deviations for {file_name}, dataset {data_set}, rowid {rowid} - see warnings above for details")
        
        conn.commit()
        
        return jsonify({
            'success': True,
            'calculated': calculated,
            'skipped': skipped,
            'errors': errors,
            'total': len(results),
            'warnings': warnings  # Include warnings in response
        })
    except Exception as e:
        conn.rollback()
        print(f"Error calculating deviations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()


@app.route('/recalculate_deviation_stat', methods=['POST'])
def recalculate_deviation_stat():
    """
    Recalculate one deviation statistic column for selected entries (rowids) or all entries.
    This is intended to power per-column "Update" buttons on the deviations page.
    Body: { stat: "<key>", entry_ids?: [rowid...] }
    """
    try:
        init_deviation_columns()
        data = request.get_json() or {}
        stat = str(data.get('stat') or '').strip()
        entry_ids = data.get('entry_ids') or []
        try:
            entry_ids = [int(x) for x in entry_ids if x is not None and str(x).strip() != '']
        except Exception:
            entry_ids = []

        if not entry_ids:
            return jsonify({'success': False, 'message': 'Please select one or more entries (checkboxes) before updating a single deviation statistic.'})

        # map UI stat -> metrics to compute + DB columns to update
        stat_map = {
            'dev_total_points': (['total_points'], ['dev_total_points']),
            'dev_avg_timestep': (['avg_timestep'], ['dev_avg_timestep']),
            'dev_db_outside_count': (['db_outside'], ['dev_db_outside_count']),
            'dev_db_percentage': (['db_pct'], ['dev_db_percentage']),
            'dev_max_db_deviation': (['max_db'], ['dev_max_db_deviation']),
            'dev_wb_outside_count': (['wb_outside'], ['dev_wb_outside_count']),
            'dev_wb_percentage': (['wb_pct'], ['dev_wb_percentage']),
            'dev_max_wb_deviation': (['max_wb'], ['dev_max_wb_deviation']),
            'dev_tsup_outside_count': (['tsup_outside'], ['dev_tsup_outside_count']),
            'dev_tsup_percentage': (['tsup_pct'], ['dev_tsup_percentage']),
            'dev_max_tsup_deviation': (['max_tsup'], ['dev_max_tsup_deviation']),
            'dev_max_db_pos': (['db_extremes'], ['dev_max_db_pos']),
            'dev_max_db_neg': (['db_extremes'], ['dev_max_db_neg']),
            'dev_max_wb_pos': (['wb_extremes'], ['dev_max_wb_pos']),
            'dev_max_wb_neg': (['wb_extremes'], ['dev_max_wb_neg']),
            'dev_max_tsup_pos': (['tsup_extremes'], ['dev_max_tsup_pos']),
            'dev_max_tsup_neg': (['tsup_extremes'], ['dev_max_tsup_neg']),
            'dev_max_dtreturn_pos': (['dtreturn_extremes'], ['dev_max_dtreturn_pos']),
            'dev_max_dtreturn_neg': (['dtreturn_extremes'], ['dev_max_dtreturn_neg']),
            'dev_max_flow_pos_pct': (['flow_extremes'], ['dev_max_flow_pos_pct']),
            'dev_max_flow_neg_pct': (['flow_extremes'], ['dev_max_flow_neg_pct']),
            'dev_dtreturn_outside_count': (['dtreturn_outside'], ['dev_dtreturn_outside_count']),
            'dev_dtreturn_percentage': (['dtreturn_pct'], ['dev_dtreturn_percentage']),
            'dev_max_dtreturn_deviation': (['max_dtreturn'], ['dev_max_dtreturn_deviation']),
            'dev_flow_outside_count': (['flow_outside'], ['dev_flow_outside_count']),
            'dev_flow_percentage': (['flow_pct'], ['dev_flow_percentage']),
            'dev_mean_flow_dev_pct': (['mean_flow_dev'], ['dev_mean_flow_dev_pct']),
        }

        if stat not in stat_map:
            return jsonify({'success': False, 'message': f'Unknown stat "{stat}"'})

        metrics, db_cols = stat_map[stat]

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            if entry_ids:
                placeholders = ",".join(["?"] * len(entry_ids))
                rows = cur.execute(
                    f"SELECT rowid, file_name, data_set, start_time, end_time, hp_id, "
                    f"profile_id, condition_set_id FROM results WHERE rowid IN ({placeholders})",
                    entry_ids,
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT rowid, file_name, data_set, start_time, end_time, hp_id, "
                    "profile_id, condition_set_id FROM results",
                ).fetchall()

            updated = 0
            errors = []
            from datetime import datetime

            for r in rows:
                rd = dict(r)
                rowid = int(rd['rowid'])
                file_name = rd.get('file_name')
                data_set = rd.get('data_set')
                start_time = rd.get('start_time')
                end_time = rd.get('end_time')
                hp_id = rd.get('hp_id')

                if not all([file_name, data_set is not None, start_time is not None, end_time is not None]):
                    continue

                try:
                    res = calculate_entry_deviation_metrics(
                        file_name, data_set, start_time, end_time,
                        metrics=metrics,
                        warning_list=None,
                        hp_id=hp_id,
                        entry=rd,
                    )
                    if not res:
                        continue

                    # Build update query for the requested DB cols only (plus dev_last_calculated)
                    set_parts = []
                    vals = []
                    for col in db_cols:
                        if col == 'dev_total_points':
                            v = res.get('total_points')
                        elif col == 'dev_avg_timestep':
                            v = res.get('avg_timestep_size')
                        elif col == 'dev_db_outside_count':
                            v = res.get('db_outside_band_count')
                        elif col == 'dev_db_percentage':
                            v = res.get('db_percentage')
                        elif col == 'dev_max_db_deviation':
                            v = res.get('max_db_deviation')
                        elif col == 'dev_wb_outside_count':
                            v = res.get('wb_outside_band_count')
                        elif col == 'dev_wb_percentage':
                            v = res.get('wb_percentage')
                        elif col == 'dev_max_wb_deviation':
                            v = res.get('max_wb_deviation')
                        elif col == 'dev_tsup_outside_count':
                            v = res.get('tsup_outside_band_count')
                        elif col == 'dev_tsup_percentage':
                            v = res.get('tsup_percentage')
                        elif col == 'dev_max_tsup_deviation':
                            v = res.get('max_tsup_deviation')
                        elif col == 'dev_dtreturn_outside_count':
                            v = res.get('dtreturn_outside_band_count')
                        elif col == 'dev_dtreturn_percentage':
                            v = res.get('dtreturn_percentage')
                        elif col == 'dev_max_dtreturn_deviation':
                            v = res.get('max_dtreturn_deviation')
                        elif col == 'dev_flow_outside_count':
                            v = res.get('flow_outside_count')
                        elif col == 'dev_flow_percentage':
                            v = res.get('flow_percentage')
                        elif col == 'dev_mean_flow_dev_pct':
                            v = res.get('mean_flow_dev_pct')
                        elif col.startswith('dev_max_') and col[len('dev_'):] in SIGNED_EXTREME_KEYS:
                            v = res.get(col[len('dev_'):])
                        else:
                            v = None

                        set_parts.append(f"{col} = ?")
                        vals.append(v)

                    set_parts.append("dev_last_calculated = ?")
                    vals.append(datetime.now().isoformat())
                    vals.append(rowid)

                    cur.execute(f"UPDATE results SET {', '.join(set_parts)} WHERE rowid = ?", vals)
                    updated += 1
                except Exception as e:
                    errors.append(f"rowid={rowid}: {e}")

            conn.commit()
            return jsonify({'success': True, 'updated': updated, 'errors': errors[:20]})
        finally:
            conn.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


def _load_results_entry(file_name, data_set, start_time, end_time):
    """Load the results row for one cycle. Empty dict if the row is missing."""
    try:
        conn = get_db_connection()
        try:
            row = conn.execute(
                "SELECT * FROM results WHERE file_name=? AND CAST(data_set AS TEXT)=CAST(? AS TEXT) "
                "AND start_time=? AND end_time=?",
                (file_name, data_set, start_time, end_time),
            ).fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()
    except Exception:
        return {}

@app.route('/plot_deviation', methods=['POST'])
def plot_deviation():
    """Generate deviation plots for a specific entry - DB/WB, Tsup, and dTreturn."""
    # Reload permissible deviations config to ensure latest values
    global PERMISSIBLE_DEVIATIONS, DB_BAND, WB_BAND, TSUP_BAND, DEVIATION_BANDS
    PERMISSIBLE_DEVIATIONS = load_permissible_deviations()
    DB_BAND = PERMISSIBLE_DEVIATIONS.get('DB', {}).get('value', 1.0)
    WB_BAND = PERMISSIBLE_DEVIATIONS.get('WB', {}).get('value', 1.0)
    TSUP_BAND = PERMISSIBLE_DEVIATIONS.get('Tsup', {}).get('value', 0.5)
    DEVIATION_BANDS = calculate_deviation_bands(DEVIATION_SETPOINTS, DB_BAND, WB_BAND)
    
    try:
        data = request.get_json()
        file_name = data.get('file_name')
        data_set = data.get('data_set')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        if not all([file_name, data_set is not None, start_time is not None, end_time is not None]):
            return jsonify({'success': False, 'message': 'Missing required parameters'})

        entry = _load_results_entry(file_name, data_set, start_time, end_time)
        hp_id = entry.get('hp_id')
        
        # Calculate deviations (includes cycle_data)
        stats = calculate_entry_deviations(
            file_name, data_set, start_time, end_time, hp_id=hp_id, entry=entry or None
        )
        if not stats or 'cycle_data' not in stats:
            return jsonify({'success': False, 'message': 'Could not load time series data'})
        
        cycle_data = stats['cycle_data']
        test_condition = stats['test_condition']
        tsup_setpoint = stats.get('tsup_setpoint')
        applicable_checks = _entry_applicable_checks({**entry, 'file_name': file_name, 'hp_id': hp_id})
        bands = DEVIATION_BANDS.get(test_condition) or {}
        has_db = 'db' in applicable_checks and 'T_outdoor (DB)' in cycle_data.columns
        has_wb = 'wb' in applicable_checks and 'T_outdoor (WB)' in cycle_data.columns

        # Outdoor plot is air-source only. Skip it for water-to-water so Tsup / dTreturn still render.
        img_base64_dbwb = None
        if (has_db or has_wb) and bands:
            plt.figure(figsize=(12, 6))
            if has_db:
                plt.plot(cycle_data['time_elapsed'], cycle_data['T_outdoor (DB)'],
                        label='Drybulb Temperature', linewidth=1.5)
            if has_wb:
                plt.plot(cycle_data['time_elapsed'], cycle_data['T_outdoor (WB)'],
                        label='Wetbulb Temperature', linewidth=1.5)

            db_band = bands.get('DB') if has_db else None
            wb_band = bands.get('WB') if has_wb else None
            if db_band:
                plt.axhspan(db_band[0], db_band[1], color='blue', alpha=0.1, label='DB Band')
                plt.axhline(y=db_band[0], color='blue', linestyle='--', linewidth=1)
                plt.axhline(y=db_band[1], color='blue', linestyle='--', linewidth=1)
            if wb_band:
                plt.axhspan(wb_band[0], wb_band[1], color='red', alpha=0.1, label='WB Band')
                plt.axhline(y=wb_band[0], color='red', linestyle='--', linewidth=1)
                plt.axhline(y=wb_band[1], color='red', linestyle='--', linewidth=1)

            plt.xlabel('Time Elapsed (s)', fontsize=12)
            plt.ylabel('Temperature (°C)', fontsize=12)
            plt.title(f'DB/WB Permissible Deviation: {file_name} (Dataset {data_set})', fontsize=14)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)

            img_buffer_dbwb = BytesIO()
            plt.savefig(img_buffer_dbwb, format='png', dpi=100, bbox_inches='tight')
            img_buffer_dbwb.seek(0)
            img_base64_dbwb = base64.b64encode(img_buffer_dbwb.getvalue()).decode('utf-8')
            plt.close()
        
        # Create Tsup plot if supply temperature column exists (prefer Ts Buh)
        img_base64_tsup = None
        # Determine which column to use: prefer Ts Buh for deviation analysis
        tsup_column = None
        if 'Ts Buh' in cycle_data.columns:
            tsup_column = 'Ts Buh'
        elif 'T_supply' in cycle_data.columns:
            tsup_column = 'T_supply'
        
        if tsup_column and tsup_setpoint is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(cycle_data['time_elapsed'], cycle_data[tsup_column], 
                    label=f'Supply Temperature ({tsup_column})', linewidth=1.5, color='green')
            
            # Calculate mean Tsup value
            mean_tsup = float(cycle_data[tsup_column].mean())
            
            # Add Tsup setpoint line
            plt.axhline(y=tsup_setpoint, color='green', linestyle='--', linewidth=2, 
                       label=f'Tsup Setpoint ({tsup_setpoint}°C)', alpha=0.7)
            
            # Add mean Tsup value line (important: deviation is for mean value)
            plt.axhline(y=mean_tsup, color='orange', linestyle='-', linewidth=2, 
                       label=f'Mean Tsup ({mean_tsup:.2f}°C)', alpha=0.8)
            
            # Add deviation band (from configuration) - centered on setpoint
            tsup_band = PERMISSIBLE_DEVIATIONS.get('Tsup', {}).get('value', 0.5)
            tsup_band_lower = tsup_setpoint - tsup_band
            tsup_band_upper = tsup_setpoint + tsup_band
            plt.axhspan(tsup_band_lower, tsup_band_upper, color='green', alpha=0.1, 
                       label=f'Tsup Band (±{tsup_band}K)')
            plt.axhline(y=tsup_band_lower, color='green', linestyle='--', linewidth=1, alpha=0.5)
            plt.axhline(y=tsup_band_upper, color='green', linestyle='--', linewidth=1, alpha=0.5)
            
            plt.xlabel('Time Elapsed (s)', fontsize=12)
            plt.ylabel('Temperature (°C)', fontsize=12)
            plt.title(f'Tsup Permissible Deviation: {file_name} (Dataset {data_set})', fontsize=14)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Convert Tsup plot to base64
            img_buffer_tsup = BytesIO()
            plt.savefig(img_buffer_tsup, format='png', dpi=100, bbox_inches='tight')
            img_buffer_tsup.seek(0)
            img_base64_tsup = base64.b64encode(img_buffer_tsup.getvalue()).decode('utf-8')
            plt.close()

        # Create dTreturn plot (T_return_emu - T_return_calc) with asymmetric bounds
        img_base64_dtreturn = None
        if 'T_return_emu' in cycle_data.columns and 'T_return_calc' in cycle_data.columns:
            dt_cfg = PERMISSIBLE_DEVIATIONS.get('dTreturn', {'lower': -2.0, 'upper': 2.0})
            try:
                dt_lower = float(dt_cfg.get('lower', -2.0))
                dt_upper = float(dt_cfg.get('upper', 2.0))
            except Exception:
                dt_lower, dt_upper = -2.0, 2.0
            dtreturn = cycle_data['T_return_emu'] - cycle_data['T_return_calc']
            plt.figure(figsize=(12, 5))
            plt.plot(cycle_data['time_elapsed'], dtreturn, label='dTreturn = T_return_emu − T_return_calc', linewidth=1.5, color='purple')
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='0 K')
            plt.axhspan(dt_lower, dt_upper, color='purple', alpha=0.12, label=f'Band ({dt_lower:+.1f}…{dt_upper:+.1f}) K')
            plt.axhline(y=dt_lower, color='purple', linestyle=':', linewidth=1, alpha=0.7)
            plt.axhline(y=dt_upper, color='purple', linestyle=':', linewidth=1, alpha=0.7)
            plt.xlabel('Time Elapsed (s)', fontsize=12)
            plt.ylabel('Temperature difference (K)', fontsize=12)
            plt.title(f'dTreturn Permissible Deviation: {file_name} (Dataset {data_set})', fontsize=14)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            img_buffer_dt = BytesIO()
            plt.savefig(img_buffer_dt, format='png', dpi=100, bbox_inches='tight')
            img_buffer_dt.seek(0)
            img_base64_dtreturn = base64.b64encode(img_buffer_dt.getvalue()).decode('utf-8')
            plt.close()
        
        return jsonify({
            'success': True,
            'image_dbwb': f'data:image/png;base64,{img_base64_dbwb}' if img_base64_dbwb else None,
            'image_tsup': f'data:image/png;base64,{img_base64_tsup}' if img_base64_tsup else None,
            'image_dtreturn': f'data:image/png;base64,{img_base64_dtreturn}' if img_base64_dtreturn else None
        })
    except Exception as e:
        print(f"Error generating deviation plot: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/plot_flow_deviation', methods=['POST'])
def plot_flow_deviation():
    """Generate flow deviation plot (time series with set value and ±band) for fixed-flow entries."""
    try:
        req = request.get_json()
        file_name = req.get('file_name')
        data_set = req.get('data_set')
        start_time = req.get('start_time')
        end_time = req.get('end_time')
        hp_id = req.get('hp_id')
        if not all([file_name, start_time is not None, end_time is not None]):
            return jsonify({'success': False, 'message': 'Missing required parameters (file_name, start_time, end_time)'})
        # Coerce to float so Excel filtering and DB lookups work (JSON may send int or string)
        try:
            start_time = float(start_time)
            end_time = float(end_time)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'message': 'Invalid start_time or end_time'})
        flow_type, flow_set_value = get_flow_set_for_hp(hp_id, file_name=file_name) if (hp_id or file_name) else (None, None)
        if not flow_type or flow_set_value is None or flow_set_value <= 0:
            return jsonify({'success': False, 'message': f'No flow set point configured for HP "{hp_id}" (check Design parameters on this page)'})
        entry = _load_results_entry(file_name, data_set, start_time, end_time)
        if hp_id is not None:
            entry.setdefault('hp_id', hp_id)
        if req.get('profile_id'):
            entry.setdefault('profile_id', req.get('profile_id'))
        entry.setdefault('file_name', file_name)
        if _entry_is_variable_flow(entry):
            return jsonify({'success': False, 'message': 'Flow band does not apply to variable-flow tests (judged on Tmean / Tsup).'})

        # Prefer cycle_data from the same path as DB/WB/Tsup plot (calculate_entry_deviations).
        # If that fails (no stored time series), load from Excel using process_file.
        cycle_data = None
        stats = calculate_entry_deviations(
            file_name, data_set, start_time, end_time, hp_id=hp_id, entry=entry or None
        )
        if stats and stats.get('cycle_data') is not None and not stats['cycle_data'].empty:
            cycle_data = stats['cycle_data']
        if cycle_data is None or cycle_data.empty:
            file_path = os.path.join(config['data_dir'], file_name)
            if not os.path.exists(file_path):
                return jsonify({'success': False, 'message': f'Excel file not found: {file_path}'})
            cycle_data = process_file(file_name, data_set, start_time, end_time)
        if cycle_data is None or cycle_data.empty:
            return jsonify({'success': False, 'message': 'Could not load time series data for this entry. The Excel file may use a different sheet or time range than expected.'})

        # Ensure mass flow is available: measured wins, empty or absent is derived
        cycle_data, mass_flow_notices = derive_mass_flow_if_needed(cycle_data)
        for note in mass_flow_notices:
            print(note)

        flow_col = 'mass flow' if flow_type == 'mass' else 'volume flow'
        if flow_col not in cycle_data.columns:
            available = [c for c in cycle_data.columns if 'flow' in c.lower()]
            return jsonify({'success': False, 'message': f'Column "{flow_col}" not found in time series data for {file_name}. Available flow-related columns: {available}'})

        flow_instantaneous_pct = load_permissible_deviations().get('flow_instantaneous_pct', 2.5)
        band_low = flow_set_value * (1 - flow_instantaneous_pct / 100.0)
        band_high = flow_set_value * (1 + flow_instantaneous_pct / 100.0)
        mean_flow = float(cycle_data[flow_col].mean())
        unit = 'kg/s' if flow_type == 'mass' else 'm³/h'
        plt.figure(figsize=(12, 5))
        plt.plot(cycle_data['time_elapsed'], cycle_data[flow_col], label=f'{flow_col}', linewidth=1.5, color='teal')
        plt.axhline(y=flow_set_value, color='green', linestyle='--', linewidth=2, label=f'Set point: {flow_set_value:.4g} {unit}')
        plt.axhline(y=mean_flow, color='orange', linestyle='-', linewidth=2, label=f'Mean: {mean_flow:.4g} {unit}', alpha=0.85)
        plt.axhspan(band_low, band_high, color='green', alpha=0.12, label=f'±{flow_instantaneous_pct}% band ({band_low:.4g}–{band_high:.4g})')
        plt.axhline(y=band_low, color='green', linestyle=':', linewidth=1, alpha=0.6)
        plt.axhline(y=band_high, color='green', linestyle=':', linewidth=1, alpha=0.6)
        plt.xlabel('Time Elapsed (s)', fontsize=12)
        plt.ylabel(f'{flow_col} ({unit})', fontsize=12)
        plt.title(f'Flow Permissible Deviation: {file_name} (Dataset {data_set})', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        return jsonify({'success': True, 'image': f'data:image/png;base64,{img_base64}'})
    except Exception as e:
        print(f"Error generating flow deviation plot: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/scatter')
def scatter():
    """Display scatter plot generation page"""
    # Initialize scatter_dataset column if needed
    init_scatter_dataset_column()
    
    conn = get_db_connection()
    try:
        # Get all entries from database
        results = conn.execute('''
            SELECT *, rowid FROM results 
            ORDER BY display_order ASC, rowid ASC
        ''').fetchall()
        results = [dict(row) for row in results]
        
        # Find the actual column names for filtering (similar to deviations page)
        column_name_map = {}
        if results:
            first_entry = results[0]
            all_keys = list(first_entry.keys())
            
            # Detect HP ID, Lab ID, Flow Config columns
            for key in all_keys:
                key_lower = key.lower()
                if 'hp' in key_lower and 'id' in key_lower and 'HP_ID' not in column_name_map:
                    column_name_map['HP_ID'] = key
                if 'lab' in key_lower and 'id' in key_lower and 'Lab_ID' not in column_name_map:
                    column_name_map['Lab_ID'] = key
                if 'flow' in key_lower and 'config' in key_lower and 'Flow_Config' not in column_name_map:
                    column_name_map['Flow_Config'] = key
        
        # Filter entries - use scatter_dataset if it has entries, otherwise show empty (user must select)
        default_entries = []
        using_scatter_dataset = False
        
        # Check if scatter_dataset has any entries
        scatter_entries = []
        for entry in results:
            scatter_value = entry.get('scatter_dataset')
            if scatter_value and str(scatter_value).upper() in ['YES', 'Y', '1', 'TRUE']:
                scatter_entries.append(entry)
        
        if scatter_entries:
            # Use scatter_dataset entries
            default_entries = scatter_entries
            using_scatter_dataset = True
        # If scatter_dataset is empty, default_entries stays empty (user needs to select entries)
        
        # Extract test condition from file_name; collect unit + flow combinations
        import re
        all_idents = []
        unresolved_files = []
        for entry in default_entries:
            file_name = entry.get('file_name', '')
            test_condition = None
            if file_name:
                parts = file_name.split('_')
                if len(parts) >= 3:
                    test_condition = parts[2][0] if len(parts[2]) > 0 else None
                    if test_condition not in ['E', 'A', 'F', 'B', 'C', 'D']:
                        name_without_ext = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                        match = re.search(r'\s+([EAFBCD])', name_without_ext, re.IGNORECASE)
                        if match:
                            test_condition = match.group(1).upper()
            entry['extracted_test_condition'] = test_condition

            ident = unit_config.unit_key_and_label(entry)
            all_idents.append(ident)
            if ident.get('unresolved') and ident.get('file_name'):
                unresolved_files.append(ident['file_name'])

        all_idents = unit_config.disambiguate_labels(all_idents)

        combo_map = {}
        for ident in all_idents:
            flow = ident.get('flow_config') or ''
            combo_key = (ident['unit_key'], flow)
            if combo_key not in combo_map:
                combo_map[combo_key] = {
                    'unit_key': ident['unit_key'],
                    'label': ident['label'],
                    'flow': flow,
                    'source': ident['source'],
                }

        available_combinations = sorted(
            combo_map.values(),
            key=lambda c: (str(c.get('label') or '').lower(), c.get('flow') or '', c.get('unit_key') or ''),
        )
        unresolved_files = sorted(set(unresolved_files))

        # Resolve and store climate × application per entry for display and filtering
        climate_app_set = set()
        for entry in default_entries:
            clim, app = _entry_climate_application(entry)
            entry['_resolved_climate'] = clim
            entry['_resolved_application'] = app
            climate_app_set.add((clim, app))
        default_climate = unit_config.get_default_climate()
        default_application = unit_config.get_default_application()
        available_climate_app = []
        for clim, app in sorted(climate_app_set):
            is_default = (clim == default_climate and app == default_application)
            lbl = f"{clim} / {app}"
            if is_default:
                lbl += " (default)"
            available_climate_app.append({
                'climate': clim,
                'application': app,
                'label': lbl,
                'is_default': is_default,
            })

        # Get available parameters for Y-axis selection
        # These are the columns that can be plotted
        available_parameters = []
        if results:
            first_entry = results[0]
            # Exclude system columns and file identifiers
            exclude_cols = {'rowid', 'file_name', 'data_set', 'start_time', 'end_time', 
                          'has_time_series', 'time_series_data', 'display_order'}
            exclude_cols.update({col for col in first_entry.keys() if col.startswith('dev_')})
            
            # Also exclude HP ID, Lab ID, Flow Config columns (these are grouping columns, not plot values)
            for key in first_entry.keys():
                key_lower = key.lower()
                if ('hp' in key_lower and 'id' in key_lower) or \
                   ('lab' in key_lower and 'id' in key_lower) or \
                   ('flow' in key_lower and 'config' in key_lower):
                    exclude_cols.add(key)
            
            for key in first_entry.keys():
                if key not in exclude_cols:
                    value = first_entry.get(key)
                    # Only include numeric columns or columns that might be numeric
                    if value is not None:
                        try:
                            float(value)
                            available_parameters.append(key)
                        except (ValueError, TypeError):
                            pass
        
        # Mean permissible deviations for scatter plot bands (plot-only; defaults from main config)
        scatter_bands_config = load_permissible_deviations()
        scatter_mean_limits = {
            'mean_db_k': scatter_bands_config.get('mean_db_k', 0.6),
            'mean_wb_k': scatter_bands_config.get('mean_wb_k', 0.4),
            'mean_tsup_k': scatter_bands_config.get('mean_tsup_k', 0.5),
            'mean_q_pct': scatter_bands_config.get('mean_q_band_pct', 5.0),
            'mean_flow_pct': scatter_bands_config.get('mean_flow_pct', 1.0),
        }
        return render_template('scatter.html',
                             all_entries=results,
                             default_entries=default_entries,
                             column_name_map=column_name_map,
                             available_parameters=available_parameters,
                             using_scatter_dataset=using_scatter_dataset,
                             available_combinations=available_combinations,
                             unresolved_files=unresolved_files,
                             scatter_mean_limits=scatter_mean_limits,
                             available_climate_app=available_climate_app)
    finally:
        conn.close()

@app.route('/generate_scatter_plots', methods=['POST'])
def generate_scatter_plots():
    """Generate scatter plots for selected entries"""
    try:
        data = request.get_json()
        entry_ids = data.get('entry_ids', [])  # List of rowids
        # Support batch: y_parameters (list) generates all in one request so image cache stays in same process
        y_parameters = data.get('y_parameters')
        if y_parameters is None or not isinstance(y_parameters, list):
            single = data.get('y_parameter')
            y_parameters = [single] if single else []
        column_name_map = data.get('column_name_map', {})
        selected_combinations = data.get('selected_combinations', [])  # List of [hp_label, flow_config] pairs
        selected_climate_app = data.get('selected_climate_app', [])   # List of [climate, application] pairs
        # Plot-only mean limits for bands (optional; defaults from config)
        plot_mean_limits = data.get('scatter_mean_limits', {})
        cfg = load_permissible_deviations()
        mean_db_k = float(plot_mean_limits.get('mean_db_k', cfg.get('mean_db_k', 0.6)))
        mean_wb_k = float(plot_mean_limits.get('mean_wb_k', cfg.get('mean_wb_k', 0.4)))
        mean_tsup_k = float(plot_mean_limits.get('mean_tsup_k', cfg.get('mean_tsup_k', 0.5)))
        mean_q_pct = float(plot_mean_limits.get('mean_q_pct', cfg.get('mean_q_band_pct', 5.0)))
        mean_flow_pct = float(plot_mean_limits.get('mean_flow_pct', cfg.get('mean_flow_pct', 1.0)))
        
        if not entry_ids:
            return jsonify({'success': False, 'message': 'No entries selected'})
        if not y_parameters:
            return jsonify({'success': False, 'message': 'No Y-axis parameter(s) selected'})
        # Optional safeguard: set to 0 to allow unlimited parameters (e.g. "all at once")
        MAX_PARAMETERS = 0  # 0 = no limit; was 30 to prevent accidental long runs
        if MAX_PARAMETERS and len(y_parameters) > MAX_PARAMETERS:
            return jsonify({
                'success': False,
                'message': f'Too many Y-axis parameters selected ({len(y_parameters)}). Please select at most {MAX_PARAMETERS} parameters.'
            })
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # Fetch selected entries
            placeholders = ','.join(['?'] * len(entry_ids))
            query = f'SELECT *, rowid FROM results WHERE rowid IN ({placeholders})'
            results = conn.execute(query, entry_ids).fetchall()
            entries = [dict(row) for row in results]
            
            if not entries:
                return jsonify({'success': False, 'message': 'No entries found'})
            
            # Get column names for grouping
            # Try multiple possible column name variations
            hp_column = column_name_map.get('HP_ID', 'HP_ID')
            flow_column = column_name_map.get('Flow_Config', 'Flow_Config')
            lab_column = column_name_map.get('Lab_ID', 'Lab_ID')
            test_column = None  # We'll extract from file_name
            
            # If column_name_map doesn't have the keys, try to detect from first entry
            if entries:
                first_entry = entries[0]
                all_keys = list(first_entry.keys())
                
                # Try to find HP_ID column (case-insensitive, handle spaces)
                if hp_column not in first_entry:
                    for key in all_keys:
                        key_lower = key.lower().replace(' ', '_')
                        if key_lower in ['hp_id', 'hpid', 'hp']:
                            hp_column = key
                            print(f"Detected HP_ID column: {key}")
                            break
                
                # Try to find Flow_Config column
                if flow_column not in first_entry:
                    for key in all_keys:
                        key_lower = key.lower().replace(' ', '_')
                        if key_lower in ['flow_config', 'flowconfig', 'flow']:
                            flow_column = key
                            print(f"Detected Flow_Config column: {key}")
                            break
                
                # Try to find Lab_ID column
                if lab_column not in first_entry:
                    for key in all_keys:
                        key_lower = key.lower().replace(' ', '_')
                        if key_lower in ['lab_id', 'labid', 'lab']:
                            lab_column = key
                            print(f"Detected Lab_ID column: {key}")
                            break
            
            # Get column types from database for comparison
            cursor.execute("PRAGMA table_info(results)")
            db_columns_info = {col[1]: {'type': col[2], 'notnull': col[3], 'dflt_value': col[4]} 
                              for col in cursor.fetchall()}
            
            # Check column types
            hp_col_type = db_columns_info.get(hp_column, {}).get('type', 'UNKNOWN')
            flow_col_type = db_columns_info.get(flow_column, {}).get('type', 'UNKNOWN')

            # Resolve climate × application per entry; filter by selected slices
            entry_climate_app = [_entry_climate_application(e) for e in entries]
            if selected_climate_app:
                ca_set = {(str(ca[0]).strip(), str(ca[1]).strip()) for ca in selected_climate_app if len(ca) >= 2}
                if ca_set:
                    keep = [i for i, (c, a) in enumerate(entry_climate_app) if (c, a) in ca_set]
                    entries = [entries[i] for i in keep]
                    entry_climate_app = [entry_climate_app[i] for i in keep]
                    if not entries:
                        return jsonify({'success': False, 'message': 'No entries match the selected climate / application filter.'})
            multi_slice = len(set(entry_climate_app)) > 1

            generated_plots = []
            entry_idents = unit_config.disambiguate_labels(
                [unit_config.unit_key_and_label(e) for e in entries]
            )
            unresolved_warning_files = sorted({
                ident['file_name']
                for ident in entry_idents
                if ident.get('unresolved') and ident.get('file_name')
            })
            for y_parameter in y_parameters:
                plot_data = []
                missing_hp = []
                missing_flow = []
                debug_info = []
                debug_info.append(f"Column mapping: HP_ID='{hp_column}', Flow_Config='{flow_column}', Lab_ID='{lab_column}'")
                debug_info.append(f"Database column types: HP_ID='{hp_column}' is {hp_col_type}, Flow_Config='{flow_column}' is {flow_col_type}")
                debug_info.append(f"Total entries received: {len(entries)}")
            
                working_entries = []
                non_working_entries = []
            
                for i, entry in enumerate(entries):
                    file_name = entry.get('file_name', 'Unknown')
                    ident = entry_idents[i]
                    hp_label = ident['unit_key']
                    unit_label = ident['label']
                    flow_config = ident.get('flow_config') or ''
                    lab_id = entry.get(lab_column) if lab_column in entry else None
                    hp_label_raw = entry.get(hp_column) if hp_column in entry else None
                    flow_config_raw = entry.get(flow_column) if flow_column in entry else None
                    e_clim, e_app = entry_climate_app[i]

                    debug_info.append(
                        f"  {file_name}: unit_key={hp_label!r} label={unit_label!r} "
                        f"source={ident['source']} HP_ID={hp_label_raw!r} Flow={flow_config_raw!r}"
                    )

                    working_entries.append({
                        'file_name': file_name,
                        'hp_label': hp_label,
                        'unit_label': unit_label,
                        'flow_config': flow_config,
                        'source': ident['source'],
                    })
                    if ident.get('unresolved'):
                        missing_hp.append(file_name)
                        debug_info.append(
                            f"  {file_name}: no profile or HP ID — plotted as its own series. "
                            f"Assign a unit profile so series stay comparable."
                        )
                    if not flow_config:
                        missing_flow.append(file_name)

                    test_condition = None
                    if file_name:
                        parts = file_name.split('_')
                        if len(parts) >= 3:
                            test_condition = parts[2][0] if len(parts[2]) > 0 else None
                            if test_condition not in ['E', 'A', 'F', 'B', 'C', 'D']:
                                import re
                                name_without_ext = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                                match = re.search(r'\s+([EAFBCD])', name_without_ext, re.IGNORECASE)
                                if match:
                                    test_condition = match.group(1).upper()

                    y_value = entry.get(y_parameter)
                    if y_value is not None:
                        try:
                            y_value = float(y_value)
                            plot_data.append({
                                'hp_label': hp_label,
                                'unit_label': unit_label,
                                'flow_config': flow_config,
                                'lab_id': lab_id,
                                'test_condition': test_condition,
                                'y_value': y_value,
                                'file_name': file_name,
                                'profile_id': ident.get('profile_id') or '',
                                'hp_id': ident.get('hp_id') or '',
                                'unit_source': ident['source'],
                                'climate': e_clim,
                                'application': e_app,
                            })
                        except (ValueError, TypeError) as e:
                            print(f"  Skipped (invalid Y value): {y_value} - {e}")
                            pass
                    else:
                        print(f"  Skipped (no Y value for parameter '{y_parameter}')")
            
                if not plot_data:
                    continue
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(plot_data)
            
                # Ensure hp_label and flow_config are strings for consistent comparison
                if len(df) > 0:
                    def normalize_hp(x):
                        if x is None or pd.isna(x):
                            return None
                        s = str(x).strip()
                        return None if s == '' or s.lower() == 'none' else s

                    def normalize_flow(x):
                        if x is None or pd.isna(x):
                            return ''
                        s = str(x).strip().lower()
                        return '' if s == '' or s == 'none' else s

                    df['hp_label'] = df['hp_label'].apply(normalize_hp)
                    df['flow_config'] = df['flow_config'].apply(normalize_flow)

                # Keep every row that has a unit key; missing flow is allowed
                initial_count = len(df)
                df = df[df['hp_label'].notna() & (df['hp_label'] != '') & (df['hp_label'] != 'None')]
                filtered_count = len(df)
            
                # Debug output
                print(f"Column mapping: HP_ID={hp_column}, Flow_Config={flow_column}, Lab_ID={lab_column}")
                print(f"Total entries: {len(entries)}, Valid plot data points: {initial_count}, After filtering: {filtered_count}")
                if missing_hp:
                    print(f"Entries missing HP_ID: {set(missing_hp)}")
                if missing_flow:
                    print(f"Entries missing Flow_Config: {set(missing_flow)}")
            
                if len(df) == 0:
                    continue
                # (batch: skip parameter when no valid df)
                # Group by HP label and flow config, generate plots (append to shared generated_plots)
                # Get unique HP/Flow combinations
                print(f"DEBUG: DataFrame shape before grouping: {df.shape}")
                print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
                print(f"DEBUG: DataFrame HP labels (first 20): {df['hp_label'].head(20).tolist()}")
                print(f"DEBUG: DataFrame flow configs (first 20): {df['flow_config'].head(20).tolist()}")
            
                hp_flow_combos = df.groupby(['hp_label', 'flow_config']).size().reset_index()
                hp_flow_combos.columns = ['hp_label', 'flow_config', 'count']
            
                print(f"Found {len(hp_flow_combos)} HP/Flow combinations for plotting")
                print(f"DEBUG: HP/Flow combinations found:")
                for _, combo in hp_flow_combos.iterrows():
                    print(f"  - HP{combo['hp_label']} ({combo['flow_config']}): {combo['count']} data points")
            
                # Filter by selected combinations if provided
                if selected_combinations:
                    # Convert selected_combinations to set of tuples for fast lookup
                    # Normalize to strings for comparison
                    selected_set = set()
                    for combo in selected_combinations:
                        if isinstance(combo, list) and len(combo) >= 1:
                            hp_val = str(combo[0]).strip() if combo[0] is not None else ''
                            flow_val = ''
                            if len(combo) >= 2 and combo[1] is not None:
                                flow_val = str(combo[1]).strip().lower()
                            if hp_val:
                                selected_set.add((hp_val, flow_val))
                
                    print(f"DEBUG: Selected combinations (normalized): {selected_set}")
                    print(f"DEBUG: HP/Flow combinations before filtering: {len(hp_flow_combos)}")
                    print(f"DEBUG: Sample combinations before filtering:")
                    for idx, row in hp_flow_combos.head(5).iterrows():
                        normalized = (str(row['hp_label']).strip(), str(row['flow_config']).strip().lower())
                        print(f"  - Row {idx}: raw=({repr(row['hp_label'])}, {repr(row['flow_config'])}), normalized={normalized}, in_set={normalized in selected_set}")
                
                    # Filter hp_flow_combos (normalize for comparison)
                    hp_flow_combos = hp_flow_combos[
                        hp_flow_combos.apply(lambda row: 
                            (str(row['hp_label']).strip(), str(row['flow_config']).strip().lower()) in selected_set, 
                            axis=1)
                    ]
                    print(f"DEBUG: Filtered to {len(hp_flow_combos)} selected combinations")
                    if len(hp_flow_combos) > 0:
                        print(f"DEBUG: Remaining combinations after filtering:")
                        for idx, row in hp_flow_combos.iterrows():
                            print(f"  - Row {idx}: hp_label={repr(row['hp_label'])}, flow_config={repr(row['flow_config'])}")
            
                if len(hp_flow_combos) == 0:
                    print(f"WARNING: hp_flow_combos is empty after filtering by selected_combinations!")
                    print(f"DEBUG: This means no combinations matched the selected_combinations: {selected_combinations}")
                    # Show what combinations were actually found
                    if 'df' in locals() and len(df) > 0:
                        all_combos = df.groupby(['hp_label', 'flow_config']).size().reset_index()
                        all_combos.columns = ['hp_label', 'flow_config', 'count']
                        print(f"DEBUG: All combinations found by groupby:")
                        for _, combo in all_combos.iterrows():
                            normalized = (str(combo['hp_label']).strip(), str(combo['flow_config']).strip().lower())
                            print(f"  - HP{combo['hp_label']} ({combo['flow_config']}): {combo['count']} data points (normalized: {normalized})")
                            print(f"    In selected_set? {normalized in selected_set if 'selected_set' in locals() else 'N/A'}")
            
                print(f"DEBUG: About to enter loop with {len(hp_flow_combos)} combinations")
                if len(hp_flow_combos) == 0:
                    print(f"ERROR: hp_flow_combos is empty - loop will not execute!")
            
                for idx, combo in hp_flow_combos.iterrows():
                    # Get raw values from combo DataFrame - these are the EXACT values from the groupby
                    combo_hp_raw = combo['hp_label']
                    combo_flow_raw = combo['flow_config']
                
                    print(f"DEBUG: Processing combination #{idx}")
                    print(f"DEBUG: Combo raw values: hp_label={repr(combo_hp_raw)} (type:{type(combo_hp_raw).__name__}), flow_config={repr(combo_flow_raw)} (type:{type(combo_flow_raw).__name__})")
                    print(f"DEBUG: DataFrame shape before filtering: {df.shape}")
                
                    # Use the EXACT SAME filtering method as the error message code (which works!)
                    # Match the error message code exactly: str(combo['hp_label']).strip() and str(combo['flow_config']).strip().lower()
                    filtered_data = df[
                        (df['hp_label'].astype(str).str.strip() == str(combo['hp_label']).strip()) & 
                        (df['flow_config'].astype(str).str.strip().str.lower() == str(combo['flow_config']).strip().lower())
                    ]
                
                    # Normalize for display and plot generation
                    hp_label = str(combo_hp_raw).strip() if pd.notna(combo_hp_raw) else ''
                    flow_config = str(combo_flow_raw).strip().lower() if pd.notna(combo_flow_raw) else ''
                
                    print(f"DEBUG: Normalized values: hp_label='{hp_label}', flow_config='{flow_config}'")
                    print(f"DEBUG: Filtering result: {len(filtered_data)} rows")
                
                    print(f"DEBUG: Filtering result: {len(filtered_data)} rows")
                
                    print(f"DEBUG: Filtered data shape: {filtered_data.shape}")
                
                    # Debug: Show what values are actually in the DataFrame
                    if len(filtered_data) == 0:
                        print(f"DEBUG: No data found - showing sample of df values")
                        print(f"DEBUG: Unique HP labels in DataFrame: {sorted(df['hp_label'].astype(str).str.strip().unique().tolist())}")
                        print(f"DEBUG: Unique flow configs in DataFrame: {sorted(df['flow_config'].astype(str).str.strip().str.lower().unique().tolist())}")
                    
                        # Try to find matches separately
                        hp_matches = df[df['hp_label'].astype(str).str.strip() == hp_label]
                        flow_matches = df[df['flow_config'].astype(str).str.strip().str.lower() == flow_config]
                        print(f"DEBUG: Rows with HP='{hp_label}': {len(hp_matches)}")
                        print(f"DEBUG: Rows with Flow='{flow_config}': {len(flow_matches)}")
                        if len(hp_matches) > 0:
                            print(f"DEBUG: Sample HP values (raw): {hp_matches['hp_label'].head(3).tolist()}")
                        if len(flow_matches) > 0:
                            print(f"DEBUG: Sample Flow values (raw): {flow_matches['flow_config'].head(3).tolist()}")
                
                    print(f"DEBUG: Final filtered data shape: {filtered_data.shape}")
                    if len(filtered_data) > 0:
                        print(f"DEBUG: Filtered HP labels: {filtered_data['hp_label'].unique().tolist()}")
                        print(f"DEBUG: Filtered flow configs: {filtered_data['flow_config'].unique().tolist()}")
                    else:
                        print(f"DEBUG: No data found - showing sample of df values")
                        print(f"DEBUG: Sample HP values: {df['hp_label'].head(10).tolist()}")
                        print(f"DEBUG: Sample Flow values: {df['flow_config'].head(10).tolist()}")
                        print(f"DEBUG: Combo HP value type: {type(combo_hp_raw).__name__}, value: {repr(combo_hp_raw)}")
                        print(f"DEBUG: Combo Flow value type: {type(combo_flow_raw).__name__}, value: {repr(combo_flow_raw)}")
                
                    if len(filtered_data) == 0:
                        print(f"WARNING: No data found for HP{hp_label} ({flow_config}) after filtering!")
                        print(f"DEBUG: Sample of df['hp_label'] values: {df['hp_label'].head(10).tolist()}")
                        print(f"DEBUG: Sample of df['flow_config'] values: {df['flow_config'].head(10).tolist()}")
                        continue
                
                    print(f"Generating plot for {hp_label} ({flow_config}) with {len(filtered_data)} data points")

                    unit_label = hp_label
                    profile_id = None
                    if 'unit_label' in filtered_data.columns and len(filtered_data):
                        unit_label = str(filtered_data['unit_label'].iloc[0] or hp_label)
                    if 'profile_id' in filtered_data.columns and len(filtered_data):
                        raw_pid = filtered_data['profile_id'].iloc[0]
                        if raw_pid is not None and not pd.isna(raw_pid):
                            profile_id = str(raw_pid).strip() or None

                    # Append climate/application to label when showing multiple slices
                    plot_unit_label = unit_label
                    plot_climate = ''
                    plot_application = ''
                    if multi_slice and 'climate' in filtered_data.columns and 'application' in filtered_data.columns:
                        slices = filtered_data[['climate', 'application']].drop_duplicates()
                        slice_labels = [f"{r['climate']}/{r['application']}" for _, r in slices.iterrows()]
                        slice_tag = ', '.join(sorted(slice_labels))
                        plot_unit_label = f"{unit_label} [{slice_tag}]"
                        if len(slices) == 1:
                            plot_climate = str(slices.iloc[0]['climate'])
                            plot_application = str(slices.iloc[0]['application'])

                    try:
                        plot_image = generate_stripplot(
                            filtered_data, hp_label, flow_config, y_parameter,
                            mean_db_k=mean_db_k, mean_wb_k=mean_wb_k, mean_tsup_k=mean_tsup_k,
                            mean_q_pct=mean_q_pct, mean_flow_pct=mean_flow_pct,
                            unit_label=plot_unit_label, profile_id=profile_id,
                        )

                        generated_plots.append({
                            'hp_label': hp_label,
                            'unit_label': plot_unit_label,
                            'flow_config': flow_config,
                            'image': plot_image,
                            'entry_count': len(filtered_data),
                            'y_parameter': y_parameter,
                            'climate': plot_climate,
                            'application': plot_application,
                        })
                    except Exception as e:
                        print(f"Error generating plot for HP{hp_label} ({flow_config}): {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            print(f"Successfully generated {len(generated_plots)} plots")
            
            # CRITICAL: Always check if plots were generated before returning success
            # Filter out any plots with missing or invalid images
            valid_plots = [p for p in generated_plots if p.get('image') and len(p.get('image', '')) > 0]
            
            print(f"DEBUG: Total plots attempted: {len(generated_plots)}, Valid plots: {len(valid_plots)}")
            if len(generated_plots) > 0 and len(valid_plots) == 0:
                print(f"WARNING: All {len(generated_plots)} plots have missing or invalid images!")
                for i, p in enumerate(generated_plots):
                    has_image = 'image' in p and p['image'] is not None
                    image_len = len(p.get('image', '')) if has_image else 0
                    print(f"  Plot {i+1}: HP={p.get('hp_label')}, Flow={p.get('flow_config')}, has_image={has_image}, image_len={image_len}")
            
            # If no valid plots were generated, provide detailed debug information
            if len(valid_plots) == 0:
                error_msg = 'No valid plots generated after filtering and processing.'
                error_msg += f'\n\nSummary:'
                error_msg += f'\n- Total entries processed: {len(entries)}'
                error_msg += f'\n- Plots attempted: {len(generated_plots)}'
                error_msg += f'\n- Valid plots (with images): {len(valid_plots)}'
                try:
                    error_msg += f'\n- HP/Flow combinations available for loop: {len(hp_flow_combos) if "hp_flow_combos" in locals() else 0}'
                    if "hp_flow_combos" in locals() and len(hp_flow_combos) == 0:
                        error_msg += f'\n  ⚠️ Loop did not execute because hp_flow_combos is empty!'
                    elif "hp_flow_combos" in locals() and len(hp_flow_combos) > 0:
                        error_msg += f'\n  ✓ Loop should have executed with {len(hp_flow_combos)} combination(s)'
                except:
                    pass
                
                # Safely access variables that may or may not be in scope
                try:
                    error_msg += f'\n- Entries with a unit key: {len(working_entries)}'
                except NameError:
                    error_msg += f'\n- Entries with a unit key: N/A'
                
                try:
                    error_msg += f'\n- Data points in DataFrame: {len(df)}'
                except NameError:
                    error_msg += f'\n- Data points in DataFrame: 0'
                
                try:
                    error_msg += f'\n- Available HP/Flow combinations: {len(hp_flow_combos)}'
                except NameError:
                    error_msg += f'\n- Available HP/Flow combinations: 0'
                
                if selected_combinations:
                    error_msg += f'\n- Selected combinations: {len(selected_combinations)}'
                    error_msg += f'\n  Selected: {selected_combinations}'
                
                try:
                    # Show all combinations found by groupby (before filtering by selected_combinations)
                    if 'df' in locals() and len(df) > 0:
                        all_combos = df.groupby(['hp_label', 'flow_config']).size().reset_index()
                        all_combos.columns = ['hp_label', 'flow_config', 'count']
                        error_msg += f'\n\nAll HP/Flow combinations found by groupby (before filtering):'
                        for _, combo in all_combos.iterrows():
                            error_msg += f'\n  - HP{combo["hp_label"]} ({combo["flow_config"]}): {combo["count"]} data points'
                    
                    if len(hp_flow_combos) > 0:
                        error_msg += f'\n\nAvailable combinations in DataFrame (after filtering by selected_combinations):'
                        for _, combo in hp_flow_combos.iterrows():
                            combo_df = df[
                                (df['hp_label'].astype(str).str.strip() == str(combo['hp_label']).strip()) & 
                                (df['flow_config'].astype(str).str.strip().str.lower() == str(combo['flow_config']).strip().lower())
                            ]
                            error_msg += f'\n  - HP{combo["hp_label"]} ({combo["flow_config"]}): {len(combo_df)} data points'
                            if len(combo_df) == 0:
                                # Show why filtering failed
                                hp_matches = df[df['hp_label'].astype(str).str.strip() == str(combo['hp_label']).strip()]
                                flow_matches = df[df['flow_config'].astype(str).str.strip().str.lower() == str(combo['flow_config']).strip().lower()]
                                error_msg += f'\n    ⚠️ Filtering issue: Found {len(hp_matches)} rows with matching HP, {len(flow_matches)} rows with matching Flow'
                                if len(hp_matches) > 0:
                                    error_msg += f'\n    Sample HP values: {hp_matches["hp_label"].head(5).tolist()}'
                                if len(flow_matches) > 0:
                                    error_msg += f'\n    Sample Flow values: {flow_matches["flow_config"].head(5).tolist()}'
                    else:
                        error_msg += f'\n\n⚠️ No combinations found after filtering by selected_combinations!'
                        if 'df' in locals() and len(df) > 0:
                            all_combos = df.groupby(['hp_label', 'flow_config']).size().reset_index()
                            all_combos.columns = ['hp_label', 'flow_config', 'count']
                            error_msg += f'\n  But groupby found {len(all_combos)} combinations:'
                            for _, combo in all_combos.head(10).iterrows():
                                normalized = (str(combo['hp_label']).strip(), str(combo['flow_config']).strip().lower())
                                error_msg += f'\n    - HP{combo["hp_label"]} ({combo["flow_config"]}): {combo["count"]} data points (normalized: {normalized})'
                except NameError:
                    pass
                except Exception as e:
                    error_msg += f'\n\nError generating combination info: {e}'
                
                try:
                    if len(df) > 0:
                        error_msg += f'\n\nDataFrame sample (first 5 rows):'
                        for idx, row in df.head(5).iterrows():
                            error_msg += f'\n  Row {idx}: HP={repr(row.get("hp_label"))}, Flow={repr(row.get("flow_config"))}, Y={repr(row.get("y_value"))}'
                except NameError:
                    pass
                
                # Include the debug_info if available
                try:
                    if debug_info and len(debug_info) > 0:
                        error_msg += f'\n\nDetailed Entry Debug Info:\n' + '\n'.join(debug_info[:50])
                except NameError:
                    pass
                
                return jsonify({
                    'success': False,
                    'message': error_msg,
                    'debug_info': debug_info if 'debug_info' in locals() else [],
                    'plots': []
                })
            
            # Final safety check: Never return success: True with empty plots
            if len(valid_plots) == 0:
                error_msg = 'No valid plots generated (all plots had missing or invalid images).'
                error_msg += f'\n\nSummary:'
                error_msg += f'\n- Total entries processed: {len(entries)}'
                error_msg += f'\n- Plots attempted: {len(generated_plots)}'
                error_msg += f'\n- Valid plots: {len(valid_plots)}'
                return jsonify({
                    'success': False,
                    'message': error_msg,
                    'debug_info': debug_info if 'debug_info' in locals() else [],
                    'plots': []
                })
            
            # Store images in cache and return image_url instead of huge base64 (avoids browser/JSON size limits)
            global SCATTER_PLOT_CACHE
            if len(SCATTER_PLOT_CACHE) > SCATTER_PLOT_CACHE_MAX:
                SCATTER_PLOT_CACHE.clear()
            response_plots = []
            for p in valid_plots:
                img_data = p.get('image') or ''
                if not img_data:
                    continue
                # Extract base64 payload (after "data:image/png;base64,")
                b64_payload = img_data.split(',', 1)[1] if ',' in img_data else img_data
                image_id = str(uuid.uuid4())
                SCATTER_PLOT_CACHE[image_id] = b64_payload
                out = {k: v for k, v in p.items() if k != 'image'}
                out['image_url'] = url_for('scatter_plot_image', image_id=image_id)
                response_plots.append(out)
            
            scatter_warnings = []
            if unresolved_warning_files:
                scatter_warnings.append(
                    'These files have no unit profile and no HP ID, so each file is plotted as its own series. '
                    'Assign a unit profile on Cycle Extract or review so series stay comparable: '
                    + ', '.join(unresolved_warning_files)
                )
            return jsonify({
                'success': True,
                'plots': response_plots,
                'warnings': scatter_warnings,
            })
            
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Error generating scatter plots: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/scatter_plot_image/<image_id>')
def scatter_plot_image(image_id):
    """Serve a cached scatter plot image by ID so the browser can load it by URL (avoids huge JSON and data URL length limits)."""
    global SCATTER_PLOT_CACHE
    if not image_id or image_id not in SCATTER_PLOT_CACHE:
        abort(404)
    b64_payload = SCATTER_PLOT_CACHE[image_id]
    try:
        raw = base64.b64decode(b64_payload)
    except Exception:
        abort(404)
    return send_file(BytesIO(raw), mimetype='image/png', max_age=300)

def generate_stripplot(data, hp_label, flow_config, y_parameter, mean_db_k=0.6, mean_wb_k=0.4, mean_tsup_k=0.5, mean_q_pct=5.0, mean_flow_pct=1.0, unit_label=None, profile_id=None):
    """Generate a stripplot for given data. Draws permissible deviation bands only for test conditions present in the data.
    mean_q_pct and mean_flow_pct are used for Q (heating capacity) and flow (mass/volume) y_parameters respectively."""
    try:
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure (moderate size to keep base64 payload within response limits)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        plot_df = pd.DataFrame(data)
        # One file name from the group: lets unit lookups resolve profiles by filename pattern
        # for units that carry no HP id (e.g. HPT_RRT*).
        band_file_name = None
        if 'file_name' in plot_df.columns:
            file_names = plot_df['file_name'].dropna()
            if not file_names.empty:
                band_file_name = str(file_names.iloc[0])
        # Ensure lab_id is usable (stripplot hue can fail on None/NaN)
        if 'lab_id' in plot_df.columns:
            plot_df = plot_df.copy()
            plot_df['lab_id'] = plot_df['lab_id'].apply(lambda x: x if x is not None and pd.notna(x) and str(x).strip() != '' else 'Unknown')
        
        # Base color palette matching the notebook
        base_lab_palette = {
            1: '#1f77b4', '1': '#1f77b4', '1a': '#1f77b4', '1b': 'navy',
            2: '#ff7f0e', '2': '#ff7f0e',
            3: '#2ca02c', '3': '#2ca02c',
            4: '#d62728', '4': '#d62728',
            5: '#9467bd', '5': '#9467bd',
            6: '#8c564b', '6': '#8c564b',
            7: '#17becf', '7': '#17becf',
            'Unknown': '#7f7f7f'
        }
        
        # Get all unique lab_ids from the data
        unique_lab_ids = plot_df['lab_id'].unique().tolist()
        
        # Start with base palette
        lab_palette = base_lab_palette.copy()
        
        # Generate colors for any missing lab_ids
        import matplotlib.colors as mcolors
        import colorsys
        
        # Extended color palette for additional labs
        extended_colors = [
            '#e377c2', '#7f7f7f', '#bcbd22', '#ffbb78', '#2ca02c',
            '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
            '#8c564b', '#c49c94', '#17becf', '#9edae5', '#aec7e8',
            '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728'
        ]
        
        # Find missing lab_ids
        missing_lab_ids = []
        for lab_id in unique_lab_ids:
            # Check if lab_id (as string or int) is in palette
            if lab_id not in lab_palette and str(lab_id) not in lab_palette:
                missing_lab_ids.append(lab_id)
        
        # Add colors for missing lab_ids
        color_index = 0
        for lab_id in missing_lab_ids:
            if color_index < len(extended_colors):
                color = extended_colors[color_index]
                # Add both string and int versions if applicable
                lab_palette[lab_id] = color
                if isinstance(lab_id, str):
                    try:
                        lab_palette[int(lab_id)] = color
                    except (ValueError, TypeError):
                        pass
                elif isinstance(lab_id, (int, float)):
                    lab_palette[str(lab_id)] = color
                color_index += 1
            else:
                # Generate a color using HSV color space
                hue = (color_index * 0.618) % 1.0  # Golden ratio for better distribution
                saturation = 0.7
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                color = mcolors.rgb2hex(rgb)
                lab_palette[lab_id] = color
                if isinstance(lab_id, str):
                    try:
                        lab_palette[int(lab_id)] = color
                    except (ValueError, TypeError):
                        pass
                elif isinstance(lab_id, (int, float)):
                    lab_palette[str(lab_id)] = color
                color_index += 1
        
        # Convert keys to strings for display (to avoid sorting issues with mixed types)
        palette_keys_str = [str(k) for k in lab_palette.keys()]
        print(f"DEBUG: Lab palette keys: {sorted(set(palette_keys_str))}")
        print(f"DEBUG: Unique lab_ids in data: {unique_lab_ids}")
        
        # Create stripplot
        sns.stripplot(data=plot_df, x='test_condition', y='y_value', 
                     hue='lab_id', jitter=True, dodge=True, 
                     palette=lab_palette, ax=ax)
        
        # Per-condition bands: only for test conditions present in this plot's data.
        # Pcorrwbuh/Puncorrwbuh get no bands (power, not temperature or Q). Qcorrwbuh/Quncorrwbuh get Q bands.
        unique_conditions = set(c for c in plot_df['test_condition'].dropna().unique().tolist() if c in DEVIATION_SETPOINTS)
        from matplotlib.patches import Rectangle
        y_param_lower = y_parameter.lower()
        # Power/capacity BUH params: do not treat as WB temperature
        is_p_or_q_buh = ('buh' in y_param_lower and ('pcorr' in y_param_lower or 'puncorr' in y_param_lower or 'qcorr' in y_param_lower or 'quncorr' in y_param_lower))
        is_wb_temperature = ('t_wb' in y_param_lower or ('wb' in y_param_lower and ('temperature' in y_param_lower or 'wet' in y_param_lower or 't_wb' in y_param_lower))) and not is_p_or_q_buh
        is_q_param = (
            ('heating' in y_param_lower and 'capacity' in y_param_lower) or
            ('capacity' in y_param_lower and 'pcorr' not in y_param_lower and 'puncorr' not in y_param_lower) or
            'qcorrwbuh' in y_param_lower or 'quncorrwbuh' in y_param_lower
        )
        band_ylo_yhi = []  # collect band extents so we can include them in y-axis range
        if unique_conditions:
            # Match x-axis order (same as stripplot categories)
            cat_order = [str(l.get_text()) for l in ax.get_xticklabels()]
            for idx, cond in enumerate(cat_order):
                if cond not in unique_conditions or cond not in DEVIATION_SETPOINTS:
                    continue
                ylo, yhi = None, None
                setpoint_val = None  # for dashed line and set-value text
                if 't_db' in y_param_lower or ('db' in y_param_lower and 'wb' not in y_param_lower):
                    setpoint = DEVIATION_SETPOINTS.get(cond)
                    half = mean_db_k
                    if setpoint is not None and half is not None:
                        ylo, yhi = setpoint - half, setpoint + half
                        setpoint_val = setpoint
                elif is_wb_temperature:
                    setpoint = DEVIATION_SETPOINTS.get(cond) - 1 if DEVIATION_SETPOINTS.get(cond) is not None else None
                    half = mean_wb_k
                    if setpoint is not None and half is not None:
                        ylo, yhi = setpoint - half, setpoint + half
                        setpoint_val = setpoint
                elif ('supply' in y_param_lower or 'tsup' in y_param_lower or 't_sup' in y_param_lower or
                      'ts buh' in y_param_lower or 'ts_buh' in y_param_lower):
                    # Tsup bands: supply temp and Ts Buh (buffer supply temp)
                    setpoint = TSUP_SETPOINTS.get(cond, TSUP_SETPOINTS['E'])
                    half = mean_tsup_k
                    if setpoint is not None and half is not None:
                        ylo, yhi = setpoint - half, setpoint + half
                        setpoint_val = setpoint
                elif is_q_param:
                    # Q (heating capacity): band = Qset ± mean_q_pct%
                    db_setpoint = DEVIATION_SETPOINTS.get(cond)
                    hp_for_design = None
                    if 'hp_id' in plot_df.columns:
                        hp_vals = plot_df['hp_id'].dropna().astype(str).str.strip()
                        hp_vals = hp_vals[hp_vals != '']
                        if not hp_vals.empty:
                            hp_for_design = hp_vals.iloc[0]
                    if hp_for_design is None:
                        m_hp = re.match(r'^HP(\d+)$', str(hp_label or ''), re.I)
                        hp_for_design = m_hp.group(1) if m_hp else None
                    pid = profile_id
                    if not pid and hp_label in unit_config.load_all_profiles():
                        pid = hp_label
                    pdesign = get_pdesign_for_hp(
                        hp_for_design,
                        file_name=band_file_name,
                        profile_id=pid,
                    )
                    if db_setpoint is not None and pdesign is not None:
                        _, qset = compute_ua_and_qset(pdesign, db_setpoint)
                        if qset is not None and mean_q_pct >= 0:
                            fac = mean_q_pct / 100.0
                            ylo = qset * (1 - fac)
                            yhi = qset * (1 + fac)
                            setpoint_val = qset
                elif ('mass' in y_param_lower and 'flow' in y_param_lower) or ('volume' in y_param_lower and 'flow' in y_param_lower) or y_param_lower.strip() in ('mass flow', 'volume flow', 'avg_mass_flow', 'avg_volume_flow'):
                    # Flow bands only for fixed-flow tests; match mass/volume to set value type.
                    is_mass_flow_param = 'mass' in y_param_lower and 'flow' in y_param_lower
                    is_volume_flow_param = 'volume' in y_param_lower and 'flow' in y_param_lower
                    scatter_entry = {
                        'file_name': band_file_name,
                        'profile_id': profile_id or (hp_label if hp_label in unit_config.load_all_profiles() else None),
                        'flow_config': flow_config,
                        'hp_id': hp_label,
                    }
                    if (is_mass_flow_param or is_volume_flow_param) and not _entry_is_variable_flow(scatter_entry):
                        flow_type, flow_set_value = get_flow_set_for_hp(hp_label, file_name=band_file_name, profile_id=scatter_entry.get('profile_id'))
                        if flow_set_value is not None and mean_flow_pct >= 0:
                            if (is_mass_flow_param and (flow_type or '').lower() == 'mass') or (is_volume_flow_param and (flow_type or '').lower() == 'volume'):
                                fac = mean_flow_pct / 100.0
                                ylo = flow_set_value * (1 - fac)
                                yhi = flow_set_value * (1 + fac)
                                setpoint_val = flow_set_value
                if ylo is not None and yhi is not None:
                    band_ylo_yhi.append((ylo, yhi))
                    rect = Rectangle((idx - 0.45, ylo), 0.9, yhi - ylo, linewidth=0, facecolor='gray', alpha=0.2, zorder=-1)
                    ax.add_patch(rect)
                    # Full-width dashed setpoint line (notebook-style); no setpoint text on figure
                    if setpoint_val is not None:
                        ax.axhline(setpoint_val, color='black', linestyle='--', linewidth=1, zorder=0, alpha=0.7)
        
        # Always set y-axis to data (+ bands if any) with tight padding so variations are visible
        y_vals = plot_df['y_value'].dropna()
        data_min = float(y_vals.min()) if len(y_vals) > 0 else 0.0
        data_max = float(y_vals.max()) if len(y_vals) > 0 else 1.0
        combined_min = data_min
        combined_max = data_max
        if band_ylo_yhi:
            band_min = min(b[0] for b in band_ylo_yhi)
            band_max = max(b[1] for b in band_ylo_yhi)
            combined_min = min(combined_min, band_min)
            combined_max = max(combined_max, band_max)
        total_range = combined_max - combined_min
        # 8% padding; for very small ranges use at least 2% of larger magnitude or 0.01
        if total_range < 1e-12:
            total_range = abs(combined_max) if abs(combined_max) >= 1e-12 else 1.0
        pad = max(0.08 * total_range, 0.02 * max(abs(combined_min), abs(combined_max), 1e-6), 0.005)
        ax.set_ylim(combined_min - pad, combined_max + pad)
        
        # Add gridlines
        ax.grid(True, alpha=0.3)
        
        # Format title
        y_param_display = y_parameter.replace('_', ' ').title()
        display_unit = unit_label or hp_label
        title = f'{y_param_display} - {unit_config.format_unit_flow_label(display_unit, flow_config)}'
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlabel('Test Condition', fontsize=18)
        ax.set_ylabel(y_param_display, fontsize=18)
        ax.tick_params(labelsize=16)
        
        # Legend
        ax.legend(title='Lab', title_fontsize='14', fontsize='12', loc='best')
        
        # Convert to base64; use lower DPI/size to keep payload small and avoid truncation in JSON response
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=72, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{img_base64}'
        
    except Exception as e:
        print(f"Error in generate_stripplot: {e}")
        import traceback
        traceback.print_exc()
        plt.close()
        raise

@app.route('/mark_for_scatter', methods=['POST'])
def mark_for_scatter():
    """Mark selected entries for scatter plots"""
    try:
        data = request.get_json()
        entry_ids = data.get('entry_ids', [])
        action = data.get('action', 'mark')  # 'mark' or 'unmark'
        
        if not entry_ids:
            return jsonify({'success': False, 'message': 'No entries selected'})
        
        # Initialize column if needed
        init_scatter_dataset_column()
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Check if column exists
            cursor.execute("PRAGMA table_info(results)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'scatter_dataset' not in columns:
                cursor.execute("ALTER TABLE results ADD COLUMN scatter_dataset TEXT")
                conn.commit()
            
            # Update selected entries
            placeholders = ','.join(['?'] * len(entry_ids))
            if action == 'mark':
                cursor.execute(f'''
                    UPDATE results 
                    SET scatter_dataset = 'YES'
                    WHERE rowid IN ({placeholders})
                ''', entry_ids)
            else:  # unmark
                cursor.execute(f'''
                    UPDATE results 
                    SET scatter_dataset = NULL
                    WHERE rowid IN ({placeholders})
                ''', entry_ids)
            
            conn.commit()
            updated_count = cursor.rowcount
            
            return jsonify({
                'success': True,
                'message': f'Successfully {action}ed {updated_count} entries for scatter plots',
                'updated_count': updated_count
            })
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Error marking entries for scatter: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/reset_scatter_selection', methods=['POST'])
def reset_scatter_selection():
    """Reset scatter_dataset column to match COP_Dataset"""
    try:
        init_scatter_dataset_column()
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Find COP_Dataset column
            cursor.execute("PRAGMA table_info(results)")
            columns = [col[1] for col in cursor.fetchall()]
            
            cop_column = None
            for col in columns:
                col_lower = col.lower()
                if (col_lower == 'cop_dataset' or col_lower == 'cop dataset' or 
                    'cop' in col_lower and 'dataset' in col_lower):
                    cop_column = col
                    break
            
            if cop_column:
                # Reset scatter_dataset from COP_Dataset
                cursor.execute(f"""
                    UPDATE results 
                    SET scatter_dataset = CASE 
                        WHEN {cop_column} IN ('YES', 'Y', '1', 'TRUE', 'True', 'true') THEN 'YES'
                        ELSE NULL
                    END
                """)
                conn.commit()
                updated_count = cursor.rowcount
                
                return jsonify({
                    'success': True,
                    'message': f'Reset scatter selection for {updated_count} entries based on COP_Dataset',
                    'updated_count': updated_count
                })
            else:
                # If no COP_Dataset, clear all
                cursor.execute("UPDATE results SET scatter_dataset = NULL")
                conn.commit()
                updated_count = cursor.rowcount
                
                return jsonify({
                    'success': True,
                    'message': f'Cleared scatter selection for {updated_count} entries (no COP_Dataset column found)',
                    'updated_count': updated_count
                })
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Error resetting scatter selection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/toggle_column_value', methods=['POST'])
def toggle_column_value():
    """Toggle a column value between YES and None for a specific entry"""
    try:
        data = request.get_json()
        rowid = data.get('rowid')
        column_name = data.get('column_name')  # 'COP_Dataset' or 'scatter_dataset'
        
        if not rowid or not column_name:
            return jsonify({'success': False, 'message': 'Missing rowid or column_name'})
        
        # Initialize column if needed (for scatter_dataset)
        if column_name == 'scatter_dataset':
            init_scatter_dataset_column()
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Check if column exists
            cursor.execute("PRAGMA table_info(results)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Find the actual column name (handle case variations)
            actual_column = None
            for col in columns:
                col_lower = col.lower()
                if column_name.lower() == 'cop_dataset':
                    if (col_lower == 'cop_dataset' or col_lower == 'cop dataset' or 
                        'cop' in col_lower and 'dataset' in col_lower):
                        actual_column = col
                        break
                elif column_name.lower() == 'scatter_dataset':
                    if col_lower == 'scatter_dataset' or col_lower == 'scatter dataset':
                        actual_column = col
                        break
            
            if not actual_column:
                return jsonify({'success': False, 'message': f'Column {column_name} not found'})
            
            # Get current value
            cursor.execute(f'SELECT {actual_column} FROM results WHERE rowid = ?', (rowid,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'success': False, 'message': 'Entry not found'})
            
            current_value = result[0]
            is_yes = current_value and str(current_value).upper() in ['YES', 'Y', '1', 'TRUE']
            
            # Toggle value
            new_value = None if is_yes else 'YES'
            cursor.execute(f'UPDATE results SET {actual_column} = ? WHERE rowid = ?', (new_value, rowid))
            conn.commit()
            
            return jsonify({
                'success': True,
                'new_value': new_value,
                'is_yes': not is_yes
            })
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Error toggling column value: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/toggle_column_value_bulk', methods=['POST'])
def toggle_column_value_bulk():
    """Add or remove entries from a dataset column (bulk operation)"""
    try:
        data = request.get_json()
        entry_ids = data.get('entry_ids', [])
        column_name = data.get('column_name')  # 'COP_Dataset' or 'scatter_dataset'
        action = data.get('action')  # 'add' or 'remove'
        
        if not entry_ids or not column_name or not action:
            return jsonify({'success': False, 'message': 'Missing required parameters'})
        
        # Initialize column if needed (for scatter_dataset)
        if column_name == 'scatter_dataset':
            init_scatter_dataset_column()
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Check if column exists
            cursor.execute("PRAGMA table_info(results)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Find the actual column name (handle case variations)
            actual_column = None
            for col in columns:
                col_lower = col.lower()
                if column_name.lower() == 'cop_dataset':
                    if (col_lower == 'cop_dataset' or col_lower == 'cop dataset' or 
                        'cop' in col_lower and 'dataset' in col_lower):
                        actual_column = col
                        break
                elif column_name.lower() == 'scatter_dataset':
                    if col_lower == 'scatter_dataset' or col_lower == 'scatter dataset':
                        actual_column = col
                        break
            
            if not actual_column:
                return jsonify({'success': False, 'message': f'Column {column_name} not found'})
            
            # Update entries
            placeholders = ','.join(['?'] * len(entry_ids))
            if action == 'add':
                cursor.execute(f'''
                    UPDATE results 
                    SET {actual_column} = 'YES'
                    WHERE rowid IN ({placeholders})
                ''', entry_ids)
                action_text = 'added to'
            else:  # remove
                cursor.execute(f'''
                    UPDATE results 
                    SET {actual_column} = NULL
                    WHERE rowid IN ({placeholders})
                ''', entry_ids)
                action_text = 'removed from'
            
            conn.commit()
            updated_count = cursor.rowcount
            
            column_display = 'COP Dataset' if column_name == 'COP_Dataset' else 'Scatter Dataset'
            
            return jsonify({
                'success': True,
                'message': f'Successfully {action_text} {updated_count} entries {action_text} {column_display}',
                'updated_count': updated_count
            })
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Error toggling column value bulk: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

def get_column_metadata_file():
    """Get the path to the column metadata JSON file"""
    # Use _APP_DIR if available, otherwise use __file__
    try:
        app_dir = _APP_DIR
    except NameError:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, 'column_metadata.json')

def load_column_metadata():
    """Load column metadata from JSON file"""
    metadata_file = get_column_metadata_file()
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading column metadata: {e}")
            return {}
    return {}

def save_column_metadata(column_name, category, extra_data=None):
    """Save column metadata to JSON file"""
    metadata_file = get_column_metadata_file()
    metadata = load_column_metadata()
    
    metadata[column_name] = {
        'category': category,
        **(extra_data or {})
    }
    
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving column metadata: {e}")

def load_column_visibility():
    """Load column visibility settings from JSON file"""
    # Use _APP_DIR if available, otherwise use __file__
    try:
        app_dir = _APP_DIR
    except NameError:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    visibility_file = os.path.join(app_dir, 'column_visibility.json')
    if os.path.exists(visibility_file):
        try:
            with open(visibility_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading column visibility: {e}")
            return {}
    return {}

def save_column_visibility(visibility):
    """Save column visibility settings to JSON file"""
    # Use _APP_DIR if available, otherwise use __file__
    try:
        app_dir = _APP_DIR
    except NameError:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    visibility_file = os.path.join(app_dir, 'column_visibility.json')
    try:
        with open(visibility_file, 'w') as f:
            json.dump(visibility, f, indent=2)
    except Exception as e:
        print(f"Error saving column visibility: {e}")

@app.route('/edit_column_metadata', methods=['POST'])
def edit_column_metadata():
    """Edit column metadata (formula, analysis type, etc.)"""
    try:
        data = request.get_json()
        column_name = data.get('column_name')
        category = data.get('category')
        
        if not column_name or not category:
            return jsonify({'success': False, 'message': 'Column name and category are required'})
        
        # Verify column exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(results)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        conn.close()
        
        if column_name not in existing_columns:
            return jsonify({'success': False, 'message': f'Column "{column_name}" does not exist'})
        
        # Prepare metadata based on category
        extra_data = {}
        if category == 'derived':
            formula = data.get('formula', '').strip()
            if not formula:
                return jsonify({'success': False, 'message': 'Formula is required for derived columns'})
            extra_data['formula'] = formula
        elif category == 'timeseries':
            analysis_type = data.get('analysis_type')
            parameters = data.get('parameters', {})
            if not analysis_type:
                return jsonify({'success': False, 'message': 'Analysis type is required for time series columns'})
            extra_data['analysis_type'] = analysis_type
            extra_data['parameters'] = parameters
        
        # Save metadata
        save_column_metadata(column_name, category, extra_data)
        
        # If it's a derived column, optionally recalculate values
        if category == 'derived' and data.get('recalculate', False):
            # Recalculate derived column values
            formula = extra_data['formula']
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT rowid, * FROM results")
            entries = cursor.fetchall()
            
            updated_count = 0
            for entry in entries:
                entry_dict = dict(entry)
                eval_context = {}
                for key, value in entry_dict.items():
                    if key != 'rowid' and key != 'time_series_data':
                        try:
                            if value is not None:
                                eval_context[key] = float(value)
                            else:
                                eval_context[key] = None
                        except (ValueError, TypeError):
                            eval_context[key] = value
                
                import math
                safe_builtins = {
                    'abs': abs, 'min': min, 'max': max, 'round': round, 'pow': pow,
                    'sum': sum, 'len': len, 'int': int, 'float': float, 'str': str,
                    'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log, 'log10': math.log10,
                    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
                    'ceil': math.ceil, 'floor': math.floor,
                }
                eval_context.update(safe_builtins)
                
                try:
                    result = eval(formula, {"__builtins__": {}}, eval_context)
                    if result is not None:
                        cursor.execute(f"UPDATE results SET {column_name} = ? WHERE rowid = ?", 
                                      (result, entry_dict['rowid']))
                        updated_count += 1
                except Exception:
                    pass
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': f'Column metadata updated and values recalculated for {updated_count} entries'
            })
        
        return jsonify({
            'success': True,
            'message': f'Column metadata updated successfully'
        })
        
    except Exception as e:
        print(f"Error editing column metadata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

_SCALAR_DEVIATION_KEYS = (
    'mean_db_k', 'mean_wb_k', 'mean_tsup_k', 'mean_tmean_k',
    'db_pct_red', 'db_pct_yellow', 'wb_pct_red', 'wb_pct_yellow',
    'db_violations_red', 'wb_violations_red',
    'tsup_pct_red', 'tsup_pct_yellow', 'tsup_violations_red',
    'dtreturn_pct_red', 'dtreturn_pct_yellow', 'dtreturn_violations_red',
    'mean_q_pct_red', 'mean_q_pct_yellow',
    'mean_flow_pct', 'flow_instantaneous_pct',
    'flow_violations_red', 'flow_pct_red', 'flow_pct_yellow',
)

@app.route('/save_permissible_deviations', methods=['POST'])
def save_permissible_deviations_route():
    """Save permissible deviations configuration (transient bands, mean limits, violation thresholds)."""
    global PERMISSIBLE_DEVIATIONS, DB_BAND, WB_BAND, TSUP_BAND, DEVIATION_BANDS
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        # Start from current config so we don't drop keys not in the form
        updated_config = dict(load_permissible_deviations())
        for key, value in data.items():
            if key in ('DB', 'WB', 'Tsup') and isinstance(value, dict) and 'value' in value:
                try:
                    numeric_value = float(value['value'])
                    updated_config[key] = {
                        'value': numeric_value,
                        'unit': value.get('unit', 'K'),
                        'description': value.get('description', ''),
                        'standard_reference': value.get('standard_reference', '')
                    }
                except (ValueError, TypeError):
                    return jsonify({'success': False, 'message': f'Invalid value for {key}'})
            elif key == 'dTreturn' and isinstance(value, dict) and 'lower' in value and 'upper' in value:
                try:
                    lower_v = float(value['lower'])
                    upper_v = float(value['upper'])
                    if lower_v >= upper_v:
                        return jsonify({'success': False, 'message': 'Invalid dTreturn bounds: lower must be < upper'})
                    updated_config[key] = {
                        'lower': lower_v,
                        'upper': upper_v,
                        'unit': value.get('unit', 'K'),
                        'description': value.get('description', ''),
                        'standard_reference': value.get('standard_reference', '')
                    }
                except (ValueError, TypeError):
                    return jsonify({'success': False, 'message': 'Invalid value for dTreturn'})
            elif key in _SCALAR_DEVIATION_KEYS:
                try:
                    updated_config[key] = float(value)
                except (ValueError, TypeError):
                    return jsonify({'success': False, 'message': f'Invalid value for {key}'})
        
        # Save configuration
        if save_permissible_deviations(updated_config):
            # Reload global variables
            PERMISSIBLE_DEVIATIONS = load_permissible_deviations()
            DB_BAND = PERMISSIBLE_DEVIATIONS.get('DB', {}).get('value', 1.0)
            WB_BAND = PERMISSIBLE_DEVIATIONS.get('WB', {}).get('value', 1.0)
            TSUP_BAND = PERMISSIBLE_DEVIATIONS.get('Tsup', {}).get('value', 0.5)
            DEVIATION_BANDS = calculate_deviation_bands(DEVIATION_SETPOINTS, DB_BAND, WB_BAND)
            
            return jsonify({
                'success': True,
                'message': 'Permissible deviations configuration saved successfully'
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to save configuration'})
    except Exception as e:
        print(f"Error saving permissible deviations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/hp_design', methods=['GET'])
def api_hp_design_list():
    """List all HP design parameters (hp_id, pdesign_kw) from database."""
    try:
        init_hp_design_table()
        items = get_all_hp_designs()
        return jsonify({'success': True, 'items': items})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/hp_design', methods=['POST'])
def api_hp_design_save():
    """Legacy: write the hp_design table directly. Body: { hp_id, pdesign_kw [, flow_type, flow_set_value ] }.

    Kept for older clients only. The Design parameters panel now writes unit profiles
    via /api/unit_design, because a profile value always overrides this table.
    """
    try:
        data = request.get_json()
        # Coerce to str: front-end may send hp_id as number (e.g. from data-hp-id)
        hp_id = str(data.get('hp_id') or '').strip()
        try:
            pdesign_kw = float(data.get('pdesign_kw'))
        except (TypeError, ValueError):
            return jsonify({'success': False, 'message': 'Valid pdesign_kw (number) required'})
        if not hp_id:
            return jsonify({'success': False, 'message': 'hp_id required'})
        if pdesign_kw < 0:
            return jsonify({'success': False, 'message': 'pdesign_kw must be non-negative'})
        flow_type = str(data.get('flow_type') or '').strip().lower()
        if flow_type not in ('', 'mass', 'volume'):
            flow_type = ''
        flow_set_value = None
        if flow_type:
            try:
                flow_set_value = float(data.get('flow_set_value'))
                if flow_set_value <= 0:
                    return jsonify({'success': False, 'message': 'flow_set_value must be positive when flow_type is set'})
            except (TypeError, ValueError):
                return jsonify({'success': False, 'message': 'Valid flow_set_value required when flow_type is mass or volume'})
        init_hp_design_table()
        conn = get_db_connection()
        try:
            conn.execute(
                """INSERT INTO hp_design (hp_id, pdesign_kw, flow_type, flow_set_value)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(hp_id) DO UPDATE SET
                     pdesign_kw = excluded.pdesign_kw,
                     flow_type = excluded.flow_type,
                     flow_set_value = excluded.flow_set_value""",
                (hp_id, pdesign_kw, flow_type or None, flow_set_value)
            )
            conn.commit()
            return jsonify({'success': True, 'message': f'HP{hp_id} design saved.'})
        finally:
            conn.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

def _parse_optional_float(value, field, minimum=None, allow_zero=True):
    """Return (value_or_None, error_message). Empty input means "not set"."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f'{field} must be a number'
    if minimum is not None and number < minimum:
        return None, f'{field} must be at least {minimum}'
    if not allow_zero and number == 0:
        return None, f'{field} must be greater than 0'
    return number, None


@app.route('/api/unit_design', methods=['POST'])
def api_unit_design_save():
    """Write Pdesign / flow mode / fixed-flow set into the unit profile that the
    calculations actually read. Creates a stub profile when the unit has none, which
    is why an alias or filename pattern is required in that case: without one the new
    profile could never be resolved from an entry.
    """
    try:
        data = request.get_json() or {}
        profile_id = str(data.get('profile_id') or '').strip()
        hp_id = str(data.get('hp_id') or '').strip()

        pdesign_kw, err = _parse_optional_float(data.get('pdesign_kw'), 'Pdesign', minimum=0)
        if err:
            return jsonify({'success': False, 'message': err})

        flow_mode = str(data.get('flow_mode') or 'per_test').strip().lower()
        if flow_mode not in FLOW_MODES:
            return jsonify({'success': False, 'message': f'flow_mode must be one of {", ".join(FLOW_MODES)}'})

        flow_type = str(data.get('flow_type') or '').strip().lower()
        if flow_type not in ('', 'mass', 'volume'):
            return jsonify({'success': False, 'message': 'flow_type must be mass, volume or empty'})
        flow_set_value, err = _parse_optional_float(
            data.get('flow_set_value'), 'Fixed-flow set value', minimum=0, allow_zero=False
        )
        if err:
            return jsonify({'success': False, 'message': err})
        if flow_mode == 'variable':
            # Variable flow has no flow setpoint to hold.
            flow_type, flow_set_value = '', None
        if flow_set_value is not None and not flow_type:
            return jsonify({'success': False, 'message': 'Choose mass or volume for the fixed-flow set value'})

        existing = unit_config.get_profile(profile_id) if profile_id else None
        if existing is None:
            if not profile_id:
                return jsonify({'success': False, 'message': 'profile_id required to create a unit profile'})
            aliases = [a.strip() for a in str(data.get('hp_id_aliases') or '').split(',') if a.strip()]
            patterns = [p.strip() for p in str(data.get('filename_patterns') or '').split(',') if p.strip()]
            if hp_id and hp_id not in aliases:
                aliases.append(hp_id)
            if not aliases and not patterns:
                return jsonify({'success': False, 'message':
                                'Give an HP id alias or a filename pattern, otherwise no entry can resolve to this unit'})
            profile = {
                'profile_id': profile_id,
                'display_name': str(data.get('display_name') or '').strip() or profile_id,
                'unit_type': str(data.get('unit_type') or 'air_water').strip(),
                'buh_power_cap_kw': None,
                'wb_rule': 'db_minus_1k',
                'hp_id_aliases': aliases,
                'filename_patterns': patterns,
            }
            destination = str(data.get('destination') or 'local').strip() or 'local'
        else:
            profile = {k: v for k, v in existing.items() if not str(k).startswith('_')}
            destination = str(data.get('destination') or existing.get('_source') or 'local').strip()
            if str(data.get('display_name') or '').strip():
                profile['display_name'] = str(data['display_name']).strip()

        profile['pdesign_kw'] = pdesign_kw
        profile['flow_mode'] = flow_mode
        profile['flow_type'] = flow_type or None
        profile['flow_set_value'] = flow_set_value

        path = unit_config.save_profile(profile, destination=destination)
        message = f'Saved {profile["profile_id"]}.'
        if flow_type and flow_set_value is None:
            message += ' Flow type is set without a value, so no flow check will run for this unit.'
        return jsonify({'success': True, 'message': message, 'path': path,
                        'profile': unit_config.get_profile(profile['profile_id'])})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/hp_design/<hp_id>', methods=['DELETE'])
def api_hp_design_delete(hp_id):
    """Remove HP design record (e.g. when retiring a unit)."""
    try:
        hp_id = str(hp_id or '').strip()
        if not hp_id:
            return jsonify({'success': False, 'message': 'hp_id required'})
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM hp_design WHERE hp_id = ?", (hp_id,))
            conn.commit()
            return jsonify({'success': True, 'message': f'HP{hp_id} design removed.'})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


def backfill_profile_and_condition_ids(only_null: bool = True) -> dict:
    """Set profile_id / condition_set_id on results from filename / hp_id (B1.3).

    Does not recompute any mean/COP/deviation numbers.
    """
    init_deviation_columns()
    conn = get_db_connection()
    updated = 0
    unresolved = 0
    try:
        rows = conn.execute(
            "SELECT rowid, file_name, hp_id AS hpid, profile_id, condition_set_id FROM results"
        ).fetchall()
        default_cs = unit_config.get_default_condition_set_id()
        for row in rows:
            d = dict(row)
            pid = str(d.get('profile_id') or '').strip() or None
            cid = d.get('condition_set_id')
            new_pid = pid
            new_cid = cid or default_cs
            if only_null and pid:
                # Keep an already-stamped profile (HPT review/cycle-extract must survive backfill).
                pass
            elif (not pid) or (not only_null):
                profile, reason = unit_config.resolve_profile(
                    file_name=d.get('file_name'), hp_id=d.get('hpid'), profile_id=None
                )
                if profile:
                    new_pid = profile.get('profile_id')
                elif not pid:
                    unresolved += 1
            if only_null and pid and cid:
                continue
            if new_pid != pid or new_cid != cid:
                conn.execute(
                    "UPDATE results SET profile_id = ?, condition_set_id = ? WHERE rowid = ?",
                    (new_pid, new_cid, d['rowid']),
                )
                updated += 1
        conn.commit()
    finally:
        conn.close()
    return {'updated': updated, 'unresolved': unresolved}


def get_profile_assignment_stats(known_ids=None):
    """Count results rows by the stamped profile_id in the open database.

    assigned: known JSON profile_id -> row count
    unassigned: empty / NULL profile_id
    orphans: stamped ids that have no JSON file
    """
    known = set(known_ids if known_ids is not None else (
        p.get('profile_id') for p in unit_config.list_profiles() if p.get('profile_id')
    ))
    assigned = {pid: 0 for pid in known}
    unassigned = 0
    orphans = []
    total = 0
    if not database_is_ready():
        return {
            'assigned': assigned,
            'unassigned': unassigned,
            'orphans': orphans,
            'total': total,
            'available': False,
        }
    try:
        conn = get_db_connection()
        try:
            cols = {row[1] for row in conn.execute('PRAGMA table_info(results)').fetchall()}
            if 'profile_id' not in cols:
                return {
                    'assigned': assigned,
                    'unassigned': unassigned,
                    'orphans': orphans,
                    'total': total,
                    'available': False,
                }
            rows = conn.execute(
                "SELECT TRIM(CAST(profile_id AS TEXT)) AS pid, COUNT(*) AS n "
                "FROM results GROUP BY pid"
            ).fetchall()
            for row in rows:
                pid = (row['pid'] if row['pid'] is not None else '').strip()
                n = int(row['n'] or 0)
                total += n
                if not pid:
                    unassigned += n
                elif pid in known:
                    assigned[pid] = assigned.get(pid, 0) + n
                else:
                    orphans.append({'profile_id': pid, 'entry_count': n})
        finally:
            conn.close()
    except Exception as exc:
        print(f'get_profile_assignment_stats: {exc}')
        return {
            'assigned': assigned,
            'unassigned': unassigned,
            'orphans': orphans,
            'total': total,
            'available': False,
        }
    orphans.sort(key=lambda o: (-o['entry_count'], o['profile_id']))
    return {
        'assigned': assigned,
        'unassigned': unassigned,
        'orphans': orphans,
        'total': total,
        'available': True,
    }


def _retarget_profile_id_on_entries(old_id, new_id):
    """Rewrite stamped profile_id in the open database. Returns rows updated."""
    if not old_id or not new_id or old_id == new_id or not database_is_ready():
        return 0
    conn = get_db_connection()
    try:
        cols = {row[1] for row in conn.execute('PRAGMA table_info(results)').fetchall()}
        if 'profile_id' not in cols:
            return 0
        cur = conn.execute(
            "UPDATE results SET profile_id = ? "
            "WHERE TRIM(CAST(profile_id AS TEXT)) = ?",
            (new_id, old_id),
        )
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


@app.route('/profiles')
def profiles_page():
    """View/edit unit profiles and condition sets (file-backed, B1.2b)."""
    unit_config.reload_all()
    profiles = unit_config.list_profiles()
    known_ids = [p.get('profile_id') for p in profiles if p.get('profile_id')]
    assignment = get_profile_assignment_stats(known_ids)
    cond_doc = unit_config.load_condition_sets_document()
    unit_types = unit_config.load_unit_types()
    default_cs_id = unit_config.get_default_condition_set_id()
    default_cs = unit_config.get_condition_set(default_cs_id)
    return render_template(
        'profiles.html',
        profiles=profiles,
        entry_counts=assignment['assigned'],
        unassigned_entry_count=assignment['unassigned'],
        orphan_profiles=assignment['orphans'],
        orphan_entry_count=sum(o['entry_count'] for o in assignment['orphans']),
        assignment_available=assignment['available'],
        assignment_total=assignment['total'],
        condition_sets_doc=cond_doc,
        unit_types=unit_types,
        default_condition_set_id=default_cs_id,
        default_condition_set_display_name=default_cs.get('display_name') or default_cs_id,
        default_condition_set_aliases=default_cs.get('aliases') or [],
    )


@app.route('/api/profiles', methods=['GET'])
def api_profiles_list():
    return jsonify({'success': True, 'items': unit_config.list_profiles()})


@app.route('/api/profiles', methods=['POST'])
def api_profiles_save():
    try:
        data = request.get_json() or {}
        destination = str(data.pop('destination', 'local') or 'local')
        original_id = str(data.pop('original_profile_id', '') or '').strip()
        original_dest = str(data.pop('original_destination', '') or destination).strip() or destination
        update_entries = bool(data.pop('update_entries', False))
        data.pop('original_entry_count', None)
        if 'profile_id' not in data or not str(data.get('profile_id') or '').strip():
            return jsonify({'success': False, 'message': 'profile_id required'})
        new_id = str(data.get('profile_id')).strip()
        data['profile_id'] = new_id
        for key in ('buh_power_cap_kw', 'pdesign_kw', 'flow_set_value'):
            if key in data and data[key] not in (None, ''):
                data[key] = float(data[key])
            elif key in data and data[key] == '':
                data[key] = None
        for key in ('default_climate', 'default_application'):
            if key in data and data[key] == '':
                data[key] = None
        if isinstance(data.get('filename_patterns'), str):
            data['filename_patterns'] = [p.strip() for p in data['filename_patterns'].split(',') if p.strip()]
        if isinstance(data.get('hp_id_aliases'), str):
            data['hp_id_aliases'] = [p.strip() for p in data['hp_id_aliases'].split(',') if p.strip()]

        is_rename = bool(original_id and original_id != new_id)
        old_path = unit_config.profile_json_path(original_id, original_dest) if original_id else ''
        new_path = unit_config.profile_json_path(new_id, destination)
        if is_rename and unit_config.get_profile(new_id) and os.path.normpath(old_path) != os.path.normpath(new_path):
            return jsonify({
                'success': False,
                'message': f'profile_id {new_id} already exists. Choose another id or edit that profile.',
            })

        path = unit_config.save_profile(data, destination=destination)
        removed_old = False
        if original_id and os.path.normpath(old_path) != os.path.normpath(new_path) and os.path.isfile(old_path):
            removed_old = unit_config.delete_profile(original_id, original_dest)

        entries_updated = 0
        leftover = 0
        if is_rename and update_entries:
            entries_updated = _retarget_profile_id_on_entries(original_id, new_id)
        elif is_rename and database_is_ready():
            try:
                conn = get_db_connection()
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) AS n FROM results "
                        "WHERE TRIM(CAST(profile_id AS TEXT)) = ?",
                        (original_id,),
                    ).fetchone()
                    leftover = int(row['n'] or 0) if row else 0
                finally:
                    conn.close()
            except Exception:
                leftover = 0

        bits = [f'Saved {new_id}.']
        if is_rename:
            bits.append(f'Renamed from {original_id}.')
            if removed_old:
                bits.append('Removed the old JSON file.')
            if update_entries:
                bits.append(f'Updated {entries_updated} entries in this database.')
            elif leftover:
                bits.append(
                    f'{leftover} entries still stamped {original_id} were left unchanged '
                    '(they will show as orphans until you update them).'
                )
        return jsonify({
            'success': True,
            'path': path,
            'profile': unit_config.get_profile(new_id),
            'renamed': is_rename,
            'entries_updated': entries_updated,
            'message': ' '.join(bits),
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/profiles/<profile_id>', methods=['DELETE'])
def api_profiles_delete(profile_id):
    destination = request.args.get('destination', 'local')
    ok = unit_config.delete_profile(profile_id, destination=destination)
    return jsonify({'success': ok, 'message': 'deleted' if ok else 'not found'})


@app.route('/api/condition_sets', methods=['GET'])
def api_condition_sets_get():
    return jsonify({'success': True, 'document': unit_config.load_condition_sets_document()})


@app.route('/api/condition_sets', methods=['POST'])
def api_condition_sets_save():
    try:
        data = request.get_json() or {}
        if 'condition_sets' not in data:
            return jsonify({'success': False, 'message': 'condition_sets object required'})
        unit_config.save_condition_sets_document(data)
        _reload_setpoint_maps()
        global DEVIATION_BANDS
        DEVIATION_BANDS = calculate_deviation_bands(DEVIATION_SETPOINTS, DB_BAND, WB_BAND)
        return jsonify({'success': True, 'document': unit_config.load_condition_sets_document()})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/backfill_profile_ids', methods=['POST'])
def api_backfill_profile_ids():
    try:
        result = backfill_profile_and_condition_ids(only_null=True)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/export')
def export_summary_page():
    """UI for T4.2 summary CSV/XLS export (C2 / C2b)."""
    tool_version = summary_export.get_tool_version(config)
    selected_columns = summary_export.load_default_columns()
    using_custom_default = selected_columns != list(summary_export.DEFAULT_EXPORT_COLUMNS)
    total_count = 0
    cop_yes_count = 0
    try:
        conn = get_db_connection()
        try:
            total_count = conn.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
            cols = [r[1] for r in conn.execute("PRAGMA table_info(results)")]
            cop_col = next(
                (c for c in cols if c.lower().replace(" ", "").replace("-", "_") == "cop_dataset"),
                None,
            )
            if cop_col:
                cop_yes_count = conn.execute(
                    f'SELECT COUNT(*) AS n FROM results '
                    f'WHERE upper(trim(cast("{cop_col}" AS TEXT))) = \'YES\''
                ).fetchone()["n"]
        finally:
            conn.close()
    except Exception as e:
        flash(f"Could not read database for export counts: {e}", "warning")
    return render_template(
        "export_summary.html",
        schema_id=summary_export.SCHEMA_ID,
        column_count=len(summary_export.DEFAULT_EXPORT_COLUMNS),
        all_columns=list(summary_export.DEFAULT_EXPORT_COLUMNS),
        column_groups=summary_export.ui_column_groups(),
        selected_columns=selected_columns,
        using_custom_default=using_custom_default,
        tool_version=tool_version,
        total_count=total_count,
        cop_yes_count=cop_yes_count,
    )


def _export_request_columns():
    """Columns from POST form / JSON / GET; else saved default; else full schema."""
    cols = request.values.getlist("columns")
    if not cols and request.is_json:
        body = request.get_json(silent=True) or {}
        raw = body.get("columns") or body.get("default_columns")
        if isinstance(raw, list):
            cols = raw
    if cols:
        return list(summary_export.resolve_export_columns(cols))
    return summary_export.load_default_columns()


@app.route("/export/summary", methods=["GET", "POST"])
def export_summary_download():
    """Download summary export matching t42_summary_v1 (C2 / C2b)."""
    fmt = (request.values.get("format") or "csv").strip().lower()
    if fmt not in ("csv", "xlsx"):
        fmt = "csv"
    cop_only = request.values.get("cop_only", "").strip().lower() in ("1", "true", "yes", "on")
    columns = _export_request_columns()
    tool_version = summary_export.get_tool_version(config)
    try:
        conn = get_db_connection()
        try:
            payload, filename, mime, _n = summary_export.build_export_bytes(
                conn,
                fmt=fmt,
                tool_version=tool_version,
                cop_dataset_only=cop_only,
                columns=columns,
            )
        finally:
            conn.close()
        return send_file(
            BytesIO(payload),
            mimetype=mime.split(";")[0],
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        flash(f"Export failed: {e}", "danger")
        return redirect(url_for("export_summary_page"))


@app.route("/api/export/columns_default", methods=["GET", "POST", "DELETE"])
def api_export_columns_default():
    """Load / save / reset local default export columns (C2b)."""
    try:
        if request.method == "GET":
            cols = summary_export.load_default_columns()
            return jsonify({
                "success": True,
                "columns": cols,
                "is_custom": cols != list(summary_export.DEFAULT_EXPORT_COLUMNS),
                "schema_id": summary_export.SCHEMA_ID,
            })
        if request.method == "DELETE":
            cols = summary_export.reset_default_columns()
            return jsonify({
                "success": True,
                "columns": cols,
                "message": "Default reset to all usual columns.",
            })
        data = request.get_json(silent=True) or {}
        raw = data.get("default_columns") or data.get("columns") or []
        if not isinstance(raw, list) or not raw:
            return jsonify({"success": False, "message": "default_columns list required"})
        cols = summary_export.save_default_columns(raw)
        return jsonify({
            "success": True,
            "columns": cols,
            "message": f"Saved default ({len(cols)} columns).",
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/datasets")
def datasets_page():
    """Create / open / switch analysis databases (B1.6). Never overwrites existing files."""
    active = get_database_path()
    ready = database_is_ready()
    dbs = dataset_manager.list_managed_databases(active_path=active if ready else None)
    active_in_managed = bool(ready and dataset_manager.is_under_datasets_dir(active))
    return render_template(
        "datasets.html",
        databases=dbs,
        datasets_dir=dataset_manager.datasets_dir(),
        active_in_managed=active_in_managed,
    )


@app.route("/api/datasets", methods=["GET"])
def api_datasets_list():
    active = get_database_path()
    return jsonify({
        "success": True,
        "ready": database_is_ready(),
        "active_path": active if database_is_ready() else None,
        "items": dataset_manager.list_managed_databases(
            active_path=active if database_is_ready() else None
        ),
    })


@app.route("/api/datasets/create", methods=["POST"])
def api_datasets_create():
    """Create a new empty DB file (refuses if name already exists) and open it."""
    try:
        data = request.get_json(silent=True) or {}
        name = data.get("name") or ""
        # Fingerprint protected BAM DB if present — must remain unchanged by create.
        protected = os.path.normpath(os.path.join(_APP_DIR, "..", "notebooks", "data_analysis.db"))
        finger_before = (
            dataset_manager.file_fingerprint(protected) if os.path.isfile(protected) else None
        )
        new_path = dataset_manager.create_new_database(name)
        activate_database(new_path, prepare=True)
        if finger_before is not None:
            finger_after = dataset_manager.file_fingerprint(protected)
            if finger_after != finger_before:
                return jsonify({
                    "success": False,
                    "message": (
                        "Safety stop: data_analysis.db changed during create. "
                        "No further action taken beyond opening the new file path check."
                    ),
                }), 500
        return jsonify({
            "success": True,
            "path": new_path,
            "message": f"Created and opened {os.path.basename(new_path)}.",
        })
    except FileExistsError as e:
        return jsonify({"success": False, "message": str(e)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/datasets/open", methods=["POST"])
def api_datasets_open():
    """Open an existing DB by path. Does not delete or overwrite any file."""
    try:
        data = request.get_json(silent=True) or {}
        path = data.get("path") or ""
        protected = os.path.normpath(os.path.join(_APP_DIR, "..", "notebooks", "data_analysis.db"))
        opening_protected = os.path.normpath(dataset_manager.resolve_db_path(path)) == protected
        finger_before = (
            dataset_manager.file_fingerprint(protected)
            if os.path.isfile(protected) and not opening_protected
            else None
        )
        # When opening data_analysis.db itself, allow ADD COLUMN growth but not shrink
        abs_path = activate_database(path, prepare=True)
        if finger_before is not None:
            finger_after = dataset_manager.file_fingerprint(protected)
            if finger_after != finger_before:
                return jsonify({
                    "success": False,
                    "message": "Safety stop: data_analysis.db was modified while opening another database.",
                }), 500
        return jsonify({
            "success": True,
            "path": abs_path,
            "message": f"Opened {os.path.basename(abs_path)}.",
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/datasets/browse", methods=["POST"])
def api_datasets_browse():
    """Native file dialog, then open the selected existing database."""
    try:
        path = dataset_manager.browse_open_database_dialog()
        if not path:
            return jsonify({"success": False, "cancelled": True, "message": "Cancelled."})
        abs_path = activate_database(path, prepare=True)
        return jsonify({
            "success": True,
            "path": abs_path,
            "message": f"Opened {os.path.basename(abs_path)}.",
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


if __name__ == '__main__':
    # Ensure auxiliary tables exist at startup only when a DB is already configured & present.
    # Never create an empty file just because the path is missing (B1.6).
    if database_is_ready():
        try:
            prepare_active_database()
            print(f"Database ready: {get_database_path()}")
        except Exception as _e:
            print(f"Warning: could not prepare database schema: {_e}")
    else:
        print(
            f"No ready database at {get_database_path()}. "
            "Open http://127.0.0.1:5001/datasets to create or open one."
        )
    print(f"Starting Flask app at http://127.0.0.1:5001/  (database: {get_database_path()})")
    print('\nRegistered routes:')
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} -> {rule.endpoint} [{', '.join(rule.methods)}]")
    print()
    app.run(debug=True, use_reloader=False, threaded=True, host='0.0.0.0', port=5001)
    