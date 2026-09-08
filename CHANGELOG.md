# Changelog

All notable changes to the T4.2 Data Analysis WebApp are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Version numbers match `flask_app/VERSION` and the `tool_version` column on summary export.

---

## [0.1.0] — 2026-09-07

First tagged release for HPT T4.2 air-to-water analysis.

### Added

- **Clone-ready workflow:** `import_hpt.ipynb`, `import_tools/hpt_aw_rev1`, empty `Rohdaten/` and `Plotdaten/`
- **Analyst Quick Guide** (`docs/ANALYST_QUICK_GUIDE.md`) — clone → import → WebApp → deviations → export
- **HPT unit profile placeholders** (`HPT_RRT1` … `HPT_RRT6_STAGED`) with generic example filenames in UI templates
- **`config_template.json`** with Plotdaten paths, required columns, and insert query for new databases
- **Unit identity:** profile required on cycle extract; HP ID from profile; clash labels when profiles share an HP digit
- **Climate / application** defaults (Average / MT) visible on scatter and in plot titles
- **Profile-aware deviations:** `checks.json` per profile; variable-flow tests skip flow-percent checks; water-to-water outdoor checks skipped when configured
- **Summary export** `t42_summary_v1` with column picker on `/export`
- **Windows launchers** `run_app.ps1` / `run_app.bat` (UTF-8 safe)
- **`FLASK_SECRET_KEY`** environment variable for session secret (optional; dev default if unset)
- **Optional monitoring means (air-to-water)** — sixteen extra cycle means from the REV1 template: `avg_lab_air_db`, `avg_lab_atm_pressure`, `avg_cp_out_sink`, `avg_cp_in_sink`, `avg_density_sink`, `avg_compressor_frequency`, `avg_fan_speed`, `avg_voltage`, `avg_current`, `avg_frequency`, `avg_cos_phi`, `avg_real_buh_power`, `avg_t_in_cond`, `avg_t_out_cond`, `avg_t_in_evap`, `avg_t_out_evap`. Configurable through `optional_average_columns`; a local `config.json` without that key falls back to the built-in list. Missing series stay **n/a**, never `0`, and no deviation or pass/fail rule reads them
- **Summary export:** the same sixteen fields appended to the end of `t42_summary_v1` (existing columns keep their position) with a new **Optional monitoring (A/W)** picker group
- **MIT licence** (`LICENSE`)

### Changed

- **Derived mass flow — one rule for BAM and HPT** (`derive_mass_flow_if_needed()` in `flask_app/app.py`): measured `mass flow` always wins and is never overwritten (zeros are measurements); an empty column counts as missing; rows without a measured value are derived as `volume flow [m³/h] / 3600 × ρ`, with ρ per row from `density_sink`, else `density`, else 997 kg/m³. A density outside 900–1100 kg/m³ is rejected for that row in favour of 997 with a warning. Analysis-time only — nothing is written back to Plotdaten
- **Import map `hpt_aw_rev1` 0.4.0:** `Backup heater Input (optional)` maps to `Real BUH power input` (monitoring mean only; never sets `has_buh`). `Virtual Back-up` → `Electrical power input BUH` remains the series used for BUH corrections. `cp_out_sink` / `cp_in_sink` keep the `_sink` suffix; `rho_sink` maps to `density_sink`; sheet `F` is expected
- Single Plotdaten column layout for BAM and HPT air-to-water paths
- Condition set display uses Ecodesign heating naming (`BAM_RRT` remains an internal alias)
- Mean Tmean deviation uses profile `flow_mode` (fixed vs variable), not filename heuristics
- Example UI text uses HPT naming
- **Means / review UI:** compact selected-rows toolbar, sticky identity columns, auto-filled HP ID and Test label on new-entry review, cycle-pipeline fields hidden on Means unless turned on, Notes `drop` / `ramp` required to split defrost and on/off sub-periods

### Compatibility

- Existing BAM databases gain new mean columns by `ALTER TABLE … ADD COLUMN` only. The `results` table is never dropped; existing rows stay valid
- BAM Plotdaten files are unaffected: `Electrical power input BUH` keeps its name and meaning

### Security / sharing

- Repository contains **code, templates and documentation only** — no measurement Excel or SQLite databases
- `.gitignore` excludes data paths (`Rohdaten/`, `Plotdaten/`, `*.xlsx`, `*.db`, local `config.json`)

### Known limitations (out of scope for v0.1)

| Area | Status in v0.1 |
|------|----------------|
| **Water-to-water import** | Not shipped — awaits REV1 template and import map |
| **Hybrid import** | Not shipped — awaits REV1 template and import map |
| **Pass/fail flags in export** | Columns present but empty — awaits T4.3 / guideline rules |
| **Real unit data on HPT_RRT1–3** | Placeholder capacities and checks only |
| **Per-test draft report** | Planned for a later release |
| **Shared multi-user server** | Local use only (`127.0.0.1:5001`) |

### Documentation

- Analyst Quick Guide, User Guide, Analysis Guide, and Requirements — Markdown in `docs/`
- Version history — this file
