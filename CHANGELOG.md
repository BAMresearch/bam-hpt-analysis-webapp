# Changelog

All notable changes to the T4.2 Data Analysis WebApp are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Version numbers match `flask_app/VERSION` and the `tool_version` column on summary export.

---

## [Unreleased]

### Added

- **Guideline clocks on Cycle Extract.** For the open file, **Propose guideline clocks** derives sub-periods on existing database entries: **D** (defrost + buffer) or **S** (standby + buffer), **H**, and — on defrost and continuous cycles — **equilibrium** then **evaluation**. Evaluation is the first `eval_min` after the first `eq_min` of H, not the last minutes of the cycle; it is omitted when H is too short. The parent row stays the full cycle
- **Clock table and period layers** (layers off by default). Edit the power-transition time or D/S end; unlock equilibrium / evaluation to type them. **Save clocks** or **Save all proposed** stores the times on the parent entry
- **Treat unknown rows as** (defrost / on–off / continuous) fills only rows that have no kind yet. Kind **`continuous`**: no D/S; H is the parent window. Stored `other` stays unknown, not continuous
- Clock lengths in `flask_app/config/cycle_periods.json` (`buffer_min` 10, `eq_min` 60, `eval_min` 70)
- **Save clocks for the open database** on Cycle Extract — the first fill (and later the gap-fill) of guideline clocks for every file that has entries in the open database. Needs a database, not a loaded file. Asks first, and the confirm names how many files and entries it will write, how many already-saved entries it will leave alone, and the **Treat unknown rows as** default for rows with no kind. It then works one file at a time with a *file i of n* status, so one unreadable sheet fails that file only. Already-saved entries are never overwritten and cost no Excel read; to change a file you can see, open it and use **Save all proposed** as before. Guideline Windows stays read-only
- **Clear saved guideline clocks** on Cycle Extract — throw a clock pass away (whole database, or the loaded file) so the next fill can derive fresh clocks: running the fill again cannot, since it skips saved rows by design. Confirm names the count. It deletes the saved guideline periods and the stored power split on those rows only; the parent cycles, their means and deviations, and the Period Statistics drop / ramp splits all stay, and the fill does not start itself
- **Stored interval scores on Guideline Windows.** Saving guideline clocks now also stores that entry's interval numbers (H / eq / eval percentages, D / S percentages, H / D / S means, evaluation Tsup / Q / P / COP and eval ΔCOP) next to the clocks, using the sheet the save already has open. Guideline Windows reads them back: opening the page shows the table straight away, no Plotdaten file is opened, and coming back to the page is as quick as the first visit. Rows whose clocks changed since, or whose interval bands were retuned, show `n/a` with a reason rather than an old percentage, and **Clear saved guideline clocks** removes the stored scores with the clocks
- **Compute missing scores** on Guideline Windows — the one-off backfill for entries that have clocks but no stored scores (clocks saved before this). It asks first with the count, then works one file at a time (*file i of n*), reading each Plotdaten file once and reporting a file it cannot read. It writes the stored scores only: no proposal, no clock, no change to the parent Deviations, and it never starts on page load. Nothing else on the page does this work
- **Guideline Windows** (`/guideline_windows`) — check **saved** clocks across many entries: D / S, H, equilibrium and evaluation times, plus the entry's Tsup, Q, P and COP beside the same four on the evaluation window. Open from **Selected rows** on Mean Values, or from the navbar for the whole open database. Evaluation Q/P/COP use the same analysis-time quantities as the entry (including files that only have uncorrected Q and P). Evaluation cells stay **empty** (never `0`) when there is no evaluation window. CSV of the table; the summary export is unchanged

### Changed

- Proposing clocks no longer requires Notes `drop` / `ramp`
- Partner **User Guide** describes how to operate guideline clocks and Guideline Windows; interval bands, ΔCOP and the dTreturn definition live in the **Analysis Guide**. The **Analyst Quick Guide** includes the optional clock step
- **Guideline Windows** now opens on **`#`** — the entry's position in the open database, the same number it carries on Data and on Deviations — instead of the internal database id, which has gaps after a deletion. A **COP dataset** column (`YES` / `NO` / blank) sits beside the other entry columns. Three filters above the table (kind; COP dataset; with or without saved clocks) narrow it in the browser: they hide rows without renumbering `#`, open no Plotdaten file and recompute nothing, **Clear filters** resets them, and the status line ends with *Showing X of Y rows*. **Download CSV** writes the filtered table, with `#` and COP dataset and without the database id. The wide table now scrolls sideways inside its own box, with the scrollbar under the table, instead of sliding the whole page. **Load all entries in the open database** is gone — the page loads that table itself; **Load the selected row(s)**, **Compute missing scores** and **Download CSV** are unchanged
- Opening **Guideline Windows** now shows the table for the open database instead of an empty page — since the scores are stored, that is a database read and costs nothing. The buttons still control what is loaded
- Scoring an interval no longer prints a `cap value` or `Electrical power input BUH` line per cycle to the console
- **Guideline Windows** now shows H / eq / eval **DB %**, **WB %**, **flow %** and **dTreturn %** (Interval H bands on each saved window) and evaluation **ΔCOP** (first 5 min against last 5 min of the saved evaluation window, limit 2.5 %) when you load the rows. There is no interval Tsup %. The extra **Deviations by guideline interval** table is removed; parent Deviations stays one table (full-cycle Tsup % and dTreturn −2…+2 K unchanged)
- Guideline Windows DB / WB / flow % use the same parent-window series as Deviations Calculate (stored time series or Plotdaten plus derived mass flow), not only the raw Excel cache. A failed window score puts the scorer's reason on the n/a cell (hover). Stored `dev_db_setpoint` is used if the live outdoor-setpoint lookup is empty
- **Guideline Windows** scores **D** and **S individual %** on the union of saved spans (D: outdoor DB ±5 K and dTreturn ±2 K; S: DB/WB ±2 K, flow ±2.5 % on fixed-flow tests, dTreturn ±1 K). Two stored D (or S) spans are one number, not two. A defrost row leaves S n/a; an on–off row leaves D n/a. Parent Deviations is unchanged
- **Guideline Windows** also shows **H / D / S mean deviations** — one signed arithmetic mean per interval against its setpoint, in K (or % of set for flow), not a share of samples. H (the saved heating or `on` window): DB ±0.3 K, WB ±0.4 K, Tsup ±0.5 K, dTreturn ±0.3 K, flow ±1 % on fixed-flow tests. D (union of saved spans): DB ±1.5 K, WB ±1.0 K. S (union of saved spans): DB ±0.6 K, WB ±0.6 K, flow ±2.5 %. Cells the draft leaves blank stay n/a, never `0`; hover gives the band or the reason. Full-cycle mean Tsup, Tmean and Q stay on Deviations, and the parent bands are unchanged
- **Cycle Extract** Period layers (still off by default) draw D/S, H, equilibrium and evaluation on the existing chart. There is no Tsup band on that page. The Deviations Plot shades saved guideline D/S, H, equilibrium and evaluation on DB/WB, Tsup, dTreturn and flow. Guideline Windows stays a table; Tsup % is unchanged
- **Deviations Plot** is now **one figure per quantity** in one modal: outdoor DB/WB (air-source only), Tsup, dTreturn, and flow on fixed-flow tests. The separate **Flow** button is gone — flow comes back with the rest. Each trace is drawn once; the band is what changes. With **saved** guideline clocks the filled band is stepped at the clock edges with the per-interval individual half-widths (DB ±5 / ±1 / ±2 K on D / H / S, dTreturn ±2 / ±0.5 / ±1 K, flow ±2.5 % on H and S) and the parent / full-cycle limits stay visible as dashed lines; without clocks it is the full-cycle band on its own and the modal says so. Equilibrium and evaluation reuse the H width; a quantity the draft leaves n/a has no fill on that stretch. Tsup never steps — the draft gives the liquid sink outlet only a mean band, so that figure stays the mean ±0.5 K check. A variable-flow test simply has no flow figure. The Deviations table still scores the full cycle and Guideline Windows stays the per-interval table; parent bands are unchanged
- **Cycle Extract** can now *look* at saved clocks without re-deriving them. Loading a file brings the stored guideline periods with it, so **show** is available for any row that already has saved clocks (as well as for any row you have proposed) and the Period layers draw the **proposal** while you edit a row and the **saved** bands otherwise. Layers stay off by default and still draw only ticked rows. **edit**, **Save clocks** and **Save all proposed** still need **Propose guideline clocks**; a proposal that no longer matches the stored rows is marked ≠ saved and nothing is written until you save
- A **second D / S start** you typed survives Save and reload: the next **Propose** reads it back from the stored clocks instead of re-detecting the drop, and a cycle you forced to one span stays one span. **reset** on that row still drops it and detects again

### Fixed

- A cycle cut from one defrost (or standby) **end** to the next now gets a **second D / S** span; a cut from start to start stays one span
- Changing **Treat unknown rows as** updates Kind and the clocks immediately. If a clock button reports that the server did not answer with JSON, restart the app and reload the page
- A missing outdoor dry-bulb setpoint no longer blanks Tsup, dTreturn or flow % on Deviations Calculate or Guideline Windows H / eq / eval
- Variable-flow n/a hover on flow % names the variable-flow skip instead of a generic “does not apply”
- Deviations Plot now finds the entry by `rowid` (and compares `data_set` as a number), so saved guideline clocks actually shade the figures
- Flow Plot uses the entry’s unit profile and any stored flow set; a missing mass-flow column is not treated as “no flow set”

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
