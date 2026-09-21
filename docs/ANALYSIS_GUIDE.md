# Analysis Guide — T4.2

**Version:** v0.1
**Date:** 2026-09-03
**Status:** Draft for T4.2 circulation (v0.1) — describes the tool as implemented; items marked **TBD** await the test guideline / T4.3
**Audience:** T4.2 analysts and validators

**Code:** `flask_app/app.py`
**Configuration read:**
`flask_app/permissible_deviations.json`,
`flask_app/config/interval_deviations.json`,
`flask_app/config/cycle_periods.json`,
`flask_app/config/condition_sets.json`,
`flask_app/config/profiles/`,
`flask_app/config/unit_types.json`,
`flask_app/config/checks.json`
(`flask_app/config.json` is local and not versioned. `t_supply_set_mapping` and the legacy `cap_values` are kept in sync or used as fallbacks; they are not the source of truth.)

> **Content rule:** this document describes only *methods, definitions and configurable limits*. It contains no unit identities, unit parameters, test file names or measured results from any test campaign.

**This document is methods.** Operation — how to install, import and click through the tool — is in [`USER_GUIDE.md`](USER_GUIDE.md), with the short single-analysis path in [`ANALYST_QUICK_GUIDE.md`](ANALYST_QUICK_GUIDE.md).

Two distinctions run through the whole document: **"as implemented today"** describes current tool behaviour and is verified against the code; **"TBD"** marks a point where the guideline, T4.3 or the consortium has not yet decided, and where the tool therefore must not be treated as authoritative.

---

## 1. Analysis window / cycle

| Rule | As implemented today |
|------|----------------------|
| Window | Inclusive: `start_time ≤ time_elapsed ≤ end_time` |
| Who sets it | Labs / keys workbook provide start–end; analyst confirms in the tool when creating/editing entries |
| Auto-detect | Optional electrical-power transition / cycle classification (`cycle_type`, `pelec_transition_time`) for defrost / on–off period splits — **not** required for mean/COP/deviation stats on the full entry window |
| Periods | Sub-periods can be stored separately; the full-cycle `results` row remains the primary analysis unit for means/COP/`dev_*` |

**Guideline clocks** (Cycle Extract, stored on the parent row) name the draft intervals. Lengths are `flask_app/config/cycle_periods.json` (defaults: buffer 10 min, equilibrium 60 min, evaluation 70 min). How to propose and save them is in the User Guide.

| Interval | Window |
|----------|--------|
| **D** | Defrost plus the buffer after heating resumes. Two stored spans on one parent are **one** interval (union of times) |
| **S** | Standby plus the buffer after restart (same one-or-two-span union) |
| **H** | Heating between D or S pieces. On a *continuous* cycle, H is the whole parent window |
| **Equilibrium** | First `eq_min` of H (defrost and continuous). Reuses Interval H bands |
| **Evaluation** | The `eval_min` **immediately after** equilibrium, not the last minutes of H or of the parent. Omitted when H is shorter than `eq_min + eval_min`. Reuses Interval H bands. On–off cycles have no eq/eval |

**TBD (guideline / T4.3):** Whether acceptance uses complete cycles only, steady sub-periods only, or both; whether auto-detect becomes mandatory.

---

## 2. Mean values

Over the analysis window:

\[
\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i
\]

(arithmetic mean of the time-series column values over the window).

Typical stored means: outdoor DB/WB, supply, return (emu/calc), heating capacity, electrical power (corrected, uncorrected, and BAM-pump-corrected variants as available). The importer raises on any missing `average_columns` entry except lab-corrected Q/P: outdoor DB/WB plus `Pressure difference`, `volume flow` and the buffer columns `T_return_calc` / `T_B_calc` / `T_H_calc` / `Q_HB` / `Q_BA` (this is why a water-to-water workbook cannot be imported yet, §12.8). The deviation stage is what tolerates a missing outdoor series.

### 2.1 Optional monitoring means

A second, **optional** group of means comes from the air-to-water REV1 template. They use the same arithmetic mean over the same window, but they are never mandatory, so a file without them — every existing BAM Plotdaten file, and template sheets B/C/D — imports and calculates unchanged.

| Plotdaten column | Stored as |
|------------------|-----------|
| `Lab air temperature (DB)` | `avg_lab_air_db` |
| `Lab atmospheric pressure` | `avg_lab_atm_pressure` |
| `cp_out_sink` / `cp_in_sink` | `avg_cp_out_sink` / `avg_cp_in_sink` |
| `density_sink` | `avg_density_sink` |
| `Compressor Frequency` / `Fan Speed` | `avg_compressor_frequency` / `avg_fan_speed` |
| `Voltage` / `Current` / `Frequency` / `cos_phi` | `avg_voltage` / `avg_current` / `avg_frequency` / `avg_cos_phi` |
| `Real BUH power input` | `avg_real_buh_power` |
| `T_in_cond` / `T_out_cond` / `T_in_evap` / `T_out_evap` | `avg_t_in_cond` / `avg_t_out_cond` / `avg_t_in_evap` / `avg_t_out_evap` |

Rules that make them safe to read:

- The mean is stored only when the column exists **and** the window holds at least one finite number. Otherwise the field is NULL and renders as **n/a** — never `0`.
- They are written after the core insert or update, addressed by row id, so the mandatory mean set and the values it produces are untouched. Recalculating an entry refreshes them.
- **No deviation, band, colour or pass/fail rule reads them.** They document the test conditions; they do not judge them.
- `avg_real_buh_power` is the real backup-heater meter (§4.3). It is not `avg_power_buh`, and it never becomes one.
- Which channels are averaged is `optional_average_columns` in the config; a local `config.json` predating v0.1.0 simply falls back to the built-in list.

Deviation page “mean DB/WB/Tsup deviation” = \(\bar{T} - T_{\mathrm{set}}\) (computed at render from stored averages + setpoints; not a separate DB column). Mean Tsup prefers the post-BUH supply (`avg_ts_buh` / `Ts Buh`) when that series exists, so BUH and non-BUH cycles are judged on the same leaving-water point.

**TBD:** Final list of mandatory mean columns for the T4.2 summary export.

---

## 3. COP

\[
\mathrm{COP} = \frac{\overline{Q}}{\overline{P}}
\]

**not** \(\overline{Q/P}\).

| Variant | Capacity column | Power column | Typical store |
|---------|-----------------|--------------|---------------|
| Corrected | `Heating Capacity (corrected)` | `Electric Power Input (corrected)` | `cop_corr` / related |
| Uncorrected | `Heating Capacity (without corr)` | `Electric Power Input (without correction)` | `cop_uncorr` / related |
| BAM pump-corrected | `Heating Capacity (BAM corrected)` | `Electric Power Input (BAM corrected)` | `cop_bam_corr` / related |
| With BUH (when present) | BUH-corrected Q | BUH-corrected P | e.g. `COPCorrwBUH` |

When the Plotdaten file has no lab-corrected Q/P, the database **corrected** fields are filled from the BAM pump-corrected values and the analyst is told (§4.2). Those stored “corrected” means are then BAM-corrected, not lab-corrected.

**TBD:** Which COP column is the official T4.2 reporting value.

---

## 4. Analysis-time fills (not written into Plotdaten)

The importer does not invent mass flow or lab-corrected Q/P in the Plotdaten file. An absent BUH column is materialised **in memory** as zeros with `has_buh = False`; nothing is written back to Plotdaten. The WebApp may derive mass flow, BAM-corrected fallbacks, and BUH gap-fills at **analysis time** and must tell the user (flashed notices / console). Gaps stay visible in the archive file.

### 4.1 Derived mass flow

One rule for BAM and HPT Plotdaten, in this order:

1. **Measured `mass flow` wins.** Rows that carry a value are never overwritten — including zeros, which are measurements, not placeholders.
2. Every row without a measured value is derived from volume flow:

\[
\dot{m}\;[\mathrm{kg/s}] = \frac{\dot{V}\;[\mathrm{m^3/h}]}{3600}\times \rho\;[\mathrm{kg/m^3}]
\]

3. ρ is taken per row from `density_sink`, else `density`, else 997 kg/m³. A partly filled density column is used where it is finite; only its gaps fall back to 997. A value outside 900–1100 kg/m³ (e.g. 0.997, i.e. g/cm³) is rejected for that row in favour of 997 and warned about. A future water-to-water source density is never used for the sink mass flow.

A **missing column and an empty column are the same case**: the HPT template always ships `mass flow_sink`, so an empty column must not block derivation. Volume flow in Plotdaten is already m³/h — the importer converted it, so there is no second ×3600 here.

The notice always names the path that ran (measured / derived with measured density / derived with 997), so a derived series is never silent. Prefer measured mass flow when the lab provides it. Nothing is written back to the Plotdaten file.

Implemented once in `derive_mass_flow_if_needed()` (`flask_app/app.py`) and used by import, recalculation, the single-column update, the cycle extract and the flow-deviation plot, so BAM and HPT entries cannot drift apart.

### 4.2 BAM pump-correction fallback

The tool always computes BAM pump-corrected Q and P from uncorrected capacity/power, hydraulic power and a pump-efficiency model. If lab-corrected columns are missing, the stored `Heating Capacity (corrected)` / `Electric Power Input (corrected)` averages are those BAM-corrected values. Do not compare them to genuinely lab-corrected figures from another unit without saying so.

### 4.3 Virtual Back-up vs real Backup heater

The WebApp does **not** rename columns. It reads only the literal column `Electrical power input BUH`. Whichever series the lab (or the Plotdaten export) puts in that column is the BUH. Any other backup-heater column is never read as BUH. “Virtual Back-up” / “Real Backup heater” are lab/template names, not a WebApp mapping.

The physical reason the two are handled differently:

| | Virtual Back-up → `Electrical power input BUH` | Real backup heater → `Real BUH power input` |
|---|---|---|
| Inside the measured total power / heating capacity? | **No** — it is added on top | **Yes** — the meter is already in the total |
| Drives the WebApp BUH adjustments (power, heat, COP with BUH, `T_supply_afterBUH`)? | **Yes** | **No** — adding it again would double-count |
| Stored as | `avg_power_buh`, `avg_power_buh_from_ts`, `avg_t_sup_buh`, `*wBUH` columns | `avg_real_buh_power` — a monitoring mean (§2.1) only |

A test uses one or the other. If a workbook carries both, the importer keeps both columns and warns; the corrections still use the virtual series alone.

| Name | Role |
|------|------|
| **WebApp BUH** | Column `Electrical power input BUH` only. `has_buh` depends on nothing else — a file with `Real BUH power input` and no virtual column still gets `has_buh = False`. |
| **Other backup-heater columns** | Never read as BUH; `Real BUH power input` is averaged as a monitoring mean. |
| **Absent or all-zero BUH column** | `has_buh = False`. No `power_BUH_from_Ts` fill; supply after BUH equals `T_supply`. A column of zeros is treated like a missing column (template placeholder, not a measured zero). The all-zero test is run on the whole file **and again on the analysis window**, so an all-zero cycle inside a BUH file is also treated as no BUH. |
| **`power_BUH_from_Ts`** | When BUH is present and the measured BUH series has NaNs: \(\dot{m}\,c_p\,(T_{\mathrm{supply,set}}-T_{\mathrm{supply}})\), then clamped to \([0,\,\texttt{buh\_power\_cap\_kw}]\) so the fill is never negative (legacy cap fallback: `config.json` `cap_values` / `default_cap`). Used only to fill gaps, not to invent a heater. \(T_{\mathrm{supply,set}}\) is `TSUP_SETPOINTS[letter]` (Average/MT). The letter is the first character of the third `_`-separated filename part (`extract_letter`); other filename shapes and letter G (no Average/MT `tsup_set`) give no setpoint, so no gap-fill occurs. The fill does not follow the entry’s climate × application. |

Tsup deviation uses `Ts Buh` when present, else `T_supply`.

---

## 5. Instantaneous permissible deviations

Limits file: **`flask_app/permissible_deviations.json`**.

Setpoints: **`flask_app/config/condition_sets.json`**, default set `ecodesign_heating` (legacy alias `BAM_RRT`). Nested as climate × application × letter. Empty climate/application on an entry resolves to the unit profile’s `default_climate` / `default_application`, and only then to **Average / MT**. Letter: the **whole-table** calculation uses stored metadata (`dev_test_condition` / `test_cond`) if present, else the filename (E/A/F/B/C/D/G); the **per-column Update** currently derives the letter from the filename only and does **not** read the entry’s stored climate/application, so its Tsup/Tmean setpoints fall back to the profile default and then Average/MT. The wet-bulb setpoint is hardcoded as \(T_{\mathrm{db,set}} - 1\,\mathrm{K}\); `wb_rule` in the condition set / profile is declarative metadata that the deviation code does not read yet.

Which checks run is **`checks.json` × `unit_types.json` × the resolved flow mode** (§7). Inapplicable checks are stored empty and shown as **N/A**, never as 0.

| Quantity | Series | Band / rule | When it applies | Reported stats |
|----------|--------|-------------|-----------------|----------------|
| DB | `T_outdoor (DB)` | \(T_{\mathrm{db,set}} \pm\) `DB.value` | `source_medium = air` and the series exists | count, %, max +, max − |
| WB | `T_outdoor (WB)` | \((T_{\mathrm{db,set}}-1) \pm\) `WB.value` | air-source with `has_wetbulb` and the series exists | count, %, max +, max − |
| Tsup | `Ts Buh` if present else `T_supply` | \(T_{\mathrm{sup,set}} \pm\) `Tsup.value` | every unit type (**parent table only** — see §5.1) | count, %, max +, max − |
| Flow | mass or volume flow vs profile flow set | set ± `flow_instantaneous_pct` % | **fixed-flow tests only** | count, %, mean flow %, max +, max − (in % of set) |
| dTreturn | see §8 | see §8 | every unit type (empty if series missing) | count, %, max +, max − |

A missing `T_outdoor (DB)` **column** does not abort the whole calculation: Tsup, dTreturn and flow are still computed; DB stays empty. WB is still scored if `T_outdoor (WB)` exists, because its band is derived from the DB setpoint, not the DB series. A missing DB **setpoint** (unresolvable letter × climate × application) aborts the whole entry only for air-source entries whose file **does** carry `T_outdoor (DB)`; without that column the entry is calculated with DB empty (WB still as above).

Violation: point outside band → count; `% = count / N × 100`.

Two deviation extremes are reported per quantity:

| Statistic | Definition |
|-----------|------------|
| Max + (`dev_max_*_pos`) | \(\max(x - x_{\mathrm{set}})\) — largest excursion above setpoint |
| Max − (`dev_max_*_neg`) | \(\min(x - x_{\mathrm{set}})\) — largest excursion below setpoint |

Both keep their sign and are not clamped to zero, so a series that stays below the setpoint reports a negative "Max +". Each cell is coloured red when it passes the transient band.

The single worst-case deviation (`dev_max_*_deviation`) is retained in the database as deprecated; the table shows Max + / Max −.

### 5.1 Guideline-interval checks (Guideline Windows)

The **Deviations** page still scores the **whole parent window** with §5 / §8 / §11. Guideline Windows scores **saved** guideline clocks only. Colouring on both pages is a reading aid, not a stored pass/fail (§10). Missing series or a window that was not saved is **n/a**, never `0`.

**Individual %** uses the same rule as the parent: share of samples outside the band, `count / N × 100`, but **every half-width on this page comes from `flask_app/config/interval_deviations.json`** — H, D and S each carry their own keys and none of them reads `permissible_deviations.json`. Equilibrium and evaluation reuse Interval **H**'s keys (they are clocks inside H) and keep their own `dtreturn_k`. The Deviations page has no band editor at all any more — it only *shows*, read-only, the three full-cycle mean bands it colours — so nothing on that page can move these percentages or the interval steps on the Deviations plot. There is **no Tsup %** on Guideline Windows: the draft outlet check is a **mean** ±0.5 K, not an individual sample band. Parent Deviations still stores full-cycle Tsup %.

\(T_{\mathrm{return,calc}}\) is the **set** liquid-sink inlet. Therefore

\[
\mathrm{dTreturn} = T_{\mathrm{return,emu}} - T_{\mathrm{return,calc}}
\]

is measured return minus set on every interval (do not invent a separate return setpoint).

| Check | Window | Individual band (as implemented) |
|-------|--------|----------------------------------|
| H / eq / eval DB, WB, flow | that saved window | DB ±1 K, WB ±1 K, flow ±2.5 % (fixed-flow only) — Interval **H**'s `db_k` / `wb_k` / `flow_instantaneous_pct` in `interval_deviations.json`, **not** the parent `DB` / `WB` / `flow_instantaneous_pct` |
| H / eq / eval **dTreturn** | that saved window | **±0.5 K** (`interval_deviations.json` `dtreturn_k`) — **not** the parent −2…+2 K |
| D individual | union of saved D spans | DB ±5.0 K; dTreturn ±2 K. No D WB %, D flow %, D Tsup % |
| S individual | union of saved S spans | DB/WB ±2.0 K; flow ±2.5 % (fixed-flow); dTreturn ±1 K. No S Tsup % |

Water-to-water skips outdoor DB/WB on these columns too (`checks.json`). A draft cell that is n/a stays n/a.

**Interval means** are a different check: one arithmetic mean of the window against the setpoint, signed, in K (flow in % of set). They are **not** a sample share.

| Mean | Window | Half-width |
|------|--------|------------|
| H mean DB / WB / Tsup / dTreturn / flow | saved H (or `on`) | ±0.3 K / ±0.4 K / ±0.5 K / ±0.3 K / ±1 % of set |
| D mean DB / WB | union of D | ±1.5 K / ±1.0 K |
| S mean DB / WB / flow | union of S | ±0.6 K / ±0.6 K / ±2.5 % of set |

DB / WB / Tsup = \(\bar T - T_{\mathrm{set}}\); dTreturn mean = mean of \(T_{\mathrm{emu}}-T_{\mathrm{calc}}\) against 0 K. Tsup uses `Ts Buh` else `T_supply`. D has no mean Tsup, flow or return; S has no mean Tsup or return; mean \(T_{\mathrm{mean,log}}\) on H/D/S is undecided (n/a). Parent mean Tsup / Tmean / Q on Deviations are unchanged (0.5 K / 0.5 K / 5 %).

**ΔCOP** (evaluation window only):

1. Slice the first and last `delta_cop_slice_min` minutes of the **saved evaluation** window (default 5 min).
2. Each slice: Q and P as on the parent entry (BAM-corr when lab-corr is missing, then BUH-corr); \(\mathrm{COP}=\bar Q/\bar P\) (§3).
3. \(\Delta\mathrm{COP}=(\mathrm{COP}_{\mathrm{last}}-\mathrm{COP}_{\mathrm{first}})/\mathrm{COP}_{\mathrm{first}}\). Compare \(|\Delta\mathrm{COP}|\) to `delta_cop_pct` (2.5 %).

Empty, not 0, when there is no evaluation window, the window is shorter than two slices, or a slice has no Q/P. Not the first/last five minutes of H.

Config: `flask_app/config/interval_deviations.json` — the single source for every interval band above, individual and mean. The numbers happen to match the parent Interval H widths today; they are separate keys and may be retuned on their own. These values are draft configuration, not agreed acceptance limits.

---

## 6. Mean water temperature (Tmean) and flow mode

Ecodesign Table 3 mean water temperature. **Variable-flow tests only.** The draft allows a higher leaving-water temperature provided the Table 3 mean of the test condition is maintained. Fixed-flow tests are judged on Tsup (and on the flow set), not on Tmean.

### 6.1 Definition stored on the entry

Leaving water prefers post-BUH supply (`Ts Buh` / `T_supply_afterBUH` / `T_supply`); return is `T_return_emu`. Indoor air for the logarithm is \(T_{\mathrm{air}} = 20^\circ\mathrm{C}\).

\[
T_{\mathrm{mean,log}} = T_{\mathrm{air}} + \frac{T_{\mathrm{L}} - T_{\mathrm{R}}}{\ln\bigl((T_{\mathrm{L}}-T_{\mathrm{air}})/(T_{\mathrm{R}}-T_{\mathrm{air}})\bigr)}
\]

when \(T_{\mathrm{L}} \neq T_{\mathrm{R}}\) and both differences from \(T_{\mathrm{air}}\) are positive; if \(T_{\mathrm{L}} = T_{\mathrm{R}}\) and \(T_{\mathrm{L}} > T_{\mathrm{air}}\), \(T_{\mathrm{mean,log}} = T_{\mathrm{L}}\). Cycle mean `avg_t_mean_log` is the arithmetic mean of that series over the window. A second value `t_mean_from_avgs` is the same formula on the period-average leaving and return (stationary check).

### 6.2 When Tmean / flow are scored

Profile `flow_mode` wins when it is `variable` or `fixed`. For `per_test` units — and for entries whose profile is unresolved or has no `flow_mode` — the entry `flow_config` (`var` / `variable` vs anything else) decides, defaulting to **fixed** when it is empty.

| Test | Mean Tmean Dev | Flow % / flow violations |
|------|----------------|---------------------------|
| Variable flow | \(\overline{T}_{\mathrm{mean,log}} - T_{\mathrm{mean,set}}\) (letter × climate × application) | **n/a** (not scored; stored flow % is hidden at render) |
| Fixed flow | **n/a (fixed flow)** | scored against the profile flow set (else the legacy `hp_design` row), if a set and a series exist |

\(T_{\mathrm{mean,set}}\) comes from `tmean_set` in the condition-set slice. Colouring uses `mean_tmean_k` (0.5 K).

---

## 7. Which checks apply (`checks.json` / `unit_types.json`)

The unit profile’s `unit_type` is looked up in `unit_types.json` (capabilities: `source_medium`, `has_wetbulb`, `has_gas_input`, …). Each check in `checks.json` applies when every key of `applies_when` matches those flags plus `entry_flow_mode` (`fixed` or `variable` from §6.2).

If the profile or `unit_type` is missing, capabilities default to air-source, so existing databases keep their DB/WB checks.

| Check id | Applies when (today) |
|----------|----------------------|
| `db` | `source_medium = air` |
| `wb` | air + `has_wetbulb` |
| `tsup` | always |
| `dtreturn` | always |
| `flow` | `entry_flow_mode = fixed` |
| `tmean` | `entry_flow_mode = variable` (cycle-mean only; not an instantaneous band) |
| `gas_input` | `has_gas_input` — listed in config; not yet scored as a deviation column |

Water-to-water (`source_medium = water`) therefore has no instantaneous dry-/wet-bulb scores. The DB setpoint may still be stored on the row because it feeds part-load Qset (§9), so **mean DB** deviation still renders when a stored mean exists. **Mean WB** deviation is empty: the WB setpoint is only stored when the WB check applies, and the render path has no on-the-fly WB fallback.

---

## 8. Controllability / dTreturn

**Instantaneous (deviation stats and insights):**

\[
\mathrm{dTreturn} = T_{\mathrm{return,emu}} - T_{\mathrm{return,calc}}
\]

Outside band if \(\mathrm{dTreturn} <\) `dTreturn.lower` or \(\mathrm{dTreturn} >\) `dTreturn.upper` (`permissible_deviations.json`: **−2 … +2 K** on the **parent** window).

Guideline Windows uses the **same difference** on saved H / eq / eval with ±0.5 K, on D with ±2 K, and on S with ±1 K (§5.1). Parent `dev_dtreturn_*` columns are not retuned.

**Mean column `DTret`:** \(\overline{T}_{\mathrm{return,emu}} - \overline{T}_{\mathrm{return,calc}}\) — the same sign convention. It is a derived column whose formula is stored in `column_metadata.json` and re-evaluated whenever an entry is created or updated; it is empty when either input mean is unavailable.

Extra cache statistics (min/max/p99_abs, etc.) support the dTreturn insights page and period splits — complementary to the Permissible Deviations table.

**TBD (guideline / T4.3):** Whether acceptance uses full-cycle dTreturn %, H/eq/eval only, or a formal pass/fail flag. The tool currently shows both the parent band and the interval columns.

---

## 9. Part-load Qset (display)

Not an instantaneous band. On the Deviations page:

\[
UA = \frac{P_{\mathrm{design}}}{T_{\mathrm{set,indoor}} - T_{\mathrm{design,MT}}},\quad
Q_{\mathrm{set}} = UA \times (T_{\mathrm{set,indoor}} - T_{\mathrm{db,set}})
\]

with \(T_{\mathrm{set,indoor}} = 16^\circ\mathrm{C}\) and \(T_{\mathrm{design,MT}} = -10^\circ\mathrm{C}\) (code constants). \(P_{\mathrm{design}}\) and the flow set come from the unit profile, else from the legacy `hp_design` table row for the entry’s `hp_id`. Mean Q deviation preference order is BUH-corrected (`QCorrwBUH` / `avg_heating_capacity_corr_wbuh`), then BAM-corrected, corrected, uncorrected.

---

## 10. Red / yellow / “pass–fail” today

**No stored boolean pass/fail flag** exists on the entry. Colouring is a UI aid, not an acceptance decision; exportable flags are planned when the acceptance rules are agreed.

UI badges from `permissible_deviations.json`:

| Display | Rule |
|---------|------|
| Instantaneous % and violation counts | Still calculated and stored per entry, but **no longer columns on Deviations** — the draft scores those sample shares by interval (§5.1), not over the whole cycle |
| Max + / Max − (Deviations, behind **Show extra columns**) | red if beyond the transient band (`DB/WB.value`, `dTreturn.lower/upper`, `flow_instantaneous_pct`); else green |
| Mean Tsup / Mean Tmean (Deviations) | green if \|mean dev\| ≤ `mean_tsup_k` / `mean_tmean_k`, red otherwise. **Two colours, no yellow and no half-limit step.** Mean Tmean only on variable flow |
| Mean Q % (Deviations) | green if \|dev\| ≤ `mean_q_band_pct` (±5 % of Qset), red otherwise. The older `mean_q_pct_red` / `mean_q_pct_yellow` pair is no longer read |
| Mean DB / Mean WB / Mean flow % (Deviations) | **Plain numbers, never coloured.** The draft defines no full-cycle permissible deviation for them; the interval checks are on Guideline Windows and Period Statistics (§5.1). Hover names the set that was subtracted |
| Interval means (Period Statistics, Guideline Windows) | green inside the draft interval mean band, red outside — the same two-colour rule. Hover names the signed deviation, the set and the band |
| **N/A** | Calculated, but the quantity has **no valid points** in the window, **or the check does not apply** (water-to-water **instantaneous** DB/WB; variable-flow flow %; fixed-flow Tmean). Not 0. Water-to-water still shows **mean DB** (setpoint stored for Qset) and shows **-** for **mean WB** (WB setpoint not stored). |
| **0** | Measured (or scored), and no violations |
| **-** | Not yet calculated |

---

## 11. Limits summary (`permissible_deviations.json`)

| Key | Current value | Role |
|-----|---------------|------|
| `DB.value` | 1.0 K | Instantaneous DB half-width |
| `WB.value` | 1.0 K | Instantaneous WB half-width |
| `Tsup.value` | 0.5 K | Instantaneous Tsup half-width |
| `dTreturn.lower` / `upper` | −2.0 / +2.0 K | Controllability band |
| `mean_tsup_k` / `mean_tmean_k` | 0.5 / 0.5 K | The two mean-temperature bands the Deviations table colours |
| `mean_q_band_pct` | 5 % | Mean Q % band the Deviations table colours, and the Q band on Scatter |
| `mean_db_k` / `mean_wb_k` / `mean_flow_pct` | 0.6 / 0.4 K / 1 % | **Scatter plot bands only.** Nothing on Deviations is coloured against them |
| `flow_instantaneous_pct` | 2.5 % | Instantaneous flow band |
| `db_pct_red` / `db_pct_yellow` | 5 / 1 | DB % colouring |
| `wb_pct_red` / `wb_pct_yellow` | 5 / 1 | WB % |
| `tsup_pct_red` / `tsup_pct_yellow` | 5 / 1 | Tsup % |
| `dtreturn_pct_red` / `dtreturn_pct_yellow` | 2 / 1 | dTreturn % |
| `flow_pct_red` / `flow_pct_yellow` | 5 / 1 | Flow % |
| `mean_q_pct_red` / `mean_q_pct_yellow` | 10 / 5 | Leftover keys, no longer read by any page |
| `*_violations_red` | 0 | Count → red if above |

These are the values shipped with v0.1 for the **parent** Deviations page. They are configuration, not agreed acceptance limits — see the TBD below.

Interval individual and mean half-widths, and ΔCOP, live in `flask_app/config/interval_deviations.json` (§5.1). Do not copy those keys onto `permissible_deviations.json` or the parent table will silently change.

### Setpoints and unit parameters

| Source | Where |
|--------|-------|
| Outdoor DB, Tsup, Tmean per letter × climate × application | `flask_app/config/condition_sets.json` |
| In-memory `DEVIATION_SETPOINTS` | Reloaded from that file as the **union of all climate × application slices** (safe: `tdb_set` per letter is the same in every slice) |
| In-memory `TSUP_SETPOINTS` | Reloaded from the **Average/MT** slice only; this is what the BUH gap-fill uses (§4.3). Deviation Tsup/Tmean setpoints for an entry use the resolved climate × application. |
| `config.json` → `t_supply_set_mapping` | Alias kept in sync; do not edit as the master |
| WB setpoint | Hardcoded \(T_{\mathrm{db,set}} - 1\,\mathrm{K}\); `wb_rule` is not read by the deviation code |
| Backup-heater cap, \(P_{\mathrm{design}}\), flow set, `flow_mode`, `unit_type` | Unit profiles under `flask_app/config/profiles/` (`local/` is not versioned). \(P_{\mathrm{design}}\) and the flow set also fall back to the legacy `hp_design` table by `hp_id`. |
| Which checks run | `unit_types.json` + `checks.json` |
| Interval individual / mean bands, ΔCOP | `flask_app/config/interval_deviations.json` |
| Guideline clock lengths | `flask_app/config/cycle_periods.json` |

**TBD:** Align the bands with the final round-robin guideline / EN table wording. Make the derived WB rule editable per condition if a campaign needs a different wet-bulb offset.

---

## 12. Open gaps (tool state)

1. **Entries can have empty deviation statistics** until they are calculated — not a formula issue; backfill from the Permissible Deviations page. The whole-table calculation uses stored metadata for the letter; the per-column Update currently derives the letter from the filename only and does not read the stored climate/application (§5).
2. Instantaneous Tsup statistics are still computed on the **parent** window (`dev_tsup_*`). That share is often large because D/S legitimately leave the H band. Guideline Windows does **not** report Tsup % (draft outlet check is a mean). Individual % on H/eq/eval is implemented for DB/WB/flow/dTreturn (**TBD** which of these is acceptance).
3. **Sign convention:** resolved — instantaneous dTreturn and the mean column `DTret` both use `emu − calc`.
4. **No formal pass/fail export flags** — UI colouring only; the summary-export flag columns stay empty in v0.1.
5. **A missing or inapplicable quantity** is **N/A**, not 0 %.
6. **`gas_input`** is declared in `checks.json` for hybrids; it is not yet a deviation column.
7. **Qset indoor / design temperatures** are still code constants (16 °C / −10 °C MT), not the nested condition-set `tdesignh`.
8. Hybrid / water-to-water **import maps** are still waiting on the REV1 templates. Analysis of a water-to-water *profile* (skip instantaneous DB/WB; mean DB still renders, mean WB does not) is implemented; importing a water-to-water workbook is not (the importer still requires the full `average_columns` set, §2).

---

## 13. Explicit TBD (guideline / T4.3 / consortium)

- [ ] Official analysis period definition (complete cycle vs steady / on sub-period)
- [ ] Official COP and mean-value column set for reporting
- [ ] Whether instantaneous % (especially Tsup) is an acceptance criterion on full cycles vs H/eq/eval; Guideline Windows has no Tsup %
- [ ] Pass/fail export flags and their thresholds — planned when the acceptance rules are agreed
- [ ] RH→WB conversion rules, if they are needed
- [ ] Multi-sheet template mapping for hybrid and water-to-water units
- [ ] Whether the final consortium Analysis Guide is this document or a separate deliverable that cites it

Operational instructions are in [`USER_GUIDE.md`](USER_GUIDE.md); analyst orientation is [`ANALYST_QUICK_GUIDE.md`](ANALYST_QUICK_GUIDE.md).

---

*Revision note: v0.1, 2026-09-03. Method statements were checked against the shipped code before circulation. 2026-09-17: guideline-interval windows, dTreturn widths, interval means and ΔCOP added as implemented (§1, §5.1, §8).*
