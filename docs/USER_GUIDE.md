# User Guide — T4.2 import and analysis tool

**Version:** v0.1
**Date:** 2026-09-03
**Status:** Draft for T4.2 circulation (v0.1)
**Audience:** T4.2 analysts and validators
**Scope of this release:** air-to-water templates only (import map `hpt_aw_rev1`); HPT unit profiles are placeholders until real unit data is confirmed.

**Start here for a single analysis:** [`ANALYST_QUICK_GUIDE.md`](ANALYST_QUICK_GUIDE.md). That guide is the short "one analysis, start to finish" path. **This document is the full operational reference** — configuration, import detail, profile assignment, export and troubleshooting.

**Companion documents:** methods, formulas and limits are in [`ANALYSIS_GUIDE.md`](ANALYSIS_GUIDE.md); installation summary and repository layout are in the [README](../README.md).

> **Content rule:** this guide describes only how to operate the tool. It contains no unit identities, unit parameters, test file names or measured results from any campaign. Screenshots, if added later, must be taken with example data only.

---

## 1. What this tool does

Two steps, deliberately separate:

```
Lab template workbook              Step 1: import_hpt        Step 2: WebApp
(lab fills the WP2 template)  -->  normalise to Plotdaten -->  analyse, deviations, export
                                   (Rohdaten/ → Plotdaten/)    (one column contract)
```

The reason for two steps: the WebApp understands **one** column contract (Plotdaten). Import is what translates a lab template into that contract. The WebApp is never taught to read raw lab templates.

| You have | Use |
|----------|-----|
| Lab-filled WP2 template workbook (air-to-water) | `import_hpt.ipynb` — this guide |
| Legacy BAM single-sheet Plotdaten files | The WebApp reads them directly; the legacy BAM import path is not part of this release and is not covered here |

One codebase serves both paths. There is no separate BAM build and no fork.

---

## 2. Install and start (once per machine)

Requirements: **Git**, **Python 3.11+** (developed on 3.13), **Jupyter** for the import notebook, and access to the T4.2 tool repository.

**Get the tool (once):**

```powershell
git clone https://github.com/BAMresearch/bam-hpt-analysis-webapp.git
cd bam-hpt-analysis-webapp
pip install -r requirements.txt
```

The clone folder **is** your working directory — the folder that contains `flask_app/`, `import_hpt.ipynb`, `import_tools/`, `Rohdaten/` and `Plotdaten/`. Do not wrap it in a second project folder.

If you already have a clone, `cd` into it and run `git pull`; re-run `pip install -r requirements.txt` only after a major update.

Lab workbooks and analysis databases are **not** in the repository. They stay on your PC (or come from SharePoint) and are excluded from version control.

**Start the WebApp (each session):** change into `flask_app` first, run the launcher, then open the local URL:

```powershell
cd bam-hpt-analysis-webapp\flask_app
.\run_app.ps1                   # or double-click run_app.bat in this folder
```

Leave that window open. When it prints `Running on http://127.0.0.1:5001`, open **http://127.0.0.1:5001** in a browser on this PC.

`127.0.0.1` always means this computer; port `5001` is fixed in `app.py`. Every machine uses that same local address — it is not a shared server. `run_app.ps1` / `run_app.bat` set `PYTHONUTF8=1` before launching `app.py`. A bare `python app.py` on Windows can fail with `UnicodeDecodeError`.

**First-run configuration:** if `flask_app/config.json` does not exist, the app falls back to `flask_app/config_template.json`, which already points `data_dir` at `../Plotdaten` — correct when you start from `flask_app/`. Copy the template to `config.json` only when you need local overrides; `config.json` is local and not versioned.

---

## 3. Step 1 — Import: template → Plotdaten

Open `import_hpt.ipynb`. **The working directory must be the clone root**, because the notebook resolves `Rohdaten/`, `Plotdaten/` and `import_tools/maps/` relative to it. Start Jupyter from that folder.

Run the first cell. It prints the working directory, the available maps, and the two data folders — if any of those look wrong, stop and fix the working directory before going further.

Run the UI cell. The controls, in the order you use them:

| Control | What it does |
|---------|--------------|
| **Rohdaten:** | Dropdown of `.xlsx` files found in `Rohdaten/` |
| **Or path:** | Absolute path, if the workbook is not in `Rohdaten/` |
| **Refresh file list** | Re-scan `Rohdaten/` after adding a file |
| **Load sheets** | Opens the workbook and lists its worksheets. Only lists them — no mapping happens yet |
| **Sheets:** | Multi-select of worksheets to export. **Nothing is pre-selected**; Ctrl/Shift-click to choose |
| **Template:** | Which template family (import map) to apply — see §4 |
| **Prefix:** | Output filename prefix. Pre-filled from the workbook name (`HPT_RRT1_Lab01.xlsx` → `HPT_RRT1_`); it must end with `_` so files become `{prefix}{sheet}.xlsx`, e.g. `Plotdaten/HPT_RRT1_E.xlsx`. Edit it if the unit profile id differs |
| **Dry-run: no files written** | On = report only, nothing written. Leave on for the first pass |
| **Skip empty sheets** | Skips worksheets with no data rows |
| **Export!** | Runs map → normalise → validate, and writes unless dry-run |

The intended sequence, which the notebook also prints:

> 1) Pick file  2) Load sheets  3) Select sheets  4) Check prefix  5) Dry-run, then uncheck dry-run to write

Two behaviours worth knowing because they are easy to misread:

- The **template map is applied at Export**, not at *Load sheets*. Changing the template and pressing *Load sheets* again changes nothing.
- **Only selected sheets are exported.** An empty selection exports nothing and says so.

### 3.1 Reading the report

`Export!` prints one block per sheet, for example:

```
=== E rows=<n> ok=True ===
```

| Line | Meaning | What to do |
|------|---------|------------|
| `ok=True` | Plotdaten contract satisfied | Proceed |
| `missing required sources` | A column the map expects was not found in the template | Check the lab's header spelling; add an alias to the map (§4.2) |
| `missing after normalize` | Contract gap that survived mapping | Must be fixed — the WebApp needs these |
| `required all-null` | Column exists but is entirely empty | Usually a paste error in the template |
| `fills` | Columns the map synthesised | Should normally be empty — see §5 |
| `warnings` | Non-fatal notes | Read them |

When the report is clean, uncheck **Dry-run** and press **Export!** again. It prints `WROTE <path>` per sheet and a final `Done. wrote=… skipped_empty=… dry_run=False`.

---

## 4. Choosing the template — the only unit-type-specific step

> **This section is the one place where the unit type matters.** Everything else in this guide is the same for every unit type. When hybrid and water-to-water templates arrive, this section gains rows; no other section changes.

| Template family | Map id | Status in v0.1 |
|-----------------|--------|----------------|
| Air-to-water, template REV1 | `hpt_aw_rev1` | **Shipped — the only option** |
| Hybrid | — | Not yet written — awaiting the REV1 template |
| Water-to-water | — | Not yet written — awaiting the REV1 `_sink` / `_source` template |

The **Template:** dropdown lists only the maps that actually exist in `import_tools/maps/`, so hybrid and water-to-water appear there when their maps are added.

If you pick the wrong template, the symptom is a long `missing required sources` list, because the template's headers do not match what the map is looking for. That is a safe failure — nothing is written in dry-run.

### 4.1 What the air-to-water REV1 map does

| Topic | Behaviour |
|-------|-----------|
| Outdoor DB | `T_outdoor (DB)` only; lab air temperature stays a separate column |
| Sink → WebApp | `T_out_sink`→`T_supply`, `T_in_sink`→`T_return_emu`, `volume flow_sink`→`volume flow` |
| Pressure difference | kPa → bar (`× 0.01`) |
| Virtual Back-up | → `Electrical power input BUH` (this is the WebApp's BUH) |
| Backup heater Input (optional) | → `Real BUH power input` — a monitoring column only; **never** used as the WebApp BUH |
| No Virtual Back-up | The BUH column is **omitted** — zeros are not invented |
| Mass flow, corrected Q/P, either back-up | Passed through only if the template contains them **and** they are not empty |
| Sink properties | `cp_out_sink`, `cp_in_sink`, `density_sink` keep the `_sink` suffix, so water-to-water can later add the matching `_source` columns without renaming anything |
| Other optional columns | Lab air / atmospheric pressure, compressor frequency, fan speed, voltage, current, frequency, cos phi, condenser and evaporator temperatures are kept even when the lab left them empty, so the field stays visible in Plotdaten |

Default sheets are `E, F, A, B, C, D`. Sheets `B, C, D` have no back-up headers at all — that is normal, and both back-up columns are simply absent from their Plotdaten files.

### 4.1.1 The two back-up heater columns

The template has two different back-up quantities, and they must not be mixed up:

| Template column | Plotdaten column | Already inside `Total Power Input (without corr)`? | Does the WebApp add it to power / heat / COP? |
|-----------------|------------------|--------------------------------------|-----------------------------------------------|
| **Virtual Back-up** | `Electrical power input BUH` | No — it comes on top | **Yes** — this is the existing BUH path |
| **Backup heater Input (optional)** | `Real BUH power input` | Yes — it is already measured in the total | **No** — stored as a monitoring mean only |

A test normally uses one or the other. If a workbook has data in both, the importer keeps both columns and prints a warning rather than merging or dropping either — check with the lab which back-up was actually active.

### 4.2 Small map edits

Most template changes are a JSON edit, not a code change. Edit the map in `import_tools/maps/`, bump its `version`, re-run. The field-by-field reference is `import_tools/maps/README.md`.

| Situation | Edit |
|-----------|------|
| Lab renamed a column | Change `source`, or add the old name to `aliases` |
| Extra column you want to keep | Add a `columns` entry, or rely on `passthrough_unmapped` |
| Unit already converted upstream | Remove or neutralise that entry's `scale` |

### 4.3 Adding a new unit type later

1. Copy `hpt_aw_rev1.json` to `hpt_<family>_rev1.json` and set `map_id` to match the filename.
2. Rewrite `columns` against the **real** template headers. Do not guess names from the air-to-water map.
3. Set `webapp_required_columns` to what that family genuinely provides.
4. Add any new column to the WebApp via **Manage Columns** (§6.6) — the results table takes new columns without a code change.
5. Add a row to the table in §4 and a condition set in `flask_app/config/condition_sets.json` if the test conditions differ.

---

## 5. What the importer will *not* do

These are deliberate and should not be "fixed" by adding fills back into a map:

| The importer never | Why |
|--------------------|-----|
| Invents mass flow | A derived value must not look like a measurement in the archive file |
| Copies uncorrected Q/P into the corrected columns | Would silently present uncorrected data as lab-corrected |
| Writes `Electrical power input BUH = 0` | A false zero column makes the WebApp run BUH logic as if a backup heater existed |

The WebApp handles these gaps at analysis time instead, and **tells you** when it does — see §6.4. The principle: gaps stay visible in the data and are resolved in the open, per analysis.

---

## 6. Step 2 — WebApp

Nav bar, left to right, roughly in workflow order.

### 6.1 Datasets — pick the database first

`/datasets`. Each dataset is a separate analysis database. Create one per campaign or per unit; switch between them here. **Do this first** — everything below writes into whichever dataset is currently open, and the active dataset is shown at the right of the nav bar.

### 6.2 Profiles — check unit and conditions

`/profiles`. **Unit profiles** hold per-unit parameters (backup-heater cap, design capacity, flow set, flow mode, unit type) and the default climate and application. **Condition sets** hold the setpoints per test-condition letter.

The default condition **set** is Ecodesign Table 3 heating conditions (`ecodesign_heating`; the legacy alias `BAM_RRT` is still accepted). Average / MT is the default climate × application **slice** inside that set, not the name of the set.

Check this before analysing a new unit. A wrong or missing profile shows up later as implausible deviations, because the setpoints the deviations are measured against came from the wrong place.

Shared profiles — including the HPT placeholders `HPT_RRT1` … `HPT_RRT6_STAGED` and the `EXAMPLE_*` profiles — are versioned in `flask_app/config/profiles/`. Profiles you create for real units belong in `flask_app/config/profiles/local/`, which is deliberately excluded from version control so unit parameters never reach the repository.

**Unit profile and HP ID**

- **Select a unit profile.** It is **required** on Cycle Extraction and on the new-entry review modal. This is what gives a file its unit identity.
- **Leave HP ID blank.** The tool auto-fills it from the profile (trailing digit, or the first numeric alias). Filename patterns can also attach a profile when the file name matches. Do not invent numeric unit IDs — assign a profile instead.
- Legacy databases whose entries carry a numeric HP ID stay valid; nothing is rewritten, and profiles can be backfilled later.

Scatter and Period Statistics group by **unit profile**, falling back to HP ID for legacy rows that have not been backfilled, and to the file name if neither is set (each unresolved file is then its own series, not merged). If the same HP ID appears under two profiles in one database, labels switch to the profile name so the series are not mixed.

An empty **climate / application** on an entry means the Average / MT **default** slice of the Ecodesign heating condition set. Scatter shows that resolved slice; if more than one slice is present, Scatter offers climate/application checkboxes.

### 6.3 Cycle Extraction — turn a Plotdaten file into entries

`/cycle_extract`. This is the normal way to create analysis entries from an imported file.

1. Pick the Plotdaten file.
2. Define cycles — the tool suggests candidates, and you can add cycles manually. **Suggest cycles** still cuts the file **drop to drop** and **Min gap** still works the same way, but on a defrost file (test condition A/B/E/F) a suggested window in which no defrost actually happened — supply − return never stays below 0.2 K for 60 s — is left out; a C/D or unlabelled file, a file without those temperature columns, and the case where that would leave nothing to suggest all keep the full list.
3. **Select a unit profile** (required). HP ID fills from the profile if it was empty.
4. **Set `ds` per cycle.** `ds` is the dataset number stored in the database for that cycle. It defaults to 1, 2, 3, … down the table and is editable per row. **Set it before pressing Apply** — it is per cycle, not one value for the file.
5. Apply. Entries appear on the main results page. The review modal shows the auto-filled Test cond, Lab ID, HP ID, Indicator and Notes so you can correct anything before Confirm. **Test label** is filled automatically from Test cond and Flow (for example `B fixed`) and is not shown on the modal. Indicator and Notes are optional reminders of how the cycle start was chosen; they are not required for Period Statistics. HP ID is filled from the profile when it was empty.

**Apply also stores the power-transition time** of each new cycle. On a defrost cycle the end of the defrost is the **last** time supply − return stays above 0.2 K for 60 s — not the first flicker, and not the compressor ramp-up on its own. A cycle the tool cannot read leaves the time empty (never `0`), and you can still type it. **Apply also stores default D / H / equilibrium / evaluation clocks** for each new cycle (on a continuous window, H only), so the sub-periods are there as soon as the entry is — you no longer have to open **Edit clocks** first. They are a starting point, not a confirmation: use **Edit clocks** in §6.3a to change them and **Save** to confirm them. A window the tool cannot classify, and one it cannot judge — a defrost letter with no usable supply / return series, or an on–off letter with no readable power — gets **no clocks at all**; the entry itself is stored either way. The test-condition letter is only a hint about the window you extracted: a window whose supply − return never stays **below** 0.2 K for 60 s is treated as **continuous** — an A letter cannot invent a defrost out of a compressor dip — and a C window with no compressor stop is treated as continuous too. Either way the time stays empty, which is not an error.

### 6.3a Guideline clocks on existing entries

Also on `/cycle_extract`, for the file you have loaded. The database entry **stays the complete cycle**. These clocks are sub-periods on that row, not a replacement. Interval definitions and bands are in the [Analysis Guide](ANALYSIS_GUIDE.md) §1 and §5.1.

**Apply** already stored the default clocks (§6.3). **Edit clocks** opens them for changing; **Save** stores what you see and marks it as yours.

It shows D or S, H, and — on defrost and continuous — equilibrium then evaluation as editable handles, and writes nothing until you Save. Re-applying or recalculating a row that already has clocks leaves them exactly as they are: to redo them, either **Clear saved guideline clocks** and Apply again, or **Edit clocks** and **Save**. Evaluation is **not** the last minutes of the cycle; it is omitted when H is too short. Lengths: `flask_app/config/cycle_periods.json` (defaults: buffer 10 min, equilibrium 60 min, evaluation 70 min).

Kinds: **defrost**, **on–off** (no equilibrium or evaluation), **continuous**, or **unknown**. Notes `drop` / `ramp` are not required. Stored `other` is unknown, not continuous. Confirm before you save.

The test-condition letter is a hint, not a verdict on the window: an A window that never holds supply − return **below** 0.2 K for 60 s, and a C window with no compressor power stop, come back as **continuous** (H is the whole parent window, equilibrium and evaluation run from its start, no D or S). The kind source on the row says so. You can still **set the kind yourself** on Edit clocks, and a kind you set, a defrost-indicator column, clocks already saved with a D or S, and a transition time already on the row are never changed this way. A window the tool cannot judge — no usable supply / return series — keeps its kind rather than becoming continuous.

**Treat unknown rows as** fills only rows that still have no kind. **Save all clocks** stores the open file (tick **show** to limit the batch). Unknown rows with no default are skipped.

**Edit** one row: move the power-transition time or D/S end to rebuild H (and locked equilibrium / evaluation). A cycle cut from one defrost **end** to the next can show a **second D** (same for S); type a start, clear it, or **reset**. **Unlock** equilibrium or evaluation to type their times. **Save clocks** writes on the parent entry.

Period layers are **off by default**. Tick **show** to draw the **saved** clocks — no click on **Edit clocks** needed — or the clocks you have open on that row, on the existing chart: the open clocks while a row is being edited, otherwise the **saved** bands. A row marked *≠ saved* has open clocks that do not match storage; nothing is written until Save. A second D/S start you typed is kept on the next **Edit clocks** after Save.

The power-transition time and the default clocks are already on the row after Apply (§6.3). **Edit clocks** reuses the time and never overwrites it; where it is still empty it fills the preview only, using the same rule, and shows where it came from. Save is still what stores the clocks.

**Save clocks for the open database** fills parent rows that still have no clocks (asks first, one file at a time) — mostly older entries, since new ones are filled at Apply. It never overwrites clocks that are already stored. To change clocks you can see, open the file, **Edit clocks**, and **Save all clocks**. It does not send edits that are open on the loaded file.

**Clear saved guideline clocks** removes those times and their stored scores so you can fill again. It does not delete parent cycles, their means or deviations. Period Statistics reads these clocks, so a cleared entry shows `n/a` there until you fill again. Duplicate the database on **Datasets** first if you may want that pass back.

Guideline Windows and the Deviations Plot use **saved** clocks only. There is no Tsup sample-band on Cycle Extract.

### 6.4 Reading the import fallback notices

After Apply, the Means page (and the review modal) may show fallback notices. These are the WebApp telling you it filled a gap the importer refused to invent. They are informational, not errors — but they change how the numbers should be read.

Notices are **collapsed by file and kind**, not by exact text. Three cycles of the same file with missing lab-corrected Q/P produce **one** warning, with the cycle count N, not three banners that differ only in `Q=… kW`. Two files in one session produce two groups. Raw per-cycle strings are still collected internally (capped at about 20); only the collapsed lines are shown. Text wraps; the list is dismissible and scrolls if it is longer than about four lines. The same collapsed list is shown at the top of the review modal so it is readable while you confirm.

| Notice | What actually happened | What to do |
|--------|------------------------|------------|
| *"{file}: lab-corrected Q/P missing on N cycles — stored BAM pump-corrected averages."* | The `corrected` fields hold **BAM pump-corrected** values, not lab-corrected ones | Do not compare these `corrected` figures against genuinely lab-corrected values from another unit without saying so |
| *"{file}: mass flow not reported — derived from volume flow × …"* | The file had volume flow but no usable mass flow, so mass flow was computed as `volume flow / 3600 × ρ`. The notice names ρ: the measured `density_sink` (HPT) or `density` (BAM) when the lab reported one, otherwise 997 kg/m³ | Acceptable for analysis. Note it in any report; ask the lab for measured mass flow in future deliveries |
| *"Mass flow: N of M rows were empty and derived …"* (shown as the derived line above when collapsed) | Only the gaps were filled. Measured values, zeros included, are unchanged | Check why the lab's mass flow has holes if N is large |
| *"{file}: Mass flow: measured values from the file used (no derivation)."* | Nothing was derived. **Info**, at most once per file | Nothing — this is the normal case |
| *"Density column 'density_sink' has … value(s) outside 900–1100 kg/m³ …"* | The density looks like g/cm³ (0.997) or is otherwise not water; those rows used 997 kg/m³. Warning, once per file | Ask the lab to deliver density in kg/m³ |
| *"Both Virtual Back-up and Backup heater Input carry data …"* | The file has data in both back-up columns. Warning, kept as-is | Check with the lab which back-up was actually active |
| *(no notice, console only)* `Column 'Electrical power input BUH' absent — has_buh=False` | No backup-heater column, so BUH logic is skipped entirely | Expected for units without a backup heater |

An **all-zero** BUH column is treated exactly like an absent one. That is intentional: a column of zeros from a template placeholder is not evidence of a measured zero.

Only `Electrical power input BUH` decides this. A file that has `Real BUH power input` but no `Electrical power input BUH` still gets `has_buh = False` and no BUH add-on, because the real heater's power is already inside the measured total.

### 6.4a Optional monitoring means

The air-to-water REV1 template carries monitoring channels that are not part of the analysis contract: lab air temperature and atmospheric pressure, the sink `cp` and density, compressor frequency, fan speed, voltage, current, frequency, cos phi, the real back-up heater power, and the condenser / evaporator temperatures. Each one that is present in the Plotdaten file is averaged over the cycle window and stored next to the other means.

On the **Mean Values** page they are **hidden by default**. Tick **Show monitoring columns** (next to the results table) to show all 16 extras, or use **Manage Columns** for a per-column override. The checkbox writes the same `column_visibility.json` as Manage Columns — it does not hide identity, energy or state columns (Blocks A–C), and it does not invent a second settings file. They can also be ticked into the summary export under **Optional monitoring (A/W)**.

Three properties are worth remembering:

- **Optional means optional.** A Plotdaten file without them imports and calculates exactly as before. This includes every existing BAM RRT file.
- **Absent is `n/a`, not `0`.** If the column is missing, or has no numeric value inside the window, the field stays empty. Nothing is invented.
- **They are not judged.** No permissible deviation, colouring or pass/fail rule reads these columns; they are documentation of the test conditions.

### 6.5 Analysis pages

| Page | Use |
|------|-----|
| **Data** (`/`) | The results table; add, edit, recalculate, plot entries. Identity, energy (including `QCorrwBUH` / `Ts_buh`) and state columns are visible by default; monitoring extras are off until **Show monitoring columns** |
| **Deviations** (`/deviations`) | Permissible deviation analysis on the **full-cycle parent** window (H+D or H+S), as cycle **means** against the setpoints. Interval (D / H / S) scores are on **Guideline Windows** and **Period Statistics**, not here. Bands are not edited on this page: **Permissible deviations (this page)** is a read-only card listing the three full-cycle mean bands the table colours — liquid sink outlet (Tsup) ±0.5 K, liquid sink mean (Tmean, variable flow) ±0.5 K and heating capacity Q ±5 % of Qset. Change them in the committed configuration files (§10). Unit parameters (Pdesign, flow mode, fixed-flow set value) are on **Profiles**. Coloured cells: **Mean Tsup Dev**, **Mean Tmean Dev** (variable flow only) and **Mean Q dev (%)** — green inside the band, red outside, no yellow. **Mean DB Dev**, **Mean WB Dev** and **Mean flow dev (%)** are plain numbers: the draft sets no full-cycle band for them; hover says so. There is no **Setpoint** column — hover the **Test Condition** badge for the letter and the climate / application slice, and hover any mean cell for the set that was subtracted. The per-quantity **Violations** and **%** columns (DB, WB, dTreturn, flow) are no longer on the table; they are still calculated and still stored. **Show extra columns**, off on every load, keeps **Total Points**, the **Max ±** excursions and **Avg Timestep (s)**; **Actions** (Plot) stays visible. **Calculate Statistics** and the per-column **Update** buttons are unchanged. **Plot** draws DB/WB, Tsup, dTreturn and flow in one modal. The table scrolls inside its own box; the column titles and the identity columns up to **Duration (s)** stay put. There is no individual **Tsup %** column: the draft gives the liquid-sink outlet a mean band only, so Tsup is judged by **Mean Tsup Dev (K)** and by the Tsup plot |
| **Period Statistics** (`/period_statistics`) | Sub-period means within an entry (D / H / D+H, Off / On / Off+On). The splits are the clocks saved on **Cycle Extract** (Apply or **Save all clocks**): 10 min after the defrost **and** 10 min after the restart. **Notes** need no `drop` / `ramp`, and a row whose cycle type is still `other` is listed as soon as it has clocks. An entry with no saved clocks stays `n/a` (never `0`) — fill it on Cycle Extract. A parent that is cut through standby reports **both** S spans: **Off** is their union (durations added, means weighted by duration) and **Off1** / **Off2** show the pieces, the same way **D** / **D1** / **D2** work on a defrost cycle. Those pieces are behind **Show split spans**, off on every load — the unions **D** / **Off** and the full cycle **D+H** / **Off+On** are always shown, and **Export CSV** writes every column whatever the checkbox says. The page shows one **T_mean** (the logged mean); the second *T_mean avgs* column is gone. Means that have a draft **mean** band are read in colour: green inside the band, red outside, with the signed deviation, the set that was subtracted and the band on hover. Coloured are **T_db** / **T_wb** on every interval group, **dT avg** and flow on **H** / **On**, and **Ts Buh**, **T_mean** and **Q** on the full cycle **D+H** / **Off+On** only. Everything else keeps the plain number, including **Ts Buh** on H / On — the outlet is a full-cycle mean check, not an interval one. A cell stays uncoloured (never red) when the mean is missing, when the test condition has no setpoint, and on the flow columns of a variable-flow test; hover says which. **Export CSV** is unchanged — numbers only, no colour. There is no buffer choice on the page; an older database that has no guideline clocks still shows its previous splits. The table scrolls inside its own box; the group and metric header rows and the identity columns up to **clocks** stay put. Above it, three dropdowns (unit, test condition, COP dataset) hide rows in the browser beside the **All / Defrost / On–Off** tabs — there is no filter box per column any more |
| **Guideline Windows** (`/guideline_windows`) | Read-only check of **saved** guideline clocks across many entries: times, parent vs evaluation means, interval % / means, eval ΔCOP. The table scrolls inside its own box; the header rows and the identity columns up to **Kind** stay put (§6.5a) |
| **Scatter** (`/scatter`) | Scatter plots across entries. Combinations are unit + flow, not a generic `HP (fixed)`. Climate / application is labelled (Average / MT default) and can be filtered when several slices exist |

Definitions of every statistic on these pages — means, COP, deviation bands, the N/A rule — are in the [Analysis Guide](ANALYSIS_GUIDE.md), not repeated here.

Three operational notes:

- A new entry has **empty** deviation statistics until it is calculated. That is not a fault; use the whole-table calculate, or a per-column **Update** — those buttons sit on the extra columns, so tick **Show extra columns** first.
- **Show split spans** (Period Statistics), **Show extra columns** (Deviations) and **Show mean deviations** (Guideline Windows) only change what you look at. They are off every time the page loads, they write no settings file — unlike **Show monitoring columns** on Data (§6.4a) — and the hidden numbers are still computed, still stored and still in the CSV.
- **N/A** and **0** mean different things. `0` = measured, no violations. `N/A` = calculated, but the quantity had no valid points in the window, or the check does not apply to that unit type / flow mode. `-` = not yet calculated.

**Plot** on Deviations opens one figure per quantity in one modal: outdoor DB/WB (air-source), Tsup, dTreturn, and flow on fixed-flow tests. If guideline clocks are **saved**, the filled band **steps** at those clocks; dashed lines are the parent / full-cycle limits the table scores. The steps come from `interval_deviations.json`; the dashed parent lines come from the parent band file. Neither is edited on the Deviations page (§10). Tsup does not step (mean check only). Interval percentages and ΔCOP are on Guideline Windows (§6.5a). Band widths: [Analysis Guide](ANALYSIS_GUIDE.md) §5.1.

### 6.5a Guideline Windows — checking the saved clocks

`/guideline_windows`. Read-only check of clocks saved in §6.3a, one row per parent. Cycle Extract stays the editor. What the columns mean (bands, ΔCOP, n/a) is in the [Analysis Guide](ANALYSIS_GUIDE.md) §5.1.

Two ways in:

- Tick rows on **Data** → **Guideline windows** in the Selected rows toolbar.
- Open the page from the navbar: it loads every entry in the open database. There is no *Load all* button.

Opening the page does **not** read Plotdaten. Scores are stored when you **save** clocks; they are read back here. You do not need Calculate on Deviations first.

Each row: identity (**#** is the same number as on Data and Deviations, not the internal id; **COP dataset** is `YES` / `NO` / blank), kind, clock times, parent Tsup/Q/P/COP beside the same four on the **evaluation** window (means in °C / kW), then interval individual %, D/S %, eval ΔCOP, and H/D/S mean deviations. Kind comes from the stored periods, never from the test letter. The table scrolls sideways and down inside its own box: the two header rows and the identity block (**#** through **Kind**) stay put.

The page itself carries one short paragraph. What the stored scores are, when a cell is `n/a` and what **Compute missing scores** does is here and on that button’s hover, not on the page.

Filters (Kind, COP dataset, Clocks) run in the browser on rows already loaded. **Clear filters** resets them. Status is **Showing X of Y rows**.

The three **mean deviation** groups (H, D, S — one arithmetic mean per interval against its setpoint, including **H mean Tsup**) start hidden. Hover a filled mean cell for the set it was read against (and the test letter and climate / application slice) together with its band; neither page has a **Setpoint** column. Tick **Show mean deviations** beside the filters for them. The individual **%**, the clock times and **eval ΔCOP** are never hidden, and **Download CSV** keeps every column whatever the checkbox says.

Empty score cells are **n/a, never `0`**: no such window (on–off, short H, unknown, unsaved clocks), or scores not stored yet. Unsaved proposals do not appear. Hover an n/a cell for the stored reason.

Hover a filled **eval ΔCOP** cell for the two evaluation-slice COPs it was made of — the first and last five minutes, each with its window in seconds — and the 2.5 % limit; the same two COPs and their times are extra columns in the CSV, not on screen. A row scored before this was added shows the % without them until its clocks are saved again or **Compute missing scores** runs.

**Compute missing scores** fills rows whose clocks are saved but whose interval numbers are not (typical on an older database, or after clocks or bands changed). It asks first, then reads Plotdaten **one file at a time**. It does not propose clocks and does not change parent Deviations. A sheet that cannot be read is reported; the rest continue. Opening the page and using the filters never starts this.

**Download CSV** writes the **filtered** table. The summary export (§6.7) is unchanged.

### 6.6 Manage Columns and Diagnose Database

`/manage_columns` adds, removes, reorders and hides columns, including derived columns with a stored formula that re-evaluates when entries change. This is what makes new unit types possible without a code change (§4.3).

**Committed defaults** live in `flask_app/column_order.json` and `flask_app/column_visibility.json` (Mean Values column order and which columns start visible). Toggling visibility on Means (**Show monitoring columns**) or on this page **writes those same JSON files on this machine**. The change only reaches git if the analyst commits the files; a fresh clone always starts from the committed defaults, not from another analyst's local toggles. `dev_*` columns and cycle-pipeline fields (cycle type, start marker, Pelec transition) stay off the Means page unless you explicitly show them here.

`/diagnose_database` and the duplicate-cleanup tool are for when the table looks wrong — inconsistent entries, duplicates after a re-import. They are maintenance tools; routine analysis does not need them.

### 6.7 Export

`/export`. Writes a summary CSV or Excel (`t42_summary_v1`) with a selectable column set. Column-selection preferences are stored per machine and are not versioned.

There are currently **no formal pass/fail flags** in the export — the pass/fail columns exist in the summary schema but stay empty, and the red/yellow colouring lives only in the UI. Populating them is planned once the acceptance rules are agreed.

---

## 7. Typical full pass

```text
1. import_hpt.ipynb   Load sheets → select sheets → Template: Air-to-water → check prefix → dry-run
2. import_hpt.ipynb   Report clean → uncheck dry-run → Export!  → Plotdaten/*.xlsx
3. WebApp /datasets       open or create the dataset
4. WebApp /profiles       confirm unit profile + condition set
5. WebApp /cycle_extract  define cycles → set ds per cycle → Apply
6. WebApp /                read the collapsed fallback notices (also inside the review modal)
7. WebApp /cycle_extract  Edit clocks → Save (optional; needed for Guideline Windows)
8. WebApp /deviations      calculate → review (Plot if you want the trace)
9. WebApp /guideline_windows  check saved clocks; Compute missing scores if cells are n/a
10. WebApp /export         summary out
```

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Notebook cannot find `Rohdaten/` or the maps | Jupyter's working directory is not the clone root (§3) |
| Long `missing required sources` list | Wrong template family for the workbook (§4), or the lab renamed headers |
| `Export!` writes nothing | No sheets selected, or Dry-run still ticked |
| Plotdaten files named `HPT_RRT1E.xlsx` instead of `HPT_RRT1_E.xlsx` | Prefix does not end with `_` (§3) |
| `UnicodeDecodeError` on start | Started `app.py` directly without `PYTHONUTF8=1`; use `run_app.ps1` (§2) |
| Deviations all empty | Entry not calculated yet (§6.5) |
| Deviations implausibly large | Wrong unit profile or condition set (§6.2); or a fixed-flow check applied to a variable-flow test (§9 note 1) |
| Guideline Windows scores all n/a with clocks present | Scores were not stored with those clocks — **Compute missing scores** (§6.5a) |
| Scatter: only a generic `HP (fixed)` series, or files merged | Entries have no profile and no HP ID; assign a unit profile (§6.2) |
| Profiles page stale after editing a profile JSON | Reload the page |
| Numbers changed after re-import | Check which dataset is open (§6.1) |

---

## 9. Known limits in v0.1

1. **Fixed-flow checks skip variable-flow tests.** Profile `flow_mode` (or per-entry `flow_config` when the profile is `per_test`) decides. Variable-flow rows show n/a for flow %; they are judged on Tmean / Tsup.
2. **No pass/fail flags in the export.** The columns exist but stay empty; they will be filled when the acceptance rules are agreed.
3. **Hybrid and water-to-water import maps are not available yet** — they await the REV1 templates (§4). Water-to-water *profiles* already skip outdoor-air deviation checks, so the analysis side is ready before the import side.
4. **Cycle auto-detection is optional.** Manual cycle definition is the supported path today.
5. **HPT unit profiles are placeholders.** Design capacity, flow set and backup-heater cap must be replaced with confirmed unit data before results are used for validation.
6. **No assembled per-test report.** Use the analysis pages plus the summary export; a per-test draft report is planned for a later release.
7. **Local use only.** The app runs on `127.0.0.1:5001` on the analyst PC; there is no shared server or multi-user mode.

---

## 10. Where configuration lives

| Setting | File | Edit via |
|---------|------|----------|
| Import column mapping | `import_tools/maps/hpt_*.json` | Text editor; bump `version` |
| Deviation bands and colour thresholds (**full-cycle parent only**) | `flask_app/permissible_deviations.json` | Text editor. **Not** edited from the Deviations page — that page only *shows* the three mean bands it colours |
| Guideline clock lengths | `flask_app/config/cycle_periods.json` | Text editor |
| Interval individual / mean bands and ΔCOP (**H, equilibrium, evaluation, D, S — all of them**) | `flask_app/config/interval_deviations.json` | Text editor. Committed file — no page edits it |
| Test-condition setpoints | `flask_app/config/condition_sets.json` | Profiles page / text editor |
| Unit parameters (Pdesign, flow mode, fixed-flow set, BUH cap, climate, application) | `flask_app/config/profiles/` (`local/` is not versioned) | **Profiles page** |
| Which checks apply per unit type | `flask_app/config/checks.json`, `flask_app/config/unit_types.json` | Text editor |
| Results table columns | `flask_app/column_metadata.json` | Manage Columns page |
| Mean Values column order and visibility | `flask_app/column_order.json`, `flask_app/column_visibility.json` | Manage Columns, or **Show monitoring columns** on Mean Values. Local toggles write these files; they reach git only if you commit them |
| Data directory and local overrides | `flask_app/config.json` (falls back to `config_template.json`) | Text editor; not versioned |
| Which optional monitoring channels are averaged | `optional_average_columns` in `config_template.json` | Text editor. An existing local `config.json` without the key uses the built-in list, so there is nothing to migrate |

---

*Document history: v0.1 — 2026-09-03, first version circulated within T4.2. 2026-09-17 — guideline clocks and Guideline Windows described as operation; methods moved to the Analysis Guide.*
