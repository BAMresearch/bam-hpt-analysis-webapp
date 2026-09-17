# Analyst Quick Guide — one analysis, start to finish

**Date:** 2026-09-17  
**Audience:** HPT T4.2 analysts

This is orientation plus one analysis.

---

## 1. What you are using

Two steps, kept separate on purpose:

```
Lab template workbook  →  import notebook  →  Plotdaten Excel  →  WebApp  →  means / deviations / scatter / export
```

The WebApp understands **one** column layout (Plotdaten). The import notebook translates the lab template into that layout. Do not feed the raw lab workbook to the WebApp.

Air-to-water templates use the **Template** dropdown (Air-to-water). Water-to-water and hybrid options appear there when those maps exist.

---

## 2. Get the tool from GitHub (once per PC)

Clone the project repository. You need **Git** and **Python 3.11+**.

```powershell
git clone https://github.com/BAMresearch/bam-hpt-analysis-webapp.git
cd bam-hpt-analysis-webapp
pip install -r requirements.txt
```

`pip install` is once per PC (or after a major update). Then work **inside the cloned folder** — that folder *is* the working directory.

If you already have a clone, skip to §3. To update: `cd` into it and run `git pull`.

Lab workbooks and analysis databases are **not** on GitHub.

---

## 3. Folders in the clone

After clone, the repository looks like this. Do not add a second working folder around it.

```
bam-hpt-analysis-webapp/
  flask_app/          WebApp
  import_hpt.ipynb    importer
  import_tools/       import maps
  Rohdaten/           put lab template workbooks here
  Plotdaten/          importer writes one Excel per test sheet here
  docs/               this guide
```

| Folder / file | What goes there |
|---------------|-----------------|
| `Rohdaten/` | Lab-filled template workbooks (input) |
| `Plotdaten/` | Normalised one-sheet Excels the WebApp reads (output of import) |
| `import_hpt.ipynb` | Jupyter UI: template → Plotdaten |
| `import_tools/` | Column maps (e.g. `hpt_aw_rev1`) |

Start Jupyter so its **working directory is the cloned folder** (the folder that contains `Rohdaten`, `Plotdaten`, and `import_tools`). If that is wrong, the notebook cannot see the maps or the two data folders.

If `flask_app/config.json` does not exist, the app uses `flask_app/config_template.json`, which already points `data_dir` at `../Plotdaten` (correct when you start the app from `flask_app`).

---

## 4. Start the WebApp (once per session)

One sequence, not two start methods:

1. Open PowerShell, **change into `flask_app` inside the clone**, then run the script:

   ```powershell
   cd bam-hpt-analysis-webapp\flask_app
   .\run_app.ps1
   ```

   `.\run_app.ps1` means “the script in *this* folder”. If you skip `cd`, it will not be found.

   You can instead double-click `run_app.bat` **inside** `flask_app`. Either script sets UTF-8 and then runs `python app.py`. Do not start with a bare `python app.py` on Windows — that often fails with `UnicodeDecodeError`.

2. Leave that window open. When it prints `Running on http://127.0.0.1:5001`, open **that same address** in a browser **on this PC**.

`127.0.0.1` always means “this computer”. Port `5001` is fixed in the app. Every analyst uses **http://127.0.0.1:5001** locally; it is not a shared server.

---

## 5. Why unit profiles exist (read this once)

A **profile** is the unit: design capacity, flow mode, backup-heater cap, and which checks apply. The **condition set** is the guideline (Ecodesign heating setpoints). **Climate / application** is the test slice (Average / MT, Colder / LT, …), not a new profile.

Without a profile, files have no unit identity: scatter series merge or show up as a generic “HP (fixed)”, and deviations can be scored against the wrong setpoints.

| Field | What to do |
|-------|------------|
| **Unit profile** | **Required.** Pick the matching HPT placeholder (`HPT_RRT1` … `HPT_RRT6_STAGED`) |
| **HP ID** | Leave blank; the tool fills it from the profile (trailing digit) |
| **Lab ID** | The lab that ran the test — this is *not* the unit |

Plots and dTreturn group by **profile**, so two units that both get HP ID `1` stay separate series.

---

## 6. One analysis (the only sequence you need)

Do this in order. The open dataset is shown on the right of the nav bar — check it before you Apply.

### A. Dataset

**Datasets** → create or open a database. One dataset per unit or campaign.

### B. Profile

**Profiles** → confirm the unit exists. The shipped names are placeholders; unit type (air-to-water, water-to-water, hybrid, …) is on the profile, not in this list:

- `HPT_RRT1`, `HPT_RRT2`, `HPT_RRT3`
- `HPT_RRT4_HYBRID`, `HPT_RRT5_TRET`, `HPT_RRT6_STAGED`

Empty climate / application means **Average / MT (default)**. Scatter shows that resolved slice; you do not have to type it.

### C. Import (if the Plotdaten file is not already there)

Open `import_hpt.ipynb` with working directory = the cloned folder (§3).

1. Pick workbook in **Rohdaten** → **Load sheets** → select sheets → **Template** Air-to-water.
2. Check **Prefix**. It is filled from the filename (`EXAMPLE_AW1_Lab01.xlsx` → `EXAMPLE_AW1_`, `HPT_RRT1_Lab01.xlsx` → `HPT_RRT1_`).  
   It must end with `_` so files are `{prefix}{sheet}.xlsx` (`HPT_RRT1_E.xlsx`, not `HPT_RRT1E.xlsx`). Edit it if the profile id is different.
3. Dry-run is on by default (no files written). If `ok` looks right, **uncheck** dry-run → **Export!**
4. Files appear in `Plotdaten/`.

### D. Cycle Extract

**Cycle Extraction**

1. Pick the Plotdaten file.
2. **Select a unit profile** (required). HP ID fills itself.
3. Confirm or add cycles. Set **ds** per cycle (1, 2, 3, …) *before* Apply.
4. Apply. Entries appear on **Data**. Review shows auto-filled Test cond, Test label, Lab ID and HP ID (HP ID from the profile). **Notes** must contain `drop` or `ramp` if the window should be split into defrost/heating or on/off sub-periods on **Period Statistics**; `other` means no split.

5. (Optional — needed for evaluation-window means and interval checks.) **Propose guideline clocks** → confirm kind → **Save all proposed** or **Save clocks**. Check them on **Guideline Windows**. Definitions: [Analysis Guide](ANALYSIS_GUIDE.md) §1 / §5.1.

You may see yellow notifications; they are not errors but notes about how missing data is handled.

### E. Means and deviations

**Data:** check start/end, profile, flow (fixed/var). Recalculate if you edited windows.

**Deviations:** calculate (empty cells mean “not calculated yet”). `0` = no violations; `N/A` = no valid points; `-` = not yet run. Variable-flow tests show **n/a** for flow columns (the check is Tmean / Tsup). **Plot** shows the traces; if clocks are saved, the band steps by interval.

**Guideline Windows:** read-only table of **saved** clocks, evaluation means, interval % / means and eval ΔCOP. If score cells are n/a on an older database, use **Compute missing scores**.

### F. Scatter (optional)

**Data** → select rows → **Add to Scatter Dataset** → **Scatter**.

- Combinations are unit + flow (profile name).
- Climate / application: if everything is Average/MT you see one info line. If several slices exist, use the checkboxes. Multi-slice plots add `[Average/MT]` (etc.) to the title.

### G. Export

**Export** → CSV or Excel. No formal pass/fail flags yet (colour is UI only).

---

## 7. Pages in one line

| Page | Use it for |
|------|------------|
| Datasets | Which `.db` you are writing into |
| Profiles | Unit parameters + heating-condition setpoints |
| Cycle Extraction | File → analysis entries; optional guideline clocks |
| Data | Means, COP, edit windows, mark scatter/COP sets |
| Deviations | Parent-cycle bands; Plot |
| Guideline Windows | Saved clocks, evaluation means, interval % / ΔCOP |
| Period Statistics | Defrost / heating sub-periods (needs Notes `drop` or `ramp`) |
| dTreturn Insights | Return-temperature controllability |
| Scatter | Compare units / flows / (if needed) climate×application |
| Export | Summary for Excel workbooks |

---

## 8. If it looks wrong

| Symptom | Fix |
|---------|-----|
| `UnicodeDecodeError` on start | Use `run_app.ps1` (or `run_app.bat`) |
| Notebook cannot see Rohdaten / maps | Jupyter working directory is not the cloned folder (§3) |
| Scatter shows `HP (fixed)` or merges files | Profile was not selected — assign it on Cycle Extract or review |
| Period Statistics empty / no sub-periods | Notes has no `drop` or `ramp` (e.g. `other`), or Test cond is not A/B/E/F or C/D |
| Deviations huge | Wrong profile, or variable-flow scored as fixed-flow |
| Profiles page stale after editing JSON | Reload the page |
| Numbers from another campaign | Wrong dataset open |

---

## 9. What this tool will not do yet

- Hybrid / water-to-water import (waiting on REV1 templates)
- Formal PASS/FAIL in the export
- An assembled per-test PDF/HTML report (use the pages + Export)
