# T4.2 Data Analysis WebApp

Local analysis tool for **HPT T4.2** round-robin test data: import lab workbooks, review cycles, compute mean values and permissible deviations, generate scatter plots, and export summary tables.

**Release scope (v0.1):** air-to-water templates only. One shared codebase serves BAM and HPT workflows; this release is configured for HPT placeholders and the `hpt_aw_rev1` import map.

---

## Licence

This software is released under the [MIT License](LICENSE). Copyright (c) 2026 Bundesanstalt für Materialforschung und -prüfung (BAM).

**No measurement data files are tracked.** Excel, databases (`.db`, `.sqlite`), and local `config.json` are excluded via `.gitignore`. Lab workbooks and analysis databases stay on your PC.

---

## Project context and funding

This tool is developed and owned by the Bundesanstalt für Materialforschung und -prüfung (BAM). It is **not** an official HPT deliverable or milestone.

Most of the analysis application was developed with support from the Deutsche Bundesstiftung Umwelt (DBU) in the project *Validierung der Kompensationsmethode für Wärmepumpen – Ringversuch und Reglersensitivitätsanalyse* (Az. 38943/01).

After the Horizon Europe project [Heat Pump Testing (HPT)](https://cordis.europa.eu/project/id/101295795) (grant agreement No **101295795**) started, the tool was cleaned, versioned and adapted for Task 4.2 round-robin data analysis.

Funded by the European Union. Views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union or the granting authority. Neither the European Union nor the granting authority can be held responsible for them.

---

## Quick start

**Full walkthrough:** [`docs/ANALYST_QUICK_GUIDE.md`](docs/ANALYST_QUICK_GUIDE.md) — one analysis from clone to export.

### Prerequisites

- **Git**
- **Python 3.11+**
- **Jupyter** (for the import notebook)

### 1. Clone and install (once per PC)

   ```powershell
git clone https://github.com/BAMresearch/bam-hpt-analysis-webapp.git
cd bam-hpt-analysis-webapp
pip install -r requirements.txt
```

The clone folder is your working directory. Do not wrap it in a second project folder.

### 2. Folders you will use

```
bam-hpt-analysis-webapp/
  flask_app/          WebApp (browser UI)
  import_hpt.ipynb    lab template → Plotdaten Excel
  import_tools/       column maps (e.g. hpt_aw_rev1)
  Rohdaten/           put lab template workbooks here
  Plotdaten/          importer output — WebApp reads from here
  docs/               analyst documentation
```

### 3. Import lab data

1. Copy filled lab workbooks into `Rohdaten/`.
2. Start Jupyter with the **clone root** as working directory.
3. Open `import_hpt.ipynb`, choose **Air-to-water**, run all cells.
4. Normalised one-sheet Excel files appear in `Plotdaten/`.

The WebApp expects Plotdaten layout only — do not point it at raw lab templates.

### 4. Configure (first run)

If `flask_app/config.json` does not exist, the app uses `flask_app/config_template.json`, which already sets `data_dir` to `../Plotdaten` when you start from `flask_app/`.

Copy the template only when you need local overrides:

```powershell
copy flask_app\config_template.json flask_app\config.json
```

Unit profiles (design capacity, flow mode, condition set) live under `flask_app/config/profiles/` — use the HPT placeholders (`HPT_RRT1` … `HPT_RRT6_STAGED`) until real unit data is confirmed.

### 5. Start the WebApp (each session)

```powershell
cd bam-hpt-analysis-webapp\flask_app
.\run_app.ps1
```

On Windows, prefer `run_app.ps1` or `run_app.bat` over a bare `python app.py` (UTF-8 encoding).

When the console shows `Running on http://127.0.0.1:5001`, open that address in a browser **on the same PC**. Port 5001 is fixed; this is not a shared server.

### 6. Typical analysis flow

1. **Data review / Cycle Extract** — import Plotdaten files, assign a **unit profile** per test, extract cycles.
2. **Mean values** (`/`) — table of cycle means; bulk recalculate and dataset tagging.
3. **Permissible deviations** (`/deviations`) — on-demand deviation statistics vs setpoints.
4. **Scatter** (`/scatter`) — compare parameters across unit/flow configurations.
5. **Export** (`/export`) — `t42_summary_v1` summary table.

Details, field meanings, and troubleshooting: Analyst Quick Guide.

---

## Documentation

| Document | Location |
|----------|----------|
| Analyst Quick Guide (start here) | [`docs/ANALYST_QUICK_GUIDE.md`](docs/ANALYST_QUICK_GUIDE.md) |
| User Guide (full operational reference) | [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) |
| Analysis Guide (methods, formulas, limits) | [`docs/ANALYSIS_GUIDE.md`](docs/ANALYSIS_GUIDE.md) |
| Version history | [`CHANGELOG.md`](CHANGELOG.md) |

---

## Main features

- **Import pipeline** — lab template → Plotdaten via `import_hpt.ipynb` and JSON column maps in `import_tools/maps/`
- **Unit profiles** — required identity for scatter grouping and deviation setpoints; HP ID derived from profile
- **Mean values** — configurable columns, derived formulas, recalculation history
- **Permissible deviations** — profile-aware checks (`checks.json`), wet-bulb handling, on-demand calculation
- **Scatter plots** — climate/application filters, multi-parameter grids, bulk download
- **Export** — column-picker summary export (`t42_summary_v1`); pass/fail columns reserved for a later release
- **Profiles & datasets** — edit unit profiles and condition sets without code changes

---

## Project layout (reference)

```
bam-hpt-analysis-webapp/
├── flask_app/
│   ├── app.py                 # Flask application entry point
│   ├── config_template.json   # default config (safe to commit)
│   ├── config/                # unit profiles, condition sets, checks
│   ├── templates/             # HTML UI
│   └── run_app.ps1            # recommended Windows launcher
├── import_hpt.ipynb
├── import_tools/
├── Rohdaten/                  # input workbooks (empty in clone)
├── Plotdaten/                 # normalised Excel (empty in clone)
├── docs/
├── requirements.txt
├── LICENSE
└── CHANGELOG.md
```

Local-only (gitignored): `flask_app/config.json`, `*.db`, `*.xlsx`, `Rohdaten/*`, `Plotdaten/*`.

---

## Support

- **Tool bugs and import-map issues:** [GitHub Issues](https://github.com/BAMresearch/bam-hpt-analysis-webapp/issues) on this repository
- **T4.2 methods and process:** [`docs/ANALYSIS_GUIDE.md`](docs/ANALYSIS_GUIDE.md) and WP4 meetings
