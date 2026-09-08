# Import maps — how to read and edit them

Maps live in this folder as JSON files. The notebook loads one by id:

```python
MAPPING = load_map("hpt_aw_rev1")  # loads hpt_aw_rev1.json
```

| File | Use for |
|------|---------|
| `hpt_aw_rev1.json` | **Current** AIR-TO-WATER template (only option until WW / hybrid maps exist) |

Water-to-water and hybrid will be separate files (`hpt_ww_rev1.json`, `hpt_hybrid_rev1.json`) when those templates exist. The notebook dropdown lists a map only if its JSON is present.

---

## What a map does

Template Excel column names → **Plotdaten / WebApp** column names (+ unit conversion + fills).

```
HPT template sheet  --detect headers-->  raw columns
                 --columns[] rename/scale-->  Plotdaten names
                 --fill_if_missing[]------>  complete required set
                 --validate--------------->  report
```

You almost never need to change Python for a **tiny rename**. Edit the JSON, bump `version`, re-run the notebook.

---

## Top-level keys

| Key | Meaning |
|-----|---------|
| `map_id` | Must match the filename stem (`hpt_aw_rev1`) |
| `version` | Your changelog stamp (e.g. `0.2.1`) — bump when you edit |
| `changelog` | Short human history of map edits |
| `template_revision` | Which Excel template this map targets |
| `header_detection` | How to find the name/units/data rows |
| `sheet_names_expected` | Default sheets to process (`E,F,A,B,C,D`) |
| `webapp_required_columns` | What validation checks after normalize |
| `columns` | **Main rename table** (edit this most often) |
| `fill_if_missing` | Create columns the WebApp needs if labs did not provide them |
| `skip_empty_targets` | Targets dropped when the template column is entirely empty (invent-risk: mass flow, corrected Q/P, both BUH columns). Every other optional column is kept even when all-empty, so the field stays visible and the WebApp stores NULL |
| `passthrough_unmapped` | Keep extra template columns under their original names |

---

## Each entry in `columns`

```json
{
  "source": "T_out_sink",
  "target": "T_supply",
  "required": true,
  "aliases": ["T_out"],
  "scale": 0.01,
  "unit_in": "kPa",
  "unit_out": "bar",
  "fill_if_target_missing": true,
  "notes": "…"
}
```

| Field | Effect |
|-------|--------|
| `source` | Exact (or whitespace-normalised) header in the **template** |
| `target` | Name written to **Plotdaten** / expected by WebApp |
| `required` | If `true` and column missing → listed in `missing_sources` (import still continues) |
| `aliases` | Alternate template spellings still accepted |
| `scale` | Multiply numeric values (ΔP: `0.01` = kPa→bar) |
| `fill_if_target_missing` | Only write if `target` not already filled (used so **Backup heater** wins over **Virtual Back-up**) |

**Order matters** when two sources share one `target` (BUH): put preferred source first, then Virtual Back-up with `fill_if_target_missing: true`.

---

## `fill_if_missing` rules (code-backed)

These run **after** renames. Rule names are implemented in `import_tools/normalize.py`:

| `rule` | Behaviour |
|--------|-----------|
| *(none by default in ≥0.3.2)* | Do not invent mass flow, corrected Q/P, or BUH=0 |

**Removed (on purpose):** auto-creating `mass flow`, copying without-corr → corrected, and writing `Electrical power input BUH = 0` when Virtual Back-up is absent (that false column made WebApp run `power_BUH_from_Ts` like a BAM file with BUH). Omit the column instead — same as BAM files without BUH.

Changing the *formula* needs a small Python edit; changing *when* it applies is JSON.

---

## Typical tiny edits (no AI needed)

1. **Template renamed a column**  
   Change `source` (and/or add old name to `aliases`). Bump `version`.

2. **You want a different Plotdaten name**  
   Change `target` — only if WebApp also knows that name (or you update WebApp).

3. **New optional column to keep**  
   Either add a `columns` entry (`source`=`target`) or rely on `passthrough_unmapped: true`.

4. **ΔP already in bar in a future template**  
   Remove `"scale": 0.01` for that entry (or set `scale` to `1`).

5. **Sheet without Virtual Back-up**  
   **Omit** `Electrical power input BUH` — do **not** write `0`. An absent column tells the WebApp "no BUH"; an all-zero column tells it a BUH exists that drew no power, which makes it run `power_BUH_from_Ts`. Real `Backup heater Input (optional)` maps to `Real BUH power input`, is a monitoring mean only, and is never used as WebApp BUH.

6. **Do not re-add silent corrected/mass-flow fills**  
   Prefer fixing the lab paste or teaching WebApp to tolerate missing columns.

---

## How to verify a change

1. Open `import_hpt.ipynb` (working directory = `05_Pythonscripte`).
2. Set `MAP_ID = "hpt_aw_rev1"` and `RAW_FILE` to your workbook.
3. Keep `DRY_RUN = True`.
4. Read the per-sheet report:
   - `renames` — what matched
   - `missing_sources` — required template columns not found
   - `fills` — what was synthesised; **should normally be empty** since ≥0.3.2
   - `MISSING required` — WebApp contract gaps after normalize

---

## Related code (only if JSON is not enough)

| File | Role |
|------|------|
| `import_tools/detect.py` | Find header / units / data start rows |
| `import_tools/normalize.py` | Apply map + fills + `time_elapsed` |
| `import_tools/validate.py` | Check `webapp_required_columns` |
| `import_hpt.ipynb` | Thin UI: pick file, run, write `Plotdaten/` |

BAM path stays on `import_v2.ipynb` — do not point this map at old BAM compensation templates.
