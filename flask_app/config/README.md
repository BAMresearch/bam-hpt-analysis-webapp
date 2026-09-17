# Configuration (B1)

JSON under this folder is the source of truth for guideline setpoints and unit parameters.

| Path | In git? | Contents |
|------|---------|----------|
| `condition_sets.json` | Yes | Outdoor DB + supply setpoints per condition letter |
| `unit_types.json` | Yes | Capability flags per unit class |
| `checks.json` | Yes | Which checks apply for which capabilities |
| `cycle_periods.json` | Yes | Guideline clock lengths: `buffer_min`, `eq_min`, `eval_min` |
| `interval_deviations.json` | Yes | Deviations per guideline interval: `delta_cop_pct`, `delta_cop_slice_min`, Interval H/eq/eval bands, D/S individual half-widths, and the `mean_*` half-widths scored as the H/D/S mean columns on Guideline Windows. Do not set D/S `"bands": "permissible_deviations"` |
| `profiles/examples/` | Yes | Fictitious schema examples |
| `profiles/shared/` | Yes | Public HPT placeholders (no private BAM data) |
| `profiles/local/` | **No** (gitignored) | Real BAM / lab unit profiles |

Edit via the **Profiles** page in the UI, or by editing the JSON files and restarting / saving through the API (condition sets reload in memory on save).

Database path is set in `../config.json` as `database_path` (relative to `flask_app/` or absolute).
Prefer the **Datasets** page (`/datasets`) to create or open a database — it updates `database_path` for you.
Lab `.db` files live under `../../notebooks/` (gitignored); they are never shipped with the tool.
