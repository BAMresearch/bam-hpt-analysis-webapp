"""Checks for the mass-flow fallback (derive_mass_flow_if_needed) in flask_app/app.py.

Run from the repository root:  python scripts/test_mass_flow_fallback.py

Covers the one rule shared by BAM and HPT Plotdaten: measured mass flow wins, an absent
column and an empty column both mean "not reported", density comes from density_sink then
density then 997 kg/m³, and an implausible density falls back to 997 with a warning.
The last case runs a real process_file + calculate_and_insert against a throw-away
database, so BAM files without a mass flow column are proven to still insert.
"""
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flask_app'))

import app as webapp  # noqa: E402
import dataset_manager  # noqa: E402

FAILURES = []


def check(name, condition, detail=''):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def base_frame(rows=5):
    """Minimal time series with a volume flow, in m³/h as in Plotdaten."""
    return pd.DataFrame({
        'time_elapsed': np.arange(rows) * 10.0,
        'volume flow': np.linspace(0.9, 1.1, rows),
    })


def expected(volume, density):
    return pd.Series(volume, dtype=float) / 3600.0 * density


def case_1_no_column_no_density():
    df = base_frame()
    volume = df['volume flow'].copy()
    df, notices = webapp.derive_mass_flow_if_needed(df)
    ok = np.allclose(df['mass flow'], expected(volume, 997.0))
    check('1 BAM-style, no mass flow column, no density -> 997', ok)
    check('1 notice names the path', any('997' in n and 'derived' in n for n in notices),
          notices[0] if notices else 'no notice')
    return df['mass flow'].to_numpy()


def case_2_empty_column_no_density(reference):
    df = base_frame()
    df['mass flow'] = np.nan
    df, notices = webapp.derive_mass_flow_if_needed(df)
    check('2 BAM-style, empty mass flow column -> same series as case 1',
          np.allclose(df['mass flow'], reference))
    check('2 notice says the column was empty',
          any('empty' in n for n in notices), notices[0] if notices else 'no notice')


def case_3_empty_mass_flow_with_density_sink():
    df = base_frame()
    volume = df['volume flow'].copy()
    df['mass flow'] = np.nan
    df['density_sink'] = 985.0
    df, notices = webapp.derive_mass_flow_if_needed(df)
    check('3 HPT-style, empty mass flow + density_sink 985 -> uses 985',
          np.allclose(df['mass flow'], expected(volume, 985.0)))
    check('3 does not fall back to 997',
          not np.allclose(df['mass flow'], expected(volume, 997.0)))
    check('3 notice names density_sink',
          any('density_sink' in n for n in notices), notices[0] if notices else 'no notice')


def case_4_density_without_sink_suffix():
    df = base_frame()
    volume = df['volume flow'].copy()
    df['density'] = 990.0
    df, notices = webapp.derive_mass_flow_if_needed(df)
    check('4 BAM-style, filled density (no _sink) -> uses 990',
          np.allclose(df['mass flow'], expected(volume, 990.0)))
    check('4 notice names density',
          any("measured density" in n for n in notices), notices[0] if notices else 'no notice')


def case_5_measured_wins():
    df = base_frame()
    measured = np.array([0.30, 0.31, 0.0, 0.32, 0.33])
    df['mass flow'] = measured
    df['density_sink'] = 985.0
    df, notices = webapp.derive_mass_flow_if_needed(df)
    check('5 measured mass flow unchanged (zeros kept)', np.allclose(df['mass flow'], measured))
    check('5 no derivation flag', not df.attrs.get('mass_flow_derived'))
    check('5 notice says measured', any('measured values' in n for n in notices),
          notices[0] if notices else 'no notice')


def case_6_partial_mass_flow():
    df = base_frame()
    volume = df['volume flow'].copy()
    measured = np.array([0.30, np.nan, 0.32, np.nan, 0.34])
    df['mass flow'] = measured
    df['density_sink'] = [985.0, 985.0, 985.0, np.nan, 985.0]
    df, notices = webapp.derive_mass_flow_if_needed(df)
    result = df['mass flow'].to_numpy()
    finite_kept = np.allclose(result[[0, 2, 4]], measured[[0, 2, 4]])
    row1 = np.isclose(result[1], volume.iloc[1] / 3600.0 * 985.0)
    row3 = np.isclose(result[3], volume.iloc[3] / 3600.0 * 997.0)
    check('6 mixed mass flow: measured rows kept', finite_kept)
    check('6 mixed mass flow: empty row uses measured density', row1)
    check('6 partial density: gap row uses 997', row3)
    check('6 notice counts the filled rows', any('2 of 5 rows' in n for n in notices),
          notices[0] if notices else 'no notice')


def case_7_density_in_wrong_unit():
    df = base_frame()
    volume = df['volume flow'].copy()
    df['mass flow'] = np.nan
    df['density_sink'] = 0.997
    df, notices = webapp.derive_mass_flow_if_needed(df)
    check('7 density_sink 0.997 g/cm³ -> 997 kg/m³',
          np.allclose(df['mass flow'], expected(volume, 997.0)))
    check('7 warning about the unit', any('g/cm' in n for n in notices),
          ' | '.join(notices))


def bam_style_excel(tmpdir, rows=6):
    """A BAM Plotdaten sheet: no mass flow column, no density, no BUH column."""
    df = pd.DataFrame({
        'time_elapsed': np.arange(rows) * 10.0,
        'T_outdoor (DB)': np.full(rows, 7.0),
        'T_outdoor (WB)': np.full(rows, 6.0),
        'T_supply': np.full(rows, 42.0),
        'T_return_emu': np.full(rows, 37.0),
        'volume flow': np.linspace(0.9, 1.1, rows),
        'Heating Capacity (corrected)': np.full(rows, 5.0),
        'Electric Power Input (corrected)': np.full(rows, 1.6),
        'Heating Capacity (without corr)': np.full(rows, 5.1),
        'Electric Power Input (without correction)': np.full(rows, 1.7),
        'Pressure difference': np.full(rows, 20000.0),
        'T_return_calc': np.full(rows, 37.1),
        'T_B_calc': np.full(rows, 36.9),
        'T_H_calc': np.full(rows, 42.2),
        'Q_HB': np.full(rows, 0.1),
        'Q_BA': np.full(rows, 0.1),
    })
    path = os.path.join(tmpdir, 'RRT_HP1_B_test.xlsx')
    df.to_excel(path, index=False)
    return path, df


def temp_database(tmpdir):
    path = os.path.join(tmpdir, 'test_results.db')
    conn = sqlite3.connect(path)
    conn.execute(dataset_manager.RESULTS_CREATE_SQL)
    conn.execute('ALTER TABLE results ADD COLUMN display_order INTEGER')
    conn.commit()
    conn.close()
    return path


def case_8_end_to_end_insert():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, source = bam_style_excel(tmpdir)
        db_path = temp_database(tmpdir)
        old_data_dir = webapp.config.get('data_dir')
        old_db_path = webapp.config.get('database_path')
        webapp.config['data_dir'] = tmpdir
        webapp.config['database_path'] = db_path
        try:
            df_filtered = webapp.process_file('RRT_HP1_B_test.xlsx', 1, 0, 100)
            check('8 process_file returns a window', df_filtered is not None and not df_filtered.empty)
            if df_filtered is None:
                return
            check('8 mass flow derived with 997',
                  np.allclose(df_filtered['mass flow'], expected(source['volume flow'], 997.0)))
            check('8 notice reaches the UI list',
                  any('derived from volume flow' in n
                      for n in df_filtered.attrs.get('import_notices', [])),
                  ' | '.join(df_filtered.attrs.get('import_notices', [])))
            check('8 has_buh stays False for an absent BUH column',
                  not bool(df_filtered['has_buh'].iloc[0]))

            conn = webapp.get_db_connection()
            try:
                notices = []
                webapp.calculate_and_insert(conn.cursor(), df_filtered, 'RRT_HP1_B_test.xlsx',
                                            1, 0, 100, notices_out=notices)
                conn.commit()
                row = conn.execute(
                    'SELECT avg_mass_flow, avg_power_buh, avg_density_sink FROM results'
                ).fetchone()
            finally:
                conn.close()
            check('8 insert still works', row is not None)
            check('8 avg_mass_flow stored from the derived series',
                  np.isclose(row['avg_mass_flow'],
                             float(expected(source['volume flow'], 997.0).mean())),
                  f"stored {row['avg_mass_flow']!r}")
            check('8 virtual BUH untouched (avg_power_buh 0)', row['avg_power_buh'] == 0)
            check('8 optional extras stay n/a (avg_density_sink NULL)',
                  row['avg_density_sink'] is None)
            check('8 notice repeated for the analyst',
                  any('derived from volume flow' in n for n in notices), ' | '.join(notices))
        finally:
            webapp.config['data_dir'] = old_data_dir
            if old_db_path is None:
                webapp.config.pop('database_path', None)
            else:
                webapp.config['database_path'] = old_db_path


def main():
    reference = case_1_no_column_no_density()
    case_2_empty_column_no_density(reference)
    case_3_empty_mass_flow_with_density_sink()
    case_4_density_without_sink_suffix()
    case_5_measured_wins()
    case_6_partial_mass_flow()
    case_7_density_in_wrong_unit()
    case_8_end_to_end_insert()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print('All mass-flow fallback checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
