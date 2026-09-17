"""Checks for Cycle Extract guideline clocks and the Guideline Windows page.

Run from the repository root:  python scripts/test_guideline_clocks.py

Synthetic numbers only — no lab data, no analyst database. D/S is the
transition plus the buffer, H is the remainder, equilibrium is the first eq_min
of H and evaluation the eval_min after it (never the last minutes of the
cycle). Evaluation is omitted when H is too short. Eq/eval apply to defrost and
continuous, not on–off. A window cut defrost-end to next defrost-end carries
two D spans with H in the middle; defrost-start to next defrost-start keeps
one. Kind ``continuous``: no D/S, H is the parent window. Stored cycle type
``other`` is unknown, never continuous.

API checks: the file-level kind default reaches the clocks; Save all with no
ticks means the whole file; every ``/api/`` reply is JSON even on failure
(throwaway SQLite + stubbed sheet).

Guideline Windows: evaluation Tsup, Q, P and COP on a sheet that has only
uncorrected Q and P use the same analysis-time path as the
stored entry. H / eq / eval dTreturn % uses the Interval H ±0.5 K inlet band;
eval ΔCOP uses the first and last five minutes of the saved evaluation window.
D / S individual % use the draft widths on the union of saved spans (D ±5 K DB
and ±2 K dTreturn; S ±2 K DB/WB, ±2.5 % flow, ±1 K dTreturn). Parent Deviations
keeps the full-cycle −2…+2 K dTreturn band.
"""
import contextlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd

# Runtime import: flask_app/app.py is not a package. basedpyright does not
# execute sys.path changes, so it cannot see this module from scripts/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flask_app'))

import app as webapp  # noqa: E402  # pyright: ignore[reportMissingImports]

FAILURES = []

# A Windows console is cp1252: Δ and the like must not turn a failing check into
# a UnicodeEncodeError that hides every check after it.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

LENGTHS = {'buffer_min': 10.0, 'eq_min': 60.0, 'eval_min': 70.0,
           'buffer_s': 600.0, 'eq_s': 3600.0, 'eval_s': 4200.0}


def check(name, condition, detail=''):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def windows(result):
    """{period_type: (start, end)} — only for results with one span per type."""
    return {p['period_type']: (p['start_time'], p['end_time']) for p in result['periods']}


def spans(result):
    """[(type, start, end)] — use this where D/S may appear twice."""
    return [(p['period_type'], p['start_time'], p['end_time']) for p in result['periods']]


def test_defrost_opens_in_d():
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'ds', 600.0, LENGTHS)
    w = windows(r)
    check('D runs from cycle start to rise + buffer', w.get('defrost') == (0.0, 1200.0), str(w.get('defrost')))
    check('H is the remainder', w.get('heating') == (1200.0, 12000.0), str(w.get('heating')))
    check('equilibrium is the first eq_min of H', w.get('equilibrium') == (1200.0, 4800.0), str(w.get('equilibrium')))
    check('evaluation follows equilibrium', w.get('evaluation') == (4800.0, 9000.0), str(w.get('evaluation')))
    check('evaluation is not the tail of H', w['evaluation'][1] < w['heating'][1],
          f"eval ends {w['evaluation'][1]}, H ends {w['heating'][1]}")


def test_defrost_opens_in_d_without_later_drop():
    # Cut defrost start → next defrost start: the next defrost is the next parent
    # row, so a drop sitting on the cycle end must not become a second span.
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'ds', 600.0, LENGTHS,
                                       next_drop=11980.0)
    s = spans(r)
    check('still one D span', len([x for x in s if x[0] == 'defrost']) == 1, str(s))
    check('H runs to the cycle end', ('heating', 1200.0, 12000.0) in s, str(s))


def test_defrost_opens_in_d_with_later_drop():
    # Cut defrost end → next defrost end: leftover buffer, H, then the next defrost.
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'ds', 600.0, LENGTHS,
                                       next_drop=10000.0)
    s = spans(r)
    check('first D is the leftover buffer', ('defrost', 0.0, 1200.0) in s, str(s))
    check('H is the gap between the D spans', ('heating', 1200.0, 10000.0) in s, str(s))
    check('second D runs from the drop to the cycle end', ('defrost', 10000.0, 12000.0) in s, str(s))
    check('equilibrium is the first eq_min of that H', ('equilibrium', 1200.0, 4800.0) in s, str(s))
    check('evaluation follows equilibrium', ('evaluation', 4800.0, 9000.0) in s, str(s))
    check('eq/eval stay inside H', all(x[1] >= 1200.0 and x[2] <= 10000.0
                                       for x in s if x[0] in ('equilibrium', 'evaluation')), str(s))
    check('reason reported', any('two spans' in n for n in r['notes']), str(r['notes']))


def test_two_span_h_too_short_drops_evaluation():
    # The second span shortens H to 1200…4000 s = 47 min, below eq_min.
    r = webapp._build_guideline_clocks(0.0, 6000.0, 'defrost', 'ds', 600.0, LENGTHS,
                                       next_drop=4000.0)
    s = spans(r)
    w = windows(r)
    check('H stops at the drop', ('heating', 1200.0, 4000.0) in s, str(s))
    check('no equilibrium in the short H', 'equilibrium' not in w, str(list(w)))
    check('no evaluation in the short H', 'evaluation' not in w, str(list(w)))


def test_on_off_two_span():
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'on_off', 'ds', 600.0, LENGTHS,
                                       next_drop=10000.0)
    s = spans(r)
    check('first S is the leftover buffer', ('off', 0.0, 1200.0) in s, str(s))
    check('H is the gap', ('on', 1200.0, 10000.0) in s, str(s))
    check('second S to the cycle end', ('off', 10000.0, 12000.0) in s, str(s))
    check('no eq/eval on on-off', not any(x[0] in ('equilibrium', 'evaluation') for x in s), str(s))


def test_analyst_moves_or_removes_the_second_span():
    moved = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'ds', 600.0, LENGTHS,
                                           override={'ds2_start': 9000.0}, next_drop=10000.0)
    s = spans(moved)
    check('typed second D wins over the detected drop', ('defrost', 9000.0, 12000.0) in s, str(s))
    check('H follows the typed second D', ('heating', 1200.0, 9000.0) in s, str(s))

    added = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'ds', 600.0, LENGTHS,
                                           override={'ds2_start': 9000.0})
    check('analyst can add a second D without detection',
          ('defrost', 9000.0, 12000.0) in spans(added), str(spans(added)))

    removed = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'ds', 600.0, LENGTHS,
                                             override={'no_second_ds': True}, next_drop=10000.0)
    s = spans(removed)
    check('analyst can drop the second D', len([x for x in s if x[0] == 'defrost']) == 1, str(s))
    check('H goes back to the cycle end', ('heating', 1200.0, 12000.0) in s, str(s))


def test_defrost_opens_in_h():
    # Window starts in heating; the transition is the drop that ends it, and the
    # first buffer_min still belongs to the previous defrost.
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'h', 10000.0, LENGTHS)
    spans = [(p['period_type'], p['start_time'], p['end_time']) for p in r['periods']]
    check('leading buffer is D', ('defrost', 0.0, 600.0) in spans, str(spans))
    check('H runs to the drop', ('heating', 600.0, 10000.0) in spans, str(spans))
    check('trailing defrost to cycle end', ('defrost', 10000.0, 12000.0) in spans, str(spans))
    check('equilibrium anchored to H start', ('equilibrium', 600.0, 4200.0) in spans, str(spans))
    check('evaluation after equilibrium', ('evaluation', 4200.0, 8400.0) in spans, str(spans))


def test_h_too_short_for_evaluation():
    # H = 1200…6000 s = 80 min: room for eq (60) but not eq + eval (130).
    r = webapp._build_guideline_clocks(0.0, 6000.0, 'defrost', 'ds', 600.0, LENGTHS)
    w = windows(r)
    check('equilibrium still fits', w.get('equilibrium') == (1200.0, 4800.0), str(w.get('equilibrium')))
    check('evaluation omitted', 'evaluation' not in w, str(list(w)))
    check('reason reported', any('evaluation omitted' in n for n in r['notes']), str(r['notes']))


def test_h_too_short_for_equilibrium():
    r = webapp._build_guideline_clocks(0.0, 3000.0, 'defrost', 'ds', 600.0, LENGTHS)
    w = windows(r)
    check('no equilibrium', 'equilibrium' not in w, str(list(w)))
    check('no evaluation', 'evaluation' not in w, str(list(w)))
    check('reason mentions eq_min', any('eq_min' in n for n in r['notes']), str(r['notes']))


def test_on_off_gets_buffer_but_no_eq_eval():
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'on_off', 'ds', 600.0, LENGTHS)
    w = windows(r)
    check('S is standby plus the buffer after restart', w.get('off') == (0.0, 1200.0), str(w.get('off')))
    check('H uses the existing "on" type', w.get('on') == (1200.0, 12000.0), str(w.get('on')))
    check('no eq/eval on on-off', not ({'equilibrium', 'evaluation'} & set(w)), str(list(w)))


def test_continuous_long_window():
    # No defrost, no standby: the parent window is H and needs no transition.
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'continuous', None, None, LENGTHS)
    s = spans(r)
    w = windows(r)
    check('no D or S on a continuous cycle',
          not any(x[0] in ('defrost', 'off') for x in s), str(s))
    check('H is the whole parent window', w.get('heating') == (0.0, 12000.0), str(w.get('heating')))
    check('equilibrium is the first eq_min from the parent start',
          w.get('equilibrium') == (0.0, 3600.0), str(w.get('equilibrium')))
    check('evaluation follows equilibrium', w.get('evaluation') == (3600.0, 7800.0), str(w.get('evaluation')))
    check('evaluation is not the tail of the window', w['evaluation'][1] < w['heating'][1],
          f"eval ends {w['evaluation'][1]}, H ends {w['heating'][1]}")


def test_continuous_short_window_has_no_evaluation():
    # 110 min: room for eq (60) but not eq + eval (130).
    r = webapp._build_guideline_clocks(0.0, 6600.0, 'continuous', None, None, LENGTHS)
    w = windows(r)
    check('H is still the whole window', w.get('heating') == (0.0, 6600.0), str(w.get('heating')))
    check('equilibrium still fits', w.get('equilibrium') == (0.0, 3600.0), str(w.get('equilibrium')))
    check('evaluation omitted', 'evaluation' not in w, str(list(w)))
    check('reason reported', any('evaluation omitted' in n for n in r['notes']), str(r['notes']))


def test_continuous_needs_no_power_transition():
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'continuous', 'ds', None, LENGTHS,
                                       next_drop=10000.0)
    s = spans(r)
    check('a missing transition is not an error for continuous', r['periods'] != [], str(r['notes']))
    check('a later drop does not create a second span',
          not any(x[0] in ('defrost', 'off') for x in s), str(s))
    check('no "no interior power transition" note',
          not any('No interior power transition' in n for n in r['notes']), str(r['notes']))


def test_other_is_unknown_not_continuous():
    kind, src = webapp._resolve_guideline_kind(None, 'other', None, None, set(), None)
    check("stored cycle_type 'other' is unknown", kind == 'unknown', f'{kind} ({src})')
    check('the reason names the stale classification', 'other' in src, src)
    check("'other' is not a kind the analyst can pick",
          webapp._normalise_guideline_kind('other') is None,
          str(webapp._normalise_guideline_kind('other')))
    check('allowed kinds are exactly the three',
          set(webapp.GUIDELINE_KINDS) == {'defrost', 'on_off', 'continuous'},
          str(webapp.GUIDELINE_KINDS))


def test_file_level_default_applies_only_to_unknown():
    def kind_of(row_kind, cycle_type, indicator=None, saved=(), default=None):
        return webapp._resolve_guideline_kind(row_kind, cycle_type, indicator, 'ind',
                                              saved, default)

    k, src = kind_of(None, None, default='continuous')
    check('the default fills an unknown row', (k, src) == ('continuous', 'file-level default'), f'{k} ({src})')
    k, src = kind_of(None, 'other', default='defrost')
    check("the default also fills a stale 'other' row", k == 'defrost', f'{k} ({src})')
    k, src = kind_of(None, 'defrost_cycle', default='continuous')
    check('a stored kind wins over the default', (k, src) == ('defrost', 'stored cycle_type'), f'{k} ({src})')
    k, src = kind_of('on_off', 'defrost_cycle', default='continuous')
    check('the analyst wins over everything', (k, src) == ('on_off', 'analyst'), f'{k} ({src})')
    k, _ = kind_of(None, None, indicator=True, default='continuous')
    check('the indicator column wins over the default', k == 'defrost', k)
    k, _ = kind_of(None, None, saved={'heating', 'equilibrium'}, default='defrost')
    check('saved continuous clocks win over the default', k == 'continuous', k)
    k, src = kind_of(None, None)
    check('no default leaves the row unknown', k == 'unknown', f'{k} ({src})')


def test_moving_ds_end_rebuilds_h_and_locked_clocks():
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'ds', 600.0, LENGTHS,
                                       override={'ds_end': 2000.0})
    w = windows(r)
    check('D end follows the analyst', w.get('defrost') == (0.0, 2000.0), str(w.get('defrost')))
    check('H rebuilt from the new D end', w.get('heating') == (2000.0, 12000.0), str(w.get('heating')))
    check('locked equilibrium follows H start', w.get('equilibrium') == (2000.0, 5600.0), str(w.get('equilibrium')))
    check('locked evaluation follows', w.get('evaluation') == (5600.0, 9800.0), str(w.get('evaluation')))


def test_unlocked_eq_eval_are_kept():
    r = webapp._build_guideline_clocks(
        0.0, 12000.0, 'defrost', 'ds', 600.0, LENGTHS,
        override={'eq_locked': False, 'eq_start': 1500.0, 'eq_end': 5000.0,
                  'eval_locked': False, 'eval_start': 5000.0, 'eval_end': 9200.0},
    )
    w = windows(r)
    check('unlocked equilibrium typed by the analyst', w.get('equilibrium') == (1500.0, 5000.0), str(w.get('equilibrium')))
    check('unlocked evaluation typed by the analyst', w.get('evaluation') == (5000.0, 9200.0), str(w.get('evaluation')))


def test_no_transition_is_reported_not_invented():
    r = webapp._build_guideline_clocks(0.0, 12000.0, 'defrost', 'ds', None, LENGTHS)
    check('no periods without a transition', r['periods'] == [], str(r['periods']))
    check('reason reported', any('No interior power transition' in n for n in r['notes']), str(r['notes']))


def test_lengths_come_from_config():
    lg = webapp._guideline_lengths()
    ok = all(k in lg for k in ('buffer_min', 'eq_min', 'eval_min', 'buffer_s', 'eq_s', 'eval_s'))
    check('config exposes all three lengths', ok, str(sorted(lg)))
    check('seconds derive from minutes',
          lg['buffer_s'] == lg['buffer_min'] * 60 and lg['eq_s'] == lg['eq_min'] * 60
          and lg['eval_s'] == lg['eval_min'] * 60,
          str(lg))
    check('buffer_s used for storage is an int', isinstance(webapp._guideline_buffer_s(), int))


def test_opens_low_detection():
    idle = np.full(200, 0.2)
    on = np.full(800, 3.0)
    check('window opening at idle begins in D/S', webapp._cycle_opens_low(np.concatenate([idle, on])) is True)
    check('window opening under load begins in H', webapp._cycle_opens_low(np.concatenate([on, idle])) is False)
    check('flat trace is undecidable', webapp._cycle_opens_low(np.full(500, 2.0)) is None)
    check('short trace is undecidable', webapp._cycle_opens_low(np.full(4, 2.0)) is None)


def test_second_drop_is_searched_after_the_first_span():
    # Idle until 600 s (the defrost the window opens in), heating, then the next
    # defrost at 10000 s. Only the part after the first D end may be searched, so
    # the opening rise cannot be returned instead.
    t = pd.Series(np.arange(0.0, 12000.0, 10.0))
    p = pd.Series(np.where((t < 600) | (t >= 10000), 0.2, 3.0))
    drop = webapp._detect_next_drop_after(p, t, 1200.0)
    check('second drop found in the heating tail',
          drop is not None and 9900.0 <= drop <= 10050.0, str(drop))
    check('pure heating tail has no second drop',
          webapp._detect_next_drop_after(pd.Series(np.full(400, 3.0)),
                                         pd.Series(np.arange(400.0)), 10.0) is None)
    check('short tail is not searched',
          webapp._detect_next_drop_after(p, t, 11950.0) is None)


# ---------------------------------------------------------------------------
# API level: the file-level default must reach the clocks, and no
# /api/ reply may be an HTML page. Throwaway DB + stubbed sheet, no lab data.
# ---------------------------------------------------------------------------

API_FILE = 'synthetic_not_a_lab_file.xlsx'


def _fake_sheet(file_name, data_set=None, usecols=None):
    """Two 200 min windows, each opening at idle power (so 'ds' would be detected)."""
    t = np.arange(0.0, 24001.0, 10.0)
    return pd.DataFrame({
        'time_elapsed': t,
        'Electric power input (without correction)': np.where((t % 12000) < 600, 0.2, 3.0),
    })


class ApiHarness:
    """Flask test client wired to a temporary database and a stubbed sheet.

    ``ROWS`` and ``SHEET`` are class attributes so a later section can reuse the
    whole wiring with a different synthetic file. Still synthetic only: no lab
    workbook and no analyst database is ever opened.
    """

    #: (file_name, data_set, start_time, end_time, cycle_type, cycle_start_marker,
    #:  pelec_transition_time, pelec_transition_source)
    #: One row with no classification at all, one left over as 'other'.
    ROWS = [(API_FILE, 1, 0, 12000, None, None, None, None),
            (API_FILE, 1, 12000, 24000, 'other', None, None, None)]
    SHEET = staticmethod(_fake_sheet)

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp(prefix='guideline_clocks_')
        self.db = os.path.join(self.tmpdir, 'test.db')
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE results (file_name TEXT, data_set REAL, start_time REAL, "
                     "end_time REAL, cycle_type TEXT, cycle_start_marker TEXT, "
                     "pelec_transition_time REAL, pelec_transition_source TEXT, "
                     "pelec_detect_failed INTEGER, HP_ID TEXT, dev_test_condition TEXT, "
                     "indicator TEXT, notes TEXT)")
        conn.executemany(
            "INSERT INTO results (file_name, data_set, start_time, end_time, cycle_type, "
            "cycle_start_marker, pelec_transition_time, pelec_transition_source) "
            "VALUES (?,?,?,?,?,?,?,?)", self.ROWS)
        conn.commit()
        conn.close()

        self._saved = {k: getattr(webapp, k) for k in
                       ('get_db_connection', 'get_database_path', '_read_excel_sheet')}
        webapp.get_db_connection = self._connect
        webapp.get_database_path = lambda: self.db
        webapp._read_excel_sheet = self.SHEET
        webapp.app.config['TESTING'] = False          # exercise the real error handling
        webapp.app.config['PROPAGATE_EXCEPTIONS'] = False
        self.client = webapp.app.test_client()
        return self

    def __exit__(self, *exc_info):
        for k, v in self._saved.items():
            setattr(webapp, k, v)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return False

    def _connect(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn

    def post(self, path, payload):
        """(response, parsed body or None) with the app's own stdout swallowed."""
        with contextlib.redirect_stdout(io.StringIO()):
            resp = self.client.post(path, json=payload)
            try:
                body = resp.get_json()
            except Exception:
                body = None
        return resp, body

    def period_types(self, rowid):
        conn = self._connect()
        try:
            rows = conn.execute("SELECT period_type FROM cycle_periods WHERE entry_rowid=? "
                                "AND detection_method=?",
                                (rowid, webapp.GUIDELINE_DETECTION_METHOD)).fetchall()
            return sorted(r['period_type'] for r in rows)
        finally:
            conn.close()

    def saved_spans(self, rowid):
        """[(period_type, start, end)] stored for one parent, ordered by start."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT period_type, start_time, end_time FROM cycle_periods "
                "WHERE entry_rowid=? AND detection_method=? ORDER BY start_time ASC, id ASC",
                (rowid, webapp.GUIDELINE_DETECTION_METHOD)).fetchall()
            return [(r['period_type'], r['start_time'], r['end_time']) for r in rows]
        finally:
            conn.close()

    def transition_of(self, rowid):
        conn = self._connect()
        try:
            r = conn.execute("SELECT pelec_transition_time, pelec_transition_source "
                             "FROM results WHERE rowid=?", (rowid,)).fetchone()
            return r['pelec_transition_time'], r['pelec_transition_source']
        finally:
            conn.close()


def test_api_file_default_makes_unknown_rows_continuous():
    with ApiHarness() as h:
        _, plain = h.post('/api/cycle_extract/propose_periods',
                          {'file_name': API_FILE, 'default_kind': ''})
        kinds = [e['kind'] for e in (plain or {}).get('entries', [])]
        check('without a default both rows stay unknown', kinds == ['unknown', 'unknown'], str(kinds))

        resp, d = h.post('/api/cycle_extract/propose_periods',
                         {'file_name': API_FILE, 'overrides': {}, 'default_kind': 'continuous'})
        check('propose answers JSON', resp.content_type.startswith('application/json'), resp.content_type)
        entries = (d or {}).get('entries', [])
        check('both rows come back', len(entries) == 2, str(len(entries)))
        check('the file-level default fills the kind',
              all(e['kind'] == 'continuous' for e in entries), str([e['kind'] for e in entries]))
        check('the row can say where its kind came from',
              all(e['kind_source'] == 'file-level default' for e in entries),
              str([e['kind_source'] for e in entries]))
        types = sorted({p['period_type'] for e in entries for p in e['periods']})
        check('clocks are H + eq/eval only, no D/S',
              types == ['equilibrium', 'evaluation', 'heating'], str(types))
        check('H is the whole parent window',
              all(e['h_start'] == e['start_time'] and e['h_end'] == e['end_time'] for e in entries),
              str([(e['h_start'], e['h_end']) for e in entries]))
        check('no transition is asked for', all(e['transition_time'] is None for e in entries),
              str([e['transition_time'] for e in entries]))

        # A kind on the row must survive the file-level default.
        _, d2 = h.post('/api/cycle_extract/propose_periods',
                       {'file_name': API_FILE, 'default_kind': 'continuous',
                        'overrides': {'1': {'kind': 'defrost'}}})
        by_row = {e['rowid']: e for e in (d2 or {}).get('entries', [])}
        check('the default does not overwrite the analyst\'s kind',
              by_row[1]['kind'] == 'defrost' and by_row[1]['kind_source'] == 'analyst',
              f"{by_row[1]['kind']} ({by_row[1]['kind_source']})")
        check('that row is back to D + H',
              'defrost' in {p['period_type'] for p in by_row[1]['periods']},
              str([p['period_type'] for p in by_row[1]['periods']]))
        check('the other row still follows the default', by_row[2]['kind'] == 'continuous',
              by_row[2]['kind'])


def test_api_save_all_without_ticks_saves_the_whole_file():
    with ApiHarness() as h:
        resp, d = h.post('/api/cycle_extract/save_all_periods',
                         {'file_name': API_FILE, 'rowids': None, 'default_kind': 'continuous'})
        check('save all answers JSON', resp.content_type.startswith('application/json'), resp.content_type)
        check('no tick means the whole file', (d or {}).get('saved_entries') == 2, str(d))
        check('nothing was skipped', (d or {}).get('skipped') == [], str((d or {}).get('skipped')))
        check('stored clocks are H + eq/eval only',
              h.period_types(1) == ['equilibrium', 'evaluation', 'heating'], str(h.period_types(1)))
        trans, source = h.transition_of(1)
        check('a continuous row stores no transition time', trans is None, str(trans))
        check('but is still marked manual, so Rebuild all leaves it alone',
              source == 'manual', str(source))

        # An empty list is a scope of "everything", not "nothing".
        _, d2 = h.post('/api/cycle_extract/save_all_periods',
                       {'file_name': API_FILE, 'rowids': [], 'default_kind': 'continuous'})
        check('an empty rowid list also means the whole file',
              (d2 or {}).get('saved_entries') == 2, str(d2))

    # Fresh database: nothing saved yet, so both rows really are unknown.
    with ApiHarness() as h:
        _, d3 = h.post('/api/cycle_extract/save_all_periods',
                       {'file_name': API_FILE, 'default_kind': ''})
        check('unknown rows are skipped, not failed',
              (d3 or {}).get('success') is True and len((d3 or {}).get('skipped', [])) == 2, str(d3))
        check('nothing was written for them', h.period_types(1) == [], str(h.period_types(1)))


def test_api_replies_are_json_even_when_the_request_fails():
    with ApiHarness() as h:
        broken = lambda: (_ for _ in ()).throw(RuntimeError('forced failure'))
        real = webapp.init_deviation_columns
        webapp.init_deviation_columns = broken
        try:
            resp, d = h.post('/api/cycle_extract/save_all_periods',
                             {'file_name': API_FILE, 'default_kind': 'continuous'})
        finally:
            webapp.init_deviation_columns = real
        check('a forced failure still answers JSON, not an HTML traceback',
              resp.content_type.startswith('application/json'), resp.content_type)
        check('the failure is reported',
              (d or {}).get('success') is False and 'forced failure' in (d or {}).get('message', ''),
              str(d))

        # A server started before the endpoint existed used to answer an HTML 404,
        # which the browser could only report as "Invalid JSON from server".
        resp404, d404 = h.post('/api/cycle_extract/save_all_periods_typo', {})
        check('an unknown API path answers JSON too',
              resp404.content_type.startswith('application/json'), resp404.content_type)
        check('and it names the likely cause',
              'restart' in (d404 or {}).get('message', ''), str(d404))

        with contextlib.redirect_stdout(io.StringIO()):
            page = h.client.get('/no_such_page')
        check('pages keep the normal HTML error page',
              page.content_type.startswith('text/html'), page.content_type)


# ---------------------------------------------------------------------------
# Cycle Extract: file load carries the saved clocks so show / Period
# layers can draw them without Propose, and the second D/S span the analyst
# placed is seeded back from those saved rows on the next Propose. Synthetic
# sheet and throwaway SQLite only.
# ---------------------------------------------------------------------------

SECOND_DROP_FILE = 'synthetic_second_drop_file.xlsx'


def _second_drop_sheet(file_name=None, data_set=None, usecols=None):
    """Three 200 min windows on one synthetic sheet.

    0-12000 opens in defrost (idle to 600 s) and drops again at 9600 s, so the
    detector has a second drop to find. 12000-24000 opens in defrost with no
    later drop. 24000-36000 opens in **heating** and has a single trailing
    defrost from 33000 s - the case that must not be read as "one span saved".
    """
    t = np.arange(0.0, 36001.0, 10.0)
    idle = ((t < 600.0)
            | ((t >= 9600.0) & (t < 10200.0))
            | ((t >= 12000.0) & (t < 12600.0))
            | (t >= 33000.0))
    return pd.DataFrame({
        'time_elapsed': t,
        'Electric power input (without correction)': np.where(idle, 0.2, 3.0),
    })


class SavedClocksHarness(ApiHarness):
    """ApiHarness on the second-drop sheet, with three defrost parents."""

    ROWS = [(SECOND_DROP_FILE, 1, 0.0, 12000.0, 'defrost_cycle', 'defrost_start', 600.0, 'auto'),
            (SECOND_DROP_FILE, 1, 12000.0, 24000.0, 'defrost_cycle', 'defrost_start', 12600.0, 'auto'),
            (SECOND_DROP_FILE, 1, 24000.0, 36000.0, 'defrost_cycle', 'defrost_end', 33000.0, 'auto')]
    SHEET = staticmethod(_second_drop_sheet)


def _ptypes(periods, want):
    return [p for p in periods or [] if p['period_type'] == want]


def test_existing_entries_carry_the_saved_clocks_without_a_sheet_read():
    """File load must answer with the stored clocks and read no sheet."""
    with SavedClocksHarness() as h:
        _, d0 = h.post('/api/cycle_extract/existing_entries', {'file_name': SECOND_DROP_FILE})
        entries0 = (d0 or {}).get('entries', [])
        check('file load returns every parent of the file', len(entries0) == 3, str(len(entries0)))
        check('with nothing saved yet no row claims saved clocks',
              all(e['has_saved'] is False and e['saved_periods'] == [] for e in entries0),
              str([(e['rowid'], e['has_saved']) for e in entries0]))

        _, sv = h.post('/api/cycle_extract/save_all_periods', {'file_name': SECOND_DROP_FILE, 'rowids': [1]})
        check('one row was saved', (sv or {}).get('saved_entries') == 1, str(sv))

        reads = []
        stub = webapp._read_excel_sheet

        def _watched(*a, **k):
            reads.append(a)
            return stub(*a, **k)

        webapp._read_excel_sheet = _watched
        try:
            _, d1 = h.post('/api/cycle_extract/existing_entries', {'file_name': SECOND_DROP_FILE})
        finally:
            webapp._read_excel_sheet = stub
        check('file load never walks the sheet - no re-derive on load', reads == [], str(reads))

        by_row = {e['rowid']: e for e in (d1 or {}).get('entries', [])}
        check('the saved row says has_saved', by_row[1]['has_saved'] is True,
              str(by_row[1]['has_saved']))
        types = sorted({p['period_type'] for p in by_row[1]['saved_periods']})
        check('and carries the stored period types, eq/eval included',
              types == ['defrost', 'equilibrium', 'evaluation', 'heating'], str(types))
        check('each saved period carries its times',
              all(p['start_time'] is not None and p['end_time'] is not None
                  for p in by_row[1]['saved_periods']),
              str(by_row[1]['saved_periods'][:2]))
        check('a sibling row without clocks stays has_saved false',
              by_row[2]['has_saved'] is False and by_row[2]['saved_periods'] == [],
              str(by_row[2]['has_saved']))


def test_saved_second_d_span_survives_the_next_propose():
    """A typed ds2_start is stored only as cycle_periods - seed it back."""
    with SavedClocksHarness() as h:
        _, d0 = h.post('/api/cycle_extract/propose_periods', {'file_name': SECOND_DROP_FILE, 'rowids': [1]})
        det = _ptypes((d0 or {}).get('entries', [{}])[0].get('periods'), 'defrost')
        check('detection on its own finds the later drop near 9600 s',
              len(det) == 2 and 9500.0 <= det[1]['start_time'] <= 9750.0, str(det))

        _, sv = h.post('/api/cycle_extract/save_all_periods',
                       {'file_name': SECOND_DROP_FILE, 'rowids': [1],
                        'overrides': {'1': {'ds2_start': 6000}}})
        check('the typed two-span row saved', (sv or {}).get('saved_entries') == 1, str(sv))
        stored = [x for x in h.saved_spans(1) if x[0] == 'defrost']
        check('both D spans are stored, the second at the typed time',
              len(stored) == 2 and stored[1][1] == 6000.0, str(stored))

        # A reloaded page sends no overrides at all.
        _, d1 = h.post('/api/cycle_extract/propose_periods', {'file_name': SECOND_DROP_FILE, 'rowids': [1]})
        e1 = (d1 or {}).get('entries', [{}])[0]
        ds = _ptypes(e1.get('periods'), 'defrost')
        check('the next Propose keeps the saved second D start, not the detected drop',
              len(ds) == 2 and ds[1]['start_time'] == 6000.0, str(ds))
        check('the row says the span came from the saved clocks',
              e1.get('second_drop_source') == 'saved clocks', str(e1.get('second_drop_source')))
        check('H still ends at that second D',
              any(p['end_time'] == 6000.0 for p in _ptypes(e1.get('periods'), 'heating')),
              str(_ptypes(e1.get('periods'), 'heating')))
        check('kind still comes from the stored data, never from a test letter',
              e1.get('kind') == 'defrost' and e1.get('kind_source') == 'stored cycle_type',
              '{} ({})'.format(e1.get('kind'), e1.get('kind_source')))

        # The analyst typing in this session still wins over the seed.
        _, d2 = h.post('/api/cycle_extract/propose_periods',
                       {'file_name': SECOND_DROP_FILE, 'rowids': [1],
                        'overrides': {'1': {'ds2_start': 7500}}})
        e2 = (d2 or {}).get('entries', [{}])[0]
        ds2 = _ptypes(e2.get('periods'), 'defrost')
        check('a typed ds2_start still beats the saved seed',
              len(ds2) == 2 and ds2[1]['start_time'] == 7500.0, str(ds2))
        check('and is reported as the analyst edit',
              e2.get('second_drop_source') == 'analyst', str(e2.get('second_drop_source')))

        # reset drops the override *and* the seed, then detects again.
        _, d3 = h.post('/api/cycle_extract/propose_periods',
                       {'file_name': SECOND_DROP_FILE, 'rowids': [1], 'redetect_ds2': [1]})
        ds3 = _ptypes((d3 or {}).get('entries', [{}])[0].get('periods'), 'defrost')
        check('reset ignores the saved seed and re-detects',
              len(ds3) == 2 and 9500.0 <= ds3[1]['start_time'] <= 9750.0, str(ds3))


def test_saved_one_span_defrost_is_not_re_split():
    """A cycle the analyst forced to one span must stay one span."""
    with SavedClocksHarness() as h:
        _, sv = h.post('/api/cycle_extract/save_all_periods',
                       {'file_name': SECOND_DROP_FILE, 'rowids': [1],
                        'overrides': {'1': {'no_second_ds': True}}})
        check('the one-span row saved', (sv or {}).get('saved_entries') == 1, str(sv))
        stored = [x for x in h.saved_spans(1) if x[0] == 'defrost']
        check('exactly one D span is stored, the leading buffer',
              len(stored) == 1 and stored[0][1] == 0.0, str(stored))

        _, d1 = h.post('/api/cycle_extract/propose_periods', {'file_name': SECOND_DROP_FILE, 'rowids': [1]})
        e1 = (d1 or {}).get('entries', [{}])[0]
        ds = _ptypes(e1.get('periods'), 'defrost')
        check('Propose does not re-insert the detected drop', len(ds) == 1, str(ds))
        check('and says the single span came from the saved clocks',
              'one span saved' in (e1.get('second_drop_source') or ''),
              str(e1.get('second_drop_source')))
        check('H runs to the parent end again',
              any(p['end_time'] == 12000.0 for p in _ptypes(e1.get('periods'), 'heating')),
              str(_ptypes(e1.get('periods'), 'heating')))

        _, d2 = h.post('/api/cycle_extract/propose_periods',
                       {'file_name': SECOND_DROP_FILE, 'rowids': [1], 'redetect_ds2': [1]})
        ds2 = _ptypes((d2 or {}).get('entries', [{}])[0].get('periods'), 'defrost')
        check('reset finds the later drop again', len(ds2) == 2, str(ds2))


def test_opens_in_h_with_one_trailing_d_is_not_seeded():
    """A window that opens in H is not the one-saved-span case."""
    with SavedClocksHarness() as h:
        _, sv = h.post('/api/cycle_extract/save_all_periods', {'file_name': SECOND_DROP_FILE, 'rowids': [3]})
        check('the opens-in-H row saved', (sv or {}).get('saved_entries') == 1, str(sv))

        _, d = h.post('/api/cycle_extract/propose_periods', {'file_name': SECOND_DROP_FILE, 'rowids': [3]})
        e = (d or {}).get('entries', [{}])[0]
        check('the window still opens in H', e.get('begins_in') == 'h', str(e.get('begins_in')))
        check('no second-span seed is applied to an opens-in-H row',
              e.get('second_drop_source') is None, str(e.get('second_drop_source')))
        ds = _ptypes(e.get('periods'), 'defrost')
        check('the leftover buffer and the trailing D both stay',
              len(ds) == 2 and ds[0]['start_time'] == 24000.0 and ds[1]['start_time'] == 33000.0,
              str(ds))


def test_saved_clock_seed_helper_reads_only_d_s_spans():
    """The saved-clock seed itself: two spans, one leading span, nothing usable."""
    two = [{'period_type': 'defrost', 'start_time': 0.0, 'end_time': 1200.0},
           {'period_type': 'heating', 'start_time': 1200.0, 'end_time': 6000.0},
           {'period_type': 'defrost', 'start_time': 6000.0, 'end_time': 12000.0}]
    check('two saved D spans seed the later start',
          webapp._saved_second_ds_seed(two, 0.0) == (6000.0, False),
          str(webapp._saved_second_ds_seed(two, 0.0)))

    one = [{'period_type': 'defrost', 'start_time': 0.0, 'end_time': 1200.0},
           {'period_type': 'heating', 'start_time': 1200.0, 'end_time': 12000.0}]
    check('a single leading D span means no second span',
          webapp._saved_second_ds_seed(one, 0.0) == (None, True),
          str(webapp._saved_second_ds_seed(one, 0.0)))

    off = [{'period_type': 'off', 'start_time': 0.0, 'end_time': 1200.0},
           {'period_type': 'off', 'start_time': 8000.0, 'end_time': 12000.0}]
    check('standby spans are read the same way as defrost ones',
          webapp._saved_second_ds_seed(off, 0.0) == (8000.0, False),
          str(webapp._saved_second_ds_seed(off, 0.0)))

    check('a row with no D/S clocks is not seeded at all',
          webapp._saved_second_ds_seed(
              [{'period_type': 'heating', 'start_time': 0.0, 'end_time': 12000.0}], 0.0)
          == (None, False))
    check('nothing saved means nothing seeded',
          webapp._saved_second_ds_seed([], 0.0) == (None, False))
    check('an unusable span is skipped, not guessed',
          webapp._saved_second_ds_seed(
              [{'period_type': 'defrost', 'start_time': None, 'end_time': 1200.0}], 0.0)
          == (None, False))


def test_cycle_extract_show_and_layers_can_use_saved_clocks():
    """In the template: show is no longer proposal-only and the
    layers fall back from the proposal to the saved bands."""
    html = open(os.path.join(REPO, 'flask_app', 'templates', 'cycle_extract.html'),
                encoding='utf-8').read()
    check('show is no longer hard-disabled for every row without a proposal',
          "(pr?'':' disabled')" not in html and "(pr ? '' : ' disabled')" not in html)
    check('show follows a proposal *or* saved clocks',
          "(canShow?'':' disabled')" in html
          and 'function hasClocksToShow' in html
          and '!!_clockProposals[rid] || hasSavedClocks(rid, entry)' in html)
    picker = html.split('function clockPeriodsToDraw')[1].split('function proposalDiffersFromSaved')[0]
    check('the picker draws the proposal when there is one',
          'if (pr) return pr.periods || [];' in picker, picker.strip()[:120])
    check('and falls back to the saved periods otherwise',
          'return savedPeriodsOf(rid, entry);' in picker, picker.strip()[:120])
    check('the Period layers loop goes through that picker',
          'clockPeriodsToDraw(e.rowid, e).forEach' in html)
    check('saved bands come from the server, never from the test letter',
          'saved_periods' in html and 'dev_test_condition' not in html.split('function layerOfPeriod')[1]
          .split('function renderExistingTable')[0])
    check('edit still needs a proposal',
          'if (pr && _clockExpanded[rid]) body.appendChild(buildClockRow(e, pr));' in html)
    check('Save clocks still needs a proposal',
          'async function saveClocks(rid) {' in html
          and 'if (!pr) return;' in html.split('async function saveClocks(rid) {')[1][:200])
    check('Save all keeps its proposal-only scope',
          'function tickedProposedRowids' in html
          and 'var ticked = tickedProposedRowids();' in html)
    check('reset asks the server to ignore the saved seed',
          'redetect_ds2: redetect || null' in html and 'proposeClocks([id], [id]);' in html)
    check('a proposal that differs from the saved rows is flagged, not auto-saved',
          'function proposalDiffersFromSaved' in html and '&ne; saved' in html)


def test_clock_layers_changed_nothing_outside_cycle_extract():
    """Saved-clock show/layers are Cycle Extract only - the neighbours are checked again here."""
    parent = webapp.load_permissible_deviations()
    check('parent dTreturn is still the full-cycle -2...+2 K',
          parent['dTreturn']['lower'] == -2.0 and parent['dTreturn']['upper'] == 2.0,
          str(parent['dTreturn']))
    check('parent DB/WB individual widths are unchanged at 1 K',
          float(parent['DB']['value']) == 1.0 and float(parent['WB']['value']) == 1.0,
          str([parent['DB']['value'], parent['WB']['value']]))

    app_src = open(os.path.join(REPO, 'flask_app', 'app.py'), encoding='utf-8').read()
    check('plot_deviation still reports band_mode interval / parent',
          "band_mode = 'interval' if clock_spans else 'parent'" in app_src
          and "'band_mode': band_mode," in app_src)
    check('no second figure per quantity crept in',
          'image_dbwb_interval' not in app_src and "'image_interval'" not in app_src)
    check('existing_entries still does not re-derive the whole file',
          '_guideline_proposals_for_file('
          not in app_src.split('def api_cycle_extract_existing_entries')[1]
          .split('def _guideline_pelec_for_window')[0])

    gw_html = open(os.path.join(REPO, 'flask_app', 'templates', 'guideline_windows.html'),
                   encoding='utf-8').read()
    check('Guideline Windows still has no plots and no Tsup %',
          '<img' not in gw_html.lower() and 'plot_deviation' not in gw_html
          and 'tsup_pct' not in gw_html)


# ---------------------------------------------------------------------------
# Whole-database clock batch: the cheap file list, `skip_saved` on the
# file save, and Clear. Two synthetic file names on a throwaway SQLite and a
# stubbed sheet — no lab workbook and no analyst database is opened.
# ---------------------------------------------------------------------------

BATCH_A = 'synthetic_batch_a.xlsx'
BATCH_B = 'synthetic_batch_b.xlsx'


class BatchHarness(ApiHarness):
    """Two files of two defrost parents each, on the second-drop sheet."""

    ROWS = [(BATCH_A, 1, 0.0, 12000.0, 'defrost_cycle', 'defrost_start', 600.0, 'auto'),
            (BATCH_A, 1, 12000.0, 24000.0, 'defrost_cycle', 'defrost_start', 12600.0, 'auto'),
            (BATCH_B, 1, 0.0, 12000.0, 'defrost_cycle', 'defrost_start', 600.0, 'auto'),
            (BATCH_B, 1, 12000.0, 24000.0, 'defrost_cycle', 'defrost_start', 12600.0, 'auto')]
    SHEET = staticmethod(_second_drop_sheet)


class BatchUnknownHarness(BatchHarness):
    """File A cannot be classified at all; file B is defrost."""

    ROWS = [(BATCH_A, 1, 0.0, 12000.0, None, None, None, None),
            (BATCH_A, 1, 12000.0, 24000.0, None, None, None, None),
            (BATCH_B, 1, 0.0, 12000.0, 'defrost_cycle', 'defrost_start', 600.0, 'auto'),
            (BATCH_B, 1, 12000.0, 24000.0, 'defrost_cycle', 'defrost_start', 12600.0, 'auto')]


class EmptyResultsHarness(BatchHarness):
    """A database with a results table and no rows at all."""

    ROWS = []


def _watch_sheet_reads():
    """(list of file names read, restore callable) around ``_read_excel_sheet``."""
    seen = []
    stub = webapp._read_excel_sheet

    def _watched(file_name=None, *a, **k):
        seen.append(file_name)
        return stub(file_name, *a, **k)

    webapp._read_excel_sheet = _watched
    return seen, (lambda: setattr(webapp, '_read_excel_sheet', stub))


def _batch_fill(h, files, default_kind=''):
    """What the browser loop does: one save_all_periods per file, skip_saved on."""
    out = []
    for f in files:
        _, d = h.post('/api/cycle_extract/save_all_periods',
                      {'file_name': f, 'rowids': None,
                       'default_kind': default_kind, 'skip_saved': True})
        out.append(d or {})
    return out


def _function_code(app_src, name):
    """One top-level function's code, docstring and neighbours stripped off.

    A check like "this route opens no sheet" must look at what the route *runs*,
    not at a docstring that names ``_read_excel_sheet`` to promise it does not.
    """
    body = app_src.split('def ' + name + '(', 1)[1]
    nl = chr(10)
    marks = (nl + '@app.route', nl + 'def ', nl + 'class ')
    stops = [body.index(s) for s in marks if s in body]
    if stops:
        body = body[:min(stops)]
    parts = body.split('"""')
    return parts[0] + ''.join(parts[2:]) if len(parts) >= 3 else body


def _period_ids(h, rowid):
    """The cycle_periods row ids of one parent - a rewrite changes them."""
    conn = h._connect()
    try:
        return [r[0] for r in conn.execute(
            "SELECT id FROM cycle_periods WHERE entry_rowid=? AND detection_method=? ORDER BY id",
            (rowid, webapp.GUIDELINE_DETECTION_METHOD)).fetchall()]
    finally:
        conn.close()


def _periods_of(h, rowid, method=None):
    conn = h._connect()
    try:
        if method is None:
            rows = conn.execute("SELECT period_type, detection_method, buffer_s FROM cycle_periods "
                                "WHERE entry_rowid=?", (rowid,)).fetchall()
        else:
            rows = conn.execute("SELECT period_type, detection_method, buffer_s FROM cycle_periods "
                                "WHERE entry_rowid=? AND detection_method=?",
                                (rowid, method)).fetchall()
        return [(r['period_type'], r['detection_method'], r['buffer_s']) for r in rows]
    finally:
        conn.close()


def test_batch_file_list_is_cheap_and_counts_what_is_saved():
    """The database fill list: one row per distinct results.file_name, no Plotdaten walk."""
    with BatchHarness() as h:
        seen, restore = _watch_sheet_reads()
        try:
            resp, d = h.post('/api/cycle_extract/database_clock_files', {})
        finally:
            restore()
        check('the file list answers JSON', resp.content_type.startswith('application/json'),
              resp.content_type)
        check('listing the database reads no sheet', seen == [], str(seen))
        files = {f['file_name']: f for f in (d or {}).get('files', [])}
        check('both files of the database are listed', sorted(files) == sorted([BATCH_A, BATCH_B]),
              str(sorted(files)))
        check('each file reports its parent count',
              files[BATCH_A]['n_entries'] == 2 and files[BATCH_B]['n_entries'] == 2, str(files))
        check('with nothing saved yet everything is unsaved',
              all(f['n_saved'] == 0 and f['n_unsaved'] == 2 for f in files.values()), str(files))
        check('the totals are the confirm numbers',
              (d or {}).get('n_files') == 2 and d.get('n_entries') == 4
              and d.get('n_saved') == 0 and d.get('n_unsaved') == 4, str(d))

        _, sv = h.post('/api/cycle_extract/save_all_periods', {'file_name': BATCH_A, 'rowids': [1]})
        check('one parent of file A was saved', (sv or {}).get('saved_entries') == 1, str(sv))
        _, d2 = h.post('/api/cycle_extract/database_clock_files', {})
        files2 = {f['file_name']: f for f in (d2 or {}).get('files', [])}
        check('the saved parent moves from unsaved to saved',
              files2[BATCH_A]['n_saved'] == 1 and files2[BATCH_A]['n_unsaved'] == 1,
              str(files2[BATCH_A]))
        check('the other file is untouched by that save',
              files2[BATCH_B]['n_saved'] == 0 and files2[BATCH_B]['n_unsaved'] == 2,
              str(files2[BATCH_B]))
        check('and the totals follow',
              d2.get('n_saved') == 1 and d2.get('n_unsaved') == 3, str(d2))

        # A leftover buffer must not make stored clocks look absent.
        conn = h._connect()
        try:
            conn.execute("UPDATE cycle_periods SET buffer_s=? WHERE entry_rowid=?", (12345, 1))
            conn.commit()
        finally:
            conn.close()
        _, d3 = h.post('/api/cycle_extract/database_clock_files', {})
        files3 = {f['file_name']: f for f in (d3 or {}).get('files', [])}
        check('clocks written at another buffer_s still count as saved',
              files3[BATCH_A]['n_saved'] == 1, str(files3[BATCH_A]))

    with EmptyResultsHarness() as h:
        resp, d = h.post('/api/cycle_extract/database_clock_files', {})
        check('an empty database is success with an empty list',
              (d or {}).get('success') is True and (d or {}).get('files') == []
              and d.get('n_unsaved') == 0, str(d))
        check('and it is still JSON', resp.content_type.startswith('application/json'),
              resp.content_type)


def test_batch_skip_saved_fills_the_gap_and_leaves_saved_rows_alone():
    """The fill writes the unsaved file and never re-derives the saved one."""
    with BatchHarness() as h:
        # File B is the "already saved" one, with a second D span the analyst
        # typed where detection would not have put one.
        _, sv = h.post('/api/cycle_extract/save_all_periods',
                       {'file_name': BATCH_B, 'rowids': None,
                        'overrides': {'3': {'ds2_start': 11000.0}}})
        check('file B starts fully saved', (sv or {}).get('saved_entries') == 2, str(sv))
        b_spans = {3: h.saved_spans(3), 4: h.saved_spans(4)}
        b_trans = {3: h.transition_of(3), 4: h.transition_of(4)}
        typed = [s for s in b_spans[3] if s[0] == 'defrost']
        check('the typed second D span is what is stored on file B',
              len(typed) == 2 and abs(typed[1][1] - 11000.0) < 1.0, str(typed))

        seen, restore = _watch_sheet_reads()
        try:
            replies = _batch_fill(h, [BATCH_A, BATCH_B])
        finally:
            restore()

        check('file A is written by the fill', replies[0].get('saved_entries') == 2, str(replies[0]))
        check('file B writes nothing', replies[1].get('saved_entries') == 0, str(replies[1]))
        check('and says why every row was left alone',
              len(replies[1].get('skipped', [])) == 2
              and all(s['reason'] == 'already has saved guideline clocks'
                      for s in replies[1]['skipped']), str(replies[1].get('skipped')))
        check('the already-saved file costs no sheet read',
              BATCH_B not in seen and BATCH_A in seen, str(seen))
        check('the typed second D span on file B is still there',
              h.saved_spans(3) == b_spans[3] and h.saved_spans(4) == b_spans[4],
              str(h.saved_spans(3)))
        check('and so are its stored transition times',
              h.transition_of(3) == b_trans[3] and h.transition_of(4) == b_trans[4],
              str(h.transition_of(3)))
        check('file A now carries clocks',
              h.period_types(1) and h.period_types(2), str(h.period_types(1)))

        # File-level Save all still overwrites, skip_saved off or omitted.
        before = _period_ids(h, 3)
        _, over = h.post('/api/cycle_extract/save_all_periods',
                         {'file_name': BATCH_B, 'rowids': None})
        check('without skip_saved the file save still rewrites a saved file',
              (over or {}).get('saved_entries') == 2 and (over or {}).get('skipped') == [],
              str(over))
        check('the stored rows really were replaced, not skipped',
              _period_ids(h, 3) and set(_period_ids(h, 3)).isdisjoint(before),
              str((before, _period_ids(h, 3))))
        check('the typed second span still survives a re-derive - that is the saved-span seed, not skip_saved',
              [s for s in h.saved_spans(3) if s[0] == 'defrost'] == typed,
              str(h.saved_spans(3)))


def test_batch_skip_saved_on_a_fully_saved_file_reads_no_sheet():
    """Nothing to do must cost zero Excel reads, not a silent re-derive."""
    with BatchHarness() as h:
        _, sv = h.post('/api/cycle_extract/save_all_periods', {'file_name': BATCH_A})
        check('file A is fully saved first', (sv or {}).get('saved_entries') == 2, str(sv))

        seen, restore = _watch_sheet_reads()
        try:
            resp, d = h.post('/api/cycle_extract/save_all_periods',
                             {'file_name': BATCH_A, 'rowids': None, 'skip_saved': True})
        finally:
            restore()
        check('the reply is JSON', resp.content_type.startswith('application/json'),
              resp.content_type)
        check('a fully saved file is a success that writes nothing',
              (d or {}).get('success') is True and d.get('saved_entries') == 0
              and d.get('saved_periods') == 0, str(d))
        check('every parent is reported as skipped', len((d or {}).get('skipped', [])) == 2,
              str(d.get('skipped')))
        check('and no sheet was opened at all', seen == [], str(seen))
        check('the buffer is still reported so the UI can name it',
              (d or {}).get('buffer_s') == webapp._guideline_buffer_s(), str(d.get('buffer_s')))


def test_batch_unknown_rows_and_one_bad_sheet_stop_only_their_own_file():
    """Database fill: kind still unknown is skipped; an unreadable sheet fails that file."""
    with BatchUnknownHarness() as h:
        replies = _batch_fill(h, [BATCH_A, BATCH_B], default_kind='')
        check('the unclassifiable file writes nothing',
              replies[0].get('success') is True and replies[0].get('saved_entries') == 0,
              str(replies[0]))
        check('its rows are skipped for the kind, not for being saved',
              len(replies[0].get('skipped', [])) == 2
              and all('kind' in s['reason'] for s in replies[0]['skipped']),
              str(replies[0].get('skipped')))
        check('the sibling file in the same loop still saves',
              replies[1].get('saved_entries') == 2, str(replies[1]))
        check('and only that file has clocks',
              h.period_types(1) == [] and h.period_types(3) != [], str(h.period_types(3)))

        # The confirm's "Treat unknown rows as" is what fills them.
        again = _batch_fill(h, [BATCH_A], default_kind='continuous')
        check('the default from the confirm fills the unknown rows',
              again[0].get('saved_entries') == 2, str(again[0]))
        check('and they store an H window with eq/eval, no D/S',
              h.period_types(1) == ['equilibrium', 'evaluation', 'heating'],
              str(h.period_types(1)))

    with BatchUnknownHarness() as h:
        good = webapp._read_excel_sheet

        def _bad_for_a(file_name=None, *a, **k):
            if file_name == BATCH_A:
                raise RuntimeError('synthetic unreadable sheet')
            return good(file_name, *a, **k)

        webapp._read_excel_sheet = _bad_for_a
        try:
            resp_a, da = h.post('/api/cycle_extract/save_all_periods',
                                {'file_name': BATCH_A, 'default_kind': 'continuous',
                                 'skip_saved': True})
            _, db = h.post('/api/cycle_extract/save_all_periods',
                           {'file_name': BATCH_B, 'default_kind': '', 'skip_saved': True})
        finally:
            webapp._read_excel_sheet = good
        check('an unreadable sheet fails JSON-style, not as an HTML page',
              resp_a.content_type.startswith('application/json')
              and (da or {}).get('success') is False, str(da))
        check('and it fails only its own file — the next file still saves',
              (db or {}).get('saved_entries') == 2, str(db))


def test_batch_clear_is_what_makes_a_pass_repeatable():
    """Clear drops the guideline rows and the stored split, nothing else."""
    with BatchHarness() as h:
        _, sv = h.post('/api/cycle_extract/save_all_periods',
                       {'file_name': BATCH_A, 'overrides': {'1': {'ds2_start': 11000.0}}})
        check('file A saved', (sv or {}).get('saved_entries') == 2, str(sv))
        h.post('/api/cycle_extract/save_all_periods', {'file_name': BATCH_B})
        typed = [s for s in h.saved_spans(1) if s[0] == 'defrost']
        check('rowid 1 carries the typed second D span',
              len(typed) == 2 and abs(typed[1][1] - 11000.0) < 1.0, str(typed))

        # A Period Statistics row and a guideline row left at another buffer.
        conn = h._connect()
        try:
            conn.execute("INSERT INTO cycle_periods (entry_rowid, period_type, start_time, "
                         "end_time, detection_method, buffer_s) VALUES (?,?,?,?,?,?)",
                         (1, 'defrost', 0.0, 600.0, 'auto_pelec', 0))
            conn.execute("INSERT INTO cycle_periods (entry_rowid, period_type, start_time, "
                         "end_time, detection_method, buffer_s) VALUES (?,?,?,?,?,?)",
                         (4, 'defrost', 12000.0, 12600.0, webapp.GUIDELINE_DETECTION_METHOD, 9999))
            conn.commit()
        finally:
            conn.close()

        seen, restore = _watch_sheet_reads()
        try:
            resp, d = h.post('/api/cycle_extract/clear_guideline_clocks', {'file_name': None})
        finally:
            restore()
        check('clear answers JSON', resp.content_type.startswith('application/json'),
              resp.content_type)
        check('clear reads no sheet', seen == [], str(seen))
        check('clear reports what it removed',
              (d or {}).get('cleared_entries') == 4 and (d or {}).get('cleared_periods', 0) > 0,
              str(d))
        _, listed = h.post('/api/cycle_extract/database_clock_files', {})
        check('no parent claims saved clocks any more',
              (listed or {}).get('n_saved') == 0 and listed.get('n_unsaved') == 4, str(listed))
        check('the guideline period rows are gone on every parent',
              all(h.period_types(r) == [] for r in (1, 2, 3, 4)),
              str([h.period_types(r) for r in (1, 2, 3, 4)]))
        check('a guideline row left at another buffer_s is gone too',
              _periods_of(h, 4, webapp.GUIDELINE_DETECTION_METHOD) == [],
              str(_periods_of(h, 4)))
        check('the stored power split no longer says manual, so detection may run again',
              all(h.transition_of(r) == (None, None) for r in (1, 2, 3, 4)),
              str([h.transition_of(r) for r in (1, 2, 3, 4)]))
        check('the Period Statistics row at buffer_s=0 stays',
              _periods_of(h, 1, 'auto_pelec') == [('defrost', 'auto_pelec', 0)],
              str(_periods_of(h, 1)))
        conn = h._connect()
        try:
            n_parents = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            markers = [r[0] for r in conn.execute(
                "SELECT cycle_start_marker FROM results ORDER BY rowid").fetchall()]
        finally:
            conn.close()
        check('the parent results rows are untouched', n_parents == 4, str(n_parents))
        check('and so is cycle_start_marker',
              markers == ['defrost_start'] * 4, str(markers))

        # The point of Clear: the gap-fill can now write fresh clocks.
        replies = _batch_fill(h, [BATCH_A, BATCH_B])
        check('after clear the fill writes clocks again',
              replies[0].get('saved_entries') == 2 and replies[1].get('saved_entries') == 2,
              str(replies))
        fresh = [s for s in h.saved_spans(1) if s[0] == 'defrost']
        check('and they are derived again, not the typed pass that was thrown away',
              len(fresh) == 2 and abs(fresh[1][1] - 11000.0) > 100.0, str(fresh))

        # Nothing left to clear on a file with no saved clocks.
        h.post('/api/cycle_extract/clear_guideline_clocks', {'file_name': None})
        _, empty = h.post('/api/cycle_extract/clear_guideline_clocks', {'file_name': None})
        check('clearing an already-clear database is success with zeros',
              (empty or {}).get('success') is True and empty.get('cleared_entries') == 0
              and empty.get('cleared_periods') == 0, str(empty))

    with BatchHarness() as h:
        h.post('/api/cycle_extract/save_all_periods', {'file_name': BATCH_A})
        h.post('/api/cycle_extract/save_all_periods', {'file_name': BATCH_B})
        _, d = h.post('/api/cycle_extract/clear_guideline_clocks', {'file_name': BATCH_A})
        check('clear with a file name clears only that file',
              (d or {}).get('cleared_entries') == 2, str(d))
        _, listed = h.post('/api/cycle_extract/database_clock_files', {})
        files = {f['file_name']: f for f in (listed or {}).get('files', [])}
        check('the other file keeps its clocks',
              files[BATCH_A]['n_saved'] == 0 and files[BATCH_B]['n_saved'] == 2, str(files))


def test_cycle_extract_has_the_database_batch_controls():
    """Fill and Clear in the template: two confirmed controls, no auto-run."""
    html = open(os.path.join(REPO, 'flask_app', 'templates', 'cycle_extract.html'),
                encoding='utf-8').read()
    check('the database fill control exists on Cycle Extract',
          'id="dbFillClocksBtn"' in html and 'Save clocks for the open database' in html)
    check('the clear control exists and is not the green save',
          'id="dbClearClocksBtn"' in html and 'Clear saved guideline clocks' in html
          and 'btn-outline-danger btn-sm mr-3" type="button"' in html)
    check('the fill is not the green Save all proposed button either',
          'id="dbFillClocksBtn" class="btn btn-outline-primary' in html)
    check('neither control is Guideline Windows',
          'gwLoadAll' not in html and 'guideline_windows' not in html)
    check('both need only an open database, not a loaded file',
          html.count('{% if not database_is_ready %}disabled{% endif %}') == 2)

    init = html.split("document.addEventListener('DOMContentLoaded'")[1]
    check('nothing in the page init runs the batch',
          'openDatabaseFillConfirm(); });' in init and 'runDatabaseFill(' not in init
          and 'runDatabaseClear(' not in init and 'clear_guideline_clocks' not in init)
    check('the page never calls the batch endpoints outside the confirmed handlers',
          html.count('/api/cycle_extract/save_all_periods') == 2
          and html.count('/api/cycle_extract/clear_guideline_clocks') == 1)

    fill = html.split('async function runDatabaseFill')[1].split('async function openDatabaseClearConfirm')[0]
    check('the fill loops one file at a time with skip_saved on',
          'skip_saved: true' in fill and 'file_name: f.file_name' in fill
          and 'rowids: null' in fill)
    check('and shows file i of n',
          "'File ' + (i + 1) + ' of ' + files.length" in fill)
    check('the database pass sends no browser overrides',
          '_clockOverrides' not in fill and 'redetect_ds2' not in fill)
    check('a failed file is listed and the loop continues',
          'failed.push(' in fill and 'continue;' not in fill)
    check('only the file the analyst already had open is refreshed',
          'if (touchedOpenFile)' in fill and 'loadExistingEntries();' in fill
          and 'load_data' not in fill)

    confirm = html.split('async function openDatabaseFillConfirm')[1].split('async function runDatabaseFill')[0]
    check('the fill confirm states the counts from the list endpoint',
          'd.n_unsaved' in confirm and 'd.n_saved' in confirm and 'd.n_files' in confirm)
    check('it says this is not the Guideline Windows page',
          'not</b> the <b>Guideline Windows</b> page' in confirm)
    check('nothing to write points at Save all proposed and at Clear',
          'Nothing to write' in confirm and 'Save all proposed' in confirm
          and 'Clear saved guideline clocks' in confirm)
    check('the confirm carries Treat unknown rows as',
          'dbClockKindRow' in confirm and 'dbClockKindSelect' in confirm
          and 'defaultKind()' in confirm)

    clear = html.split('async function openDatabaseClearConfirm')[1].split('async function runDatabaseClear')[0]
    check('the clear confirm names the saved count and what survives',
          'd.n_saved' in clear and 'Period Statistics' in clear and 'parent cycles' in clear.lower())
    check('the clear confirm does not save anything',
          'save_all_periods' not in clear and 'runDatabaseFill' not in clear)
    check('clear can be narrowed to the loaded file',
          'dbClockScopeSelect' in clear and 'this file only' in clear)

    run_clear = html.split('async function runDatabaseClear')[1].split('function wireExistingButtons')[0]
    check('clear never chains into the fill',
          'runDatabaseFill' not in run_clear and 'proposeClocks' not in run_clear)

    # The neighbours this job must not disturb.
    check('Period layers still start off',
          'id="periodLayersToggle">' in html and 'periodLayersToggle" checked' not in html)
    check('still no Tsup band on Cycle Extract',
          'tsup_band' not in html and "layer=\"tsup\"" not in html)
    check('still one Plotly figure on this page', html.count('Plotly.newPlot') == 1,
          str(html.count('Plotly.newPlot')))


def test_batch_changed_nothing_outside_cycle_extract():
    """The batch is Cycle Extract only — the neighbours are checked again."""
    app_src = open(os.path.join(REPO, 'flask_app', 'app.py'), encoding='utf-8').read()
    check('existing_entries still does not re-derive the whole file',
          '_guideline_proposals_for_file('
          not in app_src.split('def api_cycle_extract_existing_entries')[1]
          .split('def _guideline_pelec_for_window')[0])
    list_src = _function_code(app_src, 'api_cycle_extract_database_clock_files')
    check('the file list opens no sheet and derives nothing',
          '_read_excel_sheet' not in list_src and '_guideline_proposals_for_file' not in list_src,
          list_src[:80])
    clear_src = _function_code(app_src, 'api_cycle_extract_clear_guideline_clocks')
    check('clear opens no sheet either', '_read_excel_sheet' not in clear_src, clear_src[:80])
    check('clear deletes guideline periods only',
          'DELETE FROM cycle_periods WHERE {where}' in clear_src
          and 'detection_method=?' in clear_src and 'auto_pelec' not in clear_src)
    check('clear does not touch the parent results rows themselves',
          'DELETE FROM results' not in clear_src)
    check('no second clock writer was invented',
          app_src.count('def _build_guideline_clocks') == 1
          and app_src.count('def _persist_guideline_clocks') == 1)
    check('plot_deviation still reports band_mode interval / parent',
          "band_mode = 'interval' if clock_spans else 'parent'" in app_src
          and "'band_mode': band_mode," in app_src)

    parent = webapp.load_permissible_deviations()
    check('parent dTreturn is still the full-cycle -2...+2 K',
          parent['dTreturn']['lower'] == -2.0 and parent['dTreturn']['upper'] == 2.0,
          str(parent['dTreturn']))

    gw_html = open(os.path.join(REPO, 'flask_app', 'templates', 'guideline_windows.html'),
                   encoding='utf-8').read()
    check('Guideline Windows is still read-only, with no plots and no Tsup %',
          'save_all_periods' not in gw_html and 'propose_periods' not in gw_html
          and 'clear_guideline_clocks' not in gw_html
          and '<img' not in gw_html.lower() and 'tsup_pct' not in gw_html)
    check('and the batch added no nav item',
          'database_clock_files' not in open(os.path.join(REPO, 'flask_app', 'templates',
                                                          'base.html'), encoding='utf-8').read())


# ---------------------------------------------------------------------------
# The Guideline windows page reads back what Save stored — stored
# clocks, stored parent means, evaluation-window means. Throwaway SQLite and a
# stubbed sheet; never the analyst's database, never a lab workbook.
# ---------------------------------------------------------------------------

WINDOWS_FILE = 'synthetic_not_a_lab_file.xlsx'


def _ramp_sheet(file_name, data_set=None, usecols=None):
    """Monotonic ramps, so a mean identifies the window it was taken over."""
    t = np.arange(0.0, 24001.0, 10.0)
    return pd.DataFrame({
        'time_elapsed': t,
        'Ts Buh': 30.0 + t / 1000.0,
        'QCorrwBUH': 5.0 + t / 10000.0,
        'PCorrwBUH': 2.0 + t / 40000.0,
        'COPCorrwBUH': 3.0 + t / 20000.0,
    })


def sheet_mean(column, start, end):
    """Mean of a stubbed column over a window, sliced exactly as the app slices."""
    df = _ramp_sheet(WINDOWS_FILE)
    sel = df[(df['time_elapsed'] >= start) & (df['time_elapsed'] <= end)]
    return float(sel[column].mean())


def _uncorr_sheet(file_name=None, data_set=None, usecols=None, lab_corrected=False):
    """An uncorrected Q/P sheet: raw Q and P plus what the pump correction needs.

    No QCorrwBUH / PCorrwBUH timeseries and no BUH column — the case where the
    evaluation Q/P/COP used to stay empty. The columns nothing reads are filled
    from the configured average columns so the sheet stays valid if that list
    grows. Ramps, so a mean identifies the window it was taken over.
    """
    t = np.arange(0.0, 24001.0, 10.0)
    df = pd.DataFrame({'time_elapsed': t})
    for col in webapp.config['average_columns']:
        if col not in webapp.ANALYSIS_SUPPLIED_COLUMNS:
            df[col] = 1.0 + t / 100000.0
    df['T_supply'] = 30.0 + t / 1000.0
    df['T_return_emu'] = 25.0 + t / 1000.0
    df['volume flow'] = 1.0 + t / 100000.0
    df['mass flow'] = 997.0 * df['volume flow'] / 3600.0
    df['Pressure difference'] = 0.5 + t / 200000.0
    df['Heating Capacity (without corr)'] = 5.0 + t / 10000.0
    df['Electric Power Input (without correction)'] = 2.0 + t / 40000.0
    if lab_corrected:
        df['Heating Capacity (corrected)'] = 4.5 + t / 12000.0
        df['Electric Power Input (corrected)'] = 1.8 + t / 50000.0
    return df


def _lab_corr_sheet(file_name=None, data_set=None, usecols=None):
    """The same sheet, but the lab did deliver corrected Q and P."""
    return _uncorr_sheet(file_name, data_set, usecols, lab_corrected=True)


def _unreadable_sheet(file_name=None, data_set=None, usecols=None):
    return None


def sheet_mean_of(sheet, column, start, end):
    """Mean of a column of that sheet over a window, sliced as the app slices."""
    df = sheet(WINDOWS_FILE)
    sel = df[(df['time_elapsed'] >= start) & (df['time_elapsed'] <= end)]
    return float(sel[column].mean())


def analysis_four(sheet, start, end):
    """The four evaluation means the extracted no-write helper produces."""
    with contextlib.redirect_stdout(io.StringIO()):
        df, lab_corr_missing, reason = webapp._prepare_sheet_for_analysis(
            sheet(WINDOWS_FILE), WINDOWS_FILE)
        means = webapp._analysis_means_for_window(df, WINDOWS_FILE, start, end, lab_corr_missing)
    assert not reason, reason
    return (webapp.get_mean_supply_for_deviations(means), webapp.get_mean_q_for_deviations(means),
            webapp.get_mean_p_for_deviations(means), webapp.get_mean_cop_for_deviations(means))


class WindowsHarness:
    """Test client on a temp database holding parent rows and saved clocks."""

    def __init__(self, sheet=None):
        self.sheet = sheet or _ramp_sheet

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp(prefix='guideline_windows_')
        self.db = os.path.join(self.tmpdir, 'test.db')
        self.sheet_reads = []
        conn = sqlite3.connect(self.db)
        conn.execute(
            "CREATE TABLE results (file_name TEXT, data_set REAL, start_time REAL, end_time REAL, "
            "display_order INTEGER, test_cond TEXT, dev_test_condition TEXT, profile_id TEXT, HP_ID TEXT, "
            "cycle_type TEXT, cycle_start_marker TEXT, pelec_transition_time REAL, "
            "pelec_transition_source TEXT, pelec_detect_failed INTEGER, avg_ts_buh REAL, "
            "avg_heating_capacity_corr_wbuh REAL, PCorrwBUH REAL, COPCorrwBUH REAL, "
            "COP_dataset TEXT)")
        conn.commit()
        conn.close()

        self._saved = {k: getattr(webapp, k) for k in
                       ('get_db_connection', 'get_database_path', '_read_excel_sheet',
                        'load_time_series_data')}
        webapp.get_db_connection = self._connect
        webapp.get_database_path = lambda: self.db
        webapp._read_excel_sheet = self._read_sheet
        webapp.load_time_series_data = (
            lambda fn, st, et, data_set=None, skip_derived=False: self.sheet(fn, data_set))
        webapp.app.config['TESTING'] = False
        webapp.app.config['PROPAGATE_EXCEPTIONS'] = False
        self.client = webapp.app.test_client()
        with contextlib.redirect_stdout(io.StringIO()):
            webapp.ensure_cycle_periods_table()
        self.buffer_s = webapp._guideline_buffer_s()
        return self

    def __exit__(self, *exc_info):
        for k, v in self._saved.items():
            setattr(webapp, k, v)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return False

    def _connect(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn

    def _read_sheet(self, file_name, data_set=None, usecols=None):
        self.sheet_reads.append((file_name, data_set))
        return self.sheet(file_name, data_set, usecols)

    def add_entry(self, rowid, start, end, order=None, tsup=None, q=None, p=None, cop=None,
                  cop_dataset=None):
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO results (rowid, file_name, data_set, start_time, end_time, display_order, "
                "test_cond, profile_id, HP_ID, avg_ts_buh, avg_heating_capacity_corr_wbuh, PCorrwBUH, "
                "COPCorrwBUH, COP_dataset) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rowid, WINDOWS_FILE, 1, start, end, order if order is not None else rowid,
                 'E', 'HPT_RRT1', '1', tsup, q, p, cop, cop_dataset))
            conn.commit()
        finally:
            conn.close()

    def add_period(self, rowid, period_type, start, end, cache=None, method=None, buffer_s=None):
        """One stored sub-period, as Save writes it (caches optional)."""
        cache = cache or {}
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO cycle_periods (entry_rowid, period_type, start_time, end_time, "
                "detection_method, buffer_s, avg_ts_buh, avg_q_corr_wbuh, avg_p_corr_wbuh, "
                "avg_cop_corr_wbuh) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (rowid, period_type, start, end,
                 method or webapp.GUIDELINE_DETECTION_METHOD,
                 self.buffer_s if buffer_s is None else buffer_s,
                 cache.get('tsup'), cache.get('q'), cache.get('p'), cache.get('cop')))
            conn.commit()
        finally:
            conn.close()

    def save_clocks(self, rowid, kind, periods, begins_in='ds', transition_time=None):
        """Cycle Extract's per-row Save — the path that also stores the scores."""
        with contextlib.redirect_stdout(io.StringIO()):
            resp = self.client.post('/api/cycle_extract/save_periods', json={
                'rowid': rowid, 'kind': kind, 'begins_in': begins_in,
                'transition_time': transition_time, 'periods': periods})
            try:
                body = resp.get_json()
            except Exception:
                body = None
        return body or {}

    def compute_scores(self, file_name=WINDOWS_FILE):
        """The explicit backfill: store this file's interval scores.

        Clock Save writes the same cache from the same helper; the tests add
        their clocks straight to SQLite, so this is how a synthetic database
        gets the scores the page reads back.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            resp = self.client.post('/api/guideline_windows/compute_scores',
                                    json={'file_name': file_name})
            try:
                body = resp.get_json()
            except Exception:
                body = None
        return body or {}

    def load(self, rowids, compute=True):
        """The table the page shows. ``compute`` False is a cold cache.

        Guideline Windows itself only ever reads the store, so a test that wants
        numbers has to have had them computed once — which is what Save does in
        the app. ``compute=False`` is the "clocks saved, scores not stored yet"
        case.
        """
        if compute:
            self.compute_scores()
        with contextlib.redirect_stdout(io.StringIO()):
            resp = self.client.post('/api/guideline_windows', json={'rowids': rowids})
            try:
                body = resp.get_json()
            except Exception:
                body = None
        return resp, (body or {})

    def one_row(self, rowid, compute=True):
        _, d = self.load([rowid], compute=compute)
        rows = d.get('rows') or []
        return rows[0] if rows else {}


def _defrost_entry(h):
    """Parent 0…12000 s cut at a defrost start: D 0…1200, H 1200…12000."""
    h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
    h.add_period(1, 'defrost', 0.0, 1200.0)
    h.add_period(1, 'heating', 1200.0, 12000.0)
    h.add_period(1, 'equilibrium', 1200.0, 4800.0)
    h.add_period(1, 'evaluation', 4800.0, 9000.0)


def test_windows_only_the_requested_rows_are_read():
    with WindowsHarness() as h:
        for rid in (1, 2, 3):
            h.add_entry(rid, 0.0, 12000.0, tsup=40.0 + rid)
            h.add_period(rid, 'heating', 0.0, 12000.0)
        _, d = h.load([1, 3])
        got = [r['rowid'] for r in d.get('rows', [])]
        check('only the selected rowids come back', got == [1, 3], str(got))
        check('the sibling row is absent', 2 not in got, str(got))


def test_windows_defrost_evaluation_is_the_stored_window():
    with WindowsHarness() as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('kind comes from the stored period types', row.get('kind') == 'defrost', str(row.get('kind')))
        check('the parent four are the stored means, not recomputed',
              (row.get('parent_tsup'), row.get('parent_q'), row.get('parent_p'), row.get('parent_cop'))
              == (40.0, 6.0, 2.5, 2.4),
              str([row.get('parent_tsup'), row.get('parent_q'), row.get('parent_p'), row.get('parent_cop')]))
        check('the stored evaluation clock is shown',
              (row.get('eval_start'), row.get('eval_end')) == (4800.0, 9000.0),
              str([row.get('eval_start'), row.get('eval_end')]))
        expect = sheet_mean('Ts Buh', 4800.0, 9000.0)
        check('eval Tsup is the mean over that window',
              abs(row.get('eval_tsup') - expect) < 1e-9, f"{row.get('eval_tsup')} vs {expect}")
        tail = sheet_mean('Ts Buh', 12000.0 - 4200.0, 12000.0)
        check('eval is not the last eval_min of H',
              abs(row.get('eval_tsup') - tail) > 1.0, f"{row.get('eval_tsup')} vs tail {tail}")
        check('eval Q, P and COP come from the same window',
              abs(row.get('eval_q') - sheet_mean('QCorrwBUH', 4800.0, 9000.0)) < 1e-9
              and abs(row.get('eval_p') - sheet_mean('PCorrwBUH', 4800.0, 9000.0)) < 1e-9
              and abs(row.get('eval_cop') - sheet_mean('COPCorrwBUH', 4800.0, 9000.0)) < 1e-9,
              str([row.get('eval_q'), row.get('eval_p'), row.get('eval_cop')]))
        check('the parent means are never overwritten by the evaluation ones',
              row.get('parent_tsup') != row.get('eval_tsup'),
              f"{row.get('parent_tsup')} / {row.get('eval_tsup')}")


def test_windows_on_off_has_empty_evaluation_cells():
    with WindowsHarness() as h:
        h.add_entry(1, 0.0, 12000.0, tsup=38.0, q=5.0, p=2.0, cop=2.5)
        h.add_period(1, 'off', 0.0, 1200.0)
        h.add_period(1, 'on', 1200.0, 12000.0)
        row = h.one_row(1)
        check('kind is on_off', row.get('kind') == 'on_off', str(row.get('kind')))
        check('the standby span is in the S columns',
              (row.get('s1_start'), row.get('s1_end')) == (0.0, 1200.0),
              str([row.get('s1_start'), row.get('s1_end')]))
        check('the D columns stay empty', row.get('d1_start') is None and row.get('d1_end') is None,
              str([row.get('d1_start'), row.get('d1_end')]))
        check('the eval four are empty, not 0',
              all(row.get(k) is None for k in ('eval_tsup', 'eval_q', 'eval_p', 'eval_cop')),
              str([row.get('eval_tsup'), row.get('eval_q'), row.get('eval_p'), row.get('eval_cop')]))
        check('the note gives the rule', 'on–off' in (row.get('note') or '').lower(), str(row.get('note')))


def test_windows_short_h_has_empty_evaluation_cells():
    with WindowsHarness() as h:
        # H = 1200…6000 s = 80 min, below eq_min + eval_min, so Save stored no evaluation.
        h.add_entry(1, 0.0, 6000.0, tsup=41.0, q=6.5, p=2.6, cop=2.5)
        h.add_period(1, 'defrost', 0.0, 1200.0)
        h.add_period(1, 'heating', 1200.0, 6000.0)
        h.add_period(1, 'equilibrium', 1200.0, 4800.0)
        row = h.one_row(1)
        check('the row is still a defrost cycle', row.get('kind') == 'defrost', str(row.get('kind')))
        check('no evaluation clock', row.get('eval_start') is None and row.get('eval_end') is None,
              str([row.get('eval_start'), row.get('eval_end')]))
        check('the eval four are empty, not 0',
              all(row.get(k) is None for k in ('eval_tsup', 'eval_q', 'eval_p', 'eval_cop')),
              str([row.get('eval_tsup'), row.get('eval_q'), row.get('eval_p'), row.get('eval_cop')]))
        check('the note says H was too short', 'shorter' in (row.get('note') or ''), str(row.get('note')))
        check('the parent four are still there', row.get('parent_cop') == 2.5, str(row.get('parent_cop')))


def test_windows_continuous_shows_its_saved_evaluation():
    with WindowsHarness() as h:
        h.add_entry(1, 0.0, 12000.0, tsup=39.0, q=5.5, p=2.2, cop=2.5)
        h.add_period(1, 'heating', 0.0, 12000.0)
        h.add_period(1, 'equilibrium', 0.0, 3600.0)
        h.add_period(1, 'evaluation', 3600.0, 7800.0,
                     cache={'tsup': 35.5, 'q': 5.9, 'p': 2.3, 'cop': 2.57})
        row = h.one_row(1)
        check('heating without D or S is continuous', row.get('kind') == 'continuous', str(row.get('kind')))
        check('no D or S clocks',
              all(row.get(k) is None for k in ('d1_start', 'd2_start', 's1_start', 's2_start')),
              str([row.get('d1_start'), row.get('s1_start')]))
        check('H is the parent window', (row.get('h_start'), row.get('h_end')) == (0.0, 12000.0),
              str([row.get('h_start'), row.get('h_end')]))
        # The cache Save wrote is no longer allowed to answer for the
        # window while the sheet can be read — it holds Tsup but no Q/P/COP on a
        # such a row, and a partial cache used to pass as complete.
        check('the eval four are the stored window, not the cache',
              abs(row.get('eval_tsup') - sheet_mean('Ts Buh', 3600.0, 7800.0)) < 1e-9
              and abs(row.get('eval_q') - sheet_mean('QCorrwBUH', 3600.0, 7800.0)) < 1e-9
              and abs(row.get('eval_p') - sheet_mean('PCorrwBUH', 3600.0, 7800.0)) < 1e-9,
              str([row.get('eval_tsup'), row.get('eval_q'), row.get('eval_p')]))
        check('the stale cache does not win',
              (row.get('eval_tsup'), row.get('eval_q')) != (35.5, 5.9),
              str([row.get('eval_tsup'), row.get('eval_q')]))


def test_windows_two_span_defrost_shows_both():
    with WindowsHarness() as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0)
        h.add_period(1, 'defrost', 0.0, 1200.0)
        h.add_period(1, 'heating', 1200.0, 10000.0)
        h.add_period(1, 'defrost', 10000.0, 12000.0)
        h.add_period(1, 'equilibrium', 1200.0, 4800.0)
        h.add_period(1, 'evaluation', 4800.0, 9000.0, cache={'tsup': 34.0})
        row = h.one_row(1)
        check('D1 is the leftover buffer', (row.get('d1_start'), row.get('d1_end')) == (0.0, 1200.0),
              str([row.get('d1_start'), row.get('d1_end')]))
        check('D2 is the next defrost', (row.get('d2_start'), row.get('d2_end')) == (10000.0, 12000.0),
              str([row.get('d2_start'), row.get('d2_end')]))
        check('H is the gap between them', (row.get('h_start'), row.get('h_end')) == (1200.0, 10000.0),
              str([row.get('h_start'), row.get('h_end')]))
        check('eq and eval sit inside that H',
              row.get('eq_start') >= row.get('h_start') and row.get('eval_end') <= row.get('h_end'),
              str([row.get('eq_start'), row.get('eval_end')]))


def test_windows_row_without_clocks_still_shows_the_parent():
    with WindowsHarness() as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        # A legacy Period Statistics row must not be read as a guideline clock.
        h.add_period(1, 'defrost', 0.0, 900.0, method='auto_pelec')
        row = h.one_row(1)
        check('kind is unknown, never other', row.get('kind') == 'unknown', str(row.get('kind')))
        check('the clock cells are empty',
              all(row.get(k) is None for k in
                  ('d1_start', 'd2_start', 's1_start', 'h_start', 'eq_start', 'eval_start')),
              str([row.get('d1_start'), row.get('h_start'), row.get('eval_start')]))
        check('the parent four are still shown',
              (row.get('parent_tsup'), row.get('parent_cop')) == (40.0, 2.4),
              str([row.get('parent_tsup'), row.get('parent_cop')]))
        check('the note says where to save them', 'Cycle Extract' in (row.get('note') or ''),
              str(row.get('note')))


def test_windows_nothing_is_read_until_the_analyst_asks():
    with WindowsHarness() as h:
        _defrost_entry(h)
        with contextlib.redirect_stdout(io.StringIO()):
            page = h.client.get('/guideline_windows')
        check('the shell renders', page.status_code == 200, str(page.status_code))
        check('the shell reads no sheet', h.sheet_reads == [], str(h.sheet_reads))

        resp, d = h.load([], compute=False)
        check('an empty selection answers JSON', resp.content_type.startswith('application/json'),
              resp.content_type)
        check('an empty selection computes nothing', d.get('success') is True and d.get('rows') == [],
              str(d))
        check('and still reads no sheet', h.sheet_reads == [], str(h.sheet_reads))


def test_windows_toolbar_selection_opens_the_page_with_those_rowids():
    with WindowsHarness() as h:
        _defrost_entry(h)
        h.add_entry(2, 0.0, 12000.0, tsup=41.0)
        with contextlib.redirect_stdout(io.StringIO()):
            page = h.client.post('/guideline_windows',
                                 data={'file_names': WINDOWS_FILE, 'data_sets': '1', 'row_ids': '1'})
        body = page.get_data(as_text=True)
        check('the toolbar post opens the page', page.status_code == 200, str(page.status_code))
        check('the page carries the selected rowid', '[1]' in body, body[-400:])
        check('rendering the page reads no sheet', h.sheet_reads == [], str(h.sheet_reads))

        with contextlib.redirect_stdout(io.StringIO()):
            empty = h.client.post('/guideline_windows', data={})
        check('an empty selection goes back to Mean Values instead of an empty page',
              empty.status_code == 302, str(empty.status_code))


def test_windows_null_rowids_loads_every_parent():
    with WindowsHarness() as h:
        # display_order, not rowid, is the Mean Values order.
        h.add_entry(1, 0.0, 12000.0, order=2, tsup=40.0)
        h.add_entry(2, 0.0, 12000.0, order=1, tsup=41.0)
        h.add_period(2, 'heating', 0.0, 12000.0)
        _, d = h.load(None)
        got = [r['rowid'] for r in d.get('rows', [])]
        check('every parent in the open database is listed', sorted(got) == [1, 2], str(got))
        check('rows follow the Mean Values order', got == [2, 1], str(got))
        check('every kind is listed, not frost only',
              sorted(r['kind'] for r in d['rows']) == ['continuous', 'unknown'],
              str([r['kind'] for r in d['rows']]))


def test_windows_api_errors_are_json():
    with WindowsHarness() as h:
        _defrost_entry(h)
        real = webapp.ensure_cycle_periods_table
        webapp.ensure_cycle_periods_table = lambda: (_ for _ in ()).throw(RuntimeError('forced failure'))
        try:
            resp, d = h.load(None)
        finally:
            webapp.ensure_cycle_periods_table = real
        check('a forced failure still answers JSON, not an HTML traceback',
              resp.content_type.startswith('application/json'), resp.content_type)
        check('the failure is reported',
              d.get('success') is False and 'forced failure' in (d.get('message') or ''), str(d))
        check('and the table comes back empty rather than missing', d.get('rows') == [], str(d.get('rows')))


def test_windows_csv_export_is_the_loaded_table():
    with WindowsHarness() as h:
        _defrost_entry(h)
        with contextlib.redirect_stdout(io.StringIO()):
            resp = h.client.post('/api/guideline_windows/export', json={'rowids': [1]})
        body = resp.data.decode('utf-8-sig')
        header, first = body.splitlines()[0], body.splitlines()[1]
        check('a CSV comes back', resp.headers.get('Content-Disposition', '').startswith('attachment'),
              resp.headers.get('Content-Disposition', ''))
        check('the evaluation columns are not named after 70 min',
              'COP_eval_70' not in header, header)
        check('parent and evaluation means are separate columns',
              'parent_COP' in header and 'eval_COP' in header, header)
        check('H/eq/eval dTreturn % and eval ΔCOP are columns',
              'H_dTreturn_pct' in header and 'eq_dTreturn_pct' in header
              and 'eval_dTreturn_pct' in header and 'eval_dCOP_pct' in header, header)
        check('H/eq/eval DB, WB and flow % are columns',
              'H_DB_pct' in header and 'eq_WB_pct' in header and 'eval_flow_pct' in header, header)
        check('there is no Tsup % field',
              'Tsup_pct' not in header and 'Tsup %' not in header, header)
        check('the loaded row is in the file', first.startswith('1,'), first)


# ---------------------------------------------------------------------------
# The evaluation four are the same analysis-time quantities as the
# entry, computed on the evaluation window. Such a sheet carries neither
# QCorrwBUH nor lab-corrected Q/P, so reading the raw series left them empty.
# ---------------------------------------------------------------------------


def test_eval_means_on_an_uncorrected_sheet():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        _defrost_entry(h)          # eval window 4800…9000 s inside H 1200…12000 s
        row = h.one_row(1)
        four = (row.get('eval_tsup'), row.get('eval_q'), row.get('eval_p'), row.get('eval_cop'))
        check('a sheet with only uncorrected Q and P still fills the eval four',
              all(v is not None for v in four), str(four))
        expect = analysis_four(_uncorr_sheet, 4800.0, 9000.0)
        check('they are what the no-write analysis helper produces',
              all(abs(a - b) < 1e-9 for a, b in zip(four, expect)), f'{four} vs {expect}')

        q_uncorr = sheet_mean_of(_uncorr_sheet, 'Heating Capacity (without corr)', 4800.0, 9000.0)
        p_uncorr = sheet_mean_of(_uncorr_sheet, 'Electric Power Input (without correction)', 4800.0, 9000.0)
        check('the pump correction was applied, so Q and P are not the uncorrected means',
              row['eval_q'] < q_uncorr and row['eval_p'] < p_uncorr,
              f"{row['eval_q']} / {q_uncorr}, {row['eval_p']} / {p_uncorr}")
        check('but stay within a pump correction of them',
              q_uncorr - row['eval_q'] < 0.1 * q_uncorr and p_uncorr - row['eval_p'] < 0.1 * p_uncorr,
              f"{row['eval_q']} / {q_uncorr}, {row['eval_p']} / {p_uncorr}")
        check('COP is the ratio of the two means, not the mean of a ratio',
              abs(row['eval_cop'] - row['eval_q'] / row['eval_p']) < 1e-12,
              f"{row['eval_cop']} vs {row['eval_q'] / row['eval_p']}")
        check('eval Tsup is the supply mean of that window (no BUH on this sheet)',
              abs(row['eval_tsup'] - sheet_mean_of(_uncorr_sheet, 'T_supply', 4800.0, 9000.0)) < 1e-9,
              str(row['eval_tsup']))

        tail = analysis_four(_uncorr_sheet, 12000.0 - 4200.0, 12000.0)
        check('they are not the last eval_min of H',
              all(abs(a - b) > 1e-6 for a, b in zip(four, tail)), f'{four} vs tail {tail}')
        check('the parent four still come from results, not from the eval slice',
              (row.get('parent_tsup'), row.get('parent_q'), row.get('parent_p'), row.get('parent_cop'))
              == (40.0, 6.0, 2.5, 2.4),
              str([row.get('parent_tsup'), row.get('parent_q'), row.get('parent_p'), row.get('parent_cop')]))


def test_eval_means_prefer_lab_corrected_series():
    with WindowsHarness(sheet=_lab_corr_sheet) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        q_lab = sheet_mean_of(_lab_corr_sheet, 'Heating Capacity (corrected)', 4800.0, 9000.0)
        p_lab = sheet_mean_of(_lab_corr_sheet, 'Electric Power Input (corrected)', 4800.0, 9000.0)
        check('eval Q follows the lab-corrected series (plus BUH, which is zero here)',
              abs(row.get('eval_q') - q_lab) < 1e-9, f"{row.get('eval_q')} vs {q_lab}")
        check('eval P follows the lab-corrected series',
              abs(row.get('eval_p') - p_lab) < 1e-9, f"{row.get('eval_p')} vs {p_lab}")
        uncorr_q = analysis_four(_uncorr_sheet, 4800.0, 9000.0)[1]
        check('the pump correction is not used when the lab delivered corrected Q and P',
              abs(row.get('eval_q') - uncorr_q) > 1e-6,
              f"{row.get('eval_q')} vs pump-corrected {uncorr_q}")


def test_eval_means_ignore_a_tsup_only_cache():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        h.add_period(1, 'defrost', 0.0, 1200.0)
        h.add_period(1, 'heating', 1200.0, 12000.0)
        h.add_period(1, 'equilibrium', 1200.0, 4800.0)
        # What Save writes for such a defrost row: Tsup cached, Q/P/COP empty.
        h.add_period(1, 'evaluation', 4800.0, 9000.0, cache={'tsup': 99.0})
        row = h.one_row(1)
        check('a Tsup-only cache no longer counts as done',
              all(row.get(k) is not None for k in ('eval_q', 'eval_p', 'eval_cop')),
              str([row.get('eval_q'), row.get('eval_p'), row.get('eval_cop')]))
        check('and Tsup is recomputed with the entry\'s own quantities',
              abs(row.get('eval_tsup') - sheet_mean_of(_uncorr_sheet, 'T_supply', 4800.0, 9000.0)) < 1e-9,
              str(row.get('eval_tsup')))


def test_eval_means_when_the_sheet_cannot_be_read():
    # Since the score cache the sheet is read by Save / Compute missing
    # scores, never by the page. An unreadable sheet therefore stores nothing:
    # the file is reported as failed, the clocks stay, and the row keeps the
    # evaluation period's own stored mean with its score cells n/a.
    with WindowsHarness(sheet=_unreadable_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        h.add_period(1, 'defrost', 0.0, 1200.0)
        h.add_period(1, 'heating', 1200.0, 12000.0)
        h.add_period(1, 'evaluation', 4800.0, 9000.0, cache={'tsup': 34.5})
        report = h.compute_scores()
        check('the backfill reports the file instead of failing the run',
              report.get('success') is True and report.get('computed') == 0
              and any('could not be read' in x for x in (report.get('failed') or [])),
              str(report))
        row = h.one_row(1)
        check('the cached Tsup is still shown', row.get('eval_tsup') == 34.5, str(row.get('eval_tsup')))
        check('Q, P and COP stay empty rather than 0',
              all(row.get(k) is None for k in ('eval_q', 'eval_p', 'eval_cop')),
              str([row.get('eval_q'), row.get('eval_p'), row.get('eval_cop')]))
        check('the row says its scores are not stored',
              row.get('scores_cached') is False
              and 'Compute missing scores' in (row.get('delta_cop_reason') or ''),
              str(row.get('delta_cop_reason')))
        check('the table still lists the row with its parent means',
              row.get('parent_cop') == 2.4, str(row.get('parent_cop')))


def test_eval_means_stay_empty_without_an_evaluation_window():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=38.0, q=5.0, p=2.0, cop=2.5)
        h.add_period(1, 'off', 0.0, 1200.0)
        h.add_period(1, 'on', 1200.0, 12000.0)
        # H = 1200…6000 s, below eq_min + eval_min, so Save stored no evaluation.
        h.add_entry(2, 0.0, 6000.0, tsup=41.0, q=6.5, p=2.6, cop=2.5)
        h.add_period(2, 'defrost', 0.0, 1200.0)
        h.add_period(2, 'heating', 1200.0, 6000.0)
        h.add_period(2, 'equilibrium', 1200.0, 4800.0)
        _, d = h.load([1, 2])
        rows = {r['rowid']: r for r in d.get('rows', [])}
        for rid, why in ((1, 'on–off'), (2, 'short H')):
            check(f'the eval four are empty on {why}, never 0',
                  all(rows[rid].get(k) is None
                      for k in ('eval_tsup', 'eval_q', 'eval_p', 'eval_cop')),
                  str([rows[rid].get(k) for k in ('eval_tsup', 'eval_q', 'eval_p', 'eval_cop')]))
        check('the sheet is still read for H dTreturn even without an evaluation window',
              h.sheet_reads != [], str(h.sheet_reads))


# ---------------------------------------------------------------------------
# Permissible deviations per guideline interval. The parent table
# keeps scoring the whole entry; the extra block scores the saved H, equilibrium
# and evaluation windows with the Interval H bands, plus ΔCOP on the evaluation
# window. Throwaway SQLite + stubbed sheet, no lab data.
# ---------------------------------------------------------------------------

PD_FILE = 'synthetic_pd_E.xlsx'        # third part starts with E → test condition E
PD_END = 12000.0                       # parent window 0…12000 s
PD_D_END = 1200.0                      # D 0…1200 s, H 1200…12000 s
PD_EVAL = (4800.0, 9000.0)             # eq 1200…4800 s, evaluation 4800…9000 s
PD_Q0 = 5.0                            # uncorrected heating capacity, kW


def _pd_sheet(q_step=0.0, end=PD_END, bad_tsup_until=PD_D_END, dt_in_d=0.0, dt_in_h=0.0,
              db_in_d=0.0, db_in_h=0.0, wb_in_d=0.0, wb_in_h=0.0,
              d2_start=None, db_in_d2=None, dt_in_d2=None, wb_in_d2=None):
    """An uncorrected Q/P sheet whose supply temperature only misbehaves inside D.

    Flow, pressure difference and electric power are constant, so the pump
    correction is the same in every window and a step in the uncorrected heating
    capacity moves the COP by a known fraction. The step is lifted again before
    the end of the parent window, so H's own first/last five minutes give a
    different ΔCOP from the evaluation window's — that is what tells the two
    apart.

    ``dt_in_d`` / ``dt_in_h`` are the dTreturn offsets (K) inside D (t < 1200 s)
    and everywhere after: T_return,emu minus T_return,calc, with T_return,calc
    the set inlet. Default 0 / 0 keeps dTreturn at 0 so existing Tsup / ΔCOP
    checks are unchanged. ``d2_start`` optionally paints a second D region
    (t >= that time) so a two-span union can be scored as one interval.
    """
    cache = {}

    def sheet(file_name=None, data_set=None, usecols=None):
        if not cache:
            t = np.arange(0.0, end + 1.0, 10.0)
            tsup_set = webapp.get_tsup_setpoint('E', PD_FILE)
            db_set = webapp.unit_config.get_tdb_setpoint_for('E')
            df = pd.DataFrame({'time_elapsed': t})
            for col in webapp.config['average_columns']:
                if col not in webapp.ANALYSIS_SUPPLIED_COLUMNS:
                    df[col] = 1.0
            in_d1 = t < PD_D_END
            in_d2 = (t >= d2_start) if d2_start is not None else np.zeros(t.shape, dtype=bool)

            def _paint(in_d, in_h, in_d2_val):
                mid = np.where(in_d1, in_d, in_h)
                if d2_start is None:
                    return mid
                return np.where(in_d2, in_d if in_d2_val is None else in_d2_val, mid)

            df['T_outdoor (DB)'] = db_set + _paint(db_in_d, db_in_h, db_in_d2)
            df['T_outdoor (WB)'] = (db_set - 1.0) + _paint(wb_in_d, wb_in_h, wb_in_d2)
            # In band for the whole of H; 5 K high while the unit is defrosting.
            df['T_supply'] = np.where(t < bad_tsup_until, tsup_set + 5.0, tsup_set)
            df['Ts Buh'] = df['T_supply']
            tret_set = tsup_set - 5.0
            df['T_return_calc'] = tret_set
            df['T_return_emu'] = tret_set + _paint(dt_in_d, dt_in_h, dt_in_d2)
            df['volume flow'] = 1.0
            df['mass flow'] = 997.0 * df['volume flow'] / 3600.0
            df['Pressure difference'] = 0.5
            df['Heating Capacity (without corr)'] = np.where(
                (t >= 6000.0) & (t < 10000.0), PD_Q0 * (1.0 + q_step), PD_Q0)
            df['Electric Power Input (without correction)'] = 2.0
            cache['df'] = df
        return cache['df']

    return sheet


def _union_outside_pct(sheet, spans, column, setpoint, half):
    """Share of union-window samples outside setpoint ± half (same count as the app)."""
    df = sheet()
    t = pd.to_numeric(df['time_elapsed'], errors='coerce')
    mask = False
    for start, end in spans:
        part = (t >= start) & (t <= end)
        mask = part if mask is False else (mask | part)
    window = df.loc[mask]
    series = pd.to_numeric(window[column], errors='coerce')
    lo, hi = float(setpoint) - float(half), float(setpoint) + float(half)
    count = int(len(series[(series < lo) | (series > hi)]))
    return count / len(window) * 100.0


def _fixed_flow_stubs():
    """HPT_RRT1 is variable-flow; S flow % needs a fixed set to be scored at all."""
    saved_var = webapp._entry_is_variable_flow
    saved_set = webapp.get_flow_set_for_hp
    webapp._entry_is_variable_flow = lambda entry, flow_config=None: False
    webapp.get_flow_set_for_hp = (
        lambda hp_id, conn=None, file_name=None, profile_id=None: ('volume', 1.0))
    return saved_var, saved_set


def _restore_flow_stubs(saved_var, saved_set):
    webapp._entry_is_variable_flow = saved_var
    webapp.get_flow_set_for_hp = saved_set


def _pd_delta_cop(sheet, start, end, slice_s=300.0):
    """ΔCOP % the analysis-time path gives for one window, computed here."""
    with contextlib.redirect_stdout(io.StringIO()):
        df, lab_corr_missing, reason = webapp._prepare_sheet_for_analysis(sheet(PD_FILE), PD_FILE)
        assert df is not None, reason
        first = webapp._analysis_means_for_window(df, PD_FILE, start, start + slice_s, lab_corr_missing)
        last = webapp._analysis_means_for_window(df, PD_FILE, end - slice_s, end, lab_corr_missing)
    cop_first = webapp._interval_cop_from_means(first)
    cop_last = webapp._interval_cop_from_means(last)
    return 100.0 * (cop_last - cop_first) / cop_first


def _pd_q_step_for(target_pct):
    """The capacity step that puts ΔCOP at ``target_pct``.

    ΔCOP is linear in the step because the pump correction and the electric
    power are constant here, so one probe fixes the slope.
    """
    probe = 0.01
    slope = _pd_delta_cop(_pd_sheet(q_step=probe), *PD_EVAL) / probe
    return target_pct / slope


class DeviationsHarness:
    """Flask test client on a temp database, for the Deviations calculate path."""

    def __init__(self, sheet=None):
        self.sheet = sheet or _pd_sheet()

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp(prefix='interval_deviations_')
        self.db = os.path.join(self.tmpdir, 'test.db')
        conn = sqlite3.connect(self.db)
        # Identity and metadata, then every column Apply's insert fills, so the
        # same table serves both the Deviations path and calculate_and_insert.
        cols = ['file_name TEXT', 'data_set REAL', 'start_time REAL', 'end_time REAL',
                'display_order INTEGER', 'test_cond TEXT', 'profile_id TEXT', 'HP_ID TEXT',
                'cycle_type TEXT', 'pelec_transition_time REAL', 'pelec_transition_source TEXT',
                'avg_ts_buh REAL', 'avg_t_mean_log REAL', 't_mean_from_avgs REAL', 'avg_dt_ln REAL']
        named = {c.split()[0] for c in cols}
        for col in sorted(webapp._insert_column_names_from_config()) + webapp.optional_mean_db_columns():
            if col not in named:
                cols.append(f'"{col}" REAL')
                named.add(col)
        conn.execute(f"CREATE TABLE results ({', '.join(cols)})")
        conn.commit()
        conn.close()

        self._saved = {k: getattr(webapp, k) for k in
                       ('get_db_connection', 'get_database_path', '_read_excel_sheet',
                        'load_time_series_data')}
        webapp.get_db_connection = self._connect
        webapp.get_database_path = lambda: self.db
        webapp._read_excel_sheet = lambda fn, ds=None, usecols=None: self.sheet(fn, ds, usecols)
        webapp.load_time_series_data = lambda fn, st, et, data_set=None, skip_derived=False: self.sheet(fn)
        webapp.app.config['TESTING'] = False
        webapp.app.config['PROPAGATE_EXCEPTIONS'] = False
        self.client = webapp.app.test_client()
        with contextlib.redirect_stdout(io.StringIO()):
            webapp.ensure_cycle_periods_table()
        self.buffer_s = webapp._guideline_buffer_s()
        return self

    def __exit__(self, *exc_info):
        for k, v in self._saved.items():
            setattr(webapp, k, v)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return False

    def _connect(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn

    def add_entry(self, rowid, start=0.0, end=PD_END):
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO results (rowid, file_name, data_set, start_time, end_time, "
                "display_order, test_cond) VALUES (?,?,?,?,?,?,?)",
                (rowid, PD_FILE, 1, start, end, rowid, 'E'))
            conn.commit()
        finally:
            conn.close()

    def add_period(self, rowid, period_type, start, end):
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO cycle_periods (entry_rowid, period_type, start_time, end_time, "
                "detection_method, buffer_s) VALUES (?,?,?,?,?,?)",
                (rowid, period_type, start, end, webapp.GUIDELINE_DETECTION_METHOD, self.buffer_s))
            conn.commit()
        finally:
            conn.close()

    def calculate(self, rowids):
        with contextlib.redirect_stdout(io.StringIO()):
            resp = self.client.post('/calculate_deviations',
                                    json={'entry_ids': rowids, 'skip_existing': False})
            try:
                body = resp.get_json()
            except Exception:
                body = None
        return resp, (body or {})

    def parent_row(self, rowid):
        conn = self._connect()
        try:
            return dict(conn.execute("SELECT rowid, * FROM results WHERE rowid=?", (rowid,)).fetchone())
        finally:
            conn.close()

    def block(self, rowid):
        conn = self._connect()
        try:
            row = conn.execute("SELECT payload FROM entry_interval_deviations WHERE entry_rowid=?",
                               (rowid,)).fetchone()
            return json.loads(row['payload']) if row else None
        finally:
            conn.close()


def _pd_defrost_clocks(h, rowid=1):
    """D 0…1200 s, H 1200…12000 s, eq 1200…4800 s, evaluation 4800…9000 s."""
    h.add_period(rowid, 'defrost', 0.0, PD_D_END)
    h.add_period(rowid, 'heating', PD_D_END, PD_END)
    h.add_period(rowid, 'equilibrium', PD_D_END, PD_EVAL[0])
    h.add_period(rowid, 'evaluation', *PD_EVAL)


def _pd_interval(block, key):
    return {iv['key']: iv for iv in (block or {}).get('intervals', [])}.get(key)


def test_interval_pd_without_clocks_leaves_the_parent_alone():
    with DeviationsHarness() as h:
        h.add_entry(1)
        _resp, body = h.calculate([1])
        check('calculate still answers success', body.get('success') is True, str(body)[:200])
        parent = h.parent_row(1)
        check('the parent deviations were written as before',
              parent.get('dev_total_points') and parent.get('dev_tsup_percentage') is not None,
              str([parent.get('dev_total_points'), parent.get('dev_tsup_percentage')]))
        check('no guideline clocks means no extra block', h.block(1) is None, str(h.block(1)))
        check('and nothing is reported as an interval row', body.get('interval_rows') == 0, str(body))


def test_interval_pd_defrost_scores_h_eq_eval_with_the_h_bands():
    step = _pd_q_step_for(6.0)             # a clearly red ΔCOP
    with DeviationsHarness(sheet=_pd_sheet(q_step=step)) as h:
        h.add_entry(1)
        _pd_defrost_clocks(h)
        _resp, body = h.calculate([1])
        check('the entry is reported as having an extra block',
              body.get('interval_rows') == 1, str(body)[:200])

        block = h.block(1)
        check('kind comes from the stored period types', (block or {}).get('kind') == 'defrost',
              str((block or {}).get('kind')))
        scored = {iv['key'] for iv in block['intervals'] if iv['status'] == 'ok'}
        check('H, equilibrium and evaluation are scored',
              scored == {'H', 'equilibrium', 'evaluation'}, str(sorted(scored)))

        parent = h.parent_row(1)
        check('the parent still sees the D excursion (that is why it is the full cycle)',
              parent['dev_tsup_percentage'] > 5.0, str(parent['dev_tsup_percentage']))
        for key in ('H', 'equilibrium', 'evaluation'):
            iv = _pd_interval(block, key)
            check(f'{key} is scored on its own time mask, so Tsup is in band there',
                  iv['stats']['tsup_percentage'] == 0.0, str(iv['stats']['tsup_percentage']))
        check('each interval counts only its own points',
              (_pd_interval(block, 'equilibrium')['stats']['total_points']
               < _pd_interval(block, 'H')['stats']['total_points'] < parent['dev_total_points']),
              str([_pd_interval(block, 'equilibrium')['stats']['total_points'],
                   _pd_interval(block, 'H')['stats']['total_points'], parent['dev_total_points']]))
        check('equilibrium and evaluation reuse the Interval H bands',
              (_pd_interval(block, 'equilibrium')['stats']['tsup_setpoint']
               == _pd_interval(block, 'H')['stats']['tsup_setpoint']
               == parent['dev_tsup_setpoint']),
              str(_pd_interval(block, 'equilibrium')['stats']['tsup_setpoint']))

        dc = block['delta_cop']
        expect = _pd_delta_cop(_pd_sheet(q_step=step), *PD_EVAL)
        check('ΔCOP is the first five minutes of the evaluation window against its last five',
              abs(dc['value_pct'] - expect) < 1e-9, f"{dc['value_pct']} vs {expect}")
        check('the slices are inside the evaluation window',
              dc['first_window'] == [4800.0, 5100.0] and dc['last_window'] == [8700.0, 9000.0],
              str([dc['first_window'], dc['last_window']]))
        h_delta = _pd_delta_cop(_pd_sheet(q_step=step), PD_D_END, PD_END)
        check('it is not the first and last five minutes of H',
              abs(h_delta) < 1e-9 and abs(dc['value_pct']) > 1.0, f"H {h_delta}, eval {dc['value_pct']}")
        check('COP is the ratio of the two means',
              dc['cop_first'] is not None and dc['cop_last'] > dc['cop_first'],
              str([dc['cop_first'], dc['cop_last']]))

        for key in ('D', 'S'):
            iv = _pd_interval(block, key)
            check(f'{key} is not scored and says why', iv['stats'] is None, str(iv))
        check('D shows the saved span but no bands',
              _pd_interval(block, 'D')['status'] == 'no_bands'
              and 'not configured' in _pd_interval(block, 'D')['reason'],
              str(_pd_interval(block, 'D')))
        check('S is simply absent on a defrost cycle',
              _pd_interval(block, 'S')['status'] == 'absent', str(_pd_interval(block, 'S')))


def test_interval_pd_on_off_has_no_equilibrium_evaluation_or_delta_cop():
    with DeviationsHarness() as h:
        h.add_entry(1)
        h.add_period(1, 'off', 0.0, PD_D_END)
        h.add_period(1, 'on', PD_D_END, PD_END)
        h.calculate([1])
        block = h.block(1)
        check('kind is on_off', block['kind'] == 'on_off', str(block['kind']))
        check('H is still scored (the "on" span)',
              _pd_interval(block, 'H')['status'] == 'ok', str(_pd_interval(block, 'H')))
        for key in ('equilibrium', 'evaluation'):
            iv = _pd_interval(block, key)
            check(f'{key} is absent by rule on an on–off cycle',
                  iv['status'] == 'absent' and 'on–off' in iv['reason'], str(iv))
        check('ΔCOP is empty, not 0', block['delta_cop']['value_pct'] is None,
              str(block['delta_cop']))
        check('S shows the saved standby span but has no bands',
              _pd_interval(block, 'S')['status'] == 'no_bands', str(_pd_interval(block, 'S')))


def test_interval_pd_short_h_has_no_evaluation_or_delta_cop():
    with DeviationsHarness(sheet=_pd_sheet(end=6000.0)) as h:
        # H = 1200…6000 s, shorter than eq_min + eval_min, so Save stored no evaluation.
        h.add_entry(1, 0.0, 6000.0)
        h.add_period(1, 'defrost', 0.0, PD_D_END)
        h.add_period(1, 'heating', PD_D_END, 6000.0)
        h.add_period(1, 'equilibrium', PD_D_END, 4800.0)
        h.calculate([1])
        block = h.block(1)
        check('H and equilibrium are still scored',
              all(_pd_interval(block, k)['status'] == 'ok' for k in ('H', 'equilibrium')),
              str([_pd_interval(block, k)['status'] for k in ('H', 'equilibrium')]))
        ev = _pd_interval(block, 'evaluation')
        check('evaluation is absent and the reason names the H length',
              ev['status'] == 'absent' and 'shorter than' in ev['reason'], str(ev))
        check('ΔCOP is empty, not 0', block['delta_cop']['value_pct'] is None,
              str(block['delta_cop']))


def test_interval_pd_delta_cop_just_under_and_just_over_the_limit():
    limit = webapp.load_interval_deviations_config()['delta_cop_pct']
    check('the limit is the configured 2.5 %', abs(limit - 2.5) < 1e-12, str(limit))
    for target, expect_red in ((limit - 0.1, False), (limit + 0.1, True)):
        sheet = _pd_sheet(q_step=_pd_q_step_for(target))
        with DeviationsHarness(sheet=sheet) as h:
            h.add_entry(1)
            _pd_defrost_clocks(h)
            h.calculate([1])
            dc = h.block(1)['delta_cop']
            check(f'ΔCOP lands on the {target:.1f} % the fixture was built for',
                  abs(dc['value_pct'] - target) < 1e-6, str(dc['value_pct']))
            check(f'|ΔCOP| {"exceeds" if expect_red else "stays inside"} the {limit:g} % limit',
                  (abs(dc['value_pct']) > dc['limit_pct']) is expect_red,
                  f"{dc['value_pct']} vs {dc['limit_pct']}")


def test_interval_pd_short_evaluation_window_has_no_delta_cop():
    with DeviationsHarness() as h:
        h.add_entry(1)
        h.add_period(1, 'defrost', 0.0, PD_D_END)
        h.add_period(1, 'heating', PD_D_END, PD_END)
        h.add_period(1, 'equilibrium', PD_D_END, PD_EVAL[0])
        h.add_period(1, 'evaluation', 4800.0, 5220.0)   # 7 min: under two 5 min slices
        h.calculate([1])
        block = h.block(1)
        check('the evaluation window itself is still scored',
              _pd_interval(block, 'evaluation')['status'] == 'ok',
              str(_pd_interval(block, 'evaluation')['status']))
        dc = block['delta_cop']
        check('but ΔCOP is empty, not 0', dc['value_pct'] is None, str(dc))
        check('and the reason names the 10 min it needs', '10 min' in dc['reason'], dc['reason'])


def test_interval_pd_a_red_delta_cop_does_not_stop_an_insert():
    step = _pd_q_step_for(8.0)
    with DeviationsHarness(sheet=_pd_sheet(q_step=step)) as h:
        h.add_entry(1)
        _pd_defrost_clocks(h)
        h.calculate([1])
        dc = h.block(1)['delta_cop']
        check('the fixture really is red', abs(dc['value_pct']) > dc['limit_pct'], str(dc['value_pct']))

        # calculate_and_insert must not consult the interval block at all: make
        # every entry point into it raise, then run Apply's block on the same row.
        boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('interval code reached'))
        saved = {k: getattr(webapp, k) for k in
                 ('_recalculate_interval_deviations', '_interval_deviation_payload',
                  '_interval_delta_cop')}
        for name in saved:
            setattr(webapp, name, boom)
        try:
            conn = h._connect()
            with contextlib.redirect_stdout(io.StringIO()):
                # The frame Apply works on: process_file's preparation, then the window.
                df, _lab_corr_missing, reason = webapp._prepare_sheet_for_analysis(
                    h.sheet(PD_FILE), PD_FILE)
                assert df is not None, reason
                ok = webapp.calculate_and_insert(
                    conn.cursor(), df, PD_FILE, 1, 0.0, PD_END, update_existing=True)
            conn.commit()
            conn.close()
        finally:
            for name, value in saved.items():
                setattr(webapp, name, value)
        check('Apply still succeeds when the evaluation ΔCOP would be red', ok is True, str(ok))
        check('and the parent row got its means', h.parent_row(1).get('avg_t_supply') is not None,
              str(h.parent_row(1).get('avg_t_supply')))
        check('the stored extra block is untouched by the insert',
              h.block(1)['delta_cop']['value_pct'] == dc['value_pct'],
              str(h.block(1)['delta_cop']['value_pct']))


def test_interval_pd_page_no_longer_shows_the_extra_table():
    with DeviationsHarness() as h:
        h.add_entry(1)
        _pd_defrost_clocks(h)
        h.calculate([1])
        reads = []
        real = webapp._read_excel_sheet
        webapp._read_excel_sheet = lambda fn, ds=None, usecols=None: (
            reads.append(fn) or real(fn, ds, usecols))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                page = h.client.get('/deviations')
        finally:
            webapp._read_excel_sheet = real
        html = page.get_data(as_text=True)
        check('the page renders', page.status_code == 200, str(page.status_code))
        check('the extra interval table is gone', 'Deviations by guideline interval' not in html)
        check('a line points at Guideline Windows', 'Guideline Windows' in html)
        check('the parent table still has dTreturn %', 'dTreturn %' in html)
        check('Configure Deviation Bands is still there', 'Configure Deviation Bands' in html)
        check('rendering reads no sheet', reads == [], str(reads))


# ---------------------------------------------------------------------------
# H / eq / eval dTreturn % (Interval H ±0.5 K) and eval ΔCOP on
# Guideline Windows. Parent Deviations stays one table (dTreturn −2…+2 K).
# ---------------------------------------------------------------------------

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TSUP_PCT_KEYS = ('tsup_percentage', 'h_tsup_pct', 'eq_tsup_pct', 'eval_tsup_pct', 'tsup_pct')


def test_gw_dtreturn_on_saved_defrost_ignores_a_d_only_excursion():
    step = _pd_q_step_for(6.0)
    with WindowsHarness(sheet=_pd_sheet(q_step=step, dt_in_d=3.0, dt_in_h=0.0)) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        four = (row.get('h_dtreturn_pct'), row.get('eq_dtreturn_pct'),
                row.get('eval_dtreturn_pct'), row.get('eval_delta_cop'))
        check('saved defrost H+eq+eval fills the four new columns',
              all(v is not None for v in four), str(four))
        check('a return excursion only in D does not raise H dTreturn %',
              row['h_dtreturn_pct'] == 0.0, str(row['h_dtreturn_pct']))
        check('eq and eval sit inside H, so they stay 0 as well',
              row['eq_dtreturn_pct'] == 0.0 and row['eval_dtreturn_pct'] == 0.0,
              str([row['eq_dtreturn_pct'], row['eval_dtreturn_pct']]))
        expect = _pd_delta_cop(_pd_sheet(q_step=step, dt_in_d=3.0, dt_in_h=0.0), *PD_EVAL)
        check('eval ΔCOP is still filled from the evaluation window',
              abs(row['eval_delta_cop'] - expect) < 1e-9,
              f"{row['eval_delta_cop']} vs {expect}")


def test_gw_dtreturn_uses_the_half_k_h_band_not_the_parent_pm_2():
    parent_cfg = webapp.load_permissible_deviations()['dTreturn']
    check('parent dTreturn is still −2…+2 K',
          float(parent_cfg['lower']) == -2.0 and float(parent_cfg['upper']) == 2.0,
          str(parent_cfg))
    h_band = webapp._interval_dtreturn_band(webapp.load_interval_deviations_config(), 'H')
    check('Interval H inlet band is ±0.5 K', h_band == (-0.5, 0.5), str(h_band))

    sheet = _pd_sheet(dt_in_d=0.0, dt_in_h=1.0)
    with WindowsHarness(sheet=sheet) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('1 K inside H is outside ±0.5 K, so H dTreturn % is 100',
              row.get('h_dtreturn_pct') == 100.0, str(row.get('h_dtreturn_pct')))
        with contextlib.redirect_stdout(io.StringIO()):
            parent_h = webapp.calculate_entry_deviations(
                PD_FILE, 1, 1200.0, 12000.0, df=sheet(), entry={'test_cond': 'E'})
            narrow = webapp.calculate_entry_deviations(
                PD_FILE, 1, 1200.0, 12000.0, df=sheet(), entry={'test_cond': 'E'},
                dtreturn_band=h_band)
        check('the same 1 K is inside the parent ±2 K band',
              parent_h is not None and parent_h.get('dtreturn_percentage') == 0.0,
              str(None if parent_h is None else parent_h.get('dtreturn_percentage')))
        check('Guideline Windows used the interval-band override, not a second formula',
              narrow is not None
              and abs(narrow.get('dtreturn_percentage') - row['h_dtreturn_pct']) < 1e-9,
              str(None if narrow is None else narrow.get('dtreturn_percentage')))


def test_gw_on_off_and_short_h_leave_eval_dtreturn_and_delta_cop_empty():
    with WindowsHarness(sheet=_pd_sheet()) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=38.0, q=5.0, p=2.0, cop=2.5)
        h.add_period(1, 'off', 0.0, 1200.0)
        h.add_period(1, 'on', 1200.0, 12000.0)
        h.add_entry(2, 0.0, 6000.0, tsup=41.0, q=6.5, p=2.6, cop=2.5)
        h.add_period(2, 'defrost', 0.0, 1200.0)
        h.add_period(2, 'heating', 1200.0, 6000.0)
        h.add_period(2, 'equilibrium', 1200.0, 4800.0)
        _, d = h.load([1, 2])
        rows = {r['rowid']: r for r in d.get('rows', [])}

        off = rows[1]
        check('on–off: H dTreturn % is 0 (scored, no violations), not missing',
              off.get('h_dtreturn_pct') == 0.0, str(off.get('h_dtreturn_pct')))
        check('on–off: eq/eval dTreturn % and ΔCOP are n/a, never 0',
              all(off.get(k) is None
                  for k in ('eq_dtreturn_pct', 'eval_dtreturn_pct', 'eval_delta_cop')),
              str([off.get(k) for k in ('eq_dtreturn_pct', 'eval_dtreturn_pct', 'eval_delta_cop')]))

        short = rows[2]
        check('short H: equilibrium was saved, so eq dTreturn % is 0, not missing',
              short.get('eq_dtreturn_pct') == 0.0, str(short.get('eq_dtreturn_pct')))
        check('short H: eval dTreturn % and ΔCOP are n/a, never 0',
              short.get('eval_dtreturn_pct') is None and short.get('eval_delta_cop') is None,
              str([short.get('eval_dtreturn_pct'), short.get('eval_delta_cop')]))


def test_gw_delta_cop_is_from_eval_not_from_h():
    step = _pd_q_step_for(6.0)
    sheet = _pd_sheet(q_step=step)
    with WindowsHarness(sheet=sheet) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        expect = _pd_delta_cop(sheet, *PD_EVAL)
        check('ΔCOP is the first five minutes of eval against its last five',
              abs(row.get('eval_delta_cop') - expect) < 1e-9,
              f"{row.get('eval_delta_cop')} vs {expect}")
        h_delta = _pd_delta_cop(sheet, PD_D_END, PD_END)
        check('it is not the first and last five minutes of H',
              abs(h_delta) < 1e-9 and abs(row['eval_delta_cop']) > 1.0,
              f"H {h_delta}, eval {row['eval_delta_cop']}")


def test_gw_db_wb_on_saved_defrost_ignore_a_d_only_excursion():
    with WindowsHarness(sheet=_pd_sheet(db_in_d=5.0, db_in_h=0.0, wb_in_d=5.0, wb_in_h=0.0)) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('H/eq/eval DB % are scored (0, not missing) when the excursion is only in D',
              row.get('h_db_pct') == 0.0 and row.get('eq_db_pct') == 0.0
              and row.get('eval_db_pct') == 0.0,
              str([row.get('h_db_pct'), row.get('eq_db_pct'), row.get('eval_db_pct')]))
        check('H/eq/eval WB % likewise stay 0',
              row.get('h_wb_pct') == 0.0 and row.get('eq_wb_pct') == 0.0
              and row.get('eval_wb_pct') == 0.0,
              str([row.get('h_wb_pct'), row.get('eq_wb_pct'), row.get('eval_wb_pct')]))


def test_gw_db_outside_h_band_is_100_percent():
    with WindowsHarness(sheet=_pd_sheet(db_in_d=0.0, db_in_h=5.0)) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('5 K DB inside H is outside Interval H ±1 K, so H DB % is 100',
              row.get('h_db_pct') == 100.0, str(row.get('h_db_pct')))
        check('eq and eval sit inside H, so they are 100 as well',
              row.get('eq_db_pct') == 100.0 and row.get('eval_db_pct') == 100.0,
              str([row.get('eq_db_pct'), row.get('eval_db_pct')]))


def test_gw_on_off_leaves_eq_eval_db_wb_flow_empty():
    with WindowsHarness(sheet=_pd_sheet()) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=38.0, q=5.0, p=2.0, cop=2.5)
        h.add_period(1, 'off', 0.0, 1200.0)
        h.add_period(1, 'on', 1200.0, 12000.0)
        row = h.one_row(1)
        check('on–off: H DB % is scored', row.get('h_db_pct') == 0.0, str(row.get('h_db_pct')))
        check('on–off: eq/eval DB, WB, flow are n/a, never 0',
              all(row.get(k) is None for k in (
                  'eq_db_pct', 'eval_db_pct', 'eq_wb_pct', 'eval_wb_pct',
                  'eq_flow_pct', 'eval_flow_pct')),
              str([row.get(k) for k in (
                  'eq_db_pct', 'eval_db_pct', 'eq_wb_pct', 'eval_wb_pct',
                  'eq_flow_pct', 'eval_flow_pct')]))


def test_gw_db_uses_the_deviations_series_not_the_raw_excel_cache():
    """Deviations Calculate loads time_series_data / derived columns; the raw
    Excel cache used for dTreturn may not carry the outdoor/flow headers.
    Guideline Windows must score DB from the Deviations series."""
    with WindowsHarness(sheet=_pd_sheet()) as h:
        _defrost_entry(h)
        full = h.sheet(PD_FILE)
        drop = [c for c in ('T_outdoor (DB)', 'T_outdoor (WB)', 'mass flow', 'volume flow')
                if c in full.columns]
        raw = full.drop(columns=drop)
        webapp._read_excel_sheet = lambda fn, ds=None, usecols=None: raw
        row = h.one_row(1)
        check('H DB % is scored from the Deviations series even when the raw sheet has no outdoor column',
              row.get('h_db_pct') == 0.0, str(row.get('h_db_pct')))
        check('eq/eval DB % likewise',
              row.get('eq_db_pct') == 0.0 and row.get('eval_db_pct') == 0.0,
              str([row.get('eq_db_pct'), row.get('eval_db_pct')]))
        check('dTreturn still comes from the raw sheet',
              row.get('h_dtreturn_pct') == 0.0, str(row.get('h_dtreturn_pct')))


def test_interval_json_ds_widths_are_not_interval_h():
    cfg = webapp.load_interval_deviations_config()
    d = (cfg.get('intervals') or {}).get('D') or {}
    s = (cfg.get('intervals') or {}).get('S') or {}
    check('D individual DB is ±5 K', float(d.get('db_k')) == 5.0, str(d.get('db_k')))
    check('D individual dTreturn is ±2 K', float(d.get('dtreturn_k')) == 2.0, str(d.get('dtreturn_k')))
    check('S individual DB/WB are ±2 K',
          float(s.get('db_k')) == 2.0 and float(s.get('wb_k')) == 2.0,
          str([s.get('db_k'), s.get('wb_k')]))
    check('S individual flow is ±2.5 %',
          float(s.get('flow_instantaneous_pct')) == 2.5, str(s.get('flow_instantaneous_pct')))
    check('S individual dTreturn is ±1 K', float(s.get('dtreturn_k')) == 1.0, str(s.get('dtreturn_k')))
    check('D is not scored with Interval H bands',
          not webapp._interval_pd_has_bands(cfg, 'D'), str(d.get('bands')))
    check('S is not scored with Interval H bands',
          not webapp._interval_pd_has_bands(cfg, 'S'), str(s.get('bands')))
    parent = webapp.load_permissible_deviations()['dTreturn']
    check('parent dTreturn is still −2…+2 K',
          float(parent['lower']) == -2.0 and float(parent['upper']) == 2.0, str(parent))


def test_gw_defrost_scores_d_pct_and_leaves_s_na():
    with WindowsHarness(sheet=_pd_sheet(db_in_d=0.0, db_in_h=3.0, dt_in_d=0.0, dt_in_h=0.0)) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('D DB % and D dTreturn % are numbers',
              row.get('d_db_pct') is not None and row.get('d_dtreturn_pct') is not None,
              str([row.get('d_db_pct'), row.get('d_dtreturn_pct')]))
        check('an excursion only in H does not raise D DB %',
              row.get('d_db_pct') == 0.0, str(row.get('d_db_pct')))
        check('H DB % is 100 for the same 3 K (Interval H ±1 K)',
              row.get('h_db_pct') == 100.0, str(row.get('h_db_pct')))
        check('S % are n/a on a defrost row, never 0',
              all(row.get(k) is None for k in
                  ('s_db_pct', 's_wb_pct', 's_flow_pct', 's_dtreturn_pct')),
              str([row.get(k) for k in ('s_db_pct', 's_wb_pct', 's_flow_pct', 's_dtreturn_pct')]))
        check('S hover names the kind',
              'on–off' in (row.get('s_reasons') or {}).get('db', ''),
              str(row.get('s_reasons')))


def test_gw_d_db_width_is_five_k_not_interval_h():
    with WindowsHarness(sheet=_pd_sheet(db_in_d=3.0, db_in_h=0.0)) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('3 K DB inside D is inside D ±5 K, so D DB % is 0',
              row.get('d_db_pct') == 0.0, str(row.get('d_db_pct')))
        check('the same 3 K would be outside Interval H ±1 K (H stays 0 because it is not in H)',
              row.get('h_db_pct') == 0.0, str(row.get('h_db_pct')))


def test_gw_d_dtreturn_width_is_two_k_not_interval_h():
    with WindowsHarness(sheet=_pd_sheet(dt_in_d=1.5, dt_in_h=0.0)) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('1.5 K dTreturn inside D is inside D ±2 K, so D dTreturn % is 0',
              row.get('d_dtreturn_pct') == 0.0, str(row.get('d_dtreturn_pct')))
        check('H dTreturn % stays 0 (the 1.5 K is not in H)',
              row.get('h_dtreturn_pct') == 0.0, str(row.get('h_dtreturn_pct')))


def test_gw_two_d_spans_are_one_union_percent():
    d2_start = 10000.0
    sheet = _pd_sheet(db_in_d=0.0, db_in_h=0.0, d2_start=d2_start, db_in_d2=6.0)
    spans = [(0.0, PD_D_END), (d2_start, PD_END)]
    db_set = webapp.unit_config.get_tdb_setpoint_for('E')
    expect = _union_outside_pct(sheet, spans, 'T_outdoor (DB)', db_set, 5.0)
    with WindowsHarness(sheet=sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0)
        h.add_period(1, 'defrost', 0.0, PD_D_END)
        h.add_period(1, 'heating', PD_D_END, d2_start)
        h.add_period(1, 'defrost', d2_start, PD_END)
        h.add_period(1, 'equilibrium', PD_D_END, PD_EVAL[0])
        h.add_period(1, 'evaluation', *PD_EVAL)
        row = h.one_row(1)
        check('two D spans still yield one D DB %',
              'd2_db_pct' not in row and row.get('d_db_pct') is not None,
              str([k for k in row if 'd' in k and 'pct' in k]))
        check('that one D % is the union (excursion only in the second span)',
              abs(row.get('d_db_pct') - expect) < 1e-9,
              f"{row.get('d_db_pct')} vs union {expect}")
        check('D1 alone would have been 0 (the excursion is only in D2)',
              expect > 0.0, str(expect))


def test_gw_on_off_scores_s_pct_and_leaves_d_na():
    sheet = _pd_sheet(db_in_d=0.0, db_in_h=0.0, wb_in_d=0.0, wb_in_h=0.0, dt_in_d=0.75, dt_in_h=0.0)
    saved = _fixed_flow_stubs()
    try:
        with WindowsHarness(sheet=sheet) as h:
            h.add_entry(1, 0.0, 12000.0, tsup=38.0, q=5.0, p=2.0, cop=2.5)
            h.add_period(1, 'off', 0.0, 1200.0)
            h.add_period(1, 'on', 1200.0, 12000.0)
            row = h.one_row(1)
            check('S DB/WB/flow/dTreturn are scored',
                  all(row.get(k) is not None for k in
                      ('s_db_pct', 's_wb_pct', 's_flow_pct', 's_dtreturn_pct')),
                  str([row.get(k) for k in
                       ('s_db_pct', 's_wb_pct', 's_flow_pct', 's_dtreturn_pct')]))
            check('S DB/WB/flow stay 0 when the series is in band',
                  row.get('s_db_pct') == 0.0 and row.get('s_wb_pct') == 0.0
                  and row.get('s_flow_pct') == 0.0,
                  str([row.get('s_db_pct'), row.get('s_wb_pct'), row.get('s_flow_pct')]))
            check('0.75 K in S is inside S ±1 K, so S dTreturn % is 0 (would be outside H ±0.5 K)',
                  row.get('s_dtreturn_pct') == 0.0, str(row.get('s_dtreturn_pct')))
            check('D % are n/a on an on–off row, never 0',
                  row.get('d_db_pct') is None and row.get('d_dtreturn_pct') is None,
                  str([row.get('d_db_pct'), row.get('d_dtreturn_pct')]))
            check('eq/eval stay n/a',
                  all(row.get(k) is None for k in
                      ('eq_db_pct', 'eval_db_pct', 'eq_dtreturn_pct', 'eval_delta_cop')),
                  str([row.get(k) for k in
                       ('eq_db_pct', 'eval_db_pct', 'eq_dtreturn_pct', 'eval_delta_cop')]))
            check('D hover names the kind',
                  'defrost' in (row.get('d_reasons') or {}).get('db', ''),
                  str(row.get('d_reasons')))
    finally:
        _restore_flow_stubs(*saved)


def test_gw_continuous_leaves_d_and_s_na():
    with WindowsHarness(sheet=_pd_sheet()) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=39.0, q=5.5, p=2.2, cop=2.5)
        h.add_period(1, 'heating', 0.0, 12000.0)
        h.add_period(1, 'equilibrium', 0.0, 3600.0)
        h.add_period(1, 'evaluation', 3600.0, 7800.0)
        row = h.one_row(1)
        check('continuous: D and S % are n/a, never 0',
              all(row.get(k) is None for k in
                  ('d_db_pct', 'd_dtreturn_pct',
                   's_db_pct', 's_wb_pct', 's_flow_pct', 's_dtreturn_pct')),
              str([row.get(k) for k in
                   ('d_db_pct', 'd_dtreturn_pct',
                    's_db_pct', 's_wb_pct', 's_flow_pct', 's_dtreturn_pct')]))
        check('H % are still scored', row.get('h_db_pct') == 0.0, str(row.get('h_db_pct')))


def test_cycle_extract_period_layers_stay_off_and_have_no_tsup_band():
    """Layers are still opt-in and there is still one clocks-only figure.

    This no longer asserts that **show** waits for Propose: show is
    offered for a row that has saved clocks too. What must not change is that
    the master toggle starts off (100 cycles do not cover the plot), that Cycle
    Extract carries no Tsup band, and that there is exactly one Plotly figure.
    """
    html = open(os.path.join(REPO, 'flask_app', 'templates', 'cycle_extract.html'),
                encoding='utf-8').read()
    check('Period layers master toggle exists', 'id="periodLayersToggle"' in html)
    check('Period layers start unchecked',
          'id="periodLayersToggle" checked' not in html
          and '_clockLayersOn = false' in html)
    check('the layers still draw only for ticked show rows',
          'if (!_clockShow[e.rowid]) return;' in html)
    check('there is no Tsup ±0.5 K layer checkbox',
          'Tsup ±0.5' not in html and 'data-layer="tsup"' not in html)
    check('there is no Tsup band rect on y2',
          "yref: 'y2'" not in html and 'yref: "y2"' not in html and "yref:'y2'" not in html)
    check('there is still one Plotly.newPlot', html.count('Plotly.newPlot') == 1)
    guide = open(os.path.join(REPO, 'docs', 'USER_GUIDE.md'), encoding='utf-8').read()
    check('USER_GUIDE keeps Period layers off by default',
          'Period layers are **off by default**' in guide)
    check('USER_GUIDE says show also works for saved clocks',
          '**saved** clocks' in guide and 'or a proposal' in guide)
    check('USER_GUIDE says the layers fall back to the saved bands',
          'otherwise the **saved** bands' in guide)
    check('USER_GUIDE does not put a Tsup band on Cycle Extract',
          'Tsup ±0.5 K** band on the temperature pane' not in guide)


def test_missing_db_setpoint_does_not_blank_h_flow_pct():
    """H5: a missing outdoor setpoint must leave Tsup / dTreturn / flow scorable."""
    sheet = _pd_sheet()
    sheet()  # paint outdoor series with the real E setpoint, then hide the lookup
    saved_db = webapp.unit_config.get_tdb_setpoint_for
    webapp.unit_config.get_tdb_setpoint_for = lambda *a, **k: None
    saved_flow = _fixed_flow_stubs()
    try:
        with WindowsHarness(sheet=sheet) as h:
            _defrost_entry(h)
            row = h.one_row(1)
            check('H DB % is n/a when the outdoor setpoint is missing',
                  row.get('h_db_pct') is None, str(row.get('h_db_pct')))
            check('the DB hover names the missing outdoor setpoint, not a shared abort',
                  'setpoint' in ((row.get('db_reasons') or {}).get('h') or '').lower(),
                  str(row.get('db_reasons')))
            check('H flow % is still scored',
                  row.get('h_flow_pct') is not None, str(row.get('h_flow_pct')))
            check('the flow hover is not the outdoor-setpoint warning',
                  'setpoint' not in ((row.get('flow_reasons') or {}).get('h') or '').lower(),
                  str(row.get('flow_reasons')))
            check('H dTreturn % is still scored',
                  row.get('h_dtreturn_pct') is not None, str(row.get('h_dtreturn_pct')))
    finally:
        webapp.unit_config.get_tdb_setpoint_for = saved_db
        _restore_flow_stubs(*saved_flow)


def test_s_flow_hover_names_variable_flow():
    """H6: variable-flow n/a is not the same sentence as 'the flow check does not apply'."""
    with WindowsHarness(sheet=_pd_sheet()) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=38.0, q=5.0, p=2.0, cop=2.5)
        h.add_period(1, 'off', 0.0, 1200.0)
        h.add_period(1, 'on', 1200.0, 12000.0)
        row = h.one_row(1)
        s_why = (row.get('s_reasons') or {}).get('flow') or ''
        h_why = (row.get('flow_reasons') or {}).get('h') or ''
        check('S flow % is n/a on a variable-flow test, never 0',
              row.get('s_flow_pct') is None, str(row.get('s_flow_pct')))
        check('the S flow hover names variable-flow',
              'variable-flow' in s_why, s_why)
        check('the S flow hover is not the generic skip',
              'does not apply' not in s_why, s_why)
        check('H flow hover also names variable-flow',
              'variable-flow' in h_why, h_why)


def test_guideline_clock_spans_follow_saved_periods_not_the_test_letter():
    defrost = webapp._guideline_clock_spans_from_periods([
        {'period_type': 'defrost', 'start_time': 0.0, 'end_time': 1200.0},
        {'period_type': 'heating', 'start_time': 1200.0, 'end_time': 10000.0},
        {'period_type': 'defrost', 'start_time': 10000.0, 'end_time': 12000.0},
        {'period_type': 'equilibrium', 'start_time': 1200.0, 'end_time': 4800.0},
        {'period_type': 'evaluation', 'start_time': 4800.0, 'end_time': 9000.0},
    ])
    layers = [s['layer'] for s in defrost]
    check('two D spans stay two ds pieces, not two percentages',
          layers.count('ds') == 2, str(layers))
    check('H, eq and eval are present',
          layers.count('h') == 1 and layers.count('eq') == 1 and layers.count('eval') == 1,
          str(layers))
    check('kind does not come from a test letter',
          webapp._guideline_layer_of_period('E') is None
          and webapp._guideline_layer_of_period('other') is None)

    off = webapp._guideline_clock_spans_from_periods([
        {'period_type': 'off', 'start_time': 0.0, 'end_time': 1200.0},
        {'period_type': 'on', 'start_time': 1200.0, 'end_time': 12000.0},
    ])
    check('on–off maps S to ds and H to h, with no eq/eval',
          [s['layer'] for s in off] == ['ds', 'h'], str(off))
    check('no periods means no spans',
          webapp._guideline_clock_spans_from_periods([]) == [])


def test_plot_deviation_shades_saved_clocks_on_all_figures():
    with DeviationsHarness(sheet=_pd_sheet()) as h:
        h.add_entry(1)
        check('without clocks there is nothing to shade',
              webapp._saved_guideline_clock_spans(h.parent_row(1)) == [])
        _pd_defrost_clocks(h)
        parent = h.parent_row(1)
        spans = webapp._saved_guideline_clock_spans(parent)
        check('saved defrost clocks become D/H/eq/eval spans',
              {s['layer'] for s in spans} == {'ds', 'h', 'eq', 'eval'},
              str([s['layer'] for s in spans]))
        with contextlib.redirect_stdout(io.StringIO()):
            resp = h.client.post('/plot_deviation', json={
                'file_name': PD_FILE, 'data_set': 1,
                'start_time': 0.0, 'end_time': PD_END,
            })
            body = resp.get_json() or {}
        check('the Tsup plot is still produced',
              body.get('success') is True and bool(body.get('image_tsup')),
              str(body.get('message') or body.get('success')))
        check('DB/WB and dTreturn plots are still produced',
              bool(body.get('image_dbwb')) and bool(body.get('image_dtreturn')),
              str([bool(body.get('image_dbwb')), bool(body.get('image_dtreturn'))]))
        check('the plot reports that saved clocks were shaded',
              body.get('clocks_shaded') is True
              and set(body.get('clock_layers') or []) == {'ds', 'h', 'eq', 'eval'},
              str([body.get('clocks_shaded'), body.get('clock_layers')]))

        with contextlib.redirect_stdout(io.StringIO()):
            missed = h.client.post('/plot_deviation', json={
                'file_name': PD_FILE, 'data_set': 1,
                'start_time': 99.0, 'end_time': 100.0,
            }).get_json() or {}
            by_id = h.client.post('/plot_deviation', json={
                'file_name': PD_FILE, 'data_set': 1,
                'start_time': 99.0, 'end_time': 100.0,
                'rowid': 1,
            }).get_json() or {}
        check('wrong start/end without rowid cannot find the saved clocks',
              missed.get('clocks_shaded') is False, str(missed.get('clocks_shaded')))
        check('the same wrong start/end still shades when rowid is sent',
              by_id.get('success') is True and by_id.get('clocks_shaded') is True,
              str([by_id.get('success'), by_id.get('clocks_shaded')]))


def test_plot_flow_uses_profile_id_from_the_entry():
    """Flow set lives on the unit profile, not on HP digit 2 and not on mass flow in the sheet."""
    saved_var = webapp._entry_is_variable_flow
    saved_set = webapp.get_flow_set_for_hp
    webapp._entry_is_variable_flow = lambda entry, flow_config=None: False
    webapp.get_flow_set_for_hp = (
        lambda hp_id, conn=None, file_name=None, profile_id=None: (
            ('volume', 1.0) if profile_id == 'TEST_PROFILE' else (None, None)))
    try:
        with DeviationsHarness(sheet=_pd_sheet()) as h:
            h.add_entry(1)
            conn = h._connect()
            try:
                conn.execute("UPDATE results SET HP_ID=? WHERE rowid=?", ('2', 1))
                conn.commit()
            finally:
                conn.close()
            with contextlib.redirect_stdout(io.StringIO()):
                no_profile = h.client.post('/plot_flow_deviation', json={
                    'file_name': PD_FILE, 'data_set': 1,
                    'start_time': 0.0, 'end_time': PD_END,
                    'hp_id': '2',
                    'rowid': 1,
                }).get_json() or {}
            conn = h._connect()
            try:
                conn.execute("UPDATE results SET profile_id=? WHERE rowid=?", ('TEST_PROFILE', 1))
                conn.commit()
            finally:
                conn.close()
            with contextlib.redirect_stdout(io.StringIO()):
                with_profile = h.client.post('/plot_flow_deviation', json={
                    'file_name': PD_FILE, 'data_set': 1,
                    'start_time': 0.0, 'end_time': PD_END,
                    'hp_id': '2',
                    'rowid': 1,
                }).get_json() or {}
            check('HP digit 2 alone is not enough without the profile',
                  no_profile.get('success') is False
                  and 'mass-flow column' in (no_profile.get('message') or ''),
                  str(no_profile.get('message')))
            check('the entry profile supplies the volume-flow set',
                  with_profile.get('success') is True and bool(with_profile.get('image')),
                  str(with_profile.get('message') or with_profile.get('success')))
    finally:
        webapp._entry_is_variable_flow = saved_var
        webapp.get_flow_set_for_hp = saved_set


def test_gw_page_has_no_plots():
    gw_html = open(os.path.join(REPO, 'flask_app', 'templates', 'guideline_windows.html'),
                   encoding='utf-8').read()
    check('Guideline Windows does not embed Plotly',
          'plotly' not in gw_html.lower() and 'Plotly.newPlot' not in gw_html)
    check('Guideline Windows table is not a plot gallery',
          'plot_deviation' not in gw_html and '<img' not in gw_html.lower())


def test_gw_html_json_csv_have_no_tsup_percent_fields():
    gw_html = open(os.path.join(REPO, 'flask_app', 'templates', 'guideline_windows.html'),
                   encoding='utf-8').read()
    check('Guideline Windows HTML has H DB %', 'H DB %' in gw_html)
    check('Guideline Windows HTML has H flow %', 'H flow %' in gw_html)
    check('Guideline Windows HTML has D DB % and D dTreturn %',
          'D DB %' in gw_html and 'D dTreturn %' in gw_html)
    check('Guideline Windows HTML has S DB/WB/flow/dTreturn %',
          all(label in gw_html for label in ('S DB %', 'S WB %', 'S flow %', 'S dTreturn %')))
    check('Guideline Windows HTML has no Tsup % column', 'Tsup %' not in gw_html)
    check('Guideline Windows JS does not render a tsup % field',
          'tsup_percentage' not in gw_html and 'h_tsup_pct' not in gw_html)

    with WindowsHarness(sheet=_pd_sheet()) as h:
        _defrost_entry(h)
        _, d = h.load([1])
        row = (d.get('rows') or [{}])[0]
        check('JSON has the interval dTreturn and ΔCOP fields',
              all(k in row for k in
                  ('h_dtreturn_pct', 'eq_dtreturn_pct', 'eval_dtreturn_pct', 'eval_delta_cop')),
              str(sorted(row)))
        check('JSON has H/eq/eval DB, WB and flow % fields',
              all(k in row for k in
                  ('h_db_pct', 'eq_db_pct', 'eval_db_pct',
                   'h_wb_pct', 'eq_wb_pct', 'eval_wb_pct',
                   'h_flow_pct', 'eq_flow_pct', 'eval_flow_pct')),
              str(sorted(row)))
        check('JSON has D/S individual % fields',
              all(k in row for k in
                  ('d_db_pct', 'd_dtreturn_pct',
                   's_db_pct', 's_wb_pct', 's_flow_pct', 's_dtreturn_pct')),
              str(sorted(row)))
        check('JSON meta has the D/S half-widths',
              d.get('d_db_k') == 5.0 and d.get('d_dtreturn_k') == 2.0
              and d.get('s_db_k') == 2.0 and d.get('s_wb_k') == 2.0
              and d.get('s_flow_pct_band') == 2.5 and d.get('s_dtreturn_k') == 1.0,
              str({k: d.get(k) for k in
                   ('d_db_k', 'd_dtreturn_k', 's_db_k', 's_wb_k',
                    's_flow_pct_band', 's_dtreturn_k')}))
        check('JSON has no Tsup % fields',
              not any(k in row for k in TSUP_PCT_KEYS),
              str([k for k in row if 'tsup' in k.lower()]))
        with contextlib.redirect_stdout(io.StringIO()):
            resp = h.client.post('/api/guideline_windows/export', json={'rowids': [1]})
        header = resp.data.decode('utf-8-sig').splitlines()[0]
        check('CSV has no Tsup % field',
              'Tsup_pct' not in header and 'Tsup %' not in header, header)
        check('CSV has the D/S % headers',
              all(name in header for name in
                  ('D_DB_pct', 'D_dTreturn_pct', 'S_DB_pct', 'S_WB_pct',
                   'S_flow_pct', 'S_dTreturn_pct')),
              header)

    dev_html = open(os.path.join(REPO, 'flask_app', 'templates', 'deviations.html'),
                    encoding='utf-8').read()
    check('Deviations template no longer has the extra interval table',
          'Deviations by guideline interval' not in dev_html
          and 'intervalDeviationsTable' not in dev_html)
    check('parent dTreturn columns are unchanged',
          'dev_dtreturn_outside_count' in dev_html and 'dev_dtreturn_percentage' in dev_html)
    check('parent Tsup % is unchanged', 'dev_tsup_percentage' in dev_html or 'Tsup %' in dev_html)


# ---------------------------------------------------------------------------
# H / D / S MEAN deviations on Guideline Windows.
# A mean is one arithmetic number per interval against its setpoint, not the
# share of samples the individual % counts, and it has its own half-widths
# (H mean DB ±0.3 K against H individual ±1 K and the parent mean 0.6 K).
# ---------------------------------------------------------------------------

H_MEAN_KEYS = ('h_mean_db_k', 'h_mean_wb_k', 'h_mean_tsup_k',
               'h_mean_dtreturn_k', 'h_mean_flow_pct')
D_MEAN_KEYS = ('d_mean_db_k', 'd_mean_wb_k')
S_MEAN_KEYS = ('s_mean_db_k', 's_mean_wb_k', 's_mean_flow_pct')


def _union_frame(sheet, spans):
    """The rows of the synthetic sheet inside any of the spans (the app's union)."""
    df = sheet()
    t = pd.to_numeric(df['time_elapsed'], errors='coerce')
    mask = False
    for start, end in spans:
        part = (t >= start) & (t <= end)
        mask = part if mask is False else (mask | part)
    return df.loc[mask]


def _union_mean_dev(sheet, spans, column, setpoint):
    """mean(column) − setpoint over the union — the Step A formula, computed here."""
    window = _union_frame(sheet, spans)
    return float(pd.to_numeric(window[column], errors='coerce').mean()) - float(setpoint)


def _pd_db_set():
    return webapp.unit_config.get_tdb_setpoint_for('E')


def _water_source_stub():
    """Make the entry look water-to-water, so the air checks stop applying."""
    saved = webapp._entry_unit_capabilities
    webapp._entry_unit_capabilities = lambda entry: {
        'source_medium': 'water', 'has_wetbulb': False,
        'source_circuit_measured': True, 'has_gas_input': False,
    }
    return saved


def test_gw_mean_on_saved_defrost():
    db_set = _pd_db_set()
    sheet = _pd_sheet(db_in_d=2.0, db_in_h=0.0, wb_in_d=2.0, wb_in_h=0.0,
                      dt_in_d=3.0, dt_in_h=0.0)
    d_db = _union_mean_dev(sheet, [(0.0, PD_D_END)], 'T_outdoor (DB)', db_set)
    d_wb = _union_mean_dev(sheet, [(0.0, PD_D_END)], 'T_outdoor (WB)', db_set - 1.0)
    with WindowsHarness(sheet=sheet) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('H mean DB / WB / Tsup / dTreturn are numbers on a saved defrost row',
              all(row.get(k) is not None for k in
                  ('h_mean_db_k', 'h_mean_wb_k', 'h_mean_tsup_k', 'h_mean_dtreturn_k')),
              str([row.get(k) for k in H_MEAN_KEYS]))
        check('H mean flow is n/a on the variable-flow profile, never 0',
              row.get('h_mean_flow_pct') is None
              and 'variable-flow' in (row.get('h_mean_reasons') or {}).get('flow', ''),
              str((row.get('h_mean_reasons') or {}).get('flow')))
        check('an offset only in D does not move H mean DB',
              row.get('h_mean_db_k') == 0.0, str(row.get('h_mean_db_k')))
        check('H mean Tsup is 0 K — the 5 K excursion sits in D',
              row.get('h_mean_tsup_k') == 0.0, str(row.get('h_mean_tsup_k')))
        check('H mean dTreturn is 0 K — the 3 K offset sits in D',
              row.get('h_mean_dtreturn_k') == 0.0, str(row.get('h_mean_dtreturn_k')))
        check('D mean DB is the mean of the D window minus the setpoint',
              row.get('d_mean_db_k') is not None
              and abs(row['d_mean_db_k'] - d_db) < 1e-9,
              f"{row.get('d_mean_db_k')} vs {d_db}")
        check('D mean WB is the mean of the D window minus the setpoint',
              row.get('d_mean_wb_k') is not None
              and abs(row['d_mean_wb_k'] - d_wb) < 1e-9,
              f"{row.get('d_mean_wb_k')} vs {d_wb}")
        check('S means are n/a on a defrost row, never 0',
              all(row.get(k) is None for k in S_MEAN_KEYS),
              str([row.get(k) for k in S_MEAN_KEYS]))
        check('the S mean hover names the kind',
              'on–off' in (row.get('s_mean_reasons') or {}).get('db', ''),
              str(row.get('s_mean_reasons')))
        check('D has no mean Tsup / flow / inlet column at all',
              not any(k in row for k in
                      ('d_mean_tsup_k', 'd_mean_flow_pct', 'd_mean_dtreturn_k')),
              str([k for k in row if k.startswith('d_mean')]))


def test_gw_h_mean_db_band_is_zero_point_three():
    cfg = webapp.load_interval_deviations_config()
    h_spec = (cfg.get('intervals') or {}).get('H') or {}
    check('Interval H mean DB is ±0.3 K', float(h_spec.get('mean_db_k')) == 0.3,
          str(h_spec.get('mean_db_k')))
    check('Interval H mean WB / Tsup / dTreturn / flow are ±0.4 K / ±0.5 K / ±0.3 K / ±1 %',
          [float(h_spec.get(k)) for k in
           ('mean_wb_k', 'mean_tsup_k', 'mean_dtreturn_k', 'mean_flow_pct')]
          == [0.4, 0.5, 0.3, 1.0],
          str({k: h_spec.get(k) for k in
               ('mean_wb_k', 'mean_tsup_k', 'mean_dtreturn_k', 'mean_flow_pct')}))
    parent = webapp.load_permissible_deviations()
    check('the parent mean DB limit is untouched at 0.6 K',
          float(parent.get('mean_db_k')) == 0.6, str(parent.get('mean_db_k')))
    check('H individual DB is still ±1 K', float(parent['DB']['value']) == 1.0,
          str(parent['DB']['value']))

    with WindowsHarness(sheet=_pd_sheet(db_in_d=0.0, db_in_h=0.5)) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('a 0.5 K mean offset in H shows as +0.50 K',
              abs(row.get('h_mean_db_k') - 0.5) < 1e-9, str(row.get('h_mean_db_k')))
        check('that mean is outside the H mean band ±0.3 K',
              abs(row['h_mean_db_k']) > float(h_spec['mean_db_k']),
              f"{row['h_mean_db_k']} vs ±{h_spec['mean_db_k']}")
        check('it would have been inside the parent mean 0.6 K and the individual ±1 K',
              abs(row['h_mean_db_k']) < float(parent['mean_db_k'])
              and row.get('h_db_pct') == 0.0,
              str([row['h_mean_db_k'], row.get('h_db_pct')]))


def test_gw_d_mean_db_band_is_one_point_five():
    cfg = webapp.load_interval_deviations_config()
    d_spec = (cfg.get('intervals') or {}).get('D') or {}
    h_spec = (cfg.get('intervals') or {}).get('H') or {}
    check('D mean DB is ±1.5 K and D mean WB ±1.0 K',
          float(d_spec.get('mean_db_k')) == 1.5 and float(d_spec.get('mean_wb_k')) == 1.0,
          str([d_spec.get('mean_db_k'), d_spec.get('mean_wb_k')]))
    sheet = _pd_sheet(db_in_d=1.0, db_in_h=0.0)
    expect = _union_mean_dev(sheet, [(0.0, PD_D_END)], 'T_outdoor (DB)', _pd_db_set())
    with WindowsHarness(sheet=sheet) as h:
        _defrost_entry(h)
        row = h.one_row(1)
        check('the D mean is the union mean, about 1 K',
              abs(row.get('d_mean_db_k') - expect) < 1e-9,
              f"{row.get('d_mean_db_k')} vs {expect}")
        check('1 K in D is inside the D mean band ±1.5 K',
              abs(row['d_mean_db_k']) < float(d_spec['mean_db_k']),
              f"{row['d_mean_db_k']} vs ±{d_spec['mean_db_k']}")
        check('the same 1 K would be outside the H mean band ±0.3 K — D mean is not H mean',
              abs(row['d_mean_db_k']) > float(h_spec['mean_db_k']),
              f"{row['d_mean_db_k']} vs ±{h_spec['mean_db_k']}")
        check('H mean DB stays 0 (the offset is not in H)',
              row.get('h_mean_db_k') == 0.0, str(row.get('h_mean_db_k')))


def test_gw_two_d_spans_are_one_d_mean():
    d2_start = 10000.0
    sheet = _pd_sheet(db_in_d=0.0, db_in_h=0.0, d2_start=d2_start, db_in_d2=1.2)
    spans = [(0.0, PD_D_END), (d2_start, PD_END)]
    expect = _union_mean_dev(sheet, spans, 'T_outdoor (DB)', _pd_db_set())
    with WindowsHarness(sheet=sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0)
        h.add_period(1, 'defrost', 0.0, PD_D_END)
        h.add_period(1, 'heating', PD_D_END, d2_start)
        h.add_period(1, 'defrost', d2_start, PD_END)
        h.add_period(1, 'equilibrium', PD_D_END, PD_EVAL[0])
        h.add_period(1, 'evaluation', *PD_EVAL)
        row = h.one_row(1)
        check('two D spans still yield one D mean column',
              'd2_mean_db_k' not in row and row.get('d_mean_db_k') is not None,
              str([k for k in row if k.startswith('d') and 'mean' in k]))
        check('that one D mean is the union of both spans',
              abs(row['d_mean_db_k'] - expect) < 1e-9,
              f"{row['d_mean_db_k']} vs union {expect}")
        check('D1 alone would have been 0, so the union really is both spans',
              expect > 0.0, str(expect))


def test_gw_on_off_scores_s_mean_and_leaves_d_mean_na():
    db_set = _pd_db_set()
    sheet = _pd_sheet(db_in_d=0.4, db_in_h=0.0, wb_in_d=0.4, wb_in_h=0.0)
    s_db = _union_mean_dev(sheet, [(0.0, 1200.0)], 'T_outdoor (DB)', db_set)
    saved = _fixed_flow_stubs()
    try:
        with WindowsHarness(sheet=sheet) as h:
            h.add_entry(1, 0.0, 12000.0, tsup=38.0, q=5.0, p=2.0, cop=2.5)
            h.add_period(1, 'off', 0.0, 1200.0)
            h.add_period(1, 'on', 1200.0, 12000.0)
            row = h.one_row(1)
            check('S mean DB / WB / flow are scored on an on–off row',
                  all(row.get(k) is not None for k in S_MEAN_KEYS),
                  str([row.get(k) for k in S_MEAN_KEYS]))
            check('S mean DB is the mean of the off window minus the setpoint',
                  abs(row['s_mean_db_k'] - s_db) < 1e-9, f"{row['s_mean_db_k']} vs {s_db}")
            check('S mean flow is 0 % of set when the flow sits on its set value',
                  abs(row['s_mean_flow_pct']) < 1e-9, str(row['s_mean_flow_pct']))
            check('D means are n/a on an on–off row, never 0',
                  all(row.get(k) is None for k in D_MEAN_KEYS),
                  str([row.get(k) for k in D_MEAN_KEYS]))
            check('the D mean hover names the kind',
                  'defrost' in (row.get('d_mean_reasons') or {}).get('db', ''),
                  str(row.get('d_mean_reasons')))
            check('H means are still filled from the saved on window',
                  row.get('h_mean_db_k') == 0.0 and row.get('h_mean_tsup_k') is not None,
                  str([row.get('h_mean_db_k'), row.get('h_mean_tsup_k')]))
            check('H mean flow is scored once the unit is fixed-flow',
                  row.get('h_mean_flow_pct') is not None
                  and abs(row['h_mean_flow_pct']) < 1e-9,
                  str(row.get('h_mean_flow_pct')))
    finally:
        _restore_flow_stubs(*saved)


def test_gw_continuous_leaves_d_and_s_means_na():
    with WindowsHarness(sheet=_pd_sheet(bad_tsup_until=0.0)) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=39.0, q=5.5, p=2.2, cop=2.5)
        h.add_period(1, 'heating', 0.0, 12000.0)
        h.add_period(1, 'equilibrium', 0.0, 3600.0)
        h.add_period(1, 'evaluation', 3600.0, 7800.0)
        row = h.one_row(1)
        check('continuous: D and S means are n/a, never 0',
              all(row.get(k) is None for k in D_MEAN_KEYS + S_MEAN_KEYS),
              str([row.get(k) for k in D_MEAN_KEYS + S_MEAN_KEYS]))
        check('continuous: H means are still filled',
              row.get('h_mean_db_k') == 0.0 and row.get('h_mean_tsup_k') == 0.0,
              str([row.get('h_mean_db_k'), row.get('h_mean_tsup_k')]))
        check('the D/S mean hovers name the kind',
              'defrost' in (row.get('d_mean_reasons') or {}).get('db', '')
              and 'on–off' in (row.get('s_mean_reasons') or {}).get('db', ''),
              str([row.get('d_mean_reasons'), row.get('s_mean_reasons')]))


def test_gw_means_without_a_window_or_a_source_are_na_with_a_reason():
    with WindowsHarness(sheet=_pd_sheet()) as h:
        # A parent row with no saved clocks at all.
        h.add_entry(1, 0.0, 12000.0, tsup=40.0)
        row = h.one_row(1)
        check('no saved clocks: every mean is n/a, never 0',
              all(row.get(k) is None for k in H_MEAN_KEYS + D_MEAN_KEYS + S_MEAN_KEYS),
              str([row.get(k) for k in H_MEAN_KEYS]))
        check('and the hover says no window of that kind is saved',
              (row.get('h_mean_reasons') or {}).get('db') == 'no window of this kind is saved',
              str(row.get('h_mean_reasons')))

    saved_caps = _water_source_stub()
    try:
        with WindowsHarness(sheet=_pd_sheet()) as h:
            _defrost_entry(h)
            row = h.one_row(1)
            check('water-to-water: H and D mean DB / WB are n/a, never 0',
                  all(row.get(k) is None for k in
                      ('h_mean_db_k', 'h_mean_wb_k', 'd_mean_db_k', 'd_mean_wb_k')),
                  str([row.get(k) for k in
                       ('h_mean_db_k', 'h_mean_wb_k', 'd_mean_db_k', 'd_mean_wb_k')]))
            check('the hover names the check that does not apply',
                  'does not apply' in (row.get('h_mean_reasons') or {}).get('db', ''),
                  str(row.get('h_mean_reasons')))
            check('H mean Tsup and dTreturn still apply on a water source',
                  row.get('h_mean_tsup_k') is not None
                  and row.get('h_mean_dtreturn_k') is not None,
                  str([row.get('h_mean_tsup_k'), row.get('h_mean_dtreturn_k')]))
    finally:
        webapp._entry_unit_capabilities = saved_caps


def test_gw_mean_columns_in_html_json_and_csv():
    gw_html = open(os.path.join(REPO, 'flask_app', 'templates', 'guideline_windows.html'),
                   encoding='utf-8').read()
    check('Guideline Windows HTML has the H mean headers',
          all(label in gw_html for label in
              ('H mean DB K', 'H mean WB K', 'H mean Tsup K', 'H mean dTreturn K',
               'H mean flow % of set')), 'H mean headers')
    check('Guideline Windows HTML has the D and S mean headers',
          all(label in gw_html for label in
              ('D mean DB K', 'D mean WB K', 'S mean DB K', 'S mean WB K',
               'S mean flow % of set')), 'D/S mean headers')
    check('the mean headings still carry no Tsup % column', 'Tsup %' not in gw_html)

    with WindowsHarness(sheet=_pd_sheet()) as h:
        _defrost_entry(h)
        _, d = h.load([1])
        row = (d.get('rows') or [{}])[0]
        check('JSON row has the ten mean fields',
              all(k in row for k in H_MEAN_KEYS + D_MEAN_KEYS + S_MEAN_KEYS),
              str(sorted(k for k in row if 'mean' in k)))
        check('JSON meta exposes the mean half-widths',
              [d.get(k) for k in ('mean_h_db_k', 'mean_h_wb_k', 'mean_h_tsup_k',
                                  'mean_h_dtreturn_k', 'mean_h_flow_pct',
                                  'mean_d_db_k', 'mean_d_wb_k',
                                  'mean_s_db_k', 'mean_s_wb_k', 'mean_s_flow_pct')]
              == [0.3, 0.4, 0.5, 0.3, 1.0, 1.5, 1.0, 0.6, 0.6, 2.5],
              str({k: v for k, v in d.items() if k.startswith('mean_')}))
        check('there are no eq / eval mean columns in this slice',
              not any(k.startswith('eq_mean') or k.startswith('eval_mean') for k in row),
              str([k for k in row if 'mean' in k]))
        check('JSON still has no Tsup % field', not any(k in row for k in TSUP_PCT_KEYS),
              str([k for k in row if 'tsup' in k.lower()]))
        with contextlib.redirect_stdout(io.StringIO()):
            resp = h.client.post('/api/guideline_windows/export', json={'rowids': [1]})
        header = resp.data.decode('utf-8-sig').splitlines()[0]
        check('CSV has the mean headers with their unit',
              all(name in header for name in
                  ('H_mean_DB_K', 'H_mean_WB_K', 'H_mean_Tsup_K', 'H_mean_dTreturn_K',
                   'H_mean_flow_pct_of_set', 'D_mean_DB_K', 'D_mean_WB_K',
                   'S_mean_DB_K', 'S_mean_WB_K', 'S_mean_flow_pct_of_set')),
              header)
        check('CSV still has no Tsup % field',
              'Tsup_pct' not in header and 'Tsup %' not in header, header)
        check('CSV keeps every individual % column',
              all(name in header for name in
                  ('H_DB_pct', 'D_DB_pct', 'S_flow_pct', 'eval_dCOP_pct')), header)


def test_gw_mean_pds_do_not_reach_the_parent_or_the_insert():
    """Step A is a Guideline Windows read; parent bands and Apply stay as they were."""
    with DeviationsHarness(sheet=_pd_sheet(db_in_d=0.0, db_in_h=0.5)) as h:
        h.add_entry(1)
        _pd_defrost_clocks(h)
        _resp, body = h.calculate([1])
        parent = h.parent_row(1)
        check('parent deviations still come from the full cycle',
              body.get('success') is True and parent.get('dev_total_points'),
              str(body)[:160])
        check('the parent table has no interval mean columns',
              not any(k.startswith('dev_h_mean') or k.startswith('dev_d_mean')
                      for k in parent),
              str([k for k in parent if 'mean' in k]))
        block = h.block(1)
        check('the interval cache still scores no D or S',
              not any(iv['key'] in ('D', 'S') and iv['status'] == 'ok'
                      for iv in (block or {}).get('intervals', [])),
              str([iv['key'] for iv in (block or {}).get('intervals', [])]))
    dev_html = open(os.path.join(REPO, 'flask_app', 'templates', 'deviations.html'),
                    encoding='utf-8').read()
    check('Deviations still has no per-interval table',
          'Deviations by guideline interval' not in dev_html
          and 'intervalDeviationsTable' not in dev_html)


# ---------------------------------------------------------------------------
# Stepped individual bands on the Deviations Plot. Group 1 keeps
# the parent / full-cycle widths the Deviations table scores; Group 2 repeats the
# same traces with the draft per-interval individual widths, stepped at the saved
# clock edges. Tsup has no Group 2 figure - draft individual Tsup is n/a.
# ---------------------------------------------------------------------------

def _step_periods_defrost(with_eq_eval=True):
    """D 0...1200 s, H 1200...12000 s, plus eq and eval inside H."""
    out = [
        {'period_type': 'defrost', 'start_time': 0.0, 'end_time': PD_D_END},
        {'period_type': 'heating', 'start_time': PD_D_END, 'end_time': PD_END},
    ]
    if with_eq_eval:
        out += [
            {'period_type': 'equilibrium', 'start_time': PD_D_END, 'end_time': PD_EVAL[0]},
            {'period_type': 'evaluation', 'start_time': PD_EVAL[0], 'end_time': PD_EVAL[1]},
        ]
    return out


def _steps(periods, quantity, start=0.0, end=PD_END):
    return [(st['interval'], st['start'], st['end'], st['half'])
            for st in webapp._interval_individual_band_steps(periods, quantity, start, end)]


def test_step_b_helper_on_a_saved_defrost():
    periods = _step_periods_defrost()
    db = _steps(periods, 'db')
    check('DB steps on a defrost row are D then H',
          [(iv, half) for iv, _s, _e, half in db] == [('D', 5.0), ('H', 1.0)], str(db))
    check('the D DB step is the saved defrost span, H the rest',
          db and db[0][1] == 0.0 and db[0][2] == PD_D_END
          and db[1][1] == PD_D_END and db[1][2] == PD_END, str(db))
    check('equilibrium and evaluation add no DB step of their own',
          len(db) == len(_steps(_step_periods_defrost(with_eq_eval=False), 'db')), str(db))

    dt = _steps(periods, 'dtreturn')
    check('dTreturn steps are D +/-2 K then H +/-0.5 K, not the parent -2...+2 on H',
          [(iv, half) for iv, _s, _e, half in dt] == [('D', 2.0), ('H', 0.5)], str(dt))

    wb = _steps(periods, 'wb')
    check('WB has no D step - the draft D wet-bulb cell is n/a',
          [iv for iv, _s, _e, _h in wb] == ['H']
          and wb[0][1] == PD_D_END and wb[0][3] == 1.0, str(wb))

    flow = _steps(periods, 'flow')
    check('flow has no D step - the draft D flow cell is n/a',
          [(iv, half) for iv, _s, _e, half in flow] == [('H', 2.5)], str(flow))

    check('Tsup is not a stepped quantity at all',
          webapp._interval_individual_band_steps(periods, 'tsup', 0.0, PD_END) == []
          and 'tsup' not in webapp.INTERVAL_STEP_QUANTITIES)


def test_step_b_helper_two_d_spans_keep_h_in_the_gap():
    periods = [
        {'period_type': 'defrost', 'start_time': 0.0, 'end_time': PD_D_END},
        {'period_type': 'heating', 'start_time': PD_D_END, 'end_time': 10000.0},
        {'period_type': 'defrost', 'start_time': 10000.0, 'end_time': PD_END},
    ]
    db = _steps(periods, 'db')
    check('two stored D spans give D, H, D - not one rectangle over the lot',
          [(iv, half) for iv, _s, _e, half in db] == [('D', 5.0), ('H', 1.0), ('D', 5.0)],
          str(db))
    check('the middle stretch is the H width, not the D width',
          db[1][1] == PD_D_END and db[1][2] == 10000.0 and db[1][3] == 1.0, str(db))

    # The analyst may store H across the whole window; D still wins where they overlap.
    overlapping = [
        {'period_type': 'defrost', 'start_time': 0.0, 'end_time': PD_D_END},
        {'period_type': 'heating', 'start_time': 0.0, 'end_time': PD_END},
        {'period_type': 'defrost', 'start_time': 10000.0, 'end_time': PD_END},
    ]
    db2 = _steps(overlapping, 'db')
    check('a D span that overlaps H cuts H, it does not lose to it',
          [(iv, s, e, half) for iv, s, e, half in db2]
          == [('D', 0.0, PD_D_END, 5.0), ('H', PD_D_END, 10000.0, 1.0),
              ('D', 10000.0, PD_END, 5.0)], str(db2))


def test_step_b_helper_on_off_and_continuous():
    on_off = [
        {'period_type': 'off', 'start_time': 0.0, 'end_time': PD_D_END},
        {'period_type': 'on', 'start_time': PD_D_END, 'end_time': PD_END},
    ]
    check('on-off DB steps are S +/-2 K then H +/-1 K',
          [(iv, half) for iv, _s, _e, half in _steps(on_off, 'db')]
          == [('S', 2.0), ('H', 1.0)], str(_steps(on_off, 'db')))
    check('on-off dTreturn steps are S +/-1 K then H +/-0.5 K',
          [(iv, half) for iv, _s, _e, half in _steps(on_off, 'dtreturn')]
          == [('S', 1.0), ('H', 0.5)], str(_steps(on_off, 'dtreturn')))
    check('on-off flow is +/-2.5 % on both S and H',
          [(iv, half) for iv, _s, _e, half in _steps(on_off, 'flow')]
          == [('S', 2.5), ('H', 2.5)], str(_steps(on_off, 'flow')))
    check('on-off WB is +/-2 K on S and +/-1 K on H',
          [(iv, half) for iv, _s, _e, half in _steps(on_off, 'wb')]
          == [('S', 2.0), ('H', 1.0)], str(_steps(on_off, 'wb')))

    continuous = [{'period_type': 'heating', 'start_time': 0.0, 'end_time': PD_END}]
    for quantity, half in (('db', 1.0), ('wb', 1.0), ('dtreturn', 0.5), ('flow', 2.5)):
        steps = _steps(continuous, quantity)
        check(f'continuous {quantity} is one H step at +/-{half:g}',
              [(iv, h) for iv, _s, _e, h in steps] == [('H', half)], str(steps))


def test_step_b_helper_without_clocks_or_with_unknown_types():
    check('no saved periods means no steps at all',
          webapp._interval_individual_band_steps([], 'db', 0.0, PD_END) == []
          and webapp._interval_individual_band_steps(None, 'dtreturn', 0.0, PD_END) == [])
    unknown = [
        {'period_type': 'other', 'start_time': 0.0, 'end_time': PD_END},
        {'period_type': 'unknown', 'start_time': 0.0, 'end_time': PD_END},
        {'period_type': 'equilibrium', 'start_time': 0.0, 'end_time': PD_END},
        {'period_type': 'evaluation', 'start_time': 0.0, 'end_time': PD_END},
    ]
    check('unknown types and eq/eval alone never invent a step',
          webapp._interval_individual_band_steps(unknown, 'db', 0.0, PD_END) == [],
          str(webapp._interval_individual_band_steps(unknown, 'db', 0.0, PD_END)))
    gap = [{'period_type': 'defrost', 'start_time': 0.0, 'end_time': PD_D_END}]
    db = _steps(gap, 'db')
    check('a time with no saved D/S/H period gets no step, not the parent band',
          [(iv, s, e) for iv, s, e, _h in db] == [('D', 0.0, PD_D_END)], str(db))


def test_plot_deviation_is_one_figure_per_quantity():
    """One figure per quantity: the trace is drawn once. Saved clocks change the fill, not the count."""
    with DeviationsHarness(sheet=_pd_sheet()) as h:
        h.add_entry(1)
        with contextlib.redirect_stdout(io.StringIO()):
            bare = h.client.post('/plot_deviation', json={
                'file_name': PD_FILE, 'data_set': 1,
                'start_time': 0.0, 'end_time': PD_END,
            }).get_json() or {}
        check('without clocks every figure still draws',
              bare.get('success') is True and bool(bare.get('image_dbwb'))
              and bool(bare.get('image_tsup')) and bool(bare.get('image_dtreturn')),
              str(bare.get('message') or bare.get('success')))
        check('without clocks the fill is the parent / full-cycle band',
              bare.get('band_mode') == 'parent' and bare.get('clocks_shaded') is False,
              str([bare.get('band_mode'), bare.get('clocks_shaded')]))
        check('no second copy of any trace is sent',
              not bare.get('image_dbwb_interval') and not bare.get('image_dtreturn_interval'),
              str(sorted(bare.keys())))

        _pd_defrost_clocks(h)
        with contextlib.redirect_stdout(io.StringIO()):
            body = h.client.post('/plot_deviation', json={
                'file_name': PD_FILE, 'data_set': 1,
                'start_time': 0.0, 'end_time': PD_END,
            }).get_json() or {}
        check('with clocks it is still one DB/WB, one Tsup and one dTreturn figure',
              bool(body.get('image_dbwb')) and bool(body.get('image_tsup'))
              and bool(body.get('image_dtreturn')), str(body.get('message')))
        check('with clocks the fill is the stepped interval band',
              body.get('band_mode') == 'interval' and body.get('clocks_shaded') is True
              and set(body.get('clock_layers') or []) == {'ds', 'h', 'eq', 'eval'},
              str([body.get('band_mode'), body.get('clock_layers')]))
        check('the Group 2 copies are gone from the reply',
              not body.get('image_dbwb_interval') and not body.get('image_dtreturn_interval'),
              str(sorted(body.keys())))
        check('there is still exactly one Tsup image and no stepped Tsup band',
              [k for k in body if 'tsup' in k] == ['image_tsup'],
              str([k for k in body if 'tsup' in k]))

        with contextlib.redirect_stdout(io.StringIO()):
            by_id = h.client.post('/plot_deviation', json={
                'file_name': PD_FILE, 'data_set': 1,
                'start_time': 99.0, 'end_time': 100.0, 'rowid': 1,
            }).get_json() or {}
        check('a wrong start/end with rowid still finds the clocks and steps the fill',
              by_id.get('success') is True and by_id.get('clocks_shaded') is True
              and by_id.get('band_mode') == 'interval',
              str([by_id.get('clocks_shaded'), by_id.get('band_mode')]))


def test_plot_deviation_carries_flow_in_the_same_modal():
    saved_var, saved_set = _fixed_flow_stubs()
    try:
        with DeviationsHarness(sheet=_pd_sheet()) as h:
            h.add_entry(1)
            _pd_defrost_clocks(h)
            with contextlib.redirect_stdout(io.StringIO()):
                fixed = h.client.post('/plot_deviation', json={
                    'file_name': PD_FILE, 'data_set': 1,
                    'start_time': 0.0, 'end_time': PD_END, 'rowid': 1,
                }).get_json() or {}
            check('a fixed-flow entry gets its flow figure from the Plot button',
                  fixed.get('success') is True and bool(fixed.get('image_flow')),
                  str(fixed.get('flow_message') or fixed.get('message')))
            check('the flow figure is one image, not a pair',
                  not fixed.get('image_interval'), str(sorted(fixed.keys())))
            steps = webapp._interval_individual_band_steps(
                webapp._saved_guideline_periods(h.parent_row(1)), 'flow', 0.0, PD_END)
            check('the stepped flow fill carries H only, never a D fill',
                  [st['interval'] for st in steps] == ['H'], str(steps))

            # The same figure through the thin API: one image, never two.
            with contextlib.redirect_stdout(io.StringIO()):
                api = h.client.post('/plot_flow_deviation', json={
                    'file_name': PD_FILE, 'data_set': 1,
                    'start_time': 0.0, 'end_time': PD_END, 'rowid': 1,
                }).get_json() or {}
            check('the flow API returns one unified image',
                  api.get('success') is True and bool(api.get('image'))
                  and 'image_interval' not in api, str(sorted(api.keys())))

        webapp._entry_is_variable_flow = lambda entry, flow_config=None: True
        with DeviationsHarness(sheet=_pd_sheet()) as h:
            h.add_entry(1)
            with contextlib.redirect_stdout(io.StringIO()):
                variable = h.client.post('/plot_deviation', json={
                    'file_name': PD_FILE, 'data_set': 1,
                    'start_time': 0.0, 'end_time': PD_END, 'rowid': 1,
                }).get_json() or {}
            check('a variable-flow entry has no flow figure but keeps the rest',
                  variable.get('success') is True and variable.get('image_flow') is None
                  and bool(variable.get('image_dbwb')) and bool(variable.get('image_tsup'))
                  and bool(variable.get('image_dtreturn')),
                  str(variable.get('message') or variable.get('image_flow')))
            check('and the reason names the variable-flow skip',
                  'variable-flow' in (variable.get('flow_message') or ''),
                  str(variable.get('flow_message')))
    finally:
        _restore_flow_stubs(saved_var, saved_set)


def test_plot_water_to_water_keeps_tsup_and_dtreturn():
    saved = _water_source_stub()
    try:
        with DeviationsHarness(sheet=_pd_sheet()) as h:
            h.add_entry(1)
            _pd_defrost_clocks(h)
            with contextlib.redirect_stdout(io.StringIO()):
                body = h.client.post('/plot_deviation', json={
                    'file_name': PD_FILE, 'data_set': 1,
                    'start_time': 0.0, 'end_time': PD_END, 'rowid': 1,
                }).get_json() or {}
            check('water-to-water has no outdoor figure',
                  body.get('success') is True and body.get('image_dbwb') is None,
                  str(body.get('message') or body.get('image_dbwb')))
            check('Tsup and dTreturn still draw on a water source',
                  bool(body.get('image_tsup')) and bool(body.get('image_dtreturn')),
                  str([bool(body.get('image_tsup')), bool(body.get('image_dtreturn'))]))
    finally:
        webapp._entry_unit_capabilities = saved


def test_unified_plot_leaves_the_tables_and_parent_bands_alone():
    parent = webapp.load_permissible_deviations()
    check('parent dTreturn is still the full-cycle -2...+2 K',
          float(parent['dTreturn']['lower']) == -2.0
          and float(parent['dTreturn']['upper']) == 2.0, str(parent.get('dTreturn')))
    check('parent DB/WB individual widths are unchanged at 1 K',
          float(parent['DB']['value']) == 1.0 and float(parent['WB']['value']) == 1.0)
    check('parent mean DB is still 0.6 K',
          float(parent['mean_db_k']) == 0.6, str(parent.get('mean_db_k')))
    cfg = webapp.load_interval_deviations_config()
    check('D and S still do not borrow the Interval H band set',
          not webapp._interval_pd_has_bands(cfg, 'D')
          and not webapp._interval_pd_has_bands(cfg, 'S'))

    app_src = open(os.path.join(REPO, 'flask_app', 'app.py'), encoding='utf-8').read()
    stepper = app_src.split('def _draw_interval_band_steps')[1].split('def _figure_to_base64')[0]
    check('the stepped fill is still fill_between, never an unclipped axhspan',
          'fill_between' in stepper and 'axhspan(' not in stepper)
    check('the parent overlay is dashed lines, not a second fill',
          'def _draw_parent_overlay' in app_src
          and 'axhspan' not in app_src.split('def _draw_parent_overlay')[1]
                                     .split('def _entry_flow_set')[0])
    check('the reply no longer carries the Group 2 copies',
          'image_dbwb_interval' not in app_src and 'image_dtreturn_interval' not in app_src
          and "'image_interval'" not in app_src)

    dev_html = open(os.path.join(REPO, 'flask_app', 'templates', 'deviations.html'),
                    encoding='utf-8').read()
    check('the Deviations table has one Plot button and no Flow button',
          'btn-plot-flow' not in dev_html and 'plotFlowDeviation' not in dev_html
          and '> Plot' in dev_html, 'Flow button still present')
    imgs = [k for k in ('image_dbwb', 'image_tsup', 'image_dtreturn', 'image_flow')
            if dev_html.count('<img src="${response.' + k + '}"') == 1]
    check('the modal draws each quantity exactly once',
          len(imgs) == 4 and '<img src="${response.image_dbwb_interval}"' not in dev_html
          and '<img src="${response.image_dtreturn_interval}"' not in dev_html,
          str(imgs))
    check('the modal names both fills and keeps the no-clocks note',
          'interval individual' in dev_html
          and 'Save guideline clocks on Cycle Extract' in dev_html
          and 'Parent cycle (full-cycle bands)' not in dev_html)
    check('plot titles stay short so stacked figures share a width',
          'def _compact_plot_title' in app_src
          and 'PLOT_FIGSIZE' in app_src
          and 'Permissible Deviation:' not in app_src.split('def _compact_plot_title')[1]
                                    .split('def plot_deviation')[0]
          and 'shaded: saved guideline' not in app_src.split('def plot_deviation')[1]
                                                     .split('def plot_flow_deviation')[0])
    check('no new plot route or page was added',
          '/plot_interval' not in dev_html and '/plot_interval' not in app_src)

    gw_html = open(os.path.join(REPO, 'flask_app', 'templates', 'guideline_windows.html'),
                   encoding='utf-8').read()
    check('Guideline Windows still carries no plots',
          '<img' not in gw_html.lower() and 'plot_deviation' not in gw_html)
    check('Guideline Windows still has no Tsup %',
          'tsup_pct' not in gw_html and 'Tsup %' not in gw_html)


# ---------------------------------------------------------------------------
# Guideline Windows score cache. The interval scores are written when the
# clocks are **saved** and read back from SQLite: the page never opens a
# Plotdaten workbook, so a second visit to the page costs a query. A parent with clocks
# and no cache row keeps its clock times and shows n/a with a reason until the
# explicit **Compute missing scores** backfill has run.
#
# Synthetic sheet and a throwaway SQLite, as everywhere else in this file.
# ---------------------------------------------------------------------------

CACHE_PERIODS = [
    {'period_type': 'defrost', 'start_time': 0.0, 'end_time': 1200.0},
    {'period_type': 'heating', 'start_time': 1200.0, 'end_time': 12000.0},
    {'period_type': 'equilibrium', 'start_time': 1200.0, 'end_time': 4800.0},
    {'period_type': 'evaluation', 'start_time': 4800.0, 'end_time': 9000.0},
]


def _cache_defrost_clocks(h, rowid=1):
    """The same clocks as `_defrost_entry`, written straight to SQLite."""
    for p in CACHE_PERIODS:
        h.add_period(rowid, p['period_type'], p['start_time'], p['end_time'])


def _cache_rows(h):
    """{rowid: (fingerprint, payload)} straight out of the cache table."""
    conn = h._connect()
    try:
        rows = conn.execute(
            'SELECT entry_rowid, buffer_s, fingerprint, payload FROM guideline_window_scores'
        ).fetchall()
        return {int(r['entry_rowid']): (r['fingerprint'], json.loads(r['payload']), r['buffer_s'])
                for r in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def test_cache_save_clocks_stores_the_scores():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        d = h.save_clocks(1, 'defrost', CACHE_PERIODS)
        check('the per-row Save still reports its periods',
              d.get('success') is True and d.get('saved') == 4, str(d))
        check('and it scored the row it just saved', d.get('scored') == 1
              and not d.get('score_failed'), str(d))

        cache = _cache_rows(h)
        check('a cache row exists for that parent', list(cache) == [1], str(list(cache)))
        fingerprint, payload, buffer_s = cache[1]
        check('the cache row carries a fingerprint',
              isinstance(fingerprint, str) and len(fingerprint) > 16, str(fingerprint))
        check('it is stored at the configured buffer', buffer_s == h.buffer_s, str(buffer_s))
        check('the H fields are in the payload',
              payload.get('h_db_pct') is not None and payload.get('h_mean_db_k') is not None,
              str([payload.get('h_db_pct'), payload.get('h_mean_db_k')]))
        check('the evaluation four are in the payload',
              all(payload.get(k) is not None
                  for k in ('eval_tsup', 'eval_q', 'eval_p', 'eval_cop')),
              str([payload.get(k) for k in ('eval_tsup', 'eval_q', 'eval_p', 'eval_cop')]))
        check('Save read the sheet once, not once per helper',
              len(h.sheet_reads) == 1, str(h.sheet_reads))

        # The page now answers from the store: same numbers, no workbook.
        h.sheet_reads = []
        row = h.one_row(1, compute=False)
        check('the table reports the scores as stored', row.get('scores_cached') is True,
              str(row.get('scores_cached')))
        check('the stored H % comes back on the row',
              row.get('h_db_pct') == payload.get('h_db_pct'),
              str([row.get('h_db_pct'), payload.get('h_db_pct')]))
        check('the stored evaluation COP comes back on the row',
              row.get('eval_cop') == payload.get('eval_cop'),
              str([row.get('eval_cop'), payload.get('eval_cop')]))
        check('the clock times are still read from cycle_periods',
              (row.get('eval_start'), row.get('eval_end')) == (4800.0, 9000.0),
              str([row.get('eval_start'), row.get('eval_end')]))
        check('and _read_excel_sheet was not called', h.sheet_reads == [], str(h.sheet_reads))


def test_cache_second_load_all_still_reads_no_sheet():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        h.add_entry(2, 12000.0, 24000.0, tsup=41.0, q=6.1, p=2.6, cop=2.3)
        h.save_clocks(1, 'defrost', CACHE_PERIODS)
        h.save_clocks(2, 'defrost', [{'period_type': p['period_type'],
                                      'start_time': p['start_time'] + 12000.0,
                                      'end_time': p['end_time'] + 12000.0}
                                     for p in CACHE_PERIODS])
        h.sheet_reads = []
        _, first = h.load(None, compute=False)
        check('the whole-database load answers for both parents', len(first.get('rows') or []) == 2,
              str(len(first.get('rows') or [])))
        check('the first load opens no workbook', h.sheet_reads == [], str(h.sheet_reads))
        _, second = h.load(None, compute=False)
        check('the second load opens none either', h.sheet_reads == [], str(h.sheet_reads))
        check('and it is the same table',
              [r.get('h_db_pct') for r in second['rows']] == [r.get('h_db_pct') for r in first['rows']],
              str([r.get('h_db_pct') for r in second['rows']]))
        check('the status names entries, clocks and stored scores',
              first.get('n_entries') == 2 and first.get('n_with_clocks') == 2
              and first.get('n_with_scores') == 2 and first.get('n_missing_scores') == 0,
              str([first.get('n_entries'), first.get('n_with_clocks'),
                   first.get('n_with_scores'), first.get('n_missing_scores')]))


def test_cache_clocks_that_changed_are_a_miss_not_a_stale_percent():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        h.save_clocks(1, 'defrost', CACHE_PERIODS)
        stored = h.one_row(1, compute=False).get('h_db_pct')
        check('there is a stored H % to begin with', stored is not None, str(stored))

        # The clocks move without going through Save — the fingerprint must not
        # let the old percentage stand for the new window.
        conn = h._connect()
        try:
            conn.execute("UPDATE cycle_periods SET end_time=6000.0 WHERE entry_rowid=1 "
                         "AND period_type='heating'")
            conn.commit()
        finally:
            conn.close()
        h.sheet_reads = []
        row = h.one_row(1, compute=False)
        check('the moved clocks are a cache miss', row.get('scores_cached') is False,
              str(row.get('scores_cached')))
        check('no stale percentage is shown',
              all(row.get(k) is None for k in ('h_db_pct', 'h_mean_db_k', 'eval_delta_cop')),
              str([row.get('h_db_pct'), row.get('h_mean_db_k'), row.get('eval_delta_cop')]))
        check('the clock times still show', (row.get('h_start'), row.get('h_end')) == (1200.0, 6000.0),
              str([row.get('h_start'), row.get('h_end')]))
        check('the cell says why it is empty',
              'Compute missing scores' in ((row.get('db_reasons') or {}).get('h') or ''),
              str((row.get('db_reasons') or {}).get('h')))
        check('and the miss opened no workbook', h.sheet_reads == [], str(h.sheet_reads))


def test_cache_clear_saved_clocks_drops_the_cache_too():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        h.save_clocks(1, 'defrost', CACHE_PERIODS)
        check('the cache is there before the clear', list(_cache_rows(h)) == [1], str(_cache_rows(h)))

        with contextlib.redirect_stdout(io.StringIO()):
            resp = h.client.post('/api/cycle_extract/clear_guideline_clocks',
                                 json={'file_name': WINDOWS_FILE})
            d = resp.get_json() or {}
        check('the clear reports the clocks it removed', d.get('success') is True
              and d.get('cleared_entries') == 1, str(d))
        check('and it reports the cache it removed with them', d.get('cleared_scores') == 1, str(d))
        check('the cache table is empty', _cache_rows(h) == {}, str(_cache_rows(h)))

        row = h.one_row(1, compute=False)
        check('the row is back to no clocks at all', row.get('kind') == 'unknown', str(row.get('kind')))
        check('and it carries no scores', row.get('h_db_pct') is None, str(row.get('h_db_pct')))


def test_cache_compute_missing_is_the_backfill_and_never_runs_on_a_get():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        # The backfill case: clocks were saved, scores were not.
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        _cache_defrost_clocks(h, 1)

        h.sheet_reads = []
        with contextlib.redirect_stdout(io.StringIO()):
            page = h.client.get('/guideline_windows')
        check('the page still renders', page.status_code == 200, str(page.status_code))
        _, cold = h.load(None, compute=False)
        cold_row = (cold.get('rows') or [{}])[0]
        check('opening the page runs no backfill', h.sheet_reads == [], str(h.sheet_reads))
        check('the clocks are there without scores',
              cold_row.get('kind') == 'defrost' and cold_row.get('scores_cached') is False,
              str([cold_row.get('kind'), cold_row.get('scores_cached')]))
        check('the status counts the gap',
              cold.get('n_with_clocks') == 1 and cold.get('n_with_scores') == 0
              and cold.get('n_missing_scores') == 1, str(cold.get('message')))

        # The confirm list is SQLite only.
        with contextlib.redirect_stdout(io.StringIO()):
            listing = (h.client.post('/api/guideline_windows/score_files', json={}).get_json() or {})
        check('the confirm list names the file and the gap',
              listing.get('n_missing') == 1
              and [f['file_name'] for f in listing.get('files') or []] == [WINDOWS_FILE],
              str(listing))
        check('and counting opens no workbook', h.sheet_reads == [], str(h.sheet_reads))

        report = h.compute_scores()
        check('the backfill stores the missing scores',
              report.get('computed') == 1 and not report.get('failed'), str(report))
        check('it read the file once', len(h.sheet_reads) == 1, str(h.sheet_reads))

        h.sheet_reads = []
        _, warm = h.load(None, compute=False)
        warm_row = (warm.get('rows') or [{}])[0]
        check('the whole-database load is sheet-free afterwards', h.sheet_reads == [], str(h.sheet_reads))
        check('and the scores are on the row', warm_row.get('scores_cached') is True
              and warm_row.get('h_db_pct') is not None,
              str([warm_row.get('scores_cached'), warm_row.get('h_db_pct')]))

        again = h.compute_scores()
        check('running it again computes nothing and reads nothing',
              again.get('computed') == 0 and again.get('skipped') == 1, str(again))


def test_cache_opening_the_page_with_no_selection_opens_no_workbook():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        _cache_defrost_clocks(h, 1)
        h.add_entry(2, 12000.0, 24000.0, tsup=41.0)

        with contextlib.redirect_stdout(io.StringIO()):
            page = h.client.get('/guideline_windows')
        body = page.get_data(as_text=True)
        check('the page renders', page.status_code == 200, str(page.status_code))
        check('the GET opens no workbook', h.sheet_reads == [], str(h.sheet_reads))
        # The Load-all button is gone; the page loads every parent itself.
        check('and it shows the table instead of waiting for a click',
              "gwLoad(null, 'every entry in the open database')" in body, body[-600:])

        # The table the page then asks for is the same cheap read.
        _, d = h.load(None, compute=False)
        check('every parent is listed, with and without clocks',
              sorted(r['kind'] for r in d.get('rows') or []) == ['defrost', 'unknown'],
              str([r['kind'] for r in d.get('rows') or []]))
        check('and that table opened no workbook either', h.sheet_reads == [], str(h.sheet_reads))
        check('the status says nothing was read from Plotdaten',
              'no Plotdaten file was opened' in (d.get('message') or ''), str(d.get('message')))

        gw_src = _function_code(open(os.path.join(REPO, 'flask_app', 'app.py'),
                                     encoding='utf-8').read(), '_guideline_window_rows')
        check('and the row builder has no sheet call left in it',
              '_read_excel_sheet' not in gw_src and '_prepare_sheet_for_analysis' not in gw_src,
              gw_src[:200])


def test_cache_database_clock_fill_also_stores_the_scores():
    """The database fill writes clocks for the open database — and now its scores."""
    with BatchHarness() as h:
        seen, restore = _watch_sheet_reads()
        try:
            out = _batch_fill(h, [BATCH_A, BATCH_B], default_kind='defrost')
        finally:
            restore()
        check('both files saved their parents',
              [d.get('saved_entries') for d in out] == [2, 2], str(out))
        check('and both scored what they saved',
              [d.get('scored') for d in out] == [2, 2]
              and not any(d.get('score_failed') for d in out), str(out))
        # Scoring must ride on the sheet the save already held. The control is
        # the same fill with the scoring stubbed out: the read count must match.
        with BatchHarness() as control:
            real = webapp._store_scores_for_parents
            webapp._store_scores_for_parents = lambda rowids, sheets=None: ([], [])
            bare, restore_bare = _watch_sheet_reads()
            try:
                _batch_fill(control, [BATCH_A, BATCH_B], default_kind='defrost')
            finally:
                restore_bare()
                webapp._store_scores_for_parents = real
        check('scoring adds no sheet read of its own',
              sorted(seen) == sorted(bare), f'{sorted(seen)} vs {sorted(bare)}')
        check('and both files were touched', set(seen) == {BATCH_A, BATCH_B}, str(set(seen)))

        conn = h._connect()
        try:
            cached = sorted(r[0] for r in conn.execute(
                'SELECT entry_rowid FROM guideline_window_scores').fetchall())
        finally:
            conn.close()
        check('every filled parent has a cache row', cached == [1, 2, 3, 4], str(cached))

        # And the second pass over the same database is still a no-op.
        seen, restore = _watch_sheet_reads()
        try:
            again = _batch_fill(h, [BATCH_A, BATCH_B], default_kind='defrost')
        finally:
            restore()
        check('nothing is written twice',
              [d.get('saved_entries') for d in again] == [0, 0], str(again))
        check('and no sheet is opened for it', seen == [], str(seen))


def test_cache_is_its_own_table_and_leaves_the_deviations_payload_alone():
    app_src = open(os.path.join(REPO, 'flask_app', 'app.py'), encoding='utf-8').read()
    check('the cache has its own table',
          'CREATE TABLE IF NOT EXISTS guideline_window_scores' in app_src)
    check('the per-interval Deviations payload table is still created',
          'CREATE TABLE IF NOT EXISTS entry_interval_deviations' in app_src)
    for name in ('_store_guideline_scores', '_guideline_window_rows',
                 '_guideline_score_rows_on_sheet', '_store_scores_for_parents'):
        check(f'{name} does not touch entry_interval_deviations',
              'entry_interval_deviations' not in _function_code(app_src, name))
    check('and /calculate_deviations still writes it',
          '_recalculate_interval_deviations' in _function_code(app_src, 'calculate_deviations')
          and '_store_interval_deviations' in _function_code(app_src,
                                                             '_recalculate_interval_deviations'))

    with DeviationsHarness() as h:
        h.add_entry(1)
        _pd_defrost_clocks(h, 1)
        resp, d = h.calculate([1])
        check('Calculate still fills the parent dev_* columns',
              d.get('success') is True and h.parent_row(1).get('dev_tsup_percentage') is not None,
              str(d))
        conn = h._connect()
        try:
            stored = conn.execute('SELECT COUNT(*) AS n FROM entry_interval_deviations').fetchone()['n']
        finally:
            conn.close()
        check('and it still writes its own side table, not the score cache', stored == 1, str(stored))

    gw_html = open(os.path.join(REPO, 'flask_app', 'templates', 'guideline_windows.html'),
                   encoding='utf-8').read()
    check('the cache added no plot to Guideline Windows',
          '<img' not in gw_html.lower() and 'plot_deviation' not in gw_html)
    check('and no Tsup % column',
          'tsup_pct' not in gw_html and 'Tsup %' not in gw_html)
    check('Compute missing scores is a confirm, not a page-load action',
          'gwOpenComputeConfirm()' in gw_html and 'gwComputeModal' in gw_html
          and 'gwRunCompute(' in gw_html
          and 'gwRunCompute(files); }' in gw_html)


def test_cache_scoring_does_not_flood_the_console():
    # `_uncorr_sheet` has no BUH column and no configured cap, so the unguarded
    # analysis path prints both lines for every cycle it touches.
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        _cache_defrost_clocks(h, 1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            resp = h.client.post('/api/guideline_windows/compute_scores',
                                 json={'file_name': WINDOWS_FILE})
            report = resp.get_json() or {}
        printed = buf.getvalue()
        check('the backfill did compute something', report.get('computed') == 1, str(report))
        check('no cap-value line reached the console',
              'cap value for' not in printed, printed[:400])
        check('no BUH-missing line reached the console',
              'Electrical power input BUH' not in printed, printed[:400])

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            h.client.post('/api/guideline_windows', json={'rowids': None})
        check('and reading the table prints neither', 'cap value for' not in buf.getvalue()
              and 'Electrical power input BUH' not in buf.getvalue(), buf.getvalue()[:400])


# ---------------------------------------------------------------------------
# Guideline Windows UI: `#` instead of rowid, a COP dataset column,
# three client-side filters, a capped scroll box, no Load-all button.
#
# `#` is the entry's 1-based position in the whole open database in Mean Values
# order (`display_order ASC, rowid ASC`) — not the SQLite rowid, which has holes
# after a delete, and not 1…N of a Mean Values subset. Filters hide rows; they
# never renumber. Synthetic databases only.
# ---------------------------------------------------------------------------


def _gw_html():
    return open(os.path.join(REPO, 'flask_app', 'templates', 'guideline_windows.html'),
                encoding='utf-8').read()


def _two_rows_with_gappy_rowids(h):
    """Two parents whose rowids are 4 and 9 — the shape a delete leaves behind."""
    h.add_entry(4, 0.0, 12000.0, order=1, tsup=40.0, q=6.0, p=2.5, cop=2.4, cop_dataset='YES')
    h.add_entry(9, 12000.0, 24000.0, order=2, tsup=41.0, q=6.1, p=2.6, cop=2.3, cop_dataset='NO')
    for rid, off in ((4, 0.0), (9, 12000.0)):
        h.add_period(rid, 'defrost', off + 0.0, off + 1200.0)
        h.add_period(rid, 'heating', off + 1200.0, off + 12000.0)


def test_gw_ui_index_is_the_position_not_the_rowid():
    with WindowsHarness() as h:
        _two_rows_with_gappy_rowids(h)
        _, d = h.load(None, compute=False)
        rows = d.get('rows') or []
        check('both parents come back', len(rows) == 2, str(len(rows)))
        check('# counts 1, 2 in display order', [r.get('index') for r in rows] == [1, 2],
              str([r.get('index') for r in rows]))
        check('# is not the rowid', [r.get('rowid') for r in rows] == [4, 9],
              str([r.get('rowid') for r in rows]))
        check('rowid stays in the JSON for the APIs that select by it',
              all('rowid' in r for r in rows), str(sorted(rows[0])))


def test_gw_ui_a_mean_values_subset_keeps_the_full_database_index():
    with WindowsHarness() as h:
        _two_rows_with_gappy_rowids(h)
        _, d = h.load([9], compute=False)
        rows = d.get('rows') or []
        check('the subset holds only that parent', [r.get('rowid') for r in rows] == [9],
              str([r.get('rowid') for r in rows]))
        check('its # is still the whole-database 2, not 1',
              [r.get('index') for r in rows] == [2], str([r.get('index') for r in rows]))


def test_gw_ui_display_order_not_rowid_decides_the_index():
    # A row inserted later but ordered first must take #1, the way Mean Values
    # shows it — the order is `display_order ASC, rowid ASC`, never rowid alone.
    with WindowsHarness() as h:
        h.add_entry(4, 0.0, 12000.0, order=7)
        h.add_entry(9, 12000.0, 24000.0, order=2)
        _, d = h.load(None, compute=False)
        got = [(r.get('index'), r.get('rowid')) for r in (d.get('rows') or [])]
        check('display_order decides #, not the rowid', got == [(1, 9), (2, 4)], str(got))
        _, sub = h.load([4], compute=False)
        got_sub = [(r.get('index'), r.get('rowid')) for r in (sub.get('rows') or [])]
        check('and a subset of the later row keeps #2', got_sub == [(2, 4)], str(got_sub))


def test_gw_ui_cop_dataset_on_the_row_and_in_the_csv():
    with WindowsHarness() as h:
        _two_rows_with_gappy_rowids(h)
        h.add_entry(11, 24000.0, 36000.0, order=3)      # no COP dataset value at all
        _, d = h.load(None, compute=False)
        got = [(r.get('index'), r.get('cop_dataset')) for r in (d.get('rows') or [])]
        check('COP dataset is YES / NO / empty on the row',
              got == [(1, 'YES'), (2, 'NO'), (3, '')], str(got))

        with contextlib.redirect_stdout(io.StringIO()):
            resp = h.client.post('/api/guideline_windows/export', json={'rowids': None})
        lines = resp.data.decode('utf-8-sig').splitlines()
        header = lines[0].split(',')
        cells = [line.split(',') for line in lines[1:]]
        check('the CSV opens on # and has no rowid column',
              header[0] == '#' and 'rowid' not in header, str(header[:8]))
        check('the CSV has a COP_dataset column', 'COP_dataset' in header, str(header[:8]))
        cop_at = header.index('COP_dataset')
        check('# in the CSV is 1, 2, 3', [c[0] for c in cells] == ['1', '2', '3'],
              str([c[0] for c in cells]))
        check('YES / NO / empty reach the CSV',
              [c[cop_at] for c in cells] == ['YES', 'NO', ''], str([c[cop_at] for c in cells]))


def test_gw_ui_an_empty_cop_cell_is_not_a_no():
    with WindowsHarness() as h:
        h.add_entry(1, 0.0, 12000.0, order=1, cop_dataset='')
        h.add_entry(2, 12000.0, 24000.0, order=2, cop_dataset='  yes ')
        _, d = h.load(None, compute=False)
        got = [r.get('cop_dataset') for r in (d.get('rows') or [])]
        check('an empty cell stays empty and a stray case/space is normalised',
              got == ['', 'YES'], str(got))


def test_gw_ui_csv_is_the_filtered_table():
    # The three filters run in the browser; the page then posts the rowids they
    # left visible, so the file is the table on screen — in display order.
    with WindowsHarness() as h:
        _two_rows_with_gappy_rowids(h)
        h.add_entry(11, 24000.0, 36000.0, order=3)      # no clocks — "no saved clocks"
        with contextlib.redirect_stdout(io.StringIO()):
            resp = h.client.post('/api/guideline_windows/export', json={'rowids': [9]})
        rows = resp.data.decode('utf-8-sig').splitlines()[1:]
        check('only the visible row is written', len(rows) == 1, str(rows))
        check('and it keeps its whole-database #2', rows[0].startswith('2,'), rows[0])


def test_gw_ui_a_whole_database_selection_is_chunked_and_stays_in_order():
    # With no filter set the page posts every visible rowid, so an explicit list
    # the size of the database now reaches the row read. It must not pass the
    # bound-parameter limit, and the chunks must come back in Mean Values order.
    n = 1500
    with WindowsHarness() as h:
        conn = h._connect()
        try:
            conn.executemany(
                "INSERT INTO results (rowid, file_name, data_set, start_time, end_time, "
                "display_order, test_cond, profile_id, HP_ID) VALUES (?,?,?,?,?,?,?,?,?)",
                [(rid, WINDOWS_FILE, 1, 0.0, 12000.0, n + 1 - rid, 'E', 'HPT_RRT1', '1')
                 for rid in range(1, n + 1)])
            conn.commit()
        finally:
            conn.close()
        _, d = h.load(list(range(1, n + 1)), compute=False)
        rows = d.get('rows') or []
        check('every parent of a database-sized selection comes back', len(rows) == n,
              str(len(rows)))
        check('# still counts 1…n', [r.get('index') for r in rows] == list(range(1, n + 1)),
              str([r.get('index') for r in rows[:5]]))
        check('and the chunks are re-sorted into display_order, not rowid order',
              [r.get('rowid') for r in rows[:3]] == [n, n - 1, n - 2],
              str([r.get('rowid') for r in rows[:3]]))


def test_gw_ui_template_has_the_index_cop_column_and_the_filters():
    gw_html = _gw_html()
    check('the Load-all button is gone',
          'Load all entries in the open database' not in gw_html
          and 'gwLoadAllBtn' not in gw_html and 'gwLoadAll(' not in gw_html)
    check('Load the selected row(s) stays', 'gwLoadSelected()' in gw_html)
    check('Compute missing scores and Download CSV stay',
          'Compute missing scores' in gw_html and 'Download CSV' in gw_html)
    check('the table header opens on # and has no rowid header',
          '<th id="gwHdrIndex">#</th>' in gw_html and '<th>rowid</th>' not in gw_html)
    check('COP dataset is a header in the Entry group',
          '<th id="gwHdrCop">COP dataset</th>' in gw_html)
    check('the row renders r.index and r.cop_dataset, not r.rowid',
          'r.index' in gw_html and 'r.cop_dataset' in gw_html
          and "gwCell(r.rowid, 'gw-num')" not in gw_html)
    for control in ('gwFilterKind', 'gwFilterCop', 'gwFilterClocks'):
        check(control + ' is on the action bar', 'id="' + control + '"' in gw_html)
    check('the kind options are the four kinds plus All',
          all('value="' + v + '"' in gw_html
              for v in ('defrost', 'on_off', 'continuous', 'unknown')))
    check('the clocks filter offers with / without saved clocks',
          'with saved clocks' in gw_html and 'no saved clocks' in gw_html)
    check('there is a Clear filters control',
          'Clear filters' in gw_html and 'gwClearFilters(' in gw_html)
    check('the filters are client-side on the loaded array',
          'gwFilterRows' in gw_html and '_gwRows' in gw_html and 'gwApplyFilters' in gw_html)
    check('the status line shows X of Y', 'Showing ' in gw_html and "' of '" in gw_html)
    check('no per-column filter row is added', 'filter-row' not in gw_html)
    check('the CSV posts the visible rows, not the loaded selector', '_gwVisible.map' in gw_html)
    filter_js = gw_html.split('function gwFilterRows')[1].split('async function gwLoad')[0]
    check('the filters open no workbook and never reach Compute missing',
          'gwRunCompute' not in filter_js and 'compute_scores' not in filter_js
          and 'fetch(' not in filter_js, filter_js[:200])


def test_gw_ui_table_box_caps_its_width():
    gw_html = _gw_html()
    check('the grow-with-content wrap is gone', 'gw-table-wrap' not in gw_html)
    block = gw_html.split('.gw-table-scroll {')[1].split('}')[0]
    check('the scroll box caps its width instead of growing with the table',
          'max-width' in block, block)
    check('it scrolls sideways inside the box', 'overflow-x: auto' in block, block)
    check('the horizontal bar is styled so it is usable',
          '.gw-table-scroll::-webkit-scrollbar { height:' in gw_html
          and 'scrollbar-color' in block, block)
    check('Mean Values sticky columns were not copied over',
          'col-sticky' not in gw_html and 'fixed-scrollbar' not in gw_html)


def test_gw_ui_changed_nothing_it_was_told_to_leave_alone():
    app_src = open(os.path.join(REPO, 'flask_app', 'app.py'), encoding='utf-8').read()
    body = app_src.split('def _guideline_window_rows(')[1].split('\ndef ')[0]
    check('the table read still opens no workbook',
          '_read_excel_sheet' not in body and '_prepare_sheet_for_analysis' not in body)
    check('it still reads the score cache, not a sheet', '_load_guideline_scores' in body)
    check('entry_interval_deviations is untouched by this page',
          'entry_interval_deviations' not in body)
    gw_html = _gw_html()
    check('still no Tsup % on the page', 'Tsup %' not in gw_html)
    check('still no plot control on the page',
          'plot' not in gw_html.lower().replace('plotdaten', ''))

    with WindowsHarness(sheet=_pd_sheet()) as h:
        _two_rows_with_gappy_rowids(h)
        h.sheet_reads = []
        _, d = h.load(None, compute=False)
        rows = d.get('rows') or []
        check('kind still comes from the stored periods, not the test letter',
              [r.get('kind') for r in rows] == ['defrost', 'defrost'],
              str([r.get('kind') for r in rows]))
        check('a row with no stored scores is still n/a, never 0',
              all(r.get('h_db_pct') is None for r in rows),
              str([r.get('h_db_pct') for r in rows]))
        check('adding # and COP dataset opened no sheet', h.sheet_reads == [], str(h.sheet_reads))


def main():
    test_defrost_opens_in_d()
    test_defrost_opens_in_d_without_later_drop()
    test_defrost_opens_in_d_with_later_drop()
    test_two_span_h_too_short_drops_evaluation()
    test_on_off_two_span()
    test_analyst_moves_or_removes_the_second_span()
    test_defrost_opens_in_h()
    test_h_too_short_for_evaluation()
    test_h_too_short_for_equilibrium()
    test_on_off_gets_buffer_but_no_eq_eval()
    test_continuous_long_window()
    test_continuous_short_window_has_no_evaluation()
    test_continuous_needs_no_power_transition()
    test_other_is_unknown_not_continuous()
    test_file_level_default_applies_only_to_unknown()
    test_moving_ds_end_rebuilds_h_and_locked_clocks()
    test_unlocked_eq_eval_are_kept()
    test_no_transition_is_reported_not_invented()
    test_lengths_come_from_config()
    test_opens_low_detection()
    test_second_drop_is_searched_after_the_first_span()
    test_api_file_default_makes_unknown_rows_continuous()
    test_api_save_all_without_ticks_saves_the_whole_file()
    test_api_replies_are_json_even_when_the_request_fails()
    test_existing_entries_carry_the_saved_clocks_without_a_sheet_read()
    test_saved_second_d_span_survives_the_next_propose()
    test_saved_one_span_defrost_is_not_re_split()
    test_opens_in_h_with_one_trailing_d_is_not_seeded()
    test_saved_clock_seed_helper_reads_only_d_s_spans()
    test_cycle_extract_show_and_layers_can_use_saved_clocks()
    test_clock_layers_changed_nothing_outside_cycle_extract()
    test_batch_file_list_is_cheap_and_counts_what_is_saved()
    test_batch_skip_saved_fills_the_gap_and_leaves_saved_rows_alone()
    test_batch_skip_saved_on_a_fully_saved_file_reads_no_sheet()
    test_batch_unknown_rows_and_one_bad_sheet_stop_only_their_own_file()
    test_batch_clear_is_what_makes_a_pass_repeatable()
    test_cycle_extract_has_the_database_batch_controls()
    test_batch_changed_nothing_outside_cycle_extract()
    test_windows_only_the_requested_rows_are_read()
    test_windows_defrost_evaluation_is_the_stored_window()
    test_windows_on_off_has_empty_evaluation_cells()
    test_windows_short_h_has_empty_evaluation_cells()
    test_windows_continuous_shows_its_saved_evaluation()
    test_windows_two_span_defrost_shows_both()
    test_windows_row_without_clocks_still_shows_the_parent()
    test_windows_nothing_is_read_until_the_analyst_asks()
    test_windows_toolbar_selection_opens_the_page_with_those_rowids()
    test_windows_null_rowids_loads_every_parent()
    test_windows_api_errors_are_json()
    test_windows_csv_export_is_the_loaded_table()
    test_eval_means_on_an_uncorrected_sheet()
    test_eval_means_prefer_lab_corrected_series()
    test_eval_means_ignore_a_tsup_only_cache()
    test_eval_means_when_the_sheet_cannot_be_read()
    test_eval_means_stay_empty_without_an_evaluation_window()
    test_interval_pd_without_clocks_leaves_the_parent_alone()
    test_interval_pd_defrost_scores_h_eq_eval_with_the_h_bands()
    test_interval_pd_on_off_has_no_equilibrium_evaluation_or_delta_cop()
    test_interval_pd_short_h_has_no_evaluation_or_delta_cop()
    test_interval_pd_delta_cop_just_under_and_just_over_the_limit()
    test_interval_pd_short_evaluation_window_has_no_delta_cop()
    test_interval_pd_a_red_delta_cop_does_not_stop_an_insert()
    test_interval_pd_page_no_longer_shows_the_extra_table()
    test_gw_dtreturn_on_saved_defrost_ignores_a_d_only_excursion()
    test_gw_dtreturn_uses_the_half_k_h_band_not_the_parent_pm_2()
    test_gw_on_off_and_short_h_leave_eval_dtreturn_and_delta_cop_empty()
    test_gw_delta_cop_is_from_eval_not_from_h()
    test_gw_html_json_csv_have_no_tsup_percent_fields()
    test_gw_db_wb_on_saved_defrost_ignore_a_d_only_excursion()
    test_gw_db_outside_h_band_is_100_percent()
    test_gw_on_off_leaves_eq_eval_db_wb_flow_empty()
    test_gw_db_uses_the_deviations_series_not_the_raw_excel_cache()
    test_interval_json_ds_widths_are_not_interval_h()
    test_gw_defrost_scores_d_pct_and_leaves_s_na()
    test_gw_d_db_width_is_five_k_not_interval_h()
    test_gw_d_dtreturn_width_is_two_k_not_interval_h()
    test_gw_two_d_spans_are_one_union_percent()
    test_gw_on_off_scores_s_pct_and_leaves_d_na()
    test_gw_continuous_leaves_d_and_s_na()
    test_cycle_extract_period_layers_stay_off_and_have_no_tsup_band()
    test_missing_db_setpoint_does_not_blank_h_flow_pct()
    test_s_flow_hover_names_variable_flow()
    test_guideline_clock_spans_follow_saved_periods_not_the_test_letter()
    test_plot_deviation_shades_saved_clocks_on_all_figures()
    test_plot_flow_uses_profile_id_from_the_entry()
    test_gw_page_has_no_plots()
    test_gw_mean_on_saved_defrost()
    test_gw_h_mean_db_band_is_zero_point_three()
    test_gw_d_mean_db_band_is_one_point_five()
    test_gw_two_d_spans_are_one_d_mean()
    test_gw_on_off_scores_s_mean_and_leaves_d_mean_na()
    test_gw_continuous_leaves_d_and_s_means_na()
    test_gw_means_without_a_window_or_a_source_are_na_with_a_reason()
    test_gw_mean_columns_in_html_json_and_csv()
    test_gw_mean_pds_do_not_reach_the_parent_or_the_insert()
    test_step_b_helper_on_a_saved_defrost()
    test_step_b_helper_two_d_spans_keep_h_in_the_gap()
    test_step_b_helper_on_off_and_continuous()
    test_step_b_helper_without_clocks_or_with_unknown_types()
    test_plot_deviation_is_one_figure_per_quantity()
    test_plot_deviation_carries_flow_in_the_same_modal()
    test_plot_water_to_water_keeps_tsup_and_dtreturn()
    test_unified_plot_leaves_the_tables_and_parent_bands_alone()
    test_cache_save_clocks_stores_the_scores()
    test_cache_second_load_all_still_reads_no_sheet()
    test_cache_clocks_that_changed_are_a_miss_not_a_stale_percent()
    test_cache_clear_saved_clocks_drops_the_cache_too()
    test_cache_compute_missing_is_the_backfill_and_never_runs_on_a_get()
    test_cache_opening_the_page_with_no_selection_opens_no_workbook()
    test_cache_database_clock_fill_also_stores_the_scores()
    test_cache_is_its_own_table_and_leaves_the_deviations_payload_alone()
    test_cache_scoring_does_not_flood_the_console()
    test_gw_ui_index_is_the_position_not_the_rowid()
    test_gw_ui_a_mean_values_subset_keeps_the_full_database_index()
    test_gw_ui_display_order_not_rowid_decides_the_index()
    test_gw_ui_cop_dataset_on_the_row_and_in_the_csv()
    test_gw_ui_an_empty_cop_cell_is_not_a_no()
    test_gw_ui_csv_is_the_filtered_table()
    test_gw_ui_a_whole_database_selection_is_chunked_and_stays_in_order()
    test_gw_ui_template_has_the_index_cop_column_and_the_filters()
    test_gw_ui_table_box_caps_its_width()
    test_gw_ui_changed_nothing_it_was_told_to_leave_alone()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print('All guideline-clock checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
