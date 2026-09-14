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
uncorrected Q and P (the BAM case) use the same analysis-time path as the
stored entry.
"""
import contextlib
import io
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
    """[(type, start, end)] — use this where D/S may appear twice (§2.1.1)."""
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
# API level (Phase 1.2b): the file-level default must reach the clocks, and no
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
    """Flask test client wired to a temporary database and a stubbed sheet."""

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp(prefix='guideline_clocks_')
        self.db = os.path.join(self.tmpdir, 'test.db')
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE results (file_name TEXT, data_set REAL, start_time REAL, "
                     "end_time REAL, cycle_type TEXT, cycle_start_marker TEXT, "
                     "pelec_transition_time REAL, pelec_transition_source TEXT, "
                     "pelec_detect_failed INTEGER, HP_ID TEXT, indicator TEXT, notes TEXT)")
        # One row with no classification at all, one left over as 'other'.
        conn.executemany(
            "INSERT INTO results (file_name, data_set, start_time, end_time, cycle_type) VALUES (?,?,?,?,?)",
            [(API_FILE, 1, 0, 12000, None), (API_FILE, 1, 12000, 24000, 'other')])
        conn.commit()
        conn.close()

        self._saved = {k: getattr(webapp, k) for k in
                       ('get_db_connection', 'get_database_path', '_read_excel_sheet')}
        webapp.get_db_connection = self._connect
        webapp.get_database_path = lambda: self.db
        webapp._read_excel_sheet = _fake_sheet
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

        # A kind on the row must survive the file-level default (§2.2.1).
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
# Phase 2: the Guideline windows page reads back what Save stored — stored
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
    """A BAM-shaped sheet: uncorrected Q and P plus what the pump correction needs.

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
            "avg_heating_capacity_corr_wbuh REAL, PCorrwBUH REAL, COPCorrwBUH REAL)")
        conn.commit()
        conn.close()

        self._saved = {k: getattr(webapp, k) for k in
                       ('get_db_connection', 'get_database_path', '_read_excel_sheet')}
        webapp.get_db_connection = self._connect
        webapp.get_database_path = lambda: self.db
        webapp._read_excel_sheet = self._read_sheet
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

    def add_entry(self, rowid, start, end, order=None, tsup=None, q=None, p=None, cop=None):
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO results (rowid, file_name, data_set, start_time, end_time, display_order, "
                "test_cond, profile_id, HP_ID, avg_ts_buh, avg_heating_capacity_corr_wbuh, PCorrwBUH, "
                "COPCorrwBUH) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rowid, WINDOWS_FILE, 1, start, end, order if order is not None else rowid,
                 'E', 'HPT_RRT1', '1', tsup, q, p, cop))
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

    def load(self, rowids):
        with contextlib.redirect_stdout(io.StringIO()):
            resp = self.client.post('/api/guideline_windows', json={'rowids': rowids})
            try:
                body = resp.get_json()
            except Exception:
                body = None
        return resp, (body or {})

    def one_row(self, rowid):
        _, d = self.load([rowid])
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
        # Phase 2.1: the cache Save wrote is no longer allowed to answer for the
        # window while the sheet can be read — it holds Tsup but no Q/P/COP on a
        # BAM row, and a partial cache used to pass as complete.
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

        resp, d = h.load([])
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
        check('the loaded row is in the file', first.startswith('1,'), first)


# ---------------------------------------------------------------------------
# Phase 2.1: the evaluation four are the same analysis-time quantities as the
# entry, computed on the evaluation window. A BAM sheet carries neither
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
        check('the BAM pump correction was applied, so Q and P are not the uncorrected means',
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
        bam_q = analysis_four(_uncorr_sheet, 4800.0, 9000.0)[1]
        check('the BAM correction is not used when the lab delivered corrected Q and P',
              abs(row.get('eval_q') - bam_q) > 1e-6, f"{row.get('eval_q')} vs BAM {bam_q}")


def test_eval_means_ignore_a_tsup_only_cache():
    with WindowsHarness(sheet=_uncorr_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        h.add_period(1, 'defrost', 0.0, 1200.0)
        h.add_period(1, 'heating', 1200.0, 12000.0)
        h.add_period(1, 'equilibrium', 1200.0, 4800.0)
        # What Save writes for a BAM defrost row: Tsup cached, Q/P/COP empty.
        h.add_period(1, 'evaluation', 4800.0, 9000.0, cache={'tsup': 99.0})
        row = h.one_row(1)
        check('a Tsup-only cache no longer counts as done',
              all(row.get(k) is not None for k in ('eval_q', 'eval_p', 'eval_cop')),
              str([row.get('eval_q'), row.get('eval_p'), row.get('eval_cop')]))
        check('and Tsup is recomputed with the entry\'s own quantities',
              abs(row.get('eval_tsup') - sheet_mean_of(_uncorr_sheet, 'T_supply', 4800.0, 9000.0)) < 1e-9,
              str(row.get('eval_tsup')))


def test_eval_means_when_the_sheet_cannot_be_read():
    with WindowsHarness(sheet=_unreadable_sheet) as h:
        h.add_entry(1, 0.0, 12000.0, tsup=40.0, q=6.0, p=2.5, cop=2.4)
        h.add_period(1, 'defrost', 0.0, 1200.0)
        h.add_period(1, 'heating', 1200.0, 12000.0)
        h.add_period(1, 'evaluation', 4800.0, 9000.0, cache={'tsup': 34.5})
        row = h.one_row(1)
        check('the cached Tsup is still shown', row.get('eval_tsup') == 34.5, str(row.get('eval_tsup')))
        check('Q, P and COP stay empty rather than 0',
              all(row.get(k) is None for k in ('eval_q', 'eval_p', 'eval_cop')),
              str([row.get('eval_q'), row.get('eval_p'), row.get('eval_cop')]))
        check('the note says the sheet could not be read',
              'could not be read' in (row.get('note') or ''), str(row.get('note')))
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
        check('no sheet is read when there is no evaluation window', h.sheet_reads == [],
              str(h.sheet_reads))


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
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print('All guideline-clock checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
