"""Checks for the Cycle Extract guideline clocks in flask_app/app.py.

Run from the repository root:  python scripts/test_guideline_clocks.py

Synthetic numbers only — no lab data, no database. Covers the locked rules from
docs/CYCLE_PERIODS_AND_GUIDELINE_WINDOWS.md: D/S is the transition plus the
buffer, H is the remainder, equilibrium is the *first* eq_min of H and
evaluation the eval_min after it (never the last minutes of the cycle),
evaluation is dropped when H is too short, and eq/eval exist for defrost only.
A window cut defrost end → next defrost end carries two D spans with H in the
middle (§2.1.1); one cut defrost start → next defrost start keeps one.

Phase 1.2 adds the ``continuous`` kind (no D/S, H is the parent window, eq/eval
from its start) and the kind fill order of §2.2.1, including the file-level
"treat unknown rows as" default and the rule that ``cycle_type='other'`` is
unknown, never continuous.

Phase 1.2b adds API-level checks (``test_api_*``): the file-level default really
reaches the proposal, Save all takes the whole file when nothing is ticked, and
every ``/api/`` reply is JSON even when the request fails. Those run against a
throwaway SQLite file and a stubbed sheet — never the analyst's database, never
a lab workbook.
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
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print('All guideline-clock checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
