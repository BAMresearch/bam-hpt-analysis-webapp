"""Unit / condition / profile configuration loaded from flask_app/config/.

Source of truth for guideline setpoints and unit parameters (B1).
Private BAM profiles live under config/profiles/local/ (gitignored).
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

CLIMATES = ('Average', 'Warmer', 'Colder')
APPLICATIONS = ('LT', 'MT', 'HT')
CONDITION_LETTERS = ('E', 'A', 'F', 'B', 'C', 'D', 'G')

_CLIMATE_ALIASES = {
    'average': 'Average',
    'avg': 'Average',
    'warmer': 'Warmer',
    'colder': 'Colder',
}
_APPLICATION_ALIASES = {
    'lt': 'LT',
    'low': 'LT',
    'low-temperature': 'LT',
    'mt': 'MT',
    'medium': 'MT',
    'medium-temperature': 'MT',
    'ht': 'HT',
    'high': 'HT',
    'high-temperature': 'HT',
}

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(_APP_DIR, 'config')

_condition_sets_cache: Optional[Dict[str, Any]] = None
_profiles_cache: Optional[Dict[str, Dict[str, Any]]] = None
_profiles_cache_fingerprint: Optional[Tuple[Tuple[str, float], ...]] = None
_unit_types_cache: Optional[Dict[str, Any]] = None
_checks_cache: Optional[Dict[str, Any]] = None


def config_dir() -> str:
    return _CONFIG_DIR


def _read_json(path: str, default: Any = None) -> Any:
    if not os.path.isfile(path):
        return default
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write('\n')


def reload_all() -> None:
    global _condition_sets_cache, _profiles_cache, _profiles_cache_fingerprint
    global _unit_types_cache, _checks_cache
    _condition_sets_cache = None
    _profiles_cache = None
    _profiles_cache_fingerprint = None
    _unit_types_cache = None
    _checks_cache = None


# --- Condition sets ---------------------------------------------------------

def condition_sets_path() -> str:
    return os.path.join(_CONFIG_DIR, 'condition_sets.json')


def load_condition_sets_document() -> Dict[str, Any]:
    global _condition_sets_cache
    if _condition_sets_cache is None:
        data = _read_json(condition_sets_path(), default={})
        if not isinstance(data, dict):
            data = {}
        _condition_sets_cache = data
    return _condition_sets_cache


def save_condition_sets_document(data: Dict[str, Any]) -> None:
    _write_json(condition_sets_path(), data)
    reload_all()


def _resolve_condition_set_id(raw_id: str) -> str:
    """Resolve legacy condition-set IDs via the aliases stored in each set."""
    doc = load_condition_sets_document()
    sets = doc.get('condition_sets') or {}
    if raw_id in sets:
        return raw_id
    for cid, cs in sets.items():
        for alias in (cs.get('aliases') or []):
            if str(alias) == raw_id:
                return cid
    return raw_id


def get_default_condition_set_id() -> str:
    doc = load_condition_sets_document()
    raw = str(doc.get('default_condition_set_id') or 'ecodesign_heating')
    return _resolve_condition_set_id(raw)


def get_condition_set(condition_set_id: Optional[str] = None) -> Dict[str, Any]:
    doc = load_condition_sets_document()
    sets = doc.get('condition_sets') or {}
    cid = condition_set_id or get_default_condition_set_id()
    cid = _resolve_condition_set_id(cid)
    cs = sets.get(cid)
    if not cs:
        if sets:
            cid = next(iter(sets))
            cs = sets[cid]
        else:
            return {'id': cid, 'conditions': {}}
    out = deepcopy(cs)
    out['id'] = cid
    return out


def normalize_climate(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return _CLIMATE_ALIASES.get(s.lower()) or (s if s in CLIMATES else None)


def normalize_application(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return _APPLICATION_ALIASES.get(s.lower()) or (s.upper() if s.upper() in APPLICATIONS else None)


def condition_letter(test_condition: Optional[str]) -> Optional[str]:
    """Extract A–G from 'A', 'A real BUH', 'C70min', etc."""
    if test_condition is None:
        return None
    s = str(test_condition).strip()
    if not s:
        return None
    first = s[0].upper()
    if first in CONDITION_LETTERS:
        return first
    m = re.search(r'(?:^|[\s_\-])([A-Ga-g])(?:[\s_\-]|$)', s)
    return m.group(1).upper() if m else None


def get_default_climate(condition_set_id: Optional[str] = None) -> str:
    doc = load_condition_sets_document()
    cs = get_condition_set(condition_set_id)
    defaults = cs.get('defaults') or {}
    return (
        normalize_climate(defaults.get('climate'))
        or normalize_climate(doc.get('default_climate'))
        or 'Average'
    )


def get_default_application(condition_set_id: Optional[str] = None) -> str:
    doc = load_condition_sets_document()
    cs = get_condition_set(condition_set_id)
    defaults = cs.get('defaults') or {}
    return (
        normalize_application(defaults.get('application'))
        or normalize_application(doc.get('default_application'))
        or 'MT'
    )


def conditions_are_nested(conditions: Any) -> bool:
    """True if conditions is climate → application → letter → {tdb_set, ...}."""
    if not isinstance(conditions, dict) or not conditions:
        return False
    first = next(iter(conditions.values()))
    if not isinstance(first, dict):
        return False
    if 'tdb_set' in first or 'tsup_set' in first or 'tmean_set' in first:
        return False
    return True


def get_condition_map(condition_set_id: Optional[str] = None,
                      climate: Optional[str] = None,
                      application: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Letter → {tdb_set, tsup_set, tmean_set} for one climate × application slice.

    Flat legacy maps are returned as-is. Nested maps default to Average / MT.
    """
    cs = get_condition_set(condition_set_id)
    conditions = cs.get('conditions') or {}
    if not isinstance(conditions, dict):
        return {}
    if not conditions_are_nested(conditions):
        return deepcopy(conditions)
    clim = normalize_climate(climate) or get_default_climate(condition_set_id)
    app = normalize_application(application) or get_default_application(condition_set_id)
    slice_ = (conditions.get(clim) or {}).get(app) or {}
    return deepcopy(slice_) if isinstance(slice_, dict) else {}


def get_tdb_setpoints(condition_set_id: Optional[str] = None,
                      climate: Optional[str] = None,
                      application: Optional[str] = None) -> Dict[str, float]:
    """Outdoor dry-bulb by letter.

    With climate/application: that slice. Otherwise union of all nested slices
    (tdb is the same for a given letter) so G/E/F remain available for bands.
    """
    cs = get_condition_set(condition_set_id)
    conditions = cs.get('conditions') or {}
    if climate is not None or application is not None:
        cmap = get_condition_map(condition_set_id, climate=climate, application=application)
        return {
            letter: float(vals['tdb_set'])
            for letter, vals in cmap.items()
            if isinstance(vals, dict) and vals.get('tdb_set') is not None
        }
    if not conditions_are_nested(conditions):
        return {
            letter: float(vals['tdb_set'])
            for letter, vals in conditions.items()
            if isinstance(vals, dict) and vals.get('tdb_set') is not None
        }
    out: Dict[str, float] = {}
    # Prefer default slice, then fill letters from other climates (e.g. G).
    for clim in (get_default_climate(condition_set_id),) + tuple(
        c for c in CLIMATES if c != get_default_climate(condition_set_id)
    ):
        for app in APPLICATIONS:
            slice_ = (conditions.get(clim) or {}).get(app) or {}
            if not isinstance(slice_, dict):
                continue
            for letter, vals in slice_.items():
                if letter in out or not isinstance(vals, dict) or vals.get('tdb_set') is None:
                    continue
                out[letter] = float(vals['tdb_set'])
    return out


def get_tsup_setpoints(condition_set_id: Optional[str] = None,
                       climate: Optional[str] = None,
                       application: Optional[str] = None) -> Dict[str, float]:
    cmap = get_condition_map(condition_set_id, climate=climate, application=application)
    return {
        letter: float(vals['tsup_set'])
        for letter, vals in cmap.items()
        if isinstance(vals, dict) and vals.get('tsup_set') is not None
    }


def get_tmean_setpoints(condition_set_id: Optional[str] = None,
                        climate: Optional[str] = None,
                        application: Optional[str] = None) -> Dict[str, float]:
    cmap = get_condition_map(condition_set_id, climate=climate, application=application)
    return {
        letter: float(vals['tmean_set'])
        for letter, vals in cmap.items()
        if isinstance(vals, dict) and vals.get('tmean_set') is not None
    }


def get_tdb_setpoint_for(test_condition: Optional[str],
                         condition_set_id: Optional[str] = None,
                         climate: Optional[str] = None,
                         application: Optional[str] = None) -> Optional[float]:
    mapping = get_tdb_setpoints(condition_set_id, climate=climate, application=application)
    letter = condition_letter(test_condition)
    if letter and letter in mapping:
        return mapping[letter]
    return None


def get_tsup_setpoint_for(test_condition: Optional[str],
                          condition_set_id: Optional[str] = None,
                          file_name: Optional[str] = None,
                          climate: Optional[str] = None,
                          application: Optional[str] = None) -> Optional[float]:
    """Leaving-water setpoint for a condition letter. file_name kept for API compatibility."""
    mapping = get_tsup_setpoints(condition_set_id, climate=climate, application=application)
    letter = condition_letter(test_condition)
    if letter and letter in mapping:
        return mapping[letter]
    return None


def get_tmean_setpoint_for(test_condition: Optional[str],
                           condition_set_id: Optional[str] = None,
                           climate: Optional[str] = None,
                           application: Optional[str] = None) -> Optional[float]:
    """Table 3 mean-water setpoint (variable-flow target) for a condition letter."""
    mapping = get_tmean_setpoints(condition_set_id, climate=climate, application=application)
    letter = condition_letter(test_condition)
    if letter and letter in mapping:
        return mapping[letter]
    return None


def resolve_climate_application(climate: Optional[str] = None,
                                application: Optional[str] = None,
                                profile: Optional[Dict[str, Any]] = None,
                                condition_set_id: Optional[str] = None) -> Tuple[str, str]:
    """Entry value, else profile default, else condition-set default (Average / MT)."""
    clim = normalize_climate(climate)
    app = normalize_application(application)
    if profile:
        clim = clim or normalize_climate(profile.get('default_climate'))
        app = app or normalize_application(profile.get('default_application'))
    clim = clim or get_default_climate(condition_set_id)
    app = app or get_default_application(condition_set_id)
    return clim, app


# --- Unit types / checks ----------------------------------------------------

def load_unit_types() -> Dict[str, Any]:
    global _unit_types_cache
    if _unit_types_cache is None:
        _unit_types_cache = _read_json(os.path.join(_CONFIG_DIR, 'unit_types.json'), default={}) or {}
    return _unit_types_cache


def load_checks() -> Dict[str, Any]:
    global _checks_cache
    if _checks_cache is None:
        _checks_cache = _read_json(os.path.join(_CONFIG_DIR, 'checks.json'), default={}) or {}
    return _checks_cache


# --- Profiles ---------------------------------------------------------------

def _profile_dirs() -> List[Tuple[str, str]]:
    """Return (source_label, directory) pairs in load order (later overrides earlier)."""
    return [
        ('examples', os.path.join(_CONFIG_DIR, 'profiles', 'examples')),
        ('shared', os.path.join(_CONFIG_DIR, 'profiles', 'shared')),
        ('local', os.path.join(_CONFIG_DIR, 'profiles', 'local')),
    ]


def _profiles_on_disk_fingerprint() -> Tuple[Tuple[str, float], ...]:
    """Detect added/renamed/edited JSON files without restarting Flask."""
    parts: List[Tuple[str, float]] = []
    for _, directory in _profile_dirs():
        if not os.path.isdir(directory):
            continue
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            if not name.lower().endswith('.json'):
                continue
            path = os.path.join(directory, name)
            try:
                parts.append((path, os.path.getmtime(path)))
            except OSError:
                continue
    return tuple(sorted(parts))


def load_all_profiles() -> Dict[str, Dict[str, Any]]:
    """profile_id -> profile dict (local overrides shared/examples)."""
    global _profiles_cache, _profiles_cache_fingerprint
    fingerprint = _profiles_on_disk_fingerprint()
    if _profiles_cache is not None and _profiles_cache_fingerprint == fingerprint:
        return _profiles_cache
    profiles: Dict[str, Dict[str, Any]] = {}
    for source, directory in _profile_dirs():
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.lower().endswith('.json'):
                continue
            path = os.path.join(directory, name)
            try:
                data = _read_json(path, default=None)
            except Exception as exc:
                warnings.warn(f'Could not load profile {path}: {exc}')
                continue
            if not isinstance(data, dict):
                continue
            pid = data.get('profile_id') or os.path.splitext(name)[0]
            data = deepcopy(data)
            data['profile_id'] = pid
            data['_source'] = source
            data['_path'] = path
            profiles[pid] = data
    _profiles_cache = profiles
    _profiles_cache_fingerprint = fingerprint
    return profiles


def list_profiles() -> List[Dict[str, Any]]:
    return sorted(load_all_profiles().values(), key=lambda p: p.get('profile_id') or '')


def get_profile(profile_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not profile_id:
        return None
    return load_all_profiles().get(str(profile_id))


def profile_json_path(profile_id: str, destination: str = 'local') -> str:
    dest_map = {
        'local': os.path.join(_CONFIG_DIR, 'profiles', 'local'),
        'shared': os.path.join(_CONFIG_DIR, 'profiles', 'shared'),
        'examples': os.path.join(_CONFIG_DIR, 'profiles', 'examples'),
    }
    dest = destination if destination in dest_map else 'local'
    return os.path.join(dest_map[dest], f'{profile_id}.json')


def save_profile(profile: Dict[str, Any], destination: str = 'local') -> str:
    """Write a profile JSON. destination: local | shared | examples."""
    pid = profile.get('profile_id')
    if not pid:
        raise ValueError('profile_id is required')
    path = profile_json_path(str(pid), destination)
    to_save = {
        k: v for k, v in profile.items()
        if not str(k).startswith('_')
        and k not in (
            'destination', 'original_profile_id', 'original_destination',
            'update_entries', 'entry_count',
        )
    }
    existing: Dict[str, Any] = {}
    if os.path.isfile(path):
        loaded = _read_json(path, default={})
        if isinstance(loaded, dict):
            existing = loaded
    existing.update(to_save)
    _write_json(path, existing)
    reload_all()
    return path


def delete_profile(profile_id: str, destination: str = 'local') -> bool:
    path = profile_json_path(profile_id, destination)
    if os.path.isfile(path):
        os.remove(path)
        reload_all()
        return True
    return False


def _pattern_match(filename: str, pattern: str) -> bool:
    base = os.path.basename(filename or '')
    return fnmatch.fnmatch(base, pattern)


def resolve_profile(file_name: Optional[str] = None,
                    hp_id: Any = None,
                    profile_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """Resolve a unit profile.

    Returns (profile_or_None, reason) where reason is how it was resolved.
    """
    profiles = load_all_profiles()
    if profile_id:
        p = profiles.get(str(profile_id))
        if p:
            return p, 'explicit_profile_id'
        return None, f'explicit_profile_id_missing:{profile_id}'

    base = os.path.basename(file_name or '')
    # Prefer longest / most specific pattern match
    matches: List[Tuple[int, Dict[str, Any]]] = []
    for p in profiles.values():
        for pat in p.get('filename_patterns') or []:
            if fnmatch.fnmatch(base, pat):
                matches.append((len(str(pat)), p))
                break
    if matches:
        matches.sort(key=lambda t: t[0], reverse=True)
        return matches[0][1], 'filename_pattern'

    if hp_id is not None and str(hp_id).strip() != '':
        key = str(hp_id).strip()
        key_norm = key.lower()
        for p in profiles.values():
            aliases = [str(a).strip().lower() for a in (p.get('hp_id_aliases') or [])]
            if key_norm in aliases:
                return p, 'hp_id_alias'
            # Also match trailing digits of profile_id e.g. BAM_HP1 → 1
            pid = str(p.get('profile_id') or '')
            if pid.upper().endswith('HP' + key.upper()) or pid.upper().endswith('_' + key.upper()):
                return p, 'hp_id_suffix'

    return None, 'unresolved'


def resolve_buh_cap(file_name: Optional[str],
                    hp_id: Any = None,
                    profile_id: Optional[str] = None,
                    fallback_cap_values: Optional[Dict[str, float]] = None,
                    default_cap: Optional[float] = None) -> Tuple[Optional[float], str]:
    """Return (cap_kw, status). status: profile | legacy_prefix | default_warned | missing."""
    profile, reason = resolve_profile(file_name=file_name, hp_id=hp_id, profile_id=profile_id)
    if profile is not None and profile.get('buh_power_cap_kw') is not None:
        return float(profile['buh_power_cap_kw']), f'profile:{profile.get("profile_id")}:{reason}'

    # Legacy filename-prefix map (migration fallback)
    if fallback_cap_values and file_name:
        for prefix, value in fallback_cap_values.items():
            if str(file_name).startswith(prefix):
                msg = (f'BUH cap for {file_name} taken from legacy cap_values[{prefix!r}] '
                       f'(no profile buh_power_cap_kw); migrate to a unit profile.')
                print(f'WARNING: {msg}')
                warnings.warn(msg)
                return float(value), f'legacy_prefix:{prefix}'

    if default_cap is not None:
        msg = (f'No BUH power cap configured for file={file_name!r} hp_id={hp_id!r}; '
               f'using default_cap={default_cap}. Configure a unit profile to silence this.')
        print(f'WARNING: {msg}')
        warnings.warn(msg)
        return float(default_cap), 'default_warned'

    msg = f'No BUH power cap configured for file={file_name!r} hp_id={hp_id!r}'
    print(f'WARNING: {msg}')
    warnings.warn(msg)
    return None, 'missing'


def resolve_flow_set(file_name: Optional[str] = None,
                     hp_id: Any = None,
                     profile_id: Optional[str] = None) -> Tuple[Optional[str], Optional[float], str]:
    """Return (flow_type, flow_set_value, reason)."""
    profile, reason = resolve_profile(file_name=file_name, hp_id=hp_id, profile_id=profile_id)
    if profile is None:
        return None, None, reason
    ft = profile.get('flow_type')
    fv = profile.get('flow_set_value')
    if ft and fv is not None:
        return str(ft), float(fv), f'profile:{profile.get("profile_id")}:{reason}'
    return None, None, f'profile_incomplete:{profile.get("profile_id")}'


def resolve_pdesign(file_name: Optional[str] = None,
                    hp_id: Any = None,
                    profile_id: Optional[str] = None) -> Tuple[Optional[float], str]:
    profile, reason = resolve_profile(file_name=file_name, hp_id=hp_id, profile_id=profile_id)
    if profile is None:
        return None, reason
    if profile.get('pdesign_kw') is None:
        return None, f'profile_incomplete:{profile.get("profile_id")}'
    return float(profile['pdesign_kw']), f'profile:{profile.get("profile_id")}:{reason}'


def hp_id_from_profile(profile_id: Optional[str]) -> Optional[int]:
    """Derive hp_id from a profile's first numeric alias, or from the trailing
    digit of its profile_id (e.g. BAM_HP3 → 3, HPT_RRT1 → 1)."""
    if not profile_id:
        return None
    p = get_profile(profile_id)
    if not p:
        return None
    for alias in p.get('hp_id_aliases') or []:
        s = str(alias).strip()
        m = re.match(r'^(\d+)\s*[A-Za-z]?$', s)
        if m:
            return int(m.group(1))
    pid = str(p.get('profile_id') or '')
    m = re.search(r'(\d+)$', pid)
    if m:
        return int(m.group(1))
    return None


def suggest_profile_id_from_hp(hp_id: Any) -> Optional[str]:
    """Map legacy hp_id 1/2/3/9 → BAM_HPn when such a profile exists."""
    if hp_id is None:
        return None
    s = str(hp_id).strip()
    m = __import__('re').match(r'^(\d+)', s)
    if not m:
        return None
    candidate = f'BAM_HP{m.group(1)}'
    if candidate in load_all_profiles():
        return candidate
    p, _ = resolve_profile(hp_id=hp_id)
    return p.get('profile_id') if p else None


def normalize_hp_id(value: Any) -> str:
    """Collapse sub-unit hp ids onto their base heat pump number.

    e.g. '1a' -> '1', '1b' -> '1', '1' -> '1'. Whitespace-only values become ''.
    """
    s = '' if value is None else str(value).strip()
    if not s or s.lower() in ('none', 'nan'):
        return ''
    m = re.match(r'^(\d+)\s*[A-Za-z]?$', s)
    return m.group(1) if m else s


def _nonempty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            if float(value) == int(value):
                s = str(int(value))
            else:
                s = str(value)
        except (ValueError, TypeError, OverflowError):
            s = str(value)
    else:
        s = str(value).strip()
    if not s or s.lower() in ('none', 'nan'):
        return None
    return s


def _first_nonempty(entry: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key not in entry:
            continue
        s = _nonempty_str(entry.get(key))
        if s is not None:
            return s
    return None


def entry_hp_id(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    """Read HP_ID / hp_id from a results row; whitespace counts as missing."""
    raw = _first_nonempty(entry or {}, ['HP_ID', 'hp_id', 'hpid', 'Hp Id'])
    if raw is None:
        return None
    return normalize_hp_id(raw) or None


def entry_flow_config(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    raw = _first_nonempty(
        entry or {},
        ['flow_config', 'Flow_Config', 'Flow config', 'Flow_config'],
    )
    return raw.lower() if raw else None


def entry_profile_id(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    return _first_nonempty(entry or {}, ['profile_id'])


def _compact_unit_label(profile: Optional[Dict[str, Any]], hp_id: Optional[str], fallback: str) -> str:
    """BAM: HP{n} when hp_id is present. Otherwise a short profile name, not a long display_name."""
    if hp_id:
        return f'HP{hp_id}'
    if profile:
        name = str(profile.get('display_name') or '').strip()
        pid = str(profile.get('profile_id') or '').strip()
        if name and len(name) <= 32:
            return name
        if pid:
            return pid
        if name:
            return name
    return fallback


def unit_key_and_label(entry: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Stable grouping identity for scatter / dTreturn / period statistics.

    unit_key:
      stored profile_id
      else resolve_profile(file_name, hp_id).profile_id
      else HP{normalized hp_id}
      else file_name
      else 'unresolved'

    label: HP{n} when hp_id is present (BAM); otherwise a compact profile name
    or the file name so unresolved HPT files are not merged.
    """
    entry = entry or {}
    stored_pid = entry_profile_id(entry)
    hp = entry_hp_id(entry)
    fname = _nonempty_str(entry.get('file_name'))
    flow = entry_flow_config(entry) or ''

    profile: Optional[Dict[str, Any]] = None
    source = 'unresolved'
    key = 'unresolved'

    if stored_pid:
        profile = get_profile(stored_pid)
        key = stored_pid
        source = 'explicit_profile_id'
    else:
        profile, reason = resolve_profile(file_name=fname, hp_id=hp, profile_id=None)
        if profile and profile.get('profile_id'):
            key = str(profile.get('profile_id'))
            source = reason
        elif hp:
            key = f'HP{hp}'
            source = 'hp_id_fallback'
        elif fname:
            key = fname
            source = 'file_name'
        else:
            key = 'unresolved'
            source = 'unresolved'

    label = _compact_unit_label(profile, hp, key)
    return {
        'unit_key': key,
        'label': label,
        'source': source,
        'hp_id': hp or '',
        'flow_config': flow,
        'profile_id': stored_pid or ((profile or {}).get('profile_id') or '') or '',
        'file_name': fname or '',
        'unresolved': '1' if source in ('file_name', 'unresolved') else '',
    }


def format_unit_flow_label(label: str, flow_config: Optional[str] = None) -> str:
    flow = (flow_config or '').strip()
    if flow:
        return f'{label} ({flow})'
    return label


def disambiguate_labels(idents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """When the same hp_id maps to multiple profile_ids, switch labels to
    profile names so users can tell BAM_HP1 apart from HPT_RRT1."""
    hp_to_profiles: Dict[str, set] = {}
    for ident in idents:
        hp = ident.get('hp_id') or ''
        pid = ident.get('profile_id') or ''
        if hp and pid:
            hp_to_profiles.setdefault(hp, set()).add(pid)
    ambiguous_hps = {hp for hp, pids in hp_to_profiles.items() if len(pids) > 1}
    if not ambiguous_hps:
        return idents
    out = []
    for ident in idents:
        ident = dict(ident)
        if ident.get('hp_id') in ambiguous_hps and ident.get('profile_id'):
            profile = get_profile(ident['profile_id'])
            ident['label'] = _compact_unit_label(profile, None, ident['unit_key'])
        out.append(ident)
    return out


def list_units_from_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Unique units in display order (label, then unit_key)."""
    seen: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        ident = unit_key_and_label(entry)
        seen.setdefault(ident['unit_key'], ident)
    raw = list(seen.values())
    raw = disambiguate_labels(raw)
    return sorted(raw, key=lambda u: (str(u.get('label') or '').lower(), u['unit_key']))
