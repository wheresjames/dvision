"""Mission profiles: load, validate, digest, override, save.

A profile is the whole of what an operator chose -- mission and parameters,
budgets, the finish action, the area or waypoints -- in one JSON file, schema
``dvision2.mission-profile.v1``. The file is the source of truth: the Setup tab
edits a copy, marks it dirty, and saves it back; a run records the digest of
exactly what it flew.

Where profiles live (Q3): committed examples in ``assets/mission_profiles``;
operator saves default to ``~/.config/dvision2/missions`` (``DVISION2_MISSION_DIR``
overrides it), so flying does not dirty the repository. A bare name is looked
up in the user directory first, then the assets.

Everything here is bounded as the mission status snapshot is (B8): at most 256
area vertices and 256 waypoints, free text at most 120 characters.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from dcmn.mission import MAX_VERTICES, MAX_WAYPOINTS
from dmsn.schema import Field, ParamSchema

SCHEMA = 'dvision2.mission-profile.v1'
ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / 'assets' / 'mission_profiles'
#: What the profile itself ends with: hold, fly home, or land.
FINISH_ACTIONS = ('hold', 'home', 'land')
BUDGET_ACTIONS = ('hold', 'home', 'land')

GENERAL = ParamSchema((
    Field('finish_action', 'choice', default='hold', choices=FINISH_ACTIONS,
          help='what happens when the mission completes: hold where it is, fly home, or land'),
    Field('coverage_margin_m', 'float', default=10.0, minimum=0.0, maximum=100.0, unit='m',
          help='added around the area or workspace when dmsn asks dalg for coverage'),
    Field('workspace', 'bounds', optional=True, unit='m',
          help='xmin,ymin,xmax,ymax to map before takeoff; leave empty to keep dalg\'s own grid'),
    Field('arena', 'text', default='', help='the map this profile was made for, e.g. maze_020; '
                                            'a different running map is warned about'),
    Field('strategy', 'text', default='', help='goal strategy for area missions'),
))
BUDGETS = ParamSchema((
    Field('time_s', 'float', optional=True, minimum=0.0, unit='s', help='sim seconds from Start'),
    Field('time_action', 'choice', default='hold', choices=BUDGET_ACTIONS),
    Field('battery_reserve_pct', 'float', optional=True, minimum=0.0, maximum=100.0, unit='%',
          help='end the mission when the battery falls below this'),
    Field('battery_action', 'choice', default='home', choices=BUDGET_ACTIONS),
    Field('path_m', 'float', optional=True, minimum=0.0, unit='m', help='flown track length from Start'),
    Field('path_action', 'choice', default='hold', choices=BUDGET_ACTIONS),
))
DEBOUNCE = ParamSchema((
    Field('min_commit_s', 'float', default=3.0, minimum=0.0, maximum=600.0, unit='s',
          help='a strategy goal is kept at least this long before it may change'),
    Field('min_shift_m', 'float', default=1.0, minimum=0.0, maximum=100.0, unit='m',
          help='a strategy goal closer than this to the current one is not sent'),
))
ASSISTED = ParamSchema((
    Field('reference_background', 'bool', optional=True,
          help='open with the reference image shown; empty uses the mission default'),
    Field('tour_import', 'bool', default=False, help='waypoints were converted from a map-bound tour'),
))
GEOMETRY = ParamSchema((
    Field('area', 'polygon', default=[], unit='m', help='the area polygon, x,y;x,y;...'),
    Field('waypoints', 'points', default=[], unit='m', help='waypoints, x,y;x,y;...'),
))
SECTIONS = {'budgets': BUDGETS, 'debounce': DEBOUNCE, 'assisted': ASSISTED}
TOP_LEVEL = ('schema', 'name', 'mission', 'params', 'strategy_params', 'imported_from', *GENERAL.by_name,
             *SECTIONS, *GEOMETRY.by_name)


def user_dir() -> Path:
    override = os.environ.get('DVISION2_MISSION_DIR')
    return Path(override).expanduser() if override else Path.home() / '.config' / 'dvision2' / 'missions'


def resolve(name_or_path: str | Path) -> Path:
    """A path, or a bare name found in the user directory, then the assets."""
    text = str(name_or_path)
    candidate = Path(text).expanduser()
    if candidate.exists() or candidate.suffix or os.sep in text: return candidate
    for directory in (user_dir(), ASSET_DIR):
        path = directory / f'{text}.json'
        if path.exists(): return path
    return ASSET_DIR / f'{text}.json'


def available() -> list[tuple[str, Path]]:
    """Every profile on disk, labelled ``user`` or ``example``."""
    rows = []
    for label, directory in (('user', user_dir()), ('example', ASSET_DIR)):
        if directory.is_dir():
            rows.extend((label, path) for path in sorted(directory.glob('*.json')))
    return rows


def _mission_schema(mission: str) -> ParamSchema:
    from dmsn.missions import schema_for
    return schema_for(mission)


def default_strategy(mission: str) -> str:
    """The goal strategy a new profile for this mission starts with ('' when it takes none)."""
    from dmsn.missions import MISSIONS
    return str(getattr(MISSIONS.get(mission), 'default_strategy', ''))


def strategy_defaults(strategy: str) -> dict[str, Any]:
    if not strategy: return {}
    from dmsn.strategies import STRATEGIES
    if strategy not in STRATEGIES: raise ValueError(f'unknown strategy {strategy!r}; known: {", ".join(sorted(STRATEGIES))}')
    return STRATEGIES[strategy].schema.defaults()


def blank(mission: str = 'goto', name: str = 'untitled') -> dict[str, Any]:
    """A new profile with every default filled in."""
    value = dict(schema=SCHEMA, name=name, mission=mission, params=_mission_schema(mission).defaults(),
                 imported_from=None, **GENERAL.defaults(),
                 **{section: schema.defaults() for section, schema in SECTIONS.items()}, **GEOMETRY.defaults())
    value['strategy'] = default_strategy(mission)
    value['strategy_params'] = strategy_defaults(value['strategy'])
    return validate(value)


def validate(value: Any) -> dict[str, Any]:
    """The profile in canonical form, or ``ValueError`` saying what is wrong."""
    if not isinstance(value, dict): raise ValueError('a mission profile must be a JSON object')
    if value.get('schema', SCHEMA) != SCHEMA:
        raise ValueError(f'unsupported mission profile schema {value.get("schema")!r}; expected {SCHEMA}')
    unknown = set(value) - set(TOP_LEVEL)
    if unknown: raise ValueError(f'unknown profile keys: {", ".join(sorted(unknown))}')
    name = value.get('name', 'untitled')
    if not isinstance(name, str) or not name or len(name) > 120: raise ValueError('profile name must be 1-120 characters')
    mission = value.get('mission')
    if not isinstance(mission, str): raise ValueError('profile needs a mission')
    result = dict(schema=SCHEMA, name=name, mission=mission)
    try:
        result['params'] = _mission_schema(mission).validate(value.get('params'))
    except ValueError as exc:
        raise ValueError(f'params: {exc}') from None
    strategy_params = value.get('strategy_params') or {}
    if not isinstance(strategy_params, dict): raise ValueError('strategy_params must be an object')
    general = GENERAL.validate({k: value[k] for k in GENERAL.by_name if k in value})
    if general['strategy']:
        from dmsn.strategies import STRATEGIES
        if general['strategy'] not in STRATEGIES: raise ValueError(f'unknown strategy {general["strategy"]!r}')
        strategy_params = STRATEGIES[general['strategy']].schema.validate(strategy_params)
    result.update(general)
    result['strategy_params'] = strategy_params
    for section, schema in SECTIONS.items():
        try: result[section] = schema.validate(value.get(section))
        except ValueError as exc: raise ValueError(f'{section}: {exc}') from None
    for key, limit in (('area', MAX_VERTICES), ('waypoints', MAX_WAYPOINTS)):
        items = value.get(key) or []
        if isinstance(items, list) and len(items) > limit:
            raise ValueError(f'{key} has {len(items)} entries; at most {limit} are allowed')
    result.update(GEOMETRY.validate({k: value.get(k) or [] for k in GEOMETRY.by_name}))
    imported = value.get('imported_from')
    if imported is not None and not isinstance(imported, dict): raise ValueError('imported_from must be an object')
    result['imported_from'] = imported
    return result


def digest(profile: dict[str, Any]) -> str:
    """The identity of what a profile flies: everything but its name."""
    content = {k: v for k, v in validate(profile).items() if k != 'name'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(name_or_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = resolve(name_or_path)
    try: raw = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError: raise ValueError(f'no mission profile {str(name_or_path)!r} (looked for {path})') from None
    except json.JSONDecodeError as exc: raise ValueError(f'{path}: not JSON: {exc}') from None
    if isinstance(raw, dict): raw.setdefault('name', path.stem)
    try: return validate(raw), path
    except ValueError as exc: raise ValueError(f'{path}: {exc}') from None


def save(profile: dict[str, Any], path: str | Path) -> Path:
    path = Path(path).expanduser()
    value = validate(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    return path


def _schema_for_key(profile: dict[str, Any], key: str) -> tuple[str | None, str, Field]:
    """``(section, field, Field)`` for a dotted or bare override key."""
    section, dot, name = key.partition('.')
    if dot:
        if section == 'params': return 'params', name, _mission_schema(profile['mission']).field(name)
        if section in SECTIONS: return section, name, SECTIONS[section].field(name)
        if section == 'strategy_params':
            from dmsn.strategies import STRATEGIES
            if profile.get('strategy') not in STRATEGIES: raise ValueError('no strategy selected for strategy_params')
            return section, name, STRATEGIES[profile['strategy']].schema.field(name)
        raise ValueError(f'unknown profile section {section!r}')
    if key in GENERAL: return None, key, GENERAL.field(key)
    if key in GEOMETRY: return None, key, GEOMETRY.field(key)
    params = _mission_schema(profile['mission'])
    if key in params: return 'params', key, params.field(key)
    raise ValueError(f'unknown profile key {key!r}')


def set_value(profile: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """A copy with one field replaced and the whole profile revalidated."""
    result = copy.deepcopy(profile)
    if key == 'mission':
        result['mission'] = str(value)
        result['params'] = _mission_schema(result['mission']).defaults()
        result['strategy'] = default_strategy(result['mission'])
        result['strategy_params'] = strategy_defaults(result['strategy'])
        return validate(result)
    if key == 'strategy':
        # A different strategy has different parameters: start from its defaults.
        strategy = GENERAL.field('strategy').check(value)
        if strategy != result.get('strategy'):
            result['strategy'], result['strategy_params'] = strategy, strategy_defaults(strategy)
        return validate(result)
    if key == 'name':
        result['name'] = str(value)
        return validate(result)
    section, name, field = _schema_for_key(result, key)
    checked = field.check(value)
    if section is None: result[name] = checked
    else: result.setdefault(section, {})[name] = checked
    return validate(result)


def apply_overrides(profile: dict[str, Any], items: Iterable[str]) -> dict[str, Any]:
    """``--set KEY=VALUE`` in order: ``goal=52.4,2.4``, ``budgets.time_s=600``, ``finish_action=home``."""
    for item in items or ():
        key, sep, raw = str(item).partition('=')
        if not sep or not key.strip(): raise ValueError(f'--set needs KEY=VALUE, not {item!r}')
        try: profile = set_value(profile, key.strip(), raw.strip())
        except ValueError as exc: raise ValueError(f'--set {key.strip()}: {exc}') from None
    return profile


class ProfileDocument:
    """The profile an operator is editing: current value, where it came from, and whether it changed."""

    def __init__(self, profile: dict[str, Any] | None = None, path: Path | None = None) -> None:
        self.value = validate(profile) if profile is not None else blank()
        self.path = path
        self._saved = digest(self.value)
        self.error = ''
        self.revision = 0

    @property
    def digest(self) -> str:
        return digest(self.value)

    @property
    def dirty(self) -> bool:
        return self.digest != self._saved

    def _replace(self, value: dict[str, Any]) -> None:
        self.value, self.error = value, ''
        self.revision += 1

    def set(self, key: str, value: Any) -> bool:
        """Change one field; ``False`` with ``error`` set when the value is refused."""
        try: self._replace(set_value(self.value, key, value))
        except ValueError as exc:
            self.error = str(exc)
            return False
        return True

    def new(self, mission: str = 'goto') -> None:
        self._replace(blank(mission))
        self.path, self._saved = None, None

    def open(self, name_or_path: str | Path) -> None:
        value, path = load(name_or_path)
        self._replace(value)
        self.path, self._saved = path, digest(value)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None: target = user_dir() / f'{self.value["name"]}.json'
        if path is not None: self.value = set_value(self.value, 'name', Path(path).stem)
        saved = save(self.value, target)
        self.path, self._saved = saved, self.digest
        self.revision += 1
        return saved

    def revert(self) -> None:
        if self.path is None: return self.new(self.value['mission'])
        self.open(self.path)
