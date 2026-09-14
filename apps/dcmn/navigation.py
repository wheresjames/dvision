"""Bounded retained navigation/status records. No vehicle or display dependencies.

Lifetime flock ownership prevents a second live writer from unlinking an endpoint.
Readers reopen on every poll so provider restarts cannot strand an old handle.
The execution profile lives here because dnav and dway must load the same one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import tempfile
import uuid

from dvision2_common import load_pymembus, memkv_aligned_name_len, validate_id

SCHEMA = 'dvision2.navigation.v1'
STATUS_SCHEMA = 'dvision2.execution.v1'
PROFILE_SCHEMA = 'dvision2.execution-profile.v1'
MAX_BYTES = 65536
MAX_POINTS = 256
CONTEXT_KEYS = ('provider_id', 'frame_id', 'localization_epoch', 'clock_domain_id', 'clock_epoch')
EXECUTION_STATES = ('WAITING', 'READY', 'REJECTED', 'STOP_REQUIRED', 'CLOSED', 'TAKING_OFF', 'EXECUTING',
                    'BRAKING', 'HOLDING', 'COMPLETE', 'CANCELLED', 'FAILED')
DRY_RUN_STATES = ('WAITING', 'READY', 'REJECTED', 'STOP_REQUIRED', 'CLOSED')
CALIBRATION_ENVELOPE = ('stop_distance_m', 'stop_time_s', 'lateral_m', 'cruise_cross_track_m')
DISPOSITIONS = ('accepted', 'active', 'rejected', 'stopped', 'completed')
#: How dnav grants permission. ``evidence``: only swept cells with fresh free
#: evidence. ``plan``: trust the planned route through unknown space (a
#: simulation research assumption) and withdraw it only where observed
#: obstacles now block it.
PERMISSIONS = ('evidence', 'plan')
HEADINGS = ('fixed', 'travel')
PROFILE_DIR = Path(__file__).resolve().parents[2] / 'assets' / 'execution_profiles'
PLANNING_STATUSES = ('ok', 'no_route', 'goal_unreachable', 'start_blocked', 'stale_map', 'no_goal',
                     'stale_pose', 'frame_mismatch', 'outside_coverage', 'unavailable')


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value):
    return type(value) is int and value >= 0


def _text(value):
    return isinstance(value, str) and 0 < len(value.encode()) <= 512


def _xyz(value, sizes=(3,)):
    return isinstance(value, list) and len(value) in sizes and all(number(x) for x in value)


def _location(value):
    return isinstance(value, list) and len(value) == 2 and _integer(value[0]) and number(value[1])


@dataclass(frozen=True)
class ExecutionProfile:
    """Limits shared by dnav validation and dway admission; mismatches are rejected."""
    schema: str = PROFILE_SCHEMA
    calibrated: bool = False
    slab_assumption: str = 'none'
    note: str = ''
    speed_mps: float = 0.25
    body_radius_m: float = 0.15
    half_height_m: float = 0.15
    tracking_m: float = 0.1
    stopping_m: float = 0.25
    stopping_s: float = 1.0
    max_age_s: float = 2.0
    altitude_m: float = 1.5
    join_m: float = 0.25
    free_threshold: float = 0.35
    occupied_threshold: float = 0.5
    required_sources: tuple = ()
    # Executor limits (dway). Synthetic defaults; a calibrated profile states measured ones.
    stream_hz: float = 10.0
    reaction_s: float = 0.2
    hold_speed_mps: float = 0.05
    hold_dwell_s: float = 0.5
    hold_timeout_s: float = 5.0
    arrival_m: float = 0.3
    max_state_age_s: float = 0.5
    max_wind_mps: float = 0.0
    max_telemetry_latency_ms: float = 50.0
    calibration: dict | None = None
    permission: str = 'evidence'
    #: Plan permission only: an observed obstacle this close to the remaining
    #: route withdraws it. Independent of the planner's own inflation.
    plan_clearance_m: float = 0.3
    #: Executor heading: ``fixed`` holds the heading at Start; ``travel`` faces
    #: each segment, turning in place on the route before flying it.
    heading: str = 'fixed'
    #: With ``travel``: the largest heading error at which a segment may begin.
    turn_tolerance_deg: float = 15.0

    def __post_init__(self):
        if self.schema != PROFILE_SCHEMA: raise ValueError('unsupported execution profile')
        if type(self.calibrated) is not bool: raise ValueError('calibrated must be boolean')
        if self.permission not in PERMISSIONS: raise ValueError(f'permission must be one of {PERMISSIONS}')
        if self.heading not in HEADINGS: raise ValueError(f'heading must be one of {HEADINGS}')
        if self.slab_assumption not in ('none', 'vertical-extrusion'):
            raise ValueError('unsupported slab assumption')
        if not isinstance(self.note, str) or len(self.note.encode()) > 512: raise ValueError('invalid note')
        for key, value in asdict(self).items():
            if key in ('schema', 'calibrated', 'slab_assumption', 'note', 'required_sources', 'calibration',
                       'permission', 'heading'): continue
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid execution profile {key}')
        if not 0 <= self.free_threshold < self.occupied_threshold <= 1: raise ValueError('invalid thresholds')
        if min(self.speed_mps, self.max_age_s, self.stopping_s, self.join_m, self.stream_hz, self.hold_dwell_s,
               self.hold_timeout_s, self.arrival_m, self.max_state_age_s, self.hold_speed_mps) <= 0:
            raise ValueError('speed, age, stop time, join, stream, hold and arrival limits must be positive')
        if self.calibration is not None and not isinstance(self.calibration, dict):
            raise ValueError('calibration must be an object')
        if self.calibrated: self._check_calibration()
        if self.max_age_s <= self.stopping_s:
            raise ValueError('max_age_s must exceed stopping_s or no evidence can admit a route')
        if (not isinstance(self.required_sources, (list, tuple)) or len(self.required_sources) > 64
                or not all(isinstance(s, str) and s for s in self.required_sources)
                or len(set(self.required_sources)) != len(self.required_sources)):
            raise ValueError('invalid sources')

    def _check_calibration(self):
        """A calibrated profile must be supported by its own measured envelope; never extrapolated."""
        c = self.calibration
        if not c: raise ValueError('calibrated profile has no calibration record')
        for key in ('method', 'vehicle', 'conditions', 'envelope', 'margins'):
            if key not in c: raise ValueError(f'calibration record missing {key}')
        conditions, envelope, margins = c['conditions'], c['envelope'], c['margins']
        for key in ('speeds_mps', 'latencies_ms', 'winds_mps'):
            values = conditions.get(key)
            if not isinstance(values, list) or not values or not all(number(v) for v in values):
                raise ValueError(f'calibration conditions missing {key}')
        for key in CALIBRATION_ENVELOPE:
            if not number(envelope.get(key)): raise ValueError(f'calibration envelope missing {key}')
        for key in ('stop_m', 'tracking_m'):
            if not number(margins.get(key)) or margins[key] < 0: raise ValueError(f'calibration margin missing {key}')
        def need(ok, what):
            if not ok: raise ValueError(f'profile outside its calibration: {what}')
        need(self.speed_mps <= max(conditions['speeds_mps']) + 1e-9, 'speed above measured speeds')
        need(self.max_wind_mps <= max(conditions['winds_mps']) + 1e-9, 'wind above measured winds')
        need(self.max_telemetry_latency_ms <= max(conditions['latencies_ms']) + 1e-9,
             'telemetry latency above measured latencies')
        need(self.stopping_m + 1e-9 >= envelope['stop_distance_m'] + self.speed_mps / self.stream_hz + margins['stop_m'],
             'stopping_m below measured stop distance + one control period + margin')
        need(self.stopping_s + 1e-9 >= envelope['stop_time_s'] + self.reaction_s, 'stopping_s below measured stop time + reaction')
        need(self.tracking_m + 1e-9 >= envelope['lateral_m'] + envelope['cruise_cross_track_m'] + margins['tracking_m'],
             'tracking_m below measured lateral error + margin')
        need(self.join_m + 1e-9 >= self.stopping_m, 'join_m cannot rejoin a vehicle stopped short by stopping_m')
        need(self.arrival_m + 1e-9 >= self.stopping_m, 'arrival_m cannot accept a vehicle stopped short by stopping_m')
        need(self.slab_assumption != 'none', 'no declared vertical assumption')

    def require_flight(self):
        """Raise unless this profile may authorize real dynamic execution."""
        if not self.calibrated:
            raise ValueError('execution profile is synthetic (calibrated: false); it cannot authorize dynamic flight')
        self._check_calibration()

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @staticmethod
    def resolve(path):
        """A profile path, or a bare name under ``assets/execution_profiles``."""
        candidate = Path(path)
        if candidate.exists() or candidate.suffix or '/' in str(path): return candidate
        return PROFILE_DIR / f'{path}.json'

    @classmethod
    def load(cls, path=None, overrides=None):
        """Load a profile; ``overrides`` (``{key: value}`` or ``['key=value', ...]``)
        replace fields before validation, so a research dial is checked like any other."""
        value = {} if path is None else json.loads(cls.resolve(path).read_text())
        if not isinstance(value, dict): raise ValueError('execution profile must be an object')
        value.update(parse_overrides(overrides))
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown: raise ValueError(f'unknown execution profile keys: {sorted(unknown)}')
        if not isinstance(value.get('required_sources', []), list): raise ValueError('required_sources must be a list')
        value['required_sources'] = tuple(value.get('required_sources', ()))
        return cls(**value)


def parse_overrides(items):
    """``['speed_mps=1.0', 'permission=plan']`` -> ``{'speed_mps': 1.0, 'permission': 'plan'}``."""
    if not items: return {}
    if isinstance(items, dict): return dict(items)
    result = {}
    for item in items:
        key, sep, raw = str(item).partition('=')
        key = key.strip()
        if not sep or not key: raise ValueError(f'profile override must be KEY=VALUE, not {item!r}')
        if key not in ExecutionProfile.__dataclass_fields__: raise ValueError(f'unknown execution profile key {key!r}')
        try: result[key] = json.loads(raw)
        except ValueError: result[key] = raw.strip()
    return result


def encode(value):
    try:
        raw = json.dumps(value, allow_nan=False, separators=(',', ':'))
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f'invalid navigation JSON: {exc}') from None
    if len(raw.encode()) > MAX_BYTES:
        raise ValueError('navigation snapshot exceeds 64 KiB')
    return raw


def _validate_status(value):
    if value.get('state') not in EXECUTION_STATES: raise ValueError('invalid execution state')
    if type(value.get('dry_run')) is not bool: raise ValueError('missing dry_run')
    if not _text(value.get('reason')): raise ValueError('missing execution reason')
    if 'owns_control' in value and type(value['owns_control']) is not bool:
        raise ValueError('invalid owns_control')
    if value.get('commanded_target') is not None and not _xyz(value['commanded_target']):
        raise ValueError('invalid commanded target')
    if value.get('progress') is not None and not _location(value['progress']):
        raise ValueError('invalid progress')
    disposition = value.get('disposition')
    if disposition is not None:
        if (not isinstance(disposition, dict) or disposition.get('value') not in DISPOSITIONS
                or not (disposition.get('geometry_revision') is None
                        or _integer(disposition['geometry_revision']))):
            raise ValueError('invalid route disposition')
    if value.get('dry_run') and (value.get('owns_control') or value.get('commanded_target') is not None
                                 or value.get('state') not in DRY_RUN_STATES):
        raise ValueError('dry-run status cannot claim control or motion')
    return value


def validate(value, schema=SCHEMA):
    if not isinstance(value, dict) or value.get('schema') != schema:
        raise ValueError('unsupported navigation schema')
    encode(value)
    for key in ('vehicle_id', 'session'):
        if not _text(value.get(key)): raise ValueError(f'invalid {key}')
    for key in ('sequence', 'stop_generation'):
        if not _integer(value.get(key)): raise ValueError(f'invalid {key}')
    if not number(value.get('time_s')): raise ValueError('invalid time_s')
    if schema == STATUS_SCHEMA: return _validate_status(value)
    if not _integer(value.get('geometry_revision')): raise ValueError('invalid geometry_revision')
    if not _text(value.get('planner')): raise ValueError('missing planner')
    if not isinstance(value.get('context'), dict): raise ValueError('missing context')
    points = value.get('points')
    if not isinstance(points, list) or len(points) > MAX_POINTS:
        raise ValueError('route exceeds 256 points or is incomplete')
    if any(not _xyz(p) for p in points): raise ValueError('invalid XYZ point')
    if value.get('planning_status') not in PLANNING_STATUSES: raise ValueError('invalid planning status')
    clearance = value.get('clearance')
    if not isinstance(clearance, dict) or type(clearance.get('eligible')) is not bool:
        raise ValueError('missing clearance')
    if not _text(clearance.get('reason')): raise ValueError('missing clearance reason')
    if not clearance['eligible']: return value
    if value['planning_status'] != 'ok' or not points: raise ValueError('eligible route missing')
    ctx = value['context']
    for key in CONTEXT_KEYS:
        if key not in ctx or ctx[key] is None: raise ValueError(f'missing context {key}')
    goal = value.get('goal')
    if not isinstance(goal, dict) or not _xyz(goal.get('position'), (2, 3)): raise ValueError('missing goal')
    if not _text(value.get('profile')): raise ValueError('missing execution profile')
    for key in ('valid_until_s', 'speed_mps', 'tracking_m', 'distance_m', 'altitude_m'):
        if not number(clearance.get(key)) or clearance[key] < 0: raise ValueError(f'invalid {key}')
    if clearance['valid_until_s'] < value['time_s']: raise ValueError('expired clearance')
    for key in ('start', 'end'):
        loc = clearance.get(key)
        if not _location(loc): raise ValueError(f'invalid interval {key}')
        if loc[0] >= max(1, len(points)-1) or not 0 <= loc[1] <= 1:
            raise ValueError(f'interval {key} outside route')
    start, end = tuple(clearance['start']), tuple(clearance['end'])
    if len(points) == 1:
        if start != (0, 0) or end != (0, 0): raise ValueError('arrival-only interval must be empty')
    elif start >= end:
        raise ValueError('empty or reversed interval')
    return value


def context_identity(snapshot):
    return {key: snapshot.get(key) for key in CONTEXT_KEYS}


def point_at(points, location):
    """The point at segment index/fraction; segment ``i`` joins points ``i`` and ``i+1``."""
    if len(points) == 1: return list(points[0])
    i, t = location
    return [a + (b-a)*t for a, b in zip(points[i], points[i+1])]


def permitted_points(value):
    """The exact permitted interval, including a stop between waypoints."""
    if not value or not value.get('clearance', {}).get('eligible'): return []
    points = value['points']
    if len(points) == 1: return points[:]
    start, end = value['clearance']['start'], value['clearance']['end']
    return [point_at(points, start)] + points[start[0]+1:end[0]+1] + [point_at(points, end)]


def distance_along(points, start, end):
    """Path length from one location to a later one; zero if ``end`` is not later."""
    if len(points) < 2 or tuple(end) <= tuple(start): return 0.
    path = [point_at(points, start)] + points[start[0]+1:end[0]+1] + [point_at(points, end)]
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def project_progress(points, xyz, previous=(0, 0.)):
    """Progress of ``xyz`` on the ordered route, and its distance from the route.

    Only the current and the next segment are considered and progress never
    moves backwards, so a route that crosses itself cannot skip ahead to the
    later pass merely because that pass is nearer.
    """
    if len(points) < 2: return [0, 0.], math.dist(xyz, points[0])
    best = None
    for i in (previous[0], previous[0]+1):
        if i >= len(points)-1: continue
        a, b = points[i], points[i+1]
        ab = [q-p for p, q in zip(a, b)]
        span = sum(c*c for c in ab)
        t = 0. if span == 0 else max(0., min(1., sum((x-p)*c for x, p, c in zip(xyz, a, ab))/span))
        if i == previous[0]: t = max(t, previous[1])
        error = math.dist(xyz, point_at(points, (i, t)))
        if best is None or error < best[1] - 1e-9: best = ([i, t], error)
    return best


class Snapshot:
    def __init__(self, vehicle, kind='navigation', planner='dnav'):
        vehicle, planner = validate_id(vehicle), validate_id(planner)
        if kind not in ('navigation', 'execution'): raise ValueError('invalid endpoint kind')
        self.schema = SCHEMA if kind == 'navigation' else STATUS_SCHEMA
        suffix = f'{kind}.{planner}' if kind == 'navigation' else kind
        self.name = f'/dvision2.{vehicle}.{suffix}'
        if len(self.name.encode()) > 254: raise ValueError('endpoint name too long')
        self.pm = load_pymembus()
        self.handle = self.lock = None
        self.last_raw = None

    def start(self):
        if self.handle is not None: return
        path = Path(tempfile.gettempdir()) / (self.name.strip('/') + '.writer.lock')
        lock = path.open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise ValueError(f'live writer already owns {self.name}') from None
        try:
            self.pm.memkv.remove(self.name)  # stale area, protected by lifetime lock
            handle = self.pm.memkv()
            if not handle.create(self.name, 1, memkv_aligned_name_len(32, MAX_BYTES), MAX_BYTES, True):
                raise RuntimeError(self.pm.last_error_message())
            if not handle.setName(0, 'snapshot'): raise RuntimeError(self.pm.last_error_message())
            self.handle, self.lock = handle, lock
        except Exception:
            lock.close()
            raise

    def write(self, value):
        validate(value, self.schema)
        self.start()
        if not self.handle.setAll({'snapshot': encode(value)}):
            raise RuntimeError(self.pm.last_error_message())

    def read(self):
        handle = self.pm.memkv()
        if not handle.open(self.name): return {}
        try: raw = handle.getAll().get('snapshot', '')
        finally: handle.close()
        self.last_raw = raw[:MAX_BYTES]
        if not raw: return {}
        if len(raw.encode()) > MAX_BYTES: raise ValueError('oversized snapshot')
        try: value = json.loads(raw)
        except (ValueError, RecursionError): raise ValueError('malformed snapshot') from None
        return validate(value, self.schema)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.pm.memkv.remove(self.name)
            self.handle = None
        if self.lock is not None:
            self.lock.close(); self.lock = None


def new_session():
    return uuid.uuid4().hex
