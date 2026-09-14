"""Provider-neutral session, pose, goal authority and mapping-reset registry.

One small JSON snapshot per instance, held in a pymembus key/value area that
the session provider (dsim, the standalone fixture provider, later a hardware
adapter) creates and keeps open for its lifetime. It is *durable latest state*
for late readers: a module that starts after a goal was set, a reset was
requested or an epoch changed reads the current value here rather than relying
on having heard an event.

Ownership rules:

* the provider owns session identity, report root, clock/time, frame and
  localization epoch, and pose; nobody else may write those fields;
* the goal belongs to exactly one authority at a time (mission coordinator or a
  dnav CLI/UI). Only that authority may replace or clear it; another writer
  must ask for an explicit handoff, which advances ``authority_epoch``;
* anyone may request a mapping reset; dalg alone resolves and publishes the
  resulting geometry on the evidence plane.

Read/modify/write transactions take a host-wide ``flock``; this registry is a
single-host contract, like the rest of the membus planes.

Frame: ``local`` -- x east, y south, z up, metres; heading is the compass
bearing in degrees, clockwise from north; altitude datum is the provider's local
ground plane ``z = 0``. Latitude/longitude never enter this frame without a
georeferenced conversion.
"""
from __future__ import annotations

if __name__ == '__main__':
    import sys as _sys
    from pathlib import Path as _Path
    for _path in (str(_Path(__file__).resolve().parents[2]),
                  str(_Path(__file__).resolve().parents[1])):
        if _path not in _sys.path: _sys.path.insert(0, _path)

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import tempfile
import time
import uuid

from dvision2_common import load_pymembus, memkv_aligned_name_len, validate_id

SCHEMA = 'dvision2.context.v1'
MAX_BYTES = 65536
FRAME_ID = 'local'
FRAME = dict(id=FRAME_ID, axes='x_east,y_south,z_up', units='m', handedness='left',
             heading='compass degrees clockwise from north', altitude_datum='provider ground z=0')
POSE_FIELDS = ('x_m', 'y_m', 'z_m', 'heading_deg', 'roll_deg', 'pitch_deg')
EPOCH_KEYS = ('frame_id', 'localization_epoch', 'clock_domain_id', 'clock_epoch')
GOAL_ROLES = ('ui', 'mission')
#: Transitions kept in the snapshot, so a late reader can see why the goal it
#: expected is gone without having been listening at the time.
HISTORY = 32
#: Default freshness for "current" pose use, in the data clock (§9).
POSE_MAX_AGE_S = .5


def pose_context(clock_domain, clock_epoch, localization_epoch, timestamp, *, valid=True,
                 kind='ideal'):
    """What a capture-associated pose carries besides its numbers.

    ``valid`` is *pose* validity. A sensor sample's own status (a scan with no
    returns, say) says nothing about whether the pose it was taken at is known.
    """
    return dict(frame_id=FRAME_ID, localization_epoch=int(localization_epoch),
                clock_domain_id=str(clock_domain), clock_epoch=int(clock_epoch),
                capture_time_s=float(timestamp), valid=bool(valid),
                uncertainty=None, uncertainty_state='unknown', estimate_kind=str(kind))


def validate_pose(pose, snapshot, *, max_age=POSE_MAX_AGE_S):
    """The pose, if it is usable against ``snapshot``'s frame and clock; else raise.

    Age is measured in the data clock (``snapshot['time_s']``), never the wall
    clock: a paused simulation does not age poses.
    """
    if not isinstance(pose, dict): raise ValueError('pose unavailable')
    if pose.get('valid') is not True: raise ValueError('pose marked invalid by its provider')
    for key in (*POSE_FIELDS, 'capture_time_s'):
        try: value = float(pose[key])
        except (KeyError, TypeError, ValueError): raise ValueError(f'pose {key} missing') from None
        if not math.isfinite(value): raise ValueError(f'nonfinite pose {key}')
    for key in EPOCH_KEYS:
        if pose.get(key) != snapshot.get(key): raise ValueError(f'pose {key} mismatch')
    age = float(snapshot['time_s']) - float(pose['capture_time_s'])
    if age < -1e-6: raise ValueError('pose is from the future')
    if age > max_age: raise ValueError(f'pose is stale: {age:.3f} s old, limit {max_age:g} s')
    return pose


def validate_goal(goal, snapshot):
    """The goal, if it is in the current frame and epochs; else raise."""
    if not isinstance(goal, dict): raise ValueError('no goal')
    for key in ('frame_id', 'localization_epoch', 'clock_epoch'):
        if goal.get(key) != snapshot.get(key):
            raise ValueError(f'goal {key} {goal.get(key)!r} does not match current {snapshot.get(key)!r}')
    position = goal.get('position')
    if not isinstance(position, list) or len(position) not in (2, 3):
        raise ValueError('goal position malformed')
    return goal


def parse_bounds(text):
    """``xmin,ymin,xmax,ymax`` in local metres, validated but not snapped."""
    parts = [piece.strip() for piece in str(text).split(',')]
    if len(parts) != 4: raise ValueError('bounds are xmin,ymin,xmax,ymax in metres')
    values = tuple(float(piece) for piece in parts)
    if not all(math.isfinite(v) for v in values): raise ValueError('bounds must be finite')
    if values[2] <= values[0] or values[3] <= values[1]: raise ValueError('bounds must have positive extent')
    return values


def _position(position):
    if position is None: return None
    position = list(position)
    if len(position) == 3 and position[2] is None: position = position[:2]
    if len(position) not in (2, 3): raise ValueError('goal must contain x,y[,z]')
    try: position = [float(v) for v in position]
    except (TypeError, ValueError): raise ValueError('goal coordinates must be numbers') from None
    if not all(math.isfinite(v) for v in position): raise ValueError('goal must contain finite x,y[,z]')
    return position


class Context:
    """One instance's context registry: provider side and reader/writer side."""

    def __init__(self, instance):
        self.instance = validate_id(instance)
        self.name = f'/dvision2.{self.instance}.context'
        self.pm = load_pymembus()
        self.value = {}
        self.owner = None
        self._handle = None
        self.lock_path = Path(tempfile.gettempdir()) / f'dvision2.{self.instance}.context.lock'

    # -- transport -----------------------------------------------------------

    @contextmanager
    def locked(self):
        with self.lock_path.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self):
        """The current snapshot, or ``{}`` while no provider has one open."""
        kv = self._handle
        if kv is None:
            kv = self.pm.memkv()
            if not kv.open(self.name): self.value = {}; return self.value
        try:
            raw = kv.getAll().get('snapshot', '')
        finally:
            if kv is not self._handle: kv.close()
        value = json.loads(raw) if raw else {}
        if value and value.get('schema') != SCHEMA:
            raise ValueError(f'unsupported context schema {value.get("schema")!r}; expected {SCHEMA}')
        self.value = value
        return value

    def _write(self, value):
        raw = json.dumps(value, allow_nan=False, separators=(',', ':'))
        if len(raw.encode()) > MAX_BYTES: raise ValueError('context exceeds size limit')
        kv = self._handle
        if kv is None:
            kv = self.pm.memkv()
            if not kv.open(self.name): raise ValueError('session provider unavailable')
        try:
            if not kv.setAll({'snapshot': raw}): raise RuntimeError(self.pm.last_error_message())
        finally:
            if kv is not self._handle: kv.close()
        self.value = value

    @staticmethod
    def _event(value, kind, **fields):
        history = value.setdefault('history', [])
        history.append(dict(kind=kind, time_s=value.get('time_s', 0.), wall=time.time(), **fields))
        del history[:-HISTORY]

    # -- provider ------------------------------------------------------------

    def start(self, report_root, *, provider=None, provider_kind='ideal-simulation',
              clock_epoch=0, localization_epoch=0, local_ned_origin=None):
        """Create the registry and own it until :meth:`close`.

        The creating handle stays open: pymembus releases an area when its last
        handle closes, and a registry that vanished the moment it was written
        is no registry at all.
        """
        if local_ned_origin is not None:
            if len(local_ned_origin) != 3 or not all(math.isfinite(float(v)) for v in local_ned_origin):
                raise ValueError('local NED origin must be finite XYZ')
        root = Path(report_root).resolve(); root.mkdir(parents=True, exist_ok=True)
        self.owner = provider or uuid.uuid4().hex
        with self.locked():
            self.pm.memkv.remove(self.name)
            kv = self.pm.memkv()
            if not kv.create(self.name, 1, memkv_aligned_name_len(32, MAX_BYTES), MAX_BYTES, True):
                raise RuntimeError(self.pm.last_error_message())
            if not kv.setName(0, 'snapshot'): raise RuntimeError(self.pm.last_error_message())
            self._handle = kv
            value = dict(schema=SCHEMA, session_id=uuid.uuid4().hex, session_started_wall=time.time(),
                provider_id=self.owner, provider_kind=provider_kind, report_root=str(root),
                frame=FRAME, frame_id=FRAME_ID, localization_epoch=int(localization_epoch),
                vehicle_transform=None if local_ned_origin is None else dict(
                    schema='dvision2.local-ned-transform.v1', origin=list(local_ned_origin),
                    frame_id=FRAME_ID, localization_epoch=int(localization_epoch)),
                clock_domain_id=self.instance, clock_epoch=int(clock_epoch), time_s=0.,
                pose=None, pose_sequence=0, updated_wall=time.monotonic(),
                authority=None, authority_epoch=0, goal=None, goal_revision=0,
                reset_revision=0, mapping_request=None, history=[])
            self._event(value, 'session.started', session_id=value['session_id'])
            self._write(value)
        return self.value

    def _owned(self):
        value = dict(self.read())
        if not self.owner or value.get('provider_id') != self.owner:
            raise ValueError('provider ownership lost')
        return value

    def publish_pose(self, body, now, *, localization_epoch=0, clock_epoch=0, valid=True,
                     kind='ideal'):
        """Publish the provider's current pose estimate at data time ``now``.

        A change of localization or clock epoch is an announced discontinuity:
        the goal expressed in the old frame is withdrawn (its authority must
        reissue it) and the reason stays in the history for late readers.
        """
        with self.locked():
            value = self._owned()
            old = (value['localization_epoch'], value['clock_epoch'])
            new = (int(localization_epoch), int(clock_epoch))
            value.update(time_s=float(now), localization_epoch=new[0], clock_epoch=new[1],
                         updated_wall=time.monotonic())
            pose = dict({k: float(body[k]) for k in POSE_FIELDS},
                        **pose_context(self.instance, new[1], new[0], now, valid=valid, kind=kind),
                        provider_id=self.owner, sequence=value.get('pose_sequence', 0)+1)
            value['pose_sequence'] = pose['sequence']; value['pose'] = pose
            if old != new:
                cause = 'localization' if old[0] != new[0] else 'clock'
                self._event(value, f'{cause}.epoch_changed', old=list(old), new=list(new))
                if value.get('goal') is not None:
                    value['goal'] = None; value['goal_revision'] += 1
                    self._event(value, 'goal.invalidated', reason=f'{cause} epoch changed',
                                revision=value['goal_revision'])
            if valid: validate_pose(pose, value, max_age=math.inf)
            self._write(value)

    def rollover(self, report_root):
        """Start a new recording session; perception and mapping epochs continue."""
        with self.locked():
            value = self._owned()
            root = Path(report_root).resolve(); root.mkdir(parents=True, exist_ok=True)
            old = value['session_id']
            value.update(session_id=uuid.uuid4().hex, report_root=str(root),
                         session_started_wall=time.time())
            self._event(value, 'session.rollover', old=old, new=value['session_id'])
            self._write(value)
        return self.value

    def alive(self, timeout_s=5.):
        """Provider liveness on the wall clock; data freshness is a separate question."""
        return bool(self.value) and time.monotonic()-self.value.get('updated_wall', 0) < timeout_s

    def close(self):
        if self.owner and self._handle is not None:
            with self.locked():
                try: current = self.read()
                except ValueError: current = {}
                self._handle.close(); self._handle = None
                if current.get('provider_id') == self.owner: self.pm.memkv.remove(self.name)
        elif self._handle is not None:
            self._handle.close(); self._handle = None
        self.owner = None
        self.value = {}

    # -- goal authority -------------------------------------------------------

    def set_goal(self, writer, position, *, role='ui', handoff=False):
        """Install, replace or (``position=None``) clear the goal.

        Returns the goal descriptor, or None when cleared. Raises when another
        authority holds the goal and ``handoff`` was not requested.
        """
        if role not in GOAL_ROLES: raise ValueError('invalid goal authority role')
        position = _position(position)
        with self.locked():
            value = dict(self.read())
            if not value: raise ValueError('waiting for session provider')
            authority = value.get('authority')
            if authority and authority['id'] != writer and not handoff:
                raise ValueError(f"goal authority belongs to {authority['role']} {authority['id'][:12]}; "
                                 'explicit handoff required')
            if not authority or authority['id'] != writer:
                value['authority_epoch'] += 1
                self._event(value, 'goal.authority', id=writer, role=role,
                            epoch=value['authority_epoch'],
                            previous=None if not authority else authority['id'])
            value['authority'] = dict(id=writer, role=role)
            value['goal_revision'] += 1
            value['goal'] = None if position is None else dict(position=position,
                frame_id=value['frame_id'], localization_epoch=value['localization_epoch'],
                clock_epoch=value['clock_epoch'], revision=value['goal_revision'],
                authority_epoch=value['authority_epoch'], authority=writer, role=role)
            self._event(value, 'goal.cleared' if position is None else 'goal.set',
                        revision=value['goal_revision'], position=position, by=writer)
            self._write(value)
        return value['goal']

    def release_authority(self, writer):
        """Give up the goal authority without clearing the goal."""
        with self.locked():
            value = dict(self.read())
            if not value or not value.get('authority') or value['authority']['id'] != writer: return False
            value['authority'] = None
            self._event(value, 'goal.authority_released', id=writer)
            self._write(value)
        return True

    # -- mapping ---------------------------------------------------------------

    def request_reset(self, bounds=None, *, by='operator'):
        """Ask dalg for an explicit mapping reset, optionally with new bounds."""
        if bounds is not None:
            bounds = list(parse_bounds(','.join(str(v) for v in bounds)))
        with self.locked():
            value = dict(self.read())
            if not value: raise ValueError('waiting for session provider')
            value['reset_revision'] += 1
            value['mapping_request'] = dict(bounds=bounds, revision=value['reset_revision'], by=by)
            self._event(value, 'mapping.reset_requested', revision=value['reset_revision'], bounds=bounds)
            self._write(value)
        return value['reset_revision']


def main(argv=None):
    """Inspect a context or act on it: show, goal, clear-goal, reset."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='inspect or act on an instance context')
    parser.add_argument('--id', required=True, help='instance id')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('show', help='print the current snapshot')
    goal = commands.add_parser('goal', help='set the goal as x,y[,z] in local metres')
    goal.add_argument('position')
    goal.add_argument('--role', choices=GOAL_ROLES, default='mission')
    goal.add_argument('--writer', default='context-cli')
    goal.add_argument('--handoff', action='store_true', help='take authority from its holder')
    clear = commands.add_parser('clear-goal', help='clear the goal')
    clear.add_argument('--role', choices=GOAL_ROLES, default='mission')
    clear.add_argument('--writer', default='context-cli')
    clear.add_argument('--handoff', action='store_true')
    reset = commands.add_parser('reset', help='request a mapping reset')
    reset.add_argument('--bounds', default=None, help='xmin,ymin,xmax,ymax in local metres')
    argv = list(sys.argv[1:] if argv is None else argv)
    # Negative coordinates are ordinary; keep argparse from reading them as options.
    argv = [f'--bounds={token}' if index and argv[index-1] == '--bounds' else token
            for index, token in enumerate(argv)]
    argv = [token for index, token in enumerate(argv)
            if not (token == '--bounds' and index+1 < len(argv) and argv[index+1].startswith('--bounds='))]
    if 'goal' in argv:
        at = argv.index('goal')
        if at+1 < len(argv) and argv[at+1].startswith('-'): argv.insert(at+1, '--')
    args = parser.parse_args(argv)
    context = Context(args.id)
    try:
        if args.command == 'show':
            value = context.read()
            if not value: print(f'no context on {context.name}', file=sys.stderr); return 1
            print(json.dumps(value, indent=2, sort_keys=True)); return 0
        if args.command == 'goal':
            position = [float(v) for v in args.position.split(',')]
            print(json.dumps(context.set_goal(args.writer, position, role=args.role,
                                              handoff=args.handoff))); return 0
        if args.command == 'clear-goal':
            context.set_goal(args.writer, None, role=args.role, handoff=args.handoff); return 0
        revision = context.request_reset(None if args.bounds is None else parse_bounds(args.bounds),
                                         by='context-cli')
        print(f'mapping reset requested: revision {revision}'); return 0
    except (ValueError, RuntimeError) as exc:
        print(f'dcmn.context: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
