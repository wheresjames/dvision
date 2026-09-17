"""The mission planes: what a mission coordinator says it is doing, and what it asks dway for.

Two retained snapshots per vehicle, carried by :class:`dcmn.navigation.Snapshot`
(the same 64 KiB cap, lifetime writer lock and reopen-on-read):

* ``/dvision2.<id>.mission`` -- **status**, schema ``dvision2.mission.v1``: the
  mission, its profile digest, phase, outcome, the goal it set, its area and
  waypoints, progress, budgets and the one blocking fact. Bounded by
  construction: polygons and waypoint lists are capped at 256 entries and
  free text at 120 characters. Frontier lists, grids and visit counts never go
  here; they belong to the coordinator's archive.
* ``/dvision2.<id>.mission_intent`` -- **intent**, schema
  ``dvision2.mission-intent.v1``: the one vehicle action the coordinator
  currently wants (``launch``, ``hold``, ``resume``, ``cancel``, ``land``,
  ``yaw_to``). A new request is a new ``revision``; dway acts on each revision
  at most once and publishes its acknowledgement and outcome in the execution
  snapshot it already writes, so a coordinator needs no second reader.

The executor stays the only lease holder and may refuse any intent. Nothing
here commands a vehicle.
"""
from __future__ import annotations

import math

STATUS_SCHEMA = 'dvision2.mission.v1'
INTENT_SCHEMA = 'dvision2.mission-intent.v1'
#: Largest area polygon and waypoint list a profile or a snapshot may carry.
MAX_VERTICES = 256
MAX_WAYPOINTS = 256
#: Tags, reasons and blocking facts are cut to this many characters.
MAX_TEXT = 120
INTENT_KINDS = ('none', 'launch', 'hold', 'resume', 'cancel', 'land', 'yaw_to')
#: How dway reports what it did with an intent revision.
ACK_STATES = ('pending', 'accepted', 'refused', 'completed', 'failed')
PHASES = ('idle', 'starting', 'running', 'paused', 'finishing', 'done')
OUTCOMES = ('', 'complete', 'failed', 'aborted', 'budget')


def clip(text, limit: int = MAX_TEXT) -> str:
    """Free text cut to the snapshot limit, marked when it was cut."""
    text = '' if text is None else str(text)
    return text if len(text) <= limit else text[:limit - 1] + '…'


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value):
    return type(value) is int and value >= 0


def _text(value, *, empty=False):
    return isinstance(value, str) and (empty or value) and len(value) <= MAX_TEXT


def _points(value, limit):
    return (isinstance(value, list) and len(value) <= limit
            and all(isinstance(p, list) and len(p) == 2 and all(_number(c) for c in p) for p in value))


def _common(value, schema):
    if not isinstance(value, dict) or value.get('schema') != schema:
        raise ValueError(f'unsupported mission schema; expected {schema}')
    for key in ('vehicle_id', 'session'):
        if not isinstance(value.get(key), str) or not value[key] or len(value[key]) > 512:
            raise ValueError(f'invalid {key}')
    if not _integer(value.get('sequence')): raise ValueError('invalid sequence')
    if not _number(value.get('time_s')): raise ValueError('invalid time_s')


def validate_intent(value):
    """An intent snapshot, or raise."""
    _common(value, INTENT_SCHEMA)
    if not _integer(value.get('revision')): raise ValueError('invalid intent revision')
    if value.get('kind') not in INTENT_KINDS: raise ValueError(f'intent kind must be one of {INTENT_KINDS}')
    if not isinstance(value.get('args', {}), dict): raise ValueError('intent args must be an object')
    if not _text(value.get('reason', ''), empty=True): raise ValueError('invalid intent reason')
    return value


def validate_status(value):
    """A mission status snapshot, or raise."""
    _common(value, STATUS_SCHEMA)
    if value.get('phase') not in PHASES: raise ValueError(f'mission phase must be one of {PHASES}')
    if value.get('outcome', '') not in OUTCOMES: raise ValueError('invalid mission outcome')
    for key in ('mission', 'profile', 'profile_digest', 'reason', 'blocking'):
        if not _text(value.get(key, ''), empty=True): raise ValueError(f'invalid {key}')
    if not _points(value.get('area', []), MAX_VERTICES): raise ValueError('area exceeds 256 vertices or is malformed')
    waypoints = value.get('waypoints', [])
    if not isinstance(waypoints, list) or len(waypoints) > MAX_WAYPOINTS:
        raise ValueError('waypoints exceed 256 or are malformed')
    for waypoint in waypoints:
        if (not isinstance(waypoint, dict) or not _points([waypoint.get('xy')], 1)
                or not _text(waypoint.get('state', ''), empty=True)):
            raise ValueError('invalid waypoint')
    goal = value.get('goal')
    if goal is not None and (not isinstance(goal, dict) or not _points([goal.get('xy')], 1)
                             or not _text(goal.get('tag', ''), empty=True)):
        raise ValueError('invalid mission goal')
    for key in ('progress', 'budgets'):
        if not isinstance(value.get(key, {}), dict): raise ValueError(f'{key} must be an object')
    return value
