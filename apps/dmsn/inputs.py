"""One immutable `Observation` per mission step: everything a mission may know.

A mission is pure logic over these. It never opens a plane, reads a clock or
asks a module anything; the runner gathers one observation, the mission
decides, and both halves go into the archive. That is what lets a recorded
run be replayed through the same mission and checked decision for decision,
and what lets a mission be tested with no processes at all.

Only what a mission legitimately has is here: the session's pose, goal and
authority; dway's execution state; dnav's route status; the evidence coverage
extent; the provider's battery, map name and published target. No grids, no
world.

Missions that declare a target (a ``target_label`` parameter) also observe
the detections plane (:mod:`dcmn.detections`), filtered to that label: the
latest detections and the running estimates, each capped as the plane caps
them. Every other mission never opens the plane (Q27).

One bounded exception for area missions: the runner distils the evidence plane
into the frontier points ahead, the occupied points nearest the vehicle and the
observed fraction of the mission area, capped the same way every field here
is, so an explore mission still decides without a grid and a recorded run
still replays decision for decision (Q28).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

#: Execution status fields a mission may see, copied so a snapshot cannot change under it.
EXECUTION_KEYS = ('state', 'reason', 'hold_kind', 'launch_phase', 'auto', 'wait_for_mission', 'mission_hold',
                  'authorized_goal', 'mission_intent', 'owns_control', 'session', 'vehicle_fault')
NAVIGATION_KEYS = ('planning_status', 'route_mode')
#: How close a published goal must be to a mission's point to be that goal.
SAME_GOAL_M = 1e-6
#: Frontier points an observation may carry, nearest to the vehicle first.
FRONTIER_CAP = 64
#: A frontier this close to a point the mission gave up on is the same dead
#: end, and is left out before the cap so it cannot hide live frontiers.
FRONTIER_EXCLUDE_M = 2.0
#: Occupied evidence points an observation may carry, and how far from the
#: vehicle they are looked for.
WALL_CAP = 64
WALL_RADIUS_M = 8.0
#: dnav planning statuses that say this goal cannot be flown to as things stand.
ROUTE_REFUSALS = ('goal_unreachable', 'no_route', 'outside_coverage', 'start_blocked')
#: The refusals that are verdicts about the goal *point* rather than about the
#: moment: an obstacle, its margin, or ground outside the mapped area. Waiting
#: cannot change any of them, so a mission gives such a goal up at once instead
#: of spending its ``unreachable_s``. The others may be transient -- a plan made
#: for the previous revision, a vehicle still braking -- and keep the timer.
GOAL_REFUSALS = ('goal_unreachable', 'outside_coverage')
#: dnav's route modes that are not a plan to the goal but still something to
#: fly: ``retained`` is the last good route, permitted up to the last clear
#: point before the obstacle -- getting closer is how a distant phantom clears
#: -- and ``substitute`` is a route to a reachable point near the goal. While
#: either is eligible the vehicle has somewhere to go, whatever the planning
#: status says.
FALLBACK_MODES = ('retained', 'substitute')
#: dway's hold kinds for a stop short of the goal. Its arrival gate is measured
#: to the goal itself, and a substitute route deliberately ends up to
#: ``goal_substitute_m`` away, so a flown substitute ends in an ordinary
#: ``prefix`` hold ("the permitted interval ends before the goal") -- never
#: COMPLETE. ``arrival-unconfirmed`` is the edge of the same case: a route that
#: did reach the gate, stopped outside it.
SHORT_OF_GOAL_HOLDS = ('prefix', 'arrival-unconfirmed')
#: How long a verdict about the goal *point* must hold before a mission acts on
#: it. Not zero: the mechanism behind such a verdict is often one never-observed
#: cell beside an obstacle counting as solid, and a single frame of that should
#: not cost a frontier permanently -- the 15 Sep area1 phantom cell is the case
#: `plan_near` exists for. Short enough that confirming costs nothing.
GOAL_REFUSAL_S = 0.5


def _round(value: Any) -> Any:
    if isinstance(value, float): return round(value, 6)
    if isinstance(value, (list, tuple)): return [_round(v) for v in value]
    if isinstance(value, dict): return {k: _round(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Observation:
    time_s: float = 0.0
    session_id: str = ''
    #: ``[x, y, z, heading_deg]`` of a valid, fresh pose; None with ``pose_error`` otherwise.
    pose: tuple[float, float, float, float] | None = None
    pose_error: str = ''
    #: The session goal descriptor (position, revision, authority, role), whoever set it.
    goal: dict[str, Any] | None = None
    authority: dict[str, Any] | None = None
    #: This coordinator's goal-authority writer id.
    writer: str = ''
    execution: dict[str, Any] = field(default_factory=dict)
    navigation: dict[str, Any] = field(default_factory=dict)
    #: Evidence coverage ``[xmin, ymin, xmax, ymax]``, or None before dalg publishes.
    coverage: tuple[float, float, float, float] | None = None
    battery_pct: float | None = None
    arena: str = ''
    provider_target: tuple[float, float] | None = None
    provider_target_reason: str = ''
    #: Frontier points (map metres) the runner distilled from the evidence
    #: plane, at most :data:`FRONTIER_CAP`, nearest first; empty when the
    #: mission asks for no area or the evidence says none is left.
    frontiers: tuple[tuple[float, float], ...] = ()
    #: Observed fraction (0-1) of the area's cells, or None with no area asked
    #: for or no evidence to measure yet.
    area_coverage: float | None = None
    #: Occupied evidence points (map metres) within :data:`WALL_RADIUS_M` of
    #: the vehicle, at most :data:`WALL_CAP`, nearest first; empty without an
    #: area, a pose or evidence.
    walls: tuple[tuple[float, float], ...] = ()
    #: For a mission with a target: the plane's latest detections of that label
    #: (``id``, ``detector``, ``label``, ``confidence``, ``capture_time_s``,
    #: ``projectable``, ``xy_m``, ``estimate``), newest first.
    detections: tuple[dict[str, Any], ...] = ()
    #: For a mission with a target: the plane's estimates of that label (``id``,
    #: ``xy_m``, ``spread_m``, ``count``, ``revision``, ``confirmed``, ``stale``,
    #: ``first_s``, ``last_s``, ``confidence``).
    estimates: tuple[dict[str, Any], ...] = ()
    #: The plane's sim time, or None when no plane is readable; the reason says why.
    detections_time_s: float | None = None
    detections_reason: str = ''

    # -- questions a mission asks -------------------------------------------

    @property
    def xy(self) -> tuple[float, float] | None:
        return None if self.pose is None else (self.pose[0], self.pose[1])

    @property
    def goal_is_ours(self) -> bool:
        return bool(self.goal) and self.goal.get('authority') == self.writer

    def goal_at(self, xy) -> bool:
        """Whether the session goal is this coordinator's goal at ``xy``."""
        if not self.goal_is_ours or xy is None: return False
        position = self.goal.get('position') or ()
        return len(position) >= 2 and math.dist(position[:2], xy) <= SAME_GOAL_M

    def arrived(self, xy) -> bool:
        """dway reports COMPLETE for this coordinator's goal at ``xy``: the one definition of arrival."""
        if not self.goal_at(xy): return False
        authorized = self.execution.get('authorized_goal') or []
        return (self.execution.get('state') == 'COMPLETE' and len(authorized) >= 3
                and authorized[2] == self.goal.get('revision'))

    def _fallback_mode(self, xy) -> str:
        """The fallback route mode dnav publishes for this goal, or ``''``."""
        if not self.goal_at(xy): return ''
        nav = self.navigation
        if nav.get('goal_revision') != self.goal.get('revision'): return ''
        mode = nav.get('route_mode')
        return mode if mode in FALLBACK_MODES else ''

    def fallback_offered(self, xy) -> bool:
        """Whether dnav is still offering an eligible route toward this goal.

        A planning status of `goal_unreachable` does not mean the vehicle has
        nowhere to go: dnav answers with the last good route as far as it is
        still clear (`retained`), or with a route to a reachable point near the
        goal (`substitute`). While one of those is eligible there is something
        to fly, and flying it is also what can clear the refusal — a
        never-observed cell beside a wall counts as solid until a sensor looks
        at it, and the 17 Sep 14:27 area1 run had a standoff that was
        unreachable for two seconds and fine afterwards.
        """
        return bool(self._fallback_mode(xy)) and bool(self.navigation.get('eligible'))

    def fallback_reached(self, xy) -> str:
        """The fallback the vehicle has flown and parked at the end of, or ``''``.

        dway measures arrival against the goal itself, so the end of a fallback
        route is never ``COMPLETE``: it holds with one of
        :data:`SHORT_OF_GOAL_HOLDS` instead. That hold is the answer — *this is
        as close as permission allows* — not something to wait out. For an
        ordinary `planned` route the same hold kind means a vehicle stopped
        short for some other reason, so a fallback mode is required too.

        Deliberately **not** conditioned on the clearance being eligible.
        Parked at the end of a fallback is exactly where dnav stops offering
        it: the permitted interval has been used up, so eligibility goes false
        at the moment this question matters.
        """
        mode = self._fallback_mode(xy)
        if not mode or self.execution.get('state') != 'HOLDING': return ''
        return mode if self.execution.get('hold_kind') in SHORT_OF_GOAL_HOLDS else ''

    def route_refusal(self, xy) -> str | None:
        """dnav's refusal status for this coordinator's goal at ``xy``, or None.

        Only a plan made for the goal revision now in the session counts, so a
        refusal of the previous goal is never blamed on this one.
        """
        if not self.goal_at(xy): return None
        nav = self.navigation
        if nav.get('goal_revision') != self.goal.get('revision'): return None
        status = nav.get('planning_status')
        return status if status in ROUTE_REFUSALS else None

    @property
    def airborne(self) -> bool:
        """Whether dway has the vehicle up: target mode says so; dynamic mode requires it."""
        ex = self.execution
        if not ex: return False
        if ex.get('auto'): return ex.get('launch_phase') == 'airborne'
        return True

    # -- the record ----------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return _round(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'Observation':
        value = dict(value)
        for key in ('pose', 'coverage', 'provider_target'):
            if value.get(key) is not None: value[key] = tuple(value[key])
        for key in ('frontiers', 'walls'):
            if value.get(key) is not None: value[key] = tuple(tuple(p) for p in value[key])
        for key in ('detections', 'estimates'):
            if value.get(key) is not None: value[key] = tuple(dict(row) for row in value[key])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in value.items() if k in known})

    def digest(self) -> str:
        raw = json.dumps(self.as_dict(), sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()


DETECTION_KEYS = ('id', 'detector', 'label', 'confidence', 'capture_time_s', 'projectable', 'xy_m', 'sigma_m',
                  'range_m', 'estimate', 'reason')
ESTIMATE_KEYS = ('id', 'label', 'detector', 'xy_m', 'spread_m', 'count', 'revision', 'confirmed', 'stale',
                 'first_s', 'last_s', 'confidence')


def refusal_verdict(obs: 'Observation', xy, refused_since: float | None, unreachable_s: float, *,
                    closing: bool = True) -> tuple[str | None, float | None]:
    """``(reason to give this goal up, the refusal clock)`` for a refused goal.

    One definition for every mission that can give a goal up, because the
    question is the same one several times over:

    * dnav is not refusing this goal -- nothing to answer, clock cleared;
    * the vehicle has flown one of dnav's fallbacks and is holding at its end:
      give the goal up now, having got as close as permission allows;
    * otherwise run a clock, reset while the vehicle is still **closing** on
      the goal, and compare it against one of two limits:

      - a fallback is on offer (`retained` or `substitute`), so there is
        somewhere to fly: the full ``unreachable_s``, refreshed by every
        ``PROGRESS_M`` of progress. A vehicle that is still permitted to fly at
        the goal is never given up on; one that has stopped making progress
        toward it is, five seconds later;
      - no fallback, and the refusal is a verdict about the point itself
        (:data:`GOAL_REFUSALS`): :data:`GOAL_REFUSAL_S`, long enough to
        confirm the verdict across observations and no longer, because waiting
        cannot make an obstacle's margin passable;
      - anything else (``no_route``, ``start_blocked`` -- a plan made for the
        previous revision, a vehicle still braking): ``unreachable_s``.

    ``closing`` is the mission's own progress test, the same one its stuck
    clock uses, so "still getting closer" has one definition per mission.

    The 17 Sep area1 search run is what this is for: nine frontier goals were
    refused as "an obstacle or inside its 0.35 m margin" and each one cost five
    seconds of waiting on top of the three dnav spent computing a substitute
    nobody ever flew.
    """
    status = obs.route_refusal(xy)
    if status is None: return None, None
    parked = obs.fallback_reached(xy)
    if parked:
        detail = ('flew to the nearest reachable point instead' if parked == 'substitute'
                  else 'stopped at the last clear point short of it')
        return f'{status}; {detail}', None
    # The clock is never discarded while the refusal stands: restarting it would
    # hand back the seconds this exists to save.
    if refused_since is None or closing: refused_since = obs.time_s
    offered = obs.fallback_offered(xy)
    if offered or status not in GOAL_REFUSALS: limit = float(unreachable_s)
    else: limit = min(float(unreachable_s), GOAL_REFUSAL_S)
    if obs.time_s - refused_since < limit: return None, refused_since
    if offered:
        return f'{status}; the route toward it stopped closing on it', refused_since
    return status, refused_since


def detections_view(value: dict[str, Any] | None, target: dict[str, Any]) -> tuple[tuple, tuple, float | None, str]:
    """``(detections, estimates, plane time, reason)`` of one target from a detections snapshot.

    Filtered to the target's label, its detector when one is named, and its
    minimum confidence for detections (estimates carry their own confidence,
    judged by the mission).
    """
    if not value: return (), (), None, 'no detections plane is published'
    label, detector = target.get('label'), target.get('detector') or ''
    minimum = float(target.get('min_confidence') or 0.0)

    def wanted(row):
        return row.get('label') == label and (not detector or row.get('detector') == detector)
    detections = tuple({k: row.get(k) for k in DETECTION_KEYS if k in row} for row in value.get('detections', ())
                       if wanted(row) and float(row.get('confidence', 0.0)) >= minimum)
    estimates = tuple({k: row.get(k) for k in ESTIMATE_KEYS if k in row} for row in value.get('estimates', ())
                      if wanted(row))
    return detections, estimates, float(value.get('time_s', 0.0)), ''


def execution_view(status: dict[str, Any] | None) -> dict[str, Any]:
    status = status or {}
    return {k: status.get(k) for k in EXECUTION_KEYS if k in status}


def navigation_view(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    if not value: return {}
    view = {k: value.get(k) for k in NAVIGATION_KEYS if k in value}
    clearance = value.get('clearance') or {}
    view['eligible'] = bool(clearance.get('eligible'))
    view['reason'] = str(clearance.get('reason', ''))[:200]
    goal = value.get('goal') or {}
    view['goal_revision'] = goal.get('revision')
    return view
