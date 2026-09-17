"""search: explore an area until a target is confirmed, look again from a standoff, and land on it.

The mission runs no detector. It declares the label it needs (``target_label``,
optionally one ``target_detector`` and a ``min_confidence``); the runner refuses
to start it unless some dalg detector publishes that label, and hands it the
detections plane's estimates of that label in every observation (§13.6).

Phases, all decided on observations alone:

``explore``
    an :class:`~dmsn.missions.explore.ExploreMission` over the area, until an
    estimate of the label is confirmed, fresh and confident enough.
``standoff``
    fly to a point ``standoff_m`` short of the estimate, on the line from the
    vehicle, so the pad sits inside the front camera's ground footprint (B6).
``face``
    ask dway to ``yaw_to`` the estimate and wait for its answer; a refused or
    failed turn is noted and the mission looks anyway.
``observe``
    wait ``observe_s`` sim seconds and require the estimate to still be
    confirmed, fresh, and seen again since observing began. If not, look again;
    after ``max_standoff_attempts`` the estimate is given up and exploring
    resumes (Q26).
``approach``
    fly to the estimate and end ``complete`` with the ``land`` finish action:
    the landing is on the estimate, never a visual servo.

The mission ends ``failed: no confirmed target`` when exploration ends, or a
budget is spent, before it reaches the estimate (Q26). Every decision that acted
on an estimate is logged with the estimate's ID and revision, so the dmsn
summary links back to dalg's archive.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar, Sequence

from dcmn import theme
from dcmn.mapview import Shape
from dmsn.inputs import Observation, refusal_verdict
from dmsn.intents import Action, Decision, Done, Goal, Wait
from dmsn.missions.explore import PROGRESS_M, ExploreMission
from dmsn.schema import Field, ParamSchema

#: Sim seconds a turn in place may take before the mission looks anyway.
FACE_TIMEOUT_S = 60.0
#: Decisions kept in the estimate log the summary carries; the archive keeps every one.
LOG_CAP = 256
#: Estimates listed in the status snapshot's progress.
ESTIMATES_SHOWN = 8
#: How far round the estimate a retried standoff moves, in degrees. A standoff
#: is computed geometry -- ``standoff_m`` short of the estimate on the bearing
#: back to the vehicle -- so one that cannot be flown is answered by approaching
#: from somewhere else, not by trying the same point again.
STANDOFF_TURN_DEG = 60.0
#: New observations of an estimate given up for geometry before the mission will
#: consider it again. A standoff or an approach that could not be flown says
#: nothing about the target, and the evidence it could not be flown through
#: changes as the vehicle explores; a pad the detector keeps seeing deserves
#: another attempt. Counted in observations rather than seconds because it is
#: the evidence, not the clock, that has to change.
RETRY_OBSERVATIONS = 20


def _heading(from_xy, to_xy) -> float:
    """Compass heading (0 north, clockwise; x east, y south) from one point to another."""
    return math.degrees(math.atan2(to_xy[0] - from_xy[0], -(to_xy[1] - from_xy[1]))) % 360.0


class SearchMission:
    name: ClassVar[str] = 'search'
    label: ClassVar[str] = 'Search an area and land on the target'
    editors: ClassVar[tuple[str, ...]] = ('area',)
    reference_default: ClassVar[bool] = False
    default_strategy: ClassVar[str] = 'frontier'
    schema: ClassVar[ParamSchema] = ParamSchema((
        Field('target_label', 'text', default='landing_pad',
              help='the label a dalg detector must publish; search refuses to start without one'),
        Field('target_detector', 'text', default='',
              help='only this detector ID (e.g. front-red_disc); empty accepts any detector of the label'),
        Field('min_confidence', 'float', default=0.3, minimum=0.0, maximum=1.0,
              help='detections and estimates below this confidence are ignored'),
        Field('standoff_m', 'float', default=4.0, minimum=0.5, maximum=20.0, unit='m',
              help='how far short of the estimate the vehicle stops to look again'),
        Field('observe_s', 'float', default=3.0, minimum=0.5, maximum=60.0, unit='s',
              help='how long the vehicle looks from the standoff before deciding'),
        Field('max_standoff_attempts', 'int', default=2, minimum=1, maximum=10,
              help='looks at one estimate before it is given up and exploring resumes'),
        Field('stuck_s', 'float', default=20.0, minimum=1.0, unit='s',
              help='how long without getting closer to a goal counts as stuck'),
        Field('unreachable_s', 'float', default=5.0, minimum=0.0, unit='s',
              help='how long dnav may refuse to plan to a goal before it is given up'),
    ))

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.explore = ExploreMission()
        self.phase = 'explore'
        self.target: str | None = None
        self.target_xy: tuple[float, float] | None = None
        self.standoff_xy: tuple[float, float] | None = None
        self.approach_xy: tuple[float, float] | None = None
        self.attempts: dict[str, int] = {}
        self.given_up: dict[str, str] = {}
        #: Estimate ID to the observation count at which a geometry give-up may
        #: be reconsidered. Absent means the give-up was about the target.
        self.retry_at: dict[str, int] = {}
        self.log: list[dict[str, Any]] = []
        self.phase_s = 0.0
        self.observe_s: float | None = None
        self.yaw_mark = 0
        self.best: tuple[float, float] | None = None
        #: When dnav's refusal of the current goal was first seen.
        self.refused_since: float | None = None
        self.estimates: tuple[dict[str, Any], ...] = ()
        self.detections: tuple[dict[str, Any], ...] = ()
        self.landed_on: dict[str, Any] | None = None

    def start(self, params: dict[str, Any], obs: Observation, *, area: Sequence = (),
              waypoints: Sequence[tuple[float, float]] = (), strategy: Any = None) -> None:
        self.params = self.schema.validate(params)
        self.explore.start(dict(stuck_s=self.params['stuck_s'], unreachable_s=self.params['unreachable_s']), obs,
                           area=area, strategy=strategy)
        self.phase, self.target, self.target_xy = 'explore', None, None
        self.standoff_xy = self.approach_xy = None
        self.attempts, self.given_up, self.log = {}, {}, []
        self.retry_at = {}
        self.phase_s, self.observe_s, self.yaw_mark, self.best = obs.time_s, None, 0, None
        self.refused_since = None
        self.estimates, self.detections, self.landed_on = (), (), None

    def on_event(self, event) -> None:
        self.explore.on_event(event)

    # -- helpers -----------------------------------------------------------------

    def _enter(self, phase: str, obs: Observation) -> None:
        self.phase, self.phase_s, self.best, self.refused_since = phase, obs.time_s, None, None

    def _refused(self, obs: Observation, xy) -> str | None:
        """Why the goal at ``xy`` should be given up to dnav's refusal, or None.

        :func:`~dmsn.inputs.refusal_verdict`, the same rules the explore
        mission applies to a frontier: a standoff or an approach point is
        computed geometry too, and one that lands inside an obstacle margin is
        no more reachable for being waited on.
        """
        reason, self.refused_since = refusal_verdict(obs, xy, self.refused_since,
                                                     self.params['unreachable_s'],
                                                     closing=self._closing(obs, xy))
        return reason

    def _note(self, obs: Observation, decision: str, estimate: dict[str, Any] | None, detail: str = '') -> None:
        self.log.append(dict(time_s=round(obs.time_s, 4), phase=self.phase, decision=decision,
                             estimate=None if estimate is None else estimate.get('id'),
                             revision=None if estimate is None else estimate.get('revision'),
                             xy_m=None if estimate is None else estimate.get('xy_m'),
                             count=None if estimate is None else estimate.get('count'), detail=detail))
        del self.log[:-LOG_CAP]

    def _usable(self, estimate: dict[str, Any]) -> bool:
        return (bool(estimate.get('confirmed')) and not estimate.get('stale')
                and float(estimate.get('confidence', 0.0)) >= float(self.params['min_confidence'])
                and estimate.get('id') not in self.given_up)

    def _candidate(self, obs: Observation) -> dict[str, Any] | None:
        usable = [e for e in obs.estimates if self._usable(e)]
        if not usable: return None
        here = obs.xy or (0.0, 0.0)
        return min(usable, key=lambda e: (-int(e.get('count', 0)), math.dist(here, e['xy_m']), e['id']))

    def _current(self, obs: Observation) -> dict[str, Any] | None:
        return next((e for e in obs.estimates if e.get('id') == self.target), None)

    def _closing(self, obs: Observation, xy) -> bool:
        """Whether the vehicle has closed ``PROGRESS_M`` on ``xy`` since the anchor."""
        if obs.xy is None or xy is None or self.best is None: return True
        return math.dist(obs.xy, xy) < self.best[0] - PROGRESS_M

    def _stuck(self, obs: Observation, xy) -> bool:
        if obs.xy is None: return False
        if self.best is None or self._closing(obs, xy):
            self.best = (math.dist(obs.xy, xy), obs.time_s)
            return False
        return obs.time_s - self.best[1] >= float(self.params['stuck_s'])

    def _give_up(self, obs: Observation, reason: str, estimate: dict[str, Any] | None, *,
                 retry: bool = False) -> Decision:
        """Give the target up and go back to exploring.

        ``retry`` marks a give-up that was about the *geometry* -- a point that
        could not be flown to -- rather than about the target. Such an estimate
        is reconsidered once the detector has seen it :data:`RETRY_OBSERVATIONS`
        more times, because what blocked the route is evidence, and evidence
        changes as the vehicle explores. Without that, the 17 Sep 14:27 area1
        run discarded a pad it went on to observe 96 more times.
        """
        if self.target is not None:
            self.given_up[self.target] = reason
            if retry and estimate is not None:
                self.retry_at[self.target] = int(estimate.get('count') or 0) + RETRY_OBSERVATIONS
        self._note(obs, 'give up', estimate or ({'id': self.target} if self.target else None), reason)
        self.target = self.target_xy = self.standoff_xy = self.approach_xy = None
        self._enter('explore', obs)
        return self._explore(obs)

    def _revive(self, obs: Observation) -> None:
        """Re-admit an estimate given up for geometry that has been seen again."""
        for row in obs.estimates:
            target = row.get('id')
            wanted = self.retry_at.get(target)
            if wanted is None or int(row.get('count') or 0) < wanted: continue
            self.given_up.pop(target, None)
            self.retry_at.pop(target, None)
            self.attempts.pop(target, None)
            self._note(obs, 'reconsider', row, f'seen {row.get("count")} times since it was given up')

    def _retry_standoff(self, obs: Observation, estimate: dict[str, Any], why: str) -> Decision:
        """A standoff that cannot be flown costs an attempt, not the estimate.

        The point is computed geometry, so one that lands in an obstacle's
        margin says nothing about the target: the answer is to approach from
        another bearing. Only when the attempts are spent is the estimate given
        up, and then only until the detector has re-confirmed it.
        """
        attempts = self.attempts.get(self.target, 0) + 1
        self.attempts[self.target] = attempts
        if attempts >= int(self.params['max_standoff_attempts']):
            return self._give_up(obs, f'{why} after {attempts} standoff(s)', estimate, retry=True)
        turn = STANDOFF_TURN_DEG * attempts * (1 if attempts % 2 else -1)
        self._note(obs, 'move the standoff', estimate, f'{why}; {turn:+.0f}° round it (try {attempts})')
        return self._go_standoff(obs, estimate, turn_deg=turn)

    def _explore(self, obs: Observation) -> Decision:
        candidate = self._candidate(obs)
        if candidate is not None: return self._go_standoff(obs, candidate)
        decision = self.explore.step(obs)
        if isinstance(decision, Done):
            tried = f'; gave up {len(self.given_up)} estimate(s)' if self.given_up else ''
            return Done('failed', f'no confirmed target: {decision.reason}{tried}')
        return decision

    def _go_standoff(self, obs: Observation, estimate: dict[str, Any], *, turn_deg: float = 0.) -> Decision:
        """Fly to a point ``standoff_m`` short of the estimate, facing it (B6).

        On the bearing back to the vehicle, so the shortest approach is the one
        taken; ``turn_deg`` swings that bearing round the estimate for a retry,
        which keeps the distance -- and so the camera's footprint -- while
        changing the route to it.
        """
        self.target, self.target_xy = estimate['id'], tuple(estimate['xy_m'])
        here = obs.xy or self.target_xy
        distance = math.dist(here, self.target_xy)
        standoff = float(self.params['standoff_m'])
        if distance <= standoff and not turn_deg:
            self.standoff_xy = (float(here[0]), float(here[1]))
        else:
            if distance <= 1e-9: ux, uy = 1., 0.
            else: ux, uy = (here[0] - self.target_xy[0]) / distance, (here[1] - self.target_xy[1]) / distance
            angle = math.radians(turn_deg)
            rx = ux * math.cos(angle) - uy * math.sin(angle)
            ry = ux * math.sin(angle) + uy * math.cos(angle)
            self.standoff_xy = (self.target_xy[0] + rx * standoff, self.target_xy[1] + ry * standoff)
        self._enter('standoff', obs)
        turned = f', {turn_deg:+.0f}° round it' if turn_deg else ''
        self._note(obs, 'go to standoff', estimate, f'{standoff:g} m short, from {distance:.1f} m{turned}')
        return Goal(self.standoff_xy, f'standoff for {self.target}', operator=False)

    def _face(self, obs: Observation, estimate: dict[str, Any]) -> Decision:
        here = obs.xy or self.standoff_xy
        heading = _heading(here, estimate['xy_m'])
        ack = obs.execution.get('mission_intent') or {}
        self.yaw_mark = int(ack.get('revision') or 0)
        self._enter('face', obs)
        self._note(obs, 'face estimate', estimate, f'heading {heading:.0f}°')
        return Action('yaw_to', dict(heading_deg=round(heading, 3)))

    # -- the body --------------------------------------------------------------------

    def step(self, obs: Observation) -> Decision:
        self.estimates = tuple(obs.estimates)
        self.detections = tuple(obs.detections)
        self._revive(obs)
        if self.phase == 'explore': return self._explore(obs)
        estimate = self._current(obs)
        if self.phase != 'approach' and (estimate is None or estimate.get('stale')):
            # Stale is about the detector's last sighting, not about the target:
            # if it is seen again often enough, it is worth another approach.
            return self._give_up(obs, 'estimate went stale' if estimate else 'estimate no longer published',
                                 estimate, retry=estimate is not None)
        if estimate is not None: self.target_xy = tuple(estimate['xy_m'])
        if self.phase == 'standoff':
            if obs.arrived(self.standoff_xy): return self._face(obs, estimate)
            refused = self._refused(obs, self.standoff_xy)
            if refused is not None:
                return self._retry_standoff(obs, estimate, f'standoff unreachable: dnav {refused}')
            if self._stuck(obs, self.standoff_xy):
                return self._retry_standoff(
                    obs, estimate, f'stuck short of the standoff for {self.params["stuck_s"]:g} s')
            return Goal(self.standoff_xy, f'standoff for {self.target}', operator=False)
        if self.phase == 'face':
            ack = obs.execution.get('mission_intent') or {}
            answered = (ack.get('kind') == 'yaw_to' and int(ack.get('revision') or 0) > self.yaw_mark
                        and ack.get('state') in ('completed', 'failed', 'refused'))
            timed_out = obs.time_s - self.phase_s >= FACE_TIMEOUT_S
            if not answered and not timed_out: return Wait(f'turning to face {self.target}')
            detail = f"turn {ack.get('state')}: {ack.get('reason', '')}" if answered else 'turn not answered in time'
            self._enter('observe', obs)
            self.observe_s = obs.time_s
            self._note(obs, 'observe', estimate, detail)
            return Wait(f'looking at {self.target} for {self.params["observe_s"]:g} s')
        if self.phase == 'observe':
            if obs.time_s - self.observe_s < float(self.params['observe_s']):
                return Wait(f'looking at {self.target} for {self.params["observe_s"]:g} s')
            seen = float(estimate.get('last_s', -math.inf)) >= self.observe_s
            if self._usable(estimate) and seen:
                self.approach_xy = tuple(estimate['xy_m'])
                self._enter('approach', obs)
                self._note(obs, 'approach estimate', estimate, f"re-observed; spread {estimate['spread_m']:.2f} m")
                return Goal(self.approach_xy, f'approach {self.target}', operator=False)
            attempts = self.attempts.get(self.target, 0) + 1
            self.attempts[self.target] = attempts
            why = 'not seen again from the standoff' if not seen else 'no longer confirmed or confident enough'
            if attempts >= int(self.params['max_standoff_attempts']):
                return self._give_up(obs, f'{why} after {attempts} look(s)', estimate)
            self._note(obs, 'look again', estimate, f'{why} (look {attempts})')
            return self._face(obs, estimate)
        # approach
        if obs.arrived(self.approach_xy):
            final = estimate or dict(id=self.target, revision=None, spread_m=None, count=None)
            self.landed_on = dict(id=final.get('id'), revision=final.get('revision'),
                                  xy_m=list(self.approach_xy), spread_m=final.get('spread_m'),
                                  count=final.get('count'))
            self._note(obs, 'land on estimate', estimate)
            spread = '' if final.get('spread_m') is None else f"spread {float(final['spread_m']):.2f} m, "
            count = '' if final.get('count') is None else f"{final['count']} observations"
            return Done('complete', f'landed on estimate {self.target} ({spread}{count})'.replace(' ()', ''),
                        finish='land')
        refused = self._refused(obs, self.approach_xy)
        if refused is not None:
            return self._give_up(obs, f'estimate unreachable: dnav {refused}', estimate, retry=True)
        if self._stuck(obs, self.approach_xy):
            return self._give_up(obs, f'stuck short of the estimate for {self.params["stuck_s"]:g} s',
                                 estimate, retry=True)
        return Goal(self.approach_xy, f'approach {self.target}', operator=False)

    def budget_spent(self, reason: str) -> tuple[str, str]:
        """How a spent budget ends a search: without a landing, it found no target (Q26)."""
        if self.target is None: return 'failed', f'no confirmed target: {reason}'
        return 'failed', f'no confirmed target landed on: {reason} while at {self.phase} for {self.target}'

    # -- what the window and the archive see -------------------------------------------

    def overlay(self) -> list[Shape]:
        shapes = list(self.explore.overlay())
        for row in self.detections:
            if not row.get('projectable') or not row.get('xy_m'): continue
            x, y = row['xy_m']
            shapes.append(Shape('oval', x - .15, y - .15, x + .15, y + .15, fill=theme.WARN, outline=theme.WARN))
        for estimate in self.estimates:
            x, y = estimate['xy_m']
            radius = max(float(estimate.get('spread_m') or 0.0), 0.3)
            given_up = estimate['id'] in self.given_up
            colour = theme.DIM if estimate.get('stale') or given_up else theme.TARGET_EDGE
            fill = colour if estimate.get('confirmed') and not estimate.get('stale') and not given_up else None
            outline = theme.HIGHLIGHT if estimate['id'] == self.target else colour
            shapes.append(Shape('oval', x - radius, y - radius, x + radius, y + radius, fill=fill, outline=outline,
                                width=2.))
            shapes.append(Shape.label(x, y - radius - .6, estimate['id'], outline=outline))
        for xy, text in ((self.standoff_xy, 'standoff'), (self.approach_xy, 'land')):
            if xy is None: continue
            x, y = xy
            shapes.append(Shape('rect', x - .3, y - .3, x + .3, y + .3, outline=theme.GOAL, width=2.))
            shapes.append(Shape.label(x, y + .8, text, outline=theme.GOAL))
        return shapes

    def progress(self) -> dict[str, Any]:
        progress = dict(self.explore.progress())
        current = next((e for e in self.estimates if e.get('id') == self.target), None)
        progress.update(search_phase=self.phase, target=self.target,
                        target_revision=None if current is None else current.get('revision'),
                        attempts=self.attempts.get(self.target, 0) if self.target else 0,
                        estimates=[dict(id=e['id'], confirmed=e.get('confirmed'), stale=e.get('stale'),
                                        count=e.get('count')) for e in self.estimates[:ESTIMATES_SHOWN]],
                        given_up=sorted(self.given_up)[:ESTIMATES_SHOWN])
        return progress

    def summary(self) -> dict[str, Any]:
        summary = dict(self.explore.summary())
        summary.update(target=dict(label=self.params.get('target_label'), detector=self.params.get('target_detector'),
                                   min_confidence=self.params.get('min_confidence')),
                       phase=self.phase, landed_on=self.landed_on, given_up=dict(self.given_up),
                       retry_at=dict(self.retry_at),
                       attempts=dict(self.attempts), estimate_log=list(self.log))
        return summary
