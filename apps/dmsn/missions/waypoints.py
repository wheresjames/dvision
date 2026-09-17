"""waypoints: fly a list of points in order, with a policy for the unreachable ones.

The points come from the profile's ``waypoints`` list (the Mission tab edits
them, or a tour import writes them). Arrival is :meth:`Observation.arrived` --
dway's COMPLETE for this coordinator's goal -- the same single definition every
other mission uses. What to do when a point never arrives is the mission's one
policy knob: give up the whole mission (``abort``, the careful default), drop
the point and carry on (``skip``), or try the same point a few more times
(``retry N``). A point never arrives when dnav keeps refusing to plan to it
(``unreachable_s``), or when the vehicle is stuck. "Stuck" is measured as
progress, not time: the distance to the point must stop improving, so a long
legitimate flight never counts as stuck.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar, Sequence

from dcmn import theme
from dcmn.mapview import Shape
from dmsn.inputs import Observation
from dmsn.intents import Decision, Done, Goal
from dmsn.schema import Field, ParamSchema

POLICIES = ('abort', 'skip', 'retry')
#: Progress that resets the stuck clock, in metres of distance-to-waypoint.
PROGRESS_M = 0.25


class WaypointsMission:
    name: ClassVar[str] = 'waypoints'
    label: ClassVar[str] = 'Fly the waypoint list'
    editors: ClassVar[tuple[str, ...]] = ('waypoints',)
    reference_default: ClassVar[bool] = False
    schema: ClassVar[ParamSchema] = ParamSchema((
        Field('failure_policy', 'choice', default='abort', choices=POLICIES,
              help='abort: the first stuck waypoint fails the mission; skip: drop it and '
                   'carry on; retry: try the same point again'),
        Field('retries', 'int', default=0, minimum=0, maximum=10,
              help='with the retry policy: attempts per waypoint before it is given up'),
        Field('stuck_s', 'float', default=20.0, minimum=1.0, unit='s',
              help='how long without getting closer to the waypoint counts as stuck'),
        Field('unreachable_s', 'float', default=5.0, minimum=0.0, unit='s',
              help='how long dnav may refuse to plan to a waypoint before it counts as unreachable'),
    ))

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.waypoints: list[tuple[float, float]] = []
        self.states: list[str] = []  # pending | visiting | visited | skipped | failed
        self.index = 0
        self.attempts = 0
        #: (best distance to the current waypoint, time it last improved)
        self.best: tuple[float, float] | None = None
        self.visited = 0
        self.skipped = 0
        #: When dnav's refusal of the current waypoint was first seen.
        self.refused_since: float | None = None

    def start(self, params: dict[str, Any], obs: Observation, *, area: Sequence = (),
              waypoints: Sequence[tuple[float, float]] = (), strategy: Any = None) -> None:
        self.params = self.schema.validate(params)
        self.waypoints = [(float(x), float(y)) for x, y in waypoints]
        self.states = ['pending'] * len(self.waypoints)
        self.index, self.attempts, self.best = 0, 0, None
        self.visited, self.skipped = 0, 0
        self.refused_since = None

    def on_event(self, event) -> None:
        pass  # the list is edited between missions, not during one

    # -- the body ------------------------------------------------

    def _reason(self) -> str:
        text = f'visited {self.visited} waypoint{"s" if self.visited != 1 else ""}'
        if self.skipped: text += f', skipped {self.skipped}'
        return text

    def _advance(self, obs: Observation) -> Decision:
        """Past the current waypoint (arrived or given up): the next one, or done."""
        self.index += 1
        self.attempts = 0
        self._seed(obs)
        if self.index >= len(self.waypoints): return Done('complete', self._reason())
        return Goal(self.waypoints[self.index], f'waypoint {self.index + 1}', operator=False)

    def _seed(self, obs: Observation) -> None:
        """Start the stuck clock for the current waypoint from this observation."""
        self.best = None if obs.xy is None or self.index >= len(self.waypoints) else \
            (math.dist(obs.xy, self.waypoints[self.index]), obs.time_s)
        self.refused_since = None

    def _settle(self, skipped: bool) -> None:
        """Record how the current waypoint ended, and keep the states list honest."""
        self.states[self.index] = 'skipped' if skipped else 'visited'
        if skipped: self.skipped += 1
        else: self.visited += 1

    def _give_up(self, reason: str, obs: Observation) -> Decision:
        """Apply the policy to a stuck waypoint."""
        where = f'waypoint {self.index + 1} ({self.waypoints[self.index][0]:.1f}, ' \
                f'{self.waypoints[self.index][1]:.1f} m)'
        policy = self.params['failure_policy']
        if policy == 'abort':
            self.states[self.index] = 'failed'
            return Done('failed', f'{where} is unreachable: {reason}')
        if policy == 'retry' and self.attempts < int(self.params['retries']):
            self.attempts += 1
            self._seed(obs)
            return Goal(self.waypoints[self.index], f'{where} retry {self.attempts}', operator=False)
        self._settle(skipped=True)
        return self._advance(obs)

    def _stuck(self, obs: Observation) -> str | None:
        """Why the current waypoint is stuck, or None when it is still making progress.

        The anchor is the distance at the last reset and stays there until real
        progress moves it. Lowering it every step -- which this used to do --
        asks for ``PROGRESS_M`` within a single 10 Hz step, so the clock never
        resets and a long leg is failed while it is still flying (under the
        ``abort`` default, that fails the whole mission). See the explore
        mission, which had the same fault.
        """
        if obs.xy is None: return None  # no pose: nothing to measure with yet
        distance = math.dist(obs.xy, self.waypoints[self.index])
        if self.best is None or distance < self.best[0] - PROGRESS_M:
            self.best = (distance, obs.time_s)
            return None
        if obs.time_s - self.best[1] < float(self.params['stuck_s']): return None
        return f'no progress for {obs.time_s - self.best[1]:.0f} s (still {distance:.1f} m away)'

    def _refused(self, obs: Observation) -> str | None:
        """Why dnav cannot reach the current waypoint, once it has said so for ``unreachable_s``."""
        status = obs.route_refusal(self.waypoints[self.index])
        if status is None:
            self.refused_since = None
            return None
        if self.refused_since is None: self.refused_since = obs.time_s
        held = obs.time_s - self.refused_since
        if held < float(self.params['unreachable_s']): return None
        return f'dnav reports {status} for {held:.0f} s'

    def step(self, obs: Observation) -> Decision:
        if not self.waypoints:
            return Done('failed', 'no waypoints in the profile')
        if self.best is None: self._seed(obs)
        self.states[self.index] = 'visiting'
        if obs.arrived(self.waypoints[self.index]):
            self._settle(skipped=False)
            return self._advance(obs)
        refused = self._refused(obs)
        if refused is not None: return self._give_up(refused, obs)
        stuck = self._stuck(obs)
        if stuck is not None: return self._give_up(stuck, obs)
        tag = f'waypoint {self.index + 1}'
        if self.attempts: tag += f' retry {self.attempts}'
        return Goal(self.waypoints[self.index], tag, operator=False)

    # -- what the window and the archive see --------------------------------

    def overlay(self) -> list[Shape]:
        shapes = []
        for i, (x, y) in enumerate(self.waypoints):
            state = self.states[i]
            r, outline, width = 0.4, theme.OK, 2.
            if state == 'pending': outline, width = theme.ACCENT, 1.5
            elif state == 'skipped': outline, width = theme.WARN, 1.5
            elif state == 'failed': outline = theme.DANGER
            elif state != 'visiting': outline, width = theme.DIM, 1.
            shapes.append(Shape('oval', x - r, y - r, x + r, y + r, outline=outline, width=width))
            if i:  # the leg from the previous point
                px, py = self.waypoints[i - 1]
                shapes.append(Shape('line', px, py, x, y, outline=theme.DIM, width=1.))
        return shapes

    def progress(self) -> dict[str, Any]:
        return dict(waypoint_states=list(self.states),
                    waypoint_index=None if self.index >= len(self.waypoints) else self.index,
                    waypoints_done=self.visited, waypoints_skipped=self.skipped)

    def summary(self) -> dict[str, Any]:
        return dict(failure_policy=self.params.get('failure_policy'),
                    waypoints=[list(w) for w in self.waypoints], states=list(self.states),
                    visited=self.visited, skipped=self.skipped)
