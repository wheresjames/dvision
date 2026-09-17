"""explore: cover an area by asking a strategy for the next goal, until it has none.

The area is the profile's polygon; the runner distils the evidence plane into
the observation's frontier list, nearby walls and area coverage, so the mission
still decides without a grid (Q28). This mission owns what the strategy does
not: commitment (one goal at a time), the blacklist (points it has given up on,
each with its reason), and every way the mission can end -- the coverage
target, the last frontier, a strategy with nothing left to propose. A goal is
given up on when it lies outside the evidence coverage (B1: a strategy's goal
never resets the map), when dnav keeps refusing to plan to it, or when the
vehicle stops getting closer. What proposes the next goal is a plugin
(:mod:`dmsn.strategies`): nearest frontier by default, wall following by goal
stepping as the other proof. The time, battery and path budgets and the
operator's Abort are the runner's, and end this mission like any other.

The overlay shows what the last observation told the mission: the frontier dots,
the blacklist crossed out, the committed goal ringed, and the area coverage
labelled at the area's centre. Remembering them in ``step`` keeps the overlay
derived from observations only, so a replayed run draws the same picture.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar, Sequence

from dcmn import theme
from dcmn.mapview import Shape
from dmsn import geometry as G
from dmsn.inputs import Observation, refusal_verdict
from dmsn.intents import Decision, Done, Goal, Wait
from dmsn.schema import Field, ParamSchema

#: Progress that resets the stuck clock, in metres of distance-to-goal.
PROGRESS_M = 0.25
#: Proposals tried in one step before the mission waits for the next observation.
PROPOSALS_PER_STEP = 8
#: Blacklist entries the status snapshot carries; the archive and summary keep all.
BLACKLIST_SHOWN = 32


class ExploreMission:
    name: ClassVar[str] = 'explore'
    label: ClassVar[str] = 'Explore an area'
    editors: ClassVar[tuple[str, ...]] = ('area',)
    reference_default: ClassVar[bool] = False
    #: The goal strategy a new explore profile starts with.
    default_strategy: ClassVar[str] = 'frontier'
    schema: ClassVar[ParamSchema] = ParamSchema((
        Field('coverage_target_pct', 'float', optional=True, minimum=0.0, maximum=100.0,
              unit='%', help='finish once this much of the area is observed; unset: '
                             'finish when the strategy runs out of goals'),
        Field('stuck_s', 'float', default=20.0, minimum=1.0, unit='s',
              help='how long without getting closer to the goal counts as stuck; a '
                   'stuck goal is blacklisted and another is proposed'),
        Field('unreachable_s', 'float', default=5.0, minimum=0.0, unit='s',
              help='how long dnav may refuse to plan to a goal before it is blacklisted'),
    ))

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.area: tuple[tuple[float, float], ...] = ()
        self.strategy: Any = None
        self.committed: tuple[float, float] | None = None
        self.reason = ''
        self.blacklist: list[tuple[float, float]] = []
        self.blacklist_reasons: list[str] = []
        self.proposals = 0
        #: (best distance to the committed goal, time it last improved)
        self.best: tuple[float, float] | None = None
        #: When dnav's refusal of the committed goal was first seen.
        self.refused_since: float | None = None
        #: What the last observation said, for the overlay and the status snapshot.
        self.frontiers: tuple[tuple[float, float], ...] = ()
        self.coverage_pct: float | None = None

    def start(self, params: dict[str, Any], obs: Observation, *, area: Sequence = (),
              waypoints: Sequence[tuple[float, float]] = (), strategy: Any = None) -> None:
        self.params = self.schema.validate(params)
        self.area = tuple((float(x), float(y)) for x, y in area)
        self.strategy = strategy
        self.committed, self.reason = None, ''
        self.blacklist, self.blacklist_reasons = [], []
        self.proposals = 0
        self.best, self.refused_since = None, None
        self.frontiers, self.coverage_pct = (), None

    def on_event(self, event) -> None:
        pass  # a click is a goto concern; the area is edited between missions

    # -- the body ------------------------------------------------

    def _give_up(self, xy: tuple[float, float], reason: str) -> None:
        self.blacklist.append((float(xy[0]), float(xy[1])))
        self.blacklist_reasons.append(reason)
        if self.committed is not None and math.dist(self.committed, xy) < 1e-9:
            self.committed = None
        self.best, self.refused_since = None, None

    def _propose(self, obs: Observation) -> Decision:
        """Ask the strategy for the next goal, or end the mission when there is none."""
        if self.strategy is None: return Done('failed', 'no goal strategy in the profile')
        for _ in range(PROPOSALS_PER_STEP):
            candidate = self.strategy.propose(obs, self.area, self.blacklist)
            if candidate is None: break
            if obs.coverage is not None and not G.contains(obs.coverage, candidate.xy):
                self._give_up(candidate.xy, 'outside coverage')
                continue
            self.committed = candidate.xy
            self.reason = candidate.reason
            self.proposals += 1
            self.best = None if obs.xy is None else (math.dist(obs.xy, self.committed), obs.time_s)
            self.refused_since = None
            return Goal(candidate.xy, candidate.reason, operator=False)
        else:
            return Wait(f'{PROPOSALS_PER_STEP} proposals were outside evidence coverage; asking again')
        dead = f' ({len(self.blacklist)} blacklisted)' if self.blacklist else ''
        if not obs.frontiers: return Done('complete', f'no frontier left to explore{dead}')
        return Done('complete', f'no reachable goal left: {getattr(self.strategy, "name", "the strategy")} '
                                f'proposes none{dead}')

    def _closing(self, obs: Observation) -> bool:
        """Whether the vehicle has closed ``PROGRESS_M`` on the goal since the anchor.

        The anchor is the distance at the last reset and stays there until real
        progress moves it. Measuring against the *previous step* instead --
        which this used to do, by pulling the anchor down every step -- asks
        for ``PROGRESS_M`` within one 10 Hz step, which is 2.5 m/s from a
        vehicle whose profile allows 1.0. Nothing ever counted as progress, so
        any leg longer than ``stuck_s`` was called stuck while it was still
        flying: in the 17 Sep area1 search run that gave up a goal the vehicle
        had closed 14 m on.
        """
        if obs.xy is None or self.committed is None or self.best is None: return True
        return math.dist(obs.xy, self.committed) < self.best[0] - PROGRESS_M

    def _stuck(self, obs: Observation) -> bool:
        """Whether the committed goal has stopped getting closer for ``stuck_s``."""
        if obs.xy is None or self.committed is None: return False
        if self.best is None or self._closing(obs):
            self.best = (math.dist(obs.xy, self.committed), obs.time_s)
            return False
        return obs.time_s - self.best[1] >= float(self.params['stuck_s'])

    def _refused(self, obs: Observation) -> str | None:
        """Why the committed goal should be given up to dnav's refusal, or None.

        The rules -- defer to a substitute dnav is offering, give up a verdict
        about the point at once, wait out anything transient -- are
        :func:`~dmsn.inputs.refusal_verdict`, shared with the search mission so
        both read the same.
        """
        reason, self.refused_since = refusal_verdict(obs, self.committed, self.refused_since,
                                                     self.params['unreachable_s'],
                                                     closing=self._closing(obs))
        return reason

    def step(self, obs: Observation) -> Decision:
        self.frontiers = tuple(obs.frontiers)
        self.coverage_pct = None if obs.area_coverage is None else obs.area_coverage * 100.0
        if not self.area:
            return Done('failed', 'no area in the profile')
        target = self.params.get('coverage_target_pct')
        if target is not None and self.coverage_pct is not None and self.coverage_pct >= target:
            return Done('complete', f'area coverage {self.coverage_pct:.0f}% reached '
                                    f'the {target:g}% target')
        if self.committed is None:
            return self._propose(obs)
        if obs.arrived(self.committed):
            self.committed = None
            return self._propose(obs)
        refused = self._refused(obs)
        if refused is not None:
            self._give_up(self.committed, f'dnav: {refused}')
            return self._propose(obs)
        if self._stuck(obs):
            self._give_up(self.committed, f'stuck for {float(self.params["stuck_s"]):g} s')
            return self._propose(obs)
        return Goal(self.committed, self.reason, operator=False)

    # -- what the window and the archive see --------------------------------

    def overlay(self) -> list[Shape]:
        shapes = []
        for x, y in self.blacklist:  # a dead end, crossed out
            r = 0.3
            shapes.append(Shape('line', x - r, y - r, x + r, y + r, outline=theme.DANGER, width=1.5))
            shapes.append(Shape('line', x - r, y + r, x + r, y - r, outline=theme.DANGER, width=1.5))
        for x, y in self.frontiers:
            r = 0.25
            shapes.append(Shape('oval', x - r, y - r, x + r, y + r, outline=theme.ACCENT, width=1.))
        if self.committed is not None:
            x, y = self.committed
            r = 0.5
            shapes.append(Shape('oval', x - r, y - r, x + r, y + r, outline=theme.OK, width=2.))
        if self.strategy is not None: shapes.extend(self.strategy.overlay())
        if self.area:
            cx = sum(p[0] for p in self.area) / len(self.area)
            cy = sum(p[1] for p in self.area) / len(self.area)
            coverage = 'unmeasured' if self.coverage_pct is None else f'{self.coverage_pct:.0f}% covered'
            shapes.append(Shape.label(cx, cy, coverage, outline=theme.DIM))
        return shapes

    def _blacklist_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = [dict(xy=[round(x, 3), round(y, 3)], reason=reason)
                for (x, y), reason in zip(self.blacklist, self.blacklist_reasons)]
        return rows if limit is None else rows[-limit:]

    def progress(self) -> dict[str, Any]:
        return dict(coverage_pct=None if self.coverage_pct is None else round(self.coverage_pct, 1),
                    frontier_count=len(self.frontiers),
                    blacklist_count=len(self.blacklist),
                    blacklist=self._blacklist_rows(BLACKLIST_SHOWN),
                    committed=None if self.committed is None else list(self.committed),
                    proposals=self.proposals,
                    strategy=None if self.strategy is None else getattr(self.strategy, 'name', ''))

    def summary(self) -> dict[str, Any]:
        return dict(area=[list(p) for p in self.area],
                    strategy=None if self.strategy is None else getattr(self.strategy, 'name', ''),
                    coverage_target=self.params.get('coverage_target_pct'),
                    coverage_pct=None if self.coverage_pct is None else round(self.coverage_pct, 1),
                    blacklist=self._blacklist_rows(),
                    proposals=self.proposals)
