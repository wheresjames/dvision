"""Wall following by goal stepping: the second strategy, proving the seam.

Rather than reasoning about frontiers, it keeps the nearest occupied evidence
on one hand and proposes a point ``step_m`` further along it (B7's 4 m
default). The observation carries the occupied points nearest the vehicle; the
tangent to the nearest one, turned so the wall stays on the chosen hand, is
the travel direction, nudged toward or away from the wall to keep
``standoff_m``. With no wall within ``reach_m`` it carries straight on to find
one. A step that leaves the area, lands on a wall or on the blacklist is turned
away from the wall hand, 45 degrees at a time -- the right-hand rule's "blocked
ahead, turn left". Everything else -- getting there, stopping when blocked --
is what dnav and dway already do.

It steps goals and never steers the vehicle, so it fits the pipeline like any
strategy; it is expected to explore poorly, and it ends only when boxed in or
when the mission's budgets end it.
"""
from __future__ import annotations

import math
from typing import ClassVar, Sequence

from dcmn import theme
from dcmn.mapview import Shape
from dmsn.inputs import Observation
from dmsn.schema import Field, ParamSchema
from dmsn.strategies import Candidate

#: A proposed point this close to a blacklisted one is the same dead end.
AVOID_BLACKLIST_M = 1.0
#: Turns tried, 45 degrees each, before the strategy reports it is boxed in.
TURNS = 8


def _direction(heading_deg: float) -> tuple[float, float]:
    """A unit travel vector for a compass heading (0 north, clockwise; x east, y south)."""
    radians = math.radians(heading_deg)
    return (math.sin(radians), -math.cos(radians))


def _heading(direction: tuple[float, float]) -> float:
    return math.degrees(math.atan2(direction[0], -direction[1])) % 360.


def _rotate(direction: tuple[float, float], degrees: float) -> tuple[float, float]:
    """Turn a direction clockwise on the map (x east, y south) by ``degrees``; negative turns left."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return (direction[0] * c - direction[1] * s, direction[0] * s + direction[1] * c)


class WallFollowStrategy:
    name: ClassVar[str] = 'wall_follow'
    schema: ClassVar[ParamSchema] = ParamSchema((
        Field('step_m', 'float', default=4.0, minimum=0.5, maximum=50.0, unit='m',
              help='how far along the wall each proposed goal is'),
        Field('hand', 'choice', default='right', choices=('right', 'left'),
              help='which side of the vehicle the followed wall stays on'),
        Field('standoff_m', 'float', default=1.5, minimum=0.25, maximum=10.0, unit='m',
              help='distance kept from the followed wall'),
        Field('reach_m', 'float', default=6.0, minimum=0.5, maximum=8.0, unit='m',
              help='a wall further than this is not followed; the strategy carries straight on to find one'),
    ))

    def __init__(self) -> None:
        self.params: dict = {}
        self.direction: tuple[float, float] | None = None
        #: The last proposal's wall point and goal, for the overlay.
        self.wall: tuple[float, float] | None = None
        self.target: tuple[float, float] | None = None

    def _param(self, name: str):
        return self.params.get(name, self.schema.field(name).default)

    def propose(self, obs: Observation, area: Sequence[tuple[float, float]],
                blacklist: Sequence[tuple[float, float]]) -> Candidate | None:
        if obs.xy is None or obs.pose is None: return None
        x, y = obs.xy
        step, standoff = float(self._param('step_m')), float(self._param('standoff_m'))
        right = self._param('hand') == 'right'
        if self.direction is None: self.direction = _direction(obs.pose[3])
        walls = [w for w in obs.walls if math.dist(w, (x, y)) <= float(self._param('reach_m'))]
        wall = min(walls, key=lambda w: (math.dist(w, (x, y)), w)) if walls else None
        offset = (0.0, 0.0)
        if wall is not None and math.dist(wall, (x, y)) > 1e-6:
            distance = math.dist(wall, (x, y))
            normal = ((wall[0] - x) / distance, (wall[1] - y) / distance)  # toward the wall
            # The wall on the right means travelling a quarter turn left of the normal.
            self.direction = _rotate(normal, -90. if right else 90.)
            pull = max(-step / 2., min(step / 2., distance - standoff))
            offset = (normal[0] * pull, normal[1] * pull)
            how = f'following the {self._param("hand")}-hand wall {distance:.1f} m away'
        else:
            wall = None
            how = 'no wall in reach; carrying on to find one'
        clearance = min(standoff, step) * 0.5
        for turn in range(TURNS):
            direction = _rotate(self.direction, (-45. if right else 45.) * turn)
            shift = offset if turn == 0 else (0.0, 0.0)
            target = (x + direction[0] * step + shift[0], y + direction[1] * step + shift[1])
            if area:
                from dmsn.geometry import point_in_polygon
                if not point_in_polygon(target, area): continue
            if blacklist and any(math.dist(target, b) <= AVOID_BLACKLIST_M for b in blacklist): continue
            if any(math.dist(target, w) < clearance for w in obs.walls): continue
            self.direction, self.wall, self.target = direction, wall, target
            turned = f', turned {45 * turn}° {"left" if right else "right"}' if turn else ''
            return Candidate(target, f'wall-follow: {how}{turned} (heading {_heading(direction):.0f}°)')
        self.wall, self.target = wall, None
        return None  # boxed in on every side: the mission decides what that means

    def overlay(self) -> list[Shape]:
        """The wall being followed and the side it is kept on."""
        if self.wall is None or self.target is None: return []
        (wx, wy), (tx, ty) = self.wall, self.target
        r = 0.3
        return [Shape('rect', wx - r, wy - r, wx + r, wy + r, outline=theme.WARN, width=1.5),
                Shape('line', wx, wy, tx, ty, outline=theme.WARN, width=1., dash=(2, 3)),
                Shape.label(wx, wy - 0.7, 'R' if self._param('hand') == 'right' else 'L', outline=theme.WARN)]
