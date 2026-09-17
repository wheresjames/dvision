"""The nearest frontier: the baseline goal strategy for area exploration.

A frontier is an observed free cell touching a never-observed one -- the edge
of what the vehicle knows. The runner distils them from the evidence plane into
the observation (capped, nearest first), so proposing is nothing but picking
the closest one the mission has not already given up on. No map, no grid, no
geometry: the strategy only ever sees what the observation carries (Q28).
"""
from __future__ import annotations

import math
from typing import ClassVar, Sequence

from dcmn.mapview import Shape
from dmsn.inputs import FRONTIER_EXCLUDE_M, Observation
from dmsn.schema import ParamSchema
from dmsn.strategies import Candidate

#: A frontier this close to a blacklisted point is the same dead end.
AVOID_BLACKLIST_M = FRONTIER_EXCLUDE_M


class FrontierStrategy:
    name: ClassVar[str] = 'frontier'
    schema: ClassVar[ParamSchema] = ParamSchema(())

    def __init__(self) -> None:
        self.params: dict = {}

    def propose(self, obs: Observation, area: Sequence[tuple[float, float]],
                blacklist: Sequence[tuple[float, float]]) -> Candidate | None:
        points = [tuple(p) for p in obs.frontiers]
        if area:
            from dmsn.geometry import point_in_polygon
            points = [p for p in points if point_in_polygon(p, area)]
        if blacklist:
            points = [p for p in points
                      if all(math.dist(p, b) > AVOID_BLACKLIST_M for b in blacklist)]
        if not points: return None
        if obs.xy is None: return Candidate(points[0], 'nearest frontier (no pose yet)')
        xy = min(points, key=lambda p: (math.dist(p, obs.xy), p[0], p[1]))
        return Candidate(xy, f'nearest frontier {xy[0]:.1f}, {xy[1]:.1f} m '
                             f'({len(points)} ahead)')

    def overlay(self) -> list[Shape]:
        return []  # the mission draws the frontiers it was told about
