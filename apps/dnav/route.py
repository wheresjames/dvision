"""What a plan is, what it says when there is no plan, and what one costs.

Every route in this module carries the map revision and the cost-policy digest
it was planned against. A route without them is a route nobody can reproduce,
and reproducing a plan is the whole reason the policy is an artifact rather
than a constant.

The status vocabulary is small and its distinctions are deliberate. A failure
is a message, never silence: dnav publishes and displays the reason rather than
showing an empty map and letting the operator guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

#: A route was found.
OK = 'ok'
#: Start and goal are both usable, but no path connects them.
NO_ROUTE = 'no_route'
#: The goal itself cannot be occupied -- an obstacle, inflated ground, or off
#: the grid. Distinct from ``no_route`` because the fix is a different goal
#: rather than a different map.
GOAL_UNREACHABLE = 'goal_unreachable'
#: The vehicle is standing somewhere the policy calls blocked. Distinct again:
#: nothing about the goal is wrong.
START_BLOCKED = 'start_blocked'
#: The evidence is too old to plan on, or its producer is gone. Which of the
#: two is a data fact and which a liveness fact, and the reason says so.
STALE_MAP = 'stale_map'
#: No goal has been set yet. Not a failure -- an invitation.
NO_GOAL = 'no_goal'
#: The vehicle pose is unavailable, invalid or older than the freshness limit
#: in the data clock. A route planned from it would start somewhere else.
STALE_POSE = 'stale_pose'
#: Goal, pose and evidence disagree about frame or localization/clock epoch.
#: Numbers from an invalidated frame are never reused silently; the goal's
#: authority must reissue it.
FRAME_MISMATCH = 'frame_mismatch'
#: The goal (or the vehicle) lies outside the evidence grid's fixed coverage.
#: Coverage is a resource allocation: this proves nothing about the world.
OUTSIDE_COVERAGE = 'outside_coverage'

STATUSES = (OK, NO_ROUTE, GOAL_UNREACHABLE, START_BLOCKED, STALE_MAP, NO_GOAL,
            STALE_POSE, FRAME_MISMATCH, OUTSIDE_COVERAGE)


@dataclass(frozen=True)
class Waypoint:
    """One point of a route, with the speed a follower may use to reach it.

    The speed is optional and every viewer here ignores it. It is carried
    because a tour's waypoint list has the same shape, so an executor that
    later follows a route needs no translation -- and adding the field
    afterwards would have meant changing a published format.
    """

    x_m: float
    y_m: float
    z_m: float
    speed_mps: float | None = None

    def as_dict(self) -> dict[str, Any]:
        value = {'x': self.x_m, 'y': self.y_m, 'z': self.z_m}
        if self.speed_mps is not None: value['speed_mps'] = self.speed_mps
        return value


@dataclass(frozen=True)
class Route:
    """One planning answer, successful or not."""

    status: str
    reason: str = ''
    waypoints: tuple[Waypoint, ...] = ()
    cost: float = math.inf
    length_m: float = 0.0
    planner: str = ''
    map_revision: int = 0
    policy_digest: str = ''
    plan_time_s: float = 0.0
    expanded: int = 0
    sim_time_s: float = 0.0
    goal: tuple[float, float, float] | None = None
    start: tuple[float, float, float] | None = None
    #: What the search saw: whether it reached the coverage boundary, how many
    #: never-observed cells the route crosses. Never a claim of clearance.
    diagnostics: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f'unknown route status {self.status!r}')
        if self.status != OK and self.waypoints:
            raise ValueError(f'a {self.status} route may not carry waypoints')
        if self.status != OK and not self.reason:
            raise ValueError(f'a {self.status} route must say why')

    @property
    def ok(self) -> bool: return self.status == OK

    @property
    def points(self) -> list[tuple[float, float]]:
        """The route as map-frame ``(x, y)`` pairs, for drawing and costing."""
        return [(w.x_m, w.y_m) for w in self.waypoints]

    def as_dict(self) -> dict[str, Any]:
        return dict(status=self.status, reason=self.reason, planner=self.planner,
                    waypoints=[w.as_dict() for w in self.waypoints],
                    cost=(None if math.isinf(self.cost) else round(self.cost, 6)),
                    length_m=round(self.length_m, 4),
                    map_revision=self.map_revision, policy_digest=self.policy_digest,
                    plan_time_s=round(self.plan_time_s, 6), expanded=self.expanded,
                    sim_time_s=round(self.sim_time_s, 4),
                    goal=list(self.goal) if self.goal else None,
                    start=list(self.start) if self.start else None,
                    diagnostics=dict(self.diagnostics))


def failed(status: str, reason: str, **fields: Any) -> Route:
    """A route that is not one, which still has to say what happened."""
    return Route(status=status, reason=reason, **fields)


# -- costing -----------------------------------------------------------------

#: Whether a step between two cells is diagonal, as a length in cells.
STRAIGHT, DIAGONAL = 1.0, math.sqrt(2.0)


def line_cells(geometry, a: tuple[float, float], b: tuple[float, float]
               ) -> list[tuple[int, int]]:
    """Bresenham from ``a`` to ``b`` in map metres, as ``(col, row)`` cells.

    Eight-connected and one cell per step, which is exactly the move set A\\*
    searches over. That is what makes the two comparable: a straight line
    measured this way is a path A\\* could have returned, so A\\* costing more
    than it would be a bug in the search rather than a difference of units.
    Cells outside the grid are dropped, not clamped -- clamping would fold a
    route that leaves the map onto its edge and price it as though it had not.
    """
    x0, y0 = _cell_of(geometry, a)
    x1, y1 = _cell_of(geometry, b)
    cells: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx - dy
    while True:
        if 0 <= x0 < geometry.width and 0 <= y0 < geometry.height:
            cells.append((x0, y0))
        if (x0, y0) == (x1, y1): break
        doubled = 2 * error
        # A step that satisfies both tests is the diagonal one, which is what
        # keeps this on the eight-connected move set instead of taking two
        # orthogonal steps around a corner A* would have cut.
        step_x = doubled > -dy
        step_y = doubled < dx
        if step_x: error -= dy; x0 += sx
        if step_y: error += dx; y0 += sy
    return cells


def _cell_of(geometry, point: tuple[float, float]) -> tuple[int, int]:
    """A point in map metres as integer cell indices, unclamped."""
    return (int(np.floor((point[0] - geometry.origin_x_m) / geometry.cell_m)),
            int(np.floor((point[1] - geometry.origin_y_m) / geometry.cell_m)))


def path_cost(cost: np.ndarray, cells: Sequence[tuple[int, int]]) -> float:
    """What a cell path costs: each step's length times the cell it enters.

    The first cell is free because the vehicle is already standing in it. This
    is the same accounting A\\* accumulates as its ``g``, and every route in
    dnav is priced through this one function so that two planners' numbers can
    be compared at all.
    """
    total = 0.0
    for (col, row), (previous_col, previous_row) in zip(cells[1:], cells):
        step = DIAGONAL if col != previous_col and row != previous_row else STRAIGHT
        value = float(cost[row, col])
        if math.isinf(value): return math.inf
        total += step * value
    return total


def polyline_cells(geometry, points: Sequence[tuple[float, float]]
                   ) -> list[tuple[int, int]]:
    """One continuous cell path through a polyline, without doubling its joins."""
    cells: list[tuple[int, int]] = []
    for start, end in zip(points, points[1:]):
        segment = line_cells(geometry, start, end)
        cells.extend(segment[1:] if cells else segment)
    return cells


def route_cost(cost_map, points: Iterable[tuple[float, float]]) -> float:
    """Price an arbitrary polyline in map metres against a cost map."""
    points = list(points)
    if len(points) < 2: return 0.0
    return path_cost(cost_map.cost, polyline_cells(cost_map.geometry, points))


def polyline_length_m(points: Iterable[tuple[float, float]]) -> float:
    points = list(points)
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def simplify(cells: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop cells that only continue a straight run.

    A cell-by-cell path is a hundred waypoints where three would do, and every
    one of them is a point a follower would slow down for. Removing collinear
    interior cells changes no geometry at all: re-walking the result reproduces
    the identical cell path, which is what lets the cost be quoted against
    either one.
    """
    if len(cells) < 3: return list(cells)
    kept = [cells[0]]
    for previous, current, following in zip(cells, cells[1:], cells[2:]):
        before = (current[0] - previous[0], current[1] - previous[1])
        after = (following[0] - current[0], following[1] - current[1])
        if before != after: kept.append(current)
    kept.append(cells[-1])
    return kept


def shorten(cost: np.ndarray, geometry, points: Sequence[tuple[float, float]], *, allows=None
            ) -> list[tuple[float, float]]:
    """Join waypoints whose direct segment costs no more than the path it replaces.

    An eight-connected search can only turn in forty-five degree steps, so on
    open ground its route is a staircase several percent longer than the
    straight line it is imitating -- which makes "how much longer than the
    control route" read as a detour when it is only quantisation. Pulling the
    string taut removes that, and the acceptance test is what keeps it honest:
    a join is taken only when the direct segment costs no *more* than the
    corner it replaces, so the route can only get cheaper and the planner stays
    no worse than the control it is measured against.
    """
    points = list(points)
    if len(points) < 3: return points
    kept = [points[0]]
    index = 0
    while index < len(points) - 1:
        target = index + 1
        for candidate in range(len(points) - 1, index + 1, -1):
            direct = path_cost(cost, line_cells(geometry, points[index], points[candidate]))
            through = path_cost(cost, polyline_cells(geometry, points[index:candidate + 1]))
            if math.isfinite(direct) and direct <= through + 1e-9 and (allows is None or allows(points[index], points[candidate])):
                target = candidate
                break
        kept.append(points[target])
        index = target
    return kept


@dataclass
class RouteHistory:
    """Every status this planner has published, and when it changed.

    A report that only holds the last route says nothing about a run that spent
    most of it stale and recovered at the end.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)
    last: Route | None = None

    def observe(self, route: Route) -> bool:
        """Record a route; returns whether the status or reason changed."""
        changed = (self.last is None or route.status != self.last.status
                   or route.reason != self.last.reason)
        if changed:
            self.entries.append(dict(sim_time_s=round(route.sim_time_s, 4),
                                     status=route.status, reason=route.reason,
                                     planner=route.planner,
                                     map_revision=route.map_revision))
        self.last = route
        return changed

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry['status']] = counts.get(entry['status'], 0) + 1
        return counts
