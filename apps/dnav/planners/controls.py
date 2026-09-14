"""The control planner: a straight line that ignores cost.

Every measured thing in this repository is bracketed by a control -- `dalg` has
`constant` and `exact_range` for exactly this reason -- and a route is no
different. The straight line is the floor: the shortest path that could
possibly exist between two points, priced against the same cost map as the real
plan. A route from A\\* that costs more than this one has a bug in its search;
a route that costs a great deal less is a route that found a real detour, and
the ratio is the number that says whether the plan was worth computing.

It ignores cost on purpose, which is why it does not report `start_blocked` or
`goal_unreachable` for a cell the policy calls blocked: a control that declined
to answer whenever the answer was bad would stop being a baseline exactly when
one was needed. It priced its line at infinity instead, and infinity is a
perfectly good thing to be better than.
"""

from __future__ import annotations

import time

from dnav import route as R


class ControlPlanner:
    """The straight-line baseline every real route is judged against."""

    name = 'control'

    def plan(self, cost_map, start, goal, policy) -> R.Route:
        started = time.perf_counter()
        geometry = cost_map.geometry
        common = dict(planner=self.name, map_revision=cost_map.revision,
                      policy_digest=policy.digest, sim_time_s=cost_map.sim_time_s,
                      goal=tuple(goal), start=tuple(start))
        if geometry.to_cell(start[0], start[1]) is None:
            return R.failed(R.START_BLOCKED,
                            f'the vehicle at {start[0]:.2f}, {start[1]:.2f} m is '
                            'outside the mapped area', **common)
        if geometry.to_cell(goal[0], goal[1]) is None:
            return R.failed(R.GOAL_UNREACHABLE,
                            f'the goal at {goal[0]:.2f}, {goal[1]:.2f} m is '
                            'outside the mapped area', **common)
        points = [(start[0], start[1]), (goal[0], goal[1])]
        z = float(start[2]) if len(start) > 2 else 0.0
        cells = R.line_cells(geometry, *points)
        return R.Route(
            status=R.OK,
            waypoints=tuple(R.Waypoint(x, y, z) for x, y in points),
            cost=R.path_cost(cost_map.cost, cells),
            length_m=R.polyline_length_m(points),
            plan_time_s=time.perf_counter() - started,
            expanded=len(cells), **common)
