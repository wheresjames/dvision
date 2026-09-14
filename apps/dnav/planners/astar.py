"""A\\* over the combined cost map: the cheapest route, or the reason there is none.

Extracted from `daic`'s prototype rather than inherited from it. What is kept
is the shape that was right -- eight-connected moves, a cost charged per cell
entered, an infinite cost for a cell no route may occupy. What is fixed is the
part that was subtly wrong there: the heuristic has to be measured in the same
units the cost function charges, or A\\* stops being admissible and settles for
the first route it finds instead of the cheapest one. Here the heuristic is the
octile distance times the cheapest cell in the policy, which can never
overestimate, and a test holds it to that against the control route.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import heapq
import math
import time

import numpy as np

from dnav import route as R
from dnav.policy import FREE_COST, UNOBSERVED_COST

#: Per-cell cost of leaving an obstacle margin the vehicle already stands in:
#: allowed, but never preferred over open or unknown space.
ESCAPE_COST = 10.0 * max(FREE_COST, UNOBSERVED_COST)


class AStarPlanner:
    """The default planner: optimal on the grid it is given."""

    name = 'astar'

    def plan(self, cost_map, start, goal, policy) -> R.Route:
        started = time.perf_counter()
        geometry = cost_map.geometry
        common = dict(planner=self.name, map_revision=cost_map.revision,
                      policy_digest=policy.digest, sim_time_s=cost_map.sim_time_s,
                      goal=tuple(goal), start=tuple(start))
        start_cell = geometry.to_cell(start[0], start[1])
        goal_cell = geometry.to_cell(goal[0], goal[1])
        if start_cell is None:
            return R.failed(R.START_BLOCKED,
                            f'the vehicle at {start[0]:.2f}, {start[1]:.2f} m is '
                            'outside the mapped area', **common)
        if goal_cell is None:
            return R.failed(R.GOAL_UNREACHABLE,
                            f'the goal at {goal[0]:.2f}, {goal[1]:.2f} m is '
                            'outside the mapped area', **common)
        cost = cost_map.cost
        escape_cells, false_obstacle = 0, []
        if not math.isfinite(float(cost[start_cell[1], start_cell[0]])):
            obstacles = (np.logical_or.reduce([layer.obstacles for layer in cost_map.layers])
                         if cost_map.layers else None)
            # Only a start some source has observed may escape: an unobserved
            # cell enclosed by walls is as likely to be wall as margin. An
            # *observed* start cell that a source marks occupied is where the
            # vehicle physically is, so that mark is a false obstacle (seen live:
            # optical-flow triangulation marking the drone's own cell).
            never = cost_map.never_observed if cost_map.layers else None
            col, row = start_cell
            refusal = ('' if obstacles is not None and never is not None else 'no obstacle layers to separate '
                       'obstacle from margin')
            if not refusal and never[row, col]:
                refusal = 'no source has observed its cell'
            if not refusal and not getattr(policy, 'escape_margin', True):
                refusal = 'margin escape is disabled by the cost policy'
            if refusal:
                return R.failed(R.START_BLOCKED,
                                f'the vehicle is standing inside an obstacle or its '
                                f'{policy.inflation_m:g} m margin ({refusal})', **common)
            false_obstacle = [layer.source for layer in cost_map.layers if layer.obstacles[row, col]]
            region = self._margin_region(cost, obstacles, start_cell)
            cost = cost.copy()
            cost[region] = ESCAPE_COST
            escape_cells = int(region.sum())
            cost_map = replace(cost_map, cost=cost)
        if not math.isfinite(float(cost[goal_cell[1], goal_cell[0]])):
            return R.failed(R.GOAL_UNREACHABLE,
                            f'the goal is an obstacle or inside its '
                            f'{policy.inflation_m:g} m margin', **common)

        cells, expanded, boundary = self._search(cost, start_cell, goal_cell)
        if not cells:
            elapsed = time.perf_counter() - started
            # Bounded search: failing inside fixed coverage says nothing about
            # whether a physical route exists beyond it.
            reason = ('no route within current evidence; the search reached the coverage '
                      'boundary (coverage exhausted), so a detour outside coverage is not excluded'
                      if boundary else
                      'no route within current evidence: observed obstacles enclose the '
                      'vehicle or the goal inside the covered area')
            return R.failed(R.NO_ROUTE, reason, plan_time_s=elapsed, expanded=expanded,
                            diagnostics=dict(coverage_exhausted=bool(boundary)), **common)
        # The route runs between the two given points, not between the centres
        # of the cells holding them: a vehicle already standing somewhere does
        # not fly to the middle of its own cell first.
        points = [(start[0], start[1])]
        points += [geometry.cell_centre_m(col, row) for col, row in R.simplify(cells)][1:-1]
        points.append((goal[0], goal[1]))
        points = R.shorten(cost, geometry, points)
        z = float(start[2]) if len(start) > 2 else 0.0
        waypoints = tuple(R.Waypoint(x, y, z) for x, y in points)
        # A proposal through unknown cells is a hypothesis, not a verified
        # corridor; say how much of it is unknown.
        unknown = 0
        if cost_map.never_observed is not None:
            never = cost_map.never_observed
            unknown = sum(1 for col, row in cells if never[row, col])
        # Priced as published: the cost of the polyline the route actually
        # carries, so the number in the report is the number a follower would
        # pay and the comparison with the control route is like for like.
        return R.Route(status=R.OK, waypoints=waypoints,
                       reason=(f'escaping the {policy.inflation_m:g} m obstacle margin the vehicle stands in '
                               f'({escape_cells} margin cells passable at high cost'
                               + (f"; its own cell marked occupied by {', '.join(false_obstacle)} is treated "
                                  "as a false obstacle" if false_obstacle else '') + ')'
                               if escape_cells else ''),
                       cost=R.route_cost(cost_map, points),
                       length_m=R.polyline_length_m(points),
                       plan_time_s=time.perf_counter() - started,
                       expanded=expanded, diagnostics=dict(
                           unknown_cells=int(unknown), path_cells=len(cells),
                           unknown_corridor=bool(unknown),
                           coverage_exhausted=bool(boundary), escaped_margin=bool(escape_cells),
                           escape_cells=escape_cells), **common)

    @staticmethod
    def _margin_region(cost: np.ndarray, obstacles: np.ndarray, start: tuple[int, int]) -> np.ndarray:
        """The connected margin cells (blocked, but not obstacles) around the start."""
        passable = ~np.isfinite(cost) & ~obstacles
        region = np.zeros(cost.shape, bool)
        height, width = cost.shape
        queue = deque([(start[1], start[0])])
        region[start[1], start[0]] = True
        while queue:
            row, col = queue.popleft()
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    nrow, ncol = row + drow, col + dcol
                    if (0 <= nrow < height and 0 <= ncol < width and passable[nrow, ncol]
                            and not region[nrow, ncol]):
                        region[nrow, ncol] = True
                        queue.append((nrow, ncol))
        return region

    @staticmethod
    def _search(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
        """Cheapest eight-connected cell path, and how many cells were expanded.

        Flat indices and arrays rather than dictionaries of tuples: this runs
        once per evidence revision on a grid that may be forty thousand cells,
        and the difference is the whole plan budget.
        """
        height, width = cost.shape
        flat = np.asarray(cost, np.float64).ravel()
        start_index = start[1] * width + start[0]
        goal_index = goal[1] * width + goal[0]
        goal_col, goal_row = goal

        best = np.full(flat.size, math.inf)
        came = np.full(flat.size, -1, np.int64)
        closed = np.zeros(flat.size, bool)
        best[start_index] = 0.0
        heap = [(0.0, start_index)]
        expanded = 0
        while heap:
            _, index = heapq.heappop(heap)
            if closed[index]: continue
            closed[index] = True
            expanded += 1
            if index == goal_index: break
            row, col = divmod(index, width)
            here = best[index]
            for dcol, drow in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ncol, nrow = col + dcol, row + drow
                if not (0 <= ncol < width and 0 <= nrow < height): continue
                neighbour = nrow * width + ncol
                if closed[neighbour]: continue
                value = flat[neighbour]
                if not math.isfinite(value): continue
                step = R.DIAGONAL if dcol and drow else R.STRAIGHT
                tentative = here + step * value
                if tentative >= best[neighbour]: continue
                best[neighbour] = tentative
                came[neighbour] = index
                # Octile distance times the cheapest cell anything can cost:
                # never an overestimate, so the first route to the goal is the
                # cheapest one.
                dx, dy = abs(ncol - goal_col), abs(nrow - goal_row)
                heuristic = FREE_COST * (max(dx, dy) + (R.DIAGONAL - 1.0) * min(dx, dy))
                heapq.heappush(heap, (tentative + heuristic, neighbour))
        edges = closed.reshape(height, width)
        boundary = bool(edges[0].any() or edges[-1].any() or edges[:, 0].any() or edges[:, -1].any())
        if not closed[goal_index]: return [], expanded, boundary
        cells: list[tuple[int, int]] = []
        index = goal_index
        while index != -1:
            row, col = divmod(index, width)
            cells.append((col, row))
            if index == start_index: break
            index = int(came[index])
        cells.reverse()
        return cells, expanded, boundary
