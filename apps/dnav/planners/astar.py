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

from dcmn.clearance import ObstacleClearance, solid_cells
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
        prepared = self._prepare(cost_map, start_cell, policy, common)
        if isinstance(prepared, R.Route): return prepared
        cost_map, corridor, allows, escape_cells = prepared
        cost = cost_map.cost
        def position(cell):
            if cell == start_cell: return start[:2]
            if cell == goal_cell: return goal[:2]
            return geometry.cell_centre_m(*cell)
        def edge(a, b):
            return allows(position(a), position(b))
        if not math.isfinite(float(cost[goal_cell[1], goal_cell[0]])):
            return R.failed(R.GOAL_UNREACHABLE,
                            f'the goal is an obstacle or inside its '
                            f'{policy.inflation_m:g} m margin', **common)

        cells, expanded, boundary = self._search(cost, start_cell, goal_cell, allows=edge)
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
        # Retain exact start/goal positions when validating both search edges
        # and shortcuts. Substituting them after simplification changes geometry.
        points = [start[:2]] + [position(cell) for cell in R.simplify(cells[1:-1])] + [goal[:2]]
        points = R.shorten(cost, geometry, points, allows=allows)
        if (not all(allows(a, b) for a, b in zip(points, points[1:]))
                or not corridor.allows(points[-1], points[-1])):
            return R.failed(R.NO_ROUTE, 'no route with the required obstacle clearance', **common)
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
                               + ')'
                               if escape_cells else ''),
                       cost=R.route_cost(cost_map, points),
                       length_m=R.polyline_length_m(points),
                       plan_time_s=time.perf_counter() - started,
                       expanded=expanded, diagnostics=dict(
                           unknown_cells=int(unknown), path_cells=len(cells),
                           unknown_corridor=bool(unknown),
                           coverage_exhausted=bool(boundary), escaped_margin=bool(escape_cells),
                           escape_cells=escape_cells), **common)

    def plan_near(self, cost_map, start, goal, policy, radius_m) -> R.Route:
        """A route to the reachable point nearest ``goal``, within ``radius_m``.

        For a goal the planner cannot reach, typically one an errant obstacle
        cell sits on. The route ends at a cell centre that meets the same
        clearance as any route end; the goal itself is unchanged.
        """
        geometry = cost_map.geometry
        common = dict(planner=self.name, map_revision=cost_map.revision, policy_digest=policy.digest,
                      sim_time_s=cost_map.sim_time_s, goal=tuple(goal), start=tuple(start))
        start_cell = geometry.to_cell(start[0], start[1])
        if start_cell is None:
            return R.failed(R.START_BLOCKED, f'the vehicle at {start[0]:.2f}, {start[1]:.2f} m is '
                                             'outside the mapped area', **common)
        prepared = self._prepare(cost_map, start_cell, policy, common)
        if isinstance(prepared, R.Route): return prepared
        searched, corridor, allows, _ = prepared
        def position(cell):
            return start[:2] if cell == start_cell else geometry.cell_centre_m(*cell)
        reachable = self._reachable(searched.cost, start_cell, lambda a, b: allows(position(a), position(b)))
        span = math.ceil(radius_m / geometry.cell_m) + 1
        goal_col, goal_row = geometry.to_cell(goal[0], goal[1]) or start_cell
        candidates = []
        for row in range(max(0, goal_row - span), min(geometry.height, goal_row + span + 1)):
            for col in range(max(0, goal_col - span), min(geometry.width, goal_col + span + 1)):
                if not reachable[row, col] or (col, row) == start_cell: continue
                x, y = geometry.cell_centre_m(col, row)
                distance = math.dist((x, y), goal[:2])
                if distance <= radius_m and corridor.allows((x, y), (x, y)):
                    candidates.append((distance, math.dist((x, y), start[:2]), x, y))
        z = float(goal[2]) if len(goal) > 2 else float(start[2]) if len(start) > 2 else 0.0
        # The nearest few, so a candidate whose route fails its final checks
        # does not end the search.
        for distance, _, x, y in sorted(candidates)[:5]:
            route = self.plan(cost_map, start, (x, y, z), policy)
            if route.ok:
                return replace(route, diagnostics=dict(route.diagnostics, substitute=dict(
                    goal=[float(goal[0]), float(goal[1])], distance_m=round(distance, 3))))
        return R.failed(R.GOAL_UNREACHABLE, f'no reachable point within {radius_m:g} m of the goal', **common)

    def _prepare(self, cost_map, start_cell, policy, common):
        """The searchable cost map and clearance rule from the start, or why there is none.

        Returns ``(cost_map, corridor, allows, escape_cells)``, where a start
        inside an obstacle margin has had that margin made passable at high cost.
        """
        geometry, cost = cost_map.geometry, cost_map.cost
        escape_cells = 0
        obstacles = (np.logical_or.reduce([layer.obstacles for layer in cost_map.layers])
                     if cost_map.layers else np.zeros(cost.shape, bool))
        solid = solid_cells(obstacles, cost_map.never_observed) if cost_map.layers else obstacles
        corridor = ObstacleClearance(geometry, solid, policy.inflation_m, policy.body_radius_m)
        def allows(a, b):
            return corridor.allows(a, b, escape=policy.escape_margin)
        if not math.isfinite(float(cost[start_cell[1], start_cell[0]])):
            # An occupied start is never assumed to be a sensor error. Only
            # the non-obstacle margin may be escaped, along nondecreasing
            # clearance edges checked below.
            never = cost_map.never_observed if cost_map.layers else None
            col, row = start_cell
            refusal = ('' if cost_map.layers and never is not None else 'no obstacle layers to separate '
                       'obstacle from margin')
            if not refusal and obstacles[row, col]:
                refusal = 'its cell is observed occupied'
            if not refusal and never[row, col]:
                refusal = 'no source has observed its cell'
            if not refusal and not getattr(policy, 'escape_margin', True):
                refusal = 'margin escape is disabled by the cost policy'
            if refusal:
                return R.failed(R.START_BLOCKED,
                                f'the vehicle is standing inside an obstacle or its '
                                f'{policy.inflation_m:g} m margin ({refusal})', **common)
            region = self._margin_region(cost, solid, start_cell)
            cost = cost.copy()
            cost[region] = ESCAPE_COST
            escape_cells = int(region.sum())
            cost_map = replace(cost_map, cost=cost)
        return cost_map, corridor, allows, escape_cells

    @staticmethod
    def _reachable(cost: np.ndarray, start: tuple[int, int], allows) -> np.ndarray:
        """Cells an eight-connected route from ``start`` can reach over finite cost and allowed edges."""
        height, width = cost.shape
        finite = np.isfinite(cost)
        reached = np.zeros(cost.shape, bool)
        reached[start[1], start[0]] = True
        queue = deque([start])
        while queue:
            col, row = queue.popleft()
            for dcol in (-1, 0, 1):
                for drow in (-1, 0, 1):
                    ncol, nrow = col + dcol, row + drow
                    if (not (dcol or drow) or not (0 <= ncol < width and 0 <= nrow < height)
                            or reached[nrow, ncol] or not finite[nrow, ncol]
                            or not allows((col, row), (ncol, nrow))):
                        continue
                    reached[nrow, ncol] = True
                    queue.append((ncol, nrow))
        return reached

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
    def _search(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], *, allows=None):
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
                if allows is not None and not allows((col, row), (ncol, nrow)): continue
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
