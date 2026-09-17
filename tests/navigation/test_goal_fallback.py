"""Keep flying when the goal cannot be reached: the retained route prefix, then a substitute goal."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from dcmn.maps import EvidenceGrid, GridGeometry
from dcmn.navigation import ExecutionProfile, point_at, validate
from dnav import route as R
from dnav.clearance import Clearance
from dnav.execution import RoutePublisher
from dnav.planners.astar import AStarPlanner
from dnav.policy import CostPolicy, build_cost_map

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT/'assets/goal_fallback'
PROFILE = ExecutionProfile.load('sim-target')
POLICY = CostPolicy().for_execution(PROFILE)


def open_grid(*cells, time_s=10.):
    """A fully observed 10 m square with occupied ``(col, row)`` cells of 0.5 m."""
    geometry = GridGeometry.from_extent(10, 10, .5)
    occupancy = np.zeros(geometry.shape, np.uint8)
    for col, row in cells: occupancy[0, row, col] = 254
    return EvidenceGrid(geometry, occupancy, np.full(geometry.shape, int(time_s*1000), np.uint32), 'scan', 1, time_s)


def truth_clearance(points, walls):
    return min(math.hypot(max(x0-x, x-x1, 0), max(y0-y, y-y1, 0))
               for a, b in zip(points, points[1:]) for t in np.linspace(0, 1, 201)
               for x, y in [np.asarray(a[:2]) + t*(np.asarray(b[:2])-np.asarray(a[:2]))]
               for x0, x1, y0, y1 in walls)


def test_plan_permission_prefix_stops_short_of_an_obstacle_instead_of_withdrawing():
    grids = {'scan': open_grid((16, 10))}  # x in [8, 8.5], y in [5, 5.5]
    points = [[1., 5.25, 1.5], [9.5, 5.25, 1.5]]
    validator = Clearance(PROFILE)
    assert not validator.check(points, points[0], grids, 10.)['eligible']
    c = validator.check(points, points[0], grids, 10., prefix=True)
    assert c['eligible'] and not c['reaches_goal'], c['reason']
    end = point_at(points, c['end'])
    assert 8. - PROFILE.clearance_radius_m - .5 < end[0] <= 8. - PROFILE.clearance_radius_m
    assert c['distance_m'] == pytest.approx(end[0] - 1.) and len(c['lateral_m']) == 1
    # From just short of the obstacle nothing worth flying is left.
    near = [end[0] - .1, 5.25, 1.5]
    left = validator.check(points, near, grids, 10., start=(0, (near[0]-1.)/8.5), prefix=True)
    assert not left['eligible'] and 'stopping distance' in left['reason']


def test_only_a_fallback_route_may_be_eligible_while_planning_fails():
    value = json.loads((ROOT/'assets/navigation_examples/ready.json').read_text())
    value['planning_status'] = 'goal_unreachable'
    with pytest.raises(ValueError, match='eligible route missing'): validate(value)
    validate(dict(value, route_mode='retained'))
    with pytest.raises(ValueError, match='substitute'): validate(dict(value, route_mode='substitute'))
    validate(dict(value, route_mode='substitute', substitute_goal=[7., 2., 1.5]))
    with pytest.raises(ValueError, match='route mode'): validate(dict(value, route_mode='guess'))


def test_plan_near_routes_to_the_reachable_point_nearest_a_blocked_goal():
    grids = {'scan': open_grid((16, 10))}
    cost = build_cost_map(grids, POLICY)
    start, goal = [1., 5.25, 1.5], [8.25, 5.25, 1.5]
    assert AStarPlanner().plan(cost, start, goal, POLICY).status == R.GOAL_UNREACHABLE
    route = AStarPlanner().plan_near(cost, start, goal, POLICY, PROFILE.goal_substitute_m)
    assert route.ok, route.reason
    end = (route.waypoints[-1].x_m, route.waypoints[-1].y_m)
    assert math.dist(end, goal[:2]) <= PROFILE.goal_substitute_m
    assert route.diagnostics['substitute']['distance_m'] == pytest.approx(math.dist(end, goal[:2]), abs=1e-3)
    points = [[w.x_m, w.y_m, w.z_m] for w in route.waypoints]
    assert Clearance(PROFILE).check(points, start, grids, 10.)['eligible']
    # A goal walled in, with nothing reachable inside the radius, has no substitute.
    ring = [(c, r) for c in range(12, 20) for r in range(6, 15) if c in (12, 19) or r in (6, 14)]
    walled = build_cost_map({'scan': open_grid(*ring)}, POLICY)
    assert AStarPlanner().plan_near(walled, start, [7.75, 5.25, 1.5], POLICY, .5).status == R.GOAL_UNREACHABLE


def test_area1_phantom_goal_keeps_the_route_and_finds_a_point_near_the_goal():
    record = json.loads((FIXTURE/'area1-phantom-goal.json').read_text())
    geometry = GridGeometry.from_dict(record['geometry'])
    with np.load(FIXTURE/'area1-phantom-goal.npz', allow_pickle=False) as data:
        g = EvidenceGrid(geometry, data['occupancy'][None], data['observed_ms'][None],
                         record['source'], 1, record['time_s'])
    grids, now = {g.source: g}, record['time_s']
    pose, goal, points, start = record['pose'], record['goal'], record['points'], tuple(record['progress'])
    cost = build_cost_map(grids, POLICY)
    assert AStarPlanner().plan(cost, pose, goal, POLICY).status == R.GOAL_UNREACHABLE
    validator = Clearance(PROFILE)
    assert not validator.check(points, pose, grids, now, start=start)['eligible']
    kept = validator.check(points, pose, grids, now, start=start, prefix=True)
    assert kept['eligible'] and not kept['reaches_goal'] and kept['distance_m'] > 14., kept['reason']
    permitted = [pose] + points[start[0]+1:kept['end'][0]+1] + [point_at(points, kept['end'])]
    assert math.dist(permitted[-1][:2], goal[:2]) < 2.5
    near = AStarPlanner().plan_near(cost, pose, goal, POLICY, PROFILE.goal_substitute_m)
    assert near.ok, near.reason
    substitute = [[w.x_m, w.y_m, w.z_m] for w in near.waypoints]
    assert math.dist(substitute[-1][:2], goal[:2]) < 1.
    assert validator.check(substitute, pose, grids, now)['eligible']
    # Independent truth check against maze_020's walls near the top corridor.
    walls = [(0, 67, 0, 1), (31, 54, 4, 5), (53, 54, 4, 13), (7, 22, 4, 5)]
    for route in (permitted, substitute):
        assert truth_clearance(route, walls) > PROFILE.body_radius_m + PROFILE.tracking_m


class StubRun:
    """The NavRun surface ``RoutePublisher`` reads."""

    def __init__(self, grids, goal):
        self.snapshot = dict(provider_id='p', frame_id='local', localization_epoch=0,
                             clock_domain_id='v', clock_epoch=0)
        self.goal_descriptor = dict(authority='ui', authority_epoch=1, revision=1, frame_id='local',
                                    localization_epoch=0, clock_epoch=0, position=goal)
        self.session = SimpleNamespace(identity='stub', context={'mapping_epoch': 0})
        self.grids, self.attempts, self.recorder = grids, 0, None
        self.pose = [1., 5.25, 1.5]
        self.route = R.failed(R.NO_GOAL, 'no goal')
        self.near = None
        self.notes = []

    def sim_time_s(self): return 10.
    def _grids(self): return dict(self.grids)
    def _pose(self): return dict(x_m=self.pose[0], y_m=self.pose[1], z_m=self.pose[2])
    def substitute_route(self): return self.near
    def note(self, kind, **fields): self.notes.append((kind, fields))


def planned(*points):
    return R.Route(status=R.OK, waypoints=tuple(R.Waypoint(x, y, 1.5) for x, y in points))


def test_publisher_retains_then_substitutes_then_returns_to_the_goal():
    goal = [9.5, 5.25, 1.5]
    run = StubRun({'scan': open_grid()}, goal)
    publisher = RoutePublisher('fb-' + uuid.uuid4().hex[:8], PROFILE)
    executor = {}
    publisher._read_executor = lambda wall: executor
    try:
        run.route = planned((1., 5.25), (9.5, 5.25))
        first = publisher.publish(run, wall=0.)
        assert first['route_mode'] == 'planned' and first['clearance']['eligible'], first['clearance']['reason']
        # Moving along it when an obstacle appears on the goal and replanning fails.
        run.grids = {'scan': open_grid((16, 10), (18, 10))}
        run.route = R.failed(R.GOAL_UNREACHABLE, 'the goal is an obstacle or inside its 0.35 m margin')
        run.pose = [2.5, 5.25, 1.5]
        executor.update(state='EXECUTING', geometry_revision=first['geometry_revision'], progress=[0, 1.5/8.5])
        kept = publisher.publish(run, wall=0.)
        assert kept['route_mode'] == 'retained' and kept['points'] == first['points']
        assert kept['clearance']['eligible'] and not kept['clearance']['reaches_goal'], kept['clearance']['reason']
        assert kept['stop_generation'] == first['stop_generation']
        assert 'goal is an obstacle' in kept['fallback_reason']
        # Stopped and released where that permission ends: a substitute route near the goal.
        run.pose = [7.4, 5.25, 1.5]
        executor.update(state='HOLDING', progress=[0, 6.4/8.5],
                        disposition=dict(value='stopped', geometry_revision=kept['geometry_revision']))
        run.near = AStarPlanner().plan_near(build_cost_map(run.grids, POLICY), run.pose, goal, POLICY,
                                            PROFILE.goal_substitute_m)
        near = publisher.publish(run, wall=0.)
        assert near['route_mode'] == 'substitute' and near['clearance']['eligible'], near['clearance']['reason']
        assert near['substitute_goal'] == near['points'][-1]
        assert math.dist(near['substitute_goal'][:2], goal[:2]) <= PROFILE.goal_substitute_m
        assert near['stop_generation'] == kept['stop_generation']  # already stopped
        # The obstacle clears while the substitute is flown: back to the goal, from a confirmed stop.
        run.grids = {'scan': open_grid()}
        run.route = planned(run.pose[:2], goal[:2])
        executor.update(state='EXECUTING', geometry_revision=near['geometry_revision'], progress=[0, .1],
                        disposition=dict(value='active', geometry_revision=near['geometry_revision']))
        back = publisher.publish(run, wall=0.)
        assert back['route_mode'] == 'planned' and back['points'] != near['points']
        assert back['stop_generation'] == near['stop_generation'] + 1
        assert not back['clearance']['eligible'] and 'confirm stop' in back['clearance']['reason']
        assert [(kind, fields['mode']) for kind, fields in run.notes] == [
            ('navigation.mode', 'retained'), ('navigation.mode', 'substitute'), ('navigation.mode', 'planned')]
    finally:
        publisher.close()


# -- the substitute has to be cheap enough to still be valid when it arrives --
#
# The gate here is a clock. dnav computes the substitute inside its publisher,
# so whatever it costs is added to the age of every route it offers, and dway
# rejects a route older than the profile's max_age_s (3 s). In the 17 Sep area1
# run the reachability search took 2.8 s on a 200x200 grid: 49 substitutes were
# offered and all 49 were rejected as stale, and dnav's publication rate fell
# from 10 Hz to one snapshot every 2.9 s while it ran.

def big_grid(*, cells=200, cell_m=.5, time_s=10.):
    """A fully observed 100 m arena of pillars: 40,000 cells, all of it connected.

    Pillars rather than walls, because a one-cell door does not survive the
    policy's 0.35 m margin -- it blocks the door's own cell -- and a grid of
    sealed corridors would measure the wrong thing.
    """
    geometry = GridGeometry.from_extent(cells*cell_m, cells*cell_m, cell_m)
    occupancy = np.zeros(geometry.shape, np.uint8)
    occupancy[0, 4::6, 4::6] = 254
    return EvidenceGrid(geometry, occupancy, np.full(geometry.shape, int(time_s*1000), np.uint32),
                        'scan', 1, time_s)


def test_the_reachable_pass_stays_cheap_on_a_whole_arena():
    """What regressed: 2.8 s of reachability search on 40,000 cells, spent
    inside the publisher, so every route dnav then offered was already older
    than the executor's ``max_age_s`` when it arrived."""
    import time as _time

    cost_map = build_cost_map({'scan': big_grid()}, POLICY)
    assert cost_map.geometry.width * cost_map.geometry.height == 40_000, 'the size this measures'
    planner, start = AStarPlanner(), cost_map.geometry.to_cell(1.25, 1.25)
    searched = planner._prepare(cost_map, start, POLICY, {})[0]
    started = _time.perf_counter()
    reached = planner._reachable(searched.cost, start)
    elapsed = _time.perf_counter() - started
    assert reached.sum() > 30_000, 'the arena is open, so most of it is reachable'
    assert elapsed < .1, f'{elapsed*1000:.0f} ms to flood 40,000 cells'


def test_a_substitute_on_a_recorded_arena_is_well_inside_the_validity_window():
    """The archived maze corridor, at the size dnav actually plans on."""
    import time as _time

    record = json.loads((FIXTURE/'area1-phantom-goal.json').read_text())
    geometry = GridGeometry.from_dict(record['geometry'])
    with np.load(FIXTURE/'area1-phantom-goal.npz', allow_pickle=False) as data:
        grid = EvidenceGrid(geometry, data['occupancy'][None], data['observed_ms'][None],
                            record['source'], 1, record['time_s'])
    cost_map = build_cost_map({grid.source: grid}, POLICY)
    pose, goal = record['pose'], record['goal']
    assert AStarPlanner().plan(cost_map, pose, goal, POLICY).status == R.GOAL_UNREACHABLE
    started = _time.perf_counter()
    route = AStarPlanner().plan_near(cost_map, pose, goal, POLICY, PROFILE.goal_substitute_m)
    elapsed = _time.perf_counter() - started
    assert route.ok, route.reason
    # A tenth of the window: the route still has to survive the publish, the
    # validator and the trip to the executor before dway will look at it.
    assert elapsed < PROFILE.max_age_s / 10, f'{elapsed*1000:.0f} ms of a {PROFILE.max_age_s:g} s window'


def test_the_reachable_set_never_claims_more_than_the_exact_clearance_walk():
    """The flood fill trades a little reach for two orders of magnitude of speed.

    It may refuse a cell the exact per-edge test would allow -- a diagonal
    squeeze -- but it must never invent one, because `plan_near` only ever
    plans to cells it names.
    """
    # A diagonal gap between two blocks, which is where the two rules differ.
    grid = open_grid((8, 8), (9, 9), (4, 12), (5, 11), (12, 4), (11, 5))
    cost_map = build_cost_map({'scan': grid}, POLICY)
    geometry = cost_map.geometry
    start_cell = geometry.to_cell(1., 1.)
    planner = AStarPlanner()
    prepared = planner._prepare(cost_map, start_cell, POLICY, {})
    searched, corridor, allows, _ = prepared

    def exact(cost, start):
        """The per-edge walk this replaced, kept here as the thing to agree with."""
        from collections import deque
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
                            or not allows(geometry.cell_centre_m(col, row),
                                          geometry.cell_centre_m(ncol, nrow))):
                        continue
                    reached[nrow, ncol] = True
                    queue.append((ncol, nrow))
        return reached

    fill = planner._reachable(searched.cost, start_cell)
    walk = exact(searched.cost, start_cell)
    assert fill.any() and walk.any(), 'nothing was reachable, so nothing was compared'
    assert not (fill & ~walk).any(), 'the flood fill claimed a cell the exact test refuses'
    assert not (fill & ~np.isfinite(searched.cost)).any(), 'it claimed a blocked cell'


def test_a_substitute_is_only_offered_while_it_is_an_improvement():
    """Otherwise the answer oscillates: from the nearest reachable cell the
    next-nearest is the best candidate, and from there the first one is again.
    A consumer that defers to an eligible substitute would defer for ever."""
    grids = {'scan': open_grid((16, 10))}
    cost = build_cost_map(grids, POLICY)
    goal = [8.25, 5.25, 1.5]
    route = AStarPlanner().plan_near(cost, [1., 5.25, 1.5], goal, POLICY, PROFILE.goal_substitute_m)
    assert route.ok, route.reason
    end = [route.waypoints[-1].x_m, route.waypoints[-1].y_m, 1.5]
    # Standing on the point it just offered, there is nothing better to offer.
    again = AStarPlanner().plan_near(cost, end, goal, POLICY, PROFILE.goal_substitute_m)
    assert again.status == R.GOAL_UNREACHABLE
    assert 'already holds the reachable point nearest the goal' in again.reason


def test_the_reachable_pass_stays_cheap_on_corridors_not_just_open_ground():
    """The shape that matters: a maze, where a frontier-following mission works.
    A fill that grows one cell per pass costs one pass per cell of corridor --
    over a second on this grid, which is back inside the validity window this
    whole change exists to stay out of."""
    import time as _time

    cells, cell_m = 200, .5
    geometry = GridGeometry.from_extent(cells*cell_m, cells*cell_m, cell_m)
    occupancy = np.zeros(geometry.shape, np.uint8)
    for i, row in enumerate(range(4, cells, 6)):
        occupancy[0, row, :] = 254
        door = slice(2, 6) if i % 2 else slice(cells-6, cells-2)
        # Four cells wide: the margin blocks the two beside the wall ends, and
        # a narrower door would be sealed by it rather than merely pinched.
        occupancy[0, row, door] = 0
    grid = EvidenceGrid(geometry, occupancy, np.full(geometry.shape, 10_000, np.uint32), 'scan', 1, 10.)
    cost_map = build_cost_map({'scan': grid}, POLICY)
    planner, start = AStarPlanner(), geometry.to_cell(1.25, 1.25)
    searched = planner._prepare(cost_map, start, POLICY, {})[0]
    started = _time.perf_counter()
    reached = planner._reachable(searched.cost, start)
    elapsed = _time.perf_counter() - started
    assert reached.sum() > 20_000, 'the corridors connect, so most of the maze is reachable'
    assert elapsed < .25, f'{elapsed*1000:.0f} ms to flood a serpentine 200x200'
