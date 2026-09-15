"""Full-cell geometry, constrained escape, and the September 15 wall-end crash."""
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dcmn.clearance import ObstacleClearance
from dcmn.maps import EvidenceGrid, GridGeometry
from dcmn.navigation import ExecutionProfile
from dnav.clearance import Clearance
from dnav.planners.astar import AStarPlanner
from dnav.policy import CostPolicy, build_cost_map


def obstacle(radius=.3):
    geometry = GridGeometry.from_extent(10, 10, .5)
    occupied = np.zeros((geometry.height, geometry.width), bool)
    occupied[4, 4] = True  # [2, 2.5] x [2, 2.5]
    return ObstacleClearance(geometry, occupied, radius, .15)


@pytest.mark.parametrize(('a', 'b', 'distance'), [
    ((1, 1.9), (3, 1.9), .1),       # grazing the face; centres would miss it
    ((1, 2.25), (3, 2.25), 0.),     # crosses the interior without touching a vertex
    ((1.8, 1.8), (1.8, 1.8), math.sqrt(.08)),
    ((1, 2), (2, 1), math.sqrt(.5)), # closest point is inside the segment
    ((2.5, 1), (2.5, 3), 0.),       # exactly touches a face
])
def test_exact_segment_square_distance(a, b, distance):
    _, _, closest = obstacle(radius=2).distances(a, b)
    assert closest.tolist() == pytest.approx([distance])


def test_escape_cannot_cross_approach_or_reenter_a_nearby_obstacle():
    c = obstacle(radius=1.)
    assert c.allows((1.7, 2.25), (1., 2.25), escape=True)
    assert not c.allows((1.7, 2.25), (1.8, 2.25), escape=True)
    assert not c.allows((1.7, 2.25), (3., 2.25), escape=True)
    assert not c.allows((1.9, 2.25), (1., 2.25), escape=True)  # body already overlaps
    assert not c.allows((.9, 2.25), (1.7, 2.25), escape=True)  # enter margin again


def test_inflation_covers_cell_faces_and_soft_band_prefers_clearance():
    c = obstacle()
    g = EvidenceGrid(c.geometry, np.where(c.occupied[None], 254, 0).astype(np.uint8),
                     np.full(c.geometry.shape, 10000, np.uint32), 'scan', 1, 10.)
    cost = build_cost_map({'scan': g}, CostPolicy(inflation_m=.3))
    assert cost.blocked[4, 3]  # centre is .5 m away, but only .25 m from the face
    assert not cost.blocked[3, 3]  # diagonal clearance sqrt(.25² + .25²) > .3
    assert cost.cost[3, 3] > cost.cost[1, 1]  # allowed, but less attractive


@pytest.mark.parametrize('permission', ['plan', 'evidence'])
def test_narrow_corridor_uses_lateral_clearance_not_braking_radius(permission):
    geometry = GridGeometry.from_extent(10, 10, .5)
    occupancy = np.zeros(geometry.shape, np.uint8)
    occupancy[0, 2, :] = 254  # lower wall ends at y=1.5
    occupancy[0, 6, :] = 254  # upper wall starts at y=3: 1.5 m corridor
    grid = EvidenceGrid(geometry, occupancy, np.full(geometry.shape, 10000, np.uint32), 'scan', 1, 10.)
    grids = {'scan': grid}
    profile = ExecutionProfile.load('sim-target', {'permission': permission})
    start, goal = [2.25, 2.25, 1.5], [7.25, 2.25, 1.5]
    policy = CostPolicy().for_execution(profile)
    route = AStarPlanner().plan(build_cost_map(grids, policy), start, goal, policy)
    assert route.ok, route.reason
    points = [[w.x_m, w.y_m, w.z_m] for w in route.waypoints]
    verdict = Clearance(profile).check(points, start, grids, 10.)
    assert verdict['eligible'] and verdict['reaches_goal'], verdict
    # A longer braking distance must not close a corridor beside the vehicle.
    longer_stop = replace(profile, stopping_m=profile.stopping_m + 1., reaction_s=1., speed_mps=2.)
    assert longer_stop.clearance_radius_m == profile.clearance_radius_m
    assert Clearance(longer_stop).check(points, start, grids, 10.)['eligible']
    # Narrowing the corridor to 0.5 m leaves less than the hard clearance on
    # each side, and the same admission checks must reject it.
    occupancy[0, 3, :] = 254
    occupancy[0, 5, :] = 254
    narrow = EvidenceGrid(geometry, occupancy, np.full(geometry.shape, 10000, np.uint32), 'scan', 2, 10.)
    assert not Clearance(profile).check(points, start, {'scan': narrow}, 10.)['eligible']
    assert not AStarPlanner().plan(build_cost_map({'scan': narrow}, policy), start, goal, policy).ok


def test_area1_archived_crash_route_is_rejected_and_replanned():
    fixture = Path(__file__).parent/'assets/corner_clearance'
    record = json.loads((fixture/'area1.json').read_text())
    geometry = GridGeometry.from_dict(record['geometry'])
    with np.load(fixture/'area1.npz', allow_pickle=False) as data:
        g = EvidenceGrid(geometry, data['occupancy'][None], data['observed_ms'][None],
                         record['source'], 1, record['time_s'])
    grids = {g.source: g}
    profile = ExecutionProfile.load('sim-target')
    validator = Clearance(profile)
    original = validator.check(record['points'], record['pose'], grids, record['time_s'],
                               start=record['progress'])
    assert not original['eligible'] and 'blocked' in original['reason']
    policy = CostPolicy().for_execution(profile)
    assert policy.inflation_m == pytest.approx(.35)
    route = AStarPlanner().plan(build_cost_map(grids, policy), record['pose'], record['points'][-1], policy)
    assert route.ok, route.reason
    points = [[w.x_m, w.y_m, w.z_m] for w in route.waypoints]
    assert validator.check(points, record['pose'], grids, record['time_s'])['eligible']
    assert min(p[0] for p in points) < 30.  # go farther around the wall's left end
    # Independent truth check against the wall end [31,54] x [4,5]. It is used
    # only by this regression assertion, never supplied to the planner.
    minimum = math.inf
    for a, b in zip(points, points[1:]):
        for t in np.linspace(0, 1, 1001):
            x, y = np.asarray(a[:2]) + t*(np.asarray(b[:2])-a[:2])
            minimum = min(minimum, math.hypot(max(31-x, x-54, 0), max(4-y, y-5, 0)))
    assert minimum > profile.body_radius_m


def test_unobserved_cells_touching_an_obstacle_count_as_solid():
    geometry = GridGeometry.from_extent(10, 10, .5)
    occupancy = np.zeros(geometry.shape, np.uint8)
    occupancy[0, 4, 10:] = 254  # a wall from x=5, y in [2, 2.5]
    occupancy[0, 3:6, 9] = 255  # its end column x in [4.5, 5) never observed
    occupancy[0, 8, 2] = 255    # unknown far from any obstacle
    observed = np.where(occupancy == 255, 0, 10000).astype(np.uint32)
    grids = {'scan': EvidenceGrid(geometry, occupancy, observed, 'scan', 1, 10.)}
    profile = ExecutionProfile.load('sim-target')
    policy = CostPolicy().for_execution(profile)
    blocked = build_cost_map(grids, policy).blocked
    assert blocked[4, 9] and not blocked[8, 2]
    validator = Clearance(profile)
    # 0.4 m beyond the observed end clears the hard radius, but not the unseen end.
    past_end = [[4.6, 1., 1.5], [4.6, 4., 1.5]]
    assert not validator.check(past_end, past_end[0], grids, 10.)['eligible']
    through_unknown = [[1.25, 1., 1.5], [1.25, 6., 1.5]]
    assert validator.check(through_unknown, through_unknown[0], grids, 10.)['eligible']


def test_graded_band_centres_a_route_in_a_corridor_it_cannot_leave():
    geometry = GridGeometry.from_extent(10, 10, .5)
    occupancy = np.zeros(geometry.shape, np.uint8)
    occupancy[0, 1, :] = 254  # wall ends at y=1
    occupancy[0, 8, :] = 254  # wall starts at y=4: a 3 m corridor
    grid = EvidenceGrid(geometry, occupancy, np.full(geometry.shape, 10000, np.uint32), 'scan', 1, 10.)
    policy = CostPolicy().for_execution(ExecutionProfile.load('sim-target'))
    cost = build_cost_map({'scan': grid}, policy).cost[:, 10]
    assert not np.isfinite(cost[2]) and not np.isfinite(cost[7])  # 0.25 m from a face
    assert 1. < cost[3] < 1. + policy.clearance_cost               # in the band, passable
    assert cost[4] == cost[5] == pytest.approx(1.)                 # the corridor's middle
    start, goal = [.75, 1.75, 1.5], [9.25, 1.75, 1.5]
    route = AStarPlanner().plan(build_cost_map({'scan': grid}, policy), start, goal, policy)
    assert route.ok, route.reason
    assert any(2. < w.y_m < 3. for w in route.waypoints)  # leaves the wall-side band


def test_area1_corridor_with_phantom_cells_narrows_tracking_instead_of_closing():
    fixture = Path(__file__).parent/'assets/corner_clearance'
    record = json.loads((fixture/'area1-corridor.json').read_text())
    geometry = GridGeometry.from_dict(record['geometry'])
    with np.load(fixture/'area1-corridor.npz', allow_pickle=False) as data:
        g = EvidenceGrid(geometry, data['occupancy'][None], data['observed_ms'][None],
                         record['source'], 1, record['time_s'])
    grids = {g.source: g}
    profile = ExecutionProfile.load('sim-target')
    policy = CostPolicy().for_execution(profile)
    route = AStarPlanner().plan(build_cost_map(grids, policy), record['pose'], record['goal'], policy)
    assert route.ok, route.reason
    points = [[w.x_m, w.y_m, w.z_m] for w in route.waypoints]
    verdict = Clearance(profile).check(points, record['pose'], grids, record['time_s'])
    assert verdict['eligible'] and verdict['reaches_goal'], verdict['reason']
    allowance = [profile.tracking_allowance_m(x) for x in verdict['lateral_m']]
    assert profile.min_tracking_m - 1e-6 <= min(allowance) < profile.tracking_m
    # Independent truth check against maze_020's top corridor, y in [1, 4].
    minimum = math.inf
    for a, b in zip(points, points[1:]):
        for t in np.linspace(0, 1, 201):
            x, y = np.asarray(a[:2]) + t*(np.asarray(b[:2])-a[:2])
            minimum = min(minimum, y-1, math.hypot(max(31-x, x-54, 0), max(4-y, 0)))
    assert minimum > profile.body_radius_m + profile.tracking_m


def test_dway_narrows_tracking_allowance_and_speed_to_published_clearance():
    from types import SimpleNamespace
    from dway.executor import NARROW_SPEED_FRACTION, DynamicExecutor
    profile = ExecutionProfile.load('sim-target')
    value = dict(points=[[0, 0, 1.5], [5, 0, 1.5], [5, 5, 1.5]], clearance=dict(lateral_m=[.65, .45]))
    ex = SimpleNamespace(profile=profile, segment=0, tracking_allowance=None, corner_factor=1.)
    allowance = DynamicExecutor._tracking_allowance
    assert allowance(ex, value, (0, .5)) == pytest.approx(profile.tracking_m)
    ex.segment = 1  # already flying toward the narrower segment
    assert allowance(ex, value, (0, .99)) == pytest.approx(.2)
    assert allowance(ex, dict(value, clearance={}), (0, .5)) == 0.  # nothing published, nothing tolerated
    ex.tracking_allowance = .2
    assert DynamicExecutor._speed_cap(ex) == pytest.approx(profile.speed_mps * .2 / profile.tracking_m)
    ex.tracking_allowance = 0.
    assert DynamicExecutor._speed_cap(ex) == pytest.approx(profile.speed_mps * NARROW_SPEED_FRACTION)


def test_clearance_checks_actual_departure_and_does_not_erase_nearby_cells():
    c = obstacle()
    g = EvidenceGrid(c.geometry, np.where(c.occupied[None], 254, 0).astype(np.uint8),
                     np.full(c.geometry.shape, 10000, np.uint32), 'scan', 1, 10.)
    profile = ExecutionProfile(permission='plan', slab_assumption='vertical-extrusion',
                               body_radius_m=.15, tracking_m=0, mapping_margin_m=0,
                               stopping_m=0, reaction_s=0, plan_clearance_m=.15, join_m=1.)
    validator = Clearance(profile)
    points = [[1.5, 1.5, 1.5], [3., 1.5, 1.5]]
    assert validator.check(points, points[0], {'scan': g}, 10.)['eligible']
    # The planned segment is safe, but a shortcut from the actual stop clips
    # the lower-left corner of the occupied cell.
    verdict = validator.check(points, [1.5, 2.3, 1.5], {'scan': g}, 10.)
    assert not verdict['eligible'] and 'departure' in verdict['reason']
