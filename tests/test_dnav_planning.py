"""Cost policy and planners: the numbers a route is judged by.

The review gate here is the admissibility pair -- A\\* never costs more than the
straight line it is measured against, and never routes through an inflated
obstacle -- plus the status vocabulary saying which thing went wrong rather
than going quiet.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dtest.synthetic import SyntheticRoom
from dcmn.maps import EvidenceGrid, GridGeometry, quantize, stamp_ms
from dnav import route as R
from dnav.planners import DEFAULT_PLANNER, PLANNERS, Planner, build
from dnav.policy import (BLOCKED, FREE_COST, UNOBSERVED_COST, CostPolicy,
                         build_cost_map, disc_offsets, inflate, layer_cost,
                         load_policy, parse_policy)

ROOT = Path(__file__).resolve().parents[1]


def geometry(cell_m=.5, extent_m=20.):
    return GridGeometry.from_extent(extent_m, extent_m, cell_m)


def swept_room(sweeps=12, geo=None):
    """The committed fixture, fully revealed: the grid every gate is run on."""
    room = SyntheticRoom(geo or geometry())
    for step in range(sweeps): room.sweep(1. + step)
    return room.grid()


def open_grid(geo=None, *, walls=()):
    """A fully observed empty room, plus whatever walls a test asks for."""
    geo = geo or geometry(extent_m=10.)
    occupancy = np.full(geo.shape, quantize(.02), np.uint8)
    observed = np.full(geo.shape, stamp_ms(5.), np.uint32)
    for col0, row0, col1, row1 in walls:
        occupancy[0, row0:row1, col0:col1] = quantize(.95)
    return EvidenceGrid(geo, occupancy, observed, 'test', revision=3, sim_time_s=5.)


def cost_map(grid, policy, **kwargs):
    return build_cost_map({grid.source: grid}, policy, sim_time_s=grid.sim_time_s,
                          **kwargs)


@pytest.fixture
def policy():
    return load_policy('default', ROOT)


# -- the policy artifact -----------------------------------------------------

def test_the_committed_policies_load_and_carry_a_digest():
    for name in ('default', 'cautious'):
        loaded = load_policy(name, ROOT)
        assert loaded.name == name and len(loaded.digest) == 64
        assert loaded.combine == 'max'
    assert load_policy('cautious', ROOT).inflation_m > load_policy('default', ROOT).inflation_m


def test_a_policy_digest_names_the_bytes_that_were_read():
    raw = b'{"schema": "dvision2.cost-policy.v1", "name": "x", "inflation_m": 0.5}'
    first = parse_policy(raw)
    assert first.digest == parse_policy(raw).digest
    assert first.digest != parse_policy(raw.replace(b'0.5', b'0.6')).digest


@pytest.mark.parametrize('raw,message', [
    (b'[]', 'must be an object'),
    (b'{"name": "x"}', 'schema must be'),
    (b'{"schema": "dvision2.cost-policy.v1", "schema_version": 9}', 'not supported'),
])
def test_an_unusable_policy_is_refused_with_a_message(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_policy(raw)


@pytest.mark.parametrize('field,value', [
    ('occupied_threshold', 0.0), ('occupied_threshold', 1.5),
    ('inflation_m', -1.0), ('inflation_m', float('inf')),
    ('combine', 'sum'),
])
def test_a_policy_that_could_not_be_applied_is_refused(field, value):
    with pytest.raises(ValueError):
        CostPolicy(**{field: value})


def test_summing_layers_is_not_offered_at_all():
    """Two sensors' opinions of one wall are one wall; a sum counts it twice."""
    with pytest.raises(ValueError, match='sum'):
        CostPolicy(combine='sum')


# -- inflation ---------------------------------------------------------------

def test_the_margin_is_a_disc_and_not_a_square():
    """A square margin keeps 1.4x the asked-for distance diagonally."""
    offsets = set(disc_offsets(1.2))
    assert (1, 0) in offsets and (0, 1) in offsets
    assert (1, 1) not in offsets, 'the corner is 1.41 cells away, not 1.2'


def test_inflation_grows_an_obstacle_by_the_radius_and_no_further():
    mask = np.zeros((9, 9), bool)
    mask[4, 4] = True
    grown = inflate(mask, 2.0)
    assert grown[4, 2] and grown[4, 6] and grown[2, 4] and grown[6, 4]
    assert not grown[4, 1] and not grown[2, 2]        # 3 cells, and 2.83 cells
    assert not inflate(mask, 0.0)[4, 3]


def test_the_radius_quantises_to_the_grid_rather_than_rounding_up():
    """Rounding 0.6 m up to a whole 0.5 m cell seals a two-metre doorway."""
    policy = CostPolicy(inflation_m=.6)
    assert policy.inflation_radius_cells(.5) == pytest.approx(1.2)
    grid = swept_room()
    blocked = cost_map(grid, policy).blocked
    wide = cost_map(grid, CostPolicy(inflation_m=1.1)).blocked
    assert int(wide.sum()) > int(blocked.sum())


def test_a_wider_margin_blocks_more_and_never_less(policy):
    grid = swept_room()
    previous = -1
    for inflation in (0.0, 0.25, 0.5, 1.0, 1.5):
        blocked = int(cost_map(grid, CostPolicy(inflation_m=inflation)).blocked.sum())
        assert blocked >= previous
        previous = blocked


# -- evidence to cost --------------------------------------------------------

def test_never_observed_is_passable_but_never_as_cheap_as_known_free(policy):
    geo = geometry(extent_m=5.)
    grid = EvidenceGrid.blank(geo)
    occupancy = grid.occupancy.copy(); observed = grid.observed_ms.copy()
    occupancy[0, 0, 0] = quantize(.02); observed[0, 0, 0] = stamp_ms(1.)
    layer = layer_cost(EvidenceGrid(geo, occupancy, observed, 'x'), policy)
    assert layer.cost[0, 0] == pytest.approx(FREE_COST)
    assert layer.cost[4, 4] == pytest.approx(UNOBSERVED_COST)
    assert UNOBSERVED_COST > FREE_COST
    assert math.isfinite(UNOBSERVED_COST), \
        'a planner that cannot enter the unknown never leaves the room it started in'


def test_the_threshold_is_what_makes_a_cell_an_obstacle():
    geo = geometry(extent_m=5.)
    probabilities = np.linspace(0., 1., geo.width, dtype=np.float32)
    occupancy = np.tile(quantize(probabilities), (1, geo.height, 1))
    observed = np.full(geo.shape, stamp_ms(1.), np.uint32)
    grid = EvidenceGrid(geo, occupancy, observed, 'x')
    strict = layer_cost(grid, CostPolicy(occupied_threshold=.9, inflation_m=0.))
    loose = layer_cost(grid, CostPolicy(occupied_threshold=.2, inflation_m=0.))
    assert int(loose.obstacles.sum()) > int(strict.obstacles.sum())


def test_layers_combine_by_the_worst_witness_and_never_by_the_sum(policy):
    geo = geometry(extent_m=5.)
    observed = np.full(geo.shape, stamp_ms(1.), np.uint32)
    quiet = np.full(geo.shape, quantize(.02), np.uint8)
    loud = quiet.copy(); loud[0, 2, 2] = quantize(.95)
    grids = {'camera': EvidenceGrid(geo, quiet, observed, 'camera'),
             'lidar': EvidenceGrid(geo, loud, observed, 'lidar')}
    combined = build_cost_map(grids, CostPolicy(inflation_m=0.))
    assert len(combined.layers) == 2
    assert math.isinf(float(combined.cost[2, 2]))
    # One wall seen by one sensor is still one wall: everywhere else stays at
    # the price of free ground rather than twice it.
    assert combined.cost[0, 0] == pytest.approx(FREE_COST)


def test_a_source_with_a_different_cell_size_is_refused(policy):
    coarse = open_grid(geometry(cell_m=1., extent_m=10.))
    fine = open_grid(geometry(cell_m=.5, extent_m=10.))
    with pytest.raises(ValueError, match='geometry differs'):
        build_cost_map({'a': coarse, 'b': fine}, policy)


def test_no_evidence_is_no_cost_map(policy):
    assert build_cost_map({}, policy) is None


# -- the planner protocol ----------------------------------------------------

def test_every_registered_planner_satisfies_the_protocol():
    for name, planner in PLANNERS.items():
        instance = planner()
        assert isinstance(instance, Planner)
        assert instance.name == name
    assert DEFAULT_PLANNER in PLANNERS


def test_an_unknown_planner_is_refused_rather_than_defaulted():
    with pytest.raises(KeyError, match='unknown planner'):
        build('dijkstra')


# -- the admissibility gate --------------------------------------------------

@pytest.mark.parametrize('goal', [(16., 5.), (16., 15.), (17., 10.), (2., 2.),
                                  (8., 17.), (3., 18.)])
def test_astar_never_costs_more_than_the_straight_line(goal, policy):
    """The control is the floor. A plan that costs more has a broken search."""
    grid = swept_room()
    cells = cost_map(grid, policy)
    start = (5., 10., 1.5)
    plan = build('astar').plan(cells, start, (*goal, 1.5), policy)
    control = build('control').plan(cells, start, (*goal, 1.5), policy)
    if not plan.ok:
        assert math.isinf(control.cost), \
            'the straight line was clear but A* found nothing'
        return
    assert plan.cost <= control.cost + 1e-9


def test_astar_never_routes_through_an_inflated_obstacle(policy):
    grid = swept_room()
    cells = cost_map(grid, policy)
    plan = build('astar').plan(cells, (5., 10., 1.5), (16., 5., 1.5), policy)
    assert plan.ok
    for col, row in R.polyline_cells(cells.geometry, plan.points):
        assert math.isfinite(float(cells.cost[row, col])), \
            f'the route enters blocked cell {col},{row}'


def test_a_route_costs_what_it_says_it_costs(policy):
    """The quoted number has to be the price of the polyline that was published."""
    grid = swept_room()
    cells = cost_map(grid, policy)
    for goal in ((16., 5.), (16., 15.), (2., 2.)):
        plan = build('astar').plan(cells, (5., 10., 1.5), (*goal, 1.5), policy)
        assert plan.ok
        assert R.route_cost(cells, plan.points) == pytest.approx(plan.cost)


def test_pulling_the_string_taut_can_only_make_a_route_cheaper(policy):
    grid = swept_room()
    cells = cost_map(grid, policy)
    corner = [(5., 10.), (5., 5.), (8., 5.)]
    pulled = R.shorten(cells.cost, cells.geometry, corner)
    assert R.route_cost(cells, pulled) <= R.route_cost(cells, corner) + 1e-9
    assert len(pulled) <= len(corner)


def test_a_plan_across_open_ground_is_the_straight_line(policy):
    """Grid quantisation must not show up as a detour that is not there."""
    cells = cost_map(open_grid(), policy)
    plan = build('astar').plan(cells, (2., 2., 1.), (8., 7., 1.), policy)
    assert plan.ok and len(plan.waypoints) == 2
    assert plan.length_m == pytest.approx(math.dist((2., 2.), (8., 7.)))


# -- the status vocabulary ---------------------------------------------------

def test_a_blocked_doorway_is_no_route_with_a_reason():
    grid = swept_room()
    cautious = load_policy('cautious', ROOT)
    plan = build('astar').plan(cost_map(grid, cautious), (1.5, 1.5, 1.5),
                               (15.5, 10., 1.5), cautious)
    assert plan.status == R.NO_ROUTE
    assert plan.reason and 'no route' in plan.reason


def test_a_goal_inside_a_wall_is_unreachable_not_merely_unroutable(policy):
    grid = swept_room()
    plan = build('astar').plan(cost_map(grid, policy), (5., 10., 1.5),
                               (0.2, 0.2, 1.5), policy)
    assert plan.status == R.GOAL_UNREACHABLE
    assert 'margin' in plan.reason or 'obstacle' in plan.reason


def test_a_vehicle_inside_a_wall_says_so_about_the_start(policy):
    grid = swept_room()
    plan = build('astar').plan(cost_map(grid, policy), (0.2, 0.2, 1.5),
                               (5., 10., 1.5), policy)
    assert plan.status == R.START_BLOCKED
    assert 'vehicle' in plan.reason


@pytest.mark.parametrize('planner', sorted(PLANNERS))
@pytest.mark.parametrize('start,goal,status', [
    ((-5., 10., 1.5), (5., 10., 1.5), R.START_BLOCKED),
    ((5., 10., 1.5), (99., 99., 1.5), R.GOAL_UNREACHABLE)])
def test_a_point_off_the_grid_is_named_rather_than_clamped(planner, start, goal,
                                                           status, policy):
    """Clamping would price a route that leaves the map as though it had not."""
    plan = build(planner).plan(cost_map(swept_room(), policy), start, goal, policy)
    assert plan.status == status
    assert 'outside the mapped area' in plan.reason


def test_the_control_answers_even_when_the_line_goes_through_a_wall(policy):
    """A baseline that declines whenever the answer is bad is not a baseline."""
    grid = swept_room()
    control = build('control').plan(cost_map(grid, policy), (5., 10., 1.5),
                                    (16., 5., 1.5), policy)
    assert control.ok and math.isinf(control.cost)
    assert len(control.waypoints) == 2


def test_a_route_that_failed_may_not_pretend_to_have_waypoints():
    with pytest.raises(ValueError, match='may not carry waypoints'):
        R.Route(status=R.NO_ROUTE, reason='x',
                waypoints=(R.Waypoint(0., 0., 0.),))


def test_a_failure_must_say_why():
    with pytest.raises(ValueError, match='must say why'):
        R.Route(status=R.NO_ROUTE)


def test_an_unknown_status_is_not_a_status():
    with pytest.raises(ValueError, match='unknown route status'):
        R.Route(status='confused')


# -- the route's shape -------------------------------------------------------

def test_a_waypoint_carries_the_speed_a_follower_would_use():
    """The same shape as a tour's waypoints, so an executor needs no translation."""
    plain = R.Waypoint(1., 2., 3.)
    assert plain.as_dict() == {'x': 1., 'y': 2., 'z': 3.}
    assert R.Waypoint(1., 2., 3., 0.4).as_dict()['speed_mps'] == 0.4


def test_a_route_stays_inside_the_slab_it_was_planned_in(policy):
    cells = cost_map(open_grid(), policy)
    plan = build('astar').plan(cells, (2., 2., 1.75), (8., 7., 1.75), policy)
    assert {w.z_m for w in plan.waypoints} == {1.75}


def test_a_route_names_the_evidence_and_the_policy_it_was_planned_against(policy):
    grid = swept_room()
    cells = cost_map(grid, policy)
    plan = build('astar').plan(cells, (5., 10., 1.5), (16., 5., 1.5), policy)
    assert plan.policy_digest == policy.digest
    assert plan.map_revision == cells.revision == grid.revision
    assert plan.planner == 'astar'


def test_bresenham_stays_on_the_move_set_astar_searches():
    geo = geometry(extent_m=10.)
    cells = R.line_cells(geo, (1., 1.), (5., 5.))
    for (col, row), (previous_col, previous_row) in zip(cells[1:], cells):
        assert max(abs(col - previous_col), abs(row - previous_row)) == 1


def test_a_line_that_leaves_the_grid_is_dropped_rather_than_clamped():
    geo = geometry(extent_m=10.)
    cells = R.line_cells(geo, (1., 1.), (30., 1.))
    assert max(col for col, _ in cells) == geo.width - 1
    assert len(cells) <= geo.width


def test_simplifying_a_path_changes_no_geometry():
    cells = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]
    assert R.simplify(cells) == [(0, 0), (3, 0), (3, 2)]


def test_the_history_records_a_change_and_not_every_repetition():
    history = R.RouteHistory()
    first = R.failed(R.NO_GOAL, 'no goal set')
    assert history.observe(first)
    assert not history.observe(R.failed(R.NO_GOAL, 'no goal set'))
    assert history.observe(R.failed(R.STALE_MAP, 'producer gone'))
    assert history.counts() == {R.NO_GOAL: 1, R.STALE_MAP: 1}


# -- the query sidecar -------------------------------------------------------

def test_a_committed_query_supplies_a_start_and_a_goal_without_reading_a_map():
    from dtest.queries import load_query

    query = load_query(ROOT / 'tests/assets/planner_queries/maze_013.v1.json')
    assert query['goal'] == (41.5, 16.5) and query['start'] == (1.5, 1.5)
    assert len(query['map_sha']) == 64
    assert json.loads((ROOT / 'tests/assets/planner_queries/maze_013.v1.json').read_bytes())


def test_a_malformed_query_is_refused_with_the_file_named(tmp_path):
    from dtest.queries import load_query

    path = tmp_path / 'bad.json'
    path.write_text('{"queries": []}', encoding='utf-8')
    with pytest.raises(ValueError, match='carries no queries'):
        load_query(path)


@pytest.mark.parametrize('text,expected', [
    ('1,2', (1., 2., None)), ('1.5, 2.5, 3.5', (1.5, 2.5, 3.5))])
def test_a_goal_is_parsed_from_the_command_line(text, expected):
    from dnav.plan import parse_goal
    assert parse_goal(text) == expected


@pytest.mark.parametrize('text', ['1', '1,2,3,4', 'a,b'])
def test_a_goal_that_is_not_two_or_three_numbers_is_refused(text):
    from dnav.plan import parse_goal
    with pytest.raises(ValueError):
        parse_goal(text)


def test_the_blocked_sentinel_is_infinite_rather_than_merely_large():
    """A planner that can be bribed through a wall by a long detour is not one."""
    assert math.isinf(BLOCKED)


def _margin_start(cost_map_value):
    """A cell inside an obstacle's margin, not the obstacle, next to open space."""
    layer = cost_map_value.layers[0]
    margin = layer.inflated & ~layer.obstacles & ~cost_map_value.never_observed
    height, width = margin.shape
    finite = np.isfinite(cost_map_value.cost)
    for row, col in zip(*np.nonzero(margin)):
        if any(0 <= row + dr < height and 0 <= col + dc < width and finite[row + dr, col + dc]
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            return cost_map_value.geometry.cell_centre_m(int(col), int(row))
    raise AssertionError('fixture has no margin cell beside open space')


def test_a_vehicle_stopped_inside_a_margin_is_routed_out_of_it(policy):
    from dataclasses import replace
    grid = swept_room()
    costs = cost_map(grid, policy)
    x, y = _margin_start(costs)
    plan = build('astar').plan(costs, (x, y, 1.5), (5., 10., 1.5), policy)
    assert plan.status == R.OK, plan.reason
    assert plan.diagnostics['escaped_margin'] and 'escaping' in plan.reason
    assert math.isfinite(plan.cost)
    strict = replace(policy, escape_margin=False)
    assert build('astar').plan(cost_map(grid, strict), (x, y, 1.5), (5., 10., 1.5), strict).status == R.START_BLOCKED


def test_an_obstacle_marked_on_the_vehicles_own_observed_cell_is_rejected(policy):
    """Occupancy at the start cannot be dismissed as a false sensor reading."""
    grid = swept_room()
    x, y = _margin_start(cost_map(grid, policy))
    col, row = grid.geometry.to_cell(x, y)
    occupancy, observed = (np.array(a)[None].copy() for a in grid.layer(0))
    occupancy[0, row, col] = 254
    marked = EvidenceGrid(grid.geometry, occupancy, observed, grid.source, grid.revision + 1, grid.sim_time_s)
    plan = build('astar').plan(cost_map(marked, policy), (x, y, 1.5), (5., 10., 1.5), policy)
    assert plan.status == R.START_BLOCKED, plan.reason
    assert 'observed occupied' in plan.reason
