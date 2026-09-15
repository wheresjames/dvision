"""Smooth route changes: permission shortened instead of withdrawn, handoff while moving, corner flags."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from dcmn.maps import EvidenceGrid, GridGeometry
from dcmn.navigation import ExecutionProfile, validate
from dnav import route as R
from dnav.clearance import Clearance
from dnav.execution import RoutePublisher

ROOT = Path(__file__).parent
GOAL = [9.5, 5.25, 1.5]


def open_grid(*cells, time_s=10.):
    """A fully observed 10 m square with occupied ``(col, row)`` cells of 0.5 m."""
    geometry = GridGeometry.from_extent(10, 10, .5)
    occupancy = np.zeros(geometry.shape, np.uint8)
    for col, row in cells: occupancy[0, row, col] = 254
    return EvidenceGrid(geometry, occupancy, np.full(geometry.shape, int(time_s*1000), np.uint32), 'scan', 1, time_s)


class StubRun:
    """The NavRun surface ``RoutePublisher`` reads."""

    def __init__(self, grids):
        self.snapshot = dict(provider_id='p', frame_id='local', localization_epoch=0,
                             clock_domain_id='v', clock_epoch=0)
        self.goal_descriptor = dict(authority='ui', authority_epoch=1, revision=1, frame_id='local',
                                    localization_epoch=0, clock_epoch=0, position=GOAL)
        self.session = SimpleNamespace(identity='stub', context={'mapping_epoch': 0})
        self.grids, self.attempts, self.recorder = grids, 0, None
        self.pose = [1., 5.25, 1.5]
        self.route = planned((1., 5.25), GOAL[:2])

    def sim_time_s(self): return 10.
    def _grids(self): return dict(self.grids)
    def _pose(self): return dict(x_m=self.pose[0], y_m=self.pose[1], z_m=self.pose[2])


def planned(*points):
    return R.Route(status=R.OK, waypoints=tuple(R.Waypoint(x, y, 1.5) for x, y in points))


def flying(profile, run):
    """A publisher whose executor, once engaged, flies the first published route."""
    publisher = RoutePublisher('rh-' + uuid.uuid4().hex[:8], profile)
    executor = {}
    publisher._read_executor = lambda wall: executor
    first = publisher.publish(run, wall=0.)
    assert first['clearance']['eligible'], first['clearance']['reason']
    return publisher, executor, first


@pytest.mark.parametrize('handoff_m', [0., .5])
def test_a_route_blocked_ahead_is_shortened_only_with_handoff_enabled(handoff_m):
    run = StubRun({'scan': open_grid()})
    publisher, executor, first = flying(ExecutionProfile.load('sim-target', {'handoff_m': handoff_m}), run)
    try:
        run.grids = {'scan': open_grid((16, 10))}  # x in [8, 8.5]: 6 m ahead
        run.pose = [2., 5.25, 1.5]
        executor.update(state='EXECUTING', geometry_revision=first['geometry_revision'], progress=[0, 1/8.5])
        after = publisher.publish(run, wall=0.)
        if handoff_m:
            assert after['clearance']['eligible'] and not after['clearance']['reaches_goal']
            assert after['stop_generation'] == first['stop_generation']
        else:
            assert not after['clearance']['eligible']
            assert after['stop_generation'] == first['stop_generation'] + 1
    finally:
        publisher.close()


def blocked_and_shortened():
    run = StubRun({'scan': open_grid()})
    publisher, executor, first = flying(ExecutionProfile.load('sim-target'), run)
    run.grids = {'scan': open_grid((16, 10), (16, 11))}  # x in [8, 8.5], y in [5, 6]
    run.pose = [3., 5.25, 1.5]
    executor.update(state='EXECUTING', geometry_revision=first['geometry_revision'], progress=[0, 2/8.5])
    shortened = publisher.publish(run, wall=0.)
    assert shortened['clearance']['eligible'] and not shortened['clearance']['reaches_goal']
    return run, publisher, executor, first, shortened


def test_a_close_replacement_is_handed_off_while_moving():
    run, publisher, executor, first, shortened = blocked_and_shortened()
    try:
        # The planner routes around it from where the vehicle is: 14° off the current heading.
        run.route = planned((3.1, 5.25), (7., 4.25), (9., 4.25), GOAL[:2])
        taken = publisher.publish(run, wall=0.)
        assert taken['handoff'] == dict(from_revision=first['geometry_revision'], start=[0, 0.])
        assert taken['points'][1] == [7., 4.25, 1.5] and taken['geometry_revision'] > shortened['geometry_revision']
        assert taken['clearance']['eligible'] and taken['clearance']['reaches_goal'], taken['clearance']['reason']
        assert taken['stop_generation'] == first['stop_generation']
        # The executor still reports the route it is leaving: located on the new one, offer repeated.
        run.pose = [3.3, 5.2, 1.5]
        again = publisher.publish(run, wall=0.)
        assert again['handoff'] == taken['handoff'] and again['clearance']['eligible'], again['clearance']['reason']
        assert again['geometry_revision'] == taken['geometry_revision']
        # Once it reports the new revision, the offer is dropped.
        executor.update(geometry_revision=taken['geometry_revision'], progress=[0, .05])
        done = publisher.publish(run, wall=0.)
        assert 'handoff' not in done and done['clearance']['eligible']
    finally:
        publisher.close()


@pytest.mark.parametrize('replacement', [
    ((3.1, 6.0), (7., 4.25), (9., 4.25), (9.5, 5.25)),  # 0.73 m off the vehicle
    ((3.1, 5.25), (5., 3.5), (9., 3.5), (9.5, 5.25)),   # turns 43° away
])
def test_a_replacement_off_the_vehicle_or_turning_away_keeps_the_shortened_route(replacement):
    run, publisher, executor, first, shortened = blocked_and_shortened()
    try:
        run.route = planned(*replacement)
        kept = publisher.publish(run, wall=0.)
        assert 'handoff' not in kept and kept['points'] == shortened['points']
        assert kept['clearance']['eligible'] and kept['stop_generation'] == first['stop_generation']
    finally:
        publisher.close()


def test_corner_flags_clear_the_shortcut_actually_flown():
    profile = ExecutionProfile.load('sim-target', {'stopping_m': 1.})
    points = [[1., 2.25, 1.5], [5., 2.25, 1.5], [5., 8., 1.5]]
    validator = Clearance(profile)
    clear = validator.check(points, points[0], {'scan': open_grid()}, 10.)
    assert clear['eligible'] and clear['corners'] == [True]
    # Inside the turn, clear of both legs but on the shortcut from 1 m before the vertex.
    inside = validator.check(points, points[0], {'scan': open_grid((8, 6))}, 10.)
    assert inside['eligible'] and inside['corners'] == [False], inside['reason']
    # Permission ending on the next leg: the shortcut is to where it ends.
    shortened = validator.check(points, points[0], {'scan': open_grid((10, 14))}, 10., prefix=True)
    assert shortened['eligible'] and not shortened['reaches_goal'] and shortened['corners'] == [True]
    # ... but not when that is within the stopping distance of the vertex.
    stub = validator.check(points, points[0], {'scan': open_grid((10, 6))}, 10., prefix=True)
    assert stub['eligible'] and not stub['reaches_goal'] and stub['corners'] == [False], stub['reason']
    # A leg shorter than the stopping distance: the shortcut from its start.
    short = [[4.8, 2.25, 1.5], [5., 2.25, 1.5], [5., 8., 1.5]]
    assert validator.check(short, short[0], {'scan': open_grid()}, 10.)['corners'] == [True]


def executor(active, pose, heading_deg):
    from dway.executor import DynamicExecutor
    return DynamicExecutor, SimpleNamespace(
        profile=ExecutionProfile.load('sim-target'), pose=pose, segment=0, progress=[0, .4],
        active=dict(geometry_revision=3, points=active), vehicle=SimpleNamespace(heading_deg=heading_deg),
        _tracking_allowance=lambda value, progress: .4, _record=lambda *args: None, reason='')


@pytest.mark.parametrize(('bend', 'taken'), [(.3, True), (1., False)])
def test_dway_judges_a_handoff_by_route_direction_not_its_lagging_heading(bend, taken):
    DynamicExecutor, ex = executor([[0., 5., 1.5], [10., 5., 1.5]], [4., 5., 1.5], heading_deg=200.)
    value = dict(geometry_revision=4, handoff=dict(from_revision=3, start=[0, 0.]),
                 points=[[4., 5., 1.5], [6., 5. + bend, 1.5], [10., 5. + bend, 1.5]],
                 clearance=dict(end=[1, 1.], lateral_m=[.65, .65]))
    # 8.5° from the route being flown is taken, facing 200° or not; 26.6° is refused.
    assert DynamicExecutor._take_handoff(ex, value) is taken
    assert (ex.active['geometry_revision'] == 4) is taken


def test_moving_off_outside_the_allowance_brakes_only_if_the_offset_grows():
    from dway.executor import DEPARTURE_SLACK_M, DynamicExecutor
    # The area1 stall: a corner stop left the vehicle 0.19 m off a leg allowing 0.18 m.
    ex = SimpleNamespace(departure_cross=math.inf)
    limit = DynamicExecutor._cross_limit
    assert limit(ex, .19, .183) == pytest.approx(.19 + DEPARTURE_SLACK_M)
    assert limit(ex, .2, .183) >= .2               # converging within the slack: still flying
    assert limit(ex, .15, .183) == pytest.approx(.183)  # back within the allowance
    assert limit(ex, .19, .183) == pytest.approx(.183)  # so a later excursion brakes
    ex.departure_cross = math.inf
    assert .25 > limit(ex, .19, .183)                   # growing off the route brakes


def test_validate_checks_handoff_and_corner_flags():
    value = json.loads((ROOT/'assets/navigation_examples/ready.json').read_text())
    validate(dict(value, handoff=dict(from_revision=0, start=[0, .5])))
    for bad in (dict(from_revision=-1, start=[0, .5]), dict(from_revision=0, start=[1, 0.])):
        with pytest.raises(ValueError, match='handoff'): validate(dict(value, handoff=bad))
    validate(dict(value, clearance=dict(value['clearance'], corners=[])))
    with pytest.raises(ValueError, match='corner'):
        validate(dict(value, clearance=dict(value['clearance'], corners=[True])))
