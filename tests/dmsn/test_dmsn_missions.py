"""The missions and strategies as pure logic: scripted observations in, decisions out.

Every mission decides on an :class:`Observation` alone (Q28), so the tests
build observations by hand -- poses, arrivals, frontier lists, coverage -- and
check the decision each one produces: the waypoint policies, the frontier and
wall-follow proposals, every way explore can end, and the tour conversion.
The runner's evidence distillation is here too: a stub grid in, the bounded
frontier list and the coverage fraction out.
"""
from __future__ import annotations

import json
import math
import uuid

import numpy
import pytest

from dcmn.maps import NEVER_OBSERVED, GridGeometry
from dcmn.mission import MAX_WAYPOINTS
from dmsn import profiles as P
from dmsn.inputs import FRONTIER_CAP, Observation
from dmsn.intents import Done, Goal
from dmsn.missions.explore import ExploreMission
from dmsn.missions.waypoints import WaypointsMission
from dmsn.strategies import build
from dmsn.tour_import import convert, convert_file
from dmsn.runner import MissionRun
from dtest.mission_fixture import FakePorts

WRITER = 'test'
AREA = [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]
WAYPOINTS = [(10., 0.), (20., 0.)]


def obs(xy=None, *, time_s=0., heading=0., frontiers=(), coverage=None, walls=(), extent=None, **fields):
    """An observation with a pose, and whatever evidence the runner would distil."""
    return Observation(time_s=time_s,
                       pose=None if xy is None else (xy[0], xy[1], 1.0, heading),
                       writer=WRITER, frontiers=tuple(frontiers), area_coverage=coverage,
                       walls=tuple(walls), coverage=extent, **fields)


def refused_obs(xy, goal, *, time_s, status='goal_unreachable', frontiers=(),
                route_mode='planned', eligible=False, execution=None):
    """The vehicle at ``xy`` while dnav refuses to plan to this coordinator's ``goal``.

    ``route_mode='substitute'`` with ``eligible`` is dnav offering a route to a
    point near the goal; ``execution`` is dway's state, which says whether that
    route has been flown yet.
    """
    return obs(xy, time_s=time_s, frontiers=frontiers,
               goal={'authority': WRITER, 'position': [goal[0], goal[1]], 'revision': 3},
               execution=execution or {},
               navigation={'planning_status': status, 'goal_revision': 3, 'reason': 'blocked',
                           'route_mode': route_mode, 'eligible': eligible})


#: dway holding at the end of a substitute route. Its arrival gate is measured to
#: the goal, and a substitute ends up to goal_substitute_m away, so this is an
#: ordinary prefix hold -- never COMPLETE (docs/navigation.md).
SHORT_OF_GOAL = {'state': 'HOLDING', 'hold_kind': 'prefix'}


def arrived_obs(xy, *, time_s=0., frontiers=(), coverage=None):
    """An observation in which dway reports COMPLETE for this coordinator's goal."""
    x, y = float(xy[0]), float(xy[1])
    return Observation(time_s=time_s, pose=(x, y, 1.0, 0.), writer=WRITER,
                       goal={'authority': WRITER, 'position': [x, y], 'revision': 7},
                       execution={'state': 'COMPLETE', 'authorized_goal': [x, y, 7]},
                       frontiers=tuple(frontiers), area_coverage=coverage)


def waypoints_mission(policy, *, retries=0, stuck_s=20., waypoints=WAYPOINTS):
    mission = WaypointsMission()
    mission.start({'failure_policy': policy, 'retries': retries, 'stuck_s': stuck_s},
                  obs((0., 0.)), waypoints=waypoints)
    return mission


def explore_mission(*, target=None, stuck_s=20., strategy='frontier', area=AREA):
    mission = ExploreMission()
    built = None if strategy is None else build(strategy)
    mission.start({'coverage_target_pct': target, 'stuck_s': stuck_s}, obs((0., 0.)),
                  area=area, strategy=built)
    return mission


# -- the waypoint policies -----------------------------------------------------------------

def test_a_stuck_waypoint_aborts_the_mission():
    mission = waypoints_mission('abort')
    decision = mission.step(obs((0., 0.)))  # seeds the stuck clock and asks for waypoint 1
    assert isinstance(decision, Goal) and decision.xy == (10., 0.) and 'waypoint 1' in decision.tag
    assert isinstance(mission.step(obs((5., 0.), time_s=10.)), Goal), 'closer: progress'
    decision = mission.step(obs((5., 0.), time_s=35.))  # no progress for 25 s
    assert isinstance(decision, Done) and decision.outcome == 'failed'
    assert 'waypoint 1' in decision.reason and 'unreachable' in decision.reason
    assert mission.states[0] == 'failed'
    assert mission.progress()['waypoint_states'] == ['failed', 'pending']


def test_the_skip_policy_drops_the_point_and_carries_on():
    mission = waypoints_mission('skip')
    mission.step(obs((0., 0.)))
    decision = mission.step(obs((0., 0.), time_s=21.))  # stuck: dropped, next point
    assert isinstance(decision, Goal) and decision.xy == (20., 0.) and 'waypoint 2' in decision.tag
    assert mission.states[0] == 'skipped' and mission.skipped == 1
    decision = mission.step(arrived_obs((20., 0.), time_s=40.))
    assert isinstance(decision, Done) and decision.outcome == 'complete'
    assert decision.reason == 'visited 1 waypoint, skipped 1'


def test_the_retry_policy_tries_again_then_gives_up():
    mission = waypoints_mission('retry', retries=1)
    mission.step(obs((0., 0.)))
    decision = mission.step(obs((0., 0.), time_s=21.))  # stuck once: the same point again
    assert isinstance(decision, Goal) and decision.xy == (10., 0.)
    assert 'waypoint 1' in decision.tag and 'retry 1' in decision.tag
    decision = mission.step(obs((0., 0.), time_s=42.))  # stuck again: retries spent
    assert isinstance(decision, Goal) and decision.xy == (20., 0.) and 'waypoint 2' in decision.tag
    assert mission.states[0] == 'skipped' and mission.progress()['waypoints_skipped'] == 1


def test_arriving_at_each_waypoint_walks_the_list_to_completion():
    mission = waypoints_mission('abort')
    assert mission.step(obs((0., 0.))).xy == (10., 0.)
    decision = mission.step(arrived_obs((10., 0.), time_s=10.))
    assert isinstance(decision, Goal) and decision.xy == (20., 0.) and mission.states[0] == 'visited'
    decision = mission.step(arrived_obs((20., 0.), time_s=20.))
    assert isinstance(decision, Done) and decision.outcome == 'complete'
    assert decision.reason == 'visited 2 waypoints'
    assert mission.progress() == dict(waypoint_states=['visited', 'visited'], waypoint_index=None,
                                      waypoints_done=2, waypoints_skipped=0)


def test_no_waypoints_is_a_failed_mission():
    mission = waypoints_mission('abort', waypoints=())
    decision = mission.step(obs((0., 0.)))
    assert isinstance(decision, Done) and decision.outcome == 'failed'
    assert decision.reason == 'no waypoints in the profile'


def test_a_long_leg_with_steady_progress_never_counts_as_stuck():
    mission = waypoints_mission('abort', waypoints=[(100., 0.)])
    for step in range(8):  # 10 s per observation, 5 m closer every time
        decision = mission.step(obs((float(step * 5), 0.), time_s=10. * step))
        assert isinstance(decision, Goal) and decision.xy == (100., 0.)


@pytest.mark.parametrize('mission', ['waypoints', 'explore'])
def test_a_leg_flown_at_the_profile_speed_never_counts_as_stuck(mission):
    """The rate a vehicle actually reports at: 1 m/s sampled at 10 Hz, which is
    0.1 m per step -- under PROGRESS_M. The anchor has to hold across steps for
    that to read as progress; when it was pulled down every step instead, the
    clock never reset and a leg longer than ``stuck_s`` was always called stuck.
    The 17 Sep area1 run gave up a goal it had closed 14 m on.
    """
    goal = (60., 0.)
    body = (waypoints_mission('abort', waypoints=[goal]) if mission == 'waypoints'
            else explore_mission(area=[(0., -5.), (70., -5.), (70., 5.), (0., 5.)]))
    if mission == 'explore':
        assert body.step(obs((0., 0.), frontiers=(goal,))).xy == goal
    for step in range(400):           # 40 s at 10 Hz, twice stuck_s
        at = (step * 0.1, 0.)
        decision = body.step(obs(at, time_s=step * 0.1, frontiers=(goal,)))
        assert isinstance(decision, Goal) and decision.xy == goal, f'given up at {at[0]:.1f} m'


def test_the_stuck_clock_resets_on_cumulative_progress_not_per_step_progress():
    """What the anchor is for. Creeping 0.05 m a step never clears PROGRESS_M in
    one step, but five of them do clear it from the anchor -- and that is the
    reset. The clock then runs from there, not from the first observation."""
    mission = waypoints_mission('abort', waypoints=[(50., 0.)], stuck_s=20.)
    for step in range(6):                      # 0.06 m per step: past PROGRESS_M by the fifth
        decision = mission.step(obs((step * 0.06, 0.), time_s=float(step)))
        assert isinstance(decision, Goal), step
    # The anchor moved at the step that closed PROGRESS_M (t=5), so the clock
    # runs from there rather than from the first observation.
    held = mission.step(obs((0.3, 0.), time_s=24.))
    assert isinstance(held, Goal), 'given up before stuck_s had run from the reset'
    decision = mission.step(obs((0.3, 0.), time_s=26.))
    assert isinstance(decision, Done) and 'no progress' in decision.reason


# -- the goal strategies -------------------------------------------------------------------

def test_the_frontier_strategy_takes_the_nearest_reachable_frontier():
    strategy = build('frontier')
    candidate = strategy.propose(obs((1., 1.), frontiers=((9., 9.), (2., 2.), (50., 50.))), AREA, ())
    assert candidate.xy == (2., 2.) and 'nearest frontier' in candidate.reason
    assert (50., 50.) not in candidate.xy, 'a frontier outside the area is never proposed'


def test_the_frontier_strategy_avoids_the_blacklist_and_reports_no_pose():
    strategy = build('frontier')
    assert strategy.propose(obs((1., 1.), frontiers=((9., 9.), (2., 2.))), AREA,
                            ((9., 9.),)) .xy == (2., 2.), 'a blacklisted dead end is skipped'
    assert strategy.propose(obs((1., 1.), frontiers=((9., 9.),)), AREA, ((9., 9.),)) is None
    assert strategy.propose(obs(frontiers=((2., 2.),)), AREA, ()) .reason == 'nearest frontier (no pose yet)'
    assert strategy.propose(obs((1., 1.), frontiers=((50., 50.),)), AREA, ()) is None


def test_wall_following_carries_on_until_a_wall_and_turns_left_at_the_area_edge():
    strategy = build('wall_follow', {'step_m': 4.})
    area = [(0., -10.), (10., -10.), (10., 10.), (0., 10.)]
    candidate = strategy.propose(obs((5., 0.), heading=0.), area, ())  # north, no wall: straight on
    assert candidate.xy == pytest.approx((5., -4.)) and 'no wall in reach' in candidate.reason
    candidate = strategy.propose(obs((5., -9.), heading=0.), area, ())  # the edge: right hand turns left
    assert candidate.xy[1] > -10. and candidate.xy[0] < 5. and 'turned 90° left' in candidate.reason
    assert strategy.overlay() == [], 'no wall followed: nothing to mark'


def test_wall_following_keeps_the_wall_on_its_hand_at_the_standoff():
    right = build('wall_follow', {'step_m': 4., 'standoff_m': 1.5})
    wall = [(8., y) for y in (-1., 0., 1.)]  # a wall east of the vehicle, 3 m away
    candidate = right.propose(obs((5., 0.), heading=90., walls=wall), (), ())
    assert candidate.xy == pytest.approx((6.5, -4.)), 'east wall on the right: travel north, closing to 1.5 m'
    assert 'right-hand wall 3.0 m away' in candidate.reason and 'heading 0' in candidate.reason
    assert len(right.overlay()) == 3, 'the followed wall, the side line and the hand are drawn'
    left = build('wall_follow', {'step_m': 4., 'hand': 'left', 'standoff_m': 3.})
    candidate = left.propose(obs((5., 0.), heading=90., walls=wall), (), ())
    assert candidate.xy == pytest.approx((5., 4.)), 'east wall on the left: travel south, already at standoff'


def test_wall_following_turns_away_from_a_blocked_step_and_reports_boxed_in():
    strategy = build('wall_follow', {'step_m': 4.})
    blocked = strategy.propose(obs((5., 0.), heading=0., walls=[(5., -4.)]), (), ())
    assert blocked is not None, 'a wall on the step itself is turned away from'
    assert math.dist(blocked.xy, (5., -4.)) >= 0.75
    assert build('wall_follow').propose(obs(), AREA, ()) is None, 'no pose: nothing to step from'
    tiny = [(4., -6.), (6., -4.), (4., -4.)]
    assert build('wall_follow', {'step_m': 4.}).propose(obs((5., -5.)), tiny, ()) is None, \
        'every turn leaves the area: boxed in'
    with pytest.raises(ValueError): build('wall_follow', {'step_m': 100.})
    with pytest.raises(ValueError): build('wall_follow', {'hand': 'both'})
    with pytest.raises(ValueError): build('nonsense')


# -- explore: every way it can end ----------------------------------------------------------

def test_explore_without_an_area_is_a_failed_mission():
    decision = explore_mission(area=()).step(obs((0., 0.)))
    assert isinstance(decision, Done) and decision.outcome == 'failed'
    assert decision.reason == 'no area in the profile'


def test_explore_ends_at_the_coverage_target():
    mission = explore_mission(target=90.)
    decision = mission.step(obs((0., 0.), frontiers=((10., 10.),), coverage=0.92))
    assert isinstance(decision, Done) and decision.outcome == 'complete'
    assert 'area coverage 92% reached the 90% target' in decision.reason
    assert mission.progress()['coverage_pct'] == 92.0


def test_explore_ends_when_no_frontier_is_left():
    decision = explore_mission().step(obs((0., 0.), frontiers=()))
    assert isinstance(decision, Done) and decision.outcome == 'complete'
    assert decision.reason == 'no frontier left to explore'


def test_a_stuck_goal_is_blacklisted_and_the_mission_ends_when_nothing_is_reachable():
    mission = explore_mission()
    assert isinstance(mission.step(obs((0., 0.), frontiers=((10., 0.),))), Goal)
    decision = mission.step(obs((0., 0.), time_s=21., frontiers=((10., 0.),)))
    assert isinstance(decision, Done) and decision.outcome == 'complete'
    assert decision.reason == 'no reachable goal left: frontier proposes none (1 blacklisted)'
    assert mission.blacklist == [(10., 0.)]
    assert mission.progress()['blacklist'] == [dict(xy=[10., 0.], reason='stuck for 20 s')]
    assert mission.progress()['blacklist_count'] == 1


def test_arriving_at_a_frontier_proposes_the_next_one():
    mission = explore_mission()
    assert mission.step(obs((0., 0.), frontiers=((10., 0.),))).xy == (10., 0.)
    # the arrived cell is observed now, so the next observation no longer lists it
    decision = mission.step(arrived_obs((10., 0.), time_s=30., frontiers=((8., 8.),)))
    assert isinstance(decision, Goal) and decision.xy == (8., 8.)
    assert mission.proposals == 2


def test_a_stuck_goal_is_blacklisted_and_the_next_frontier_is_proposed():
    mission = explore_mission()
    assert mission.step(obs((0., 0.), frontiers=((10., 0.), (8., 8.)))).xy == (10., 0.)
    decision = mission.step(obs((0., 0.), time_s=21., frontiers=((10., 0.), (8., 8.))))
    assert isinstance(decision, Goal) and decision.xy == (8., 8.), \
        f'the blacklisted point is not proposed again (blacklist {mission.blacklist})'
    assert mission.blacklist == [(10., 0.)] and mission.proposals == 2


def test_explore_reports_what_the_last_observation_said():
    mission = explore_mission()
    mission.step(obs((0., 0.), frontiers=((10., 0.), (20., 0.)), coverage=0.5))
    assert mission.progress()['frontier_count'] == 2
    assert mission.progress()['committed'] == [10., 0.]
    assert mission.summary()['strategy'] == 'frontier'


def test_a_proposal_outside_evidence_coverage_is_blacklisted_not_flown():
    mission = explore_mission()
    decision = mission.step(obs((4., 4.), frontiers=((6., 6.), (1., 1.)), extent=(0., 0., 5., 5.)))
    assert isinstance(decision, Goal) and decision.xy == (1., 1.), 'the nearest frontier is outside: the next'
    assert mission.progress()['blacklist'] == [dict(xy=[6., 6.], reason='outside coverage')]


def test_a_transient_refusal_is_blacklisted_only_after_unreachable_s():
    """``no_route`` can pass: a plan made for the previous revision, a vehicle
    still braking. That is what the timer is for."""
    mission = explore_mission()
    frontiers = ((9., 0.), (0., 10.))
    assert mission.step(obs((0., 0.), frontiers=frontiers)).xy == (9., 0.)
    held = mission.step(refused_obs((0.5, 0.), (9., 0.), time_s=1., status='no_route', frontiers=frontiers))
    assert isinstance(held, Goal) and held.xy == (9., 0.), 'a transient refusal must last before it counts'
    decision = mission.step(refused_obs((0.5, 0.), (9., 0.), time_s=6.5, status='no_route', frontiers=frontiers))
    assert isinstance(decision, Goal) and decision.xy == (0., 10.)
    assert mission.progress()['blacklist'] == [dict(xy=[9., 0.], reason='dnav: no_route')]


def test_a_verdict_about_the_goal_itself_is_given_up_as_soon_as_it_is_confirmed():
    """An obstacle, its margin or ground off the map will not become reachable
    by being waited on -- in the 17 Sep area1 run, nine frontier goals each
    cost five seconds of waiting out a fact that was never going to change.
    One observation is not enough, though: the usual mechanism is a single
    never-observed cell beside a wall counting as solid, and one frame of that
    must not cost a frontier permanently."""
    from dmsn.inputs import GOAL_REFUSAL_S

    for status in ('goal_unreachable', 'outside_coverage'):
        mission = explore_mission()
        frontiers = ((9., 0.), (0., 10.))
        assert mission.step(obs((0., 0.), frontiers=frontiers)).xy == (9., 0.)
        held = mission.step(refused_obs((0.5, 0.), (9., 0.), time_s=0.1, status=status,
                                        frontiers=frontiers))
        assert isinstance(held, Goal) and held.xy == (9., 0.), f'{status} acted on one frame'
        assert not mission.blacklist
        decision = mission.step(refused_obs((0.5, 0.), (9., 0.), time_s=0.1 + GOAL_REFUSAL_S,
                                            status=status, frontiers=frontiers))
        assert isinstance(decision, Goal) and decision.xy == (0., 10.), status
        assert mission.progress()['blacklist'] == [dict(xy=[9., 0.], reason=f'dnav: {status}')]
        assert GOAL_REFUSAL_S < 1., 'confirming has to stay far cheaper than waiting it out'


def test_a_refusal_does_not_count_while_a_substitute_is_closing_on_the_goal():
    """dnav answers an unreachable goal with a route to a point near it. While
    the vehicle is flying that route there is somewhere to go, so the refusal
    does not count however long it takes -- but the moment it stops closing,
    the clock is what ends it. Suppression with no bound would pin the mission
    to an unreachable goal for ever."""
    mission = explore_mission()
    frontiers = ((9., 0.), (0., 10.))
    assert mission.step(obs((0., 0.), frontiers=frontiers)).xy == (9., 0.)

    def offered(x, time_s, execution=None):
        return refused_obs((x, 0.), (9., 0.), time_s=time_s, frontiers=frontiers,
                           route_mode='substitute', eligible=True,
                           execution=execution or {'state': 'EXECUTING'})

    for step in range(15):                       # 0.5 m every 2 s, well past unreachable_s
        decision = mission.step(offered(0.5 + step * 0.5, 2. * step))
        assert isinstance(decision, Goal) and decision.xy == (9., 0.), step
        assert not mission.blacklist, 'given up while the substitute was closing'
    # Stopped closing, two metres short -- where a substitute ends. Now the
    # clock runs, and unreachable_s later the goal is given up.
    held = mission.step(offered(7.5, 32.))
    assert isinstance(held, Goal) and held.xy == (9., 0.) and not mission.blacklist
    decision = mission.step(offered(7.5, 38.))
    assert isinstance(decision, Goal) and decision.xy == (0., 10.)
    assert mission.progress()['blacklist'] == [dict(
        xy=[9., 0.], reason='dnav: goal_unreachable; the route toward it stopped closing on it')]


def test_a_retained_route_in_flight_suppresses_the_refusal_too():
    """dnav has two fallbacks, and `retained` is the one that means *keep going*:
    the last good route, permitted as far as it is still clear, because getting
    closer is how a distant phantom clears. The 17 Sep 14:27 area1 run lost a
    confirmed pad to a two-second refusal it was already flying through."""
    mission = explore_mission()
    frontiers = ((9., 0.), (0., 10.))
    assert mission.step(obs((0., 0.), frontiers=frontiers)).xy == (9., 0.)
    # Creeping out of a stop at 0.2 m/s: never PROGRESS_M inside the window a
    # goal verdict allows, which is what made the real run fatal.
    for step in range(45):
        held = mission.step(refused_obs((0.5 + step * 0.02, 0.), (9., 0.), time_s=0.1 * step,
                                        frontiers=frontiers, route_mode='retained', eligible=True,
                                        execution={'state': 'EXECUTING'}))
        assert isinstance(held, Goal) and held.xy == (9., 0.), step
        assert not mission.blacklist, 'given up while still flying a permitted route'
    # dnav plans it again, as it did there after two seconds: nothing was lost.
    cleared = mission.step(obs((3., 0.), time_s=4.5, frontiers=frontiers,
                               goal={'authority': WRITER, 'position': [9., 0.], 'revision': 3},
                               navigation={'planning_status': 'ok', 'goal_revision': 3,
                                           'route_mode': 'planned', 'eligible': True}))
    assert isinstance(cleared, Goal) and cleared.xy == (9., 0.) and not mission.blacklist


def test_a_substitute_that_has_been_flown_ends_the_goal_at_once():
    """dway measures arrival against the goal, so a substitute's end holds with
    ``prefix`` instead of completing. For a substitute that hold is the answer:
    as close as the policy allows.

    This is the window before dnav re-plans and its own fixed point stops it
    offering the substitute; after that the goal-verdict path ends the same
    goal half a second later with a plainer reason.
    """
    mission = explore_mission()
    frontiers = ((9., 0.), (0., 10.))
    assert mission.step(obs((0., 0.), frontiers=frontiers)).xy == (9., 0.)
    flown = refused_obs((7.2, 0.), (9., 0.), time_s=8., frontiers=frontiers,
                        route_mode='substitute', eligible=True, execution=SHORT_OF_GOAL)
    decision = mission.step(flown)
    assert isinstance(decision, Goal) and decision.xy == (0., 10.)
    assert mission.progress()['blacklist'] == [
        dict(xy=[9., 0.], reason='dnav: goal_unreachable; flew to the nearest reachable point instead')]


def test_the_same_hold_without_a_substitute_is_not_an_answer():
    """``arrival-unconfirmed`` on an ordinary route means a vehicle stopped just
    outside the gate, which is transient and keeps its timer."""
    mission = explore_mission()
    frontiers = ((9., 0.), (0., 10.))
    assert mission.step(obs((0., 0.), frontiers=frontiers)).xy == (9., 0.)
    held = mission.step(refused_obs((8.7, 0.), (9., 0.), time_s=1., status='no_route',
                                    frontiers=frontiers, execution=SHORT_OF_GOAL))
    assert isinstance(held, Goal) and held.xy == (9., 0.) and not mission.blacklist


def test_a_stale_refusal_of_another_goal_revision_is_not_blamed_on_this_one():
    mission = explore_mission()
    assert mission.step(obs((0., 0.), frontiers=((9., 0.),))).xy == (9., 0.)
    stale = dict(planning_status='goal_unreachable', goal_revision=2)
    goal = {'authority': WRITER, 'position': [9., 0.], 'revision': 3}
    for t in (1., 10.):
        decision = mission.step(obs((0.5 + t / 10., 0.), time_s=t, frontiers=((9., 0.),), goal=goal, navigation=stale))
        assert isinstance(decision, Goal) and not mission.blacklist


def test_the_status_snapshot_carries_a_bounded_blacklist():
    from dmsn.missions.explore import BLACKLIST_SHOWN
    mission = explore_mission()
    for i in range(BLACKLIST_SHOWN + 10): mission._give_up((float(i), 0.), 'test')
    assert len(mission.progress()['blacklist']) == BLACKLIST_SHOWN
    assert mission.progress()['blacklist_count'] == BLACKLIST_SHOWN + 10
    assert len(mission.summary()['blacklist']) == BLACKLIST_SHOWN + 10


def test_a_waypoint_dnav_refuses_is_given_up_under_the_policy():
    mission = waypoints_mission('skip', stuck_s=1000.)
    mission.step(obs((0., 0.)))
    mission.step(refused_obs((0., 0.), WAYPOINTS[0], time_s=1., status='no_route'))
    decision = mission.step(refused_obs((0.1, 0.), WAYPOINTS[0], time_s=7., status='no_route'))
    assert isinstance(decision, Goal) and decision.xy == WAYPOINTS[1]
    assert mission.states[0] == 'skipped'
    aborted = waypoints_mission('abort', stuck_s=1000.)
    aborted.step(obs((0., 0.)))
    aborted.step(refused_obs((0., 0.), WAYPOINTS[0], time_s=1.))
    decision = aborted.step(refused_obs((0., 0.), WAYPOINTS[0], time_s=7.))
    assert isinstance(decision, Done) and decision.outcome == 'failed'
    assert 'waypoint 1' in decision.reason and 'dnav reports goal_unreachable for 6 s' in decision.reason


# -- the tour import ------------------------------------------------------------------------

def tour_document(waypoints=None, **overrides):
    tour = dict(tour_id='maze_020.default.v1', map='assets/maps/maze_020.txt', map_sha='0f' * 16,
                waypoints=[{'x': 1, 'y': 2, 'h': 90}, {'x': 3, 'y': 4, 'h': 180}]
                if waypoints is None else waypoints)
    tour.update(overrides)
    return tour


def test_a_tour_becomes_a_waypoints_profile_that_ends_in_a_landing():
    profile = convert(tour_document())
    assert profile['mission'] == 'waypoints' and profile['waypoints'] == [[1., 2.], [3., 4.]]
    assert profile['finish_action'] == 'land' and profile['arena'] == 'maze_020'
    from dmsn.tour_import import digest
    assert profile['imported_from'] == dict(kind='tour', tour_id='maze_020.default.v1',
                                            digest=digest(tour_document()),
                                            map='assets/maps/maze_020.txt', map_sha='0f' * 16,
                                            waypoints=2)
    assert digest(tour_document()) != digest(tour_document(map_sha='1e' * 16)), 'the digest names the tour'
    assert profile['assisted']['tour_import'] is True
    assert profile['name'] == 'maze_020.default.v1'
    assert convert(tour_document(), name='renamed')['name'] == 'renamed'


def test_a_bad_tour_is_refused_with_a_reason():
    for tour, reason in (
            (tour_document(waypoints=[]), 'no waypoints'),
            (tour_document(waypoints=[{'x': 1}]), 'no numeric x,y'),
            (tour_document(waypoints=[{'x': 'here', 'y': 2}]), 'no numeric x,y'),
            (tour_document(waypoints=[{'x': 1, 'y': 2}] * (MAX_WAYPOINTS + 1)), 'at most'),
            (tour_document(map=''), 'names no map')):
        with pytest.raises(ValueError, match=reason):
            convert(tour)


def test_convert_file_writes_a_loadable_profile(tmp_path):
    tour_path = tmp_path / 'circuit.default.v1.json'
    tour_path.write_text(json.dumps(tour_document()), encoding='utf-8')
    out_path = tmp_path / 'user' / 'circuit.json'
    written, profile = convert_file(tour_path, out_path)
    assert written == out_path and profile['waypoints'] == [[1., 2.], [3., 4.]]
    assert json.loads(out_path.read_text(encoding='utf-8'))['waypoints'] == [[1., 2.], [3., 4.]]
    assert P.validate(json.loads(out_path.read_text(encoding='utf-8')))['finish_action'] == 'land'
    broken = tmp_path / 'broken.json'
    broken.write_text('not json', encoding='utf-8')
    with pytest.raises(ValueError, match='not JSON'):
        convert_file(broken, tmp_path / 'out.json')
    with pytest.raises(ValueError, match='no tour at'):
        convert_file(tmp_path / 'missing.json', tmp_path / 'out.json')


def test_the_converter_child_runs_without_the_isolation_hook_and_nothing_else_changes(tmp_path):
    import os
    from dmsn.tour_import import converter_env
    hook = tmp_path / 'isolation'
    base = dict(PATH='/bin', PYTHONPATH=os.pathsep.join([str(hook), '/src/apps']),
                DVISION2_ISOLATION_LOG=str(hook / 'violations.log'), DVISION2_FORBID_DIRS='/x',
                DVISION2_FORBID_MODULES='a,b', DVISION2_MISSION_DIR='/missions')
    assert converter_env(base) == dict(PATH='/bin', PYTHONPATH='/src/apps', DVISION2_MISSION_DIR='/missions')
    assert converter_env(dict(PATH='/bin', PYTHONPATH='/src')) == dict(PATH='/bin', PYTHONPATH='/src'), \
        'no hook: the environment is passed through untouched'


# -- the runner's evidence distillation -----------------------------------------------------

class StubGrid:
    """The one thing the runner asks of an evidence grid: a geometry and a layer."""
    def __init__(self, occupancy, geometry):
        self.geometry = geometry
        self.occupancy = occupancy
        self.source, self.revision = 'stub', 1

    def layer(self, _index):
        return self.occupancy, numpy.zeros(self.occupancy.shape, dtype=numpy.uint32)


@pytest.fixture(autouse=True)
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('DVISION2_MISSION_DIR', str(tmp_path / 'user'))
    return tmp_path / 'user'


def evidence_run(tmp_path):
    ports = FakePorts(tmp_path / 'run', instance='ev-' + uuid.uuid4().hex[:8], coverage=None)
    name = 'maze_020-explore'
    profile, path = P.load(name)[0], P.resolve(name)
    return MissionRun(ports.instance, profile, profile_path=path, ports=ports), ports


def test_frontiers_are_free_cells_touching_the_unknown_and_coverage_is_the_observed_share(tmp_path):
    run, ports = evidence_run(tmp_path)
    try:
        geometry = GridGeometry(0., 0., 1., 4, 4)
        occupancy = numpy.full((4, 4), NEVER_OBSERVED, numpy.uint8)
        occupancy[0, :] = 0  # the near row observed and free; the rest never observed
        ports._grids['dalg'] = StubGrid(occupancy, geometry)
        area = [[0., 0.], [4., 0.], [4., 4.], [0., 4.]]
        frontiers, coverage, walls = run._evidence_view(area, (0.5, 0.5, 1., 0.))
        assert frontiers == ((0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5))
        assert coverage == 0.25  # 4 of the area's 16 cells observed
        assert run._evidence_view(area, None)[1] == 0.25, 'no pose: still measured, unordered'
        ports._grids.clear()
        assert run._evidence_view(area, (0.5, 0.5, 1., 0.)) == ((), None, ()), 'no evidence yet'
        assert run._evidence_view((), (0.5, 0.5, 1., 0.)) == ((), None, ()), 'no area: not measured'
    finally:
        run.close()


def test_the_frontier_list_is_capped_at_frontier_cap_nearest_first(tmp_path):
    run, ports = evidence_run(tmp_path)
    try:
        geometry = GridGeometry(0., 0., 1., 20, 20)
        occupancy = numpy.full((20, 20), NEVER_OBSERVED, numpy.uint8)
        occupancy[::2, :] = 0  # every other row free, each touching unknown rows: 200 frontiers
        ports._grids['dalg'] = StubGrid(occupancy, geometry)
        area = [[0., 0.], [20., 0.], [20., 20.], [0., 20.]]
        frontiers, _, _ = run._evidence_view(area, (0.5, 0.5, 1., 0.))
        assert len(frontiers) == FRONTIER_CAP
        assert frontiers[0] == (0.5, 0.5), 'nearest first'
        assert all(numpy.isfinite(frontiers).ravel())
    finally:
        run.close()

def test_walls_are_the_occupied_cells_nearest_the_vehicle_and_the_blacklist_hides_no_frontier(tmp_path):
    from dmsn.inputs import FRONTIER_EXCLUDE_M, WALL_RADIUS_M
    run, ports = evidence_run(tmp_path)
    try:
        geometry = GridGeometry(0., 0., 1., 40, 4)
        occupancy = numpy.full((4, 40), NEVER_OBSERVED, numpy.uint8)
        occupancy[0, :] = 0          # a free corridor along the top row, unknown below: all frontier
        occupancy[3, :] = 254        # and a wall along the bottom row
        ports._grids['dalg'] = StubGrid(occupancy, geometry)
        area = [[0., 0.], [40., 0.], [40., 4.], [0., 4.]]
        frontiers, _, walls = run._evidence_view(area, (0.5, 0.5, 1., 0.))
        assert walls[0] == (0.5, 3.5), 'nearest occupied cell first'
        assert all(math.dist(w, (0.5, 0.5)) <= WALL_RADIUS_M for w in walls)
        assert run._evidence_view(area, None)[2] == (), 'no pose: no walls'
        excluded, _, _ = run._evidence_view(area, (0.5, 0.5, 1., 0.), exclude=((0.5, 0.5),))
        assert all(math.dist(p, (0.5, 0.5)) > FRONTIER_EXCLUDE_M for p in excluded)
        assert excluded[0] == (3.5, 0.5), 'the next live frontier moves up instead of being hidden'
    finally:
        run.close()
