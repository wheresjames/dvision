"""The mission runner on in-memory ports: phases, goals, debounce, authority, budgets, reports."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import pytest

from dcmn.mission import MAX_VERTICES, MAX_WAYPOINTS, validate_status
from dcmn.navigation import encode
from dmsn import profiles as P
from dmsn.inputs import Observation
from dmsn.intents import Goal, Wait
from dmsn.missions import MISSIONS, MissionEvent
from dmsn.runner import MissionRun, StartRefused, replay
from dmsn.schema import Field, ParamSchema
from dtest.mission_fixture import FakeExecutor, FakePorts, drive


def goto(*overrides):
    return P.apply_overrides(P.blank('goto'), ['goal_source=xy', 'goal=10,5', *overrides])


@pytest.fixture
def chain(tmp_path):
    def build(profile=None, *, executor=True, **options):
        ports = FakePorts(tmp_path / 'run', **{k: options.pop(k) for k in ('pose', 'coverage') if k in options})
        ex = FakeExecutor(ports) if executor else None
        run = MissionRun('fake', profile or goto(), ports=ports, **options)
        run.step()
        return run, ports, ex
    return build


def intents(ports):
    return [i['kind'] for i in ports.intents]


# -- goto end to end --------------------------------------------------------------------

def test_goto_launches_once_flies_as_the_mission_authority_and_completes(chain):
    run, ports, ex = chain()
    assert run.phase == 'idle' and not ports.intents, 'nothing is sent before Start'
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done')
    assert (run.outcome, run.exit_code()) == ('complete', 0)
    assert intents(ports) == ['launch']
    assert ports.context['authority']['role'] == 'mission'
    assert ports.context['authority']['id'] == run.writer_id
    assert [g['outcome'] for g in run.goals] == ['reached']
    assert [p['phase'] for p in run.phases] == ['idle', 'starting', 'running', 'done']
    assert ports.written and all(validate_status(v) for v in ports.written)
    directory = run.close()
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['outcome'] == 'complete' and summary['profile_digest'] == P.digest(goto())
    assert (directory / 'mission.png').exists() and (directory / 'report.html').exists()
    events = [json.loads(line)['type'] for line in (directory / 'events.jsonl').read_text().splitlines()]
    assert 'mission.start' in events and 'mission.goal' in events and 'mission.done' in events
    assert 'mission.step' not in events, 'per-step records belong to the archive, not the event log'


def test_dway_without_wait_for_mission_is_not_sent_a_launch(chain, tmp_path):
    run, ports, _ = chain(executor=False)
    ex = FakeExecutor(ports, wait_for_mission=False)
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done')
    assert intents(ports) == []


def test_a_dway_failure_before_takeoff_fails_the_mission(chain):
    run, ports, ex = chain()
    run.start()
    drive(run, ex, ports=ports, steps=1)
    ports.execution = dict(ports.execution, state='FAILED', reason='arm rejected')
    ex.step = lambda: None
    assert drive(run, None, ports=ports, until=lambda r: r.phase == 'done')
    assert run.outcome == 'failed' and 'arm rejected' in run.reason


# -- clicks and retargets ---------------------------------------------------------------

def test_a_click_while_idle_edits_the_profile_and_while_flying_retargets_at_once(chain):
    run, ports, ex = chain(P.blank('goto'))
    assert 'goal set' in run.click(3., 4.)
    assert run.document.value['params']['goal'] == [3., 4.] and run.document.dirty
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None)
    assert run.goal_sent['xy'] == (3., 4.)
    ex.flight_steps = 1000
    run.click(20., -3.)
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent['xy'] == (20., -3.), steps=3)
    assert [g['outcome'] for g in run.goals] == ['replaced', 'active']
    assert run.suppressed == 0, 'an operator retarget never waits out the debounce'


# -- debounce -----------------------------------------------------------------------------

class Drift:
    """A strategy-like mission proposing a new, nearby point every step."""
    name: ClassVar[str] = 'drift'
    label: ClassVar[str] = 'drift'
    editors: ClassVar[tuple] = ()
    reference_default: ClassVar[bool] = False
    schema: ClassVar[ParamSchema] = ParamSchema((Field('step_m', 'float', default=0.25),))

    def start(self, params, obs, *, area=(), waypoints=(), strategy=None):
        self.params, self.x = params, 0.
    def on_event(self, event): pass
    def step(self, obs):
        self.x += self.params['step_m']
        return Goal((self.x, 0.), 'drift', operator=False)
    def overlay(self): return []
    def progress(self): return {}
    def summary(self): return {}


def test_strategy_goals_wait_for_the_commit_time_and_the_minimum_shift(chain, monkeypatch):
    monkeypatch.setitem(MISSIONS, 'drift', Drift)
    profile = P.apply_overrides(P.blank('drift'), ['debounce.min_commit_s=2', 'debounce.min_shift_m=1'])
    run, ports, ex = chain(profile)
    ex.flight_steps = 10**6
    run.start()
    drive(run, ex, ports=ports, steps=40, dt=0.2)
    sent = [g['xy'] for g in run.goals]
    assert len(sent) >= 3, sent
    assert run.suppressed > len(sent), 'most proposals were debounced'
    for a, b in zip(run.goals, run.goals[1:]):
        assert b['set_s'] - a['set_s'] >= 2. - 1e-9
        assert math.dist(a['xy'], b['xy']) >= 1. - 1e-9


# -- authority and pause --------------------------------------------------------------------

def test_losing_the_goal_authority_pauses_without_holding_and_never_takes_it_back(chain):
    run, ports, ex = chain()
    ex.flight_steps = 10**6
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None)
    ports.take_goal('dnav:operator', [1., 1.], role='ui')
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'paused', steps=3)
    assert 'goal authority taken by ui' in run.blocking
    assert 'hold' not in intents(ports), 'whoever took the goal is flying it; dmsn does not hold underneath them'
    drive(run, ex, ports=ports, steps=10)
    assert ports.context['authority']['id'] == 'dnav:operator', 'dmsn never takes the authority back on its own'
    run.resume()
    assert ports.context['authority']['id'] == run.writer_id, 'Resume is the explicit handoff'
    assert ports.context['goal']['position'] == [10., 5.]


def test_operator_pause_holds_dway_and_resume_releases_it(chain):
    run, ports, ex = chain()
    ex.flight_steps = 20
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: ports.execution.get('state') == 'EXECUTING')
    run.pause()
    assert intents(ports)[-1] == 'hold'
    drive(run, ex, ports=ports, steps=3)
    assert ports.execution['mission_hold'] and ports.execution['state'] == 'HOLDING'
    run.resume()
    assert intents(ports)[-1] == 'resume'
    assert drive(run, ex, ports=ports, steps=80, until=lambda r: r.phase == 'done')
    assert run.outcome == 'complete'


def test_abort_holds_releases_the_authority_and_ends_aborted(chain):
    run, ports, ex = chain()
    ex.flight_steps = 10**6
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None)
    run.abort()
    assert run.phase == 'done' and run.outcome == 'aborted' and run.exit_code() == 1
    assert intents(ports)[-1] == 'hold' and ports.context['authority'] is None


# -- budgets -----------------------------------------------------------------------------------

def test_a_time_budget_with_hold_ends_the_mission_and_holds(chain):
    run, ports, ex = chain(goto('budgets.time_s=2', 'budgets.time_action=hold'))
    ex.flight_steps = 10**6
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=60)
    assert run.outcome == 'budget' and 'time budget' in run.reason
    assert intents(ports)[-1] == 'hold'


def test_a_path_budget_with_home_flies_home_before_ending(chain):
    run, ports, ex = chain(goto('budgets.path_m=5', 'budgets.path_action=home'))
    ex.flight_steps = 3
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=120)
    assert run.outcome == 'budget' and 'path length budget' in run.reason
    assert [g['tag'] for g in run.goals][-1] == 'home'
    assert tuple(ports.context['goal']['position']) == run.home
    assert 'finishing' in [p['phase'] for p in run.phases]


def test_a_battery_reserve_ends_the_mission(chain):
    run, ports, ex = chain(goto('budgets.battery_reserve_pct=30', 'budgets.battery_action=hold'))
    ports.status = {'drone.battery_pct': '80'}
    ex.flight_steps = 10**6
    run.start()
    drive(run, ex, ports=ports, steps=10)
    assert run.phase == 'running'
    ports.status = {'drone.battery_pct': '25'}
    run._provider_checked = -math.inf
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=20)
    assert run.outcome == 'budget' and 'battery' in run.reason


def test_a_time_budget_with_land_ends_the_mission_on_the_ground(chain):
    run, ports, ex = chain(goto('budgets.time_s=2', 'budgets.time_action=land'))
    ex.flight_steps = 10**6
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=80)
    assert run.outcome == 'budget' and 'time budget' in run.reason
    assert intents(ports)[-1] == 'land' and ex.landings == 1
    assert 'finishing' in [p['phase'] for p in run.phases]
    assert ports.context['authority'] is None, 'on the ground the goal authority is given back'


def test_a_land_finish_action_and_a_failed_landing_says_why(chain):
    run, ports, ex = chain(goto('finish_action=land'))
    ex.land_result = 'failed'
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=80)
    assert run.outcome == 'failed' and run.reason == 'landing failed: landing not confirmed within 60 s'
    assert run.exit_code() == 1


def test_the_land_button_lands_where_the_vehicle_is_and_ends_aborted(chain):
    run, ports, ex = chain()
    assert run.land() == 'nothing to land (idle)'
    ex.flight_steps = 10**6
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None)
    assert run.land() == 'landing'
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=40)
    assert (run.outcome, run.reason) == ('aborted', 'operator landed the vehicle')
    assert intents(ports)[-1] == 'land' and ex.landings == 1


def test_explore_on_the_runner_flies_frontiers_until_none_is_left(chain):
    import numpy
    from dcmn.maps import NEVER_OBSERVED, GridGeometry

    class Grid:
        geometry = GridGeometry(0., 0., 1., 20, 10)
        source, revision = 'stub', 1

        def __init__(self): self.occupancy = numpy.full((10, 20), NEVER_OBSERVED, numpy.uint8)
        def layer(self, _index): return self.occupancy, numpy.zeros(self.occupancy.shape, numpy.uint32)

    profile = P.apply_overrides(P.blank('explore'), ['strategy=frontier', 'area=1,1;19,1;19,9;1,9',
                                                     'debounce.min_commit_s=0', 'coverage_margin_m=0'])
    run, ports, ex = chain(profile, pose=(2.5, 2.5, 1.5, 90.), coverage=(0., 0., 20., 10.))
    grid = Grid()
    grid.occupancy[:, :6] = 0  # the west third observed and free: its east edge is the frontier
    ports._grids['dalg'] = grid
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None, steps=40), run.blocking
    assert run.goal_sent['xy'] == (5.5, 2.5), 'the nearest frontier cell'
    assert 0. < run.observation.area_coverage < 1.
    grid.occupancy[:, :] = 0  # everything observed: nothing left at the edge of the known
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=60), run.blocking
    assert (run.outcome, run.reason) == ('complete', 'no frontier left to explore')
    assert run.status_value(run.observation)['progress']['coverage_pct'] == 100.0
    directory = run.close()
    assert replay(directory / 'archive')['mismatches'] == [], 'an explore run replays decision for decision'


def test_a_goal_explore_gives_up_on_is_archived_with_its_reason_and_replays(chain):
    import numpy
    from dcmn.archive import ArchiveReader
    from dcmn.maps import NEVER_OBSERVED, GridGeometry

    class Grid:
        geometry = GridGeometry(0., 0., 1., 20, 10)
        source, revision = 'stub', 1

        def __init__(self): self.occupancy = numpy.full((10, 20), NEVER_OBSERVED, numpy.uint8)
        def layer(self, _index): return self.occupancy, numpy.zeros(self.occupancy.shape, numpy.uint32)

    profile = P.apply_overrides(P.blank('explore'), ['strategy=frontier', 'area=1,1;19,1;19,9;1,9',
                                                     'debounce.min_commit_s=0', 'coverage_margin_m=0',
                                                     'stuck_s=2', 'budgets.time_s=30'])
    run, ports, ex = chain(profile, pose=(2.5, 2.5, 1.5, 90.), coverage=(0., 0., 20., 10.))
    ex.flight_steps = 10**6  # never arrives: every goal gets stuck
    grid = Grid()
    grid.occupancy[:, :6] = 0
    ports._grids['dalg'] = grid
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=400), run.blocking
    assert run.mission.blacklist and run.mission.blacklist_reasons[0] == 'stuck for 2 s'
    directory = run.close()
    events = [e for e in ArchiveReader(directory / 'archive').events() if e['type'] == 'mission.blacklist']
    assert [e['data']['reason'] for e in events] == run.mission.blacklist_reasons
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['mission_summary']['blacklist'][0]['reason'] == 'stuck for 2 s'
    assert replay(directory / 'archive')['mismatches'] == []


# -- coverage (B1, Q24) ----------------------------------------------------------------------------

def test_goto_without_a_workspace_requests_no_coverage(chain):
    run, ports, ex = chain()
    run.start()
    assert ports.resets == []


def test_a_workspace_is_requested_with_its_margin_before_takeoff(chain):
    run, ports, ex = chain(goto('workspace=0,0,70,40', 'coverage_margin_m=5'), coverage=(-10., -10., 10., 10.))
    run.start()
    assert ports.resets == [dict(bounds=[-5., -5., 75., 45.], by='dmsn', revision=1)]
    drive(run, ex, ports=ports, steps=10)
    assert not ports.intents and run.phase == 'starting'
    assert 'requested mission coverage' in run.blocking
    ports.coverage_bounds = tuple(ports.resets[-1]['bounds'])
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done')


def test_a_refused_coverage_request_never_launches(chain):
    run, ports, ex = chain(goto('workspace=0,0,70,40'), coverage=(-10., -10., 10., 10.))
    def refuse(*args, **kwargs):
        raise ValueError('mapping budget exceeded')
    ports.request_reset = refuse
    run.start()
    drive(run, ex, ports=ports, steps=10)
    assert not ports.intents and 'mapping budget exceeded' in run.blocking


def test_launch_publication_failure_is_retried(chain):
    run, ports, ex = chain()
    publish = ports.write_intent
    def unavailable(value):
        raise RuntimeError('intent plane unavailable')
    ports.write_intent = unavailable
    run.start()
    drive(run, ex, ports=ports, steps=3)
    assert not run.launch_sent
    ports.write_intent = publish
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done')


def test_time_budget_also_bounds_waiting_for_coverage(chain):
    run, ports, ex = chain(goto('workspace=0,0,70,40', 'budgets.time_s=1'), coverage=(-10., -10., 10., 10.))
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done')
    assert run.outcome == 'budget' and 'launch' not in intents(ports)


def test_mapping_rejection_heartbeat_shows_the_allocator_reason(chain):
    from types import SimpleNamespace
    run, ports, ex = chain(goto('workspace=0,0,70,40'), coverage=(-10., -10., 10., 10.))
    run.start()
    ports.inbox.append(SimpleNamespace(type='module.heartbeat', role='algorithm',
                                      payload=dict(reset_revision=1, state='RUNNING',
                                                   reason='mapping reset rejected: maximum side exceeded')))
    drive(run, ex, ports=ports, steps=3)
    assert not ports.intents and 'maximum side exceeded' in run.blocking


def test_lease_loss_while_waiting_for_a_route_pauses_without_taking_control(chain):
    run, ports, ex = chain()
    ex.flight_steps = 10**6
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None)
    ports.execution.update(state='WAITING', owns_control=False, vehicle_fault='control lease lost after waiting')
    before = len(ports.intents)
    drive(run, None, ports=ports, steps=3)
    assert run.phase == 'paused' and 'control lease lost' in run.blocking
    assert len(ports.intents) == before


def test_return_home_retries_a_goal_after_coverage_expansion(chain):
    run, ports, ex = chain()
    ex.flight_steps = 10**6
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None)
    ports.coverage_bounds = (5., 0., 20., 20.)
    drive(run, ex, ports=ports, steps=2)
    run.go_home()
    drive(run, ex, ports=ports, steps=2)
    assert run.phase == 'finishing' and 'outside evidence coverage' in run.blocking
    run.expand_coverage()
    ports.coverage_bounds = tuple(ports.resets[-1]['bounds'])
    ex.flight_steps = 2
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done')
    assert run.goals[-1]['tag'] == 'home' and run.goals[-1]['outcome'] == 'reached'


def test_a_goal_outside_coverage_is_held_back_until_the_operator_expands_it(chain):
    run, ports, ex = chain(goto(), coverage=(-5., -5., 5., 5.))
    run.start()
    drive(run, ex, ports=ports, steps=10)
    assert run.goal_sent is None and run.expand_offer is not None
    assert 'outside evidence coverage' in run.blocking and 'Expand coverage' in run.blocking
    run.expand_coverage()
    assert ports.resets and ports.resets[-1]['bounds'][2] >= 10.
    ports.coverage_bounds = tuple(ports.resets[-1]['bounds'])
    assert drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None, steps=5)


# -- arena (Q4) ------------------------------------------------------------------------------------

def test_a_headless_start_on_the_wrong_arena_is_refused_unless_allowed(chain):
    run, ports, _ = chain(goto('arena=maze_020'), headless=True)
    ports.status = {'sim.map': '/somewhere/maze_012.txt'}
    run.provider = None
    with pytest.raises(StartRefused, match='maze_020.*maze_012'):
        run.start()
    assert run.phase == 'idle' and not ports.intents
    allowed, ports2, _ = chain(goto('arena=maze_020'), headless=True, allow_arena_mismatch=True)
    ports2.status = {'sim.map': 'maze_012.txt'}
    allowed.provider = None
    allowed.start()
    assert allowed.arena['mismatch'] and allowed.arena['allowed']


def test_the_window_warns_about_the_arena_but_starts(chain):
    run, ports, _ = chain(goto('arena=maze_020'))
    ports.status = {'sim.map': 'maze_012.txt'}
    run.provider = None
    run.start()
    assert run.arena['mismatch'] and run.phase == 'starting'


# -- bounded snapshots (B8) ----------------------------------------------------------------------------

def test_the_largest_legal_status_snapshot_fits(chain):
    area = [[100 * math.cos(2 * math.pi * i / MAX_VERTICES), 100 * math.sin(2 * math.pi * i / MAX_VERTICES)]
            for i in range(MAX_VERTICES)]
    waypoints = [[float(i), float(-i)] for i in range(MAX_WAYPOINTS)]
    profile = P.validate(dict(goto(), name='n' * 120, area=area, waypoints=waypoints))
    run, ports, _ = chain(profile)
    run.blocking = run.reason = 'x' * 5000
    value = run.status_value(Observation(time_s=12345.678, writer=run.writer_id))
    validate_status(value)
    assert len(encode(value).encode()) <= 65536


def test_the_largest_explore_status_snapshot_fits(chain):
    from dmsn.inputs import FRONTIER_CAP
    from dmsn.missions.explore import ExploreMission
    area = [[100 * math.cos(2 * math.pi * i / MAX_VERTICES), 100 * math.sin(2 * math.pi * i / MAX_VERTICES)]
            for i in range(MAX_VERTICES)]
    profile = P.validate(dict(P.apply_overrides(P.blank('explore'), ['strategy=wall_follow']), name='n' * 120,
                              area=area))
    run, ports, _ = chain(profile)
    mission = ExploreMission()
    mission.start({}, Observation(), area=[tuple(p) for p in area])
    for i in range(5000): mission._give_up((float(i), 1e6 + i), 'r' * 120)
    mission.frontiers = tuple((float(i), float(i)) for i in range(FRONTIER_CAP))
    run.mission = mission
    run.blocking = run.reason = 'x' * 5000
    value = run.status_value(Observation(time_s=12345.678, writer=run.writer_id))
    validate_status(value)
    assert len(encode(value).encode()) <= 65536


# -- replay (Q29) ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['goto', 'goto_retarget', 'waypoints_skip', 'explore_frontier',
                                  'explore_wall_follow'])
def test_saved_mission_archives_replay_without_changing_decisions(name):
    from dcmn.archive import ArchiveReader
    directory = Path(__file__).parents[1] / 'fixtures' / 'missions' / name
    assert ArchiveReader(directory).validate()['complete']
    result = replay(directory)
    assert result['steps'] > 0 and result['mismatches'] == []

def test_a_recorded_mission_replays_decision_for_decision(chain, monkeypatch):
    run, ports, ex = chain(P.blank('goto'))
    run.click(3., 4.)
    run.start()
    ex.flight_steps = 12
    drive(run, ex, ports=ports, steps=8)
    run.click(-6., 2.)
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=80)
    directory = run.close()
    result = replay(directory / 'archive')
    assert result['steps'] >= 5 and result['mismatches'] == []
    from dmsn.missions.goto import GotoMission
    monkeypatch.setattr(GotoMission, 'step', lambda self, obs: Wait('changed behaviour'))
    assert replay(directory / 'archive')['mismatches'], 'a mission that decides differently must be caught'


def test_events_and_observations_round_trip():
    obs = Observation(time_s=1.5, pose=(1., 2., 1.5, 90.), coverage=(0., 0., 5., 5.), writer='w',
                      execution=dict(state='COMPLETE', authorized_goal=['w', 1, 3]),
                      goal=dict(position=[1., 2.], revision=3, authority='w', role='mission'))
    assert Observation.from_dict(json.loads(json.dumps(obs.as_dict()))) == obs
    assert obs.arrived((1., 2.)) and not obs.arrived((1., 2.1))
    event = MissionEvent('retarget', (3, 4), dict(a=1))
    assert MissionEvent.from_dict(event.as_dict()) == event
