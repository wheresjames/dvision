"""The search mission: scripted detection observations in, decisions out; then on the runner.

Search runs no detector. Its observations carry the detections plane's
estimates of its label, and these tests script them: a target that is found,
approached from a standoff and landed on; one that is never confirmed; one that
goes stale; one that is not seen again from the standoff. On the runner: Start
refuses without a publisher of the label, a mission with no target never opens
the plane (Q27), and a whole search replays decision for decision.
"""
from __future__ import annotations

import json
import math

import pytest

from dcmn import detections as D
from dmsn import profiles as P
from dmsn.inputs import Observation
from dmsn.intents import Action, Done, Goal, Wait
from dmsn.missions.search import SearchMission
from dmsn.runner import MissionRun, StartRefused, replay
from dmsn.strategies import build
from dtest.mission_fixture import FakeExecutor, FakePorts, drive

WRITER = 'test'
AREA = [(0., -10.), (30., -10.), (30., 10.), (0., 10.)]
PAD = (20., 0.)


def estimate(xy=PAD, *, id='landing_pad-1', confirmed=True, stale=False, count=5, revision=5, last_s=0.,
             confidence=0.8, spread=0.2):
    return dict(id=id, label='landing_pad', detector='front-red_disc', xy_m=list(xy), spread_m=spread, count=count,
                revision=revision, confirmed=confirmed, stale=stale, first_s=0., last_s=last_s, confidence=confidence)


def obs(xy=(0., 0.), *, time_s=0., estimates=(), frontiers=((5., 5.),), arrived=None, ack=None):
    """An observation; ``arrived`` makes dway report COMPLETE for this coordinator's goal there."""
    execution, goal = {}, None
    if arrived is not None:
        goal = {'authority': WRITER, 'position': list(arrived), 'revision': 7}
        execution = {'state': 'COMPLETE', 'authorized_goal': [arrived[0], arrived[1], 7]}
        xy = arrived
    if ack is not None: execution['mission_intent'] = ack
    return Observation(time_s=time_s, pose=(xy[0], xy[1], 1.5, 90.), writer=WRITER, goal=goal, execution=execution,
                       frontiers=tuple(frontiers), area_coverage=0.2, estimates=tuple(estimates))


def mission(**params):
    search = SearchMission()
    search.start(dict(params), obs(), area=AREA, strategy=build('frontier'))
    return search


def to_standoff(search):
    decision = search.step(obs((0., 0.), time_s=1., estimates=[estimate(last_s=1.)]))
    assert isinstance(decision, Goal) and decision.xy == pytest.approx((16., 0.)), 'standoff 4 m short, on the line'
    return decision


def test_a_confirmed_target_is_looked_at_from_the_standoff_approached_and_landed_on():
    search = mission()
    assert isinstance(search.step(obs(time_s=0.)), Goal), 'no estimate yet: it explores'
    to_standoff(search)
    turn = search.step(obs(time_s=10., estimates=[estimate(last_s=9.)], arrived=(16., 0.),
                           ack=dict(kind='launch', revision=1, state='completed')))
    assert isinstance(turn, Action) and turn.kind == 'yaw_to' and turn.args['heading_deg'] == pytest.approx(90.)
    assert isinstance(search.step(obs((16., 0.), time_s=10.2, estimates=[estimate(last_s=9.)],
                                      ack=dict(kind='yaw_to', revision=2, state='accepted'))), Wait)
    looking = search.step(obs((16., 0.), time_s=11., estimates=[estimate(last_s=9.)],
                              ack=dict(kind='yaw_to', revision=2, state='completed')))
    assert isinstance(looking, Wait) and search.phase == 'observe'
    assert isinstance(search.step(obs((16., 0.), time_s=12., estimates=[estimate(last_s=12., revision=6)])), Wait)
    approach = search.step(obs((16., 0.), time_s=14.5, estimates=[estimate((20.2, 0.), last_s=14., revision=8)]))
    assert isinstance(approach, Goal) and approach.xy == pytest.approx((20.2, 0.))
    done = search.step(obs(time_s=30., estimates=[estimate((20.2, 0.), last_s=14., revision=8)], arrived=(20.2, 0.)))
    assert isinstance(done, Done) and done.outcome == 'complete' and done.finish == 'land'
    assert done.reason.startswith('landed on estimate landing_pad-1 (spread 0.20 m, 5 observations')
    log = search.summary()['estimate_log']
    assert [row['decision'] for row in log] == ['go to standoff', 'face estimate', 'observe', 'approach estimate',
                                                'land on estimate']
    assert [row['revision'] for row in log][-2:] == [8, 8] and all(row['estimate'] == 'landing_pad-1' for row in log)
    assert search.summary()['landed_on']['revision'] == 8


def test_a_target_never_confirmed_ends_failed_when_exploring_ends():
    search = mission()
    unconfirmed = [estimate(confirmed=False, count=1)]
    assert isinstance(search.step(obs(time_s=1., estimates=unconfirmed)), Goal) and search.phase == 'explore'
    low = [estimate(confidence=0.1)]
    assert isinstance(search.step(obs(time_s=2., estimates=low)), Goal) and search.phase == 'explore', \
        'below min_confidence is not a target'
    finished = mission()  # nothing left to explore and nothing confirmed
    done = finished.step(obs(time_s=3., estimates=unconfirmed, frontiers=()))
    assert isinstance(done, Done) and done.outcome == 'failed'
    assert done.reason == 'no confirmed target: no frontier left to explore'


def test_a_stale_estimate_is_given_up_and_exploring_resumes():
    search = mission()
    to_standoff(search)
    decision = search.step(obs((5., 0.), time_s=5., estimates=[estimate(stale=True)]))
    assert isinstance(decision, Goal) and search.phase == 'explore'
    assert search.given_up == {'landing_pad-1': 'estimate went stale'}
    assert search.step(obs(time_s=6., estimates=[estimate(last_s=6.)])).xy != pytest.approx((16., 0.)), \
        'a given-up estimate is not chased again'


def test_a_target_not_seen_again_is_looked_at_again_then_given_up():
    search = mission(max_standoff_attempts=2, observe_s=2.)
    to_standoff(search)
    search.step(obs(time_s=10., estimates=[estimate(last_s=1.)], arrived=(16., 0.)))
    search.step(obs((16., 0.), time_s=11., estimates=[estimate(last_s=1.)],
                    ack=dict(kind='yaw_to', revision=3, state='completed')))
    again = search.step(obs((16., 0.), time_s=13.5, estimates=[estimate(last_s=1.)]))
    assert isinstance(again, Action) and again.kind == 'yaw_to' and search.attempts == {'landing_pad-1': 1}
    search.step(obs((16., 0.), time_s=14., estimates=[estimate(last_s=1.)],
                    ack=dict(kind='yaw_to', revision=4, state='failed', reason='not airborne')))
    assert search.phase == 'observe' and 'turn failed' in search.log[-1]['detail']
    gave_up = search.step(obs((16., 0.), time_s=16.5, estimates=[estimate(last_s=1.)]))
    assert search.phase == 'explore' and isinstance(gave_up, Goal)
    assert search.given_up['landing_pad-1'] == 'not seen again from the standoff after 2 look(s)'


def refused_at(search, time_s, *, status='no_route', pose=(2., 0.), route_mode='planned',
               eligible=False, execution=None, count=5):
    """dnav refusing the standoff the mission is currently asking for."""
    standoff = search.standoff_xy
    return Observation(time_s=time_s, pose=(pose[0], pose[1], 1.5, 90.), writer=WRITER,
                       goal={'authority': WRITER, 'position': list(standoff), 'revision': 7},
                       execution=execution or {},
                       navigation={'planning_status': status, 'goal_revision': 7,
                                   'route_mode': route_mode, 'eligible': eligible},
                       estimates=(estimate(last_s=time_s, count=count, revision=count),),
                       frontiers=((5., 5.),))


def test_a_standoff_is_kept_while_a_transient_refusal_lasts():
    search = mission(unreachable_s=5.)
    to_standoff(search)
    for time_s in (7., 8.):  # entered at 1 s: a first refusal six seconds in is not yet persistent
        assert isinstance(search.step(refused_at(search, time_s)), Goal) and search.phase == 'standoff'
    assert isinstance(search.step(refused_at(search, 9., status='ok')), Goal), 'a plan clears the refusal'
    assert search.standoff_xy == pytest.approx((16., 0.)), 'and the standoff is where it was'


def test_a_refusal_is_not_counted_while_a_route_toward_the_standoff_is_still_permitted():
    """The 17 Sep 14:27 area1 run: dnav lost the plan to the standoff for two
    seconds, published the last good route as `retained` and kept the vehicle
    flying, and the mission discarded a pad it went on to see 96 more times."""
    search = mission(unreachable_s=5.)
    to_standoff(search)
    # Creeping, as a vehicle accelerating out of a stop does: 0.2 m/s, so it
    # never closes PROGRESS_M inside the window a goal verdict would allow.
    for step in range(60):                    # six seconds of refusal, under permission
        held = refused_at(search, 2. + step * 0.1, status='goal_unreachable',
                          pose=(2. + step * 0.02, 0.), route_mode='retained', eligible=True,
                          execution={'state': 'EXECUTING'})
        assert isinstance(search.step(held), Goal) and search.phase == 'standoff', step
        assert not search.given_up, 'the estimate was discarded while the route was still permitted'
    assert isinstance(search.step(refused_at(search, 8.1, status='ok', pose=(3.2, 0.))), Goal)
    assert search.phase == 'standoff' and not search.given_up


def test_an_unreachable_standoff_moves_round_the_estimate_before_the_target_is_given_up():
    """A standoff is computed geometry: one that lands in a margin says nothing
    about the target, so the answer is another bearing, not another target."""
    search = mission(unreachable_s=5., max_standoff_attempts=2, standoff_m=4.)
    to_standoff(search)
    first = search.standoff_xy
    search.step(refused_at(search, 7.))
    moved = search.step(refused_at(search, 12.))     # unreachable_s after the first refusal
    assert isinstance(moved, Goal) and search.phase == 'standoff', 'the estimate was given up too early'
    assert search.standoff_xy != pytest.approx(first), 'the same blocked point was tried again'
    assert math.dist(search.standoff_xy, PAD) == pytest.approx(4., abs=1e-6), 'still 4 m from the pad'
    assert search.attempts['landing_pad-1'] == 1
    # The second failure spends the attempts, and now the estimate is given up.
    search.step(refused_at(search, 13.))
    search.step(refused_at(search, 18.5))
    assert search.phase == 'explore'
    assert 'after 2 standoff(s)' in search.given_up['landing_pad-1']


def test_an_estimate_given_up_for_geometry_is_reconsidered_once_it_is_seen_again():
    """What the 14:27 run needed after the fact: the pad was observed 96 times
    while the mission ignored it, because a give-up was for ever."""
    from dmsn.missions.search import RETRY_OBSERVATIONS

    search = mission(unreachable_s=5., max_standoff_attempts=1)
    to_standoff(search)
    search.step(refused_at(search, 7.))
    search.step(refused_at(search, 12.))
    assert search.phase == 'explore' and 'landing_pad-1' in search.given_up
    seen = search.retry_at['landing_pad-1']
    # Seen a few more times: not enough, and the mission keeps exploring.
    decision = search.step(obs((3., 0.), time_s=13., estimates=[estimate(last_s=13., count=seen - 1)]))
    assert isinstance(decision, Goal) and search.phase == 'explore'
    assert 'landing_pad-1' in search.given_up
    # Re-confirmed: the estimate is admitted again and the standoff retried.
    decision = search.step(obs((3., 0.), time_s=14., estimates=[estimate(last_s=14., count=seen)]))
    assert search.phase == 'standoff' and not search.given_up
    assert isinstance(decision, Goal) and search.attempts.get('landing_pad-1') is None
    assert RETRY_OBSERVATIONS > 1, 'one sighting must not be enough to undo a give-up'


def test_a_spent_budget_is_no_confirmed_target():
    search = mission()
    assert search.budget_spent('time budget spent: 60.0 of 60') == ('failed',
                                                                     'no confirmed target: time budget spent: 60.0 of 60')
    to_standoff(search)
    outcome, reason = search.budget_spent('time budget spent')
    assert outcome == 'failed' and 'no confirmed target landed on' in reason and 'landing_pad-1' in reason


def test_the_overlay_marks_estimates_the_standoff_and_the_progress_is_bounded():
    search = mission()
    to_standoff(search)
    texts = [s.text for s in search.overlay() if s.kind == 'text']
    assert 'landing_pad-1' in texts and 'standoff' in texts
    progress = search.progress()
    assert progress['search_phase'] == 'standoff' and progress['target'] == 'landing_pad-1'
    assert progress['target_revision'] == 5


# -- on the runner ------------------------------------------------------------------------------

def plane(estimates=(), *, labels=('landing_pad',), detector='front-red_disc', time_s=0.):
    value = D.empty('fake', 'dalg', 'dalg-session', time_s)
    value['detectors'] = [dict(id=detector, name='red_disc', sensor='front', labels=list(labels), settings_digest='x',
                               state='running', processed=10, skipped=0, failed=0)]
    value['estimates'] = [dict(e) for e in estimates]
    return D.validate(value)


def search_profile(*overrides):
    return P.apply_overrides(P.blank('search'), ['area=0,-10;30,-10;30,10;0,10', 'coverage_margin_m=0',
                                                 'debounce.min_commit_s=0', 'observe_s=1', *overrides])


class Grid:
    """Evidence with a frontier: the strip around the start observed and free, the rest unknown."""
    def __init__(self):
        import numpy
        from dcmn.maps import NEVER_OBSERVED, GridGeometry
        self.geometry = GridGeometry(-5., -15., 1., 40, 30)
        self.source, self.revision = 'stub', 1
        self.occupancy = numpy.full((30, 40), NEVER_OBSERVED, numpy.uint8)
        self.occupancy[10:20, :12] = 0

    def layer(self, _index):
        import numpy
        return self.occupancy, numpy.zeros(self.occupancy.shape, numpy.uint32)


@pytest.fixture
def chain(tmp_path):
    made = []

    def build_chain(profile, **options):
        ports = FakePorts(tmp_path / f'run{len(made)}', pose=(0., 0., 1.5, 90.), coverage=(-5., -15., 35., 15.))
        ports._grids['dalg'] = Grid()
        ex = FakeExecutor(ports)
        run = MissionRun('fake', profile, ports=ports, **options)
        run.step()
        made.append(run)
        return run, ports, ex
    yield build_chain
    for run in made:
        try: run.close()
        except Exception: pass


def test_start_refuses_without_a_publisher_of_the_label(chain):
    run, ports, _ = chain(search_profile())
    with pytest.raises(StartRefused, match='no detector publishing `landing_pad`: no detections plane'):
        run.start()
    ports.detections = plane(labels=('person',))
    with pytest.raises(StartRefused, match=r'no detector publishing `landing_pad` \(the plane publishes person\)'):
        run.start()
    run.document.set('params.target_detector', 'down-red_disc')
    ports.detections = plane()
    with pytest.raises(StartRefused, match='as detector down-red_disc'):
        run.start()
    assert run.phase == 'idle'


def test_missions_without_a_target_never_open_the_plane(chain):
    for profile in (P.apply_overrides(P.blank('goto'), ['goal_source=xy', 'goal=4,0']),
                    P.apply_overrides(P.blank('explore'), ['area=0,-10;30,-10;30,10;0,10', 'coverage_margin_m=0'])):
        run, ports, ex = chain(profile)
        ports.detections = plane([estimate()])
        run.start()
        drive(run, ex, ports=ports, steps=30)
        assert ports.detections_reads == 0, profile['mission']
        assert run.observation.estimates == ()


def test_a_search_on_the_runner_lands_on_the_estimate_and_replays(chain):
    run, ports, ex = chain(search_profile())
    ports.detections = plane()
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'running', steps=40)
    ports.detections = plane([estimate(last_s=0.)])
    for _ in range(400):  # the pad keeps being seen: the estimate's last sighting follows sim time
        if run.phase == 'done': break
        ports.detections = plane([estimate(last_s=ports.time_s, revision=int(ports.time_s * 5) + 5)],
                                 time_s=ports.time_s)
        drive(run, ex, ports=ports, steps=1)
    assert (run.outcome, run.exit_code()) == ('complete', 0), (run.reason, run.blocking)
    assert run.reason.startswith('landed on estimate landing_pad-1')
    assert [i['kind'] for i in ports.intents] == ['launch', 'yaw_to', 'land'] and ex.landings == 1
    assert ex.turns == [pytest.approx(90.)]
    directory = run.close()
    summary = json.loads((directory / 'summary.json').read_text())
    log = summary['mission_summary']['estimate_log']
    assert {row['estimate'] for row in log} == {'landing_pad-1'} and all(row['revision'] for row in log)
    assert summary['mission_summary']['landed_on']['id'] == 'landing_pad-1'
    assert replay(directory / 'archive')['mismatches'] == []


def test_a_search_whose_budget_runs_out_before_a_target_fails_with_that_reason(chain):
    run, ports, ex = chain(search_profile('budgets.time_s=3', 'budgets.time_action=hold'))
    ports.detections = plane()
    ex.flight_steps = 10 ** 6
    run.start()
    assert drive(run, ex, ports=ports, until=lambda r: r.phase == 'done', steps=100)
    assert run.outcome == 'failed' and run.reason.startswith('no confirmed target: time budget spent')
    assert run.exit_code() == 1
    assert math.isfinite(run.observation.detections_time_s)
