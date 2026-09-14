"""The planning loop: what dnav does with a context and a plane, and what it records.

dnav reads a provider-neutral session context (data clock, frame and epochs,
labelled pose, goal and its authority, report root) and one coherent evidence
generation. Stale evidence and a departed producer are both `stale_map` with
different reasons; a stale pose, a frame/epoch mismatch and a goal outside
coverage each have their own status. Every attempt is archived with its exact
inputs, and an archived attempt replans to the identical route.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from dcmn.archive import ArchiveReader
from dcmn.context import Context
from dcmn.maps import GridGeometry, MapPublisher
from dcmn.module_bus import PymembusModuleBus
from dnav import route as R
from dnav.plan import NavRun, replay_attempt
from dnav.policy import load_policy
from dtest.synthetic import SOURCE_ID, SyntheticRoom

ROOT = Path(__file__).resolve().parents[1]


class Provider:
    """The session side of a rig: a real context registry this test owns."""

    def __init__(self, instance, report_root):
        self.context = Context(instance)
        self.clock_epoch = 7
        self.localization_epoch = 0
        self.time_s = 0.
        self.pose = (1.5, 1.5, 1.5, 90.)
        self.valid = True
        if report_root is not None: self.context.start(report_root, clock_epoch=self.clock_epoch)

    def publish(self, time_s=None, pose=None):
        if time_s is not None: self.time_s = float(time_s)
        if pose is not None: self.pose = pose
        x, y, z, heading = self.pose
        self.context.publish_pose(dict(x_m=x, y_m=y, z_m=z, heading_deg=heading, roll_deg=0., pitch_deg=0.),
                                  self.time_s, localization_epoch=self.localization_epoch,
                                  clock_epoch=self.clock_epoch, valid=self.valid)

    def evidence_context(self):
        return dict(frame_id='local', localization_epoch=self.localization_epoch,
                    clock_domain_id=self.context.instance, clock_epoch=self.clock_epoch)


@pytest.fixture
def rig(tmp_path):
    """A provider, an evidence publisher, and a NavRun wired to both."""
    made = []

    def build(*, goal=(15.5, 10.0, None), sweeps=12, policy='default',
              vehicle=(1.5, 1.5, 1.5, 90.), provider=True, announce=True, extent=20.):
        instance = f'nav-{uuid.uuid4().hex[:8]}'
        geometry = GridGeometry.from_extent(extent, extent, .5)
        owner = PymembusModuleBus(instance, 'simulator', 'test-sim', create=True)
        assert owner.connect()
        side = Provider(instance, tmp_path / f'session-{instance}' if provider else None)
        if provider and vehicle is not None:
            side.publish(0., vehicle)
        publisher = MapPublisher(instance, geometry,
                                 [dict(id=SOURCE_ID, sensor=SOURCE_ID, sensor_type='fixture',
                                       algorithm='synthetic')], context=side.evidence_context())
        room = SyntheticRoom(geometry)
        run = NavRun(instance, ROOT, policy=load_policy(policy, ROOT))
        for step in range(sweeps):
            room.sweep(1. + step)
            publisher.publish(SOURCE_ID, room.occupancy, room.observed_ms, 1. + step)
        if provider and vehicle is not None: side.publish(float(sweeps))
        if goal is not None: run.set_goal(*goal)
        if announce:
            producer = PymembusModuleBus(instance, 'algorithm', 'dalg-synthetic')
            producer.publish('module.heartbeat', payload={
                'state': 'PUBLISHING', 'capabilities': {'maps': [SOURCE_ID]}})
            producer.close()
        run.step()
        made.append((run, publisher, owner, side))
        return run, publisher, room, side

    yield build
    for run, publisher, owner, side in made:
        run.close(); publisher.close(); side.context.close(); owner.remove()


def advance(run, side, seconds):
    side.publish(side.time_s + seconds)
    run.connect()


# -- planning against a live plane -------------------------------------------

def test_a_goal_through_the_doorway_plans(rig):
    run, _, _, _ = rig()
    route = run.replan(force=True)
    assert route.status == R.OK, route.reason
    assert len(route.waypoints) >= 2
    assert route.waypoints[0].x_m == pytest.approx(1.5)
    assert route.waypoints[-1].x_m == pytest.approx(15.5)
    # The goal's z defaults to the vehicle's altitude: the slab it is already in.
    assert route.waypoints[-1].z_m == pytest.approx(1.5)


def test_a_route_through_unknown_cells_says_so_and_claims_no_clearance(rig):
    run, _, _, _ = rig(sweeps=2)
    route = run.replan(force=True)
    assert route.status == R.OK, route.reason
    assert route.diagnostics['unknown_corridor'] is True
    assert route.diagnostics['unknown_cells'] > 0


def test_a_cautious_policy_closes_the_doorway_and_says_so(rig):
    run, _, _, _ = rig(policy='cautious')
    route = run.replan(force=True)
    assert route.status == R.NO_ROUTE
    assert 'within current evidence' in route.reason


def test_a_reloaded_policy_replans_against_the_file_on_disk(rig, tmp_path):
    run, _, _, _ = rig()
    assert run.replan(force=True).status == R.OK
    swap = tmp_path / 'wide.json'
    swap.write_text(json.dumps({'schema': 'dvision2.cost-policy.v1',
                                'name': 'wide', 'inflation_m': 1.1}), encoding='utf-8')
    run.policy = load_policy(str(swap), ROOT)
    reloaded = run.reload_policy()
    assert reloaded.inflation_m == 1.1
    assert run.replan(force=True).status == R.NO_ROUTE


def test_no_goal_is_an_invitation_rather_than_a_failure(rig):
    run, _, _, _ = rig(goal=None)
    route = run.replan(force=True)
    assert route.status == R.NO_GOAL and route.reason
    assert not route.waypoints


def test_a_goal_outside_coverage_is_named_and_never_called_impossible(rig):
    run, _, _, _ = rig(goal=(35., 10., None))
    route = run.replan(force=True)
    assert route.status == R.OUTSIDE_COVERAGE
    assert 'outside the evidence coverage' in route.reason
    assert 'does not mean no route exists' in route.reason


# -- pose ----------------------------------------------------------------------

def test_no_pose_at_all_is_not_planned_from(rig):
    run, _, _, _ = rig(vehicle=None)
    route = run.replan(force=True)
    assert route.status == R.STALE_POSE
    assert 'pose' in route.reason


def test_a_stale_pose_suspends_planning_in_the_data_clock(rig):
    run, _, _, side = rig()
    assert run.replan(force=True).status == R.OK
    # Data time moves on without a new pose: the provider stopped estimating.
    pose = dict(run.snapshot['pose'])
    side.publish(side.time_s + 2.)
    snapshot = side.context.read()
    snapshot['pose'] = pose
    run.snapshot = snapshot
    route = run.replan(force=True)
    assert route.status == R.STALE_POSE and 'stale' in route.reason


def test_an_invalid_pose_suspends_planning(rig):
    run, _, _, side = rig()
    side.valid = False; side.publish()
    run.connect()
    route = run.replan(force=True)
    assert route.status == R.STALE_POSE and 'invalid' in route.reason


# -- goal authority ---------------------------------------------------------------

def test_a_goal_set_before_the_provider_exists_is_submitted_when_it_appears(tmp_path):
    instance = f'nav-{uuid.uuid4().hex[:8]}'
    run = NavRun(instance, ROOT, policy=load_policy('default', ROOT), goal=(5., 5., None))
    side = None
    try:
        run.step()
        assert run.goal_xy is None and 'queued' in run.goal_error
        side = Provider(instance, tmp_path / 'late'); side.publish(1.)
        run.step()
        assert run.goal_xy == (5., 5.)
        assert run.authority['id'] == run.writer_id and run.authority['role'] == 'ui'
    finally:
        run.close()
        if side: side.context.close()


def test_another_authority_is_never_silently_overwritten(rig):
    run, _, _, side = rig()
    mission = Context(run.id)
    mission.set_goal('mission-1', (10., 10.), role='mission', handoff=True)
    run.connect()
    assert run.goal_xy == (10., 10.)
    assert run.set_goal(12., 12.) is False
    assert 'explicit handoff required' in run.goal_error
    run.connect()
    assert run.goal_xy == (10., 10.)
    assert run.set_goal(12., 12., handoff=True) is True
    run.connect()
    assert run.goal_xy == (12., 12.) and run.snapshot['authority_epoch'] == 3


def test_clearing_the_goal_is_the_authoritys_to_do(rig):
    run, _, _, _ = rig()
    assert run.clear_goal() is True
    run.connect()
    assert run.goal_xy is None
    assert run.replan(force=True).status == R.NO_GOAL


# -- epochs and generations --------------------------------------------------------------

def test_a_localization_reset_withdraws_the_goal_and_the_old_evidence(rig):
    run, publisher, room, side = rig()
    assert run.replan(force=True).status == R.OK
    side.localization_epoch = 1
    side.publish(side.time_s + .1)
    run.connect()
    route = run.replan(force=True)
    assert route.status == R.NO_GOAL and 'goal withdrawn' in route.reason
    run.set_goal(15.5, 10.)
    route = run.replan(force=True)
    assert route.status == R.FRAME_MISMATCH and 'localization_epoch' in route.reason


def test_a_new_evidence_generation_invalidates_the_route_and_is_never_mixed(rig):
    run, publisher, room, side = rig()
    assert run.replan(force=True).status == R.OK
    replacement = MapPublisher(run.id, publisher.geometry,
        [dict(id=SOURCE_ID, sensor=SOURCE_ID, sensor_type='fixture', algorithm='synthetic')],
        context=dict(side.evidence_context(), mapping_epoch=2, geometry_revision=2), generation=2)
    try:
        run.session._last_probe = -1e9
        run.step()
        assert run.transitions == 1
        assert run.route.status == R.STALE_MAP and 'generation changed' in run.route.reason
        assert run.session.latest(SOURCE_ID) is None
        replacement.publish(SOURCE_ID, room.occupancy, room.observed_ms, side.time_s)
        run.session.poll()
        assert run.replan(force=True).status == R.OK
    finally:
        replacement.close()


# -- the two facts behind stale_map ------------------------------------------

def test_old_evidence_is_stale_and_the_reason_names_the_clock(rig):
    run, _, _, side = rig()
    assert run.replan(force=True).status == R.OK
    advance(run, side, 30.)
    route = run.replan(force=True)
    assert route.status == R.STALE_MAP
    assert 'data-clock seconds old' in route.reason and 'horizon' in route.reason


def test_a_departed_producer_is_stale_and_the_reason_names_the_bus(rig):
    run, publisher, _, side = rig(announce=False)
    assert run.replan(force=True).status == R.OK
    publisher.close()
    advance(run, side, 30.)
    run.session._backoff_s = 0.
    run.session.poll()
    route = run.replan(force=True)
    assert route.status == R.STALE_MAP
    assert 'heartbeat' in route.reason, route.reason


def test_no_producer_at_all_is_named_as_such(tmp_path):
    instance = f'nav-{uuid.uuid4().hex[:8]}'
    side = Provider(instance, tmp_path / 'np'); side.publish(4., (1., 1., 1.5, 0.))
    run = NavRun(instance, ROOT, policy=load_policy('default', ROOT), goal=(5., 5., None))
    try:
        run.step()
        route = run.replan(force=True)
        assert route.status == R.STALE_MAP
        assert 'no evidence producer' in route.reason
    finally:
        run.close(); side.context.close()


def test_a_running_producer_without_grids_is_named_in_the_reason(tmp_path):
    instance = f'nav-{uuid.uuid4().hex[:8]}'
    owner = PymembusModuleBus(instance, 'simulator', 'test-sim', create=True)
    assert owner.connect()
    side = Provider(instance, tmp_path / 'np'); side.publish(4., (1., 1., 1.5, 0.))
    run = NavRun(instance, ROOT, policy=load_policy('default', ROOT), goal=(5., 5., None))
    waiting = PymembusModuleBus(instance, 'algorithm', 'dalg')
    try:
        waiting.publish('module.heartbeat', payload={
            'state': 'WAITING_SENSORS', 'profile': 'lidar-baseline',
            'capabilities': {'maps': False, 'algorithms': ['lidar_inverse']}})
        run.step()
        route = run.replan(force=True)
        assert route.status == R.STALE_MAP
        assert 'lidar-baseline' in route.reason
        waiting.publish('module.goodbye', payload={'state': 'STOPPED'})
        run.step()
        assert 'lidar-baseline' not in run.replan(force=True).reason
    finally:
        waiting.close(); run.close(); side.context.close(); owner.remove()


def test_a_producer_that_publishes_no_grids_is_not_a_live_map(rig):
    run, _, _, _ = rig(announce=False)
    bus = PymembusModuleBus(run.id, 'algorithm', 'dalg-waiting')
    try:
        bus.connect()
        bus.publish('module.heartbeat', payload={
            'state': 'WAITING_POSE', 'capabilities': {'algorithms': ['sgbm'], 'maps': False}})
        run.step()
        assert not run.producer_alive()
    finally:
        bus.close()


# -- re-planning -------------------------------------------------------------

def test_a_plan_is_not_recomputed_while_nothing_has_changed(rig):
    run, _, _, _ = rig()
    run.replan(force=True)
    before = run.plans
    for _ in range(5): run.replan()
    assert run.plans == before


def test_new_evidence_replans(rig):
    run, publisher, room, side = rig()
    run.replan(force=True)
    before = run.plans
    room.sweep(20.)
    side.publish(20.)
    publisher.publish(SOURCE_ID, room.occupancy, room.observed_ms, 20.)
    run.connect(); run.session.poll()
    run.replan()
    assert run.plans == before + 1


def test_moving_the_goal_replans(rig):
    run, _, _, _ = rig()
    first = run.replan(force=True)
    run.set_goal(17., 10.)
    second = run.replan()
    assert second.waypoints[-1].x_m == pytest.approx(17.)
    assert first.waypoints[-1].x_m == pytest.approx(15.5)


def test_the_straight_line_diagnostic_is_planned_alongside_every_real_route(rig):
    run, _, _, _ = rig()
    run.replan(force=True)
    assert run.control_route is not None and run.control_route.planner == 'control'
    detour = run.summary_detour()
    assert detour is not None
    assert detour['control_blocked'] or detour['cost_ratio'] <= 1.0 + 1e-9


# -- presence ----------------------------------------------------------------

def test_dnav_registers_as_a_planner_and_claims_no_lease(rig):
    run, _, _, _ = rig()
    reader = PymembusModuleBus(run.id, 'observer', 'test-reader', read_only=True)
    try:
        assert reader.connect()
        run._hello_sent = False
        run.step()
        events = [e for e in reader.receive() if e.implementation == 'dnav']
        assert events, 'dnav announced nothing'
        assert {e.role for e in events} == {'planner'}
        hello = next(e for e in events if e.type == 'module.hello')
        assert hello.payload['capabilities']['holds_control_lease'] is False
        assert 'astar' in hello.payload['capabilities']['planners']
    finally:
        reader.close()


def test_a_route_is_published_with_its_goal_revision_and_evidence_context(rig):
    run, _, _, _ = rig()
    reader = PymembusModuleBus(run.id, 'observer', 'test-reader', read_only=True)
    try:
        assert reader.connect()
        run.replan(force=True)
        published = [e for e in reader.receive() if e.type == 'route.planned']
        assert published
        payload = published[-1].payload
        assert payload['status'] == R.OK
        assert payload['waypoints'][0].keys() >= {'x', 'y', 'z'}
        assert payload['policy_digest'] == run.policy.digest
        assert payload['goal_revision'] == run.goal_descriptor['revision']
        assert payload['evidence_context']['mapping_epoch'] == 1
    finally:
        reader.close()


# -- the report and the archive ---------------------------------------------------

def test_a_finished_run_reports_its_numbers_and_its_picture(rig):
    run, _, _, _ = rig()
    run.replan(force=True)
    directory = run.close()
    assert directory is not None and directory.name == 'dnav'
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    assert summary['route']['status'] == R.OK
    assert summary['partial'] is False
    assert summary['policy']['digest'] == run.policy.digest
    assert summary['goal']['position'] == [15.5, 10.0]
    assert summary['map']['sources'][SOURCE_ID]['revision'] > 0
    assert summary['detour'] is not None and 'not an oracle' in summary['control_note']
    assert summary['recording']['complete'] is True
    assert (directory / 'events.jsonl').exists()
    assert (directory / 'route.png').exists()


def test_every_attempt_replans_exactly_from_the_archive_alone(rig):
    run, publisher, room, side = rig()
    run.replan(force=True)
    run.set_goal(17., 10.); run.replan()
    run.clear_goal(); run.replan()
    run.set_goal(35., 10.); run.replan()
    recorded = run.route
    directory = run.close()
    reader = ArchiveReader(directory / 'archive')
    report = reader.validate()
    assert report['complete'], report
    attempts = reader.attempts()
    statuses = [a['data']['route']['status'] for a in attempts]
    assert statuses[:2] == [R.OK, R.OK] and R.NO_GOAL in statuses and statuses[-1] == R.OUTSIDE_COVERAGE
    assert recorded.status == R.OUTSIDE_COVERAGE
    for attempt in attempts[:2]:
        rebuilt = reader.reconstruct_attempt(attempt['sequence'])
        assert rebuilt['grids'][SOURCE_ID].revision == attempt['data']['inputs']['revisions'][SOURCE_ID]
        assert rebuilt['pose']['localization_epoch'] == 0
        replayed = replay_attempt(rebuilt)
        assert replayed.as_dict()['waypoints'] == attempt['data']['route']['waypoints']
        assert replayed.as_dict()['cost'] == attempt['data']['route']['cost']


def test_an_aborted_run_still_reports_and_says_it_was_partial(rig):
    run, _, _, _ = rig()
    run.replan(force=True)
    run.reason = 'interrupted'
    directory = run.close(partial=True)
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    assert summary['partial'] is True and summary['reason'] == 'interrupted'


def test_the_report_records_every_status_the_run_passed_through(rig):
    run, _, _, side = rig()
    run.replan(force=True)
    advance(run, side, 30.)
    run.replan(force=True)
    directory = run.close()
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    assert [entry['status'] for entry in summary['statuses']] == [R.OK, R.STALE_MAP]
    lines = [json.loads(line) for line in
             (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any(entry['type'] == 'route.status' for entry in lines)


def test_session_rollover_opens_a_new_archive_with_the_retained_state(rig, tmp_path):
    run, _, _, side = rig()
    run.replan(force=True)
    first = run.recorder.directory
    side.context.rollover(tmp_path / 'second-session')
    run.step()
    assert run.recorder.directory != first
    assert run.recorder.directory.parent == tmp_path / 'second-session' / 'dnav'
    assert ArchiveReader(first).validate()['complete']
    run.replan(force=True)
    directory = run.close()
    reader = ArchiveReader(directory / 'archive')
    events = reader.events()
    assert events[0]['type'] == 'retained_state'
    assert SOURCE_ID in reader.reconstruct(events[0])['reconstructed_grids']


def test_reporting_never_raises_into_the_run(rig, monkeypatch):
    run, _, _, _ = rig()
    run.replan(force=True)
    def explode(*args, **kwargs): raise OSError('the disk is full')
    monkeypatch.setattr('dnav.report.write_report', explode)
    run.close()          # must not raise
    assert run.closed


def test_a_run_with_no_session_provider_writes_nothing(rig):
    run, _, _, _ = rig(provider=False)
    run.replan(force=True)
    assert run.close() is None
