"""Route transport, clearance geometry and read-only integration contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from dcmn.context import Context
from dcmn.maps import EvidenceGrid, GridGeometry, MapPublisher
from dcmn.module_bus import PymembusModuleBus
from dcmn.navigation import (MAX_BYTES, SCHEMA, STATUS_SCHEMA, Snapshot, distance_along, permitted_points,
                             project_progress, validate)
from dnav.clearance import Clearance, ExecutionProfile
from dnav.execution import RoutePublisher
from dnav.plan import NavRun
from dnav.policy import load_policy
from dway.dynamic import DynamicRouteSource, DryRun
from dway.frames import ProviderFrame

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT/'assets/execution_profiles/dry-run.json'
EXAMPLES = ROOT/'tests/assets/navigation_examples'


def grid(occupancy=None, observed=None, time_s=10.):
    geom = GridGeometry.from_extent(10, 10, .5)
    occ = np.zeros(geom.shape, np.uint8) if occupancy is None else occupancy
    obs = np.full(geom.shape, int(time_s*1000), np.uint32) if observed is None else observed
    return EvidenceGrid(geom, occ, obs, 'scan', 1, time_s)


def context(goal=(7., 2., 1.5)):
    from dcmn.context import FRAME, pose_context
    return dict(provider_id='p', frame_id='local', localization_epoch=0, clock_domain_id='v',
        clock_epoch=0, time_s=10., frame=FRAME, vehicle_transform=dict(
            schema='dvision2.local-ned-transform.v1', frame_id='local', localization_epoch=0,
            origin=[5., 5., 0.]), goal=dict(authority='ui', authority_epoch=1, revision=1,
            frame_id='local', localization_epoch=0, clock_epoch=0, position=list(goal)),
        pose=dict(x_m=2., y_m=2., z_m=1.5, heading_deg=90., roll_deg=0., pitch_deg=0.,
                  **pose_context('v', 0, 0, 10.)))


def message():
    from dcmn.navigation import context_identity
    p = ExecutionProfile.load(PROFILE)
    points = [[2., 2., 1.5], [7., 2., 1.5]]
    return dict(schema=SCHEMA, vehicle_id='v', planner='dnav', session='n', sequence=1,
        time_s=10., stop_generation=0, geometry_revision=1, points=points, context=context_identity(context()),
        goal=context()['goal'], planning_status='ok', profile=p.digest,
        clearance=Clearance(p).check(points, points[0], {'scan': grid()}, 10.))


def example(name):
    return json.loads((EXAMPLES/f'{name}.json').read_text())


def source():
    return DynamicRouteSource('v', 'dnav', ExecutionProfile.load(PROFILE))


# -- schema and transport -----------------------------------------------------

@pytest.mark.parametrize('edit', [
    lambda v: v.update(schema='other'), lambda v: v.update(points=[[0, 0, float('nan')]]),
    lambda v: v.update(points=[[0, 0, 0]]*257), lambda v: v.update(sequence=-1),
    lambda v: v['clearance'].update(end=[9, 0.]), lambda v: v['clearance'].update(end=[0, 1.1]),
    lambda v: v['clearance'].update(valid_until_s=9.), lambda v: v.update(extra='x'*MAX_BYTES),
    lambda v: v.update(points=[]), lambda v: v.update(sequence=True),
    lambda v: v['clearance'].update(start=[0, .5], end=[0, .5]),
    lambda v: v['goal'].pop('position'), lambda v: v['context'].pop('clock_epoch'),
    lambda v: v.update(points=[[2., 2., 1.5]]) or v['clearance'].update(end=[0, .5])])
def test_malformed_route_is_rejected(edit):
    value = message(); edit(value)
    with pytest.raises(ValueError): validate(value)


def test_status_cannot_claim_control_while_dry_run():
    status = example('execution-stopped')
    validate(status, STATUS_SCHEMA)
    for edit in (dict(dry_run=True), dict(state='EXECUTING', dry_run=True), dict(commanded_target=[1, 2])):
        with pytest.raises(ValueError): validate(dict(status, **edit), STATUS_SCHEMA)
    with pytest.raises(ValueError):
        validate(dict(status, disposition=dict(value='flying')), STATUS_SCHEMA)


def test_profile_loader_is_strict(tmp_path):
    value = json.loads(PROFILE.read_text())
    for edit in (dict(calibrated=True), dict(unexpected=1), dict(max_age_s=1.), dict(speed_mps=float('inf'))):
        path = tmp_path/'p.json'; path.write_text(json.dumps(dict(value, **edit)))
        with pytest.raises((ValueError, TypeError)): ExecutionProfile.load(path)


def test_retained_transport_late_reader_conflict_restart():
    instance = 'wire-'+uuid.uuid4().hex[:8]
    writer = Snapshot(instance); competitor = Snapshot(instance); reader = Snapshot(instance)
    value = message(); value['vehicle_id'] = instance
    try:
        assert reader.read() == {}
        writer.write(value)
        assert reader.read() == value
        with pytest.raises(ValueError, match='live writer'): competitor.start()
        assert reader.read() == value
        writer.close()
        competitor.write(dict(value, session='new'))
        assert reader.read()['session'] == 'new'
    finally: writer.close(); competitor.close(); reader.close()


def test_transport_maximum_sized_value_roundtrip_and_oversize_refused():
    instance = 'max-'+uuid.uuid4().hex[:8]
    writer = Snapshot(instance)
    value = message(); value['vehicle_id'] = instance
    from dcmn.navigation import encode
    value['padding'] = ''
    value['padding'] = 'x'*(MAX_BYTES-len(encode(value).encode()))
    try:
        writer.write(value)
        assert Snapshot(instance).read() == value
        with pytest.raises(ValueError, match='64 KiB'): writer.write(dict(value, padding=value['padding']+'x'))
        assert Snapshot(instance).read() == value  # a refused write never truncates the retained record
    finally: writer.close()


def test_fixtures_are_complete_and_behave_as_named():
    digest = ExecutionProfile.load(PROFILE).digest
    names = {p.stem for p in EXAMPLES.glob('*.json')}
    assert names >= {'ready', 'unavailable', 'expired-invalid', 'stop-required', 'arrival-only', 'execution-stopped'}
    for name in names:
        assert example(name)['profile'] == digest, name
    assert source().consume(example('ready'), context(), wall=0) == 'READY'
    src = source()
    assert src.consume(example('unavailable'), context(), wall=0) == 'REJECTED'
    assert src.reason == 'evidence unavailable'
    with pytest.raises(ValueError, match='expired'): validate(example('expired-invalid'))
    assert source().consume(example('expired-invalid'), context(), wall=0) == 'REJECTED'
    arrival = source()
    assert arrival.consume(example('arrival-only'), context(goal=(2., 2., 1.5)), wall=0) == 'READY'
    assert permitted_points(example('arrival-only')) == [[2., 2., 1.5]]
    # The stop-required fixture latches only an active route; an idle executor adopts it.
    assert source().consume(example('stop-required'), context(), wall=0) == 'READY'
    active = source()
    assert active.consume(example('ready'), context(), wall=0) == 'READY'
    active.activate()
    assert active.consume(example('stop-required'), context(), wall=0) == 'STOP_REQUIRED'
    validate(example('execution-stopped'), STATUS_SCHEMA)


# -- clearance ---------------------------------------------------------------

def test_clearance_stops_inside_segment_and_never_crosses_unknown():
    g = grid(); g.occupancy[:, :, 10:] = 255; g.observed_ms[:, :, 10:] = 0
    p = ExecutionProfile.load(PROFILE)
    value = message(); value['clearance'] = Clearance(p).check(value['points'], value['points'][0], {'scan': g}, 10.)
    c = value['clearance']
    assert c['eligible'] and not c['reaches_goal']
    end = permitted_points(value)[-1]
    assert 2 < end[0] < 5-p.body_radius_m-p.tracking_m-p.stopping_m
    assert end != value['points'][-1]


def test_clearance_republish_does_not_refresh_cells_and_conflict_veto_persists():
    p = ExecutionProfile.load(PROFILE); c = Clearance(p); pts = message()['points']
    g = grid(); bad = grid(); bad.occupancy[0, 4, 5] = 254
    assert not c.check(pts, pts[0], {'free': g, 'bad': bad}, 10.)['reaches_goal']
    bad.occupancy[0, 4, 5] = 120  # ambiguous does not clear a prior obstacle
    assert not c.check(pts, pts[0], {'free': g, 'bad': bad}, 10.)['reaches_goal']
    bad.occupancy[0, 4, 5] = 0
    assert c.check(pts, pts[0], {'free': g, 'bad': bad}, 10.)['reaches_goal']
    g = replace(g, sim_time_s=20.)
    assert not c.check(pts, pts[0], {'free': g}, 20.)['eligible']


def test_veto_survives_its_source_disappearing():
    p = ExecutionProfile.load(PROFILE); c = Clearance(p); pts = message()['points']
    bad = grid(); bad.occupancy[0, 4, 5] = 254
    assert not c.check(pts, pts[0], {'free': grid(), 'bad': bad}, 10.)['reaches_goal']
    assert not c.check(pts, pts[0], {'free': grid()}, 10.)['reaches_goal']
    assert Clearance(p).check(pts, pts[0], {'free': grid()}, 10.)['reaches_goal']  # explicit reset


def test_clearance_requires_slab_and_full_footprint_and_fresh_sources():
    pts = message()['points']; g = grid()
    assert not Clearance(ExecutionProfile()).check(pts, pts[0], {'s': g}, 10.)['eligible']
    p = ExecutionProfile.load(PROFILE)
    edge = [[.1, 2., 1.5], [7, 2, 1.5]]
    assert not Clearance(p).check(edge, edge[0], {'s': g}, 10.)['eligible']
    above = [[2., 2., 3.], [7., 2., 3.]]
    assert not Clearance(p).check(above, above[0], {'s': g}, 10.)['eligible']
    required = ExecutionProfile(slab_assumption='vertical-extrusion', required_sources=('missing',))
    assert not Clearance(required).check(pts, pts[0], {'s': g}, 10.)['eligible']
    # Observations older than max_age minus the stop time cannot admit a route.
    assert not Clearance(p).check(pts, pts[0], {'s': grid(time_s=7.5)}, 10.)['eligible']


def test_clearance_arrival_only_and_corner_cover():
    p = ExecutionProfile.load(PROFILE); g = grid()
    assert Clearance(p).check([[2., 2., 1.5]], [2., 2., 1.5], {'s': g}, 10.)['reaches_goal']
    # A diagonal centerline misses this adjacent corner cell; swept area does not.
    g.occupancy[0, 5, 4] = 254
    c = Clearance(p).check([[2., 2., 1.5], [4., 4., 1.5]], [2., 2., 1.5], {'s': g}, 10.)
    assert not c['reaches_goal']


def test_remaining_interval_does_not_require_return_to_old_start():
    profile = ExecutionProfile.load(PROFILE)
    points = [[2., 2., 1.5], [7., 2., 1.5], [7., 7., 1.5]]
    c = Clearance(profile).check(points, [7., 3., 1.5], {'s': grid()}, 10., start=(1, .2))
    assert c['eligible'] and c['start'] == [1, .2] and c['end'] == [1, 1.]
    value = message(); value.update(points=points, clearance=c)
    assert permitted_points(value)[0] == [7., 3., 1.5]
    assert permitted_points(value)[-1] == [7., 7., 1.5]
    # Validation from the old start fails when the vehicle is no longer there.
    assert not Clearance(profile).check(points, [7., 3., 1.5], {'s': grid()}, 10.)['eligible']


# -- progress ------------------------------------------------------------------

def test_progress_cannot_skip_ahead_at_a_crossing_or_move_backwards():
    z = 1.5
    route = [[0, 0, z], [4, 0, z], [4, 4, z], [2, 4, z], [2, -2, z]]  # last leg crosses the first
    progress, error = project_progress(route, [2, 0, z], (0, .4))
    assert progress == [0, .5] and error == pytest.approx(0)
    progress, _ = project_progress(route, [1, 0, z], (0, .6))
    assert progress == [0, .6]
    progress, _ = project_progress(route, [4, 1, z], (0, .9))
    assert progress[0] == 1
    assert distance_along(route, [0, .5], [1, .5]) == pytest.approx(4.)
    assert distance_along(route, [1, .5], [0, .5]) == 0.


def test_observation_reports_remaining_interval_and_unavailable_values():
    src = source()
    assert src.consume(example('ready'), context(), wall=0) == 'READY'
    o = src.observe(context()['pose'], 10.)
    assert o['remaining_permitted_m'] == pytest.approx(5.)
    assert o['stopping_margin_m'] == pytest.approx(5.-.25)
    assert o['validity_remaining_s'] == pytest.approx(2.)
    assert 'speed_mps' in o['unavailable'] and 'commanded_target' in o['unavailable']
    assert source().observe(context()['pose'], 10.)['remaining_permitted_m'] is None


# -- admission ----------------------------------------------------------------

def test_duplicate_reads_are_harmless_and_older_publications_ignored():
    src = source(); v = message()
    assert src.consume(v, context(), wall=0) == 'READY'
    assert src.consume(deepcopy(v), context(), wall=1) == 'READY'
    newer = dict(v, sequence=5)
    assert src.consume(newer, context(), wall=1) == 'READY'
    assert src.consume(v, context(), wall=1) == 'REJECTED' and 'out-of-order' in src.reason
    assert src.sequence == 5
    assert src.consume(dict(newer, time_s=9.9), context(), wall=1) == 'REJECTED'
    assert 'reused' in src.reason
    assert src.consume(newer, context(), wall=4.5) == 'REJECTED'  # same retained bytes aren't liveness
    assert 'stalled' in src.reason


def test_stop_generation_latch_binds_active_route_even_when_withdrawal_missed():
    src = source(); v = message(); ctx = context()
    assert src.consume(v, ctx, wall=0) == 'READY'
    assert src.consume(dict(v, sequence=2, stop_generation=1), ctx, wall=0) == 'READY'  # idle: nothing to stop
    assert src.handled_stop == 1
    src.activate()
    assert src.disposition['value'] == 'active'
    # The unavailable publication (sequence 3) was never read; sequence 4 is valid again.
    later = dict(v, sequence=4, stop_generation=2)
    assert src.consume(later, ctx, wall=0) == 'STOP_REQUIRED'
    assert 'unhandled' in src.reason
    with pytest.raises(ValueError): src.confirm_stopped(1)
    src.confirm_stopped(2)
    assert src.disposition['value'] == 'stopped' and not src.active
    assert src.consume(later, ctx, wall=0) == 'READY'
    ctx['time_s'] = 11.5
    assert src.consume(later, ctx, wall=0) == 'REJECTED'  # too little lifetime left to stop


def test_planner_restart_under_active_route_needs_stop_and_new_start():
    src = source(); v = message(); ctx = context()
    assert src.consume(v, ctx, wall=0) == 'READY'
    src.activate()
    restarted = dict(v, session='restart', sequence=1, stop_generation=0)
    assert src.consume(restarted, ctx, wall=1) == 'STOP_REQUIRED'
    assert 'restarted' in src.reason
    src.confirm_stopped(0)
    assert src.consume(dict(restarted, sequence=2), ctx, wall=1) == 'READY'
    with pytest.raises(ValueError, match='Start'): src.activate()
    src.start(); src.activate()
    ctx['localization_epoch'] = 1
    assert src.consume(dict(restarted, sequence=3), ctx, wall=1) == 'STOP_REQUIRED'


def test_revision_cannot_change_geometry_and_goal_cannot_change_silently():
    src = source()
    v = message(); assert src.consume(v, context()) == 'READY'
    changed = deepcopy(v); changed['sequence'] = 2; changed['points'][1][0] = 6.
    assert src.consume(changed, context()) == 'REJECTED'
    assert 'geometry revision' in src.reason
    ctx = context(); ctx['goal']['revision'] = 2
    assert src.consume(v, ctx) == 'REJECTED'
    ctx = context(); ctx['provider_id'] = 'other'
    assert src.consume(v, ctx) == 'REJECTED' and 'context' in src.reason


def test_provider_frame_nonzero_origin_and_invalidated_epoch():
    ctx = context(); transform = ProviderFrame.from_context(ctx)
    assert transform.map_to_ned(2, 7, 1.5) == (-2, -3, -1.5)
    assert transform.ned_to_map(-2, -3, -1.5) == (2, 7, 1.5)
    ctx['localization_epoch'] = 1
    with pytest.raises(ValueError): ProviderFrame.from_context(ctx)
    ctx = context(); ctx.pop('vehicle_transform')
    with pytest.raises(ValueError): ProviderFrame.from_context(ctx)


# -- dnav publisher with a fake executor -------------------------------------

class FakeRun:
    """The NavRun surface RoutePublisher reads, without a provider or planner."""
    def __init__(self, instance):
        from dcmn.navigation import context_identity
        self.snapshot = dict(context_identity(context()), clock_domain_id='v')
        self.goal_descriptor = context()['goal']
        self.session = SimpleNamespace(identity='maps', context={'mapping_epoch': 0})
        self.attempts, self.recorder = 1, None
        self.time_s, self.grids = 10., {'scan': grid()}
        self.pose = [2., 2., 1.5]
        self.plan([[2., 2.], [7., 2.]])

    def plan(self, xy):
        self.route = SimpleNamespace(ok=True, status='ok', reason='',
            waypoints=[SimpleNamespace(x_m=float(x), y_m=float(y), z_m=1.5) for x, y in xy])

    def sim_time_s(self): return self.time_s
    def _grids(self): return self.grids
    def _pose(self): return dict(zip(('x_m', 'y_m', 'z_m'), self.pose))


def executor_status(instance, publisher, sequence, state, **fields):
    return {**dict(schema=STATUS_SCHEMA, vehicle_id=instance, session='exec', sequence=sequence, time_s=10.,
                   stop_generation=0, state=state, reason='fixture executor', dry_run=False, owns_control=True,
                   planner_session=publisher.session), **fields}


def test_publisher_pins_executing_geometry_and_waits_for_confirmed_stop():
    instance = 'pin-'+uuid.uuid4().hex[:8]
    run = FakeRun(instance)
    publisher = RoutePublisher(instance, ExecutionProfile.load(PROFILE))
    executor = Snapshot(instance, 'execution')
    try:
        first = publisher.publish(run, wall=0.)
        assert first['clearance']['eligible'] and first['geometry_revision'] == 1
        executor.write(executor_status(instance, publisher, 1, 'EXECUTING', geometry_revision=1, progress=[0, .2]))
        run.pose = [3., 2., 1.5]; run.plan([[3., 2.], [6., 3.]])  # an optional improvement appears
        moving = publisher.publish(run, wall=.1)
        assert moving['points'] == first['points'] and moving['geometry_revision'] == 1
        assert moving['clearance']['start'] == [0, .2] and moving['clearance']['eligible']
        assert moving['executor_reference']['progress'] == [0, .2]

        run.time_s = 20.  # evidence becomes stale while executing
        executor.write(executor_status(instance, publisher, 2, 'EXECUTING', geometry_revision=1, progress=[0, .2]))
        withdrawn = publisher.publish(run, wall=.2)
        assert not withdrawn['clearance']['eligible'] and withdrawn['stop_generation'] == 1
        assert withdrawn['points'] == first['points']

        # A dead executor (unchanged sequence) neither pins geometry nor confirms the stop.
        run.grids = {'scan': grid(time_s=20.)}
        dead = publisher.publish(run, wall=5.)
        assert not dead['clearance']['eligible'] and 'confirm stop' in dead['clearance']['reason']
        assert dead['points'] != first['points']

        executor.write(executor_status(instance, publisher, 3, 'HOLDING', geometry_revision=1,
                                       progress=[0, .2], stop_generation=1))
        replacement = publisher.publish(run, wall=5.1)
        assert replacement['clearance']['eligible'], replacement['clearance']
        assert replacement['points'][0] == [3., 2., 1.5] and replacement['stop_generation'] == 1
        assert replacement['clearance']['start'] == [0, 0.]

        executor.write(executor_status(instance, publisher, 4, 'HOLDING', geometry_revision=1,
                                       progress=[0, .2], stop_generation=1))
        run.plan([[3., 2.], [6., 4.]])
        pinned = publisher.publish(run, wall=5.2)
        assert pinned['points'] == replacement['points']
        assert pinned['geometry_revision'] == replacement['geometry_revision']
    finally:
        executor.close(); publisher.close()


def test_unpublishable_route_replaces_eligible_record_without_truncation():
    instance = 'big-'+uuid.uuid4().hex[:8]
    run = FakeRun(instance)
    publisher = RoutePublisher(instance, ExecutionProfile.load(PROFILE))
    try:
        assert publisher.publish(run, wall=0.)['clearance']['eligible']
        run.plan([(2. + i*.01, 2.) for i in range(300)])
        value = publisher.publish(run, wall=.1)
        assert value['points'] == [] and value['planning_status'] == 'unavailable'
        assert not value['clearance']['eligible'] and value['stop_generation'] >= 1
        assert Snapshot(instance).read() == value
    finally:
        publisher.close()


# -- live integration ---------------------------------------------------------

@pytest.fixture
def live(tmp_path):
    instance = 'dry-'+uuid.uuid4().hex[:8]
    bus = PymembusModuleBus(instance, 'simulator', 'fixture', create=True); assert bus.connect()
    provider = Context(instance); provider.start(tmp_path/'run', local_ned_origin=(5., 5., 0.))
    provider.publish_pose(dict(x_m=2., y_m=2., z_m=1.5, heading_deg=90., roll_deg=0., pitch_deg=0.), 10.)
    provider.set_goal('fixture', [7., 2., 1.5], role='mission')
    g = grid()
    pub = MapPublisher(instance, g.geometry, [dict(id='scan', sensor='scan', algorithm='fixture')],
        context=dict(frame_id='local', localization_epoch=0, clock_domain_id=instance, clock_epoch=0))
    pub.publish('scan', g.occupancy, g.observed_ms, 10.)
    alg = PymembusModuleBus(instance, 'algorithm', 'fixture')
    alg.publish('module.heartbeat', payload=dict(capabilities={'maps': ['scan']}))
    nav = NavRun(instance, ROOT, policy=load_policy('default', ROOT), execution_profile=PROFILE)
    dry = DryRun(instance, profile=PROFILE)
    try:
        nav.step(); yield provider, nav, dry, pub, tmp_path
    finally:
        dry.close(); nav.close(); pub.close(); alg.close(); provider.close(); bus.remove()


def test_live_dry_run_and_archive_rollover_without_any_vehicle_link(live, monkeypatch):
    import dway.link
    monkeypatch.setattr(dway.link, 'DsimLink', lambda *a, **k: pytest.fail('vehicle link constructed'))
    provider, nav, dry, pub, tmp = live
    assert nav.navigation.last['clearance']['eligible'], nav.navigation.last
    status = dry.step()
    assert status['state'] == 'READY', status
    assert status['disposition']['value'] == 'accepted'
    assert status['remaining_permitted_m'] > 0 and status['speed_mps'] is None
    assert not status['owns_control'] and status['commanded_target'] is None
    with pytest.raises(ValueError, match='live writer'): DryRun(dry.id, profile=PROFILE)
    first = dry.report_dir
    provider.rollover(tmp/'run2'); dry.step()
    assert first != dry.report_dir
    for name in ('summary.json', 'route.png', 'report.html', 'events.csv', 'events.jsonl'):
        assert (first/name).exists(), name
    from dcmn.archive import ArchiveReader
    reader = ArchiveReader(first)
    assert reader.validate()['complete']
    e = next(e for e in reader.events() if e['type'] == 'navigation.observed')
    assert reader.reconstruct(e)['reconstructed_grids']
    assert dry.events[0]['kind'] == 'retained_state'  # the new segment starts from a full baseline
    provider.set_goal('fixture', None, role='mission'); nav.step(); nav.replan(force=True); nav.navigation.publish(nav)
    assert dry.step()['state'] == 'REJECTED'
    assert dry.status['disposition']['value'] == 'rejected'


def test_dynamic_cli_flies_only_without_tours_or_world_maps():
    from dway.dway import parse_args
    # Dynamic mode is real flight: with --dry-run it plans without motion,
    # without it the executor commands the vehicle.
    args = parse_args(['--id', 'v', '--mode', 'dynamic'])
    assert not args.dry_run
    dry = parse_args(['--id', 'v', '--mode', 'dynamic', '--dry-run', '--no-ui'])
    assert dry.dry_run and dry.tour is None
    with pytest.raises(SystemExit): parse_args(['--id', 'v', '--mode', 'dynamic', '--dry-run', '--tour', 'bad'])
    with pytest.raises(SystemExit): parse_args(['--id', 'v', '--mode', 'dynamic', '--edit-map', 'bad'])
    with pytest.raises(SystemExit): parse_args(['--id', 'v', '--dry-run', '--tour', 'bad'])


def test_route_source_adapters_keep_tours_and_dynamic_apart():
    from dway.route_source import TourRouteSource, route_source
    tour = next((ROOT/'assets/tours').glob('*.json'), None) or next((ROOT/'assets/tours').iterdir())
    static = TourRouteSource(tour)
    assert static.kind == 'tour' and static.tour.waypoints
    dynamic = route_source(SimpleNamespace(mode='dynamic', id='v', planner='dnav'),
                           ExecutionProfile.load(PROFILE))
    assert dynamic.kind == 'dynamic' and not hasattr(dynamic, 'tour')


def test_proposal_route_is_dashed_and_distinct_from_permission():
    from dcmn.map_pane import Overlay, overlay_shapes, snapshot_image
    shapes = list(overlay_shapes(Overlay(route=((1., 1.), (4., 1.)), proposal=True,
                                         permitted=((1., 1.), (3., 1.))), cell_m=.5))
    lines = [s for s in shapes if s.kind == 'line']
    assert lines[0].dash and lines[0].width == 1.
    assert any(not s.dash and s.width > 1 for s in lines)  # the permitted interval is solid
    assert any(s.kind == 'rect' for s in shapes)            # with a square stop marker
    assert snapshot_image(grid(), overlay=Overlay(route=((1., 1.), (4., 1.)), proposal=True)) is not None


def test_dry_window_has_no_flight_controls_and_shows_permission(live):
    from dtest.tkfixture import hidden_tk
    from dway.dynamic import DryWindow
    provider, nav, dry, pub, tmp = live
    with hidden_tk() as root:
        window = DryWindow(dry, root=root)
        window.tick(); root.update_idletasks()
        text = window.text.get()
        assert 'READY' in text and 'SYNTHETIC' in text and 'stopping margin' in text
        assert window.pane.state.overlay.permitted and window.pane.state.overlay.proposal
        assert window.pane.state.overlay.target is None
        assert window.pane.snapshot() is not None
        import tkinter as tk
        def buttons(widget):
            for child in widget.winfo_children():
                yield from ([child] if isinstance(child, (tk.Button,)) or child.winfo_class() == 'TButton' else [])
                yield from buttons(child)
        assert not list(buttons(root))
        provider.set_goal('fixture', None, role='mission')
        nav.connect(); nav.replan(force=True); nav.navigation.publish(nav)
        window._map.reset(); window._text.reset(); window.tick()
        assert not window.pane.state.overlay.permitted
        assert 'REJECTED' in window.text.get()


def test_nav_records_full_validation_inputs_even_when_plan_unchanged(live):
    provider, nav, dry, pub, tmp = live
    nav.navigation.publish(nav)
    directory = nav.recorder.directory
    nav.close()
    from dcmn.archive import ArchiveReader
    reader = ArchiveReader(directory)
    events = [e for e in reader.events() if e['type'] == 'navigation.snapshot']
    assert len(events) >= 2
    assert 'scan' in reader.reconstruct(events[-1])['reconstructed_grids']
    assert events[-1]['data']['profile']['calibrated'] is False


# -- plan permission (research): trust the planned route, withdraw on observed obstacles ----------

def plan_profile(**overrides):
    return ExecutionProfile.load(ROOT/'assets/execution_profiles/sim-target.json', overrides)


def test_plan_permission_trusts_unknown_space_and_reports_the_evidence_verdict():
    p, pts = plan_profile(), [[2., 2., 1.5], [7., 2., 1.5]]
    unknown = np.full(grid().geometry.shape, 255, np.uint8)
    c = Clearance(p).check(pts, pts[0], {'scan': grid(unknown, np.zeros(grid().geometry.shape, np.uint32))}, 10.)
    assert c['eligible'] and c['reaches_goal'] and c['permission'] == 'plan'
    assert c['end'] == [0, 1.] and c['valid_until_s'] == pytest.approx(10. + p.max_age_s)
    # The strict verdict on the same inputs is published alongside, for comparison.
    assert c['evidence_check']['eligible'] is False and 'unknown' in c['evidence_check']['reason']
    value = message(); value.update(profile=p.digest, clearance=c)
    validate(value)


def test_plan_permission_is_withdrawn_where_an_observed_obstacle_blocks_the_route():
    p, pts = plan_profile(), [[2., 2., 1.5], [7., 2., 1.5], [7., 4., 1.5]]
    occ = np.zeros(grid().geometry.shape, np.uint8); occ[0, 4, 10] = 254  # cell centre (5.25, 2.25)
    c = Clearance(p).check(pts, pts[0], {'scan': grid(occ)}, 10.)
    assert not c['eligible'] and 'blocked' in c['reason'] and 2. < c['blocked_at_m'] < 3.5
    # Behind the vehicle it no longer matters: the remaining route is clear.
    assert Clearance(p).check(pts, [7., 2., 1.5], {'scan': grid(occ)}, 10., start=(1, 0.))['eligible']
    # Outside the evidence grid is unknown space, which plan permission trusts.
    assert Clearance(p).check([[2., 2., 1.5], [14., 2., 1.5]], [2., 2., 1.5], {'scan': grid()}, 10.)['eligible']


def test_profile_dials_override_fields_and_change_the_digest():
    base = plan_profile()
    faster = plan_profile(speed_mps=2.5, permission='evidence')
    assert faster.speed_mps == 2.5 and faster.permission == 'evidence' and faster.digest != base.digest
    assert ExecutionProfile.load('sim-target', ['speed_mps=2.5', 'permission=evidence']) == faster
    with pytest.raises(ValueError, match='unknown execution profile key'):
        ExecutionProfile.load('sim-target', ['sped_mps=2'])
    with pytest.raises(ValueError, match='permission'):
        ExecutionProfile.load('sim-target', ['permission=maybe'])


def test_plan_permission_ignores_remembered_vetoes_the_planner_no_longer_sees():
    """Seen live: an old optical-flow veto on every fresh route withdrew plan
    permission forever. Plan permission judges the evidence as it is now."""
    p, pts = plan_profile(), [[2., 2., 1.5], [7., 2., 1.5]]
    occ = np.zeros(grid().geometry.shape, np.uint8); occ[0, 4, 10] = 254
    c = Clearance(p)
    assert not c.check(pts, pts[0], {'scan': grid(occ)}, 10.)['eligible']          # observed now: withdrawn
    # Another source now reports the cell free; the strict veto is still remembered ...
    later = c.check(pts, pts[0], {'scan': grid(occ), 'other': grid()}, 11.)
    assert c.veto['scan'][4, 10]
    # ... but once the occupied mark is gone from current evidence the route is permitted.
    assert c.check(pts, pts[0], {'scan': grid(), 'other': grid()}, 12.)['eligible']
    # The strict check still honours the remembered veto: it permits only the
    # stretch before the cell and does not reach the goal.
    assert later['evidence_check']['reaches_goal'] is False
    assert 'occupied' in later['evidence_check']['reason']
