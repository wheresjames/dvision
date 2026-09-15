"""dalg as a continuous, truth-independent evidence producer.

Everything here runs against the deterministic fixture provider: a real session
context, real sensor plane and real module bus, and no simulator, world file,
tour or truth channel anywhere in the process.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from dalg.profiles import load_profiles
from dalg.run import DalgRun
from dcmn.archive import ArchiveReader
from dcmn.context import Context
from dcmn.mapping import MappingConfig
from dcmn.maps import MapSession
from dcmn.module_bus import PymembusModuleBus
from dnav import route as R
from dnav.plan import NavRun
from dnav.policy import load_policy
from dtest.provider import FixtureProvider

ROOT = Path(__file__).resolve().parents[1]


class Rig:
    """A provider, a dalg run and (optionally) a dnav run on one instance."""

    def __init__(self, tmp_path, profiles=('lidar-baseline',), *, sensors=('scan',), provider=True,
                 mapping=None, nav=False, goal=None, start=(0., 0., 1.5, 90.)):
        self.instance = 'dalg-' + uuid.uuid4().hex[:8]
        self.root = tmp_path
        self.sensors, self.start = sensors, start
        self.provider = FixtureProvider(self.instance, tmp_path/'session', sensors=sensors, start=start) \
            if provider else None
        self.run = DalgRun(self.instance, load_profiles(list(profiles), ROOT), ROOT, mapping=mapping)
        self.nav = (NavRun(self.instance, ROOT, policy=load_policy('default', ROOT), goal=goal)
                    if nav else None)
        if self.nav is not None:
            # Discovery backoff and replan pacing are wall-clock; these loops
            # run far faster than real time, so both are neutralised here. A
            # tick still steps the real loop -- it must not skip the replan
            # just because the wall clock says a tenth of a second has not
            # passed since the last one.
            self.nav.session.probe_interval_s = self.nav.session.max_probe_interval_s = 0.
            self.nav._pace.period_s = 0.

    def start_provider(self):
        self.provider = FixtureProvider(self.instance, self.root/'session', sensors=self.sensors,
                                        start=self.start)

    def tick(self, count=1):
        for _ in range(count):
            if self.provider is not None: self.provider.step()
            # The sensor registry probe is wall-clock throttled; the other
            # wall-clock paces were neutralised at construction.
            self.run.sensor_session.probe.last_probe = -1e9
            self.run.step()
            if self.nav is not None: self.nav.step()

    def close(self):
        if self.nav is not None: self.nav.close()
        self.run.close()
        if self.provider is not None: self.provider.close()


@pytest.fixture
def rig(tmp_path):
    made = []
    def build(*args, **kwargs):
        made.append(Rig(tmp_path, *args, **kwargs)); return made[-1]
    yield build
    for item in made: item.close()


def grid(rig, source='scan-lidar_inverse'):
    return rig.run.evidence_grids().get(source)


# -- baselines, singly and composed --------------------------------------------------

def test_the_lidar_baseline_publishes_evidence_dnav_plans_on(rig):
    r = rig(nav=True, goal=(8., 2.))
    r.provider.fly([(2., 0.), (2., 2.5)])
    r.tick(120)
    assert r.run.state == 'RUNNING' and r.run.admission == 'admitting'
    lidar = grid(r)
    assert lidar is not None and lidar.observed.any()
    # Coverage came from the pose and the goal, never from a world size.
    assert r.run.geometry_basis == 'goal'
    assert r.nav.cost_map is not None
    assert r.nav.route.status == R.OK, r.nav.route.reason


def test_a_missing_lidar_leaves_camera_output_usable_by_dnav(rig):
    r = rig(('ground-plane-baseline', 'lidar-baseline'), sensors=('front',), nav=True, goal=(6., 0.))
    r.provider.fly([(1., 0.), (1., 1.)])
    r.tick(150)
    assert r.run.state == 'RUNNING'
    assert set(r.run.evidence_grids()) == {'front-ground_plane'}
    assert 'absent' in r.run.unavailable_sources['scan-lidar_inverse']
    camera = r.run.evidence_grids()['front-ground_plane']
    assert camera.observed.any()
    assert set(r.nav.session.sources) == {'front-ground_plane'}
    assert r.nav.route.status == R.OK, r.nav.route.reason


def test_camera_and_lidar_together_publish_one_coherent_generation(rig):
    r = rig(('ground-plane-baseline', 'lidar-baseline'), sensors=('scan', 'front'))
    r.tick(90)
    session = MapSession(r.instance)
    try:
        session.poll()
        assert set(session.sources) == {'front-ground_plane', 'scan-lidar_inverse'}
        entries = session.sources.values()
        assert len({(e['mapping_epoch'], e['localization_epoch'], e['clock_epoch'], e['geometry_revision'])
                    for e in entries}) == 1
    finally:
        session.close()


def test_no_sensors_leaves_an_observable_waiting_process(rig):
    r = rig(sensors=())
    reader = PymembusModuleBus(r.instance, 'observer', 'test', read_only=True)
    try:
        assert reader.connect()
        r.run._last_presence = -1e9
        r.tick(10)
        assert r.run.state == 'WAITING_SENSORS' and r.run.sources is None
        beats = [e for e in reader.receive() if e.implementation == 'dalg' and e.type == 'module.heartbeat']
        assert beats and beats[-1].payload['state'] == 'WAITING_SENSORS'
        assert beats[-1].payload['capabilities']['maps'] is False
    finally:
        reader.close()


def test_an_incompatible_sensor_is_a_configuration_error_not_a_missing_device(rig, tmp_path):
    path = tmp_path/'wrong.json'
    path.write_text(json.dumps({'name': 'wrong', 'sources': [
        {'sensor': 'front', 'algorithm': 'lidar_inverse', 'settings': {}}]}))
    r = rig((str(path),), sensors=('scan', 'front'))
    r.tick(20)
    assert r.run.state == 'CONFIGURATION_ERROR'
    assert 'needs lidar.scan2d' in r.run.reason


# -- lifecycle ---------------------------------------------------------------------------

def test_a_late_provider_is_waited_for_and_then_used(rig):
    r = rig(provider=False)
    r.tick(5)
    assert r.run.state == 'WAITING_PROVIDER'
    r.start_provider()
    r.tick(40)
    assert r.run.state == 'RUNNING' and grid(r) is not None


def test_without_a_goal_coverage_is_a_minimum_square_around_the_first_pose(rig):
    r = rig(start=(-30., -40., 1.5, 0.))
    r.tick(30)
    geometry = r.run.geometry
    assert r.run.geometry_basis == 'pose'
    assert geometry.bounds_m() == (-50., -60., -10., -20.)
    assert geometry.origin_x_m < 0 and geometry.origin_y_m < 0


def test_a_goal_present_at_first_pose_sizes_coverage_and_a_later_one_never_resizes(rig):
    r = rig()
    Context(r.instance).set_goal('mission', (40., 0.), role='mission')
    r.tick(30)
    first = r.run.geometry
    assert r.run.geometry_basis == 'goal'
    assert first.extent_m[0] == pytest.approx(100.)       # 2.5 x 40 m, side not radius
    Context(r.instance).set_goal('mission', (10., 10.), role='mission')
    r.tick(10)
    assert r.run.geometry == first


def test_a_coincident_goal_gets_the_minimum_side(rig):
    r = rig()
    Context(r.instance).set_goal('mission', (0., 0.), role='mission')
    r.tick(30)
    assert r.run.geometry.extent_m == pytest.approx((40., 40.), abs=.5)


def test_a_goal_outside_coverage_is_reported_and_not_clipped(rig):
    r = rig(nav=True)
    r.tick(30)
    r.nav.set_goal(500., 0.)
    r.tick(10)
    assert r.run.coverage == 'goal outside coverage'
    assert r.nav.route.status == R.OUTSIDE_COVERAGE
    assert r.run.geometry.bounds_m()[2] < 500


@pytest.mark.parametrize('mapping,needle', [
    (MappingConfig(budget_bytes=1 << 20), 'budget'),
    (MappingConfig(bounds=(0., 0., 5000., 5000.)), 'limit is 4,000,000'),
])
def test_allocation_failures_are_explicit_and_nothing_is_clipped(rig, mapping, needle):
    r = rig(mapping=mapping)
    r.tick(20)
    assert r.run.state == 'ALLOCATION_FAILED'
    assert needle in r.run.reason
    assert r.run.sources is None


def test_mission_completion_is_recorded_and_perception_continues(rig):
    r = rig()
    r.tick(30)
    bus = PymembusModuleBus(r.instance, 'navigator', 'test')
    bus.publish('run.completed', run_id='m1', payload={'outcome': 'complete'})
    bus.close()
    r.tick(30)
    assert r.run.state == 'RUNNING' and not r.run.done
    assert r.run.provenance['last_mission_outcome'] == 'complete'


# -- pose admission ----------------------------------------------------------------------

def test_an_invalid_pose_pauses_admission_and_keeps_evidence(rig):
    r = rig()
    r.tick(60)
    admitted = r.run.counters['admitted']
    r.provider.pose_valid = False
    r.tick(35)
    assert r.run.admission == 'paused' and 'invalid' in r.run.reason
    assert r.run.counters['admitted'] == admitted
    before = grid(r)
    r.tick(60)
    after = grid(r)
    # Retained and republished with its true observation times: the record
    # time advances, the content (and so the revision) does not.
    assert np.array_equal(after.observed_ms, before.observed_ms)
    assert after.revision == before.revision and after.sim_time_s > before.sim_time_s
    assert r.run._mapping_epoch == 1
    r.provider.pose_valid = True
    r.tick(30)
    assert r.run.admission == 'admitting' and r.run.counters['admitted'] > admitted
    assert r.run._mapping_epoch == 1


# -- resets ------------------------------------------------------------------------------

@pytest.mark.parametrize('event,cause', [
    ('announce_localization_reset', 'localization epoch changed'),
    ('announce_clock_reset', 'clock epoch changed'),
    ('sensor_reset', 'sensor transport reset'),
])
def test_provider_discontinuities_rebuild_unknown_without_ending_anything(rig, event, cause):
    r = rig(nav=True, goal=(8., 2.))
    r.provider.fly([(2., 0.), (2., 2.5)])
    r.tick(120)
    assert r.nav.route.status == R.OK
    old = grid(r)
    reset_at = r.provider.time_s if event != 'announce_clock_reset' else 0.
    getattr(r.provider, event)()
    r.tick(60)
    assert r.run.state == 'RUNNING' and not r.run.done
    assert r.run._mapping_epoch == 2
    assert r.run.counters['resets'] == 1
    new = grid(r)
    assert new.entry['mapping_epoch'] == 2
    # Nothing carried over: no cell predates the discontinuity.
    stamps = new.observed_ms[new.observed]
    assert stamps.size and stamps.min() >= reset_at * 1000 - 1
    reader_dir = r.run.recorder.directory
    r.run.recorder.close()
    kinds = [(e['type'], e['data'].get('cause') or e['data'].get('reason'))
             for e in ArchiveReader(reader_dir).events() if e['type'] in ('epoch.changed', 'mapping.retired')]
    assert ('epoch.changed', cause) in kinds


def test_a_localization_reset_rejects_old_epoch_samples_and_goals(rig):
    r = rig(nav=True, goal=(8., 2.))
    r.tick(60)
    r.provider.announce_localization_reset()
    r.tick(3)
    assert r.nav.route.status in (R.NO_GOAL, R.STALE_MAP, R.FRAME_MISMATCH)
    r.tick(60)
    # The goal was expressed in the invalidated frame and must be reissued.
    assert r.nav.route.status == R.NO_GOAL and 'reissue' in r.nav.route.reason
    r.nav.set_goal(8., 2.)
    r.tick(30)
    assert r.nav.route.status == R.OK, r.nav.route.reason
    assert r.nav.route.as_dict()['map_revision'] >= 1


def test_temporal_pairs_never_cross_a_sensor_epoch(rig):
    r = rig(('optical-flow-baseline',), sensors=('scan', 'front'))
    r.tick(60)
    state = next(iter(r.run.sources.states.values()))
    assert state.evidence.algorithm.previous is not None
    r.provider.sensor_reset()
    r.tick(3)
    fresh = next(iter(r.run.sources.states.values()))
    assert fresh is not state
    assert fresh.evidence.algorithm.previous is None or fresh.samples <= 1


def test_an_operator_mapping_reset_installs_new_bounds(rig):
    r = rig()
    r.tick(60)
    Context(r.instance).request_reset((-5., -5., 15., 15.))
    r.tick(10)
    assert r.run.geometry.bounds_m() == (-5., -5., 15., 15.)
    assert r.run._mapping_epoch == 2


def test_a_mapping_reset_that_does_not_fit_leaves_the_current_map(rig):
    r = rig(mapping=MappingConfig(budget_bytes=16 << 20))
    r.tick(60)
    before = r.run.geometry
    Context(r.instance).request_reset((-500., -500., 500., 500.))
    r.tick(10)
    assert r.run.state == 'RUNNING' and 'reset rejected' in r.run.reason
    assert r.run.geometry == before and r.run._mapping_epoch == 1
    assert grid(r) is not None


# -- membership and gaps -----------------------------------------------------------------

def test_a_sensor_gap_recovers_without_a_rebuild(rig):
    r = rig(('ground-plane-baseline', 'lidar-baseline'), sensors=('scan', 'front'))
    r.tick(60)
    lidar = r.run.sources.states['scan-lidar_inverse']
    camera = r.run.sources.states['front-ground_plane']
    r.provider.muted.add('scan')
    before = (lidar.samples, camera.samples)
    r.tick(60)
    assert lidar.samples == before[0] and camera.samples > before[1]
    r.provider.muted.clear()
    r.tick(30)
    assert r.run.sources.states['scan-lidar_inverse'] is lidar and lidar.samples > before[0]
    assert r.run._mapping_epoch == 1


def test_a_removed_sensor_cannot_strand_dnav_and_a_new_one_waits_for_a_reset(rig):
    r = rig(('ground-plane-baseline', 'lidar-baseline'), sensors=('scan', 'front'),
            nav=True, goal=(6., 0.))
    r.tick(90)
    assert set(r.nav.session.sources) == {'front-ground_plane', 'scan-lidar_inverse'}
    r.provider.set_sensors(('front',))
    r.tick(90)
    assert set(r.run.sources.states) == {'front-ground_plane'}
    assert 'removed' in r.run.unavailable_sources['scan-lidar_inverse'] or \
        'absent' in r.run.unavailable_sources['scan-lidar_inverse']
    r.tick(30)
    assert set(r.nav.session.sources) == {'front-ground_plane'}
    assert r.nav.route.status == R.OK, r.nav.route.reason
    r.provider.set_sensors(('scan', 'front'))
    r.tick(60)
    assert set(r.run.sources.states) == {'front-ground_plane'}
    assert 'explicit mapping reset' in r.run.unavailable_sources['scan-lidar_inverse']
    Context(r.instance).request_reset()
    r.tick(60)
    assert set(r.run.sources.states) == {'front-ground_plane', 'scan-lidar_inverse'}


# -- records -----------------------------------------------------------------------------

def test_every_publication_is_archived_and_content_is_stored_once(rig):
    r = rig()
    r.tick(150)
    directory = r.run.recorder.directory
    r.run.recorder.close()
    reader = ArchiveReader(directory)
    report = reader.validate()
    assert report['complete'], report
    published = [e for e in reader.events() if e['type'] == 'evidence.published']
    assert len(published) >= 4
    contents = [e['grids']['scan-lidar_inverse']['content'] for e in published]
    chunks = reader.chunks.values()
    stored = [c for chunk in chunks for c in chunk['contents']]
    assert len(stored) == len(set(stored)) == len(set(contents))
    rebuilt = reader.reconstruct(published[-1])['reconstructed_grids']['scan-lidar_inverse']
    live = grid(r)
    assert np.array_equal(rebuilt.occupancy, live.occupancy)
    assert np.array_equal(rebuilt.observed_ms, live.observed_ms)
    admitted = [e for e in reader.events() if e['type'] == 'sample.admitted']
    assert admitted and admitted[0]['data']['pose_context']['estimate_kind'] == 'ideal'


def test_session_rollover_opens_a_new_archive_with_a_full_baseline(rig, tmp_path):
    r = rig()
    r.tick(60)
    first = r.run.recorder.directory
    r.provider.rollover(tmp_path/'next')
    r.tick(10)
    second = r.run.recorder.directory
    assert second.parent == tmp_path/'next'/'dalg'
    assert ArchiveReader(first).validate()['complete']
    assert (first.parent/'summary.json').is_file()
    r.run.recorder.close()
    events = ArchiveReader(second).events()
    assert events[0]['type'] == 'retained_state'
    assert r.run._mapping_epoch == 1


def test_a_restarted_provider_is_a_new_frame_and_mapping_rebuilds(rig):
    r = rig()
    r.tick(60)
    assert r.run._mapping_epoch == 1
    r.provider.close(); r.provider = None
    r.tick(5)
    assert r.run.state == 'WAITING_PROVIDER'
    # Evidence is not republished as if it were live while the provider is gone.
    assert r.run.admission == 'paused'
    r.start_provider()
    r.tick(60)
    assert r.run.state == 'RUNNING' and r.run._mapping_epoch == 2
    assert grid(r).entry['pose_provider_id'] == r.provider.context.owner
