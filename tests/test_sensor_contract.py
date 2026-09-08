"""Sensor wire, lifecycle, validation and transform contracts.

These are the guarantees producers and consumers are written against: the
record envelope, the registry commit, generation and session identity, matched
camera intake, and the separation between the shared compact ring and the
dedicated bulk-array rings.
"""
import json
import math
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from dcmn.sensors import (CAMERA_FRAME, LIDAR_FRAME, PACKED_ARRAY, RANGE_SAMPLE,
                          HEADER, RecordRing, SensorSamples, SensorVideo,
                          decode_record, encode_record, unpack_array)
from dsim.dsim import DroneState
from dsim.profiles import (DroneProfile, camera_profile, default_profile,
                           stereo_pair)
from dsim.realism import REALISM_DEFAULTS, Realism
from dsim.sensor_manager import SensorManager
from dsim.transforms import pose_angles, resolve, rotation


@pytest.fixture
def profile():
    """A tiny camera whose 30 Hz rate divides a 30 Hz physics step, so every
    tick is a capture and a test does not have to count schedule slots."""
    return DroneProfile.parse(camera_profile(16, 12, physics_hz=30.))


class Renderer:
    """Fills each requested view with a distinct constant, and records the pass."""
    def __init__(self): self.passes = []
    def render_views(self, requests):
        self.passes.append([name for name, *_ in requests])
        for index, (_name, _model, _pose, frame) in enumerate(requests):
            frame[:] = [12 + index, 34, 56]
    def drop_views(self, camera_ids): pass


def instance(): return 'sensor-test-' + uuid.uuid4().hex[:10]


class _Vehicle:
    """The context the state sensors read, with a flat local datum."""
    realism = Realism.from_settings(REALISM_DEFAULTS)
    origin_alt_m = 34.0

    @staticmethod
    def map_to_gps(x, y, z):
        return 52.0 + y * 1e-5, 13.0 + x * 1e-5, 34.0 + z


def manager(profile, *, name=None, objects=(), renderer=None, seed=0, vehicle=None):
    return SensorManager(name or instance(), profile,
                         SimpleNamespace(objects=list(objects)),
                         renderer or Renderer(), vehicle=vehicle, seed=seed)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def test_canonical_roundtrip(profile, tmp_path):
    path = tmp_path / 'p.json'; profile.save(path)
    assert DroneProfile.load(path).digest == profile.digest
    d = profile.data; d['name'] = 'changed'
    assert profile.data['name'] == 'camera-only'


@pytest.mark.parametrize('mutate,field', [
    (lambda d: d.update(physics_hz=100), 'rate_hz'),
    (lambda d: d.update(primary_camera='missing'), 'primary_camera'),
    (lambda d: d['sensors'][0].update(parent='missing'), 'parent'),
    (lambda d: d['sensors'][0].update(id='body'), 'id'),
    (lambda d: d['sensors'][0].update(type='sonar.magic'), 'type'),
    (lambda d: d['sensors'][0]['model'].update(width_px=0), 'width_px'),
    (lambda d: d['sensors'][0]['model'].update(fx_px=100), 'exactly one'),
    (lambda d: d['sensors'][0]['model'].update(samples=8), 'unknown field'),
    (lambda d: d['sensors'][0].update(sync_group='alone'), 'at least two'),
    (lambda d: d['sensors'][0]['model'].update(width_px=1000000), 'memory budget'),
])
def test_invalid_profiles(mutate, field):
    d = default_profile(); mutate(d)
    with pytest.raises(ValueError, match=field): DroneProfile.parse(d)


def test_a_sync_group_requires_a_matching_lens_and_rate():
    d = default_profile()
    d['sensors'] = stereo_pair('nav', pose=dict(z_m=.1))
    d['primary_camera'] = 'nav_left'
    assert DroneProfile.parse(d).sync_group('nav_stereo')[0]['id'] == 'nav_left'
    d['sensors'][1]['rate_hz'] = 10.
    with pytest.raises(ValueError, match='identical type, rate and lens'):
        DroneProfile.parse(d)


def test_a_stereo_pair_stores_explicit_left_and_right_poses():
    left, right = stereo_pair('nav', baseline_m=.12)
    assert left['pose_parent']['y_m'] == pytest.approx(-.06)
    assert right['pose_parent']['y_m'] == pytest.approx(.06)
    assert left['sync_group'] == right['sync_group'] == 'nav_stereo'


def test_an_array_sensor_reserves_its_own_ring():
    d = camera_profile()
    d['sensors'].append(dict(id='scan', type='lidar.scan2d', parent='body',
                             rate_hz=10., model=dict(samples=720)))
    plan = DroneProfile.parse(d).plan
    assert plan['arrays']['scan']['record_bytes'] == HEADER.size + 720 * 5
    assert plan['arrays']['scan']['capacity'] >= plan['arrays']['scan']['record_bytes']
    assert plan['compact_hz'] == 40.


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def test_rotation_golden():
    assert np.allclose(rotation(yaw_deg=90) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(rotation(pitch_deg=90) @ [1, 0, 0], [0, 0, 1])
    assert np.allclose(rotation(roll_deg=90) @ [0, 1, 0], [0, 0, -1])


@pytest.mark.parametrize('yaw,expected', [(270, (2, 2, 3)), (0, (1, 1, 3)),
                                          (90, (0, 2, 3)), (180, (1, 3, 3))])
def test_mount_translation_rotates(yaw, expected):
    d = default_profile(); d['sensors'][0]['pose_parent'] = {'y_m': 1}
    t = resolve(DroneProfile.parse(d).data, 'front', DroneState(1, 2, 3, yaw_deg=yaw))
    assert np.allclose(t[:3, 3], expected)


def test_nested_mount_and_cycle():
    d = default_profile()
    d['mounts'] = [dict(id='rig', type='mount.fixed', parent='body', pose_parent={'yaw_deg': 90})]
    d['sensors'][0].update(parent='rig', pose_parent={'x_m': 1})
    p = DroneProfile.parse(d)
    assert np.allclose(resolve(p.data, 'front', DroneState(0, 0, 0, yaw_deg=270))[:3, 3], [1, 0, 0])
    d['mounts'][0]['parent'] = 'rig'
    with pytest.raises(ValueError, match='cycle'): DroneProfile.parse(d)


def test_a_ptz_mount_composes_its_state_before_its_children():
    """body -> PTZ -> rig -> camera, in the documented order."""
    d = default_profile()
    d['mounts'] = [
        dict(id='ptz', type='mount.ptz', parent='body', pose_parent={'z_m': .5},
             state=dict(pan_deg=90.)),
        dict(id='rig', type='rig.fixed', parent='ptz', pose_parent={'x_m': 2.}),
    ]
    d['sensors'][0].update(parent='rig', pose_parent={})
    t = resolve(DroneProfile.parse(d).data, 'front', DroneState(0, 0, 0, yaw_deg=270))
    # Heading 0 is -Y; panning 90 degrees right turns the rig arm to +X.
    assert np.allclose(t[:3, 3], [2, 0, .5])
    assert pose_angles(t)['heading_deg'] == pytest.approx(90.)


def test_a_ptz_state_outside_its_limits_is_rejected():
    d = default_profile()
    d['mounts'] = [dict(id='ptz', type='mount.ptz', parent='body',
                        state=dict(tilt_deg=-95.), limits=dict(tilt_deg=[-90., 30.]))]
    with pytest.raises(ValueError, match='outside limits'): DroneProfile.parse(d)


# ---------------------------------------------------------------------------
# Record envelope
# ---------------------------------------------------------------------------

def test_envelope_fixture():
    raw = encode_record('00' * 16, 1, 'front', 2, 3, 4000, {'value': None})
    assert HEADER.size == 128
    assert raw[:8].hex() == '4456533101008000'
    decoded = decode_record(raw)
    assert decoded['sequence'] == 2 and decoded['payload'] == {'value': None}
    with pytest.raises(ValueError): decode_record(raw[:-1])
    with pytest.raises(ValueError): encode_record('00' * 16, 1, 'front', 2, 3, 4, {'v': float('nan')})


def test_a_packed_array_payload_round_trips_through_its_layout():
    from dsim.sensor_models import array_layout, pack_array
    ranges = np.array([1.5, np.nan, 3.25], np.float32)
    confidence = np.array([255, 0, 128], np.uint8)
    layout = array_layout('lidar.scan2d', dict(samples=3))
    raw = pack_array(ranges, confidence)
    assert len(raw) == 3 * 5
    assert raw[:4] == np.float32(1.5).tobytes()
    fields = unpack_array(decode_record(
        encode_record('00' * 16, 1, 'scan', 1, 1, 0, raw, payload_type=PACKED_ARRAY))['payload'],
        layout)
    assert np.array_equal(fields['range_m'], ranges, equal_nan=True)
    assert np.array_equal(fields['confidence'], confidence)


def test_record_ring_overrun_and_binary():
    name = '/dvision2.records.' + uuid.uuid4().hex
    w = RecordRing(name, 4096, create=True); r = RecordRing(name, 4096)
    try:
        raw = encode_record('00' * 16, 1, 'lidar', 1, 1, 0, b'\0\xff\x80', payload_type=PACKED_ARRAY)
        w.write(raw); assert list(r.drain())[0]['payload'] == b'\0\xff\x80'
        for seq in range(100): w.write(encode_record('00' * 16, 1, 'imu', seq, seq, 0, {'x': 'x' * 300}))
        list(r.drain()); assert r.overruns > 0
    finally:
        w.close(); r.close(); w.unlink()


def test_a_record_too_large_for_its_ring_is_refused_rather_than_corrupting_it():
    """The visible-failure rule: a record that cannot fit is an error at the
    writer, not a truncated write the next reader has to notice."""
    name = '/dvision2.records.' + uuid.uuid4().hex
    w = RecordRing(name, 4096, create=True); r = RecordRing(name, 4096)
    try:
        with pytest.raises(ValueError, match='exceeds ring capacity'):
            w.write(b'x' * 4096)
        assert list(r.drain()) == [], "the refused record wrote nothing"
    finally:
        w.close(); r.close(); w.unlink()


# ---------------------------------------------------------------------------
# Discovery and camera intake
# ---------------------------------------------------------------------------

def test_matching_rollover_restart(profile):
    sensors = manager(profile)
    client = SensorVideo(sensors.publisher.instance)
    try:
        assert client.connect()
        assert client.getSeq() == 0
        sensors.tick(DroneState(1, 2, 3), .1)
        assert client.getSeq() > 0
        assert np.all(client[0] == [12, 34, 56])
        assert client.record['sim_time_us'] == 100000
        video = sensors.publisher.channels.cameras['front']
        assert client.record['payload']['video_sequence'] == video.getSeq()
        session = client.identity[0]

        sensors.apply(profile); client.last_probe = -1e9
        assert client.connect() and client.identity == (session, 2)
        assert client.getSeq() == 0
        sensors.tick(DroneState(1, 2, 3), .2)
        assert client.getSeq() > 0

        sequence = sensors.publisher.sequences['front']
        capture = sensors.capture_id
        sensors.reset(); sensors.tick(DroneState(1, 2, 3), .3)
        assert sensors.publisher.sequences['front'] > sequence
        assert sensors.capture_id > capture
        client.getSeq(); assert client.record['reset_epoch'] == 1

        sensors.close(); sensors = manager(profile, name=sensors.publisher.instance)
        client.last_probe = -1e9; assert client.connect()
        assert client.identity[0] != session and client.getSeq() == 0
        sensors.tick(DroneState(1, 2, 3), 0.)
        assert client.getSeq() > 0
    finally:
        client.close(); sensors.close()


def test_unmatched_frame_is_pending(profile):
    sensors = manager(profile)
    client = SensorVideo(sensors.publisher.instance)
    try:
        assert client.connect()
        ring = sensors.publisher.channels.records
        original, deferred = ring.write, []
        ring.write = deferred.append
        sensors.tick(DroneState(0, 0, 0), .1)
        assert client.getSeq() == 0 and client.pending
        original(deferred[0])
        assert client.getSeq() > 0 and not client.pending
    finally:
        client.close(); sensors.close()


def test_every_camera_gets_a_channel_and_a_synchronized_capture():
    d = camera_profile(16, 12, physics_hz=30.)
    d['sensors'] = stereo_pair('nav', pose=dict(z_m=.1), model=dict(
        width_px=16, height_px=12, fov_h_deg=70., near_m=.15, far_m=150.))
    d['primary_camera'] = 'nav_left'
    renderer = Renderer()
    sensors = manager(DroneProfile.parse(d), renderer=renderer)
    try:
        manifest = sensors.manifest
        channels = {sid: manifest['sensors'][sid]['channel'] for sid in ('nav_left', 'nav_right')}
        assert len(set(channels.values())) == 2
        sensors.tick(DroneState(1, 2, 3), .5)
        # One render pass covers both members of the group.
        assert renderer.passes == [['nav_left', 'nav_right']]
        samples = SensorSamples(sensors.publisher.instance)
        try:
            records = {r['sensor_id']: r for r in samples.drain()}
        finally:
            samples.close()
        assert set(records) == {'nav_left', 'nav_right'}
        left, right = records['nav_left'], records['nav_right']
        assert left['capture_id'] == right['capture_id']
        assert left['sim_time_us'] == right['sim_time_us'] == 500000
        assert left['payload']['sync_group'] == 'nav_stereo'
        # Distinct extrinsics, identical capture: the baseline is in the pose,
        # rotated into the map by the body heading rather than added to an axis.
        assert left['payload']['pose'] != right['payload']['pose']
        assert left['payload']['body'] == right['payload']['body']
    finally:
        sensors.close()


def test_a_second_camera_can_be_opened_by_id():
    d = camera_profile(16, 12, physics_hz=30.)
    d['sensors'] = stereo_pair('nav', pose=dict(z_m=.1), model=dict(
        width_px=16, height_px=12, fov_h_deg=70., near_m=.15, far_m=150.))
    d['primary_camera'] = 'nav_left'
    sensors = manager(DroneProfile.parse(d))
    right = SensorVideo(sensors.publisher.instance, 'nav_right')
    try:
        assert right.connect() and right.selected == 'nav_right'
        sensors.tick(DroneState(0, 0, 0), .1)
        assert right.getSeq() > 0
        assert np.all(right[0] == [13, 34, 56])
    finally:
        right.close(); sensors.close()


# ---------------------------------------------------------------------------
# Compact and bulk sample transport
# ---------------------------------------------------------------------------

def _sensing_profile(scan_hz=10., sonar_hz=15.):
    d = camera_profile(16, 12, physics_hz=30.)
    d['sensors'].extend([
        dict(id='scan', type='lidar.scan2d', parent='body', rate_hz=scan_hz,
             pose_parent=dict(z_m=.05), model=dict(samples=720, max_range_m=25.)),
        dict(id='sonar', type='range.ultrasonic', parent='body', rate_hz=sonar_hz,
             pose_parent=dict(pitch_deg=-90.)),
    ])
    return DroneProfile.parse(d)


def test_compact_and_array_records_carry_their_declared_schemas():
    sensors = manager(_sensing_profile(scan_hz=30., sonar_hz=30.))
    samples = SensorSamples(sensors.publisher.instance)
    try:
        assert samples.connect()
        sensors.tick(DroneState(3, 3, 2), .1)
        records = {r['sensor_id']: r for r in samples.drain()}
        assert records['front']['payload_type'] == CAMERA_FRAME
        assert records['sonar']['payload_type'] == RANGE_SAMPLE
        assert records['scan']['payload_type'] == LIDAR_FRAME

        sonar = records['sonar']['payload']
        assert sonar['schema'] == 'range.sample.v1' and sonar['reducer'] == 'nearest'
        assert sonar['range_m'] == pytest.approx(2.0, abs=.2)

        lidar = records['scan']['payload']
        assert lidar['array_sequence'] == records['scan']['sequence']
        assert lidar['calibration']['angle_increment_deg'] == pytest.approx(.5)

        arrays = samples.drain_array('scan')
        assert len(arrays) == 1 and arrays[0]['payload_type'] == PACKED_ARRAY
        assert arrays[0]['sequence'] == records['scan']['sequence']
        assert arrays[0]['fields']['range_m'].shape == (720,)
        assert arrays[0]['fields']['confidence'].dtype == np.uint8
    finally:
        samples.close(); sensors.close()


def test_an_invalid_reading_is_null_with_an_invalid_status():
    d = camera_profile(16, 12, physics_hz=30.)
    d['sensors'].append(dict(id='sky', type='range.laser', parent='body', rate_hz=30.,
                             pose_parent=dict(pitch_deg=90.), model=dict(max_range_m=5.)))
    sensors = manager(DroneProfile.parse(d))
    samples = SensorSamples(sensors.publisher.instance)
    try:
        assert samples.connect()
        sensors.tick(DroneState(3, 3, 2), .1)
        record = next(r for r in samples.drain() if r['sensor_id'] == 'sky')
        assert record['status'] == 0
        assert record['payload']['range_m'] is None
        assert record['payload']['returns'] == 0
        assert 'NaN' not in json.dumps(record['payload'])
    finally:
        samples.close(); sensors.close()


def test_bulk_lidar_cannot_evict_compact_records():
    """Array traffic has its own ring, so it never competes for compact slots."""
    sensors = manager(_sensing_profile(scan_hz=30., sonar_hz=30.))
    samples = SensorSamples(sensors.publisher.instance)
    try:
        assert samples.connect()
        manifest = sensors.manifest
        assert manifest['sensors']['scan']['channel'] != manifest['sample_channel']
        assert manifest['sensors']['sonar']['channel'] == manifest['sample_channel']
        # Two simulated seconds against rings sized for one second of retention.
        for tick in range(60):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        records = samples.drain()
        assert samples.overruns == 0
        assert sum(r['sensor_id'] == 'sonar' for r in records) == 60
        assert sum(r['sensor_id'] == 'scan' for r in records) == 60
        # The array ring is the one that had to drop, and its per-sensor
        # sequence gaps say exactly how much.
        assert len(samples.drain_array('scan')) < 60
        assert samples.skipped['scan'] > 0
        assert samples.skipped['sonar'] == 0
    finally:
        samples.close(); sensors.close()


def test_a_new_generation_reopens_every_channel():
    first = _sensing_profile()
    sensors = manager(first)
    samples = SensorSamples(sensors.publisher.instance)
    try:
        assert samples.connect()
        before = dict(sample=samples.manifest['sample_channel'],
                      scan=samples.manifest['sensors']['scan']['channel'])
        reduced = DroneProfile.parse({**first.data, 'sensors': [
            s for s in first.data['sensors'] if s['id'] != 'sonar']})
        sensors.apply(reduced)
        samples.last_probe = -1e9
        assert samples.connect()
        assert samples.identity[1] == 2
        assert samples.manifest['sample_channel'] != before['sample']
        assert samples.manifest['sensors']['scan']['channel'] != before['scan']
        assert 'sonar' not in samples.manifest['sensors']
    finally:
        samples.close(); sensors.close()


def test_a_disabled_sensor_leaves_no_manifest_entry_and_no_channel():
    d = _sensing_profile().data
    for sensor in d['sensors']:
        if sensor['id'] == 'scan': sensor['enabled'] = False
    sensors = manager(DroneProfile.parse(d))
    try:
        assert 'scan' not in sensors.manifest['sensors']
        assert 'scan' not in sensors.publisher.channels.arrays
        assert sensors.production().keys() == {'front', 'sonar'}
    finally:
        sensors.close()


def test_metadata_that_never_matches_an_image_is_dropped_and_counted():
    """A consumer must not hold pose records for frames it never took."""
    sensors = manager(_sensing_profile(scan_hz=30., sonar_hz=30.))
    client = SensorVideo(sensors.publisher.instance)
    try:
        assert client.connect()
        # Publish a second of captures without the consumer ever looking, so
        # every image but the newest has gone past by the time it does.
        for tick in range(40):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        assert client.getSeq() > 0
        assert client.association_drops > 0
        # Both caches are bounded rather than emptied: about a second of
        # retention each, so unmatched entries cannot accumulate for a run.
        limit = math.ceil(client.getFps()) + 2
        assert len(client.metadata) <= limit and len(client.pending) <= limit
        # And the stream keeps working afterwards.
        before = client.getSeq()
        sensors.tick(DroneState(3, 3, 2), 40 / 30)
        assert client.getSeq() > before
    finally:
        client.close(); sensors.close()


def test_the_shared_ring_carries_every_compact_sensor_at_once():
    """State and range records sharing one ring keep their own sequences."""
    d = camera_profile(16, 12, physics_hz=30.)
    d['sensors'].extend([
        dict(id='gnss', type='position.gnss', parent='body', rate_hz=10.),
        dict(id='imu', type='motion.imu', parent='body', rate_hz=30.),
        dict(id='baro', type='altimeter.barometric', parent='body', rate_hz=15.),
        dict(id='ambient', type='environment.temperature', parent='body', rate_hz=1.),
        dict(id='sonar', type='range.ultrasonic', parent='body', rate_hz=10.,
             pose_parent=dict(pitch_deg=-90.)),
        dict(id='scan', type='lidar.scan2d', parent='body', rate_hz=30.,
             model=dict(samples=720, max_range_m=25.)),
    ])
    sensors = manager(DroneProfile.parse(d), vehicle=_Vehicle())
    samples = SensorSamples(sensors.publisher.instance)
    try:
        assert samples.connect()
        for tick in range(30):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        by_sensor = {}
        for record in samples.drain():
            by_sensor.setdefault(record['sensor_id'], []).append(record)
        assert {sid: len(rows) for sid, rows in by_sensor.items()} == {
            'front': 30, 'gnss': 10, 'imu': 30, 'baro': 15, 'ambient': 1,
            'sonar': 10, 'scan': 30}
        for sid, rows in by_sensor.items():
            assert [r['sequence'] for r in rows] == list(range(1, len(rows) + 1)), sid
            assert all(r['sim_time_us'] >= 0 for r in rows)
            assert rows[-1]['sim_time_us'] > rows[0]['sim_time_us'] or len(rows) == 1
        # Bulk LiDAR shares none of that capacity, and nothing was evicted.
        assert samples.overruns == 0 and set(samples.skipped.values()) == {0}
    finally:
        samples.close(); sensors.close()


def test_a_drone_reset_keeps_the_profile_and_restarts_the_cadence():
    profile = _sensing_profile()
    sensors = manager(profile)
    try:
        for tick in range(9):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        assert sensors.tick_index == 9
        generation, digest = sensors.generation, sensors.profile.digest
        sequences = dict(sensors.publisher.sequences)

        sensors.reset()

        assert sensors.profile.digest == digest, "a reset is not a profile change"
        assert sensors.generation == generation
        assert sensors.publisher.reset_epoch == 1
        assert sensors.publisher.sequences == sequences, "sequences are transport"
        assert sensors.tick_index == 0, "cadence restarts"
        sensors.tick(DroneState(3, 3, 2), 1.0)
        assert sensors.publisher.sequences['front'] == sequences['front'] + 1
    finally:
        sensors.close()


def test_each_sensor_publishes_at_its_own_configured_cadence():
    """Rates divide the physics step, so a capture always lands on a snapshot."""
    sensors = manager(_sensing_profile(scan_hz=10., sonar_hz=15.))
    samples = SensorSamples(sensors.publisher.instance)
    try:
        assert samples.connect()
        for tick in range(30):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        counts = {}
        for record in samples.drain():
            counts[record['sensor_id']] = counts.get(record['sensor_id'], 0) + 1
        assert counts == {'front': 30, 'scan': 10, 'sonar': 15}
        production = sensors.production()
        assert production['scan']['configured_hz'] == 10.
        assert production['scan']['published'] == 10
    finally:
        samples.close(); sensors.close()


def test_apply_releases_the_render_targets_of_removed_cameras():
    """A generation that drops a camera must drop its buffer with it."""
    class Tracking(Renderer):
        def __init__(self): super().__init__(); self.dropped = []
        def drop_views(self, camera_ids): self.dropped.extend(sorted(camera_ids))

    d = camera_profile(16, 12, physics_hz=30.)
    d['sensors'] = stereo_pair('nav', pose=dict(z_m=.1), model=dict(
        width_px=16, height_px=12, fov_h_deg=70., near_m=.15, far_m=150.))
    d['primary_camera'] = 'nav_left'
    renderer = Tracking()
    sensors = manager(DroneProfile.parse(d), renderer=renderer)
    try:
        sensors.tick(DroneState(0, 0, 0), .1)
        single = camera_profile(16, 12, physics_hz=30.)
        single['sensors'][0]['id'] = single['primary_camera'] = 'nav_left'
        sensors.apply(DroneProfile.parse(single))
        assert renderer.dropped == ['nav_right']
        assert set(sensors.manifest['sensors']) == {'nav_left'}
        assert sensors.tick_index == 0 and sensors.capture_id == 0
        # The surviving camera keeps working, on the new generation.
        sensors.tick(DroneState(0, 0, 0), .2)
        assert sensors.publisher.sequences['nav_left'] == 1
    finally:
        sensors.close()


def test_an_invalid_profile_never_disturbs_the_running_generation():
    sensors = manager(_sensing_profile())
    try:
        before = sensors.manifest['sample_channel']
        broken = _sensing_profile().data
        broken['sensors'].append(dict(broken['sensors'][0], id='clash'))
        broken['sensors'][-1]['id'] = 'front'
        with pytest.raises(ValueError, match='duplicate'):
            sensors.apply(DroneProfile.parse(broken))
        assert sensors.generation == 1
        assert sensors.manifest['sample_channel'] == before
        sensors.tick(DroneState(3, 3, 2), .1)
    finally:
        sensors.close()


def test_transport_retention_sizes_video_arrays_and_compact_history():
    draft = _sensing_profile().data
    original = DroneProfile.parse(draft)
    draft['transport'] = dict(retention_s=2, memory_limit_mib=128)
    longer = DroneProfile.parse(draft)
    assert longer.memory_limit_bytes == 128 * 1048576
    assert longer.plan['sample_capacity'] > original.plan['sample_capacity']
    for sensor in original.plan['cameras']:
        assert longer.plan['cameras'][sensor]['slots'] > original.plan['cameras'][sensor]['slots']
    for sensor in original.plan['arrays']:
        assert longer.plan['arrays'][sensor]['capacity'] > original.plan['arrays'][sensor]['capacity']
    draft['transport']['memory_limit_mib'] = .01
    with pytest.raises(ValueError, match='memory budget'):
        DroneProfile.parse(draft)


def test_reset_invalidates_camera_history_and_reports_copy_memory(profile):
    sensors = manager(profile)
    client = SensorVideo(sensors.publisher.instance)
    try:
        assert client.connect()
        sensors.tick(DroneState(0, 0, 1), .1)
        assert client.getSeq() > 0
        revision = client.getSessionId()
        assert client.observation()['cache_bytes'] >= 16 * 12 * 3
        sensors.reset()
        sensors.tick(DroneState(0, 0, 1), .2)
        assert client.getSeq() > 0
        assert client.getSessionId() > revision
        assert client.subscription()['reset_epoch'] == 1
    finally:
        client.close()
        sensors.close()
