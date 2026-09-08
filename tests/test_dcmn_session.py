"""Shared discovery, one compact decode, bounded caching and attach baselines."""
import time

import pytest

from dcmn.sensors import SensorSession
from dsim.dsim import DroneState
from dsim.profiles import DroneProfile, camera_profile
from tests.test_sensor_contract import manager, _Vehicle


def numeric_profile():
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(id='beam', type='environment.temperature', parent='body', rate_hz=30.))
    return DroneProfile.parse(draft)


@pytest.fixture
def vehicle():
    sensors = manager(numeric_profile(), vehicle=_Vehicle())
    try: yield sensors
    finally: sensors.close()


def test_single_decode_and_refcount(vehicle, monkeypatch):
    import dcmn.sensors as wire
    session = SensorSession(vehicle.publisher.instance)
    try:
        one = session.open('beam'); two = session.open('beam')
        assert one is two and session.refs['beam'] == 2
        calls = []
        original = wire.decode_record
        def decode(raw):
            calls.append(1); return original(raw)
        monkeypatch.setattr(wire, 'decode_record', decode)
        for i in range(10): vehicle.tick(DroneState(0, 0, 1), i/30)
        session.poll()
        assert len(calls) == session.decoded_records == 20
        assert len(one.history) == 10
        session.release('beam'); assert session.latest('beam') is not None
        session.release('beam'); assert not session.streams and session.cache_bytes == 0
    finally: session.close()


def test_late_attach_has_explicit_baseline_and_healthy_rate(vehicle):
    session = SensorSession(vehicle.publisher.instance)
    try:
        session.connect()
        # A subscriber that has drained ten minutes of earlier compact data.
        vehicle.publisher.sequences['beam'] = 18000
        session.poll()
        stream = session.open('beam', accounting='from_attach')
        for i in range(31):
            vehicle.tick(DroneState(0, 0, 1), 600+i/30); session.poll()
        assert stream.attached_at == 18001 and stream.skipped == 0
        assert session.report()['devices']['beam']['grade'] == 'ok'
        assert not session.subscriptions()[0]['required']
        session.release('beam')
        stream = session.open('beam', required=True)
        vehicle.tick(DroneState(0, 0, 1), 602); session.poll()
        assert stream.skipped == 18031
    finally: session.close()


def test_oldest_first_global_eviction_and_reset(vehicle):
    session = SensorSession(vehicle.publisher.instance, cache_bytes=1800)
    try:
        stream = session.open('beam')
        for i in range(20):
            vehicle.tick(DroneState(0, 0, 1), i/30); session.poll()
            assert session.cache_bytes <= session.cache_limit
        assert session.cache_drops > 0
        assert session.latest('beam').sequence == 20
        assert next(iter(stream.history)) > 1
        vehicle.reset(); vehicle.tick(DroneState(0, 0, 1), 1); session.poll()
        assert len(stream.history) == 1
        assert stream.reset_epoch == 1
    finally: session.close()


def test_generation_reattaches_same_stream_and_closes_handles():
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    session = SensorSession(sensors.publisher.instance)
    try:
        stream = session.open('front')
        sensors.tick(DroneState(0, 0, 1), .1); session.poll(); stream.refresh()
        assert stream.latest is not None
        previous = stream.video
        sensors.apply(sensors.profile)
        session.probe.last_probe = -1e9
        assert session.connect()
        assert session.streams['front'] is stream
        assert stream.video is not previous and stream.latest is None
        assert stream.identity[1] == 2
        sensors.tick(DroneState(0, 0, 1), .2); session.poll(); stream.refresh()
        assert stream.record['generation'] == 2
    finally: session.close(); sensors.close()


def test_poll_does_no_bulk_work_and_reference_budget(monkeypatch):
    from pathlib import Path
    profile = DroneProfile.load(Path('assets/drone_profiles/stereo-nav-and-proximity.json'))
    sensors = manager(profile, vehicle=_Vehicle())
    session = SensorSession(sensors.publisher.instance)
    try:
        for sid in session.probe.probe()['sensors']:
            session.open(sid, accounting='from_attach')
        for stream in session.streams.values():
            monkeypatch.setattr(stream, 'refresh', lambda: pytest.fail('poll read bulk data'))
        durations = []
        for i in range(90):
            for j in range(10):
                sensors.tick(DroneState(0, 0, 1), (i*10+j)/300)
            start = time.perf_counter(); session.poll()
            durations.append(time.perf_counter()-start)
        assert sum(durations)/len(durations) < .002
        assert session.decoded_records > 0
    finally: session.close(); sensors.close()


def test_open_after_close_refuses_instead_of_leaking_handles(vehicle):
    session = SensorSession(vehicle.publisher.instance)
    session.close()
    with pytest.raises(RuntimeError):
        session.open('beam')


def test_first_record_is_ungraded_not_a_zero_hz_flash(vehicle):
    session = SensorSession(vehicle.publisher.instance)
    try:
        session.connect()
        vehicle.tick(DroneState(0, 0, 1), 0.); session.poll()
        first = session.report()['devices']['beam']
        assert first['achieved_hz'] is None and first['grade'] == 'unknown'
        vehicle.tick(DroneState(0, 0, 1), 1/30); session.poll()
        assert session.report()['devices']['beam']['grade'] == 'ok'
    finally: session.close()


def test_released_stream_cannot_reopen_and_oversize_sample_is_evicted(vehicle):
    session = SensorSession(vehicle.publisher.instance, cache_bytes=1)
    try:
        stream = session.open('beam')
        vehicle.tick(DroneState(0, 0, 1), .1); session.poll()
        assert session.cache_bytes == stream.cache_bytes == 0
        assert session.latest('beam') is None and session.cache_drops == 1
        session.release('beam'); stream.refresh()
        assert session.cache_bytes == 0
    finally: session.close()
