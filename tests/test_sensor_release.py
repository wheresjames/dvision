"""Release verification: the budget the reference profile has to meet, and the
failures that must be visible without corrupting the running generation.

The performance gate is a live process at real time, because the thing being
measured is whether the simulator holds its pace against a wall clock while
producing every sample it promised. It runs a small deterministic profile in
the normal suite; the committed reference profile at full resolution is the
nightly benchmark beside it.
"""

from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace

import pytest

from dcmn.sensors import CAMERA_FRAME, RecordRing, SensorVideo
from dsim.backend import SimulatorBackend
from dsim.dsim import DroneSimulator, DroneState
from dsim.profiles import (MAX_COMPONENTS, MAX_ID_BYTES, DroneProfile,
                           camera_profile, default_profile)
from dsim.sensor_manager import SensorManager
from dtest.artifacts import artifact_directory
from dtest.process_harness import DsimProcessHarness

#: The gate from DV-SENSORS: pace, publisher overruns, and delivered samples.
MIN_SPEED_RATIO = 0.90
MIN_SAMPLE_RATIO = 0.90


def reference_profile(*, width=160, height=120, samples=180, image=(32, 24)):
    """The reference vehicle's shape at a size CI can afford.

    Every sensor type, the same mount tree and the same rates as the committed
    reference; only the pixel and ray counts are reduced. Shrinking those is
    what the design says to do when the budget is missed, so measuring a
    reduced one is measuring the same thing.
    """
    lens = dict(width_px=width, height_px=height, fov_h_deg=70., near_m=.15, far_m=150.)
    return DroneProfile.parse(dict(
        camera_profile(width, height),
        name="reference-ci", primary_camera="nav_left",
        mounts=[dict(id="nav_ptz", type="mount.ptz", parent="body",
                     pose_parent=dict(x_m=.08, z_m=.1),
                     state=dict(tilt_deg=-5.))],
        sensors=[
            dict(id="nav_left", type="camera.rgb", rate_hz=30., parent="nav_ptz",
                 pose_parent=dict(y_m=-.06), model=dict(lens), sync_group="nav_stereo"),
            dict(id="nav_right", type="camera.rgb", rate_hz=30., parent="nav_ptz",
                 pose_parent=dict(y_m=.06), model=dict(lens), sync_group="nav_stereo"),
            dict(id="scan", type="lidar.scan2d", rate_hz=10., parent="body",
                 pose_parent=dict(z_m=.05), model=dict(samples=samples, max_range_m=25.)),
            dict(id="flash", type="lidar.range_image", rate_hz=10., parent="body",
                 pose_parent=dict(x_m=.05, pitch_deg=-5.),
                 model=dict(width_px=image[0], height_px=image[1], fov_h_deg=70.)),
            dict(id="down_sonar", type="range.ultrasonic", rate_hz=20., parent="body",
                 pose_parent=dict(z_m=-.03, pitch_deg=-90.)),
            dict(id="front_ir", type="range.infrared", rate_hz=20., parent="body",
                 pose_parent=dict(x_m=.12)),
            dict(id="front_tof", type="range.laser", rate_hz=25., parent="nav_ptz",
                 pose_parent=dict(x_m=.02, z_m=.03)),
            dict(id="gnss", type="position.gnss", rate_hz=10., parent="body"),
            dict(id="imu", type="motion.imu", rate_hz=100., parent="body"),
            dict(id="baro", type="altimeter.barometric", rate_hz=25., parent="body"),
            dict(id="compass", type="heading.magnetometer", rate_hz=25., parent="body"),
            dict(id="ambient", type="environment.temperature", rate_hz=1., parent="body"),
        ]))


def flown(tmp_path, profile, *, seconds, name):
    """Run a real simulator at real time and return its run summary."""
    artifacts = artifact_directory(tmp_path, name)
    harness = DsimProcessHarness(artifacts, drone_profile=profile,
                                 frames=int(seconds * 30))
    with harness:
        harness.arm()
        harness.wait_status(lambda s: s.get("drone.armed") == "1",
                            description="armed for the reference run")
        harness.send_body_velocity(0.6, 0.0, 0.0, 12.0)
        # --frames ends the run on its own; waiting for the exit is the point,
        # so the harness's "the simulator died" guard must not fire on it.
        assert harness.process.wait(timeout=seconds * 6 + 30.0) == 0
    return json.loads((harness.report_dir / "dsim" / "summary.json").read_text())


def assert_within_budget(summary, seconds):
    speed = summary["health"]["achieved_speed"]
    assert speed["mean"] is not None, "the run produced no health samples"
    assert speed["mean"] >= MIN_SPEED_RATIO, (
        f"simulated time advanced at {speed['mean']:.3f}x of real time; "
        f"the gate is {MIN_SPEED_RATIO}x")
    for sensor_id, record in summary["sensors"].items():
        assert record["drops"] == 0, f"{sensor_id}: {record['drops']} publisher drops"
        expected = record["configured_hz"] * summary["sim_time_s"]
        assert record["published"] >= MIN_SAMPLE_RATIO * expected, (
            f"{sensor_id}: published {record['published']} of about "
            f"{expected:.0f} configured samples")
    return summary


def test_the_reference_profile_holds_real_time_and_produces_every_sample(tmp_path):
    """Every sensor type at once, at real time, inside the documented budget."""
    seconds = 4.0
    summary = flown(tmp_path, reference_profile(), seconds=seconds, name="reference-ci")
    assert set(summary["sensors"]) == {
        "nav_left", "nav_right", "scan", "flash", "down_sonar", "front_ir",
        "front_tof", "gnss", "imu", "baro", "compass", "ambient"}
    assert_within_budget(summary, seconds)
    assert summary["sensor_plan"]["total_bytes"] < 256 * 1024 * 1024
    assert summary["sensor_generation"] == 1


@pytest.mark.nightly
@pytest.mark.skipif(os.environ.get("DVISION_NIGHTLY") != "1",
                    reason="set DVISION_NIGHTLY=1 to run the reference benchmark")
def test_the_committed_reference_profile_meets_the_budget(tmp_path):
    """The documented 60-second run, at the resolutions the profile ships with."""
    seconds = float(os.environ.get("DVISION_REFERENCE_SECONDS", "60"))
    summary = flown(tmp_path, DroneProfile.load("stereo-nav-and-proximity"),
                    seconds=seconds, name="reference-full")
    assert_within_budget(summary, seconds)


# ---------------------------------------------------------------------------
# Failures that must be visible without corrupting the running generation
# ---------------------------------------------------------------------------

class Renderer:
    def render_views(self, requests):
        for _id, _model, _pose, frame in requests:
            frame[:] = 5
    def drop_views(self, camera_ids): pass


def manager(profile):
    return SensorManager("sensor-release-" + uuid.uuid4().hex[:8], profile,
                         SimulatorBackend(SimpleNamespace(objects=[]), Renderer()))


@pytest.mark.parametrize("mutate,field", [
    (lambda d: d.update(name=""), "name"),
    (lambda d: d.update(schema="dvision2.drone-profile.v99"), "schema"),
    (lambda d: d.update(sensors="not a list"), "mounts/sensors"),
    (lambda d: d["sensors"].extend(
        dict(d["sensors"][0], id=f"extra{n}") for n in range(MAX_COMPONENTS)),
     "maximum 64"),
    (lambda d: d["sensors"][0].update(id="x" * (MAX_ID_BYTES + 1)), "48 bytes"),
    (lambda d: d["sensors"][0]["model"].update(width_px=4000, height_px=4000),
     "memory budget"),
    (lambda d: d["sensors"][0].update(pose_parent=dict(elevation_m=1.0)),
     "pose_parent"),
])
def test_a_malformed_profile_is_rejected_by_the_field_that_is_wrong(mutate, field):
    draft = camera_profile()
    mutate(draft)
    with pytest.raises(ValueError, match=field):
        DroneProfile.parse(draft)


def test_a_profile_larger_than_the_manifest_allows_is_rejected(tmp_path):
    draft = camera_profile()
    draft["name"] = "n" * 70000
    with pytest.raises(ValueError, match="64 KiB"):
        DroneProfile.parse(draft)
    path = tmp_path / "huge.json"
    path.write_text(json.dumps(draft))
    with pytest.raises(ValueError, match="64 KiB"):
        DroneProfile.load(path)


def test_an_allocation_failure_leaves_the_running_generation_untouched(monkeypatch):
    """Construction is staged, so a channel that cannot be created costs nothing."""
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    client = SensorVideo(sensors.publisher.instance)
    try:
        assert client.connect()
        sensors.tick(DroneState(0, 0, 1), 0.1)
        assert client.getSeq() > 0
        before = sensors.manifest["sample_channel"]

        original = RecordRing.__init__

        def refuse(self, name, size, *, create=False):
            if create:
                raise RuntimeError("shared memory exhausted")
            original(self, name, size, create=create)

        monkeypatch.setattr(RecordRing, "__init__", refuse)
        with pytest.raises(RuntimeError, match="exhausted"):
            sensors.apply(DroneProfile.parse(camera_profile(24, 16, physics_hz=30.)))
        monkeypatch.undo()

        assert sensors.generation == 1
        assert sensors.manifest["sample_channel"] == before
        sensors.tick(DroneState(0, 0, 1), 0.2)
        assert client.getSeq() > 0, "the surviving generation still publishes"
    finally:
        client.close()
        sensors.close()


def test_a_restarted_consumer_rediscovers_without_help():
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    first = SensorVideo(sensors.publisher.instance)
    try:
        assert first.connect()
        sensors.tick(DroneState(0, 0, 1), 0.1)
        assert first.getSeq() > 0
        first.close()

        second = SensorVideo(sensors.publisher.instance)
        try:
            assert second.connect()
            sensors.tick(DroneState(0, 0, 1), 0.2)
            assert second.getSeq() > 0
            assert second.identity == (sensors.publisher.session, 1)
        finally:
            second.close()
    finally:
        first.close()
        sensors.close()


def test_a_consumer_on_an_old_generation_reopens_rather_than_serving_stale_frames():
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    client = SensorVideo(sensors.publisher.instance)
    try:
        assert client.connect()
        sensors.tick(DroneState(0, 0, 1), 0.1)
        stale = client.getSeq()
        assert stale > 0

        sensors.apply(DroneProfile.parse(camera_profile(24, 16, physics_hz=30.)))
        client.last_probe = -1e9
        assert client.connect()
        assert client.identity[1] == 2
        assert client.getSeq() == 0, "no frame of the new generation has arrived yet"
        assert client.frame is None
        sensors.tick(DroneState(0, 0, 1), 0.2)
        assert client.getSeq() > 0
        assert client[0].shape == (16, 24, 3), "the new generation's dimensions"
    finally:
        client.close()
        sensors.close()


def test_removing_a_sensor_closes_its_publisher_and_its_manifest_entry():
    draft = camera_profile(16, 12, physics_hz=30.)
    draft["sensors"].append(dict(id="scan", type="lidar.scan2d", parent="body",
                                 rate_hz=10., model=dict(samples=90)))
    sensors = manager(DroneProfile.parse(draft))
    try:
        assert "scan" in sensors.publisher.channels.arrays
        sensors.apply(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
        assert "scan" not in sensors.manifest["sensors"]
        assert sensors.publisher.channels.arrays == {}
        assert "scan" not in sensors.production()
    finally:
        sensors.close()


def test_an_armed_apply_is_refused_before_anything_is_staged():
    """The gate itself, not just the button that leads to it: an armed
    vehicle answers no, and the refusal comes before a channel moves."""
    sim = DroneSimulator.__new__(DroneSimulator)
    sim.state = DroneState(0, 0, 1, armed=True)
    with pytest.raises(ValueError, match='Disarm'):
        sim.apply_profile(DroneProfile.parse(camera_profile(24, 16)))


def test_a_compact_record_beyond_the_ring_slot_is_refused_rather_than_truncated():
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    try:
        with pytest.raises(ValueError, match='front: compact record exceeds'):
            sensors.publisher.write_compact('front', 1, 1, 0, CAMERA_FRAME,
                                            dict(blob='x' * 9000))
        assert sensors.production()['front']['drops'] == 0, \
            "a caller's bad payload is not the sensor's fault"
    finally:
        sensors.close()


def test_the_default_profile_stays_inside_every_documented_limit():
    profile = DroneProfile.parse(default_profile())
    plan = profile.plan
    data = profile.data
    assert len(data["mounts"]) + len(data["sensors"]) <= MAX_COMPONENTS
    assert all(len(s["id"].encode()) <= MAX_ID_BYTES for s in data["sensors"])
    assert len(profile.encoded.encode()) <= 65536
    assert plan["total_bytes"] <= 256 * 1024 * 1024
    # Every committed profile has to load and resolve to itself.
    for name in ("default", "fast-sweep", "stereo-nav-and-proximity"):
        loaded = DroneProfile.load(name)
        assert DroneProfile.parse(loaded.data).digest == loaded.digest


@pytest.mark.parametrize('value', [None, [], 'bad', 12])
def test_ptz_objects_fail_with_a_field_specific_error(value):
    draft = camera_profile(16, 12)
    draft['mounts'] = [dict(id='head', type='mount.ptz', parent='body', state=value)]
    with pytest.raises(ValueError, match='head.state/limits'):
        DroneProfile.parse(draft)


def test_a_removed_selected_camera_cannot_return_its_last_frame():
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(draft['sensors'][0], id='aux'))
    sensors = manager(DroneProfile.parse(draft))
    client = SensorVideo(sensors.publisher.instance, 'aux')
    try:
        assert client.connect()
        sensors.tick(DroneState(0, 0, 1), .1)
        assert client.getSeq() > 0
        sensors.apply(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
        client.last_probe = -1e9
        assert not client.connect()
        assert client.getSeq() == 0
        assert client.frame is None and client.record is None
        assert client.video is None and client.records is None
    finally:
        client.close()
        sensors.close()


def test_renderer_allocation_failure_precedes_registry_commit():
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    try:
        before = sensors.manifest
        def refuse(profile):
            raise RuntimeError('camera allocation refused')
        sensors.backend.renderer.prepare_profile = refuse
        with pytest.raises(RuntimeError, match='camera allocation refused'):
            sensors.apply(DroneProfile.parse(camera_profile(24, 16, physics_hz=30.)))
        assert sensors.generation == 1
        assert sensors.manifest == before
        sensors.tick(DroneState(0, 0, 1), .1)
        assert sensors.production()['front']['published'] == 1
    finally:
        sensors.close()
