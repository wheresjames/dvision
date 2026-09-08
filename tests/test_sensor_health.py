"""Per-sensor health: what dsim produced, what each module took in, and the
one grade an operator reads.

The rule the whole design turns on is that production and intake are separate
measurements. A producer publishing perfectly is exactly what a stalled reader
looks like from the other side, so neither may be derived from the other, and
the summary grade must never let a healthy stream hide a failed one.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from dcmn import health
from dcmn.health import SensorIntake
from dcmn.module_bus import (SENSOR_HEALTH_EVENT, ModuleEvent, PipelineView)
from dsim.dsim import DroneState
from dsim.health import SimulationHealth, required_sensor_grade
from dsim.profiles import DroneProfile, camera_profile, stereo_pair
from dsim.sensor_manager import SensorManager


# ---------------------------------------------------------------------------
# Consumer intake
# ---------------------------------------------------------------------------

def subscription(sensor_id="front", *, expected_hz=30.0, required=True,
                 sync_group=None, generation=1):
    return dict(sensor_id=sensor_id, expected_hz=expected_hz, required=required,
                sync_group=sync_group, generation=generation)


def feed(intake, sensor_id, sequences, *, start=0.0, period=1 / 30, capture=None):
    """Deliver a run of samples on their own cadence."""
    for sequence in sequences:
        intake.observe(dict(sensor_id=sensor_id, sequence=sequence,
                            sim_time_s=start + sequence * period,
                            generation=1, capture_id=capture or sequence))
    return start + max(sequences) * period


def test_an_input_is_starting_before_its_first_period_has_passed():
    intake = SensorIntake()
    intake.follow([subscription()])
    assert intake.report(0.0)["front"]["state"] == health.STARTING
    assert intake.grade() == health.UNKNOWN, "a starting input degrades nothing"


def test_a_stream_at_its_configured_rate_is_healthy():
    intake = SensorIntake()
    intake.follow([subscription()])
    intake.report(0.0)
    end = feed(intake, "front", range(1, 31))
    record = intake.report(end)["front"]
    assert record["observed_hz"] == pytest.approx(30.0, rel=0.05)
    assert record["state"] == health.OK
    assert record["skipped"] == 0 and record["last_sequence"] == 30


def test_gaps_in_the_sequence_are_the_samples_this_module_never_saw():
    intake = SensorIntake()
    intake.follow([subscription()])
    intake.report(0.0)
    end = feed(intake, "front", [1, 2, 5, 6])
    record = intake.report(end)["front"]
    assert record["skipped"] == 2
    # And it is not the transport's overrun count, which is a different thing.
    assert record["overruns"] == 0


def test_a_repeated_sample_is_not_a_new_one():
    """Consumers re-read the same matched frame between publications."""
    intake = SensorIntake()
    intake.follow([subscription()])
    intake.report(0.0)
    for _ in range(20):
        intake.observe(dict(sensor_id="front", sequence=7, sim_time_s=0.25,
                            generation=1, capture_id=7))
    record = intake.report(1.0)["front"]
    assert record["last_sequence"] == 7
    assert record["observed_hz"] == pytest.approx(1.0)


def test_a_stopped_stream_fails_even_though_its_last_window_looked_full():
    intake = SensorIntake()
    intake.follow([subscription()])
    intake.report(0.0)
    end = feed(intake, "front", range(1, 31))
    assert intake.report(end)["front"]["state"] == health.OK
    # Nothing more arrives. The rate window empties, and lateness alone would
    # already have caught it.
    assert intake.report(end + 1.0)["front"]["state"] == health.BAD
    assert intake.grade() == health.BAD


def test_an_optional_input_never_degrades_the_module():
    intake = SensorIntake()
    intake.follow([subscription("aux", required=False)])
    intake.report(0.0)
    assert intake.report(5.0)["aux"]["state"] == health.BAD
    assert intake.grade() == health.UNKNOWN


def test_a_diverged_synchronized_pair_is_unhealthy_at_full_rate():
    """The failure a per-module frame rate cannot see."""
    intake = SensorIntake()
    intake.follow([subscription("left", sync_group="nav"),
                   subscription("right", sync_group="nav")])
    intake.report(0.0)
    end = 0.0
    for sequence in range(1, 31):
        end = feed(intake, "left", [sequence], capture=sequence)
        feed(intake, "right", [sequence], capture=sequence)
    records = intake.report(end)
    assert records["left"]["sync"] == "ok" and records["left"]["state"] == health.OK

    # Both members keep arriving at full rate, but from different captures.
    for sequence in range(31, 61):
        end = feed(intake, "left", [sequence], capture=sequence)
        feed(intake, "right", [sequence], capture=sequence + 1)
    records = intake.report(end)
    assert records["left"]["observed_hz"] == pytest.approx(30.0, rel=0.05)
    assert records["left"]["sync"] == "diverged"
    assert records["left"]["state"] == health.BAD and intake.grade() == health.BAD


def test_a_new_generation_restarts_the_window_rather_than_reporting_a_gap():
    intake = SensorIntake()
    intake.follow([subscription()])
    intake.report(0.0)
    end = feed(intake, "front", range(1, 31))
    intake.report(end)
    intake.follow([subscription(generation=2)])
    record = intake.report(end)["front"]
    assert record["generation"] == 2
    assert record["last_sequence"] is None and record["skipped"] == 0
    assert record["state"] == health.STARTING


def test_a_sensor_dropped_from_a_new_generation_leaves_the_report():
    intake = SensorIntake()
    intake.follow([subscription("front"), subscription("aux")])
    intake.follow([subscription("front")])
    assert set(intake.report(0.0)) == {"front"}


def test_track_declares_and_samples_one_handle():
    """The call every consumer makes where it polls for frames."""
    class Handle:
        def __init__(self): self.sequence = 0
        def subscription(self, required=True):
            return subscription(required=required)
        def observation(self):
            self.sequence += 1
            return dict(sensor_id="front", sequence=self.sequence,
                        sim_time_s=self.sequence / 30, generation=1,
                        capture_id=self.sequence, overruns=0, drops=2)

    intake, handle = SensorIntake(), Handle()
    intake.track(handle)
    intake.report(0.0)
    for _ in range(30):
        intake.track(handle)
    record = intake.report(31 / 30)["front"]
    assert record["state"] == health.OK and record["drops"] == 2
    intake.track(None)  # a consumer whose handle is not open yet
    assert set(intake.report(2.0)) == set()


# ---------------------------------------------------------------------------
# Aggregation in dsim
# ---------------------------------------------------------------------------

def test_the_module_grade_is_the_worst_required_input():
    inputs = {"left": {"required": True, "state": health.OK},
              "right": {"required": True, "state": health.BAD},
              "aux": {"required": False, "state": health.BAD}}
    assert required_sensor_grade(inputs) == health.BAD
    assert required_sensor_grade({k: v for k, v in inputs.items() if k != "right"}) \
        == health.OK
    assert required_sensor_grade({}) == health.UNKNOWN
    assert required_sensor_grade(None) == health.UNKNOWN


def event(kind, payload, process_id="p1", role="controller"):
    return ModuleEvent("e" + uuid.uuid4().hex, "inst", role, "daic", process_id,
                       1, 0.0, kind, "", payload)


def test_a_failed_stereo_member_cannot_hide_behind_a_healthy_one():
    """The acceptance criterion the whole per-sensor design exists for."""
    view = PipelineView()
    view.observe(event("module.hello", {"state": "ready", "intake": {
        "wanted_hz": 30.0, "achieved_hz": 30.0, "grade": health.OK}}), now=100.0)
    view.observe(event(SENSOR_HEALTH_EVENT, {"sensor_inputs": {
        "nav_left": {"required": True, "state": health.OK, "observed_hz": 29.9},
        "nav_right": {"required": True, "state": health.BAD, "observed_hz": 0.0},
    }}), now=100.0)

    monitor = SimulationHealth()
    record = monitor.sample(wall_now=100.0, sim_now=1.0, requested=1.0,
                            members=view.members(now=100.0))
    module = record["modules"][0]
    assert module["sensor_grade"] == health.BAD
    # The module's own loop rate is fine; the summary must not say so.
    assert module["achieved_hz"] == 30.0
    assert module["grade"] == health.BAD
    assert set(module["sensor_inputs"]) == {"nav_left", "nav_right"}


def test_a_stale_sensor_report_is_dropped_rather_than_shown_as_healthy():
    view = PipelineView(expiry_s=3.0)
    view.observe(event("module.hello", {"state": "ready"}), now=100.0)
    view.observe(event(SENSOR_HEALTH_EVENT, {"sensor_inputs": {
        "front": {"required": True, "state": health.OK}}}), now=100.0)

    member, _age = view.members(now=101.0, include_expired=True)[0]
    assert member.sensors["front"]["state"] == health.OK
    member, _age = view.members(now=110.0, include_expired=True)[0]
    assert member.sensors == {}
    assert member.sensors_age_s == pytest.approx(10.0)


def test_goodbye_clears_the_sensor_report_with_the_module():
    view = PipelineView()
    view.observe(event("module.hello", {"state": "ready"}), now=100.0)
    view.observe(event(SENSOR_HEALTH_EVENT, {"sensor_inputs": {"front": {}}}), now=100.0)
    view.observe(event("module.goodbye", {}), now=100.5)
    assert view.members(now=100.5, include_expired=True) == []
    view.observe(event("module.hello", {"state": "ready"}), now=101.0)
    member, _age = view.members(now=101.0)[0]
    assert member.sensors == {}


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

class Renderer:
    def render_views(self, requests):
        for _id, _model, _pose, frame in requests:
            frame[:] = 7
    def drop_views(self, camera_ids): pass


def manager(profile, **kwargs):
    return SensorManager('sensor-health-' + uuid.uuid4().hex[:8], profile,
                         SimpleNamespace(objects=[]), Renderer(), **kwargs)


def sensing_profile():
    draft = camera_profile(16, 12, physics_hz=30.)
    draft["sensors"].append(dict(id="sonar", type="range.ultrasonic", parent="body",
                                 rate_hz=10., pose_parent=dict(pitch_deg=-90.)))
    return DroneProfile.parse(draft)


def test_production_reports_every_configured_sensor_separately():
    sensors = manager(sensing_profile())
    try:
        for tick in range(30):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        production = sensors.production()
        assert set(production) == {"front", "sonar"}
        assert production["front"]["published"] == 30
        assert production["front"]["observed_hz"] == pytest.approx(30.0)
        assert production["sonar"]["published"] == 10
        assert production["sonar"]["configured_hz"] == 10.0
        assert production["sonar"]["last_sequence"] == 10
        assert production["sonar"]["last_sim_time_s"] == pytest.approx(29 / 30, abs=1e-3)
        assert all(record["state"] == health.OK for record in production.values())
        assert all(record["drops"] == 0 for record in production.values())
    finally:
        sensors.close()


def test_a_sensor_is_starting_until_its_first_period_could_have_passed():
    sensors = manager(sensing_profile())
    try:
        assert sensors.production()["sonar"]["state"] == health.STARTING
        sensors.tick(DroneState(3, 3, 2), 0.0)
        assert sensors.production()["front"]["state"] == health.OK
        assert sensors.production()["sonar"]["state"] == health.STARTING
    finally:
        sensors.close()


def test_a_publisher_failure_is_a_drop_and_a_fault_rather_than_a_crash():
    """The provider is not allowed to stop because a client cannot keep up."""
    sensors = manager(sensing_profile())
    try:
        sensors.tick(DroneState(3, 3, 2), 0.0)

        def refuse(*args, **kwargs):
            raise RuntimeError("ring write refused")

        sensors.publisher.write_compact = refuse
        for tick in range(1, 10):
            sensors.tick(DroneState(3, 3, 2), tick / 30)

        record = sensors.production()["front"]
        assert record["drops"] == 9 and record["published"] == 1
        assert record["fault"] == "ring write refused"
        assert record["state"] == health.BAD
        assert record["last_sequence"] == 1, "a dropped record advances nothing"
    finally:
        sensors.close()


def test_production_reaches_the_health_record_and_the_run_summary():
    sensors = manager(sensing_profile())
    monitor = SimulationHealth()
    try:
        for tick in range(30):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        record = monitor.sample(wall_now=100.0, sim_now=1.0, requested=1.0,
                                production=sensors.production())
        assert set(record["sensors"]) == {"front", "sonar"}
        assert monitor.summary()["sensors"]["front"]["published"] == 30
    finally:
        sensors.close()


def test_production_reports_where_each_sensors_own_work_went():
    """Per-sensor work time, and nothing double-counted: a shared render pass
    is split between the cameras that took part, so the per-sensor figures
    sum to the time the sensor plane actually spent."""
    draft = camera_profile(16, 12, physics_hz=30.)
    draft["sensors"] = stereo_pair('nav', rate_hz=30.,
                                   model=dict(width_px=16, height_px=12,
                                              fov_h_deg=70., near_m=.15, far_m=150.))
    draft["primary_camera"] = 'nav_left'
    draft["sensors"].append(dict(id="sonar", type="range.ultrasonic", parent="body",
                                 rate_hz=10., pose_parent=dict(pitch_deg=-90.)))
    sensors = manager(DroneProfile.parse(draft))
    try:
        for tick in range(30):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        production = sensors.production()
        assert set(production) == {"nav_left", "nav_right", "sonar"}
        # The pair rendered in one shared pass, so each member carried half of it.
        assert production["nav_left"]["work_s"] == pytest.approx(sensors.render_total_s / 2)
        assert production["nav_right"]["work_s"] == pytest.approx(sensors.render_total_s / 2)
        assert production["sonar"]["work_s"] == pytest.approx(sensors.sample_total_s)
        total = sum(record["work_s"] for record in production.values())
        assert total == pytest.approx(sensors.render_total_s + sensors.sample_total_s)
    finally:
        sensors.close()


def test_reset_preserves_counts_without_inflating_production_rate():
    sensors = manager(sensing_profile())
    try:
        for tick in range(30):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        sensors.reset()
        for tick in range(30, 60):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        record = sensors.production()['front']
        assert record['scheduled'] == record['generated'] == record['published'] == 60
        assert record['observed_hz'] == pytest.approx(30)
    finally:
        sensors.close()


@pytest.mark.parametrize('failure', ['render', 'commit'])
def test_camera_failures_are_reported_while_other_sensors_continue(failure):
    sensors = manager(sensing_profile())
    try:
        def refuse(*args, **kwargs):
            raise RuntimeError('camera unavailable')
        if failure == 'render':
            sensors.renderer.render_views = refuse
        else:
            sensors.publisher.commit_camera = refuse
        for tick in range(30):
            sensors.tick(DroneState(3, 3, 2), tick / 30)
        record = sensors.production()['front']
        assert record['scheduled'] == record['drops'] == 30
        assert record['generated'] == (30 if failure == 'commit' else 0)
        assert record['published'] == 0
        assert record['fault'] == 'camera unavailable'
        assert record['state'] == health.BAD
        assert sensors.production()['sonar']['published'] == 10
    finally:
        sensors.close()


@pytest.mark.parametrize('changed', [dict(provider_session_id='replacement'), dict(reset_epoch=1)])
def test_intake_restarts_history_on_provider_restart_or_drone_reset(changed):
    intake = SensorIntake()
    initial = dict(subscription(), provider_session_id='original', reset_epoch=0)
    intake.follow([initial])
    intake.report(0)
    feed(intake, 'front', range(1, 31))
    intake.follow([dict(initial, **changed)])
    record = intake.report(1)['front']
    assert record['last_sequence'] is None
    assert record['state'] == health.STARTING
