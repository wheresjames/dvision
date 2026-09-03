"""Operational realism contracts shared by dsim and dway."""

from __future__ import annotations

import json
import random

import pytest

from dsim.dsim import parse_args
from dsim.realism import Geofence, Realism, TelemetryDelay
from dtest.dway_rig import FORWARD_TOUR, Rig
from dway.mission import MissionState
from dway.report import repeatability_summary


def test_gps_denial_keeps_local_estimator_and_invalidates_global() -> None:
    realism = Realism(gps_mode="off", local_estimator=True)
    assert realism.estimators() == {
        "attitude": True, "local": True, "global": False, "velocity": True}
    realism.set_estimator(local=False)
    assert realism.estimators()["local"] is False
    assert realism.estimators()["velocity"] is False


def test_wind_uses_compass_from_direction() -> None:
    realism = Realism(wind_mps=2.0, wind_dir_deg=0.0)
    east, south = realism.wind_vector()
    assert east == pytest.approx(0.0, abs=1e-9)
    assert south == pytest.approx(2.0)


def test_telemetry_delay_releases_latest_due_sample_without_reordering() -> None:
    delay = TelemetryDelay(random.Random(4), latency_ms=100, jitter_ms=20)
    delay.push(1.0, {"sample": "one"})
    delay.push(1.01, {"sample": "two"})
    assert delay.release(1.05) is None
    first = delay.release(1.2)
    assert first is not None and first["sample"] == "two"


def test_geofence_normalizes_corners_and_checks_ceiling() -> None:
    fence = Geofence.parse("10,8,2,3,4")
    assert fence is not None
    assert fence.contains(5, 5, 4)
    assert not fence.contains(5, 5, 4.01)
    assert not fence.contains(1.99, 5, 2)


def test_realism_fields_are_published_for_flight_reports() -> None:
    realism = Realism(
        sensor_noise="light", battery_failsafe_pct=20,
        telemetry=TelemetryDelay(random.Random(1), 120, 15), seed=9)
    fields = realism.status_fields()
    assert fields["realism.sensor_noise"] == "light"
    assert fields["realism.telemetry_latency_ms"] == "120.000"
    assert fields["realism.telemetry_jitter_ms"] == "15.000"
    assert fields["realism.battery_failsafe_pct"] == "20.000"
    assert fields["realism.seed"] == "9"


def test_setpoint_timeout_defaults_to_two_seconds() -> None:
    args = parse_args([
        "--id", "realism-default", "--map", "assets/maps/test_direct.txt",
        "--no-ui"])
    assert args.setpoint_timeout == 2.0


def test_explicit_realism_flag_overrides_profile(tmp_path) -> None:
    profile = tmp_path / "vehicle.json"
    profile.write_text(json.dumps({"gps": "degraded", "wind_mps": 3.0}),
                       encoding="utf-8")
    args = parse_args([
        "--id", "profile", "--map", "assets/maps/test_direct.txt", "--no-ui",
        "--vehicle-profile", str(profile), "--gps", "rtk"])
    assert args.gps == "rtk"
    assert args.wind_mps == 3.0


def test_five_run_repeatability_publishes_path_and_arrival_variance() -> None:
    runs = [{
        "outcome": "complete", "path_length_m": 10.0 + index,
        "waypoints": [{"arrival_s": 4.0 + index}, {"arrival_s": 8.0 + index}],
    } for index in range(5)]
    aggregate = repeatability_summary(runs)
    assert aggregate["runs"] == aggregate["complete_runs"] == 5
    assert aggregate["path_length_variance_m2"] == pytest.approx(2.0)
    assert aggregate["arrival"][0]["variance_s2"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The environment and fault matrix, flown
#
# Each case below is one row of the operational matrix: the same baseline tour
# flown with one part of the environment switched on, asserting the behaviour
# the contract documents rather than the internals of the noise model.
# ---------------------------------------------------------------------------

def test_a_good_fix_is_the_baseline_every_other_row_is_compared_against(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path, realism={"gps": "good"})
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    state = rig.mission.last_state
    assert state.global_position_valid and state.local_position_valid
    assert rig.summary()["conditions"]["gps.fix_type"] == "3"


def test_gps_denial_still_flies_on_a_valid_local_estimate(
        monkeypatch, tmp_path) -> None:
    """The GPS-denied branch: no fix, but VIO/SLAM still locates the vehicle."""
    rig = Rig(monkeypatch, tmp_path,
              realism={"gps": "off", "local_estimator": "on"})
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    state = rig.mission.last_state
    assert state.local_position_valid and not state.global_position_valid
    conditions = rig.summary()["conditions"]
    assert conditions["gps.fix_type"] == "0"
    assert conditions["est.global_position_valid"] == "0"
    assert conditions["est.local_position_valid"] == "1"


def test_no_position_estimate_at_all_refuses_to_fly_and_names_the_fact(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path,
              realism={"gps": "off", "local_estimator": "off"})
    assert rig.fly(limit_s=10.0) is MissionState.FAILED
    assert "local position estimate is invalid" in rig.mission.reason
    assert not rig.sim.state.armed
    # The vehicle page answers "why will it not fly" with that same fact.
    assert "local position estimate is invalid" in rig.mission.blocking_fact()


def test_a_local_estimator_that_fails_in_flight_stops_the_vehicle(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path)
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    rig.sim.realism.set_estimator(local=False)
    assert rig.fly(limit_s=10.0) is MissionState.FAILED
    assert "local position estimate is invalid" in rig.mission.reason
    assert rig.summary()["partial"] is True


@pytest.mark.parametrize("strategy", ("position", "velocity"))
def test_a_steady_wind_is_trimmed_out_rather_than_hovered_downwind_of(
        monkeypatch, tmp_path, strategy: str) -> None:
    """Both loops hold the waypoint in wind, and they do it the same way.

    A purely proportional approach parks at an offset of exactly the wind
    divided by its gain, which no arrival gate on these tours would ever
    accept. Onboard and off-board loops carry the same trim so a tour behaves
    the same whichever side of the seam closes it.
    """
    rig = Rig(monkeypatch, tmp_path, strategy=strategy, finish_action="hold",
              realism={"wind_mps": 0.4, "wind_dir_deg": 270.0,
                       "wind_gust_mps": 0.1})
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    last = rig.tour.waypoints[-1]
    assert rig.sim.state.x == pytest.approx(last.x, abs=0.05)
    assert rig.sim.state.y == pytest.approx(last.y, abs=0.05)
    conditions = rig.summary()["conditions"]
    assert float(conditions["wind.speed_mps"]) == pytest.approx(0.4, abs=0.15)
    assert float(conditions["wind.dir_deg"]) == pytest.approx(270.0)


def test_stronger_wind_takes_longer_to_settle_and_says_so_when_it_cannot(
        monkeypatch, tmp_path) -> None:
    """Trim settles, but not instantly, and the leg timeout is what notices.

    At 0.8 m/s the committed tour's default leg timeout expires before the
    vehicle has trimmed the wind out. That is a stated failure with a reason,
    not a flight that quietly continues -- and a tour that expects to be flown
    in wind widens its own ``leg_timeout_s``, which the follower never does
    for it.
    """
    impatient = Rig(monkeypatch, tmp_path, finish_action="hold",
                    realism={"wind_mps": 0.8, "wind_dir_deg": 270.0})
    assert impatient.fly(limit_s=120.0) is MissionState.FAILED
    assert "leg timeout" in impatient.mission.reason
    assert impatient.sim.state.mode == "HOLD"

    payload = json.loads(FORWARD_TOUR.read_text())
    payload["leg_timeout_s"] = 90.0
    patient_tour = tmp_path / "patient.json"
    patient_tour.write_text(json.dumps(payload))
    patient = Rig(monkeypatch, tmp_path, tour_path=patient_tour,
                  finish_action="hold",
                  realism={"wind_mps": 0.8, "wind_dir_deg": 270.0})
    assert patient.fly(limit_s=400.0) is MissionState.COMPLETE, patient.mission.reason
    last = patient.tour.waypoints[-1]
    assert patient.sim.state.x == pytest.approx(last.x, abs=0.05)


def test_delayed_telemetry_is_flown_on_without_reordering(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path,
              realism={"telemetry_latency_ms": 120.0,
                       "telemetry_jitter_ms": 30.0})
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    conditions = rig.summary()["conditions"]
    assert float(conditions["realism.telemetry_latency_ms"]) == pytest.approx(120.0)
    assert float(conditions["realism.telemetry_jitter_ms"]) == pytest.approx(30.0)


def test_light_sensor_noise_stays_inside_the_committed_arrival_gates(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path, realism={"sensor_noise": "light"})
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    assert rig.summary()["conditions"]["realism.sensor_noise"] == "light"


def test_heavy_noise_needs_a_tour_that_widens_its_own_gates(
        monkeypatch, tmp_path) -> None:
    """The follower never secretly widens a gate to make a run succeed."""
    payload = json.loads(FORWARD_TOUR.read_text())
    payload["heading_tolerance_deg"] = 12.0
    payload["waypoint_tolerance_m"] = 0.6
    payload["arrival_speed_mps"] = 0.4
    widened = tmp_path / "widened.json"
    widened.write_text(json.dumps(payload))

    strict = Rig(monkeypatch, tmp_path, realism={"sensor_noise": "heavy"})
    assert strict.tour.heading_tolerance_deg == 5.0
    assert strict.fly(limit_s=60.0) is MissionState.FAILED
    assert "leg timeout" in strict.mission.reason

    relaxed = Rig(monkeypatch, tmp_path, tour_path=widened,
                  realism={"sensor_noise": "heavy"})
    assert relaxed.fly() is MissionState.COMPLETE, relaxed.mission.reason


def test_a_low_battery_returns_home_and_lands(monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path, finish_action="hold",
              realism={"battery_failsafe_pct": 99.0,
                       "battery_drain_pct_s": 2.0})
    assert rig.fly(limit_s=60.0) is MissionState.FAILED
    assert "battery_low" in rig.mission.reason
    # The vehicle flies its own failsafe out: home, then down.
    for _ in range(2000):
        rig.sim.integrate(0.05)
        rig.clock.advance(0.05)
        if not rig.sim.state.armed:
            break
    assert rig.sim.state.mode in ("LAND", "DISARMED")
    assert rig.sim.state.x == pytest.approx(rig.sim.state.home_x, abs=0.3)
    assert rig.sim.state.y == pytest.approx(rig.sim.state.home_y, abs=0.3)
    assert [entry["reason"] for entry in rig.summary()["failsafes"]] == ["battery_low"]


@pytest.mark.parametrize(("action", "expected_mode"),
                         (("hold", "HOLD"), ("rtl", "RTL")))
def test_a_vehicle_found_outside_the_fence_acts_as_configured(
        monkeypatch, tmp_path, action: str, expected_mode: str) -> None:
    """A fence the vehicle is already outside of, which is how one is breached.

    ``dsim`` refuses a setpoint that would cross the fence, so the vehicle
    cannot be commanded out of one. It ends up outside when the fence moves
    under it -- a new fence uploaded mid-flight, or one narrowed after takeoff
    -- which is exactly what is installed here.
    """
    rig = Rig(monkeypatch, tmp_path, finish_action="hold",
              realism={"geofence_action": action})
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    rig.sim.realism.geofence = Geofence(0.0, 0.0, 2.0, 2.0)

    assert rig.fly(limit_s=60.0) is MissionState.FAILED
    assert "geofence" in rig.mission.reason
    assert rig.sim.state.mode == expected_mode
    assert [entry["reason"] for entry in rig.summary()["failsafes"]] == ["geofence"]


def test_a_target_outside_the_fence_is_refused_before_it_is_flown(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path,
              realism={"geofence": "34.0,3.0,35.0,4.0"})
    assert rig.fly(limit_s=20.0) is MissionState.FAILED
    assert "outside geofence" in rig.mission.reason
    assert rig.sim.state.mode == "HOLD"


def test_wind_does_not_carry_a_parked_vehicle_away(monkeypatch, tmp_path) -> None:
    """A disarmed vehicle is parked, whatever the weather.

    Wind acting on the start pose blows the simulator's own drone across the
    map while a client is still connecting, so a flight can fail before
    anything has armed.
    """
    rig = Rig(monkeypatch, tmp_path, autostart=False,
              realism={"wind_mps": 2.0, "wind_dir_deg": 90.0})
    start = (rig.sim.state.x, rig.sim.state.y)
    for _ in range(600):          # 30 s on the ground
        rig.sim.integrate(0.05)
        rig.clock.advance(0.05)
    assert not rig.sim.state.armed and not rig.sim.state.crashed
    assert (rig.sim.state.x, rig.sim.state.y) == pytest.approx(start)

    # Once it is flying, the same wind is felt again.
    assert rig.fly(until=lambda m: m.state is MissionState.READY)
    assert rig.mission.start()
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    airborne = rig.sim.state.x
    for _ in range(20):
        rig.step()
    assert rig.sim.state.x != pytest.approx(airborne)
