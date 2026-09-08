"""The state sensors: what they measure, in which axes, and with what noise.

These read the vehicle and the environment rather than the scene, so the
checks are against physical conventions rather than geometry: which way a
positive gyro reading points, what an accelerometer reads at rest, that a
receiver which is not fitted is a different answer from one with no fix, and
that a per-sample error depends on its logical capture and nothing else.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dsim import state_sensors
from dsim.dsim import DroneState, compass_heading_to_sim_yaw
from dsim.profiles import DroneProfile, camera_profile, default_profile
from dsim.realism import REALISM_DEFAULTS, Realism
from dsim.sensor_models import capture_rng
from dsim.transforms import mount_chain, resolve

DT = 0.01


def realism(**settings):
    return Realism.from_settings({**REALISM_DEFAULTS, **settings})


def vehicle(**settings):
    """The context a state sensor reads: an environment and a geodetic datum."""
    model = realism(**settings)
    return SimpleNamespace(
        realism=model, origin_alt_m=34.0,
        # A flat local datum: the sensors are checked on what they do with a
        # position, not on the geodesy of the map, which has its own tests.
        map_to_gps=lambda x, y, z: (52.0 + y * 1e-5, 13.0 + x * 1e-5, 34.0 + z))


def profile_with(*sensors):
    draft = camera_profile()
    draft["sensors"].extend(sensors)
    return DroneProfile.parse(draft).data


def rng(index=0, sensor="s"):
    return capture_rng(0, sensor, 0, index)


def quiet():
    """No white noise, so a reading is exactly the quantity it measures."""
    return dict(gyro_noise_std_dps=0.0, accel_noise_std_mps2=0.0)


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------

def imu_profile(**pose):
    return profile_with(dict(id="imu", type="motion.imu", parent="body",
                             rate_hz=100.0, pose_parent=pose, model=quiet()))


def imu_sample(data, before, after, dt=DT):
    model = next(s for s in data["sensors"] if s["id"] == "imu")["model"]
    payload, valid = state_sensors.imu(model, data, "imu", after, before, dt, rng())
    assert valid
    return payload


def test_an_accelerometer_reads_one_g_upward_at_rest():
    """Specific force, not acceleration: at rest the Z axis reads +g."""
    data = imu_profile()
    payload = imu_sample(data, DroneState(0, 0, 1), DroneState(0, 0, 1))
    force = payload["specific_force_mps2"]
    assert force["z"] == pytest.approx(state_sensors.GRAVITY_MPS2)
    assert (force["x"], force["y"]) == pytest.approx((0.0, 0.0))
    assert payload["angular_rate_dps"] == dict(x=0.0, y=0.0, z=0.0)


def test_free_fall_reads_zero_specific_force():
    data = imu_profile()
    before = DroneState(0, 0, 10, vz=0.0)
    after = DroneState(0, 0, 10, vz=-state_sensors.GRAVITY_MPS2 * DT)
    force = imu_sample(data, before, after)["specific_force_mps2"]
    assert force["z"] == pytest.approx(0.0, abs=1e-9)


def test_forward_acceleration_shows_on_the_forward_axis():
    data = imu_profile()
    before = DroneState(0, 0, 1, yaw_deg=compass_heading_to_sim_yaw(90.0))
    after = DroneState(0, 0, 1, vx=2.0 * DT,
                       yaw_deg=compass_heading_to_sim_yaw(90.0))
    force = imu_sample(data, before, after)["specific_force_mps2"]
    # Heading 90 is map +X, so a +X world acceleration is straight ahead.
    assert force["x"] == pytest.approx(2.0)
    assert force["y"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("axis,before,after", [
    ("z", dict(yaw_deg=compass_heading_to_sim_yaw(0.0)),
     dict(yaw_deg=compass_heading_to_sim_yaw(1.0))),
    ("x", dict(roll_deg=0.0), dict(roll_deg=1.0)),
    ("y", dict(pitch_deg=0.0), dict(pitch_deg=1.0)),
])
def test_positive_rates_follow_the_public_angle_conventions(axis, before, after):
    """Right turn, right wing down, nose up: each is a positive body rate."""
    data = imu_profile()
    payload = imu_sample(data, DroneState(0, 0, 1, **before),
                         DroneState(0, 0, 1, **after))
    rate = payload["angular_rate_dps"]
    assert rate[axis] == pytest.approx(100.0, rel=1e-3)
    for other in set("xyz") - {axis}:
        assert rate[other] == pytest.approx(0.0, abs=1e-6)
    reversed_payload = imu_sample(data, DroneState(0, 0, 1, **after),
                                  DroneState(0, 0, 1, **before))
    assert reversed_payload["angular_rate_dps"][axis] == pytest.approx(-100.0, rel=1e-3)


def test_a_mounted_imu_measures_in_its_own_axes():
    """Rotating the mount rotates the measurement, not just the manifest."""
    upright = imu_profile()
    inverted = imu_profile(roll_deg=180.0)
    level = DroneState(0, 0, 1)
    assert imu_sample(upright, level, level)["specific_force_mps2"]["z"] == pytest.approx(
        state_sensors.GRAVITY_MPS2)
    assert imu_sample(inverted, level, level)["specific_force_mps2"]["z"] == pytest.approx(
        -state_sensors.GRAVITY_MPS2)


def test_the_first_sample_has_nothing_to_differentiate():
    data = imu_profile()
    model = next(s for s in data["sensors"] if s["id"] == "imu")["model"]
    payload, valid = state_sensors.imu(model, data, "imu",
                                       DroneState(0, 0, 1), None, 0.0, rng())
    assert valid and payload["angular_rate_dps"] == dict(x=0.0, y=0.0, z=0.0)
    assert payload["specific_force_mps2"]["z"] == pytest.approx(
        state_sensors.GRAVITY_MPS2)


# ---------------------------------------------------------------------------
# GNSS
# ---------------------------------------------------------------------------

def gnss(state=None, **settings):
    return state_sensors.gnss({}, vehicle(**settings), state or DroneState(3, 4, 5), rng())


def test_a_fix_carries_the_same_error_vehicle_status_publishes():
    context = vehicle(gps="degraded")
    context.realism.update(1.0)
    payload, valid = state_sensors.gnss({}, context, DroneState(3, 4, 5), rng())
    north, east, up = context.realism.gps_offset_m()
    assert valid and payload["fix_type"] == 2
    assert payload["error_north_m"] == north != 0.0
    assert payload["lat_deg"] == pytest.approx(52.0 + (4.0 - north) * 1e-5)
    assert payload["lon_deg"] == pytest.approx(13.0 + (3.0 + east) * 1e-5)
    assert payload["alt_m"] == pytest.approx(34.0 + 5.0 + up)


def test_no_fix_is_a_null_position_rather_than_a_wrong_one():
    payload, valid = gnss(gps="off")
    assert not valid
    assert payload["fix_type"] == 0 and payload["satellites"] == 0
    assert payload["lat_deg"] is None and payload["vel_north_mps"] is None


def test_velocity_is_over_the_ground_including_the_wind():
    payload, _valid = gnss(DroneState(0, 0, 1, vx=1.0, vy=-2.0, vz=0.5),
                           wind_mps=3.0, wind_dir_deg=0.0)
    # Wind from the north blows the vehicle south, which is map +Y.
    assert payload["vel_north_mps"] == pytest.approx(2.0 - 3.0)
    assert payload["vel_east_mps"] == pytest.approx(1.0)
    assert payload["vel_down_mps"] == pytest.approx(-0.5)


def test_installation_fix_and_estimator_are_three_separate_answers():
    """Not fitted, fitted with no fix, and fitted but rejected."""
    absent = realism(gps="rtk")
    absent.set_gnss_installed(False)
    assert absent.gps_fix()["fix_type"] == 0
    assert absent.estimators()["global"] is False

    denied = realism(gps="off")
    assert denied.gnss_installed and denied.gps_fix()["fix_type"] == 0
    assert denied.estimators()["global"] is False

    rejected = realism(gps="rtk")
    rejected.set_estimator(**{"global": False})
    assert rejected.gps_fix()["fix_type"] == 4
    assert rejected.estimators()["global"] is False


def test_the_profile_decides_whether_a_receiver_is_fitted():
    assert any(s["type"] == "position.gnss" for s in
               DroneProfile.parse(default_profile()).data["sensors"])
    assert not any(s["type"] == "position.gnss" for s in
                   DroneProfile.parse(camera_profile()).data["sensors"])


# ---------------------------------------------------------------------------
# Barometer, magnetometer, temperature
# ---------------------------------------------------------------------------

def state_model(kind, **overrides):
    data = profile_with(dict(id="s", type=kind, parent="body", rate_hz=10.0,
                             model=overrides))
    return next(s for s in data["sensors"] if s["id"] == "s")["model"]


def test_pressure_and_altitude_invert_each_other():
    model = state_model("altimeter.barometric", noise_std_m=0.0)
    context = vehicle()
    payload, valid = state_sensors.barometer(model, context, DroneState(0, 0, 12.0), rng())
    assert valid and payload["altitude_m"] == pytest.approx(12.0)
    assert payload["pressure_pa"] == pytest.approx(
        state_sensors.pressure_pa(34.0 + 12.0, 101325.0))
    assert payload["pressure_pa"] < state_sensors.pressure_pa(0.0, 101325.0)


def test_the_barometer_reads_the_environment_drift_rather_than_rerolling_it():
    """One wander, reaching status and the sensor record as the same number."""
    model = state_model("altimeter.barometric", noise_std_m=0.0)
    context = vehicle(sensor_noise="heavy")
    for _ in range(50):
        context.realism.update(0.1)
    drift = context.realism.baro_drift_m
    assert drift != 0.0
    payload, _valid = state_sensors.barometer(model, context, DroneState(0, 0, 5.0),
                                              _ZeroNoise())
    assert payload["altitude_m"] == pytest.approx(5.0 + drift)


class _ZeroNoise:
    """A generator that draws nothing, isolating the deterministic part."""
    def standard_normal(self, shape=None):
        return np.zeros(shape) if shape else 0.0


def test_a_quantized_barometer_reports_on_its_own_grid():
    model = state_model("altimeter.barometric", noise_std_m=0.0, quantization_m=0.5)
    payload, _valid = state_sensors.barometer(model, vehicle(), DroneState(0, 0, 5.3),
                                              _ZeroNoise())
    assert payload["altitude_m"] == pytest.approx(5.5)


def test_the_compass_reports_a_heading_with_its_noise_combined():
    model = state_model("heading.magnetometer", noise_std_deg=0.0)
    context = vehicle()
    payload, valid = state_sensors.magnetometer(
        model, context, DroneState(0, 0, 1, yaw_deg=compass_heading_to_sim_yaw(123.0)),
        _ZeroNoise())
    assert valid and payload["heading_deg"] == pytest.approx(123.0)
    # Two independent sources of the same quantity add in quadrature.
    assert state_sensors.combined(3.0, 4.0) == pytest.approx(5.0)


def test_temperature_falls_with_height_from_the_environment_setting():
    model = state_model("environment.temperature", noise_std_c=0.0, quantization_c=0.0)
    context = vehicle(ambient_temp_c=18.0)
    ground, _ = state_sensors.temperature(model, context, DroneState(0, 0, 0.0), _ZeroNoise())
    aloft, _ = state_sensors.temperature(model, context, DroneState(0, 0, 100.0), _ZeroNoise())
    assert ground["temperature_c"] == pytest.approx(18.0)
    assert aloft["temperature_c"] == pytest.approx(18.0 - 0.65)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_a_state_sample_depends_on_its_capture_and_not_on_call_order():
    model = state_model("altimeter.barometric", noise_std_m=0.2)
    context = vehicle()
    state = DroneState(0, 0, 4.0)

    def altitude(index):
        return state_sensors.barometer(model, context, state,
                                       capture_rng(0, "baro", 0, index))[0]["altitude_m"]

    reference = altitude(9)
    for skipped in range(9):
        altitude(skipped)
    assert altitude(9) == reference
    assert altitude(8) != reference


def test_the_mount_chain_is_the_airframe_alone():
    """Where a sensor sits does not depend on where the vehicle is."""
    data = imu_profile(x_m=0.2, z_m=-0.05, pitch_deg=-90.0)
    chain = mount_chain(data, "imu")
    assert np.allclose(chain[:3, 3], [0.2, 0.0, -0.05])
    for heading in (0.0, 90.0, 217.0):
        state = DroneState(7, 9, 2, yaw_deg=compass_heading_to_sim_yaw(heading))
        world = resolve(data, "imu", state)
        assert np.allclose(world, np.asarray(
            [[math.sin(math.radians(heading)), math.cos(math.radians(heading)), 0, 7],
             [-math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0, 9],
             [0, 0, 1, 2], [0, 0, 0, 1]]) @ chain, atol=1e-9)
